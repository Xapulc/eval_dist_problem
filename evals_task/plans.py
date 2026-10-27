"""Построение планов n по оценкам параметров (или по истинным — оракул).

Релизный критерий pi(n) <= A:
  chain_model   — вложенные планы 0 -> все по R, жадно по убыванию -log g_j на рубль
  first_ok      — первый план цепочки, у которого оценка критерия не выше цели
  V_freq, chain_freq — то же без модели, по частотам (несмещённая оценка видимой части)
  plan_delta    — подстановка + z * погрешность (дельта-метод) при фиксированном плане
  plan_bootpct  — бутстреп-квантиль оценки для каждого плана цепочки, без переотбора
  plan_bootsel  — бутстреп всей процедуры: запас, при котором переотобранные планы проходят в доле level повторов
  plan_holdout  — обучение на одной половине истории, проверка на другой границей Клоппера–Пирсона
  oracle_rel    — точный оптимум целочисленным программированием (scipy.optimize.milp)
Критерии по числу упущенных проблем (E N, P(N >= 1)): table, value, chain, first_ok_count.
"""
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.stats import beta, norm

from .model import R, g_tab, L, qtab, en, missed_releases
from .estimators import est, est_mom, est_mle0, H, HK, LHK, fisher

Z90 = norm.ppf(0.9)


# ----------------------------------------------------------------------------- релизный критерий, модель
def chain_model(p, s, c):
    W = -np.log(np.maximum(g_tab(p, s), 1e-300)); M = len(c); n = np.zeros(M, int); plans = [n.copy()]; idx = np.arange(M)
    while (n < R).any():
        gain = np.where(n < R, (W[idx, np.minimum(n + 1, R)] - W[idx, n]) / c, -np.inf)
        if gain.max() <= 0:                                   # учиться больше не на чем: добавляем самый дешёвый прогон
            gain = np.where(n < R, -c, -np.inf)
        n[int(np.argmax(gain))] += 1; plans.append(n.copy())
    return plans


def first_ok(plans, f, A):
    return next((n for n in plans if f(n) <= A), plans[-1])


def p1_hat(pe):
    """оценка P(релиз проблемный) по модели"""
    return float(1 - np.prod(1 - pe))


# ----------------------------------------------------------------------------- релизный критерий, частоты
def V_freq(X, n):
    """несмещённая оценка видимой части pi(n): среднее по релизам P(первые n_j всех тестов зелёные) - [красных нет вовсе]"""
    return float(np.mean(np.prod(HK[X, n[None, :]], 1) - (X.sum(1) == 0)))


def chain_freq(X, c):
    N, M = X.shape; n = np.zeros(M, int); plans = [n.copy()]; S = LHK[X, 0].sum(1); rows = np.arange(N)
    while (n < R).any():
        cur = np.exp(S).mean(); best, bi = -np.inf, -1
        for i in np.where(n < R)[0]:
            with np.errstate(invalid="ignore"):
                Si = S + LHK[X[:, i], n[i] + 1] - LHK[X[:, i], n[i]]
            Si = np.where(np.isfinite(S), Si, -np.inf)
            g = (cur - np.exp(Si).mean()) / c[i]
            if g > best + 1e-15:
                best, bi = g, i
        if best <= 0:
            bi = int(np.argmin(np.where(n < R, c, np.inf)))
        with np.errstate(invalid="ignore"):
            S = np.where(np.isfinite(S), S + LHK[X[:, bi], n[bi] + 1] - LHK[X[:, bi], n[bi]], -np.inf)
        n[bi] += 1; plans.append(n.copy())
    return plans


# ----------------------------------------------------------------------------- погрешность при фиксированном плане
def se_delta_mom(X, n):
    N, M = X.shape; p, s = est_mom(X); a = X.mean(0) / R; b = (X * (X - 1)).mean(0) / (R * (R - 1)); s0 = b.sum() / max(a.sum(), 1e-12)
    G = g_tab(p, s)[np.arange(M), n]; Pg = np.prod(G); P0 = np.prod(1 - p); var = 0.0
    def pi_si(aa, bb, i):
        si = bb / aa if b[i] > 0 else s0; si = min(max(si, 1e-3), 1.0); return min(aa / si, 1.0), si
    for i in range(M):
        if a[i] <= 0:
            continue
        def f(aa, bb):
            pp, ss = pi_si(aa, bb, i); gi = 1 - pp * (1 - (1 - ss) ** n[i])
            return Pg / G[i] * gi - P0 / (1 - p[i]) * (1 - pp) if p[i] < 1 else Pg / G[i] * gi
        y1, y2 = X[:, i] / R, X[:, i] * (X[:, i] - 1) / (R * (R - 1)); ea, eb = 1e-6 * a[i], max(1e-6 * b[i], 1e-9)
        ga = (f(a[i] + ea, b[i]) - f(a[i] - ea, b[i])) / (2 * ea)
        if b[i] > 0:
            gb = (f(a[i], b[i] + eb) - f(a[i], b[i] - eb)) / (2 * eb); g = np.array([ga, gb])
            var += g @ np.cov(np.stack([y1, y2])) @ g / N
        else:
            var += ga ** 2 * y1.var(ddof=1) / N
    return float(np.sqrt(var))


def se_delta_mle(X, n):
    N, M = X.shape; p, s = est_mle0(X); pp, ss = np.clip(p, 1e-4, 1 - 1e-4), np.clip(s, 1e-3, 1 - 1e-3)
    Cov = np.linalg.inv(fisher(pp, ss) + np.eye(2) * 1e-6) / N; var = 0.0; base = L(pp, ss, n)
    for i in range(M):
        g = np.zeros(2)
        for k, h in ((0, 1e-5), (1, 1e-5)):
            q, t = pp.copy(), ss.copy()
            if k == 0:
                q[i] += h
            else:
                t[i] += h
            g[k] = (L(q, t, n) - base) / h
        var += g @ Cov[i] @ g
    return float(np.sqrt(var))


# ----------------------------------------------------------------------------- критерии по числу упущенных проблем
def table(X, kind):
    """Q_hat_j(n), n = 0..R: по частотам (kind='freq') или по оценкам параметров"""
    if kind == "freq":
        return np.minimum.accumulate(H[X].mean(0), 1)
    p, s = est(X, kind)
    return qtab(p, s)


def value(T, n, crit):
    Q = T[np.arange(len(n)), n]
    return float(Q.sum()) if crit == "E" else float(1 - np.prod(1 - Q))


def chain(T, c, crit):
    """вложенные планы 0 -> все по R, жадно по снижению критерия на рубль"""
    W = T if crit == "E" else -np.log1p(-np.minimum(T, 1 - 1e-12))
    M = len(c); n = np.zeros(M, int); plans = [n.copy()]; idx = np.arange(M)
    while (n < R).any():
        g = np.where(n < R, (W[idx, n] - W[idx, np.minimum(n + 1, R)]) / c, -np.inf); n[int(np.argmax(g))] += 1; plans.append(n.copy())
    return plans


def first_ok_count(plans, T, crit, target):
    return next((n for n in plans if value(T, n, crit) <= target), plans[-1])


# ----------------------------------------------------------------------------- планы с запасом
def plan_delta(X, c, kind, A):
    p, s = est(X, kind); se = se_delta_mom if kind == "mom" else se_delta_mle
    return first_ok(chain_model(p, s, c), lambda n: L(p, s, n) + Z90 * se(X, n), A)


def plan_bootpct(X, c, A, rng, B=60):
    p, s = est_mom(X); ch = chain_model(p, s, c); ests = [est_mom(X[rng.integers(0, len(X), len(X))]) for _ in range(B)]
    return first_ok(ch, lambda n: np.quantile([L(pb, sb, n) for pb, sb in ests], 0.9), A)


def plan_bootsel(X, c, kind, A, rng, B=40, crit="L", level=0.9):
    """бутстреп всей процедуры; crit='L' — релизный критерий, crit='E' — ожидаемое число упущенных проблем"""
    p, s = est(X, kind)
    val = (lambda pp, ss, n: L(pp, ss, n)) if crit == "L" else (lambda pp, ss, n: en(pp, ss, n))
    chn = (lambda pp, ss: chain_model(pp, ss, c)) if crit == "L" else (lambda pp, ss: chain(qtab(pp, ss), c, "E"))
    grid = np.arange(0, 0.6001, 0.01) * A; ok = np.zeros(len(grid))
    for _ in range(B):
        pb, sb = est(X[rng.integers(0, len(X), len(X))], kind); chb = chn(pb, sb)
        e = np.array([val(pb, sb, n) for n in chb]); ref = np.array([val(p, s, n) for n in chb])
        for gi, cc in enumerate(grid):
            k = int(np.argmax(e <= A - cc)) if (e <= A - cc).any() else len(chb) - 1
            ok[gi] += ref[k] <= A
    cc = grid[np.argmax(ok / B >= level)] if (ok / B >= level).any() else grid[-1]
    return first_ok(chn(p, s), lambda n: val(p, s, n), A - cc)


def plan_holdout(runs, c, A, rng, learn=35, delta=0.1, eps_bar=0.02, cond=False):
    perm = rng.permutation(len(runs)); Lr, T = runs[perm[:learn]], runs[perm[learn:]]; m = len(T)
    p, s = est_mom(Lr.sum(2)); chosen = np.full(runs.shape[1], R)
    if cond:                                      # цель при условии проблемного релиза: A * P_hat(релиз проблемный)
        A = A * p1_hat(p)
    for n in reversed(chain_model(p, s, c)):
        k = int(missed_releases(T, n).sum()); ub = 1.0 if k == m else float(beta.ppf(1 - delta, k + 1, m - k))
        if ub > A - eps_bar:                      # по теореме: видимая часть должна уложиться в A - eps_bar
            break
        chosen = n
    return chosen


# ----------------------------------------------------------------------------- оракул
def oracle_rel(p, s, c, A):
    """min c.n  при  sum log g_j(n_j) <= log(A + P0)  (разделимо после логарифма), булевы переменные с монотонностью"""
    M = len(p); G = np.log(g_tab(p, s)); d = (G[:, :-1] - G[:, 1:]).ravel(); rhs = G[:, 0].sum() - np.log(A + np.prod(1 - p))
    if rhs <= 0:
        return np.zeros(M, int)
    rows = []
    for i in range(M):
        for k in range(1, R):
            r_ = np.zeros(M * R); r_[i * R + k] = 1; r_[i * R + k - 1] = -1; rows.append(r_)
    res = milp(np.repeat(c, R), constraints=[LinearConstraint(d[None, :], lb=rhs, ub=np.inf), LinearConstraint(np.array(rows), lb=-np.inf, ub=0)],
               integrality=np.ones(M * R), bounds=Bounds(0, 1))
    return np.full(M, R) if res.x is None else res.x.reshape(M, R).round().sum(1).astype(int)
