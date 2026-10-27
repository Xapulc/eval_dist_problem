"""Зоопарк решений: что могут построить команды, и как это выглядит по истине и на валидации.

solutions(w, runs, A, seed, quick, cond) -> {имя: план}
  all1, all2, all5               все по разу / по два / по пять
  naive_{mom,mle,pooled,freq}_L  подстановка оценок в pi(n), самый дешёвый план цепочки с оценкой <= цели
  naive_mom_E, naive_mom_P       команда контролирует E N или P(N >= 1) вместо pi (строже и дороже)
  delta_{mom,mle}_L              подстановка + z * погрешность (дельта-метод), отбор не учтён
  bootpct_mom_L                  бутстреп-квантиль оценки для планов цепочки, без переотбора
  bootsel_{mom,mle}_L            бутстреп всей процедуры, уровень 0.9, B = 40
  bootsel95_mom_L                то же, уровень 0.95, B = 100  [официальный эталон]
  bootsel_mom_E                  бутстреп всей процедуры для E N
  holdout_cp                     35/35: цепочка по одной половине, граница Клоппера–Пирсона на другой
  oracle                         точный оптимум по истинным параметрам
cond=True: требование P(пропуск | релиз проблемный) <= A; каждый метод целится в A * своя оценка P(релиз проблемный),
оракул — в A * истинную, частоты — в A * долю релизов истории с красным.
"""
import numpy as np

from .model import R, L, en, draw, missed_releases
from .estimators import est
from .plans import (chain_model, first_ok, p1_hat, V_freq, chain_freq, table, chain, first_ok_count,
                    plan_delta, plan_bootpct, plan_bootsel, plan_holdout, oracle_rel)


def solutions(w, runs, A, seed, quick=False, cond=False):
    p, s, c = w["p"], w["s"], w["c"]; M = len(c); X = runs.sum(2)
    tgt = (lambda pe: A * p1_hat(pe)) if cond else (lambda pe: A)
    Af = A * float(runs.any((1, 2)).mean()) if cond else A
    out = {"all1": np.ones(M, int), "all2": np.full(M, 2), "all5": np.full(M, R)}
    for kind in ("mom", "mle", "pooled"):
        pe, se = est(X, kind); out[f"naive_{kind}_L"] = first_ok(chain_model(pe, se, c), lambda n: L(pe, se, n), tgt(pe))
    out["naive_freq_L"] = first_ok(chain_freq(X, c), lambda n: V_freq(X, n), Af)
    pm, sm = est(X, "mom"); Am = tgt(pm); Al = tgt(est(X, "mle")[0])
    Tm = table(X, "mom")
    out["naive_mom_E"] = first_ok_count(chain(Tm, c, "E"), Tm, "E", Am)
    out["naive_mom_P"] = first_ok_count(chain(Tm, c, "P"), Tm, "P", Am)
    out["delta_mom_L"] = plan_delta(X, c, "mom", Am)
    if not quick:
        out["delta_mle_L"] = plan_delta(X, c, "mle", Al)
    out["bootpct_mom_L"] = plan_bootpct(X, c, Am, np.random.default_rng(seed + 1))
    out["bootsel_mom_L"] = plan_bootsel(X, c, "mom", Am, np.random.default_rng(seed + 2))
    out["bootsel95_mom_L"] = plan_bootsel(X, c, "mom", Am, np.random.default_rng(seed + 2), B=100, level=0.95)   # официальный эталон
    if not quick:
        out["bootsel_mle_L"] = plan_bootsel(X, c, "mle", Al, np.random.default_rng(seed + 2))
        out["bootsel_mom_E"] = plan_bootsel(X, c, "mom", Am, np.random.default_rng(seed + 4), crit="E")
    out["holdout_cp"] = plan_holdout(runs, c, A, np.random.default_rng(seed + 3), cond=cond)
    out["oracle"] = oracle_rel(p, s, c, A * (1 - np.prod(1 - p)) if cond else A)
    return out


def evaluate(make_world, A, seeds, ntr=70, nval=30, quick=True, cond=False):
    """по одной истории и одной валидации на мир; столбцы: истинный критерий (pi или pi / P1 при cond), укладывается ли,
    стоимость к «все по 5», пропущено релизов на валидации, проблемных релизов на валидации"""
    res = {}
    for sd in seeds:
        w = make_world(sd); p, s, c = w["p"], w["s"], w["c"]; c5 = c.sum() * R; P1 = 1 - np.prod(1 - p)
        runs = draw(np.random.default_rng(1000 + sd), p, s, ntr)[1]; val = draw(np.random.default_rng(5000 + sd), p, s, nval)[1]
        for k, n in solutions(w, runs, A, sd, quick, cond).items():
            v = L(p, s, n) / (P1 if cond else 1.0); res.setdefault(k, []).append((v, v <= A + 1e-12, c @ n / c5, int(missed_releases(val, n).sum()), int(val.any((1, 2)).sum())))
    return {k: np.array(v, float) for k, v in res.items()}


def report(res, A, K, label=""):
    print(f"\n=== {label}  (A = {A}, {len(next(iter(res.values())))} worlds; cost relative to all 5; val = missed problem releases of 30; pass: <= {K})")
    print(f"{'solution':>14} {'L':>6} {'P(ok)':>6} {'cost':>6} | {'val mean':>8} {'10-90%':>7} {'pass':>5} {'P(> ref)':>8}")
    g = res["bootsel_mom_L"][:, 3]
    for k, v in res.items():
        lo, hi = np.quantile(v[:, 3], [0.1, 0.9])
        print(f"{k:>14} {v[:, 0].mean():6.3f} {v[:, 1].mean():6.2f} {v[:, 2].mean():6.3f} | {v[:, 3].mean():8.2f} {lo:3.0f}-{hi:<3.0f} {np.mean(v[:, 3] <= K):5.2f} {np.mean(v[:, 3] > g):8.2f}")
    print(f"problem releases in validation: {res['all1'][:, 4].mean():.1f} of 30")


# ----------------------------------------------------------------------------- типичные студенческие решения без модели
def emp_rel(runs, n):
    """доля релизов истории, которые план пропустил бы (по записанному порядку прогонов)"""
    return float(missed_releases(runs, n).mean())


def emp_any(runs, n):
    """доля релизов истории, где план упустил хотя бы одну проблему (P(N >= 1))"""
    first = np.cumsum(runs, 2) > 0
    caught = np.where(n[None, :] > 0, first[:, np.arange(len(n)), np.maximum(n - 1, 0)], False)
    return float((runs.any(2) & ~caught).any(1).mean())


def greedy(runs, c, f):
    """жадный набор прогонов прямо по эмпирическому критерию"""
    M = len(c); n = np.zeros(M, int); plans = [n.copy()]; cur = f(runs, n)
    while (n < R).any():
        best, bi = -np.inf, None
        for i in np.where(n < R)[0]:
            m = n.copy(); m[i] += 1; g = (cur - f(runs, m)) / c[i]
            if g > best: best, bi = g, i
        if best <= 0: bi = int(np.argmin(np.where(n < R, c, np.inf)))
        n[bi] += 1; cur = f(runs, n); plans.append(n.copy())
    return plans


def student_plans(runs, c, alphas, cond=False):
    """cond=True: доли берутся среди релизов истории с красным (при условии проблемного релиза)"""
    out = {}; den = float(runs.any((1, 2)).mean()) if cond else 1.0
    pr, pa = greedy(runs, c, emp_rel), greedy(runs, c, emp_any)
    for a in alphas:
        out[f"emp_rel@{a}"] = next(n for n in pr if emp_rel(runs, n) / den <= a)
        out[f"emp_any@{a}"] = next(n for n in pa if emp_any(runs, n) / den <= a)
    return out
