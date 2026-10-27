"""Оценки параметров одного теста по матрице X (релизы x тесты) числа красных среди R прогонов.

est_mom     — метод моментов: a = mean K/R = p s, b = mean K(K-1)/(R(R-1)) = p s^2; тестам без релиза с двумя красными — общая s
est_mle     — ОМП через EM с ускорением (SQUAREM-подобный шаг); тесты без красных -> nan
est_mle0    — ОМП, тесты без красных получают p = 0, s = 0.5
est_pooled  — грубая модель: одна s на все тесты, p_j по доле релизов с красным
H, HK, LHK  — таблицы h(k, n) = C(R-n, k)/C(R, k): вероятность, что первые n из R записей зелёные при k красных
fisher      — информация Фишера одного релиза для (p, s)
"""
import numpy as np
from math import comb

R = 5
EPS = 1e-9

H = np.zeros((R + 1, R + 1))  # H[k, n], H[0, :] = 0 (для несмещённой оценки видимой части)
for _k in range(1, R + 1):
    for _n in range(R + 1):
        H[_k, _n] = comb(R - _n, _k) / comb(R, _k) if _k <= R - _n else 0.0
HK = np.array(
    [[comb(R - n, k) / comb(R, k) if k <= R - n else 0.0 for n in range(R + 1)] for k in range(R + 1)]
)  # HK[0, :] = 1
with np.errstate(divide="ignore"):
    LHK = np.log(HK)


# ----------------------------------------------------------------------------- ОМП (EM)
def suff(X):
    pos = X > 0
    return X.shape[0], (~pos).sum(0).astype(float), pos.sum(0).astype(float), X.sum(0).astype(float)


def objective(p, s, n0, n1, K, r=R, kp=0.0, ks=0.0, p0=0.5, s0=0.5):
    z = (1 - s) ** r
    ll = n0 * np.log(1 - p + p * z) + n1 * np.log(p) + K * np.log(s) + (r * n1 - K) * np.log(1 - s)
    return ll + kp * (p0 * np.log(p) + (1 - p0) * np.log(1 - p)) + ks * (s0 * np.log(s) + (1 - s0) * np.log(1 - s))


def em_map_step(p, s, n, n0, n1, K, r, kp, ks, p0, s0):
    z = (1 - s) ** r
    q = p * z / (1 - p + p * z)
    W = n1 + n0 * q
    pn = (W + kp * p0) / (n + kp)
    sn = (K + ks * s0) / (r * W + ks)
    return np.clip(pn, EPS, 1 - EPS), np.clip(sn, EPS, 1 - EPS)


def em_map(X, kp=0.0, ks=0.0, p0=0.5, s0=0.5, r=R, tol=1e-10, max_iter=20000, p_init=None, s_init=None):
    n, n0, n1, K = suff(X)
    M = X.shape[1]
    p = np.full(M, 0.2) if p_init is None else np.clip(p_init.copy(), EPS, 1 - EPS)
    s = np.full(M, 0.5) if s_init is None else np.clip(s_init.copy(), EPS, 1 - EPS)
    args = (n, n0, n1, K, r, kp, ks, p0, s0)
    obj = lambda p, s: objective(p, s, n0, n1, K, r, kp, ks, p0, s0)
    for it in range(max_iter):
        p1, s1 = em_map_step(p, s, *args)
        p2, s2 = em_map_step(p1, s1, *args)
        rp, rs = p1 - p, s1 - s
        vp, vs = p2 - 2 * p1 + p, s2 - 2 * s1 + s
        nr = np.sqrt(rp**2 + rs**2)
        nv = np.sqrt(vp**2 + vs**2)
        alpha = np.where(nv > 0, -nr / np.maximum(nv, 1e-300), -1.0)
        alpha = np.minimum(alpha, -1.0)
        pa = np.clip(p - 2 * alpha * rp + alpha**2 * vp, EPS, 1 - EPS)
        sa = np.clip(s - 2 * alpha * rs + alpha**2 * vs, EPS, 1 - EPS)
        pa, sa = em_map_step(pa, sa, *args)  # стабилизирующий EM-шаг
        ok = obj(pa, sa) >= obj(p2, s2) - 1e-12
        pn = np.where(ok, pa, p2)
        sn = np.where(ok, sa, s2)
        if max(np.abs(pn - p).max(), np.abs(sn - s).max()) < tol:
            p, s = pn, sn
            break
        p, s = pn, sn
    return p, s


def est_mle(X, r=R):
    """ОМП по каждому тесту; тесты без красных -> nan (правдоподобие плоское по s при p (1 - (1 - s)^r) -> 0)"""
    p, s = em_map(X, r=r)
    nored = ~(X > 0).any(0)
    p[nored] = np.nan
    s[nored] = np.nan
    return p, s


def est_mle0(X):
    p, s = est_mle(X)
    nored = np.isnan(p)
    p[nored] = 0.0
    s[nored] = 0.5
    return p, s


# ----------------------------------------------------------------------------- метод моментов и грубые варианты
def est_mom(X):
    """a = E K/5 = p s, b = E K(K-1)/20 = p s^2  ->  s = b/a, p = a^2/b; тестам без релиза с >= 2 красными — общая s"""
    a = X.mean(0) / R
    b = (X * (X - 1)).mean(0) / (R * (R - 1))
    s0 = b.sum() / max(a.sum(), 1e-12)
    s = np.where(b > 0, b / np.maximum(a, 1e-12), s0)
    p = np.where(a > 0, a / s, 0.0)
    return np.clip(p, 0, 1), np.clip(s, 1e-3, 1)


def est_pooled(X):
    """грубая модель: одна s на всех (общий метод моментов), p_j по доле релизов с красным"""
    a = X.mean(0) / R
    b = (X * (X - 1)).mean(0) / (R * (R - 1))
    s0 = float(np.clip(b.sum() / max(a.sum(), 1e-12), 0.05, 0.99))
    p = (X > 0).mean(0) / (1 - (1 - s0) ** R)
    return np.clip(p, 0, 1), np.full(len(p), s0)


def est(X, kind):
    return {"mom": est_mom, "mle": est_mle0, "pooled": est_pooled}[kind](X)


def fisher(p, s):
    """ожидаемая информация Фишера одного релиза для (p, s) в биноме с избытком нулей, форма (M, 2, 2)"""
    M = len(p)
    I = np.zeros((M, 2, 2))
    z = (1 - s) ** R
    P0 = 1 - p + p * z
    g0 = np.stack([(z - 1) / P0, -p * R * (1 - s) ** (R - 1) / P0], 1)  # градиент log P(K=0)
    I += P0[:, None, None] * g0[:, :, None] * g0[:, None, :]
    for k in range(1, R + 1):
        Pk = p * comb(R, k) * s**k * (1 - s) ** (R - k)
        gk = np.stack([1 / p, k / s - (R - k) / (1 - s)], 1)
        I += Pk[:, None, None] * gk[:, :, None] * gk[:, None, :]
    return I
