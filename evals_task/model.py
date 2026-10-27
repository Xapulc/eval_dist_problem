"""Модель мира и данных.

Тест j на релизе i: проблема есть с вероятностью p_j (независимо по тестам и релизам); при проблеме каждый из R прогонов
красный независимо с вероятностью s_j; без проблемы все прогоны зелёные. План n = (n_1..n_M), n_j in 0..R.
Релиз проблемный, если проблема есть хотя бы в одном тесте; план пропускает его, если все запланированные прогоны зелёные:
  pi(n) = prod_j g_j(n_j) - prod_j (1 - p_j),   g_j(n) = 1 - p_j + p_j (1 - s_j)^n.
"""
import numpy as np

R = 5  # прогонов каждого теста на релизе в истории

# архетипы: (имя, доля тестов, диапазон s, диапазон p до масштабирования, множитель стоимости)
BASE = [
    ("ideal", 0.15, (0.80, 0.95), (0.6, 1.4), 1.0),
    ("shy", 0.15, (0.55, 0.70), (0.6, 1.4), 1.0),
    ("weak", 0.08, (0.50, 0.65), (0.6, 1.4), 1.0),
    ("heavy_exact", 0.08, (0.90, 0.99), (0.6, 1.6), 10.0),
    ("cheap_mid", 0.29, (0.55, 0.75), (0.6, 1.4), 0.3),
    ("never_breaks", 0.12, (0.85, 0.95), (0.0, 0.04), 1.0),
    ("often", 0.13, (0.65, 0.85), (2.5, 4.0), 1.0),
]
HARD = ("shy", "weak", "cheap_mid", "often")  # у этих архетипов чувствительность понижена на d_hard


def arch_rel(p_scale, d_hard, d_all=0.0):
    out = []
    for n, sh, sr, pr, cm in BASE:
        dd = d_all + (d_hard if n in HARD else 0.0)
        out.append((n, sh, (max(sr[0] - dd, 0.08), max(sr[1] - dd, 0.12)), tuple(x * p_scale for x in pr), cm))
    return out


def make_world_arch(seed, arch, p_scale=1.0, cost_sigma=0.4, M=40):
    """arch: list of (name, share, s_range, p_range, cost_mult); p = p_scale * U(p_range).
    Стоимости зависят только от seed и долей архетипов: при любых диапазонах p и s они одни и те же."""
    rng = np.random.default_rng(seed)
    cnt = np.round(np.array([a[1] for a in arch]) * M).astype(int)
    cnt[np.argmax(cnt)] += M - cnt.sum()
    s, p, c, lab = [], [], [], []
    for (nm, _, sr, pr, cm), k in zip(arch, cnt):
        s += list(rng.uniform(*sr, k))
        p += list(rng.uniform(*pr, k) * p_scale)
        c += list(rng.lognormal(np.log(8), cost_sigma, k) * cm)
        lab += [nm] * k
    perm = rng.permutation(M)
    return dict(
        p=np.clip(np.array(p)[perm], 0, 0.95),
        s=np.array(s)[perm],
        c=np.round(np.array(c)[perm], 2),
        arch=[lab[i] for i in perm],
    )


def make_world(seed, p_scale=0.036, d_hard=0.2, d_all=0.0):
    """мир задачи: опубликованный — seed 24, p_scale 0.036, d_hard 0.2"""
    return make_world_arch(seed, arch_rel(p_scale, d_hard, d_all))


def draw(rng, p, s, n_rel, r=R):
    """данные n_rel релизов: скрытые индикаторы проблем I (n_rel, M) и прогоны runs (n_rel, M, r)"""
    I = rng.random((n_rel, len(p))) < p
    runs = (rng.random((n_rel, len(p), r)) < s[None, :, None]) & I[:, :, None]
    return I, runs


def wilson(k, n, z=1.96):
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


# ----------------------------------------------------------------------------- вероятности
def g_tab(p, s):
    """g_j(n) = P(первые n прогонов теста j зелёные), столбцы n = 0..R"""
    return 1 - p[:, None] * (1 - (1 - s[:, None]) ** np.arange(R + 1))


def L(p, s, n):
    """pi(n): вероятность, что релиз проблемный, а все прогоны плана зелёные"""
    G = g_tab(p, s)
    return float(np.prod(G[np.arange(len(n)), n]) - np.prod(1 - p))


def p1_true(p):
    """P(релиз проблемный)"""
    return float(1 - np.prod(1 - p))


def qtab(p, s):
    """Q_j(n) = p_j (1 - s_j)^n: вероятность упустить проблему теста j при n прогонах"""
    return p[:, None] * (1 - s[:, None]) ** np.arange(R + 1)


def en(p, s, n):
    """ожидаемое число упущенных проблем на релизе"""
    return float((p * (1 - s) ** n).sum())


def missed_releases(runs, n):
    """по релизам: 1, если у какого-то теста есть красный среди записей, но в первых n_j записях всех тестов красных нет"""
    first = np.cumsum(runs, 2) > 0
    seen = np.where(n[None, :] > 0, first[:, np.arange(len(n)), np.maximum(n - 1, 0)], False).any(1)
    return runs.any((1, 2)) & ~seen
