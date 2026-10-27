"""Числа для документа решения (результаты пишутся в results/).

  python3 -m evals_task.numbers zoo        — зоопарк на 60 мирах, условный критерий (results/relc_zoo.txt)
  python3 -m evals_task.numbers w24        — надёжность методов на 200 историях мира 24 (results/relc_w24.txt)
  python3 -m evals_task.numbers students   — студенческие подстановки без модели и вырожденный план (results/relc_students.txt)
  python3 -m evals_task.numbers extra      — наивная подстановка с целью alpha/2 и проигрыш жадного плана точному (results/relc_extra.txt)
  python3 -m evals_task.numbers fixed      — смещение и погрешность оценок pi при фиксированном плане (results/rel_numbers_fixed.txt)
  python3 -m evals_task.numbers levels     — надёжность бутстрепа при разных уровнях и B (results/rel_bootsel_levels_60worlds.txt)
"""
import os, sys
import numpy as np

from .model import R, make_world, draw, L, g_tab, missed_releases, wilson
from .estimators import est_mom, est_mle0, HK
from .plans import chain_model, first_ok, p1_hat, plan_delta, plan_bootsel, oracle_rel, se_delta_mom, se_delta_mle, V_freq
from .zoo import evaluate, report, student_plans

PS, DH, A, K = 0.036, 0.2, 0.1, 3
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def tee(name):
    os.makedirs(OUT, exist_ok=True); f = open(os.path.join(OUT, name), "w")
    def pr(*a):
        line = " ".join(str(x) for x in a); print(line, flush=True); f.write(line + "\n"); f.flush()
    return pr


def run_zoo():
    r = evaluate(lambda sd: make_world(sd, PS, DH), A, range(60), quick=False, cond=True)
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report(r, A, K, "CONDITIONAL criterion, full zoo, 60 worlds (column L = true P(miss | problem release))")
    print(buf.getvalue()); open(os.path.join(OUT, "relc_zoo.txt"), "w").write(buf.getvalue())


def run_w24():
    pr = tee("relc_w24.txt")
    w = make_world(24, PS, DH); p, s, c = w["p"], w["s"], w["c"]; c5 = c.sum() * R; P1 = 1 - np.prod(1 - p); out = {}
    for h in range(200):
        X = draw(np.random.default_rng(400000 + h), p, s, 70)[1].sum(2); pm, sm = est_mom(X); Am = A * p1_hat(pm)
        plans = {"naive_mom_L": first_ok(chain_model(pm, sm, c), lambda n: L(pm, sm, n), Am), "delta_mom_L": plan_delta(X, c, "mom", Am),
                 "bootsel_mom_L": plan_bootsel(X, c, "mom", Am, np.random.default_rng(h)), "bootsel95_mom_L": plan_bootsel(X, c, "mom", Am, np.random.default_rng(h), B=100, level=0.95)}
        for k, n in plans.items():
            out.setdefault(k, []).append((L(p, s, n) / P1 <= A, c @ n / c5))
    for k, v in out.items():
        v = np.array(v, float); lo, hi = wilson(int(v[:, 0].sum()), len(v))
        pr(f"world 24, 200 histories, conditional target: {k}: P(ok) {v[:, 0].mean():.3f} ({lo:.2f}-{hi:.2f}), mean cost {v[:, 1].mean():.3f}")


def run_students():
    pr = tee("relc_students.txt"); res = {}
    for sd in range(60):
        w = make_world(sd, PS, DH); p, s, c = w["p"], w["s"], w["c"]; c5 = c.sum() * R; P1 = 1 - np.prod(1 - p)
        runs = draw(np.random.default_rng(1000 + sd), p, s, 70)[1]; val = draw(np.random.default_rng(5000 + sd), p, s, 30)[1]
        plans = student_plans(runs, c, (0.05, 0.1), cond=True)
        freq = runs.any(2).sum(0); order = np.argsort(-freq, kind="stable"); n = np.zeros(40, int); n[order[:15]] = 3; plans["deg_top15x3"] = n
        for k, n in plans.items():
            v = L(p, s, n) / P1; res.setdefault(k, []).append((v, v <= A, c @ n / c5, missed_releases(val, n).sum()))
    pr(f"{'plan':>16} {'cond pi':>7} {'P(ok)':>5} {'cost':>5} {'val missed':>10} {'P(pass K=3)':>11}")
    for k, v in res.items():
        v = np.array(v, float); pr(f"{k:>16} {v[:,0].mean():7.3f} {v[:,1].mean():5.2f} {v[:,2].mean():5.2f} {v[:,3].mean():10.2f} {np.mean(v[:,3] <= K):11.2f}")


def run_extra():
    pr = tee("relc_extra.txt"); half, gap = [], []
    for sd in range(60):
        w = make_world(sd, PS, DH); p, s, c = w["p"], w["s"], w["c"]; c5 = c.sum() * R; P1 = 1 - np.prod(1 - p)
        X = draw(np.random.default_rng(1000 + sd), p, s, 70)[1].sum(2); val = draw(np.random.default_rng(5000 + sd), p, s, 30)[1]
        pe, se = est_mom(X); n = first_ok(chain_model(pe, se, c), lambda m: L(pe, se, m), 0.05 * p1_hat(pe))
        v = L(p, s, n) / P1; half.append((v, v <= 0.1, c @ n / c5, missed_releases(val, n).sum()))
        if sd < 20:
            o = oracle_rel(p, s, c, 0.1 * P1); g = first_ok(chain_model(p, s, c), lambda m: L(p, s, m), 0.1 * P1); gap.append(c @ g / (c @ o) - 1)
    h = np.array(half, float).mean(0)
    pr("naive MoM at cond alpha/2 (0.05), 60 worlds: pi_c %.3f, P(ok) %.2f, cost %.3f, val missed %.2f" % tuple(h))
    pr("greedy over exact MILP at cond target, 20 worlds: mean %.3f max %.3f" % (np.mean(gap), np.max(gap)))


def run_fixed():
    """оценки pi(n) при фиксированном плане (безусловная цель 0.1 для плана оракула): смещение, разброс, калибровка погрешности"""
    pr = tee("rel_numbers_fixed.txt"); rows = {}
    for sd in range(10):
        w = make_world(sd, PS, DH); p, s, c = w["p"], w["s"], w["c"]; M = len(p)
        plans = {"all2": np.full(M, 2), "oracle": oracle_rel(p, s, c, A)}
        rng = np.random.default_rng(600 + sd)
        for h in range(60):
            X = draw(rng, p, s, 70)[1].sum(2); pm, sm = est_mom(X); pl, sl = est_mle0(X)
            for name, n in plans.items():
                Lt = L(p, s, n)
                vals = np.prod(HK[X, n[None, :]], 1) - (X.sum(1) == 0); sef = vals.std(ddof=1) / np.sqrt(len(vals))
                for estn, e, se_ in (("MoM", L(pm, sm, n), se_delta_mom(X, n)), ("MLE", L(pl, sl, n), se_delta_mle(X, n)), ("freq", V_freq(X, n), sef)):
                    rows.setdefault((name, estn), []).append((e, se_, Lt, sd))
    for (name, estn), v in rows.items():
        v = np.array(v, float); out = []
        for sd in range(10):
            u = v[v[:, 3] == sd]
            w = make_world(sd, PS, DH); p, s = w["p"], w["s"]; M = len(p)
            n = {"all2": np.full(M, 2), "oracle": oracle_rel(p, s, w["c"], A)}[name]
            G = g_tab(p, s); tgt = np.prod(G[np.arange(M), n]) - np.prod(G[:, 5]) if estn == "freq" else u[0, 2]   # частоты оценивают видимую часть
            out.append((np.mean(u[:, 0]) / tgt - 1, np.std(u[:, 0]) / tgt, np.mean(u[:, 1]) / np.std(u[:, 0]), np.mean(u[:, 0] + 1.2816 * u[:, 1] >= tgt)))
        out = np.array(out).mean(0)
        pr(f"{name:>7} {estn:>5}: bias {out[0]:+.0%}, sd {out[1]:.0%}, est.se / true sd {out[2]:.2f}, coverage of 90% bound {out[3]:.2f}")


def run_levels():
    pr = tee("rel_bootsel_levels_60worlds.txt"); out = {k: [] for k in ("l90_B40", "l95_B40", "l95_B100")}
    for sd in range(60):
        w = make_world(sd, PS, DH); p, s, c = w["p"], w["s"], w["c"]; c5 = c.sum() * R; P1 = 1 - np.prod(1 - p)
        X = draw(np.random.default_rng(1000 + sd), p, s, 70)[1].sum(2); Am = A * p1_hat(est_mom(X)[0])
        for k, (lvl, B) in zip(out, ((0.9, 40), (0.95, 40), (0.95, 100))):
            n = plan_bootsel(X, c, "mom", Am, np.random.default_rng(sd + 2), B=B, level=lvl); out[k].append((L(p, s, n) / P1 <= A, c @ n / c5))
    for k, v in out.items():
        v = np.array(v, float); pr(f"60 worlds, conditional target, {k}: P(ok) {v[:, 0].mean():.3f}, cost {v[:, 1].mean():.3f}")


if __name__ == "__main__":
    modes = dict(zoo=run_zoo, w24=run_w24, students=run_students, extra=run_extra, fixed=run_fixed, levels=run_levels)
    if len(sys.argv) < 2 or sys.argv[1] not in modes:
        sys.exit(__doc__)
    modes[sys.argv[1]]()
