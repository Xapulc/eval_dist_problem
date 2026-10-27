"""Отбор реализации данных и запись файлов задачи.

  python3 -m evals_task.generate scan [n]            — перебор реализаций истории и валидации мира 24 (см. критерии в scan)
  python3 -m evals_task.generate gen <k> <out_dir>   — записать реализацию k:
      public/evals.csv              eval_id, cost
      public/dataset_train.csv      release_id, eval_id, run_no, red_flag   (70 релизов)
      hidden/dataset_valid.csv      то же для 30 скрытых релизов, release_id 100..129
      hidden/problems_valid.csv     истинные индикаторы проблем на скрытых релизах
      hidden/params.json            истинные параметры, архетипы, критерий и порог прохождения
      hidden/plans/*.csv            планы всех решений зоопарка
Опубликованные данные: мир seed 24 (p_scale 0.036, d_hard 0.2), реализация 190; запасные — 146 и 153.
Стоимости зависят только от seed, поэтому evals.csv один и тот же для любых p и s.
"""
import json, os, sys
import numpy as np
import pandas as pd

from .model import R, make_world, draw, L, missed_releases, p1_true
from .plans import plan_bootsel
from .zoo import solutions

SEED, PS, DH = 24, 0.036, 0.2
A, K, COND = 0.1, 3, True  # требование P(пропуск | релиз проблемный) <= A; порог на валидации: пропущено <= K из 30
NTR, NVAL = 70, 30


def world():
    return make_world(SEED, PS, DH)


def data(k):
    w = world()
    p, s = w["p"], w["s"]
    I_tr, runs = draw(np.random.default_rng(100000 + k), p, s, NTR)
    I_v, val = draw(np.random.default_rng(200000 + k), p, s, NVAL)
    return w, runs, I_v, val


def scan(n=200):
    """реализации, на которых лестница решений видна и по истине, и на валидации, а стоимость эталона типична для мира"""
    w = world()
    p, s, c = w["p"], w["s"], w["c"]
    c5 = c.sum() * R
    ok = []
    typ = np.median(
        [
            c
            @ plan_bootsel(
                draw(np.random.default_rng(300000 + h), p, s, NTR)[1].sum(2), c, "mom", A, np.random.default_rng(h)
            )
            / c5
            for h in range(40)
        ]
    )
    print(f"world {SEED}: P(problem release) {p1_true(p):.3f}, typical reference cost {typ:.3f}", flush=True)
    for k in range(n):
        _, runs, _, val = data(k)
        sol = solutions(w, runs, A, k, quick=True, cond=COND)
        r = {m: (L(p, s, x), c @ x / c5, int(missed_releases(val, x).sum())) for m, x in sol.items()}
        bad = ("naive_freq_L", "naive_pooled_L", "naive_mom_L", "all1")
        good = ("bootsel_mom_L", "oracle")
        if not (
            all(r[m][0] > A for m in bad)
            and r["bootsel_mom_L"][0] <= A
            and all(r[m][2] > K for m in ("naive_freq_L", "all1"))
            and r["naive_mom_L"][2] >= K
            and all(r[m][2] <= K - 1 for m in good)
            and abs(r["bootsel_mom_L"][1] / typ - 1) <= 0.15
        ):
            continue
        score = (
            r["naive_freq_L"][2]
            + r["naive_pooled_L"][2]
            + r["naive_mom_L"][2]
            + r["delta_mom_L"][2]
            - 4 * r["bootsel_mom_L"][2]
        )
        ok.append((score, k, r))
        print(
            f"  k {k:3d} score {score}: " + ", ".join(f"{m} {v[0]:.3f}/{v[1]:.2f}/{v[2]}" for m, v in r.items()),
            flush=True,
        )
    ok.sort(key=lambda t: -t[0])
    print(f"\n{len(ok)} of {n} draws pass; best: {[t[1] for t in ok[:5]]}")


def generate(k, out):
    w, runs, I_v, val = data(k)
    p, s, c = w["p"], w["s"], w["c"]
    M = len(p)
    for sub in ("public", "hidden/plans"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)

    def rows(r, off):
        rel, t, j = np.meshgrid(np.arange(r.shape[0]), np.arange(M), np.arange(5), indexing="ij")
        return pd.DataFrame(
            dict(release_id=rel.ravel() + off, eval_id=t.ravel(), run_no=j.ravel(), red_flag=r.ravel().astype(int))
        )

    pd.DataFrame(dict(eval_id=range(M), cost=c)).to_csv(os.path.join(out, "public", "evals.csv"), index=False)
    rows(runs, 0).to_csv(os.path.join(out, "public", "dataset_train.csv"), index=False)
    rows(val, 100).to_csv(os.path.join(out, "hidden", "dataset_valid.csv"), index=False)
    rel, t = np.meshgrid(np.arange(NVAL), np.arange(M), indexing="ij")
    pd.DataFrame(dict(release_id=rel.ravel() + 100, eval_id=t.ravel(), problem=I_v.ravel().astype(int))).to_csv(
        os.path.join(out, "hidden", "problems_valid.csv"), index=False
    )
    P1 = p1_true(p)
    json.dump(
        dict(
            seed=SEED,
            draw=k,
            target="P(missed | release has a problem) <= A_cond" if COND else "P(missed problem release) <= A",
            A_cond=A if COND else None,
            A=A * P1 if COND else A,
            P_problem_release=round(P1, 5),
            k_pass=K,
            R=5,
            p_scale=PS,
            d_hard=DH,
            p=p.round(5).tolist(),
            s=s.round(5).tolist(),
            cost=c.tolist(),
            archetype=w["arch"],
        ),
        open(os.path.join(out, "hidden", "params.json"), "w"),
        ensure_ascii=False,
        indent=1,
    )
    for m, x in solutions(w, runs, A, k, cond=COND).items():
        pd.DataFrame(dict(eval_id=range(M), n_runs=x)).to_csv(
            os.path.join(out, "hidden", "plans", f"{m}.csv"), index=False
        )
    print(f"{out}: world {SEED}, draw {k}, P(problem release) {P1:.3f}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("scan", "gen"):
        sys.exit(__doc__)
    scan(int(sys.argv[2]) if len(sys.argv) > 2 else 200) if sys.argv[1] == "scan" else generate(
        int(sys.argv[2]), sys.argv[3]
    )
