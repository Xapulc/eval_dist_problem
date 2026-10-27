"""Скоринг планов команд на скрытых релизах.

  python3 -m evals_task.score <data_dir> [<plan.csv> ...] [--exact]     (без планов — все CSV из <data_dir>/hidden/plans)

plan.csv: eval_id, n_runs (0..5; тесты, которых нет в файле, получают 0 прогонов — с предупреждением).
Валидация hidden/dataset_valid.csv: 30 релизов, у каждого теста записаны все 5 прогонов.
Релиз проблемный, если у какого-то теста среди записей есть красный. План ПРОПУСТИЛ его, если ни у одного теста нет
красного в первых n_j записях: план выпустил бы релиз как чистый. Для каждого плана печатается:
  missed releases — пропущено проблемных релизов (главная метрика) и сколько их всего; pass — не больше k_pass из params.json;
  missed problems — не пойманных отдельных проблем (справочно); cost — стоимость плана и её отношение к «все по разу» и «все по 5»;
  evals, runs — сколько тестов запускается и сколько прогонов.
Вывод ранжирован: прошедшие по стоимости, затем не прошедшие по числу пропусков.
--exact (для организаторов): истинные P(пропуск релиза), P(пропуск | релиз проблемный) и E N по params.json.
"""
import json, os, sys
import numpy as np
import pandas as pd

R = 5


def read_plan(path, M):
    pl = pd.read_csv(path)
    if not {"eval_id", "n_runs"} <= set(pl.columns):
        raise ValueError(f"{path}: columns eval_id and n_runs are required, got {list(pl.columns)}")
    if len(pl) == 0:
        raise ValueError(f"{path}: no rows")
    e = pd.to_numeric(pl.eval_id, errors="coerce")
    if e.isna().any() or (e != np.round(e)).any() or e.duplicated().any() or not set(e.astype(int)) <= set(range(M)):
        raise ValueError(f"{path}: eval_id must be distinct integers in 0..{M - 1}")
    v = pd.to_numeric(pl.n_runs, errors="coerce")
    if v.isna().any() or (v != np.round(v)).any() or (v < 0).any() or (v > R).any():
        raise ValueError(f"{path}: n_runs must be integers in 0..{R}")
    n = np.zeros(M, int); n[e.astype(int).values] = v.astype(int).values
    if len(pl) < M:
        print(f"warning: {path} lists {len(pl)} of {M} evals; the missing ones count as 0 runs", file=sys.stderr)
    return n


def load_runs(path, M):
    """(releases, evals, 5) bool array from a long CSV release_id, eval_id, run_no, red_flag"""
    df = pd.read_csv(path).sort_values(["release_id", "eval_id", "run_no"])
    return df.red_flag.values.reshape(df.release_id.nunique(), M, R).astype(bool)


def main(argv):
    args = [a for a in argv if a != "--exact"]; want_exact = "--exact" in argv
    if not args:
        sys.exit("usage: python3 -m evals_task.score <data_dir> [plan.csv ...] [--exact]")
    d, paths = args[0], args[1:]
    if not paths:
        pdir = os.path.join(d, "hidden", "plans"); paths = sorted(os.path.join(pdir, f) for f in os.listdir(pdir) if f.endswith(".csv"))
    c = pd.read_csv(os.path.join(d, "public", "evals.csv")).sort_values("eval_id").cost.values.astype(float); M = len(c)
    runs = load_runs(os.path.join(d, "hidden", "dataset_valid.csv"), M); first = np.cumsum(runs, 2) > 0
    prm = json.load(open(os.path.join(d, "hidden", "params.json"))); k_pass = prm.get("k_pass")
    has = runs.any(2); prob_rel = has.any(1); n_rel = runs.shape[0]
    print(f"validation: {n_rel} releases, {int(prob_rel.sum())} problematic (some eval red among its 5 records), "
          f"{int(has.sum())} problems in total")
    print(f"{'plan':>24} | {'missed releases':>15} {'pass':>4} | {'missed problems':>15} | {'cost':>7} {'/all once':>9} {'/all 5':>6} | {'evals':>5} {'runs':>5}"
          + (f" | {'P(miss rel)':>11} {'P(miss|prob)':>12} {'E N':>6}" if want_exact else "") + (f"   (pass: missed releases <= {k_pass})" if k_pass is not None else ""))
    if want_exact:
        p, s = np.array(prm["p"]), np.array(prm["s"])
    pdir = os.path.abspath(os.path.join(d, "hidden", "plans")); rows = []
    for path in paths:
        label = os.path.basename(path) if os.path.dirname(os.path.abspath(path)) == pdir else os.path.relpath(path)
        n = read_plan(path, M)
        caught = np.where(n[None, :] > 0, first[:, np.arange(M), np.maximum(n - 1, 0)], False)
        miss_rel = int((prob_rel & ~caught.any(1)).sum()); miss_prob = int((has & ~caught).sum()); cost = float(c @ n)
        passed = None if k_pass is None else miss_rel <= k_pass; verdict = "" if passed is None else ("yes" if passed else "no")
        line = (f"{label:>24} | {str(miss_rel) + ' / ' + str(int(prob_rel.sum())):>15} {verdict:>4} | "
                f"{str(miss_prob) + ' / ' + str(int(has.sum())):>15} | {cost:7.1f} {cost / c.sum():9.3f} {cost / c.sum() / R:6.3f} | "
                f"{int((n > 0).sum()):5d} {int(n.sum()):5d}")
        if want_exact:
            Lr = np.prod(1 - p * (1 - (1 - s) ** n)) - np.prod(1 - p); P1 = 1 - np.prod(1 - p)
            line += f" | {Lr:11.3f} {Lr / P1:12.3f} {(p * (1 - s) ** n).sum():6.3f}"
        rows.append((0 if passed else 1, cost if passed else miss_rel, line))   # прошедшие по цене, затем не прошедшие по пропускам
    for _, _, line in sorted(rows, key=lambda r: (r[0], r[1])):
        print(line)


if __name__ == "__main__":
    main(sys.argv[1:])
