#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flat PySR baselines for the surrogates of the paper.

  python pysr_baselines.py timedep   --data log3c.txt        --outdir out_td
  python pysr_baselines.py pointwise --data res_const_0.txt  --outdir out_pw

timedep   : ONE explicit model  C(x=0.3, t) = f(varying parameters, t)  (flat time-dependent baseline).
            Evaluation on whole parameter sets that are held out (train/test split by set).

pointwise : the pointwise GP/NN setting of the paper (Section "Surrogates for specific space-time points"):
            * data: ln C and ln P at 13 times x 5 locations = 65 points per parameter set (P := column U);
            * direct surrogates : one model per space-time point, ln C_ij = f(S)         (65 models);
            * inverse surrogates: one model per parameter, S_m = f(ln P_ij) for u_inp, kappa(ae), tau_r and
                                  S_m = f(ln C_ij) for the others                         (9 models);
              built on the full time range and (variant 'half') on t <= T/2 only;
            * train/test split of parameter sets 50/50, min-max scaling of all variables to [0, 1];
            * every model is trained 5 times independently (PySR runs with different seeds);
            * combined test: inverse surrogates -> parameters -> direct surrogates -> ln C and C at the
              65 points on the test sets, error = total relative error eps_tr of the paper.

Reporting separates what the paper's protocol does (best of the runs chosen on the TEST R2 - optimistic)
from selection that never looks at the test sets (best run by TRAIN R2) and from the mean over runs.

PySR cannot reproduce the GP/NN hybrid pipeline of the paper (no refinement on a larger or synthetic dataset,
no neural networks); this is a flat symbolic-regression baseline on the same data and the same split.
"""
import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
import sympy as sp

PARAMS = ["u_inp", "c_dur", "c_inp", "D", "kmu", "a", "b", "ae", "tr"]
P_BASED = {"u_inp", "ae", "tr"}      # inverse models trained on pressures in the paper
NAMES = {"a": "al", "b": "be", "ae": "kap", "tr": "tau_r", "u_inp": "u_inp",
         "c_dur": "c_dur", "c_inp": "c_inp", "D": "dd", "kmu": "kmu"}


# =========================================================================== common helpers
def load_log(path):
    """Header line 'u_inp ...' starts a parameter set; following lines are space-time points."""
    rows, sid, hdr = [], -1, {}
    with open(path) as f:
        for line in f:
            p = line.split()
            if not p:
                continue
            kv = dict(zip(p[0::2], map(float, p[1::2])))
            if p[0] == "u_inp":
                sid += 1
                hdr = kv
            else:
                rows.append({"set": sid, **hdr, **kv})
    return pd.DataFrame(rows)


def r2(y, yh):
    y, yh = np.asarray(y, float), np.nan_to_num(np.asarray(yh, float), nan=float(np.mean(y)),
                                                 posinf=float(np.mean(y)), neginf=float(np.mean(y)))
    ss = np.sum((y - y.mean()) ** 2)
    return float(1 - np.sum((y - yh) ** 2) / ss) if ss > 0 else float("nan")


def eps_tr(v, vh):
    """total relative error of the paper: ||v - vh||_2 / ||v||_2"""
    vh = np.nan_to_num(np.asarray(vh, float), nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.linalg.norm(v - vh) / np.linalg.norm(v))


def fit_pysr(X, y, args, niterations, maxsize, seed):
    """Runs PySR; returns the Pareto front [{complexity, loss, expr(sympy)}].
    The ONLY function that touches PySR - adapt keyword names here if your PySR version differs."""
    from pysr import PySRRegressor

    unary = list(args.unary)
    nested = {}
    for op in unary:
#        if op in ("sin", "cos"):
#            nested[op] = {"sin": 0, "cos": 0}
        if op in ("sin"):
            nested[op] = {"sin": 0}
        if op in ("cos"):
            nested[op] = { "cos": 0}
        if op == "exp":
            nested[op] = {"exp": 0}
    print(unary)
    print(nested)
    kw = dict(niterations=niterations, maxsize=maxsize, populations=args.populations,
              population_size=args.population_size, parsimony=args.parsimony,
              binary_operators=["+", "-", "*", "/", "^"], unary_operators=unary,
              constraints={"^": (-1, 3)}, nested_constraints=nested or None,
              elementwise_loss="loss(x, y) = (x - y)^2", random_state=seed,
              verbosity=0, progress=False, temp_equation_file=True, delete_tempfiles=True)
    if args.deterministic:
        kw.update(deterministic=True, parallelism="serial")
    try:
        model = PySRRegressor(**kw)
    except TypeError:  # older PySR versions
        if args.deterministic:
            kw.pop("parallelism", None)
            kw.update(procs=0, multithreading=False)
        model = PySRRegressor(**kw)
    model.fit(X, y)
    return [{"complexity": int(r["complexity"]), "loss": float(r["loss"]), "expr": r["sympy_format"]}
            for _, r in model.equations_.iterrows()]


def pick_best(front):
    """PySR's 'best' rule: max score (loss drop per complexity) among equations with loss <= 1.5 * min loss."""
    front = sorted(front, key=lambda r: r["complexity"])
    lo = min(r["loss"] for r in front)
    ok = [r for r in front if r["loss"] <= 1.5 * lo]
    best, best_score, prev = ok[-1], -np.inf, None
    for r in front:
        if prev is not None and r["loss"] > 0 and prev["loss"] > 0 and r["complexity"] > prev["complexity"]:
            sc = -np.log(r["loss"] / prev["loss"]) / (r["complexity"] - prev["complexity"])
            if r in ok and sc > best_score:
                best, best_score = r, sc
        prev = r
    return best


def make_func(expr, names):
    """sympy expression -> numpy callable f(*columns); NaN/complex values become NaN."""
    syms = [sp.Symbol(n) for n in names]
    expr = expr.xreplace({s: sp.Symbol(s.name) for s in expr.free_symbols})
    f = sp.lambdify(syms, expr, "numpy")

    def g(*cols):
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = np.asarray(f(*cols))
            if np.iscomplexobj(out):
                out = np.where(np.abs(out.imag) < 1e-12, out.real, np.nan)
            out = out.astype(float)
        return out * np.ones_like(np.asarray(cols[0], float))
    return g


def func_from_string(eq, names):
    return make_func(sp.sympify(eq, locals={n: sp.Symbol(n) for n in names}), names)


class MinMax:
    """Column-wise min-max scaling to [0, 1]."""

    def __init__(self, A):
        self.lo, self.hi = A.min(0), A.max(0)
        self.rng = np.where(self.hi - self.lo > 0, self.hi - self.lo, 1.0)

    def fwd(self, A):
        return (A - self.lo) / self.rng

    def inv(self, A):
        return A * self.rng + self.lo


# =========================================================================== TIME-DEPENDENT (flat)
def run_timedep(args):
    os.makedirs(args.outdir, exist_ok=True)
    df = load_log(args.data)
    all_p = [c for c in PARAMS if c in df.columns]
    per_set = df.groupby("set")[all_p].first()
    varying = [c for c in all_p if per_set[c].std() > 1e-12]
    pn = [NAMES[c] for c in varying]
    sets = {}
    for sid, g in df.sort_values("t").groupby("set"):
        g = g[g["x"] == args.x] if "x" in g and g["x"].nunique() > 1 else g
        sets[int(sid)] = {"t": g["t"].to_numpy(float), "C": g["C"].to_numpy(float),
                          "p": g[varying].iloc[0].to_numpy(float)}
    bad = [i for i, s in sets.items() if (not np.all(np.isfinite(s["C"]))) or s["C"].min() <= 0
           or s["C"].max() > args.c_max]
    for i in bad:
        del sets[i]
    print("%d parameter sets (%d removed as corrupted), varying: %s" % (len(sets), len(bad), varying))
    ids = np.array(sorted(sets))
    perm = np.random.default_rng(args.seed).permutation(ids)
    n_te = int(round(args.test_frac * len(ids)))
    te, tr = perm[:n_te], perm[n_te:]
    print("train sets %d, held-out test sets %d (split by parameter set)" % (len(tr), len(te)))

    cmax = max(sets[i]["C"].max() for i in tr)                     # scaling from TRAIN only
    ln = args.target == "logC"
    if ln:
        L = np.concatenate([np.log(sets[i]["C"]) for i in tr])
        mu, sd = L.mean(), L.std()
    fwd = (lambda C: (np.log(C) - mu) / sd) if ln else (lambda C: C / cmax)
    inv = (lambda y: np.exp(np.clip(y * sd + mu, -80, 20))) if ln else (lambda y: y * cmax)

    names = pn + ["t"]
    def stack(ids_):
        X = np.vstack([np.column_stack([np.tile(sets[i]["p"], (len(sets[i]["t"]), 1)), sets[i]["t"]]) for i in ids_])
        return pd.DataFrame(X, columns=names), np.concatenate([fwd(sets[i]["C"]) for i in ids_])
    X, y = stack(tr)
    front = fit_pysr(X, y, args, args.niterations, args.maxsize, args.seed)
    best = pick_best(front)
    f = make_func(best["expr"], names)
    print("Pareto front (complexity, MSE of scaled target):")
    for r in sorted(front, key=lambda r: r["complexity"]):
        print("  %3d  %.3e  %s%s" % (r["complexity"], r["loss"], r["expr"], "   <-- selected" if r is best else ""))

    def evaluate(ids_):
        out = []
        for i in ids_:
            s = sets[i]
            Ch = inv(f(*([np.full_like(s["t"], v) for v in s["p"]] + [s["t"]])))
            out.append({"set": int(i), "r2": r2(s["C"], Ch), "eps_tr": eps_tr(s["C"], Ch)})
        return out
    res = {"equation": str(best["expr"]), "target": args.target}
    for lab, ids_ in (("train (in-sample)", tr), ("TEST (held-out sets)", te)):
        m = evaluate(ids_)
        R = np.array([a["r2"] for a in m]); E = 100 * np.array([a["eps_tr"] for a in m])
        res[lab] = {"n": len(m), "R2_mean": float(np.nanmean(R)), "R2_median": float(np.nanmedian(R)),
                    "R2>0.9_%": float(100 * np.mean(R > 0.9)), "eps_mean_%": float(E.mean()),
                    "eps_median_%": float(np.median(E)), "eps_best_%": float(E.min())}
        print("  %-22s n=%3d  R2 mean %.3f median %.3f  R2>0.9 %5.1f%%  eps mean %.2f%% median %.2f%% best %.2f%%"
              % (lab, len(m), res[lab]["R2_mean"], res[lab]["R2_median"], res[lab]["R2>0.9_%"],
                 res[lab]["eps_mean_%"], res[lab]["eps_median_%"], res[lab]["eps_best_%"]))
        if lab.startswith("TEST"):
            pd.DataFrame(m).to_csv(os.path.join(args.outdir, "timedep_test_metrics.csv"), index=False)
    json.dump({"args": vars(args), "train_sets": tr.tolist(), "test_sets": te.tolist(), **res},
              open(os.path.join(args.outdir, "timedep_results.json"), "w"), indent=2, default=str)


# =========================================================================== POINTWISE
class PointData:
    """Pointwise dataset in the form used in the paper."""

    def __init__(self, args):
        df = load_log(args.data)
        n_sets = df["set"].nunique()
        df = df.sort_values(["set", "t", "x"])
        self.ts, self.xs = sorted(df["t"].unique()), sorted(df["x"].unique())
        npts = len(self.ts) * len(self.xs)
        cnt = df.groupby("set").size()
        if (cnt != npts).any():
            raise ValueError("every parameter set must have %d (t, x) points" % npts)
        ids = np.array(sorted(df["set"].unique()))
        C = df["C"].to_numpy(float).reshape(len(ids), npts)
        P = df[args.pressure_col].to_numpy(float).reshape(len(ids), npts)
        S = df.groupby("set")[PARAMS].first().loc[ids].to_numpy(float)
        ok = (np.isfinite(C).all(1) & np.isfinite(P).all(1) & (C > 0).all(1) & (C <= args.c_max).all(1)
              & (P > 0).all(1))
        print("%d parameter sets in file; %d removed as corrupted/non-positive (C<=0, C>%g, P<=0 or non-finite);"
              " %d kept" % (n_sets, (~ok).sum(), args.c_max, ok.sum()))
        self.S, self.lnC, self.lnP = S[ok], np.log(C[ok]), np.log(P[ok])
        self.tcol = np.repeat(self.ts, len(self.xs))               # time of every column (t-major)
        self.ti = np.repeat(np.arange(len(self.ts)), len(self.xs))
        self.xi = np.tile(np.arange(len(self.xs)), len(self.ts))
        self.n, self.npts = ok.sum(), npts
        varying = [PARAMS[k] for k in range(len(PARAMS)) if self.S[:, k].std() > 1e-12]
        if len(varying) != len(PARAMS):
            print("WARNING: non-varying parameters:", sorted(set(PARAMS) - set(varying)))


def run_pointwise(args):
    os.makedirs(os.path.join(args.outdir, "models"), exist_ok=True)
    D = PointData(args)
    perm = np.random.default_rng(args.seed).permutation(D.n)
    n_tr = int(round(args.train_frac * D.n))
    tr, te = perm[:n_tr], perm[n_tr:]
    print("train sets %d, test sets %d (split by parameter set, seed %d); scaling on %s"
          % (len(tr), len(te), args.seed, args.scale_on))
    fit_rows = tr if args.scale_on == "train" else np.arange(D.n)   # the paper scales on the complete dataset
    sS, sC, sP = MinMax(D.S[fit_rows]), MinMax(D.lnC[fit_rows]), MinMax(D.lnP[fit_rows])
    S_sc, C_sc, P_sc = sS.fwd(D.S), sC.fwd(D.lnC), sP.fwd(D.lnP)
    pn = [NAMES[p] for p in PARAMS]

    def feat_names(prefix, cols):
        return ["%s_%d_%d" % (prefix, D.ti[c], D.xi[c]) for c in cols]

    # ---- task definitions
    tasks = {}
    for c in range(D.npts):
        tasks["direct_t%02d_x%d" % (D.ti[c], D.xi[c])] = {"kind": "direct", "col": c}
    for variant in args.variants:
        cols = np.arange(D.npts) if variant == "full" else np.where(D.tcol <= args.half_tmax)[0]
        for m, p in enumerate(PARAMS):
            tasks["inverse_%s_%s" % (variant, p)] = {"kind": "inverse", "param": m, "variant": variant,
                                                      "cols": cols, "src": "P" if p in P_BASED else "C"}
    todo = [t for t in tasks if args.only == "all" or tasks[t]["kind"] == args.only]

    def data_for(tid, rows):
        T = tasks[tid]
        if T["kind"] == "direct":
            return S_sc[rows], C_sc[rows, T["col"]], pn
        A = (P_sc if T["src"] == "P" else C_sc)[rows][:, T["cols"]]
        return A, S_sc[rows, T["param"]], feat_names("p" if T["src"] == "P" else "c", T["cols"])

    # ---- train every model `runs` times
    for k, tid in enumerate(todo):
        path = os.path.join(args.outdir, "models", tid + ".json")
        if os.path.exists(path):
            continue                                                  # resume
        Xtr, ytr, names = data_for(tid, tr)
        Xte, yte, _ = data_for(tid, te)
        runs = []
        for r in range(args.runs):
            front = fit_pysr(pd.DataFrame(Xtr, columns=names), ytr, args, args.niterations, args.maxsize,
                             args.seed + 7919 * r)
            b = pick_best(front)
            f = make_func(b["expr"], names)
            runs.append({"run": r, "equation": str(b["expr"]), "complexity": b["complexity"],
                         "r2_train": r2(ytr, f(*Xtr.T)), "r2_test": r2(yte, f(*Xte.T))})
        json.dump(runs, open(path, "w"), indent=1)
        print("[%d/%d] %-22s test R2 of runs: %s" % (k + 1, len(todo), tid,
              " ".join("%.2f" % q["r2_test"] for q in runs)), flush=True)

    models = {t: json.load(open(os.path.join(args.outdir, "models", t + ".json"))) for t in todo}

    def choose(runs, rule):
        if rule == "train":
            return max(runs, key=lambda q: q["r2_train"])
        return max(runs, key=lambda q: q["r2_test"])               # paper protocol (looks at the test set)

    def stats(vals):
        v = np.array(vals)
        return {"n": len(v), "R2_mean": float(v.mean()), "R2_median": float(np.median(v)),
                ">0.7_%": float(100 * np.mean(v > 0.7)), ">0.8_%": float(100 * np.mean(v > 0.8)),
                ">0.9_%": float(100 * np.mean(v > 0.9))}

    summary = {}
    print("\n================ test R2 of individual models (analogue of Tables 2a/2b) ================")
    groups = {"direct": [t for t in todo if t.startswith("direct")]}
    for variant in args.variants:
        groups["inverse_" + variant] = [t for t in todo if t.startswith("inverse_" + variant)]
    for gname, tl in groups.items():
        if not tl:
            continue
        rules = {"mean over runs": [np.mean([q["r2_test"] for q in models[t]]) for t in tl],
                 "best run by TRAIN R2 (no test leakage)": [choose(models[t], "train")["r2_test"] for t in tl],
                 "best run by TEST R2 (paper protocol, optimistic)": [choose(models[t], "test")["r2_test"] for t in tl]}
        summary[gname] = {}
        print("\n%s (%d models)" % (gname, len(tl)))
        for rn, vals in rules.items():
            s = stats(vals)
            summary[gname][rn] = s
            print("  %-50s R2 mean %.3f median %.3f | >0.7: %5.1f%%  >0.8: %5.1f%%  >0.9: %5.1f%%"
                  % (rn, s["R2_mean"], s["R2_median"], s[">0.7_%"], s[">0.8_%"], s[">0.9_%"]))
        if gname.startswith("inverse"):
            print("  per parameter, test R2 (mean over runs / best-by-train / best-by-test):")
            for t in tl:
                q = models[t]
                print("    %-8s %6.3f / %6.3f / %6.3f" % (t.split("_", 2)[2], np.mean([x["r2_test"] for x in q]),
                      choose(q, "train")["r2_test"], choose(q, "test")["r2_test"]))

    # ---- combined test: inverse -> direct on the test sets
    if args.only == "all":
        print("\n================ combined test: inverse -> direct on test sets (analogue of Tables 3a-3c) ==========")
        summary["combined"] = {}
        names_direct = {t: data_for(t, te)[2] for t in groups["direct"]}
        for variant in args.variants:
            for rule in ("train", "test"):
                Shat = np.zeros((len(te), len(PARAMS)))
                for m, p in enumerate(PARAMS):
                    tid = "inverse_%s_%s" % (variant, p)
                    run = choose(models[tid], rule)
                    Xte, _, nm = data_for(tid, te)
                    Shat[:, m] = func_from_string(run["equation"], nm)(*Xte.T)
                Shat = np.nan_to_num(Shat, nan=0.5)
                if not args.no_clip:
                    Shat = np.clip(Shat, 0.0, 1.0)                     # physical range of the parameters
                lnC_hat = np.zeros((len(te), D.npts))
                for c in range(D.npts):
                    tid = "direct_t%02d_x%d" % (D.ti[c], D.xi[c])
                    run = choose(models[tid], rule)
                    lnC_hat[:, c] = func_from_string(run["equation"], pn)(*Shat.T)
                S_phys, lnC_hat = sS.inv(Shat), sC.inv(lnC_hat)
                with np.errstate(all="ignore"):
                    C_hat = np.exp(np.clip(lnC_hat, -80, 20))
                eS = np.array([eps_tr(D.S[i], S_phys[j]) for j, i in enumerate(te)]) * 100
                eL = np.array([eps_tr(D.lnC[i], lnC_hat[j]) for j, i in enumerate(te)]) * 100
                eC = np.array([eps_tr(np.exp(D.lnC[i]), C_hat[j]) for j, i in enumerate(te)]) * 100
                key = "inverse '%s' + direct, selection by %s R2" % (variant, rule.upper())
                summary["combined"][key] = {}
                print("\n%s%s" % (key, "  (optimistic: paper protocol)" if rule == "test" else ""))
                for nm_, e in (("parameters S", eS), ("ln C (65 points)", eL), ("C (65 points)", eC)):
                    s = {"worst_%": float(e.max()), "mean_%": float(e.mean()), "median_%": float(np.median(e)),
                         "best_%": float(e.min())}
                    summary["combined"][key][nm_] = s
                    print("   eps_tr %-18s worst %8.2f%%  mean %8.2f%%  median %7.2f%%  best %6.2f%%"
                          % (nm_, s["worst_%"], s["mean_%"], s["median_%"], s["best_%"]))
                pd.DataFrame({"set_index": te, "eps_S_%": eS, "eps_lnC_%": eL, "eps_C_%": eC}).to_csv(
                    os.path.join(args.outdir, "combined_%s_%s.csv" % (variant, rule)), index=False)

    rows = [{"task": t, **{k: q[k] for k in ("run", "equation", "complexity", "r2_train", "r2_test")}}
            for t in todo for q in models[t]]
    pd.DataFrame(rows).to_csv(os.path.join(args.outdir, "pointwise_all_runs.csv"), index=False)
    json.dump({"args": vars(args), "train_sets": tr.tolist(), "test_sets": te.tolist(), "summary": summary},
              open(os.path.join(args.outdir, "pointwise_summary.json"), "w"), indent=2, default=str)
    print("\nresults saved to", args.outdir)


# =========================================================================== CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--data", required=True)
        p.add_argument("--outdir", default="pysr_out")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--c-max", type=float, default=1.5,
                       help="sets with any C above this (numerical blow-up) are removed")
        p.add_argument("--unary", nargs="*", default=["exp", "sin", "cos", "sqrt", "log"])
        p.add_argument("--populations", type=int, default=24)
        p.add_argument("--population-size", type=int, default=40)
        p.add_argument("--parsimony", type=float, default=0.003)
        p.add_argument("--deterministic", action="store_true", help="serial, reproducible, slow")

    t = sub.add_parser("timedep", help="flat time-dependent surrogate C(S, t) at one point")
    common(t)
    t.add_argument("--target", choices=["C", "logC"], default="C")
    t.add_argument("--test-frac", type=float, default=0.3)
    t.add_argument("--x", type=float, default=0.3, help="space point if the file has several")
    t.add_argument("--niterations", type=int, default=200)
    t.add_argument("--maxsize", type=int, default=40)
    t.set_defaults(func=run_timedep)

    p = sub.add_parser("pointwise", help="pointwise direct/inverse surrogates as in the paper")
    common(p)
    p.add_argument("--train-frac", type=float, default=0.5, help="paper: 50/50")
    p.add_argument("--scale-on", choices=["train", "full"], default="train",
                   help="'full' = min-max on the complete dataset, as in the paper (leaks test ranges)")
    p.add_argument("--runs", type=int, default=5, help="independent runs per model, as in the paper")
    p.add_argument("--variants", nargs="+", choices=["full", "half"], default=["full", "half"],
                   help="inverse models on all times / on t <= --half-tmax")
    p.add_argument("--half-tmax", type=float, default=1.5, help="T/2; use 1.51 to include t=1.501")
    p.add_argument("--only", choices=["all", "direct", "inverse"], default="all")
    p.add_argument("--pressure-col", default="U", help="column used as P")
    p.add_argument("--no-clip", action="store_true", help="do not clip restored parameters to the data range")
    p.add_argument("--niterations", type=int, default=100)
    p.add_argument("--maxsize", type=int, default=25)
    p.set_defaults(func=run_pointwise)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
