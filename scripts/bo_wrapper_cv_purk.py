"""
Bayesian optimization wrapper for CV tuning using validate_myocardial_mesh_smoke.py.
"""

from __future__ import annotations

import argparse
import logging
import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EvalResult:
    cv_purk: float
    rmse_norm_mean: float
    rho_mean: float
    abs_rho_mean: float
    metrics_csv: Path
    sim_npz: Path
    run_dir: Path


def create_run_dir(base_out_dir: Path) -> Path:
    base_out_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M")

    run_idx = 1
    for child in base_out_dir.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if not name.startswith(f"{date_str}-"):
            continue
        if "-run" not in name:
            continue
        tail = name.rsplit("-run", 1)[-1]
        try:
            idx = int(tail)
        except ValueError:
            continue
        if idx >= run_idx:
            run_idx = idx + 1

    run_dir = base_out_dir / f"{date_str}-{time_str}-run{run_idx}"
    while run_dir.exists():
        run_idx += 1
        run_dir = base_out_dir / f"{date_str}-{time_str}-run{run_idx}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def configure_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("bo_wrapper_cv_purk")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def log_info(logger: logging.Logger, msg: str, *args) -> None:
    logger.log(logging.INFO, msg, *args)


def run_cmd(cmd: list[str], logger: logging.Logger, cwd: Optional[Path] = None) -> None:
    log_info(logger, "Running: %s", " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        shell=False,
        bufsize=1,
    )
    if proc.stdout is None:
        raise RuntimeError("Failed to capture process output.")
    for line in proc.stdout:
        log_info(logger, line.rstrip())
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"Command failed with exit code {ret}: {' '.join(cmd)}")


def find_single_run_dir(base_dir: Path) -> Path:
    run_dirs = [p for p in base_dir.iterdir() if p.is_dir()]
    if len(run_dirs) != 1:
        raise RuntimeError(
            f"Expected exactly one run directory in {base_dir}, found {len(run_dirs)}."
        )
    return run_dirs[0]


def read_metrics(metrics_csv: Path) -> tuple[float, float, float]:
    """
    Expect columns: rmse_norm, rho, abs_rho.
    Returns mean(rmse_norm), mean(rho), mean(abs_rho).
    """
    df = pd.read_csv(metrics_csv)
    required = {"rmse_norm", "rho", "abs_rho"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"metrics CSV missing columns: {sorted(missing)} in {metrics_csv}"
        )
    return (
        float(df["rmse_norm"].mean()),
        float(df["rho"].mean()),
        float(df["abs_rho"].mean()),
    )


def objective_value(
    rmse_norm_mean: float,
    rho_mean: float,
    objective: str,
    lambda_rho: float,
) -> float:
    """
    Always minimize.
    - rmse: objective = rmse_norm
    - mixed: objective = rmse_norm - lambda * rho_mean
    """
    if objective == "rmse":
        return rmse_norm_mean
    if objective == "mixed":
        return rmse_norm_mean - lambda_rho * rho_mean
    raise ValueError(f"Unknown objective={objective}")


def propose_next_cv_ei(
    xs: np.ndarray,
    ys: np.ndarray,
    cv_min: float,
    cv_max: float,
    rng: np.random.Generator,
    n_candidates: int = 256,
    xi: float = 0.01,
) -> float:
    """
    1D BO with GP(EI). Requires scikit-learn.
    """
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    except Exception as e:
        raise RuntimeError(
            "scikit-learn is required for the GP. Install: pip install scikit-learn"
        ) from e

    X = xs.reshape(-1, 1)
    y = ys.reshape(-1, 1)

    kernel = 1.0 * RBF(length_scale=0.5) + WhiteKernel(noise_level=1e-6)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=0)
    gp.fit(X, y.ravel())

    cand = rng.uniform(cv_min, cv_max, size=n_candidates)
    cand = np.unique(np.concatenate([cand, [cv_min, cv_max]]))
    C = cand.reshape(-1, 1)

    mu, sigma = gp.predict(C, return_std=True)
    sigma = np.maximum(sigma, 1e-12)

    y_best = float(np.min(ys))
    imp = y_best - mu - xi
    Z = imp / sigma

    cdf = 0.5 * (1.0 + np.vectorize(math.erf)(Z / math.sqrt(2.0)))
    pdf = (1.0 / math.sqrt(2.0 * math.pi)) * np.exp(-0.5 * Z * Z)

    ei = imp * cdf + sigma * pdf
    ei[sigma <= 1e-12] = 0.0

    best_idx = int(np.argmax(ei))
    return float(cand[best_idx])


def evaluate_cv(
    cv_purk: float,
    *,
    base: Path,
    obs_json: Path,
    smoke_script: Path,
    pmj_mode: str,
    metrics_mode: str,
    normalize: str,
    sim_dt: float,
    qrs_pre: float,
    qrs_post: float,
    baseline0: float,
    baseline1: float,
    lv_roots: list[int],
    rv_roots: list[int],
    out_dir: Path,
    npz_name: Optional[str],
    skip_plots: bool,
    logger: logging.Logger,
) -> EvalResult:
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(smoke_script),
        "--base",
        str(base),
        "--obs-json",
        str(obs_json),
        "--out-dir",
        str(out_dir),
        "--pmj-mode",
        pmj_mode,
        "--metrics-mode",
        metrics_mode,
        "--normalize",
        normalize,
        "--sim-dt",
        f"{sim_dt}",
        "--qrs-pre",
        f"{qrs_pre}",
        "--qrs-post",
        f"{qrs_post}",
        "--baseline0",
        f"{baseline0}",
        "--baseline1",
        f"{baseline1}",
        "--cv-purk",
        f"{cv_purk:.6f}",
        "--lv-roots",
        *[str(x) for x in lv_roots],
        "--rv-roots",
        *[str(x) for x in rv_roots],
    ]
    if npz_name:
        cmd.extend(["--npz-name", npz_name])
    if skip_plots:
        cmd.append("--skip-plots")

    run_cmd(cmd, logger)

    run_dir = find_single_run_dir(out_dir)
    metrics_name = "metrics_qrs_lsq.csv" if metrics_mode == "lsq" else "metrics_qrs.csv"
    metrics_csv = run_dir / metrics_name
    if not metrics_csv.exists():
        raise RuntimeError(f"Metrics did not produce expected CSV: {metrics_csv}")

    if npz_name:
        sim_npz = run_dir / npz_name
    else:
        sim_npz = run_dir / f"out_forward_ecg_{pmj_mode}.npz"

    rmse_norm_mean, rho_mean, abs_rho_mean = read_metrics(metrics_csv)
    return EvalResult(
        cv_purk=cv_purk,
        rmse_norm_mean=rmse_norm_mean,
        rho_mean=rho_mean,
        abs_rho_mean=abs_rho_mean,
        metrics_csv=metrics_csv,
        sim_npz=sim_npz,
        run_dir=run_dir,
    )


def save_convergence_plot(history_csv: Path, out_png: Path) -> None:
    import matplotlib.pyplot as plt

    df = pd.read_csv(history_csv)
    df = df.sort_values("eval_id")

    best_so_far = np.minimum.accumulate(df["objective"].to_numpy())
    iters = df["eval_id"].to_numpy()

    plt.figure()
    plt.plot(iters, best_so_far)
    plt.xlabel("Evaluation")
    plt.ylabel("Best-so-far objective")
    plt.title("BO convergence (best-so-far)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="BO wrapper for CV optimization using smoke script."
    )
    ap.add_argument(
        "--base",
        type=Path,
        required=True,
        help="Base data folder (Karli dataset root).",
    )
    ap.add_argument(
        "--obs-json",
        type=Path,
        required=True,
        help="Observed ECG mean-beat JSON (12-lead).",
    )
    ap.add_argument(
        "--smoke-script",
        type=Path,
        required=True,
        help="Path to validate_myocardial_mesh_smoke.py",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        required=True,
        help="Output folder (base; run subfolder created inside).",
    )

    ap.add_argument("--pmj-mode", choices=["tree", "zero"], default="tree")
    ap.add_argument("--metrics-mode", choices=["lsq", "plain"], default="lsq")
    ap.add_argument("--normalize", choices=["none", "rms", "z"], default="rms")
    ap.add_argument("--sim-dt", type=float, default=0.001)
    ap.add_argument("--qrs-pre", type=float, default=0.04)
    ap.add_argument("--qrs-post", type=float, default=0.12)
    ap.add_argument("--baseline0", type=float, default=-0.04)
    ap.add_argument("--baseline1", type=float, default=-0.02)
    ap.add_argument("--npz-name", type=str, default=None)
    ap.add_argument("--skip-plots", action="store_true")

    ap.add_argument("--lv-roots", type=int, nargs=2, default=[742, 984])
    ap.add_argument("--rv-roots", type=int, nargs=2, default=[282, 195])

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--objective", choices=["rmse", "mixed"], default="rmse")
    ap.add_argument("--lambda-rho", type=float, default=0.2)
    ap.add_argument("--cv-min", type=float, default=1.0)
    ap.add_argument("--cv-max", type=float, default=4.0)
    ap.add_argument("--baseline-cv", type=float, default=2.0)
    ap.add_argument("--n-init", type=int, default=3)
    ap.add_argument("--n-iter", type=int, default=10)

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    base = args.base.expanduser().resolve()
    obs_json = args.obs_json.expanduser()
    if not obs_json.is_absolute():
        obs_json = base / obs_json

    smoke_script = args.smoke_script.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()
    run_root = create_run_dir(out_root)

    log_path = run_root / "bo_wrapper_cv_purk.log"
    logger = configure_logger(log_path)

    log_info(logger, "Run output dir: %s", run_root)
    log_info(logger, "Log file: %s", log_path)
    log_info(logger, "Smoke script: %s", smoke_script)

    rng = np.random.default_rng(args.seed)

    baseline_dir = run_root / "baseline"
    log_info(logger, "Baseline CV: %.6f", args.baseline_cv)
    baseline = evaluate_cv(
        args.baseline_cv,
        base=base,
        obs_json=obs_json,
        smoke_script=smoke_script,
        pmj_mode=args.pmj_mode,
        metrics_mode=args.metrics_mode,
        normalize=args.normalize,
        sim_dt=args.sim_dt,
        qrs_pre=args.qrs_pre,
        qrs_post=args.qrs_post,
        baseline0=args.baseline0,
        baseline1=args.baseline1,
        lv_roots=list(args.lv_roots),
        rv_roots=list(args.rv_roots),
        out_dir=baseline_dir,
        npz_name=args.npz_name,
        skip_plots=args.skip_plots,
        logger=logger,
    )
    baseline_obj = objective_value(
        baseline.rmse_norm_mean,
        baseline.rho_mean,
        args.objective,
        args.lambda_rho,
    )

    rows = []
    rows.append(
        dict(
            eval_id=0,
            kind="baseline",
            cv_purk=baseline.cv_purk,
            rmse_norm_mean=baseline.rmse_norm_mean,
            rho_mean=baseline.rho_mean,
            abs_rho_mean=baseline.abs_rho_mean,
            objective=baseline_obj,
            metrics_csv=str(baseline.metrics_csv),
            sim_npz=str(baseline.sim_npz),
            run_dir=str(baseline.run_dir),
        )
    )

    xs = [baseline.cv_purk]
    ys = [baseline_obj]

    for i in range(1, args.n_init + 1):
        cv = float(rng.uniform(args.cv_min, args.cv_max))
        out_dir = run_root / f"eval_{i:03d}_init_cv_{cv:.6f}"
        log_info(logger, "Init eval %d CV: %.6f", i, cv)
        res = evaluate_cv(
            cv,
            base=base,
            obs_json=obs_json,
            smoke_script=smoke_script,
            pmj_mode=args.pmj_mode,
            metrics_mode=args.metrics_mode,
            normalize=args.normalize,
            sim_dt=args.sim_dt,
            qrs_pre=args.qrs_pre,
            qrs_post=args.qrs_post,
            baseline0=args.baseline0,
            baseline1=args.baseline1,
            lv_roots=list(args.lv_roots),
            rv_roots=list(args.rv_roots),
            out_dir=out_dir,
            npz_name=args.npz_name,
            skip_plots=args.skip_plots,
            logger=logger,
        )
        obj = objective_value(
            res.rmse_norm_mean, res.rho_mean, args.objective, args.lambda_rho
        )

        xs.append(res.cv_purk)
        ys.append(obj)

        rows.append(
            dict(
                eval_id=i,
                kind="init",
                cv_purk=res.cv_purk,
                rmse_norm_mean=res.rmse_norm_mean,
                rho_mean=res.rho_mean,
                abs_rho_mean=res.abs_rho_mean,
                objective=obj,
                metrics_csv=str(res.metrics_csv),
                sim_npz=str(res.sim_npz),
                run_dir=str(res.run_dir),
            )
        )

    for t in range(args.n_iter):
        eval_id = args.n_init + 1 + t

        cv_next = propose_next_cv_ei(
            xs=np.asarray(xs, dtype=float),
            ys=np.asarray(ys, dtype=float),
            cv_min=args.cv_min,
            cv_max=args.cv_max,
            rng=rng,
        )

        out_dir = run_root / f"eval_{eval_id:03d}_bo_cv_{cv_next:.6f}"
        log_info(logger, "BO eval %d CV: %.6f", eval_id, cv_next)
        res = evaluate_cv(
            cv_next,
            base=base,
            obs_json=obs_json,
            smoke_script=smoke_script,
            pmj_mode=args.pmj_mode,
            metrics_mode=args.metrics_mode,
            normalize=args.normalize,
            sim_dt=args.sim_dt,
            qrs_pre=args.qrs_pre,
            qrs_post=args.qrs_post,
            baseline0=args.baseline0,
            baseline1=args.baseline1,
            lv_roots=list(args.lv_roots),
            rv_roots=list(args.rv_roots),
            out_dir=out_dir,
            npz_name=args.npz_name,
            skip_plots=args.skip_plots,
            logger=logger,
        )
        obj = objective_value(
            res.rmse_norm_mean, res.rho_mean, args.objective, args.lambda_rho
        )

        xs.append(res.cv_purk)
        ys.append(obj)

        rows.append(
            dict(
                eval_id=eval_id,
                kind="bo",
                cv_purk=res.cv_purk,
                rmse_norm_mean=res.rmse_norm_mean,
                rho_mean=res.rho_mean,
                abs_rho_mean=res.abs_rho_mean,
                objective=obj,
                metrics_csv=str(res.metrics_csv),
                sim_npz=str(res.sim_npz),
                run_dir=str(res.run_dir),
            )
        )

    hist = pd.DataFrame(rows).sort_values("eval_id")
    history_csv = run_root / "bo_history.csv"
    hist.to_csv(history_csv, index=False)

    best_row = hist.loc[hist["objective"].idxmin()]
    summary = pd.DataFrame(
        [
            dict(
                label="baseline",
                cv_purk=baseline.cv_purk,
                rmse_norm_mean=baseline.rmse_norm_mean,
                rho_mean=baseline.rho_mean,
                abs_rho_mean=baseline.abs_rho_mean,
                objective=baseline_obj,
            ),
            dict(
                label="best",
                cv_purk=float(best_row["cv_purk"]),
                rmse_norm_mean=float(best_row["rmse_norm_mean"]),
                rho_mean=float(best_row["rho_mean"]),
                abs_rho_mean=float(best_row["abs_rho_mean"]),
                objective=float(best_row["objective"]),
            ),
        ]
    )
    summary_csv = run_root / "baseline_vs_best.csv"
    summary.to_csv(summary_csv, index=False)

    conv_png = run_root / "convergence_best_so_far.png"
    save_convergence_plot(history_csv, conv_png)

    log_info(logger, "History: %s", history_csv)
    log_info(logger, "Summary: %s", summary_csv)
    log_info(logger, "Convergence: %s", conv_png)
    log_info(logger, "Best row:\n%s", best_row.to_string(index=False))


if __name__ == "__main__":
    main()
