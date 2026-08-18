"""
test_shp_calibration.py
=======================
SHP (Static Hydrogen Pool) 교정 독립 테스트 스크립트

기존 calibration_debug_data.xlsx 를 읽어 다음 세 가지 방법을 비교합니다.
  1. Standard  : 기존 결과 (고정 Wlat, N0만 최적화)
  2. SHP-Joint : N0 + a2 동시 RMSE 최적화 (data-driven)
  3. SHP-2pt   : 건조기/습윤기 2점 해석적 풀이

사용법:
    python test_shp_calibration.py --station HC
    python test_shp_calibration.py --station HC --output_dir ./shp_results
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ── 물리 상수 (Desilets et al.) ─────────────────────────────────────────────
A0 = 0.0808   # [m³/m³]  고정 (particle-physics universal constant)
A1 = 0.372    # [-]       고정


# ════════════════════════════════════════════════════════════════════════════
# 핵심 함수
# ════════════════════════════════════════════════════════════════════════════

def vwc_from_N(N: np.ndarray, N0: float, a2: float) -> np.ndarray:
    """Desilets 역산: θ = a0 / (N/N0 − a1) − a2"""
    denom = N / N0 - A1
    with np.errstate(divide='ignore', invalid='ignore'):
        theta = np.where(denom > 0, A0 / denom - a2, np.nan)
    return theta


def rmse(obs: np.ndarray, pred: np.ndarray) -> float:
    valid = np.isfinite(obs) & np.isfinite(pred)
    if valid.sum() < 2:
        return np.nan
    return float(np.sqrt(np.mean((obs[valid] - pred[valid]) ** 2)))


def metrics(obs: np.ndarray, pred: np.ndarray) -> dict:
    valid = np.isfinite(obs) & np.isfinite(pred)
    o, p = obs[valid], pred[valid]
    if len(o) < 2:
        return dict(RMSE=np.nan, MAE=np.nan, Bias=np.nan, R=np.nan, n=0)
    r_val = float(np.corrcoef(o, p)[0, 1])
    return dict(
        RMSE=float(np.sqrt(np.mean((o - p) ** 2))),
        MAE=float(np.mean(np.abs(o - p))),
        Bias=float(np.mean(p - o)),
        R=r_val,
        n=int(len(o)),
    )


# ════════════════════════════════════════════════════════════════════════════
# 교정 메서드 1: Standard (기존 결과 재사용)
# ════════════════════════════════════════════════════════════════════════════

def calibrate_standard(theta: np.ndarray, N: np.ndarray,
                        N0_existing: float, a2_existing: float) -> dict:
    """기존 결과를 그대로 사용 (재계산하지 않음)."""
    vwc = vwc_from_N(N, N0_existing, a2_existing)
    vwc = np.clip(vwc, 0, 1)
    return dict(N0=N0_existing, a2=a2_existing, vwc=vwc,
                method="Standard", **metrics(theta, vwc))


# ════════════════════════════════════════════════════════════════════════════
# 교정 메서드 2: SHP-Joint (N0 + a2 동시 최적화)
# ════════════════════════════════════════════════════════════════════════════

def calibrate_shp_joint(theta: np.ndarray, N: np.ndarray,
                         initial_N0: float = 1000.0) -> dict:
    """
    RMSE(N0, a2) 를 Nelder-Mead 로 최소화.
    격자 탐색 → Nelder-Mead 정밀화 순서로 수행.
    """

    def obj(params):
        N0, a2 = params
        if N0 <= 0 or a2 < -0.01:
            return 1e6
        vwc = vwc_from_N(N, N0, a2)
        valid = np.isfinite(vwc) & (vwc >= 0) & (vwc <= 1)
        if valid.sum() < 2:
            return 1e6
        return float(np.sqrt(np.mean((vwc[valid] - theta[valid]) ** 2)))

    # 격자 탐색
    best = (1e6, initial_N0, 0.03)
    for n0 in np.linspace(300, 3500, 14):
        for a2 in np.linspace(0.0, 0.25, 10):
            r = obj([n0, a2])
            if r < best[0]:
                best = (r, n0, a2)

    result = minimize(obj, x0=[best[1], best[2]], method="Nelder-Mead",
                      options={"xatol": 1e-5, "fatol": 1e-7, "maxiter": 10000})

    N0_opt, a2_opt = result.x
    vwc = np.clip(vwc_from_N(N, N0_opt, a2_opt), 0, 1)
    return dict(N0=float(N0_opt), a2=float(a2_opt), vwc=vwc,
                method="SHP-Joint", success=result.success,
                **metrics(theta, vwc))


# ════════════════════════════════════════════════════════════════════════════
# 교정 메서드 3: SHP-2pt (해석적 2점 교정)
# ════════════════════════════════════════════════════════════════════════════

def calibrate_shp_twopoint(theta: np.ndarray, N: np.ndarray,
                            n_percentile: int = 10) -> dict:
    """
    건조기(하위 n%)와 습윤기(상위 n%) 대표값으로 이차방정식 풀이.

    두 상태점 (θ1,N1), (θ2,N2)에서:
        N0 = N / (a0/(θ+a2) + a1)  →  두 식 같다고 놓으면 a2에 대한 이차방정식.

    이차방정식:  A·a2² + B·a2 + C = 0
        A = a1
        B = a1(θ1+θ2) − a0
        C = a1·θ1·θ2 − a0·(N1·θ2 − N2·θ1)/(N1−N2)
    """
    # 대표값: percentile 집합의 중앙값 사용 (이상치 영향 감소)
    dry_mask = theta <= np.percentile(theta, n_percentile)
    wet_mask = theta >= np.percentile(theta, 100 - n_percentile)

    if dry_mask.sum() == 0 or wet_mask.sum() == 0:
        return dict(N0=np.nan, a2=np.nan, vwc=np.full_like(theta, np.nan),
                    method="SHP-2pt", RMSE=np.nan, MAE=np.nan,
                    Bias=np.nan, R=np.nan, n=0, note="Insufficient spread")

    t1 = float(np.median(theta[dry_mask]))
    n1 = float(np.median(N[dry_mask]))
    t2 = float(np.median(theta[wet_mask]))
    n2 = float(np.median(N[wet_mask]))

    dN = n1 - n2
    if abs(dN) < 1.0 or abs(t2 - t1) < 0.005:
        return dict(N0=np.nan, a2=np.nan, vwc=np.full_like(theta, np.nan),
                    method="SHP-2pt", RMSE=np.nan, MAE=np.nan,
                    Bias=np.nan, R=np.nan, n=0,
                    note="Dry/wet contrast too small")

    A = A1
    B = A1 * (t1 + t2) - A0
    C = A1 * t1 * t2 - A0 * (n1 * t2 - n2 * t1) / dN
    disc = B ** 2 - 4 * A * C

    if disc < 0:
        return dict(N0=np.nan, a2=np.nan, vwc=np.full_like(theta, np.nan),
                    method="SHP-2pt", RMSE=np.nan, MAE=np.nan,
                    Bias=np.nan, R=np.nan, n=0, note="No real roots")

    roots = [(-B + np.sqrt(disc)) / (2 * A), (-B - np.sqrt(disc)) / (2 * A)]
    valid_roots = sorted([r for r in roots if r >= -0.01])

    if not valid_roots:
        return dict(N0=np.nan, a2=np.nan, vwc=np.full_like(theta, np.nan),
                    method="SHP-2pt", RMSE=np.nan, MAE=np.nan,
                    Bias=np.nan, R=np.nan, n=0, note="No non-negative roots")

    a2_opt = float(max(valid_roots[0], 0.0))
    N0_opt = float(n1 / (A0 / (t1 + a2_opt) + A1))

    vwc = np.clip(vwc_from_N(N, N0_opt, a2_opt), 0, 1)
    note = (f"Dry: θ={t1:.3f} N={n1:.1f} | "
            f"Wet: θ={t2:.3f} N={n2:.1f} | "
            f"disc={disc:.4f}")
    return dict(N0=float(N0_opt), a2=float(a2_opt), vwc=vwc,
                method="SHP-2pt", note=note,
                **metrics(theta, vwc))


# ════════════════════════════════════════════════════════════════════════════
# 데이터 로드
# ════════════════════════════════════════════════════════════════════════════

def load_debug_data(station_id: str, project_root: Path) -> pd.DataFrame:
    """기존 calibration_debug_data.xlsx 로드."""
    path = (project_root / "data" / "output" / station_id /
            "calibration" / f"{station_id}_calibration_debug_data.xlsx")
    if not path.exists():
        raise FileNotFoundError(f"Debug data not found: {path}\n"
                                "먼저 run_calibration.py 를 실행하세요.")
    df = pd.read_excel(path, sheet_name="Calibration_Data")
    df["date"] = pd.to_datetime(df["date"])
    print(f"✅ Loaded {len(df)} matched data points from {path.name}")
    return df


def load_existing_result(station_id: str, project_root: Path) -> dict:
    """기존 calibration_result.json 로드."""
    path = (project_root / "data" / "output" / station_id /
            "calibration" / f"{station_id}_calibration_result.json")
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ════════════════════════════════════════════════════════════════════════════
# 시각화
# ════════════════════════════════════════════════════════════════════════════

COLORS = {
    "Standard":  "#4477AA",
    "SHP-Joint": "#EE6677",
    "SHP-2pt":   "#228833",
    "FDR":       "#333333",
}


def plot_results(df: pd.DataFrame, results: dict,
                 station_id: str, output_dir: Path) -> None:

    dates = df["date"].values
    theta = df["Field_SM"].values
    N     = df["Daily_N"].values

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(f"{station_id}  —  SHP Calibration Test", fontsize=16, fontweight="bold")

    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35)
    ax_ts   = fig.add_subplot(gs[0, :])      # 시계열 전체
    ax_sc   = fig.add_subplot(gs[1, 0])      # 산포도
    ax_res  = fig.add_subplot(gs[1, 1])      # 잔차 시계열
    ax_curv = fig.add_subplot(gs[1, 2])      # N-θ 감쇠 곡선
    ax_bar  = fig.add_subplot(gs[2, 0])      # 성능 지표 막대
    ax_a2   = fig.add_subplot(gs[2, 1])      # SHP 민감도
    ax_txt  = fig.add_subplot(gs[2, 2])      # 텍스트 요약

    methods_order = ["Standard", "SHP-Joint", "SHP-2pt"]

    # ── 시계열 ───────────────────────────────────────────────────────────────
    ax_ts.plot(dates, theta, "k-", linewidth=1.5, label="FDR (reference)", zorder=5)
    for m in methods_order:
        r = results.get(m)
        if r is None or not np.any(np.isfinite(r["vwc"])):
            continue
        ax_ts.plot(dates, r["vwc"], linestyle="--", linewidth=1.4,
                   color=COLORS[m], label=f"{m}  (RMSE={r['RMSE']:.4f})", alpha=0.85)
    ax_ts.set_ylabel("VWC (m³/m³)")
    ax_ts.set_title("Time Series Comparison")
    ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_ts.xaxis.set_major_locator(mdates.MonthLocator())
    ax_ts.tick_params(axis="x", rotation=45)
    ax_ts.legend(fontsize=9)
    ax_ts.grid(alpha=0.3)

    # ── 산포도 ───────────────────────────────────────────────────────────────
    lim = (max(0, theta.min() - 0.02), min(1, theta.max() + 0.02))
    ax_sc.plot(lim, lim, "k--", linewidth=1, label="1:1")
    for m in methods_order:
        r = results.get(m)
        if r is None or not np.any(np.isfinite(r["vwc"])):
            continue
        ax_sc.scatter(theta, r["vwc"], s=30, alpha=0.7,
                      color=COLORS[m], label=m, zorder=3)
    ax_sc.set_xlim(lim); ax_sc.set_ylim(lim)
    ax_sc.set_xlabel("FDR VWC"); ax_sc.set_ylabel("CRNP VWC")
    ax_sc.set_title("Scatter (1:1)")
    ax_sc.legend(fontsize=8); ax_sc.grid(alpha=0.3)

    # ── 잔차 시계열 ──────────────────────────────────────────────────────────
    ax_res.axhline(0, color="k", linewidth=0.8)
    for m in methods_order:
        r = results.get(m)
        if r is None or not np.any(np.isfinite(r["vwc"])):
            continue
        res = r["vwc"] - theta
        ax_res.plot(dates, res, linewidth=1.2, color=COLORS[m],
                    label=f"{m}", alpha=0.8)
    ax_res.set_ylabel("Residual (CRNP − FDR)")
    ax_res.set_title("Residuals")
    ax_res.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax_res.tick_params(axis="x", rotation=45)
    ax_res.legend(fontsize=8); ax_res.grid(alpha=0.3)

    # ── N-θ 감쇠 곡선 ────────────────────────────────────────────────────────
    theta_range = np.linspace(0.01, 0.60, 200)
    for m in methods_order:
        r = results.get(m)
        if r is None or not np.isfinite(r.get("N0", np.nan)):
            continue
        N_curve = r["N0"] * (A0 / (theta_range + r["a2"]) + A1)
        ax_curv.plot(theta_range, N_curve, color=COLORS[m], linewidth=2,
                     label=f"{m}  N0={r['N0']:.0f}, a2={r['a2']:.4f}")
    ax_curv.scatter(theta, N, s=20, color="k", alpha=0.5, label="Observations", zorder=5)
    ax_curv.set_xlabel("VWC (m³/m³)"); ax_curv.set_ylabel("Neutron count")
    ax_curv.set_title("N–θ Calibration Curves")
    ax_curv.legend(fontsize=8); ax_curv.grid(alpha=0.3)

    # ── 성능 지표 막대 ───────────────────────────────────────────────────────
    metric_keys = ["RMSE", "MAE", "Bias"]
    x = np.arange(len(metric_keys))
    width = 0.25
    plotted = [m for m in methods_order if results.get(m) and np.isfinite(results[m].get("RMSE", np.nan))]
    for i, m in enumerate(plotted):
        r = results[m]
        vals = [abs(r.get(k, 0)) for k in metric_keys]
        ax_bar.bar(x + i * width, vals, width, color=COLORS[m], label=m, alpha=0.85)
    ax_bar.set_xticks(x + width)
    ax_bar.set_xticklabels(metric_keys)
    ax_bar.set_ylabel("m³/m³")
    ax_bar.set_title("Performance Metrics (|values|)")
    ax_bar.legend(fontsize=8); ax_bar.grid(alpha=0.3, axis="y")

    # ── a2 민감도 (SHP-Joint 기준) ───────────────────────────────────────────
    r_j = results.get("SHP-Joint")
    if r_j and np.isfinite(r_j.get("a2", np.nan)):
        a2_center = r_j["a2"]
        a2_grid = np.linspace(max(0, a2_center - 0.08), a2_center + 0.08, 60)
        rmse_grid = []
        for a2_t in a2_grid:
            v = np.clip(vwc_from_N(N, r_j["N0"], a2_t), 0, 1)
            rmse_grid.append(rmse(theta, v))
        ax_a2.plot(a2_grid, rmse_grid, color=COLORS["SHP-Joint"], linewidth=2)
        ax_a2.axvline(a2_center, color="red", linestyle="--",
                      label=f"Optimal a2={a2_center:.4f}")
        ax_a2.set_xlabel("a2 (SHP equivalent)")
        ax_a2.set_ylabel("RMSE")
        ax_a2.set_title("RMSE sensitivity to a2\n(N0 fixed at SHP-Joint optimum)")
        ax_a2.legend(fontsize=9); ax_a2.grid(alpha=0.3)
    else:
        ax_a2.text(0.5, 0.5, "SHP-Joint not available",
                   ha="center", va="center", transform=ax_a2.transAxes)
        ax_a2.axis("off")

    # ── 텍스트 요약 ──────────────────────────────────────────────────────────
    ax_txt.axis("off")
    lines = [f"{'Station':>12s}:  {station_id}",
             f"{'N points':>12s}:  {len(df)}",
             f"{'FDR range':>12s}:  {theta.min():.3f} – {theta.max():.3f}",
             f"{'N range':>12s}:  {N.min():.1f} – {N.max():.1f}",
             ""]
    header = f"{'Method':<12} {'N0':>7} {'a2':>8} {'RMSE':>8} {'R':>6}"
    lines.append(header)
    lines.append("─" * len(header))
    for m in methods_order:
        r = results.get(m, {})
        N0s  = f"{r['N0']:.1f}"   if np.isfinite(r.get("N0", np.nan))   else "  N/A"
        a2s  = f"{r['a2']:.4f}"  if np.isfinite(r.get("a2", np.nan))   else "  N/A"
        rms  = f"{r['RMSE']:.4f}" if np.isfinite(r.get("RMSE", np.nan)) else "  N/A"
        rs   = f"{r['R']:.3f}"   if np.isfinite(r.get("R", np.nan))    else " N/A"
        lines.append(f"{m:<12} {N0s:>7} {a2s:>8} {rms:>8} {rs:>6}")
    if results.get("SHP-2pt", {}).get("note"):
        lines.append(f"\n2pt note:\n  {results['SHP-2pt']['note']}")

    ax_txt.text(0.02, 0.97, "\n".join(lines), transform=ax_txt.transAxes,
                fontsize=9, va="top", fontfamily="monospace",
                bbox=dict(facecolor="lightyellow", edgecolor="gray", alpha=0.9))

    out_path = output_dir / f"{station_id}_shp_test.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"📊 Plot saved → {out_path}")


# ════════════════════════════════════════════════════════════════════════════
# 메인
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SHP/DHP Calibration Test")
    parser.add_argument("--station", "-s", required=True,
                        help="Station ID (HC, PC, …)")
    parser.add_argument("--output_dir", "-o", default=None,
                        help="Output directory (default: same as calibration folder)")
    parser.add_argument("--percentile", type=int, default=10,
                        help="Percentile for 2-point dry/wet selection (default: 10)")
    args = parser.parse_args()

    station_id  = args.station
    project_root = Path(__file__).resolve().parent   # scripts/ 와 같은 레벨이면 .parent.parent
    # 스크립트 위치 자동 감지
    if (project_root / "data").exists():
        pass
    elif (project_root.parent / "data").exists():
        project_root = project_root.parent
    else:
        project_root = Path.cwd()

    # ── 데이터 로드 ──────────────────────────────────────────────────────────
    df = load_debug_data(station_id, project_root)
    existing = load_existing_result(station_id, project_root)

    theta = df["Field_SM"].values.astype(float)
    N     = df["Daily_N"].values.astype(float)

    # 기존 파라미터 추출
    N0_exist = float(existing.get("N0_rdt", 1000))
    a2_exist = float(existing.get("lattice_water", 0.03))

    print(f"\n{'='*60}")
    print(f"  Station : {station_id}")
    print(f"  Data pts: {len(df)}")
    print(f"  FDR     : {theta.min():.3f} – {theta.max():.3f}")
    print(f"  N       : {N.min():.1f} – {N.max():.1f}")
    print(f"  Existing: N0={N0_exist:.1f}, a2(Wlat)={a2_exist:.4f}")
    print(f"{'='*60}\n")

    # ── 세 가지 교정 수행 ────────────────────────────────────────────────────
    print("▶ [1/3] Standard (기존 결과 재사용)")
    r_std = calibrate_standard(theta, N, N0_exist, a2_exist)
    print(f"         N0={r_std['N0']:.1f}, a2={r_std['a2']:.4f}, "
          f"RMSE={r_std['RMSE']:.4f}, R={r_std['R']:.3f}")

    print("▶ [2/3] SHP-Joint (N0+a2 동시 최적화) …")
    r_joint = calibrate_shp_joint(theta, N, initial_N0=N0_exist)
    print(f"         N0={r_joint['N0']:.1f}, a2={r_joint['a2']:.4f}, "
          f"RMSE={r_joint['RMSE']:.4f}, R={r_joint['R']:.3f}, "
          f"success={r_joint['success']}")

    print(f"▶ [3/3] SHP-2pt (해석적 {args.percentile}% 건조/습윤 대표값)")
    r_2pt = calibrate_shp_twopoint(theta, N, n_percentile=args.percentile)
    print(f"         N0={r_2pt['N0']:.1f}, a2={r_2pt['a2']:.4f}, "
          f"RMSE={r_2pt['RMSE']:.4f}, R={r_2pt['R']:.3f}")
    if "note" in r_2pt:
        print(f"         note: {r_2pt['note']}")

    results = {
        "Standard":  r_std,
        "SHP-Joint": r_joint,
        "SHP-2pt":   r_2pt,
    }

    # ── RMSE 개선율 ──────────────────────────────────────────────────────────
    base_rmse = r_std["RMSE"]
    print(f"\n{'─'*50}")
    print(f"  RMSE 개선율 (Standard 기준)")
    for m, r in [("SHP-Joint", r_joint), ("SHP-2pt", r_2pt)]:
        if np.isfinite(r["RMSE"]):
            imp = (base_rmse - r["RMSE"]) / base_rmse * 100
            sign = "↓" if imp > 0 else "↑"
            print(f"  {m:<12}: {imp:+.1f}% {sign}  ({r['RMSE']:.4f} vs {base_rmse:.4f})")
    print(f"{'─'*50}")

    # ── SHP 해석 ─────────────────────────────────────────────────────────────
    a2_j = r_joint["a2"]
    print(f"\n📌 SHP 해석 (SHP-Joint a2 = {a2_j:.4f} m³/m³ 상당)")
    if a2_j < 0.02:
        hint = "SHP 매우 작음 (농경지/사막 특성 또는 Wlat 과소평가)"
    elif a2_j < 0.05:
        hint = "SHP 보통 (전형적 농경지·초지)"
    elif a2_j < 0.10:
        hint = "SHP 큰 편 (식생 바이오매스·유기탄소 풍부)"
    else:
        hint = "SHP 매우 큼 (산림·습지 등 높은 정적 수소 환경)"
    print(f"  → {hint}")

    # ── 출력 디렉토리 ────────────────────────────────────────────────────────
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = (project_root / "data" / "output" / station_id / "calibration")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 시각화 ───────────────────────────────────────────────────────────────
    plot_results(df, results, station_id, out_dir)

    # ── JSON 저장 ─────────────────────────────────────────────────────────────
    summary = {}
    for m, r in results.items():
        summary[m] = {k: (float(v) if isinstance(v, (float, np.floating)) else v)
                      for k, v in r.items() if k != "vwc"}
    json_path = out_dir / f"{station_id}_shp_test_result.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"💾 Result saved → {json_path}")
    print("\n✅ Done.")


if __name__ == "__main__":
    main()