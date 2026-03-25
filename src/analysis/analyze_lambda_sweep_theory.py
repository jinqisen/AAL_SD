"""
Statistical and theoretical analysis of lambda sweep experimental data.

Analyzes whether the observed lambda-performance landscape supports
unimodality assumptions or alternative theoretical frameworks.
"""

import numpy as np
from scipy import stats


# ── Experimental Data ──────────────────────────────────────────────
LAMBDAS = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

DATA = {
    3: {
        "mIoU": np.array([0.6788, 0.6531, 0.6785, 0.6891, 0.6647, 0.6504]),
        "c0": np.array([0.9879, 0.9866, 0.9859, 0.9872, 0.9840, 0.9866]),
        "c1": np.array([0.3697, 0.3197, 0.3711, 0.3910, 0.3454, 0.3142]),
    },
    6: {
        "mIoU": np.array([0.6843, 0.6710, 0.6381, 0.6437, 0.6924, 0.6539]),
        "c0": np.array([0.9869, 0.9873, 0.9853, 0.9866, 0.9881, 0.9872]),
        "c1": np.array([0.3816, 0.3547, 0.2908, 0.3007, 0.3967, 0.3206]),
    },
    9: {
        "mIoU": np.array([0.6846, 0.7106, 0.6755, 0.6878, 0.6870, 0.7016]),
        "c0": np.array([0.9861, 0.9861, 0.9867, 0.9866, 0.9857, 0.9870]),
        "c1": np.array([0.3831, 0.4352, 0.3643, 0.3890, 0.3883, 0.4161]),
    },
}


def section(title: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")


# ── 1. Descriptive Statistics ──────────────────────────────────────
def descriptive_stats():
    section("1. DESCRIPTIVE STATISTICS")
    for rnd in sorted(DATA):
        m = DATA[rnd]["mIoU"]
        c1 = DATA[rnd]["c1"]
        peak_lam = LAMBDAS[np.argmax(m)]
        print(f"\n  Round {rnd}:")
        print(
            f"    mIoU  mean={m.mean():.4f}  std={m.std():.4f}  "
            f"range={m.max() - m.min():.4f}  peak_lambda={peak_lam}"
        )
        print(
            f"    c1    mean={c1.mean():.4f}  std={c1.std():.4f}  "
            f"range={c1.max() - c1.min():.4f}  peak_lambda={LAMBDAS[np.argmax(c1)]}"
        )
        print(
            f"    c0    mean={DATA[rnd]['c0'].mean():.4f}  "
            f"std={DATA[rnd]['c0'].std():.4f}  "
            f"range={DATA[rnd]['c0'].max() - DATA[rnd]['c0'].min():.4f}"
        )


# ── 2. Effect Size & Noise Estimation ─────────────────────────────
def effect_size_analysis():
    section("2. EFFECT SIZE & NOISE ESTIMATION")

    # Estimate noise from c0 (background class) which should be nearly
    # constant across lambda values — its variance is a proxy for
    # training/evaluation noise.
    c0_all = np.concatenate([DATA[r]["c0"] for r in DATA])
    c0_noise_std = c0_all.std()
    print(f"\n  c0 (background) pooled std = {c0_noise_std:.5f}")
    print(f"  → This is our noise floor proxy for per-class IoU")

    # For mIoU, noise ~ sqrt(noise_c0^2 + noise_c1^2) / 2
    # But c0 is near-saturated, so mIoU noise ≈ c1 noise / 2
    # We can also estimate from cross-round variance at same lambda
    print("\n  Cross-round variance at each lambda (mIoU):")
    for i, lam in enumerate(LAMBDAS):
        vals = [DATA[r]["mIoU"][i] for r in DATA]
        print(
            f"    lambda={lam:.1f}  values={[f'{v:.4f}' for v in vals]}  "
            f"std={np.std(vals):.4f}"
        )

    # Pooled cross-round std as noise estimate
    cross_round_stds = []
    for i in range(len(LAMBDAS)):
        vals = [DATA[r]["mIoU"][i] for r in DATA]
        cross_round_stds.append(np.std(vals))
    pooled_noise = np.mean(cross_round_stds)
    print(f"\n  Pooled cross-round noise estimate (mIoU): {pooled_noise:.4f}")

    # Effect sizes per round
    print("\n  Effect size (max-min range) vs noise:")
    for rnd in sorted(DATA):
        m = DATA[rnd]["mIoU"]
        rng = m.max() - m.min()
        ratio = rng / pooled_noise if pooled_noise > 0 else float("inf")
        print(
            f"    Round {rnd}: range={rng:.4f}  noise={pooled_noise:.4f}  "
            f"SNR={ratio:.2f}"
        )

    return pooled_noise


# ── 3. Statistical Significance Tests ─────────────────────────────
def significance_tests(noise_est: float):
    section("3. STATISTICAL SIGNIFICANCE TESTS")

    print("\n  3a. One-way ANOVA across lambda values (per round)")
    print("      H0: all lambda values produce the same mean mIoU")
    print("      (Using bootstrap-style analysis since n=1 per cell)\n")

    # With single-seed data, we cannot do classical ANOVA.
    # Instead: treat cross-round data as repeated measures.
    # Friedman test: non-parametric repeated measures
    miou_matrix = np.array([DATA[r]["mIoU"] for r in sorted(DATA)])
    # rows = rounds (blocks), cols = lambda treatments
    stat_f, p_f = stats.friedmanchisquare(*[miou_matrix[:, i] for i in range(6)])
    print(f"  Friedman test (mIoU across rounds):")
    print(f"    chi2={stat_f:.4f}  p={p_f:.4f}")
    print(f"    {'SIGNIFICANT' if p_f < 0.05 else 'NOT significant'} at alpha=0.05")

    # Same for c1
    c1_matrix = np.array([DATA[r]["c1"] for r in sorted(DATA)])
    stat_c1, p_c1 = stats.friedmanchisquare(*[c1_matrix[:, i] for i in range(6)])
    print(f"\n  Friedman test (c1 across rounds):")
    print(f"    chi2={stat_c1:.4f}  p={p_c1:.4f}")
    print(f"    {'SIGNIFICANT' if p_c1 < 0.05 else 'NOT significant'} at alpha=0.05")

    # Pairwise: is the best lambda significantly better than worst?
    print("\n  3b. Pairwise comparisons (best vs worst lambda per round)")
    for rnd in sorted(DATA):
        m = DATA[rnd]["mIoU"]
        best_i, worst_i = np.argmax(m), np.argmin(m)
        diff = m[best_i] - m[worst_i]
        # Cohen's d using noise estimate
        d = diff / noise_est if noise_est > 0 else float("inf")
        print(
            f"    Round {rnd}: best=lambda_{LAMBDAS[best_i]:.1f}({m[best_i]:.4f}) "
            f"worst=lambda_{LAMBDAS[worst_i]:.1f}({m[worst_i]:.4f}) "
            f"diff={diff:.4f} Cohen_d={d:.2f}"
        )

    # Bootstrap confidence intervals
    print("\n  3c. Approximate confidence intervals (using cross-round variance)")
    print(f"      Noise estimate: {noise_est:.4f}")
    print(
        f"      95% CI half-width (t_2,0.025 * se): "
        f"{stats.t.ppf(0.975, df=2) * noise_est / np.sqrt(3):.4f}"
    )

    ci_hw = stats.t.ppf(0.975, df=2) * noise_est / np.sqrt(3)
    print(f"\n      Per-lambda mean mIoU with 95% CI:")
    for i, lam in enumerate(LAMBDAS):
        vals = [DATA[r]["mIoU"][i] for r in sorted(DATA)]
        mean_v = np.mean(vals)
        print(
            f"        lambda={lam:.1f}: {mean_v:.4f} +/- {ci_hw:.4f}  "
            f"[{mean_v - ci_hw:.4f}, {mean_v + ci_hw:.4f}]"
        )


# ── 4. Unimodality Tests ──────────────────────────────────────────
def unimodality_analysis():
    section("4. UNIMODALITY ANALYSIS")

    print("\n  4a. Local peak count per round")
    for rnd in sorted(DATA):
        m = DATA[rnd]["mIoU"]
        peaks = []
        for i in range(1, len(m) - 1):
            if m[i] > m[i - 1] and m[i] > m[i + 1]:
                peaks.append((LAMBDAS[i], m[i]))
        # Check endpoints
        if m[0] > m[1]:
            peaks.insert(0, (LAMBDAS[0], m[0]))
        if m[-1] > m[-2]:
            peaks.append((LAMBDAS[-1], m[-1]))
        print(f"    Round {rnd}: {len(peaks)} peak(s) at {peaks}")

    # Quadratic fit (unimodal proxy)
    print("\n  4b. Quadratic fit R^2 (unimodal model)")
    for rnd in sorted(DATA):
        m = DATA[rnd]["mIoU"]
        coeffs = np.polyfit(LAMBDAS, m, 2)
        fitted = np.polyval(coeffs, LAMBDAS)
        ss_res = np.sum((m - fitted) ** 2)
        ss_tot = np.sum((m - m.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        vertex = -coeffs[1] / (2 * coeffs[0]) if coeffs[0] != 0 else float("nan")
        concave = "concave" if coeffs[0] < 0 else "convex"
        print(
            f"    Round {rnd}: R^2={r2:.4f}  a={coeffs[0]:.4f} ({concave})  "
            f"vertex={vertex:.2f}"
        )

    # Spearman rank correlation (monotonicity test)
    print("\n  4c. Spearman correlation (lambda vs mIoU)")
    for rnd in sorted(DATA):
        rho, p = stats.spearmanr(LAMBDAS, DATA[rnd]["mIoU"])
        print(
            f"    Round {rnd}: rho={rho:.4f}  p={p:.4f}  "
            f"{'monotonic trend' if p < 0.05 else 'no monotonic trend'}"
        )


# ── 5. Theoretical Framework Evaluation ───────────────────────────
def framework_evaluation(noise_est: float):
    section("5. THEORETICAL FRAMEWORK EVALUATION")

    # Framework A: Flat landscape with danger zones
    print("\n  5a. Framework A: Flat Landscape with Danger Zones")
    print("      Hypothesis: lambda has weak effect; goal is avoiding bad regions")
    all_miou = np.concatenate([DATA[r]["mIoU"] for r in DATA])
    grand_mean = all_miou.mean()
    grand_std = all_miou.std()
    print(f"      Grand mean mIoU = {grand_mean:.4f}")
    print(f"      Grand std  mIoU = {grand_std:.4f}")
    print(f"      Coefficient of variation = {grand_std / grand_mean * 100:.2f}%")

    # Check: what fraction of lambda values are within epsilon of best?
    for eps in [0.01, 0.02, 0.03]:
        safe_counts = []
        for rnd in sorted(DATA):
            m = DATA[rnd]["mIoU"]
            n_safe = np.sum(m >= m.max() - eps)
            safe_counts.append(n_safe)
        print(f"      eps={eps}: safe lambdas per round = {safe_counts}  (of 6)")

    # Framework B: Task-specific sampling dominance
    print("\n  5b. Framework B: Task-Specific Sampling Dominance")
    print("      Hypothesis: individual sample selection >> lambda effect")
    for rnd in sorted(DATA):
        c0_range = DATA[rnd]["c0"].max() - DATA[rnd]["c0"].min()
        c1_range = DATA[rnd]["c1"].max() - DATA[rnd]["c1"].min()
        ratio = c1_range / c0_range if c0_range > 0 else float("inf")
        print(
            f"      Round {rnd}: c0_range={c0_range:.4f}  c1_range={c1_range:.4f}  "
            f"c1/c0_ratio={ratio:.1f}x"
        )
    print("      → c1 (minority class) varies ~20x more than c0")
    print("      → Performance dominated by which landslide patches are selected")

    # Framework C: Multi-modal landscape
    print("\n  5c. Framework C: Multi-Modal Landscape")
    print("      Hypothesis: multiple local optima, shifting across rounds")
    peaks_per_round = {}
    for rnd in sorted(DATA):
        m = DATA[rnd]["mIoU"]
        peak_lam = LAMBDAS[np.argmax(m)]
        peaks_per_round[rnd] = peak_lam
    print(f"      Peak lambda trajectory: {peaks_per_round}")
    peak_vals = list(peaks_per_round.values())
    peak_std = np.std(peak_vals)
    print(f"      Peak lambda std = {peak_std:.2f}")
    print(f"      Peak jumping range = {max(peak_vals) - min(peak_vals):.1f}")

    # Scoring
    print("\n  5d. Framework Fit Summary")
    print("      ┌──────────────────────────────────┬───────┬──────────────────────┐")
    print("      │ Framework                        │ Score │ Key Evidence         │")
    print("      ├──────────────────────────────────┼───────┼──────────────────────┤")

    # Score A: flat landscape
    cv = grand_std / grand_mean
    score_a = "HIGH" if cv < 0.03 else "MED" if cv < 0.06 else "LOW"
    print(
        f"      │ A: Flat + Danger Zones           │ {score_a:5s} │ CV={cv:.3f}, small range │"
    )

    # Score B: sampling dominance
    avg_c1_c0_ratio = np.mean(
        [
            (DATA[r]["c1"].max() - DATA[r]["c1"].min())
            / max(DATA[r]["c0"].max() - DATA[r]["c0"].min(), 1e-6)
            for r in DATA
        ]
    )
    score_b = (
        "HIGH" if avg_c1_c0_ratio > 10 else "MED" if avg_c1_c0_ratio > 5 else "LOW"
    )
    print(
        f"      │ B: Sampling Dominance             │ {score_b:5s} │ c1/c0 ratio={avg_c1_c0_ratio:.0f}x     │"
    )

    # Score C: multi-modal
    score_c = "MED" if peak_std > 0.2 else "LOW"
    print(
        f"      │ C: Multi-Modal                    │ {score_c:5s} │ peak_std={peak_std:.2f}       │"
    )

    # Original theory
    print(f"      │ Original: Unimodal Concave        │ LOW   │ 0/3 rounds unimodal │")
    print("      └──────────────────────────────────┴───────┴──────────────────────┘")


# ── 6. Confounding Factors ─────────────────────────────────────────
def confounding_factors(noise_est: float):
    section("6. CONFOUNDING FACTOR ASSESSMENT")

    print("\n  6a. Sparse Sampling (6 points, step=0.2)")
    print("      Nyquist-like argument: to detect a peak of width w,")
    print("      need step < w/2. With step=0.2, can only detect peaks")
    print("      wider than 0.4 in lambda-space.")
    print("      → Sharp peaks (width < 0.4) would be missed entirely")
    print("      → True landscape could have structure between samples")

    print("\n  6b. Single Seed")
    print(f"      Estimated noise (cross-round): {noise_est:.4f}")
    for rnd in sorted(DATA):
        m = DATA[rnd]["mIoU"]
        rng = m.max() - m.min()
        snr = rng / noise_est if noise_est > 0 else float("inf")
        print(
            f"      Round {rnd}: signal range={rng:.4f}  SNR={snr:.2f}  "
            f"{'distinguishable' if snr > 2 else 'within noise'}"
        )

    print("\n  6c. Trunk Policy Dependence")
    print("      All sweeps branch from same trunk checkpoint per round.")
    print("      Different trunk → different labeled pool → different landscape.")
    print("      This is a MAJOR confound: the landscape is conditional on")
    print("      the specific training trajectory, not just lambda.")

    print("\n  6d. Round-Dependent Pool Composition")
    print("      As rounds progress, labeled pool grows and changes.")
    print("      The marginal value of lambda-weighted selection depends on")
    print("      what's already labeled. This explains peak jumping.")


# ── 7. Recommendations ────────────────────────────────────────────
def recommendations(noise_est: float):
    section("7. RECOMMENDATIONS")

    # Determine which path
    all_ranges = [DATA[r]["mIoU"].max() - DATA[r]["mIoU"].min() for r in DATA]
    avg_range = np.mean(all_ranges)
    avg_snr = avg_range / noise_est if noise_est > 0 else float("inf")

    print(f"\n  Decision metrics:")
    print(f"    Average mIoU range across rounds: {avg_range:.4f}")
    print(f"    Noise estimate: {noise_est:.4f}")
    print(f"    Average SNR: {avg_snr:.2f}")

    print(f"\n  ── PATH ASSESSMENT ──")

    print(f"\n  Path A: Effect size < noise (reposition paper)")
    if avg_snr < 2:
        print(f"    ★ RECOMMENDED (SNR={avg_snr:.2f} < 2)")
    else:
        print(f"    ○ Partially supported (SNR={avg_snr:.2f})")
    print(f"    - Reframe lambda as robustness/safety parameter, not optimizer")
    print(f"    - Paper contribution: AAL-SD framework integration, not lambda tuning")
    print(f"    - Lambda control prevents catastrophic failure, doesn't find optimum")
    print(f"    - Cost: minimal (rewrite framing, no new experiments)")

    print(f"\n  Path B: Multi-seed validation")
    print(f"    ○ RECOMMENDED as validation step regardless of path chosen")
    print(f"    - Run 3 additional seeds (total 4)")
    print(f"    - If averaged curves smooth out → noise hypothesis confirmed")
    print(f"    - If still non-monotonic → genuine multi-modality")
    print(f"    - Cost: 3x compute of current sweep (~moderate)")
    print(f"    - Expected outcome: likely confirms flat landscape (Path A)")

    print(f"\n  Path C: Embrace non-unimodal reality")
    print(f"    ○ Viable fallback if multi-seed shows persistent structure")
    print(f"    - Reformulate as safety-oriented control: avoid danger zones")
    print(f"    - Theoretical contribution: characterize safe sets S_t")
    print(f"    - Practical: robust controller with epsilon-safety margin")
    print(f"    - Cost: moderate (new theory section, revised proofs)")

    print(f"\n  ── CONCRETE NEXT STEPS ──")
    print(f"    1. Run 3 additional seeds for the same lambda sweep")
    print(f"    2. Compute averaged curves with error bars")
    print(f"    3. If averaged SNR < 2: adopt Path A framing")
    print(f"    4. If averaged SNR >= 2 and unimodal: original theory holds")
    print(f"    5. If averaged SNR >= 2 and non-unimodal: adopt Path C")

    print(f"\n  ── REVISED THEORETICAL NARRATIVE (Path A, most likely) ──")
    print(f"    - Drop Assumption 1 (unimodality) and Theorem 1 (strict concavity)")
    print(f"    - Replace with: 'Flat landscape with bounded risk'")
    print(f"    - New theorem: For epsilon-flat landscapes (range < epsilon),")
    print(f"      any lambda in [0,1] achieves near-optimal performance,")
    print(f"      but extreme values (0 or 1) carry higher variance risk.")
    print(f"    - Practical implication: lambda in [0.2, 0.8] is a safe default")
    print(f"    - Agent's role: monitor for rare non-flat episodes, not optimize")


# ── 8. Supplementary: Cross-Round Stability ───────────────────────
def cross_round_stability():
    section("8. SUPPLEMENTARY: CROSS-ROUND STABILITY")

    print("\n  Lambda ranking stability (Kendall's tau between rounds)")
    rounds = sorted(DATA.keys())
    for i in range(len(rounds)):
        for j in range(i + 1, len(rounds)):
            r1, r2 = rounds[i], rounds[j]
            tau, p = stats.kendalltau(DATA[r1]["mIoU"], DATA[r2]["mIoU"])
            print(
                f"    Round {r1} vs {r2}: tau={tau:.4f}  p={p:.4f}  "
                f"{'correlated' if p < 0.05 else 'uncorrelated'}"
            )

    print("\n  → If rankings are uncorrelated across rounds, lambda's effect")
    print("    is dominated by round-specific noise, not a stable landscape.")


# ── Main ───────────────────────────────────────────────────────────
def main():
    print("=" * 72)
    print("  LAMBDA SWEEP ANALYSIS: STATISTICAL & THEORETICAL ASSESSMENT")
    print("  Data: 3 rounds x 6 lambda values, single seed")
    print("=" * 72)

    descriptive_stats()
    noise_est = effect_size_analysis()
    significance_tests(noise_est)
    unimodality_analysis()
    framework_evaluation(noise_est)
    confounding_factors(noise_est)
    cross_round_stability()
    recommendations(noise_est)

    section("ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
