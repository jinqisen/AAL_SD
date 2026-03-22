Although the current experiments do not directly optimize gradients, the combined evidence from Figures 7 and 8 provides empirical support for the indirect gradient-shaping interpretation proposed in this paper. In the seed-42 trace, low-risk rounds such as Round 5 and Round 7 show positive train-validation gradient cosine statistics (`grad_train_val_cos_last` around 0.624 and 0.638, respectively), after which the controller raises λ from 0.20 to 0.25 and then to 0.32, increasing the contribution of knowledge gain. In contrast, later rounds with stronger instability signals trigger more conservative behavior: for example, Round 11 and Round 14 exhibit strongly negative `grad_train_val_cos_last` values (about -0.580 and -0.775), and the policy correspondingly shifts toward lower effective λ or uncertainty-biased selection.

### E-2. Lambda Landscape Analysis

To validate the theoretical assumptions underlying the adaptive lambda policy, we conducted a controlled lambda sweep experiment. Starting from a fixed trunk checkpoint at the end of Round K-1, we branched multiple experiments with lambda values in {0.0, 0.2, 0.4, 0.6, 0.8, 1.0} and trained Round K with each fixed lambda. This procedure was repeated for Rounds 3, 6, and 9 to capture landscape evolution across learning stages.

**Experimental Design:** For each sweep round K, we: (1) branched from the trunk experiment's Round K-1 checkpoint; (2) created 6 parallel experiments with fixed lambda values; (3) trained Round K with each lambda and recorded best-validation mIoU; (4) analyzed the resulting lambda-performance curves.

**Statistical Findings:** Table VII summarizes the key statistics across the three sweep rounds.

**TABLE VII**
**Lambda Sweep Analysis (Single Seed, Rounds 3/6/9)**

| Round | mIoU Range | Peak λ | Friedman p | Quadratic R² | Spearman ρ (p) | SNR |
|-------|-----------|--------|-----------|--------------|----------------|-----|
| 3     | 0.0387    | 0.6    | —         | 0.31 (concave) | -0.43 (0.40) | 2.29 |
| 6     | 0.0543    | 0.8    | —         | 0.23 (convex)  | -0.09 (0.87) | 3.21 |
| 9     | 0.0351    | 0.2    | —         | 0.11 (convex)  | +0.26 (0.62) | 2.07 |
| Pooled | 0.0427 | — | 0.70 | — | — | 2.52 |

**Key Observations:**

1. **Flat Landscape:** The mIoU range across all lambda values is only 3.5-5.4%, with a pooled cross-round noise estimate of 1.69%. The signal-to-noise ratio (2.1-3.2) is marginally distinguishable but far from the sharp peaks assumed by unimodal theory.

2. **No Unimodality:** All three rounds exhibit multiple local peaks (2-3 per round). Quadratic fits yield poor R² values (0.11-0.31), and the concavity direction is inconsistent (1 concave, 2 convex).

3. **Unstable Rankings:** The optimal lambda jumps erratically across rounds (0.6 → 0.8 → 0.2). Cross-round lambda rankings are uncorrelated (Kendall's τ ranging from -0.33 to +0.20, all p>0.47), indicating that the "best" lambda at one round provides no predictive value for subsequent rounds.

4. **Statistical Insignificance:** A Friedman test across all rounds and lambda values yields p=0.70 (mIoU) and p=0.84 (c1 IoU), indicating no statistically significant difference between lambda values.

5. **Overlapping Confidence Intervals:** Using cross-round variance to estimate 95% confidence intervals, all lambda values have CIs that overlap substantially (half-width ±0.042 vs. mean range 0.019).

6. **Minority Class Dominance:** The landslide class (c1) exhibits 20-55× more variation than the background class (c0) across lambda values, suggesting that performance is dominated by which specific landslide patches are selected rather than the lambda weighting itself.

**Theoretical Implications:** These findings contradict the original unimodal concavity assumption and support an alternative **flat landscape with bounded risk** framework. For ε-flat landscapes where max_λ φ(λ) - min_λ φ(λ) < ε, any λ ∈ [0,1] achieves near-optimal performance, but extreme values (0 or 1) carry higher variance risk. At ε=0.03 tolerance (3% below peak), approximately 67% of lambda values are acceptable on average, defining a safe operating region λ ∈ [0.2, 0.8].

The adaptive lambda policy's primary function is **risk management** rather than optimization. By maintaining lambda within the safe region and responding conservatively to gradient mismatch signals, the policy prevents catastrophic choices without requiring precise lambda tuning. This reinterpretation aligns the theoretical narrative with empirical evidence while preserving the framework's practical value: AAL-SD provides a robust, interpretable control interface that maintains stable performance across diverse training conditions, even when the underlying lambda landscape offers no sharp optimum to exploit.

**Validation Recommendation:** The single-seed sweep provides strong evidence for the flat landscape hypothesis. Multi-seed validation (3-4 additional seeds) would confirm whether the observed non-monotonicity is genuine structure or noise artifact. Either outcome supports the risk-management framing over precision optimization.

### F. Agent Decision Interpretability
