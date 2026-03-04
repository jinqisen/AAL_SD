# Anomaly Analysis: uncertainty_only

## Anomalies
[
  "Significant performance drop: mIoU 0.6993 -> 0.6465"
]

## LLM Insight
**1. Training Health Assessment:**
Training is unstable. While the model was improving through Round 8 (mIoU ~0.699), Round 9 introduced a significant and persistent performance drop (mIoU ~0.646). The loss spiked dramatically at the start of Round 9 (0.3518) before recovering, but mIoU did not fully rebound. This indicates a recent disruptive event, likely the new data batch.

**2. Possible Causes for Anomaly:**
The primary cause is the new batch of 88 samples added in **Round 9**. The sharp loss increase in epoch 1 of that round suggests the model encountered challenging or dissimilar data. Likely causes:
*   **Poor Quality New Labels:** The uncertainty-based query may have selected ambiguous, noisy, or mislabeled examples.
*   **Distribution Shift:** The new batch may come from a different data distribution than the previous training set, causing temporary model confusion.
*   **Aggressive Learning Rate:** The LR might be too high to smoothly integrate the new, potentially difficult, samples.

**3. Recommendations:**
*   **Query Strategy:** Augment the **uncertainty-only** strategy with a **diversity** component (e.g., CoreSet, BADGE) to avoid selecting a cluster of similarly challenging or outlier samples.
*   **Learning Rate:** Implement a learning rate schedule that reduces LR at the start of each new active learning round to allow gentler adaptation to new data. Consider a warm-up restart.
*   **Data Audit:** Manually inspect the samples queried in Round 9 for potential labeling errors or extreme outliers.
*   **Monitoring:** Watch the next round closely. If performance does not recover above Round 8 levels, the issue is likely systemic with the queried data.
