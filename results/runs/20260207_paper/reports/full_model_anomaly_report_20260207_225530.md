# Anomaly Analysis: full_model

## Anomalies
[
  "Performance stagnation detected (mIoU change < 0.002 over 3 rounds)"
]

## LLM Insight
**1. Training Health Assessment:**  
Training is stable but stagnant. Loss is decreasing healthily (0.099 → 0.082 in recent epochs), indicating the model is still learning. However, **mIoU has plateaued** around ~0.745 for the last 5 rounds despite adding ~350 new labeled samples. Epoch-level mIoU shows some instability (e.g., drop to 0.706 in epoch 6), suggesting potential overfitting or noisy batches.

**2. Possible Causes for Anomalies:**  
- **Query strategy inefficiency:** New samples may be redundant or not informative enough.  
- **Model capacity saturation:** Current architecture/hyperparameters may have reached a performance ceiling.  
- **Noisy labels or difficult samples** in recent batches causing instability.  
- **Learning rate too high/low**, preventing steady convergence.

**3. Recommendations:**  
- **Change query strategy** (e.g., switch to uncertainty-based or diversity sampling).  
- **Reduce learning rate** slightly to stabilize epoch-level metrics.  
- **Increase batch size** if hardware allows, to smooth gradient updates.  
- **Inspect new labeled data** for quality/representativeness.  
- Consider **early stopping** if stagnation persists next round.
