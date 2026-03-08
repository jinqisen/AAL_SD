# AAL-SD Framework Figure Draft

This file is a ready-to-draw specification for the main method framework figure in the IEEE JSTARS paper.

## Recommended Figure Title

Overall Architecture of the Proposed AAL-SD Framework

## Figure Goal

Show the complete active learning loop and make clear that AAL-SD is not only a scoring rule, but a closed-loop system with:

- model training and evaluation
- uncertainty and feature extraction
- AD-KUCS scoring
- LLM-agent-assisted selection
- feedback through annotation, checkpoints, and risk monitoring

## Recommended Layout

Use a left-to-right pipeline with one top feedback loop and one side monitoring branch.

### Main Row

1. **Dataset Split**
   - Initial labeled pool `L0`
   - Unlabeled pool `U`
   - Fixed test set `T`

2. **Segmentation Model Training**
   - DeepLabV3+
   - train on current labeled pool `Lt`
   - evaluate on `T`

3. **Unlabeled Pool Inference**
   - pixel-wise probability maps
   - uncertainty scores `U(x)`
   - deep feature embeddings `f_x`

4. **AD-KUCS Scoring Block**
   - uncertainty branch
   - knowledge-gain branch
   - dynamic lambda controller
   - final score: `Score(x) = (1 - λ_t)U(x) + λ_tK(x)`

5. **LLM Agent Decision Block**
   - system prompt + toolbox
   - get system status
   - get top-k candidates
   - finalize selection
   - interpretable reason output

6. **Annotation and Pool Update**
   - queried subset `Qt`
   - oracle / expert annotation
   - update `Lt -> Lt+1`

7. **Next Round**
   - arrow back to the training block

### Upper Monitoring Branch

From the training block, add a side branch to:

- mIoU trajectory
- F1 trajectory
- gradient alignment / overfit risk
- rollback threshold and checkpoint manager

This branch should feed both:

- the dynamic lambda controller
- the LLM agent toolbox

## Block Text You Can Put Directly Into the Figure

### Block 1
Dataset Preparation

`L0` / `U` / `T`

### Block 2
Round-wise Segmentation Training

DeepLabV3+
Train on `Lt`
Evaluate on `T`

### Block 3
Unlabeled Inference

Probability maps
Entropy / BALD-style uncertainty
Feature embeddings

### Block 4
AD-KUCS Scoring

`U(x)` uncertainty
`K(x)` coreset-to-labeled knowledge gain
Adaptive `λ_t`
Final fused ranking

### Block 5
LLM Agent Controller

Prompt + toolbox
Read status and candidate list
Finalize queried subset
Generate rationale

### Block 6
Annotation Feedback

Expert labels for `Qt`
Update labeled pool
Remove from unlabeled pool

### Side Block
Monitoring and Safety

mIoU / F1 history
gradient train-val cosine
overfit risk
rollback + checkpoint

## Mermaid Draft

```mermaid
flowchart LR
    A[Dataset Split\nInitial labeled pool L0\nUnlabeled pool U\nFixed test set T] --> B[Round-wise Training\nDeepLabV3+ on Lt\nEvaluate on T]
    B --> C[Unlabeled Pool Inference\nProbability maps\nUncertainty scores U(x)\nFeature embeddings f_x]
    C --> D[AD-KUCS Scoring\nKnowledge gain K(x)\nAdaptive lambda λ_t\nScore(x)=(1-λ_t)U(x)+λ_tK(x)]
    D --> E[LLM Agent Controller\nPrompt + Toolbox\nRead status\nRead top-k candidates\nFinalize queried subset Qt]
    E --> F[Oracle Annotation and Pool Update\nAnnotate Qt\nLt -> Lt+1\nU -> U\Qt]
    F --> B

    B --> G[Monitoring and Safety\nmIoU / F1 trajectory\nGradient alignment\nOverfit risk\nRollback and checkpoint]
    G --> D
    G --> E
```

## Suggested Caption

Figure 3. Overall architecture of AAL-SD. In each active learning round, the current segmentation model is trained on the labeled pool and evaluated on the fixed test set. The unlabeled pool is then processed to obtain uncertainty estimates and feature embeddings, which are fused by AD-KUCS through an adaptive weighting factor λ_t. An LLM agent accesses system state, ranked candidates, and safety signals through a toolbox interface, then finalizes the queried subset with an interpretable rationale. The annotated samples are added back to the labeled pool, while checkpoints and overfitting signals are recorded for the next round.

## Drawing Suggestions

- Use blue for the training/inference path.
- Use orange for AD-KUCS scoring.
- Use green for the LLM-agent block.
- Use gray or red for monitoring/safety.
- Keep the monitoring branch visually separated so reviewers can immediately see the closed-loop control aspect.
- If the full figure is too crowded, split it into two panels:
  - Panel (a): main active learning loop
  - Panel (b): agent-control and safety mechanism

## Source Mapping to Code

- Main pipeline: `src/main.py`
- AD-KUCS sampler: `src/core/sampler.py`
- Agent toolbox: `src/agent/toolbox.py`
- Agent ReAct loop: `src/agent/agent_manager.py`
- Prompt system: `src/agent/prompt_template.py`
