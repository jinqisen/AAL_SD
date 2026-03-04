## AAL-SD: Agent-Augmented Active Learning for Landslide Detection in Remote Sensing Imagery

Author A^1, Author B^2, Author C^1,*  
^1 Affiliation 1, City, Country  
^2 Affiliation 2, City, Country  
* Corresponding author: email@example.com

### Abstract

Active learning (AL) is a practical approach for reducing the high annotation cost of remote sensing semantic segmentation, including landslide mapping. Most AL methods rely on fixed acquisition heuristics (e.g., uncertainty or diversity), which may be mismatched to the learning stage and the evolving unlabeled pool. We present Agent-Augmented Active Learning for Landslide Detection (AAL-SD), a framework that integrates an Agent component as a constrained controller interface in the AL loop. The core acquisition strategy is Adaptive Diversity–Knowledge–Uncertainty Sampling (AD-KUCS), which scores each unlabeled sample using a convex combination of uncertainty and a feature-space novelty score defined by coreset-to-labeled distance. In the default setting, the mixing weight is produced by a deterministic warmup + risk-based closed-loop policy driven by training signals and logged for auditing; in separate ablations, the LLM can be granted permission to directly propose control actions (e.g., λ or query size) under explicit constraints. We evaluate AAL-SD on the Landslide4Sense benchmark and compare it with multiple baseline and ablation variants under the same labeled-budget protocol; we further report a 4-seed robustness check (seeds 43/44/45/46) and a cross-split generalization setting (TrainData as the AL pool, ValidData as per-round evaluation). Under a fixed training budget (10 epochs per round), AAL-SD remains in the same performance regime as strong baselines; multi-seed results indicate small gaps but no consistent dominance. This motivates positioning AAL-SD primarily as an auditable, constraint-aware control interface with gradient-level evidence for interpreting learning-curve behavior, rather than claiming large numerical gains under a single fixed budget.

### Keywords

Active learning; landslide detection; semantic segmentation; remote sensing; large language model; agent

### 1. Introduction

Semantic segmentation of remote sensing imagery is a fundamental task in Earth observation, with critical applications in disaster management, environmental monitoring, and urban planning (Zhu et al., 2017). Landslide detection, in particular, poses a significant challenge due to the subtle and heterogeneous visual characteristics of landslides, often requiring large amounts of accurately labeled data for training deep learning models (Ghorbanzadeh et al., 2022a). However, pixel-wise annotation is labor-intensive, time-consuming, and expensive, and it commonly requires domain expertise (Ren et al., 2021).

Active learning (AL) has emerged as a promising paradigm to mitigate this annotation burden by selecting informative samples for labeling under a limited budget (Settles, 2009; Ren et al., 2021). Conventional AL strategies—such as uncertainty-based sampling (Settles, 2009) and diversity-based sampling via core-set selection (Sener and Savarese, 2018)—typically rely on a single, fixed heuristic. While often effective, static strategies can be mismatched to the evolving model state and the shifting informativeness of the unlabeled pool across rounds (Ren et al., 2021). For example, uncertainty-focused querying may be beneficial early on, whereas later rounds may benefit from stronger emphasis on coverage and novelty to avoid diminishing returns from repeatedly sampling within a narrow region of the data space.

Recently, Large Language Models (LLMs) have demonstrated strong capabilities in reasoning, planning, and decision-making (Zhao et al., 2023). This motivates treating active learning acquisition as a control problem and exposing a constrained controller interface that can condition decisions on observable training signals. In AAL-SD, controller actions are auditable and reproducible; in dedicated ablations, an LLM can be granted permission to propose control actions under explicit constraints.

Our key contributions are as follows:

1.  **A Constrained Controller Interface for Active Learning**: We propose AAL-SD, a framework that treats acquisition as a constrained control problem and logs controller actions for auditing in an AL loop for remote sensing image segmentation.
2.  **AD-KUCS Acquisition with Closed-Loop Weight Scheduling**: We introduce Adaptive Diversity-Knowledge-Uncertainty Sampling (AD-KUCS), where the uncertainty–novelty trade-off weight is not a fixed hyperparameter but a per-round control variable produced by a warmup + risk-based closed-loop policy (and can be optionally delegated to the LLM in dedicated ablations under explicit constraints).
3.  **Comprehensive Validation and Robustness Check**: We compare AAL-SD against multiple baselines and ablations on Landslide4Sense, and provide multi-seed evaluation to check robustness of key comparisons.

Our results show that exposing acquisition as a constrained, state-aware control interface yields a competitive and interpretable active learning process under fixed training budgets. While multi-seed results do not show consistent numerical superiority over the strongest baselines in this setting, they support the robustness of AAL-SD as a stable, auditable acquisition framework and motivate future work on improving cross-seed controller stability and statistical testing.

### 2. Related Work

This work relates to three research threads. (1) **Remote sensing landslide segmentation** focuses on robust semantic segmentation under heterogeneous appearance and class imbalance, and is supported by benchmark datasets such as Landslide4Sense (Ghorbanzadeh et al., 2022a). (2) **Active learning for dense prediction** commonly combines uncertainty estimation (Settles, 2009), diversity/representativeness via core-set selection (Sener and Savarese, 2018), and Bayesian approximations for acquisition (Gal et al., 2017) to select informative samples under a labeling budget. (3) **Controllers for iterative learning systems** study how to steer iterative processes using state summaries and constrained actions; in our setting, acquisition is treated as a constrained control interface, and LLM-based control is evaluated as an optional ablation (Zhao et al., 2023).

### 3. Materials and Methods

Our proposed AAL-SD framework integrates an Agent component into the active learning loop to expose acquisition as a constrained control interface. Importantly, AD-KUCS is the acquisition (query) algorithm inside the AL loop rather than the segmentation network itself; the segmentation model (e.g., DeepLabV3+ as the base learner) performs training and inference, while AD-KUCS only uses the model’s probability maps and feature embeddings to rank unlabeled samples and select a query batch. The component view and the end-to-end loop are shown in Figure 1a–1b.

![Figure 1a: AAL-SD Framework Architecture (Component View)](./Figure1_AAL_SD_Architecture.png)

![Figure 1b: Active Learning Loop and Where AD-KUCS Operates](./Figure1b_AAL_SD_Active_Learning_Loop.png)

#### 3.1. Adaptive Diversity-Knowledge-Uncertainty Sampling (AD-KUCS)

The core of our sample selection is the AD-KUCS algorithm, which computes a score for each unlabeled sample `x` based on a weighted combination of its uncertainty `U(x)` and knowledge gain `K(x)`.

**Uncertainty Score U(x)**: We quantify the uncertainty of an image `x` by computing the mean of pixel-wise entropy across the entire image:

$$U(x) = \frac{1}{N} \sum_{i=1}^{N} H(p_i)$$

where N is the total number of pixels, and `H(p_i)` is the Shannon entropy of the predicted probability distribution for pixel `i`. This provides a holistic measure of model uncertainty.

**Knowledge Gain Score K(x)**: The knowledge gain score measures feature-space novelty with respect to the labeled set using a coreset-to-labeled distance. Let `f(x)` be the feature embedding of sample `x` extracted via a forward hook on the encoder backbone followed by global average pooling. Given labeled features `F_L = {f(l) | l ∈ L}`, we define:

-   **Nearest-labeled distance**:  `d(x, L) = min_{l ∈ L} ||f(x) - f(l)||_2`.
-   **Normalization**: we normalize distances by the maximum pairwise distance among labeled features, `D_L = max_{l_i,l_j ∈ L} ||f(l_i) - f(l_j)||_2`.

The resulting knowledge gain score is:

$K(x) = \frac{d(x, L)}{D_L}.$

This definition encourages selecting samples that are far from the current labeled coverage in the representation space.

**Score Normalization**: In each round, we normalize the raw per-sample scores across the unlabeled pool using min–max scaling:

$$S_{norm}(x) = \frac{S(x) - \min_{x' \in U} S(x')}{\max_{x' \in U} S(x') - \min_{x' \in U} S(x')}.$$

**Combined Score**: The final score for a sample `x` is a linear combination of normalized uncertainty and normalized knowledge gain:

$$Score(x) = (1 - \lambda_t) \cdot U_{norm}(x) + \lambda_t \cdot K_{norm}(x)$$

Crucially, the weighting parameter `λ_t` is not a fixed hyperparameter but a per-round control variable. In our default setting, `λ_t` is produced by a deterministic warmup + risk-based closed-loop policy based on the observed training state; in separate ablations, the LLM can be granted permission to propose `λ_t` under explicit constraints.

#### 3.2. Agent as a Constrained Controller Interface

AAL-SD exposes the acquisition strategy as a constrained control interface. At the beginning of each active learning round, the controller is provided with a state summary, including:

-   The current round number and total rounds.
-   The model's performance on a validation set (mIoU, F1-score).
-   The performance change since the last round.
-   The distribution of U(x) and K(x) scores across the unlabeled pool.
-   The previously applied control value (e.g., `λ_t`).

In the default setting, a deterministic warmup + risk-based closed-loop policy maps the observed training signals to the per-round control value `λ_t`, and the applied action is logged for auditing and reproducibility. In selected ablations, the system can grant an LLM permission to directly propose control actions (e.g., `λ_t` or per-round query size) under explicit constraints and hard bounds.

To avoid confounding from “more training” and keep the focus on label efficiency, we use a fixed training protocol in this paper: all methods use 10 epochs per round; adaptive training-budget scheduling (e.g., dynamically controlling per-round epochs) is left to future work.

### 4. Experiments and Results

#### 4.1. Experimental Setup

- **Dataset**: We evaluate our framework on the Landslide4Sense dataset (Ghorbanzadeh et al., 2022a), a benchmark for landslide detection containing 3,799 images with pixel-level annotations.
- **Model**: We use DeepLabV3+ implemented by `segmentation_models_pytorch`, with a ResNet-50 encoder initialized from ImageNet pre-training and adapted to 14-channel inputs.
- **Training Protocol**: To keep training budgets comparable, all methods use 10 epochs per round.
- **Active Learning Protocol**: We split the 3,799 images into a test pool of 760 images and a training pool of 3,039 images. We initialize AL with 151 labeled images (5% of the training pool) and 2,888 unlabeled images. We run 15 training rounds with a per-round query size of 88 for the first 14 rounds (no acquisition after the final training round), yielding a final labeled size of 1,383. The total labeling budget is set to 1,519 images and is used to normalize ALC.
- **Cross-split Generalization Protocol**: To better approximate external-domain evaluation (cross-scene validation), we additionally run a cross-split setting where TrainData is used as the AL pool (labeled/unlabeled) and ValidData is used as the per-round evaluation set (run: `run_src_full_model_with_baselines_seed42__eval_val`). In this setting, we report performance at the budget-aligned point |L|=1,383 for fair comparison to the main protocol.
- **Reproducibility**: Per-experiment configuration snapshots are stored in the run manifest.

#### 4.2. Evaluation Metric

Our primary evaluation metric is the **Area Under the Learning Curve (ALC)**, which measures the overall learning efficiency. The learning curve plots the model's mIoU on a fixed test set as a function of the number of labeled samples. A higher ALC indicates a more efficient learning strategy.

Formally, given mIoU values evaluated after each round at labeled sizes $\{n_t\}_{t=0}^T$ with corresponding $\{m_t\}_{t=0}^T$, we normalize the labeled sizes by the total budget $B$ and compute:

$$ALC = \int_0^1 m(x)\, dx,$$

where $x_t = n_t / B$ and $m(x)$ is obtained by trapezoidal interpolation over the points $\{(x_t, m_t)\}$. If the final labeled size is smaller than $B$, we pad the curve to $x=1$ using the last observed performance.

#### 4.2.1. Gradient Evidence and a Proxy of the Generalization Trajectory

To make the claim “$\lambda_t$ / query size changes the labeled-data distribution, which in turn changes optimization and the generalization trajectory” empirically testable, we record per-epoch gradient statistics in the trace: the global gradient norm on training batches (with backbone/head breakdown), and gradient direction consistency (cosine similarity between consecutive training batches, and cosine similarity between the mean training gradient and a probed gradient on a sampled test batch). These quantities serve as optimization proxies to help interpret why different acquisition strategies yield different learning-curve shapes (and thus different ALC) under the same fixed training budget.

#### 4.3. Baselines and Ablations

We include both standard active learning baselines and structured ablations to isolate each component of AAL-SD.

- **Baselines**: Random, Entropy, Core-set, and BALD. We also report two LLM baselines: an LLM-driven uncertainty sampler (LLM-US) and an LLM-driven random sampler (LLM-RS) that remove our AD-KUCS design while keeping the LLM component.
- **Ablations**: (i) `no_agent` removes the agent and uses the fixed, non-agent AD-KUCS rule, (ii) `fixed_lambda` fixes `λ=0.5`, (iii) `uncertainty_only (λ=0)` and (iv) `knowledge_only (λ=1)` isolate each score term, (v) `agent_control_lambda` and (vi) `agent_control_budget` test partial control.

#### 4.4. Main Results

Table 1 presents the main results under the fixed 10-epoch protocol (run: `20260207_paper`, seed=42).

**Table 1: Main Experimental Results on Landslide4Sense.**

| Method | Category | ALC | Final mIoU | Final F1 |
|---|---:|---:|---:|---:|
| AAL-SD (Full) | Proposed | 0.6654 | 0.7607 | 0.8453 |
| AAL-SD (λ+query-size control) | Ablation | 0.6677 | 0.7660 | 0.8498 |
| Agent control λ | Ablation | 0.6701 | 0.7639 | 0.8479 |
| Agent control query size | Ablation | 0.6680 | 0.7638 | 0.8482 |
| AD-KUCS (no agent, fixed rule) | Ablation | 0.6692 | 0.7639 | 0.8482 |
| AD-KUCS (fixed λ=0.5) | Ablation | 0.6684 | 0.7624 | 0.8467 |
| Uncertainty-only (λ=0) | Ablation | 0.6668 | 0.7613 | 0.8458 |
| Knowledge-only (λ=1) | Ablation | 0.6546 | 0.7497 | 0.8360 |
| Random | Baseline | 0.6586 | 0.7585 | 0.8435 |
| Entropy | Baseline | **0.6702** | **0.7696** | **0.8527** |
| Core-set | Baseline | 0.6547 | 0.7420 | 0.8295 |
| BALD | Baseline | 0.6670 | 0.7571 | 0.8422 |
| LLM-RS baseline | Baseline | 0.6554 | 0.7585 | 0.8435 |
| LLM-US baseline | Baseline | 0.6691 | 0.7628 | 0.8470 |

Figure 2 shows the learning curves for AAL-SD and key baselines under the fixed-epoch protocol.

![Figure 2: Learning Curves](./Figure2_Learning_Curves_generated.png)

Figure 6 visualizes the gradient diagnostics aggregated from the trace, providing an optimization/gradient perspective on learning-curve differences.

![Figure 6: Gradient Diagnostics](./Figure6_Gradient_Diagnostics.png)

From the baseline comparison, Entropy is the strongest traditional heuristic in this setting. Core-set underperforms in both ALC and final metrics, suggesting that pure diversity selection based on the current feature embedding may be less effective than uncertainty-driven querying for landslide segmentation. Under this run, the LLM-US baseline is competitive, indicating that LLM-driven scoring can be effective even without our full controller structure; we therefore include a multi-seed robustness check for the full model and major baselines.

#### 4.4.1. Multi-Seed Results

To assess robustness for submission-quality reporting, we run a 4-seed evaluation (seeds 43/44/45/46) for the full model and major baselines (runs: `run_src_full_model_with_baselines_seed43`–`seed46`). In contrast to the single-seed “full ablation matrix”, this multi-seed set focuses on the most important comparisons: the AAL-SD full model against traditional baselines (Entropy/Core-set/BALD/Random), diversity-driven baselines (DIAL-style/Wang-style), and LLM baselines (LLM-US/LLM-RS). Table 2 reports mean±std across seeds.

**Table 2: Multi-Seed Summary (mean±std over 4 seeds).**

| Method | ALC (mean±std) | Final mIoU (mean±std) | Final F1 (mean±std) |
|---|---:|---:|---:|
| Entropy | **0.6695±0.0019** | 0.7614±0.0031 | 0.8459±0.0026 |
| Wang-style | 0.6691±0.0041 | 0.7591±0.0052 | 0.8440±0.0044 |
| LLM-US baseline | 0.6683±0.0020 | **0.7637±0.0028** | **0.8478±0.0023** |
| AAL-SD (Full) | 0.6678±0.0023 | 0.7603±0.0043 | 0.8450±0.0036 |
| DIAL-style | 0.6641±0.0041 | 0.7574±0.0043 | 0.8425±0.0037 |
| BALD | 0.6640±0.0025 | 0.7564±0.0045 | 0.8415±0.0038 |
| LLM-RS baseline | 0.6562±0.0033 | 0.7521±0.0078 | 0.8381±0.0065 |
| Core-set | 0.6551±0.0032 | 0.7487±0.0039 | 0.8351±0.0034 |
| Random | 0.6538±0.0040 | 0.7511±0.0060 | 0.8372±0.0050 |

These results indicate that AAL-SD is stable and competitive but does not consistently outperform the strongest baselines under this fixed-budget setup. Accordingly, we avoid “significant improvement” wording and emphasize AAL-SD’s auditable constrained-control interface and trace-based diagnostics as the primary contribution.

#### 4.4.2. Cross-split Generalization (TrainData AL Pool, ValidData Evaluation)

To complement in-split evaluation, we report a cross-split setting where the AL pool is restricted to TrainData and each round is evaluated on ValidData (run: `run_src_full_model_with_baselines_seed42__eval_val`). Since this run starts from a larger initial labeled set and can exceed the total budget used for ALC normalization, we report the budget-aligned point |L|=1,383 as the primary comparison point.

**Table 2b: Cross-split results at |L|=1,383 (seed=42, external-domain evaluation on ValidData).**

| Method | mIoU@|L|=1383 | F1@|L|=1383 |
|---|---:|---:|
| AD-KUCS (fixed λ=0.5) | **0.7318** | **0.8191** |
| BALD | 0.7227 | 0.8109 |
| AAL-SD (V5, calibrated risk) | 0.7219 | 0.8101 |
| Entropy | 0.7107 | 0.7992 |
| Core-set | 0.7040 | 0.7930 |
| LLM-US baseline | 0.6931 | 0.7818 |
| Random | 0.6885 | 0.7771 |

These cross-split results show a noticeable performance drop relative to in-split evaluation, consistent with a domain shift between the AL pool and the evaluation split. Importantly, relative differences among acquisition strategies remain measurable under this more generalization-oriented protocol, providing complementary evidence beyond same-split testing.

#### 4.5. Ablation Study

The ablation results suggest that the relative benefit of the agent/controller can vary under a fixed training budget. In this run, several non-agent variants (e.g., `no_agent`, `fixed_lambda`) are competitive in ALC, while single-term scoring (`knowledge_only`) is notably weaker. This supports the need to balance uncertainty and knowledge gain, and also indicates that controller design and constraints are important for consistent gains.

**Table 3: Focused Ablation Summary.**

| Variant | ALC | Final mIoU | Final F1 |
|---|---:|---:|---:|
| AAL-SD (Full) | 0.6654 | 0.7607 | 0.8453 |
| AD-KUCS (no agent, fixed rule) | 0.6692 | 0.7639 | 0.8482 |
| AD-KUCS (fixed λ=0.5) | 0.6684 | 0.7624 | 0.8467 |
| Uncertainty-only (λ=0) | 0.6668 | 0.7613 | 0.8458 |
| Knowledge-only (λ=1) | 0.6546 | 0.7497 | 0.8360 |
| AAL-SD (λ+query-size control) | 0.6677 | 0.7660 | 0.8498 |

#### 4.6. Controller Behavior

Figure 3 visualizes an example controller trajectory for AAL-SD. Under the fixed training protocol (10 epochs per round), we only analyze controllable acquisition decisions (λ and query size) and do not treat training-budget control as a controller action in this paper.

![Figure 3: Controller Trajectory of AAL-SD (Full)](./Figure3_Controller_Trajectory_Full_Model.png)

For completeness, we provide bar-chart summaries of ALC and final mIoU across all variants in Figures 4–5.

![Figure 4: ALC Comparison](./Figure4_ALC_Bar.png)

![Figure 5: Final mIoU Comparison](./Figure5_Final_mIoU_Bar.png)

### 5. Conclusion

In this paper, we proposed AAL-SD, a framework that exposes active learning acquisition as a constrained controller interface for remote sensing image segmentation. By combining uncertainty with a feature-space novelty term and scheduling their trade-off via a warmup + risk-based closed-loop policy (with optional LLM control in dedicated ablations), AAL-SD adapts its acquisition focus across rounds while keeping actions auditable. Under a fixed training protocol (10 epochs per round), AAL-SD is competitive with strong baselines in label efficiency (ALC) in a single-seed full comparison, and a 4-seed robustness check shows small gaps but no consistent dominance over the strongest baselines. We further report a cross-split generalization setting (TrainData as AL pool, ValidData as per-round evaluation) to complement same-split testing and to better approximate cross-scene validation. Future work should focus on improving controller stability (across seeds and under stricter selection constraints) and adding stronger statistical testing (more seeds, confidence intervals, and paired tests), while continuing to strengthen trace-based interpretability and reproducibility.

### Code Availability

The implementation used in this study is available in the accompanying repository and will be archived in a long-term public repository upon acceptance.

### Data Availability

The Landslide4Sense dataset is publicly available (Ghorbanzadeh et al., 2022a, 2022b).

### Declaration of Competing Interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

### CRediT Author Statement

Author A: Conceptualization, Methodology, Software, Writing – original draft.  
Author B: Investigation, Validation, Writing – review & editing.  
Author C: Supervision, Funding acquisition, Writing – review & editing.

### References

Gal, Y., Islam, R., Ghahramani, Z., 2017. Deep Bayesian Active Learning with Image Data. arXiv:1703.02910. https://doi.org/10.48550/arXiv.1703.02910

Ghorbanzadeh, O., Xu, Y., Ghamisi, P., Kopp, M., Kreil, D., 2022a. Landslide4Sense: Reference Benchmark Data and Deep Learning Models for Landslide Detection. IEEE Transactions on Geoscience and Remote Sensing 60, 1–17. https://doi.org/10.1109/TGRS.2022.3215209

Ghorbanzadeh, O., Xu, Y., Zhao, H., Wang, J., Zhong, Y., Zhao, D., Zang, Q., Wang, S., Zhang, F., Shi, Y., Zhu, X.X., Bai, L., Li, W., Peng, W., Ghamisi, P., 2022b. The Outcome of the 2022 Landslide4Sense Competition: Advanced Landslide Detection From Multisource Satellite Imagery. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing 15, 9927–9942. https://doi.org/10.1109/JSTARS.2022.3220845

He, K., Zhang, X., Ren, S., Sun, J., 2016. Deep Residual Learning for Image Recognition. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 770–778. https://doi.org/10.1109/CVPR.2016.90

Ren, P., Xiao, Y., Chang, X., Huang, P.-Y., Li, Z., Gupta, B.B., Chen, X., Wang, X., 2021. A Survey of Deep Active Learning. ACM Computing Surveys 54(9), 180:1–180:40. https://doi.org/10.1145/3472291

Ronneberger, O., Fischer, P., Brox, T., 2015. U-Net: Convolutional Networks for Biomedical Image Segmentation. In: Medical Image Computing and Computer-Assisted Intervention (MICCAI), pp. 234–241. https://doi.org/10.1007/978-3-319-24574-4_28

Sener, O., Savarese, S., 2018. Active Learning for Convolutional Neural Networks: A Core-Set Approach. In: International Conference on Learning Representations (ICLR). https://doi.org/10.48550/arXiv.1708.00489

Settles, B., 2009. Active Learning Literature Survey. Computer Sciences Technical Report 1648, University of Wisconsin–Madison. https://burrsettles.com/pub/settles.activelearning.pdf

Zhao, W.X., Zhou, K., Li, J., Tang, T., Wang, X., Hou, Y., Min, Y., Zhang, B., Zhang, J., Dong, Z., Du, Y., Yang, C., Chen, Y., Chen, Z., Jiang, J., Ren, R., Li, Y., Tang, X., Liu, Z., Liu, P., Nie, J.-Y., Wen, J.-R., 2023. A Survey of Large Language Models. arXiv:2303.18223. https://doi.org/10.48550/arXiv.2303.18223

Zhu, X.X., Tuia, D., Mou, L., Xia, G.-S., Zhang, L., Xu, F., Fraundorfer, F., 2017. Deep Learning in Remote Sensing: A Comprehensive Review and List of Resources. IEEE Geoscience and Remote Sensing Magazine 5(4), 8–36. https://doi.org/10.1109/MGRS.2017.2762307
