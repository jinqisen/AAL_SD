## AAL-SD: Agent-Augmented Active Learning for Landslide Detection in Remote Sensing Imagery

Author A^1, Author B^2, Author C^1,*  
^1 Affiliation 1, City, Country  
^2 Affiliation 2, City, Country  
* Corresponding author: email@example.com

### Abstract

Active learning (AL) is a practical approach for reducing the high annotation cost of remote sensing semantic segmentation, including landslide mapping. Most AL methods rely on fixed acquisition heuristics (e.g., uncertainty or diversity), which may be mismatched to the learning stage and the evolving unlabeled pool. We present Agent-Augmented Active Learning for Landslide Detection (AAL-SD), a framework that integrates an Agent component as a constrained controller interface in the AL loop. The core acquisition strategy is Adaptive Diversity-Knowledge-Uncertainty Sampling (AD-KUCS), which scores each unlabeled sample using a convex combination of uncertainty and a feature-space novelty score defined by coreset-to-labeled distance. In the default setting, the mixing weight is produced by a deterministic warmup + risk-based closed-loop policy driven by training signals and logged for auditing; in a dedicated variant, the LLM can be granted permission to directly control lambda under explicit constraints. We evaluate AAL-SD on Landslide4Sense using four seeds (42-45) and compare two full-model variants against six strong baselines under the same fixed-budget protocol. The representative seed-42 run shows that the policy-controlled full model attains the strongest final mIoU (0.7651), while the agent-lambda variant is competitive in ALC. Across four seeds, no single method dominates all metrics: Wang-style achieves the highest mean ALC (0.6682+-0.0037 std), the policy-controlled AAL-SD remains close behind (0.6674+-0.0029), and the agent-lambda variant attains the best mean final mIoU (0.7616+-0.0014). These results motivate positioning AAL-SD primarily as an auditable, constraint-aware control interface with optimization-proxy diagnostics, rather than claiming uniform numerical superiority under a single fixed training budget.

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
- **Seeds and Reporting Scope**: The refreshed paper assets are generated from four runs (`baseline_20260228_124857_seed42` to `seed45`). We use seed 42 as the representative single-seed case study and report four-seed statistics for the methods that are complete across all runs.
- **Reproducibility**: Per-experiment configuration snapshots are stored in the run manifest.

#### 4.2. Evaluation Metric

Our primary evaluation metric is the **Area Under the Learning Curve (ALC)**, which measures the overall learning efficiency. The learning curve plots the model's mIoU on a fixed test set as a function of the number of labeled samples. A higher ALC indicates a more efficient learning strategy.

Formally, given mIoU values evaluated after each round at labeled sizes $\{n_t\}_{t=0}^T$ with corresponding $\{m_t\}_{t=0}^T$, we normalize the labeled sizes by the total budget $B$ and compute:

$$ALC = \int_0^1 m(x)\, dx,$$

where $x_t = n_t / B$ and $m(x)$ is obtained by trapezoidal interpolation over the points $\{(x_t, m_t)\}$. If the final labeled size is smaller than $B$, we pad the curve to $x=1$ using the last observed performance.

#### 4.2.1. Gradient Evidence and a Proxy of the Generalization Trajectory

To make the claim “the acquisition policy changes the labeled-data distribution, which in turn changes optimization and the generalization trajectory” empirically testable, we record per-epoch gradient statistics in the trace: the global gradient norm on training batches (with backbone/head breakdown), and gradient direction consistency (cosine similarity between consecutive training batches, and cosine similarity between the mean training gradient and a probed gradient on a sampled test batch). These quantities serve as optimization proxies to help interpret why different acquisition strategies yield different learning-curve shapes (and thus different ALC) under the same fixed training budget.

#### 4.3. Baselines and Ablations

We include both strong active learning baselines and two full-model AAL-SD variants.

- **Baselines**: Random, Entropy, Core-set, BALD, DIAL-style, and Wang-style.
- **Full-model variants**: (i) `full_model_A_lambda_policy`, the policy-controlled version used as the primary full model in this paper, and (ii) `full_model_B_lambda_agent`, which exposes direct lambda control to the agent under explicit constraints.
- **Legacy ablations**: Earlier seed-42-only ablations (e.g., `no_agent`, `fixed_lambda`, `knowledge_only`, query-size-control variants) remain useful for internal diagnosis, but they are not included in the refreshed multi-seed tables because they are not available across all four runs.

#### 4.4. Main Results

Table 1 presents the refreshed representative single-seed results (seed 42) under the fixed 10-epoch protocol.

**Table 1: Main Experimental Results on Landslide4Sense.**

| Method | Category | ALC | Final mIoU | Final F1 |
|---|---:|---:|---:|---:|
| AAL-SD (policy full, A) | Proposed | 0.6693 | **0.7651** | **0.8490** |
| AAL-SD (agent-lambda, B) | Proposed | 0.6701 | 0.7634 | 0.8476 |
| Entropy | Baseline | 0.6674 | 0.7555 | 0.8409 |
| BALD | Baseline | 0.6673 | 0.7644 | 0.8483 |
| DIAL-style | Baseline | 0.6681 | 0.7626 | 0.8469 |
| Wang-style | Baseline | **0.6714** | 0.7560 | 0.8412 |
| Core-set | Baseline | 0.6578 | 0.7507 | 0.8370 |
| Random | Baseline | 0.6563 | 0.7391 | 0.8268 |

The seed-42 comparison already reveals an important pattern: no method dominates all metrics. Wang-style achieves the best ALC in this representative run, whereas the policy-controlled AAL-SD variant attains the best final mIoU and F1. The agent-lambda variant is also competitive, but its strength in this run is concentrated more in ALC than in terminal segmentation accuracy. This split motivates a multi-seed analysis rather than a single-seed winner-take-all interpretation.

Figure 2 shows the refreshed multi-seed learning curves for the two AAL-SD variants and the six baselines.

![Figure 2: Learning Curves](./Figure2_Learning_Curves_generated.png)

The learning curves confirm that the performance gaps are small but structured. Random and Core-set remain consistently weaker, while Entropy, BALD, DIAL-style, Wang-style, and both AAL-SD variants form the leading group. The curves also suggest that the AAL-SD variants are not merely early-round heuristics; they stay competitive deep into the labeling budget, which is important for the final mapping model used in practice.

#### 4.4.1. Multi-Seed Results

Table 2 reports the refreshed four-seed summary over seeds 42-45. The comparison is restricted to the methods that are complete in all four runs.

**Table 2: Multi-Seed Summary (mean+-std over 4 seeds).**

| Method | ALC (mean+-std) | Final mIoU (mean+-std) | Final F1 (mean+-std) |
|---|---:|---:|---:|
| AAL-SD (policy full, A) | 0.6674+-0.0029 | 0.7607+-0.0046 | 0.8453+-0.0039 |
| AAL-SD (agent-lambda, B) | 0.6664+-0.0036 | **0.7616+-0.0014** | **0.8461+-0.0012** |
| Entropy | 0.6672+-0.0006 | 0.7589+-0.0047 | 0.8438+-0.0039 |
| BALD | 0.6657+-0.0028 | 0.7595+-0.0036 | 0.8443+-0.0030 |
| DIAL-style | 0.6647+-0.0030 | 0.7570+-0.0064 | 0.8421+-0.0055 |
| Wang-style | **0.6682+-0.0037** | 0.7585+-0.0019 | 0.8434+-0.0016 |
| Core-set | 0.6532+-0.0035 | 0.7498+-0.0047 | 0.8362+-0.0039 |
| Random | 0.6545+-0.0043 | 0.7466+-0.0088 | 0.8332+-0.0077 |

The four-seed results reinforce the main message of the refreshed analysis: the ranking depends on which criterion is emphasized. Wang-style has the highest mean ALC, but the margin over the policy-controlled AAL-SD variant is small. The policy-controlled AAL-SD remains ahead of Entropy, BALD, DIAL-style, Core-set, and Random in mean ALC, which supports its use as the primary full model when label efficiency across the whole budget is prioritized. By contrast, the agent-lambda variant reaches the highest mean final mIoU and F1, suggesting that direct lambda control can be beneficial for terminal accuracy even if it does not yield the strongest average ALC. Accordingly, we avoid claiming universal superiority and instead emphasize the two complementary strengths exposed by the refreshed multi-seed evidence.

#### 4.5. Ablation Study

The most relevant refreshed ablation is the comparison between the two full-model variants. Variant A (`full_model_A_lambda_policy`) keeps lambda under deterministic warmup + risk-aware policy control, whereas Variant B (`full_model_B_lambda_agent`) grants the agent explicit lambda-setting authority under constraints.

**Table 3: Focused Full-Model Comparison.**

| Variant | Seed-42 ALC | Seed-42 Final mIoU | 4-seed ALC (mean+-std) | 4-seed Final mIoU (mean+-std) | 4-seed Mean Overfit Risk |
|---|---:|---:|---:|---:|---:|
| AAL-SD (policy full, A) | 0.6693 | **0.7651** | **0.6674+-0.0029** | 0.7607+-0.0046 | 0.5580 |
| AAL-SD (agent-lambda, B) | **0.6701** | 0.7634 | 0.6664+-0.0036 | **0.7616+-0.0014** | **0.4682** |

This table explains why we retain Variant A as the primary full model for the paper. Variant B is attractive from an innovation standpoint and achieves the best mean final mIoU across seeds, but Variant A offers a stronger label-efficiency profile in the multi-seed average and keeps the control logic primarily in the explicit closed-loop policy rather than delegating lambda updates to the agent itself. In other words, Variant A is the better match for the paper's central theoretical claim, while Variant B is the stronger "direct agent control" contrast.

#### 4.6. Controller Behavior

Figure 3 compares the seed-42 controller trajectories of the two full-model variants. The top panel shows mIoU by round, the middle panels show effective lambda, and the lower panels report optimization-proxy signals (`grad_train_val_cos_last` and overfit risk).

![Figure 3: Controller Trajectory of AAL-SD (Full)](./Figure3_Controller_Trajectory_Full_Model.png)

The figure highlights the qualitative difference between the two full-model variants. Variant A stays in a more conservative lambda regime and repeatedly falls back to uncertainty-biased behavior when severe overfitting rules are triggered. Variant B keeps a higher average effective lambda and shows smoother final-accuracy behavior, which is consistent with its stronger mean final mIoU in Table 2. At the same time, Variant A's tighter risk-aware policy appears to support stronger whole-curve efficiency, which is consistent with its better multi-seed mean ALC.

For completeness, Figures 4-5 summarize the refreshed multi-seed ALC and final mIoU comparisons.

![Figure 4: ALC Comparison](./Figure4_ALC_Bar.png)

![Figure 5: Final mIoU Comparison](./Figure5_Final_mIoU_Bar.png)

Figure 6 aggregates optimization-proxy diagnostics across methods.

![Figure 6: Gradient Diagnostics](./Figure6_Gradient_Diagnostics.png)

These diagnostics support a cautious version of the paper's deeper claim. We do not argue that the controller directly changes the optimizer or analytically optimizes gradients. Instead, the refreshed evidence shows that acquisition policy, gradient-derived risk signals, and subsequent optimization behavior are coupled in practice. Across four seeds, the policy-controlled AAL-SD has a lower mean effective lambda than the agent-lambda variant (0.181 versus 0.211), a higher negative-gradient rate, and a higher mean overfit-risk proxy, yet it still achieves the stronger mean ALC. This pattern is consistent with the idea that the controller is indirectly shaping later optimization trajectories through data acquisition, rather than simply following a fixed uncertainty/diversity heuristic.

### 5. Conclusion

In this paper, we proposed AAL-SD, a framework that exposes active learning acquisition as a constrained controller interface for remote sensing image segmentation. By combining uncertainty with a feature-space novelty term and scheduling their trade-off via a warmup + risk-based closed-loop policy (with optional direct agent control in a dedicated variant), AAL-SD adapts its acquisition focus across rounds while keeping actions auditable. The refreshed four-seed analysis shows that AAL-SD is genuinely competitive but not uniformly dominant: the policy-controlled full model is among the strongest methods in mean ALC, the agent-lambda variant attains the highest mean final mIoU, and Wang-style remains a very strong baseline in whole-curve efficiency. This evidence supports a measured conclusion. AAL-SD's main value lies in making acquisition state-aware, auditable, and theoretically interpretable through optimization-proxy diagnostics, rather than in claiming universal numerical superiority under a fixed 10-epoch-per-round budget. Future work should strengthen multi-seed statistical testing, improve policy stability under severe overfitting signals, and further study how acquisition control indirectly shapes downstream optimization trajectories.

### Code Availability

The implementation used in this study is publicly available at `https://github.com/jinqisen/AAL_SD`.

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
