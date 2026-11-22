# TTC Paper Notes & Hyperparameters
Reference: Xing et al. (CVPR 2025) - "CLIP is strong enough to fight back: Test-Time Counterattack for Zero-Shot Adversarial Robustness"
Secondary reference: Liu et al. (ICML 2026) - "Self-Calibrated Consistency can Fight Back for Adversarial Robustness in Vision-Language Models" (SCC)

## Core Concept of TTC (Test-Time Counterattack)
1. **Adversarial Detection / Triggering**:
   - TTC observes that adversarial perturbation pulls image features away from natural image-text manifold, causing pseudo-stability or low maximum similarity.
   - Confidence / similarity threshold $\tau$: if $\max_k \cos(f(x), t_k) < \tau$ (or based on prediction entropy/margin), the sample is flagged as under attack.
   - For benchmark evaluation under guaranteed attack (PGD-10), counterattack is applied to all test samples.

2. **Counterattack Objective**:
   - Maximizes feature deviation from adversarial input:
     $$L_{\text{drift}}(\delta) = \|f_{\text{img}}(x_{\text{adv}} + \delta) - f_{\text{img}}(x_{\text{adv}})\|_2^2$$
   - This escapes the local adversarial pseudo-basin without requiring ground-truth labels or suffering from biased hard-negative label traps.

3. **Step-Weighted Fusion**:
   - Across $S$ gradient ascent iterations, intermediate perturbations $\delta^{(1)}, \delta^{(2)}, \dots, \delta^{(S)}$ are recorded.
   - Weighted fusion combines them:
     $$\delta_{\text{fused}} = \sum_{s=1}^S w_s \delta^{(s)}, \quad w_s = \frac{s}{\sum_{j=1}^S j}$$
   - Fused perturbation is projected back to the $\ell_\infty$ ball: $\Pi_{\epsilon_d}(\delta_{\text{fused}})$.

4. **Reference Hyperparameters (from SCC Paper Table 1 & Sec 4.1)**:
   - Counterattack budget: $\epsilon_d = 4/255$
   - Number of counterattack steps: $S = 2$ (Table 1 setting; for warmup $w = 5$)
   - Step size: $\alpha = 1/255$
   - Norm: $\ell_\infty$
   - Backbone: CLIP ViT-B/32 (OpenAI weights)
