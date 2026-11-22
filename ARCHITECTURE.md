# Architecture & Research Specification

## Title
**Adversarial Robustness of Vision-Language Models using PGD and Self-Calibrated Consistency**

Primary Research Reference:
> Jiaxiang Liu, Jiawei Du, Xiao Liu, Shangyang Li, Songchen Ma, Changshuo Wang, Prayag Tiwari, Mingkun Xu.  
> *"Self-Calibrated Consistency can Fight Back for Adversarial Robustness in Vision-Language Models"*, Proceedings of the 43rd International Conference on Machine Learning (ICML 2026).

---

## 1. System Architecture Overview

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Browser Frontend (React 18 + Tailwind)               │
│                               http://127.0.0.1:4173                         │
│   ├── Comparison Dashboard: Clean vs Attacked vs Defended (Side-by-Side)   │
│   ├── Amplified Perturbation Maps (15x contrast heatmap)                    │
│   ├── Interactive Hyperparameters: eps_a, steps_a, eps_d, lambda_cm, T, V   │
│   ├── Batch Benchmarking Engine: Clean Acc, Robust Acc, ASR, DRR           │
│   └── ICML 2026 Algorithm Deep-Dive & Empirical Verification Table          │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ REST / JSON (HTTP)
┌──────────────────────────────────────▼──────────────────────────────────────┐
│                        FastAPI Backend Server (:8000)                       │
│   ├── GET  /api/health            (System & Model status)                   │
│   ├── GET  /api/samples           (Preloaded benchmark datasets & labels)   │
│   ├── POST /api/classify          (Real zero-shot CLIP ViT-B/32 baseline)   │
│   ├── POST /api/attack            (Untargeted white-box PGD attack)         │
│   ├── POST /api/defend            (Self-Calibrated Consistency test-time)   │
│   ├── POST /api/pipeline          (Full Clean -> Attack -> Defend pipeline) │
│   └── POST /api/evaluate/batch    (Real batch dataset evaluation)           │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ Differentiable PyTorch Tensors
┌──────────────────────────────────────▼──────────────────────────────────────┐
│                    Machine Learning Implementation (backend/scc/)           │
│                                                                             │
│   1. clip_model.py (CLIPEngine)                                             │
│      - Frozen OpenAI CLIP ViT-B/32 backbone (512-dim embedding)             │
│      - Differentiable image normalization: (x - μ) / σ                      │
│      - Text tokenization & normalized prompt embeddings T                   │
│      - Logits: z = τ · ⟨f_img(x), t_k⟩, Softmax probabilities               │
│      - Semantic margin: m(x) = ⟨f, t_y*⟩ - max_{j ≠ y*} ⟨f, t_j⟩            │
│                                                                             │
│   2. attacks.py (PGD Adversarial Attack)                                    │
│      - Untargeted white-box PGD under L_inf norm ball                       │
│      - Random uniform initialization in [-ε_a, ε_a]                         │
│      - Maximizes classification cross-entropy: L_CE = -log p(y* | x)        │
│      - Gradient ascent: δ ← clamp(δ + α · sign(∇_δ L_CE), -ε_a, ε_a)        │
│      - Clamping image to valid pixel domain [0, 1]                          │
│                                                                             │
│   3. defense.py (Self-Calibrated Consistency - SCC Defense)                 │
│      - Step 1: TTC warm-up optimizing L_drift = ||f(x+δ) - f(x)||_2^2       │
│      - Step 2: Multi-view augmentation (flip + Gaussian noise σ = 6/255)    │
│      - Step 3: Temperature-sharpened soft pseudo-label (T = 0.5)            │
│      - Step 4: Soft prototype construction: t_soft = Σ q_k^sharp · t_k      │
│      - Step 5: Shared corrective perturbation optimization (Sign-PGD):      │
│                L_cm = ⟨f, t_soft⟩ - max_{k ≠ y_hat} ⟨f, t_k⟩                │
│                L_SCC = λ_cm · L_cm + ||f - f_orig||_2^2                     │
│      - Step 6: Step-weighted fusion across iterations: w_s = s / Σ s        │
│      - Step 7: Final multi-view logit aggregation                           │
│                                                                             │
│   4. evaluator.py (Batch Benchmark Engine)                                  │
│      - Clean Accuracy (%), Robust Accuracy (%), Defended Accuracy (%)       │
│      - Attack Success Rate (ASR %), Defense Recovery Rate (DRR %)           │
│                                                                             │
│   5. sample_data.py (Curated Benchmark Samples & Taxonomies)                │
│      - High-contrast visual benchmark samples (Dog, Wolf, Airplane, etc.)   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Mathematical Formulations

### 2.1 CLIP Zero-Shot Baseline
Given an image $x \in [0, 1]^{3 \times 224 \times 224}$ and candidate text prompts $\{t_k\}_{k=1}^K$:
$$f = \frac{f_{img}(\text{norm}(x))}{\|f_{img}(\text{norm}(x))\|_2}, \quad t_k = \frac{g_{text}(t_k)}{\|g_{text}(t_k)\|_2}$$
$$z_k = \tau \langle f, t_k \rangle, \quad p(y = k \mid x) = \frac{\exp(z_k)}{\sum_j \exp(z_j)}$$

### 2.2 Semantic Margin (Eq. 3)
$$m(x) = \langle f(x), t_{y^*} \rangle - \max_{j \neq y^*} \langle f(x), t_j \rangle$$
- When clean: $m(x) > 0$.
- Under adversarial attack: $m(x + \delta_{atk}) < 0$, indicating that a competitor text embedding dominates.

### 2.3 PGD Adversarial Attack
Untargeted white-box attack maximizing cross-entropy loss against ground truth $y^*$:
$$\delta^{(0)} \sim \text{Uniform}(-\epsilon_a, \epsilon_a)$$
$$\delta^{(t+1)} = \Pi_{\|\delta\|_\infty \le \epsilon_a} \left[ \delta^{(t)} + \alpha_a \cdot \text{sign}\left(\nabla_\delta \mathcal{L}_{CE}(z(x + \delta^{(t)}), y^*)\right) \right]$$
$$x_{adv} = \text{clamp}(x + \delta^{(T)}, 0, 1)$$

### 2.4 Self-Calibrated Consistency (SCC Defense - Algorithm 1)
1. **TTC Warm-up**:
   $$\delta_{warm}^{(t+1)} = \Pi_{\|\delta\|_\infty \le \epsilon_d} \left[ \delta^{(t)} + \alpha_w \cdot \text{sign}\left(\nabla_\delta \|f(x_{adv} + \delta) - f(x_{adv})\|_2^2\right) \right]$$
   $$\delta_{warm} = \sum_{s=1}^{S_w} \frac{s}{\sum j} \delta^{(s)}, \quad x_w = \text{clamp}(x_{adv} + \delta_{warm}, 0, 1)$$

2. **Multi-View Augmentation & Soft Pseudo-Label (Eq. 6-7)**:
   $$v_i(x_w) = \text{clamp}(\text{flip}_h^{\mathbf{1}_{\{i \text{ odd}\}}}(x_w) + \eta_i, 0, 1), \quad \eta_i \sim \mathcal{N}(0, (\sigma/255)^2 I)$$
   $$\bar{z} = \frac{1}{V} \sum_{i=1}^V \langle f(v_i(x_w)), t_k \rangle, \quad q = \text{softmax}(\bar{z})$$
   $$q_k^{sharp} = \frac{(q_k)^{1/T}}{\sum_j (q_j)^{1/T}}, \quad \hat{y} = \arg\max_k q_k^{sharp}$$
   $$t_{soft} = \sum_k q_k^{sharp} t_k, \quad \hat{t}_{soft} = \frac{t_{soft}}{\|t_{soft}\|_2}$$

3. **Joint Counterattack Optimization (Eq. 5, 8, 9)**:
   $$\mathcal{L}_{cm}(x_{adv}, \delta) = \langle f(x_{adv} + \delta), \hat{t}_{soft} \rangle - \max_{k \neq \hat{y}} \langle f(x_{adv} + \delta), t_k \rangle$$
   $$\mathcal{L}_{drift} = \|f(x_{adv} + \delta) - f(x_{adv})\|_2^2$$
   $$\mathcal{L}_{SCC} = \lambda_{cm} \mathcal{L}_{cm} + \mathcal{L}_{drift}$$
   $$\delta^{(s+1)} = \Pi_{\|\delta\|_\infty \le \epsilon_d} \left[ \delta^{(s)} + \alpha_d \cdot \text{sign}\left(\nabla_\delta \mathcal{L}_{SCC}\right) \right]$$

4. **Step-Weighted Fusion & Final Aggregation**:
   $$\delta_{def} = \sum_{s=1}^S \frac{s}{\sum j} \delta^{(s)}, \quad x_{def} = \text{clamp}(x_{adv} + \delta_{def}, 0, 1)$$
   $$\bar{z}_{final} = \frac{1}{V} \sum_{i=1}^V \tau \langle f(v_i(x_{def})), t_k \rangle, \quad \hat{y}_{def} = \arg\max_k \bar{z}_{final}$$
