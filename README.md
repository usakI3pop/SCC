# Adversarial Robustness of Vision-Language Models using PGD and Self-Calibrated Consistency

[![Paper](https://img.shields.io/badge/ICML_2026-Research_Paper-blue.svg)](https://jxliu-ai.github.io/scc-page/)
[![Model](https://img.shields.io/badge/Model-CLIP_ViT--B%2F32-emerald.svg)](https://huggingface.co/openai/clip-vit-base-patch32)
[![Framework](https://img.shields.io/badge/Backend-FastAPI_+_PyTorch-red.svg)]()
[![UI](https://img.shields.io/badge/Frontend-React_18_+_Tailwind_CSS-cyan.svg)]()

A fully functional research demonstration web application implementing **Self-Calibrated Consistency (SCC)**, a test-time defense framework against untargeted white-box Projected Gradient Descent (PGD) attacks on Vision-Language Models (CLIP ViT-B/32).

> **Primary Research Reference:**  
> Jiaxiang Liu, Jiawei Du, Xiao Liu, Shangyang Li, Songchen Ma, Changshuo Wang, Prayag Tiwari, Mingkun Xu.  
> *"Self-Calibrated Consistency can Fight Back for Adversarial Robustness in Vision-Language Models"*, Proceedings of the 43rd International Conference on Machine Learning (ICML 2026).  
> Project Page: [jxliu-ai.github.io/scc-page](https://jxliu-ai.github.io/scc-page/)

---

## Key Features

1. **CLIP Zero-Shot Baseline**
   - Upload any image or choose curated benchmark samples (Dog, Wolf, Airplane, Automobile, Cat, Ship).
   - Enter custom candidate class labels or select taxonomy presets (CIFAR-10, Animals, Vehicles).
   - Real zero-shot CLIP classification with top-5 confidence probabilities and cosine similarity logits.

2. **Untargeted White-Box PGD Attack**
   - Direct back-propagation of gradients through the CLIP image encoder.
   - Configurable $L_\infty$ perturbation budget $\epsilon_a$ ($1/255, 2/255, 4/255, 8/255, 16/255$).
   - Configurable iteration count ($1$ to $20$ steps) with random uniform initialization inside the $L_\infty$ ball.
   - Untargeted cross-entropy maximization against the ground-truth label.
   - Strict projection to the allowed $L_\infty$ ball and pixel clamping to $[0, 1]$.
   - Visualizes adversarial image and amplified perturbation heatmap ($15\times$ contrast).

3. **Self-Calibrated Consistency (SCC) Test-Time Defense (Algorithm 1)**
   - **TTC Counterattack Warm-up**: Optimizes feature-deviation drift $\mathcal{L}_{drift} = \|f_{img}(x + \delta) - f_{img}(x)\|_2^2$ to escape pseudo-stability without label bias.
   - **Multi-View Augmentation**: Form $V$ augmented views (horizontal flip + Gaussian noise $\sigma = 6/255$) to reduce view sensitivity and variance by $1/V$.
   - **Soft Pseudo-Label Construction**: Sharpened temperature $T = 0.5$ over view-averaged logits to form soft prototype $t_{soft} = \sum_k q_k^{sharp} t_k$.
   - **Semantic Consistency Objective with Hard-Negative Separation**: $\mathcal{L}_{cm} = \langle f, \hat{t}_{soft} \rangle - \max_{k \neq \hat{y}} \langle f, t_k \rangle$ repels confusable nearest competitors (Proposition 3.1).
   - **Spatial Consistency**: Couples all views under a shared corrective perturbation $\delta$.
   - **Step-Weighted Fusion**: Fuses intermediate perturbations across iterations with weights $w_s = s / \sum j$ for smooth recovery.
   - **Final Multi-View Logit Aggregation**: Averages logits across views of $x + \delta_{def}$ for final calibrated prediction.

4. **Side-by-Side Comparison Dashboard**
   - Side-by-side presentation of **Clean Baseline**, **PGD Attacked**, and **SCC Defended** stages.
   - Top-5 prediction probability bars with raw logit values and Ground Truth badges.
   - Live semantic margin tracking: $m(x) = \langle \hat{f}(x), \hat{t}_{y^*} \rangle - \max_{j \neq y^*} \langle \hat{f}(x), \hat{t}_j \rangle$.
   - Perturbation metrics ($L_\infty$ and $L_2$ norms in $[0, 255]$ and $[0, 1]$).
   - Execution time breakdown per stage.

5. **Batch Evaluation Benchmark**
   - Real batch testing on curated benchmark datasets.
   - Reports exact research metrics:
     - **Clean Accuracy (%)**: Accuracy on clean images.
     - **Robust Accuracy (%)**: Accuracy under PGD attack without defense.
     - **Defended Accuracy (%)**: Accuracy restored by SCC defense.
     - **Attack Success Rate (ASR %)**: Proportion of clean-correct samples successfully fooled.
     - **Defense Recovery Rate (DRR %)**: Proportion of fooled samples recovered by SCC.
   - Per-sample audit table showing margins, predictions, and status badges.

---

## Repository Structure

```text
├── backend/
│   ├── main.py                 # FastAPI application and endpoint handlers
│   ├── requirements.txt        # Python dependencies
│   ├── scc/
│   │   ├── __init__.py
│   │   ├── clip_model.py       # Differentiable CLIP ViT-B/32 engine & frozen encoder
│   │   ├── attacks.py          # Untargeted white-box PGD attack implementation
│   │   ├── defense.py          # SCC test-time defense algorithm (Liu et al., ICML 2026)
│   │   ├── evaluator.py        # Batch evaluation benchmark engine (ASR, DRR, Acc)
│   │   ├── sample_data.py      # Benchmark samples & taxonomy presets
│   │   └── schemas.py          # Pydantic request/response schemas
│   └── tests/
│       ├── conftest.py         # Test configuration & path setup
│       ├── test_clip_baseline.py # CLIP zero-shot & normalization tests
│       ├── test_pgd_attack.py  # PGD perturbation bounds & gradient tests
│       ├── test_scc_defense.py # SCC defense & TTC warm-up tests
│       └── test_api.py         # End-to-end API pipeline tests
├── frontend/
│   ├── index.html              # HTML shell loading React 18, Tailwind CSS, styles
│   ├── app.js                  # Modular React 18 application with comparison lab & tabs
│   ├── styles.css              # Custom themes, color variables, layout
│   └── serve.mjs               # Lightweight Node.js HTTP static server (:4173)
├── ARCHITECTURE.md             # Formal mathematical formulation & architecture spec
├── package.json                # Project scripts
├── start.ps1                   # One-click startup script (Windows PowerShell)
└── README.md                   # Complete documentation
```

---

## Installation & Setup

### Prerequisites
- Python 3.10+ with PyTorch 2.2+
- Node.js 18+ (for frontend static server)

### 1. Install Backend Dependencies
```bash
pip install -r backend/requirements.txt
```

### 2. Run Backend Unit Tests
```bash
python -m pytest backend/tests -v
```
All 12 unit tests verify:
- Frozen CLIP ViT-B/32 parameters and differentiable pixel encoding.
- PGD perturbation bounds $\|\delta\|_\infty \le \epsilon$ and pixel clamping to $[0, 1]$.
- TTC feature deviation warm-up and step-weighted fusion.
- Temperature sharpening, soft prototype construction, and hard-negative repulsion.
- Full end-to-end pipeline execution.

### 3. Start the Application
You can launch both the backend and frontend with one command:

**Using PowerShell:**
```powershell
./start.ps1
```

**Or manually in two terminals:**
```bash
# Terminal 1: Start FastAPI backend (:8000)
python -m uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000 --reload

# Terminal 2: Start Frontend interface (:4173)
node frontend/serve.mjs
```

Open your browser to: **`http://127.0.0.1:4173`**

---

## Reference Hyperparameters (ICML 2026 Paper)

| Component | Parameter | Default Value | Description |
|---|---|---|---|
| **Backbone** | Model | `CLIP ViT-B/32` | Frozen vision-language foundation model |
| **Attack** | $\epsilon_a$ | $4/255$ | Adversarial perturbation budget (or $1/255$) |
| **Attack** | Steps | $10$ | Iterations of PGD ascent |
| **Attack** | Random Init | `True` | Uniform initialization within $L_\infty$ ball |
| **Defense** | $\epsilon_d$ | $4/255$ | Corrective counterattack budget |
| **Defense** | $\alpha_d$ | $1/255$ | Defense step size |
| **Defense** | $S$ | $5$ | Number of SCC optimization steps |
| **Defense** | $S_{warm}$ | $5$ | Short TTC warm-up steps ($\mathcal{L}_{drift}$ only) |
| **Defense** | $V$ | $2$ | Number of augmented views (horizontal flip + noise) |
| **Defense** | $\sigma$ | $6/255$ | View Gaussian noise scale |
| **Defense** | $\lambda_{cm}$ | $4.0$ | Cross-modal semantic consistency weight |
| **Defense** | $T$ | $0.5$ | Soft pseudo-label sharpening temperature |

---

## API Reference

### `GET /api/health`
Returns backend health status, PyTorch version, active compute device (`cuda` or `cpu`), and model metadata.

### `GET /api/samples`
Returns the catalog of benchmark images (Dog, Wolf, Airplane, Automobile, Cat, Ship) and taxonomy presets (`cifar10`, `animals`, `vehicles`).

### `POST /api/classify`
Runs real zero-shot CLIP classification on a clean image:
```json
{
  "sample_id": "dog",
  "candidate_labels": ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
  "ground_truth": "dog"
}
```

### `POST /api/attack`
Executes untargeted white-box PGD attack maximizing cross-entropy against ground truth. Returns the adversarial image, amplified perturbation map, top-5 predictions, semantic margin, and $L_\infty / L_2$ metrics.

### `POST /api/defend`
Executes Self-Calibrated Consistency (SCC) defense on an attacked image.

### `POST /api/pipeline`
Executes the full end-to-end research comparison pipeline (**Clean $\to$ PGD $\to$ SCC**) in one atomic request. Returns side-by-side stage results, margins, and recovery diagnostics.

### `POST /api/evaluate/batch`
Executes a real batch evaluation over benchmark dataset samples. Reports Clean Accuracy, Robust Accuracy, Defended Accuracy, Attack Success Rate (ASR), and Defense Recovery Rate (DRR).
