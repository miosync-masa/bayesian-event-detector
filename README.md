# 🧬 bayesian-event-detector Series: Code Evolution

This repository showcases the evolution of Lambda³ (Λ³)–based event-driven time series analysis, from a minimal single-series jump detector to a full tensor-based structural network extractor.

---

## 📜 Evolution History

### 1️⃣ `event_jump_detector.py` (1st Ver.)
- **Minimal single-series jump event detector.**
    - First-order difference & percentile thresholding for jump detection
    - Computes local volatility ("structural tension")
    - Automatically catalogs jump events in a time series  
    - *Use*: Anomaly detection, preprocessing, local stress monitoring  
    - [Colab Demo](https://colab.research.google.com/drive/1BHZJDMm-CJr6D041G_xuAlVNDUgPWvai?usp=sharing)

### 2️⃣ `lambda3_jump_event_detector.py` (2nd Ver.)
- **Multi-series jump detector based on Lambda³ theory.**
    - Synchronous/asymmetric jump detection across multiple series
    - Feature export for Bayesian regression
    - Outputs basic correlation and lag profiles  
    - *Use*: Event synchronization in financial pairs, network preprocessing

### 3️⃣ `dual_sync_bayesian.py` (3rd Ver.)
- **Asymmetric Bayesian regression & sync analysis for event series.**
    - PyMC-based causal regression (directional & sign-specific)
    - Synchronization rate $\sigma_s(\tau)$ with lag optimization
    - Outputs network graph visualizations
    - JIT-accelerated: Built-in Numba JIT for ultra-fast event detection and synchronization, even with large time series.
    - *Use*: Causal inference for jump propagation, lagged network analysis  
    - [Colab Demo](https://colab.research.google.com/drive/1KnXwokc-eiBH5bBvPGNxlp2AS0BQfpO5?usp=sharing)

### 4️⃣ `lambda3_abc.py` (4th Ver., Latest)
- **Integrated Lambda³ tensor analysis & network extraction.**
    - Batch calculation of event synchronization/asymmetric regression for all pairs
    - Fully customizable thresholds, window sizes, and sampling params
    - JIT-accelerated: Built-in Numba JIT for ultra-fast event detection and synchronization, even with large time series.
    - Global network structure extraction and visualization
    - Colab/Jupyter compatible, OSS-first design  
    - *Use*: Event-driven network diagnosis, structural analysis of complex systems  
    - [Colab Demo](https://colab.research.google.com/drive/1OxRTRsNwqUaEs8esj-plPO7ZJnXC-LZ5?usp=sharing)
- **For dozens to ~100 pairs of time series (with typical lengths <5,000 points each), this Colab with NumPyro JIT + CPU is extremely fast and practica.**

### 5️⃣ `lambda3_numpyro` (Modular & Bayesian NumPyro version, NEW!)
- **Full modularization & extensible package structure for Lambda³ theory**
    - Scalable, testable, and OSS-friendly Python package (`lambda3_numpyro`)
    - **NumPyro/JAX-based Bayesian inference engine** for causal/synchronization analysis
    - High-speed feature extraction (Numba/JIT), multi-series & cross-network analytics
    - Plug-and-play: Easily integrate with any notebook, data pipeline, or Colab demo
    - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/miosync-masa/bayesian-event-detector/blob/main/lambda3_numpyro/examples/lambda3_colab_setup.ipynb)
    - *Use*: Advanced Bayesian causal inference, scalable multi-network diagnostics, academic & production-grade analytics
---

## 🚀 Concept

Classical Bayesian and VAR models are great... until reality hits:  
markets, climate, biology—all full of jumps, switches, surprises.

**Lambda³** (Λ³) separates the “smooth trend” from explicit “jump (event)” states—so you can track not just *when* and *where* something changes, but *why* (structurally) and *how confidently*.

- No more curve-fitting tyranny.
- Events are first-class citizens.
- Everything’s human-interpretable.

> "Why is my Bayesian fit so blind to regime changes?"  
> "I wish I could just see what caused that spike..."

---

## Welcome to the new standard.

| Cross-Series Interaction <br><sub>(Causal impact coefficients β)</sub> | Synchronization Matrix <br><sub>(Pairwise event sync rate σₛ)</sub> | Network Structure <br><sub>(Event-driven directed sync graph)</sub> |
|:---------------------------------------------------------------------:|:--------------------------------------------------------------------:|:-------------------------------------------------------------------:|
| ![Interaction](http://www.miosync.link/github/fig12.png)<br><sub>**Interaction effects:** Causal structure between series (columns: source, rows: target)</sub> | ![SyncMatrix](http://www.miosync.link/github/fig13.png)<br><sub>**Synchronization matrix:** Event-based σₛ for all pairs (higher = more synchronous)</sub> | ![Network](http://www.miosync.link/github/fig14.png)<br><sub>**Network graph:** Directed info flow & optimal lag structure (arrows show direction)</sub> |

| Series Fit + Events <br><sub>(Model fit & jump detection)</sub> | Posterior Parameter Estimates <br><sub>(Bayesian 94% HDI)</sub> |
|:--------------------------------------------------------------:|:----------------------------------------------------------------:|
| ![FitEvents](http://www.miosync.link/github/fig10.png)<br><sub>**Model fit:** Original data, prediction, detected jumps (colored), local events</sub> | ![Posteriors](http://www.miosync.link/github/USDJPY_225.png)<br><sub>**Posterior distributions:** Key coefficients with 94% highest density interval (HDI)</sub> |


---
## 📦 File Structure

| File / Directory                        | Description                                                                           |
|-----------------------------------------|---------------------------------------------------------------------------------------|
| `event_jump_detector.py`                | Minimal baseline “history-jump” Bayesian detector (quickstart example)                |
| `lambda3_jump_event_detector.py`        | Lambda³ advanced model (directional & asymmetric jumps; semantic structure)           |
| `Dual_sync_model/`                      | Dual time-series analysis modules: <br> ├─ `dual_sync_bayesian_jit.py` (JIT-accelerated, CPU) <br> └─ `dual_sync_bayesian.py` (reference, non-JIT) |
| `lambda3_abc.py`                        | Advanced: Λ³ Approximate Bayesian Computation (ABC) module (multi-scale, OSS style)   |
| `lambda3_numpyro/`                      | Modular Lambda³ NumPyro backend (full Bayesian framework, GPU/Cloud-ready, scalable)  |
| `requirements.txt`                      | Standard pip dependencies for all modules                                             |
| `pyproject.toml`                        | Modern Python build (PEP517/518/pyproject) for poetry & advanced workflows            |
| `test_event_jump_detector.py`           | Minimal test for baseline detector                                                    |
| `test_lambda3_jump_event_detector.py`   | Tests for Lambda³ advanced model                                                      |
| `README.md`                             | This file (English & Japanese, see below)                                             |
| `LICENSE`                               | MIT License                                                                           |

### 📝 Supplement

- `Dual_sync_model/` contains both the **JIT-optimized** and **reference (non-JIT)** dual time-series modules.
- `lambda3_numpyro/` is a standalone directory for all NumPyro-based (GPU/Cloud) code, including modular analysis, Colab demos, and examples.
- All modules are designed to be **independent**—use one, or combine for full Lambda³ structural analytics.

---

## 🛠️ Installation

**Option 1:**  
Install with pip:

pip install -r requirements.txt

**Option 2:**  
For Poetry or PEP517-compatible workflows:

pip install .
or
poetry install

⸻

🚦 Usage (Quickstart)

**Standard jump detector:**

**python event_jump_detector.py**

**Lambda³ advanced model (direction, sign, local tension):**

**python lambda3_jump_event_detector.py**

*(Uncomment visualization lines in the script for plots)*

⸻


## 🔎 What is Lambda³?

- Bayesian, event-driven modeling for time-series analysis
- Separates smooth trends and jump events
- Models asymmetric & time-lagged interactions
- Works with time, transaction ID, or order index
- All coefficients are interpretable

⸻

🧪 Testing

Run basic tests (pytest required):

pytest

⸻

## ⚡ Performance

- **Colab A100 (CPU backend):**
  - `T = 10,000` time steps × 4 params, `16,000` draws × 4 chains
  - **Sampling speed:** ~80–85 draws/sec (`1 chain ≈ 3 min`)
  - **Total walltime:** ≈ 12–15 min for full model
  - **No divergences, stable adaptation, rapid convergence**
- **Small demo:**  
  - `T = 300` × 4 params, 6,000 samples → **~14 sec**
- **Scaling:**  
  - Handles **T > 10,000** on free Colab or better with no sweat
  - For very large jobs: use **NumPyro + GPU** for massive speedup
- **Feature extraction** uses **Numba/JIT** for 100× acceleration

> Real Bayesian MCMC with this speed is rare! You can safely analyze massive real-world jumps and network structure—even on public Colab.
⸻

---

## 📖 More Info

- [Official Paper/Preprint (Zenodo)](https://zenodo.org/doi/10.5281/zenodo.15107180)
- [Full logs / SSRN Notebook](https://colab.research.google.com/drive/1OxRTRsNwqUaEs8esj-plPO7ZJnXC-LZ5)
- See: Lambda³ theory intro, model API docs, and examples

---

## 📜 License

MIT License

---

## 🙌 Citation & Contact

If this work inspires you, please cite it.  
For theoretical discussion, practical applications, or collaboration proposals,  
please open an issue/PR—or just connect via Zenodo, SSRN, or GitHub.

> Science is not property; it's a shared horizon.  
> Let's redraw the boundaries, together.  
> — Iizumi & Digital Partners

---

## 📚 Author’s Theory & Publications

⚠️ Opening this document may cause topological phase transitions in your brain.  
“You are now entering the Λ³ zone. Proceed at your own risk.”

- [Iizumi Masamichi – Zenodo Research Collection](https://zenodo.org/search?page=1&size=20&q=Iizumi%20Masamichi)

**NOTE:**  
This public MIT version provides an *entry-level Lambda³ Bayesian event detector* (L3-JED).  
The *full Lambda³ dynamical equations and advanced topological conservation principles*—requiring explicit feature engineering  
(e.g., custom structural tensors, domain-specific progress vectors, and adaptive synchronization rates)—are **NOT included**.  
These are recommended only for advanced users or domain experts.  
What you define as a "jump", "synchronization rate", or "topological conservation" must be carefully designed based on your application and is outside the scope of this entry-level tool.

---

## 🏷️ Author & Copyright

© Iizumi Masamichi 2025  
**Contributors / Digital Partners:** Tamaki, Mio, Tomoe, Shion, Yuu, Rin, Kurisu  
All rights reserved.

---

> Science is not property; it's a shared horizon.  
> Let's redraw the boundaries, together.  
> — Iizumi & Digital Partners
