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

Welcome to the new standard.

---

## 📦 File Structure

| File                           | Description                                                        |
|--------------------------------|--------------------------------------------------------------------|
| event_jump_detector.py         | Minimal baseline “history-jump” Bayesian detector                  |
| lambda3_jump_event_detector.py | Lambda³ advanced model (directional, asymmetric jumps)             |
| dual_sync_bayesian.py          | Dual time-series analysis (asymmetric lag/sync detection, experimental) |
| lambda3_abc.py                 | Advanced: Λ³ Approximate Bayesian Computation module               |
| requirements.txt               | Quick pip dependencies                                            |
| pyproject.toml                 | For modern Python workflows (poetry, PEP517/518, etc.)            |
| test_event_jump_detector.py    | Minimal tests for baseline detector                                |
| test_lambda3_jump_event_detector.py | Tests for Lambda³ advanced model                         |
| README.md                      | This file                                                         |
| LICENSE                        | MIT License                                                       |

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

- Colab A100 (CPU): 300 time steps × 4 params, 6000 samples, ~14 sec
- No divergences, rapid convergence (PyMC tuning)
- Scales to T > 1000 with more tuning
- NumPyro + GPU = even faster!

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
