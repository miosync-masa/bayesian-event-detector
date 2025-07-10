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
    - **NumPyro/JAX-based Bayesian inference engine** for causal and synchronization analysis
    - High-speed feature extraction (Numba/JIT), multi-series & cross-network analytics
    - **Built-in structural regime detection**:  
      Cluster and analyze time series by *structural regime* (season, market phase, physiological state, etc.)—not just by time!
    - Plug-and-play: Easily integrate with any notebook, data pipeline, or Colab demo
    - [Colab Demo](https://colab.research.google.com/drive/1Crygnt8hQsGlPO0dc2uTtVQVe4tCFERW?usp=sharing)
    - *Use*: Advanced Bayesian causal inference, regime-aware analytics, scalable multi-network diagnostics, academic & production-grade applications

---
# 🚀 Concept

Classical Bayesian and VAR models are great... until reality hits:  
markets, climate, biology—all full of jumps, switches, surprises.

**Lambda³** (Λ³) separates the “smooth trend” from explicit “jump (event)” states—so you can track not just *when* and *where* something changes, but *why* (structurally) and *how confidently*.

- No more curve-fitting tyranny.
- Events are first-class citizens.
- Everything’s human-interpretable.

> "Why is my Bayesian fit so blind to regime changes?"  
> "I wish I could just see what caused that spike..."

### What is a "jump-event"?

A **jump-event** is an abrupt, discrete structural change in a time series — a sudden “jump” rather than a slow drift or regular fluctuation. Unlike classical *change-point detection*, which finds broader regime shifts, or *outlier detection*, which flags rare extreme values, a jump-event specifically marks a moment where the underlying process rapidly changes state (for example: price shock, heart rhythm flip, regime switch).  
Jump-events capture both the direction (positive/negative) and magnitude of these structural pulses, enabling you to analyze how, when, and why critical transitions occur — not just whether the mean or variance has shifted.

> In Lambda³, jump-events are treated as first-class structural events — the core “particles” of change, not just noise or anomalies.

| Method                   | What it detects                                   | Typical Output          | Example Use Case                | Limitation                          | Lambda³ Usage      |
|--------------------------|---------------------------------------------------|------------------------|----------------------------------|--------------------------------------|--------------------|
| **Jump-event detection** | *Abrupt, local, signed* structural changes        | List of jump events<br>(location, sign, magnitude) | Causal impact, shock propagation, structural analysis | May be “hidden” if only looking at means/variance | **Core primitive (“event-pulse”)** |
| **Change-point detection** | Broad regime shifts or statistical changes<br>(mean/variance/trend) | Change-point indices<br>(segment boundaries) | Regime segmentation, volatility regime, drift | Misses small, rapid events; only coarse boundaries | Used for regime annotation |
| **Outlier detection**    | Rare, extreme values<br>(anomalies, noise, errors) | Outlier indices/flags  | Data cleaning, anomaly detection | Not always meaningful<br>for structure; may mix noise & real jumps | Used for data QC<br>(not structural) |

> **Tip:**  
> - Jump-events = Local, structural "pulses" that drive system evolution.  
> - Change-points = Big regime shifts (segments, plateaus).  
> - Outliers = Rare anomalies, usually noise or data error.  
>
> *Lambda³ makes jump-events the main unit of analysis: they are not “noise”—they ARE the change.*

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
| `1st_ver_event_jump_detector`           | Minimal baseline “history-jump” Bayesian detector (quickstart example)                |
| `2nd?ver_lambda3_jump_event_detector`   | Lambda³ advanced model (directional & asymmetric jumps; semantic structure)           |
| `3rd_ver_Dual_sync_model/`              | Dual time-series analysis modules: <br> ├─ `dual_sync_bayesian_jit` (JIT-accelerated, CPU) <br> └─ `dual_sync_bayesia` (reference, non-JIT) |
| `lambda3_abc.py`                        | Advanced: Λ³ Approximate Bayesian Computation (ABC) module (multi-scale, OSS style)   |
| `lambda3_numpyro/`                      | Modular Lambda³ NumPyro backend (full Bayesian framework, GPU/Cloud-ready, scalable)  |
| `requirements.txt`                      | Standard pip dependencies for all modules                                             |
| `pyproject.toml`                        | Modern Python build (PEP517/518/pyproject) for poetry & advanced workflows            |
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


# 🦋 Lambda³ Butterfly Effect Analysis on GCP Cloud Run

## Why Cloud Run is the Ultimate Choice

### The Essence of the Butterfly Effect
*"A butterfly flapping its wings in Brazil can cause a tornado in Texas..."*  
→ In Lambda³ theory, even the tiniest structural change (ΔΛC) can propagate into global-scale phenomena.

### Lambda³ × Cloud Run: A Perfect Match
- **Auto-scaling:** From one "butterfly flap" (single API call) to a thousand tornadoes (1000+ concurrent requests)
- **True serverless:** Pay only for what you use (analyzing one butterfly event may cost less than $0.001)
- **Global reach:** Monitor "flaps" from around the world, simultaneously

## 🚀 Deploy in 5 Minutes

### 1. Minimal Configuration (single-file example)

```python
# main.py
from flask import Flask, request, jsonify
import numpy as np
from lambda3_abc import calc_lambda3_features_v2, L3Config

app = Flask(__name__)

@app.route('/butterfly', methods=['POST'])
def detect_butterfly():
    """Detect micro-structural changes (the 'flap')"""
    data = request.json

    # Lambda³ analysis
    features = calc_lambda3_features_v2(
        np.array(data['timeseries']),
        L3Config(delta_percentile=99.9)  # Detect only rarest events
    )

    # Butterfly effect score calculation
    butterfly_score = np.sum(features[0]) * features[2].max()  # jumps × max_tension

    return jsonify({
        'butterfly_score': float(butterfly_score),
        'will_cause_tornado': butterfly_score > 10.0,
        'affected_region': predict_impact_zone(features)  # you can define this
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```
---
## Simple Dockerfile

```Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD exec gunicorn --bind :8080 main:app
```
---

## Deploy Command
```
export PROJECT_ID=your-butterfly-project
export REGION=asia-northeast1  # e.g., Tokyo region

gcloud run deploy butterfly-detector \
  --source . \
  --region=$REGION \
  --platform=managed \
  --memory=1Gi \
  --cpu=1 \
  --max-instances=1000 \
  --allow-unauthenticated

# When complete, you'll get a public endpoint like:
# https://butterfly-detector-xxx-an.a.run.app
```

## Multi-region "Butterfly Network" Example

```butterfly-global.yaml
regions:
  - asia-northeast1     # Tokyo
  - us-central1         # Texas
  - europe-west1        # London
  - southamerica-east1  # Brazil

deploy_script: |
  for region in ${regions[@]}; do
    gcloud run deploy butterfly-detector \
      --region=$region \
      --image=gcr.io/$PROJECT_ID/butterfly:latest &
  done
```
## 🔥 Real-World Use Cases

- **Financial Markets:**  
  Detect the "butterfly trade" that triggers global crashes.

- **Social Networks:**  
  Analyze when a single tweet becomes a viral phenomenon.

- **Medical (EEG):**  
  Predict epileptic seizures from subtle brainwave anomalies.

- **Earth Monitoring:**  
  Real-time global anomaly detection for disaster prevention.

---

## 🚀 Why is this revolutionary?

- **No supercomputers needed:**  
  Cloud Run's auto-scaling with efficient Lambda³ means you can monitor Earth-scale complexity for the price of a coffee.

- **Real-time, global, event-driven analysis** for any spatio-temporal data.

- **Open science:**  
  Anyone can deploy and experiment in minutes.

---

> *"Let your butterflies fly! — With Lambda³ and Cloud Run, the tiniest change can now be detected, traced, and even predicted, at global scale, by anyone."*


> Science is not property; it's a shared horizon.  
> Let's redraw the boundaries, together.  
> — Iizumi & Digital Partners
