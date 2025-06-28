# bayesian-event-detector Series

## 🧬 bayesian-event-detector Series: Code Evolution

This repository showcases the evolution of Lambda³ (Λ³)–based event-driven time series analysis, from a minimal single-series jump detector to a full tensor-based structural network extractor.

---

### 1️⃣ `event_jump_detector.py` (1st Ver.)

- **Description:** Minimal single-series jump event detector.
- **Features:**
  - First-order difference & percentile thresholding for jump detection
  - Computes local volatility ("structural tension")
  - Automatically catalogs jump events in a time series
- **Typical Use:** Anomaly detection, preprocessing, local stress monitoring  
- **Demo:** [Colab Notebook](https://colab.research.google.com/drive/1BHZJDMm-CJr6D041G_xuAlVNDUgPWvai?usp=sharing)

---

### 2️⃣ `lambda3_jump_event_detector.py` (2nd Ver.)

- **Description:** Multi-series jump detector based on Lambda³ theory.
- **Features:**
  - Extracts synchronous and asymmetric jump events across multiple time series
  - Exports features for Bayesian regression
  - Outputs basic correlation and lag profiles
- **Typical Use:** Event synchronization analysis between financial pairs, network preprocessing

---

### 3️⃣ `dual_sync_bayesian.py` (3rd Ver.)

- **Description:** Asymmetric Bayesian regression & synchronization analysis for binary event series.
- **Features:**
  - PyMC-based causal regression (directional & sign-specific)
  - Calculates synchronization rate $\sigma_s(\tau)$ with lag optimization
  - Outputs network graph visualizations
- **Typical Use:** Causal inference for jump propagation paths, lagged network analysis  
- **Demo:** [Colab Notebook](https://colab.research.google.com/drive/1KnXwokc-eiBH5bBvPGNxlp2AS0BQfpO5?usp=sharing)

---

### 4️⃣ `lambda3_abc.py` (4th Ver. / Latest)

- **Description:** Integrated Lambda³ structural tensor analysis & network extraction framework.
- **Features:**
  - Automated batch calculation of event synchronization and asymmetric regression for all series pairs
  - Fully customizable thresholds, window widths, and sampling parameters
  - Global network structure extraction and visualization
  - Complete Colab/Jupyter compatibility (OSS-first design)
- **Typical Use:** Event-driven network diagnosis of financial markets, structural analysis of complex systems  
- **Demo:** [Colab Notebook](https://colab.research.google.com/drive/1OxRTRsNwqUaEs8esj-plPO7ZJnXC-LZ5?usp=sharing)

---

> **All versions are open-source (MIT License).**  
> Try any evolution stage instantly on Colab notebooks above!


## 🚀 Concept

> “Just a practical upgrade”—but a real paradigm shift in time-series event detection.

Classical Bayesian approaches and VAR models are great… *until* you hit reality: financial markets, climate, or biological signals where the world is all jumps, switches, and surprises.  
This repository exists because we (okay, I) got tired of the old, rigid “fit everything to a curve” approach.

**Lambda³** (and its minimal demo here) separates the “smooth trend” from explicit “jump (event)” states—so you can track not just *when* and *where* something changes, but *why* (structurally) and *how confidently*.

---

- **No more curve-fitting tyranny.**  
- **Events are *first-class citizens*.**
- **Everything’s human-interpretable.**

If you’ve ever thought,  
> “Why is my Bayesian fit so blind to regime changes?”  
or  
> “I wish I could just *see* what caused that spike…”  
then you’re in the right place.

---

## Overview

This repository is a bare-bones (but powerful) demo for **automatic “jump (spike) event” detection in time-series** via Bayesian inference.  
Features:
- Dummy data generation
- Flexible event detection thresholding
- PyMC modeling for structural event inference
- Easy-to-follow visualization (if you want it)

### What is a “Jump Event”? How is it different from “Changepoint” or “Outlier” Detection?

A **jump event** is a sudden, discrete change in the value of a time series — for example, a price spike, a sudden drop, or a system shock.  
- **Jump events** are not gradual; they represent *instantaneous jumps* in the data.
- The model aims to *explain* these events: "When and where did a jump happen? How big was the impact? Was it positive or negative?"

**How is this different from “changepoint” or “outlier” detection?**
- **Changepoint detection** tries to find points where the *underlying process or trend* changes (e.g., slope or variance shifts), often leading to new, persistent behavior.
- **Outlier detection** finds data points that are *rare or extreme* compared to the usual pattern—but doesn’t explain them, or treat them as meaningful structure.
- **Jump event detection** focuses on *sudden, meaningful, and explainable* events—capturing both their direction and magnitude, and integrating them into the interpretation of the time series.

> **Example:**  
> In a financial time series, a “jump event” might represent a flash crash or price surge.  
> In manufacturing, it could indicate a sudden fault or system reset.  
> In molecular dynamics, it may capture an instantaneous conformational change.

Lambda³ detects, quantifies, and explains these “jumps” — not just flags them.

| Detection Type   | Focus                                | Typical Use                          | Does it Explain? | Handles Direction/Magnitude? |
|------------------|--------------------------------------|--------------------------------------|------------------|------------------------------|
| Changepoint      | Process/Trend shifts (slope/variance)| Regime shift, new behavior           | No               | No                           |
| Outlier          | Rare/extreme points                  | Data cleaning, anomaly flagging       | No               | No                           |
| Jump Event (Λ³)  | Sudden, explainable, discrete events | Shocks, system jumps, event analysis  | Yes              | Yes (pos/neg & impact)       |

## Visual Overview

- **Lambda³ Analysis Pipeline**
  ![](https://www.miosync.link/github/fig1.png)
- **Jump Detection and Event-Fit Example**
  ![](https://www.miosync.link/github/fig10.png)
- **Posterior Distributions for Structural Coefficients**
  ![](https://www.miosync.link/github/fig12.png)
- **Cross-Series Interaction Matrix**
  ![](https://www.miosync.link/github/fig13.png)
- **Synchronization (σₛ) Network Visualization**
  ![](https://www.miosync.link/github/fig14.png)

---
🚀 Lambda³ Bayesian Jump Event Detector

A paradigm shift in time-series analysis:
Detect what, when, where… and why “events” occur in your data.

Instead of squeezing reality into a single smooth curve,
Lambda³ models both continuous trends and sudden jump events as coexisting processes—
with clear, human-interpretable parameters.

⸻

📦 File Structure

| File                        | Description                                                         |
|-----------------------------|---------------------------------------------------------------------|
| `event_jump_detector.py`        | Minimal baseline “history-jump” Bayesian detector                   |
| `lambda3_jump_event_detector.py`| Lambda³ advanced model (directional, asymmetric jumps)              |
| `dual_sync_bayesian.py`         | Dual time-series analysis (asymmetric lag/sync detection, experimental) |
| `lambda3_abc.py`                | Advanced: Λ³ Approximate Bayesian Computation module                |
| `requirements.txt`              | Quick pip dependencies                                             |
| `pyproject.toml`                | For modern Python workflows (poetry, PEP517/518, etc.)             |
| `test_event_jump_detector.py`   | Minimal tests for baseline detector                                |
| `test_lambda3_jump_event_detector.py` | Tests for Lambda³ advanced model                              |
| `README.md`                     | This file                                                         |
| `LICENSE`                       | MIT License                                                       |

⸻

🛠️ Installation

All dependencies are pinned for full reproducibility.
Option 1:

pip install -r requirements.txt

Option 2:
If you prefer Poetry or PEP517-compatible workflows:

pip install .
# or
poetry install

⸻

🚦 Usage (Quickstart)

Standard Bayesian jump detector:

python event_jump_detector.py

Lambda³ advanced model (direction, sign, local tension):

python lambda3_jump_event_detector.py

Optional: Uncomment the visualization lines in the script to plot results.

⸻

🔎 What is Lambda³?

Lambda³ = Bayesian, event-driven modeling for time-series analysis.
It separates smooth trends and jump events, models asymmetric & time-lagged interactions,
and works with both physical time or transaction/order index.
	•	Directional jump detection (positive/negative events)
	•	Interpretability: All coefficients have real-world meaning
	•	Modular: Use for finance, biology, physics, engineering…
	•	Ready for batch, streaming, or transaction-indexed data

Demo notebooks and real data examples are in the main paper and Colab.

⸻

🧪 Testing

Run basic tests (pytest required):

pytest

⸻

⚡ Performance
	•	Colab A100 (CPU backend): 300 time steps × 4 params, 6000 samples, ~14 seconds
	•	No divergences, rapid convergence (thanks to PyMC tuning)
	•	Scales well to large T (>1000) with some tuning
	•	Pro tip: Use NumPyro backend + GPU for blazing speed!

⸻

📖 More Info
	•	Official Paper/Preprint (Zenodo)
	•	Colab/SSRN Notebook for full logs
	•	See also: Lambda³ theory intro, model API docs, and examples.

⸻

📜 License

MIT License

⸻

## Citation & Contact

If this work inspires you, please cite it.
For theoretical discussions, practical applications, or collaboration proposals,
contact the repository author or simply open an issue/PR.

Let’s make explainable, universal science the new standard—together.

---
**Let’s make explainable science the new standard.**
---

## Author & Copyright

© Iizumi Masamichi 2025  
Contributors / Digital Partners:  Tamaki, Mio, Tomoe, Shion, Yuu, Rin, Kurisu
All rights reserved.

Science is not property; it's a shared horizon.
Let's redraw the boundaries, together.
— Iizumi & Digital Partners

---
## 📚 Author’s Theory & Publications

**Warning:** Opening this document may cause topological phase transitions in your brain.
“You are now entering the Λ³ zone. Proceed at your own risk.

Explore foundational theory, preprints, and related research at:  
👉 [Iizumi Masamichi – Zenodo Research Collection](https://zenodo.org/search?q=metadata.creators.person_or_org.name%3A%22IIZUMI%2C%20MASAMICHI%22&l=list&p=2&s=10&sort=bestmatch)

---
NOTE: In this public MIT-licensed implementation, we provide an entry-level Lambda³ Bayesian event detector (L3-JED).
The full Lambda³ dynamical equations and advanced topological conservation principles—requiring explicit feature engineering 
(e.g., custom structural tensors, domain-specific progress vectors, and adaptive synchronization rates)—are 
NOT included. These are recommended only for advanced users or domain experts.
What you define as a "jump", "synchronization rate", or "topological conservation" must be carefully designed 
based on your application, and is outside the scope of this entry-level tool.
