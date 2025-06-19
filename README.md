# bayesian-event-detector

A minimal sample for detecting time-series jump events using Bayesian inference.
---

## 🚀 NEW: Dual EYE Mode for Jump Event Detection

Lambda³ now supports **Dual EYE Mode**, an advanced feature that enables simultaneous detection of both **global (macro) structural changes** and **local (micro) contextual anomalies** in time series data.

### What is Dual EYE Mode?

- **Global Eye**: Detects history-wide, statistically significant jump events ("phase shifts") using global percentiles (ΔΛC). Perfect for capturing major regime changes, such as financial crises or structural breaks in physical systems.
- **Local Eye**: Detects context-sensitive, locally significant jumps using moving-window normalized scores. Ideal for finding subtle "precursors," micro-anomalies, or localized events that may not stand out globally but are surprising in their immediate context.

**Visualizations now highlight both types of events:**
- Blue/Orange markers: Global jumps (positive/negative)
- Magenta markers: Local jumps (contextual anomalies)

### Why is this important?

By providing both "forest-level" (macro) and "tree-level" (micro) perspectives,  
Dual EYE Mode allows users to:
- Spot **major disruptions** and **minor anomalies** in a single unified framework.
- Gain deeper insights into the interplay between **system-wide phase transitions** and **localized precursors** or warnings.
- Apply Lambda³ to real-world problems—ranging from finance and engineering to biology and geophysics—where both scales of anomaly matter.

---

> **Lambda³ is evolving from a single-eye (mono) anomaly detector  
> to a true "stereo vision" AI—capable of seeing the whole and the details together.**

Check out the new examples and docs to see Dual EYE Mode in action!


**Fit:**

![Lambda3 fit](http://miosync.link/github/download-12.png)

**Posterior:**

![Lambda3 posterior](http://miosync.link/github/download-11.png)

---

## 🚀 Concept

This repository demonstrates a paradigm shift in time-series analysis:

> Instead of forcing all data to fit a single smooth law, our model explicitly separates "smooth trend" and "jump (event)" states, expressing reality as a *mixture of processes*.
> Each parameter has a clear, human-interpretable meaning—allowing users not only to detect *when* and *where* an event occurred, but also *why* it occurred and with what certainty.

---

## Overview

This repository provides a minimal example for automatically detecting “jump (spike) events” in time-series data using Bayesian inference.
It includes dummy data generation, PyMC modeling, and optional result visualization—all in one script.

---

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

---

## Dependencies

All dependencies are pinned for reproducibility.  
You can use either `requirements.txt` (for quick pip install) or `pyproject.toml` (for modern Python workflows).

**Option 1:**  
pip install -r requirements.txt

**Option 2:**  
If you use poetry or pip with pyproject.toml:
pip install .or poetry install .

## Usage
**Run the sample code:**
python event_jump_detector.py        # Standard Bayesian history-jump detector
python lambda3_jump_event_detector.py # Lambda³ version (directional, more advanced)

. **(Optional)**
Uncomment the visualization lines in the script to plot the results.

---
## File Description
* `event_jump_detector.py` ... Minimal baseline (history-jump detector)
* `lambda3_jump_event_detector.py`... Lambda³ advanced model (directional jumps etc.)
* `pyproject.toml`... Modern dependency management
* `requirements.txt` ... List of required Python packages
* `README.md` ... This description

---

## Example Output

### Bayesian Decomposition (Standard Bayesian history-jump detector)

![Bayesian trend decomposition](http://miosync.link/github/sample1.png)

**Posterior for event probability:**

![Posterior event probability](http://miosync.link/github/sample2.png)

**Posterior for trend and event parameters:**

![Posterior for parameters](http://miosync.link/github/sample3.png)


## Advanced Usage: Visualization Example

The following script shows how to visualize the decomposition of your time series into the inferred trend and detected event points (where the probability of an event exceeds 50%).

```python
import matplotlib.pyplot as plt
import numpy as np

# Obtain summary statistics from the trace
summary = az.summary(trace, var_names=['trend', 'event_indicator', 'jump_effect'])

# Extract posterior means for trend and event indicator
mean_trend = summary['mean'][summary.index.str.startswith('trend')]
event_prob = summary['mean'][summary.index.str.startswith('event_indicator')]

fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(data, 'o', color='gray', markersize=4, alpha=0.6, label='Original Data')
ax.plot(mean_trend, color='C0', lw=2, label='Inferred Trend')

# Highlight detected events (posterior probability > 0.5)
event_detected_indices = np.where(event_prob > 0.5)[0]
for idx in event_detected_indices:
    ax.axvline(x=idx, color='C1', linestyle='--', alpha=0.7, label=f'Event Detected (t={idx})')

ax.set_title('Decomposition of Time Series into Trend and Events', fontsize=16)
ax.set_xlabel('Time Step', fontsize=12)
ax.set_ylabel('Value', fontsize=12)

# Remove duplicate labels
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), fontsize=12)

plt.grid(axis='y', linestyle=':', alpha=0.7)
plt.show()

print("\nDetected event time points:")
print(event_detected_indices)
```

---

## Lambda³ Model: Paradigm Shift for Transactional Time-Series

**File:** `lambda3_jump_event_detector.py`

A next-generation Bayesian regression model implementing Lambda³ theory (Λ³), focusing on “jump events,” trends, and local volatility. This approach uses **transaction index** rather than physical time, and separates jump directions.

### Key features:

* Directional jump detection (positive/negative)
* Full Bayesian coefficient estimation for interpretability
* Transaction-index based progress (can be time, transaction ID, or order)
* Plug-and-play for science, finance, biology, and engineering

## 🧪 Testing

Basic tests for the Lambda³ jump event detector are included in `test_lambda3_event_jump_detector.py`.

To run all tests:
pip install pytest

### Lambda³ version (directional, more advanced) Example Output 

**Fit:**

![Lambda3 fit](http://miosync.link/github/Lambda_sample_fit.png)

**Posterior:**

![Lambda3 posterior](http://miosync.link/github/Lambda_sample_posterior.png)

---

## License

MIT License


---

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
