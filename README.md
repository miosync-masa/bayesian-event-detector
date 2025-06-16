# bayesian-event-detector

A minimal sample for detecting time-series jump events using Bayesian inference.

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

## Usage

1. **Install required packages:**

   ```bash
   pip install -r requirements.txt
   ```
2. **Run the sample code:**

   ```bash
   python event_jump_detector.py
   ```
3. **(Optional)**

   Uncomment the visualization lines in the script to plot the results.

---

## File Description

* `event_jump_detector.py` ... Main sample code
* `requirements.txt` ... List of required Python packages
* `README.md` ... This description

---

## License

MIT License

---
lambda3_jump_event_detector.py
Overview
lambda3_jump_event_detector.py is an open-source Python sample for explainable, Bayesian time-series modeling that detects and quantifies “jump events” (discontinuous transitions) using Lambda³ theory (Λ³).
Unlike traditional time-series models, it separates trend (smooth progression), jump events (discontinuous changes), and transactional progress (structural history), offering a fully transparent, interpretable framework for analyzing dynamic systems—from physics to biology and engineering.

Features
Bayesian regression with Lambda³-style features:

Directional jump event indicators (ΔΛC_pos, ΔΛC_neg)

Local volatility (ρT, "tension density")

Transactional index (structural progress, not physical time)

Full posterior inference for all coefficients—quantifies which factors matter, and how much

Automatic detection and visualization of jump events in synthetic or real-world time series

Reproducible, fully open-source, and easy to extend for your domain

How it works
Data generation:
Synthetic time series with trend, noise, and artificial jump events is created for demonstration.

Lambda³ feature extraction:

Calculates directional jump indicators (ΔΛC_pos, ΔΛC_neg) by comparing each point to its history

Computes local volatility (ρT) over a moving window

Uses a transactional index for progress (can be time, transaction ID, etc.)

Bayesian regression:

Fits a model using these features to explain the observed data

Estimates the effect size and credibility for each explanatory variable

Visualization:

Plots mean model prediction and detected events

Plots posterior distributions for all main coefficients (trend, jump events, volatility)


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

print("\\nDetected event time points:")
print(event_detected_indices)
