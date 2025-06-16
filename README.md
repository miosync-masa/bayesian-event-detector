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
