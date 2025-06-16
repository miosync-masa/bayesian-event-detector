#bayesian-event-detector
A minimal sample for detecting time-series jump events using Bayesian inference

🚀 Concept / コンセプト
This repository demonstrates a fundamental paradigm shift in time-series analysis:
Instead of forcing all data to fit a single smooth law, our model explicitly separates "smooth trend" and "jump (event)" states, expressing reality as a mixture of processes.
Each parameter has a clear, human-interpretable meaning—allowing users not only to detect when and where an event occurred, but also why it occurred and with what certainty.
The innovation lies not in code complexity, but in the model's transparent structure and explanatory power.

このリポジトリは、時系列データ解析の「パラダイム転換」を体現しています。
すべてのデータを単一の法則で説明するのではなく、「滑らかなトレンド」と「ジャンプ（イベント）」という異なる状態の混合として現実世界を捉えます。
各パラメータは人間にとって直感的な意味を持ち、「いつ・どこでイベントが起きたか」だけでなく、「なぜそれが起きたのか」「その確信度」まで推論できます。
革新性は複雑なコードにではなく、「説明可能性」とシンプルな構造にあります。

Overview
This repository provides a minimal example for automatically detecting “jump (spike) events” in time-series data using Bayesian inference.
It includes dummy data generation, PyMC modeling, and optional result visualization—all in one script.

Usage
Install required packages:
pip install -r requirements.txt
Run the sample code:

python event_jump_detector.py

(Optional)
Uncomment the visualization lines in the script to plot the results.

File Description
event_jump_detector.py ... Main sample code

requirements.txt ... List of required Python packages

README.md ... This description

License
MIT License
