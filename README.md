# bayesian-event-detector
A minimal sample for detecting time-series jump events using Bayesian inference

Bayesian Time-Series Jump Event Detector
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

# ベイズ時系列ジャンプイベント検出器 / Bayesian Time-Series Jump Event Detector

## 概要 / Overview

**日本語:**  
このリポジトリは、ベイズ推論で時系列データの中の「ジャンプ（スパイク）イベント」を自動検出する最小サンプルです。  
ダミーデータ・PyMCモデル・可視化まで一括で動かせます。

**English:**  
This repository provides a minimal sample for detecting “jump/spike events” in time-series data using Bayesian inference.  
Generate dummy data, build the PyMC model, and visualize results easily.

## 使い方 / Usage

1. 必要なライブラリをインストール  
   `pip install -r requirements.txt`
2. 実行  
   `python event_jump_detector.py`
3. （必要なら）可視化のコメントを外してグラフを確認

## ファイル説明 / File Description

- `event_jump_detector.py` ... メインのサンプルコード
- `requirements.txt` ... 必要なパッケージ一覧
- `README.md` ... 本説明

## ライセンス / License

MIT License

