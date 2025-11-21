# Nebula-Real: Sandbox Trace Classifier, Zero-FP Calibration & Saliency-Guided Evasion Attack

[![Made with Python](https://img.shields.io/badge/Made%20with-Python-3670A0?logo=python&logoColor=white)](https://www.python.org/)
[![Trained using PyTorch](https://img.shields.io/badge/Trained%20using-PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Build-Active-blue)](https://github.com/Mozeel-V/nebula-real)


This repository contains the full implementation of the Nebula project. It is a practical malware trace classifier trained on real sandbox logs, calibrated for zero false positives, and evaluated against a strong gradient-based adversarial attack.  
The project integrates data normalization, vocabulary construction, transformer-based sequence modeling (CLS-pooled), cost-sensitive training, threshold search with consecutive-window rules, and a multi-step saliency-guided evasion attack.

---

## Key Features
- Real sandbox trace dataset (goodware + malware, balanced 2000-sample subset)
- Normalization pipeline (timestamp replacement, path canonicalization, argument simplification)
- Vocabulary building from normalized traces (vocab_n.json)
- CLS-pooled Transformer model
- Cost-sensitive training (penalize mistakes on goodware more heavily)
- Window-based evaluation (sliding windows over event streams)
- Zero-FP threshold & consecutive-window calibration
- Strong saliency-guided adversarial attack:
  - gradient-based saliency per token
  - iterative multi-step replacement
  - in-vocabulary semantic candidates
  - greedy probability-minimizing search
- Before/After comparison suite with full metrics & per-sample timelines
- Side-by-side class plots for malware & goodware

---

## Architecture Overview

### 1. Normalization & Tokenization
Real sandbox traces are long event lines such as:

`timestamp|...|origin|C:\pin\malware.exe|target|KernelBase.dll|function|CreateFileW|...`

We normalize high-variance elements (timestamps, PIDs, numeric args)  
and map tokens using a vocabulary generated from real traces.

**Key Scripts:**  
- [tokenizer.py](https://github.com/Mozeel-V/nebula-real/blob/main/new_src/tokenizer.py): normalization-aware token mapping  
- [normalize.py](https://github.com/Mozeel-V/nebula-real/blob/main/src/normalize.py): rule-based canonicalization  

---

### 2. CLS-Pooled Transformer Model
The classifier uses a compact Transformer encoder with CLS token pooling for global sequence classification.

**Implementation:**  
- [nebula_model.py](https://github.com/Mozeel-V/nebula-real/blob/main/new_src/nebula_model.py) — CLS-pooled architecture  
- 2 transformer layers, hidden dimension 192  
- positional encodings adapted to real trace length  

---

### 3. Cost-Sensitive Supervised Training
Goodware errors are heavily penalized to reduce false positives.  
Training is CPU-friendly and optimized for <25 minute runs.

**Implementation:**  
- [train_supervised.py](https://github.com/Mozeel-V/nebula-real/blob/main/new_src/train_supervised.py) 
- Weighted CrossEntropy (weight_goodware >> weight_malware)  
- Logging of accuracy, weighted loss, per-epoch checkpoints  

---

### 4. Window-Level Evaluation
Sliding windows extract fixed-size event chunks; each window is independently classified.

**Outputs:**  
- Per-window logits & probabilities  
- Per-sample class timelines  
- Window-level and sample-level confusion matrices  

**Implementation:**  
- [window_eval_plot.py](https://github.com/Mozeel-V/nebula-real/blob/main/new_src/window_eval_plot.py)  

---

### 5. Zero-FP Threshold Search (with Consecutive Windows)
To mirror real AV behavior ("any malicious window ⇒ block sample"), we compute:

- Window-level score distribution  
- OR-rule prediction  
- k-consecutive malicious windows  
- Zero-FP threshold (FP = 0 while maximizing TP)

**Implementation:**  
- [find_zero_fp.py](https://github.com/Mozeel-V/nebula-real/blob/main/new_src/find_zero_fp.py)  

---

### 6. Strong Saliency-Guided Evasion Attack
The attack identifies high-saliency positions via gradient backpropagation and performs greedy, in-vocabulary replacements that significantly reduce the malware score.

**Implementation:**  
- [saliency_attack_strong.py](https://github.com/Mozeel-V/nebula-real/blob/main/new_src/saliency_attack_strong.py)  
- [estimate_time.py](https://github.com/Mozeel-V/nebula-real/blob/main/new_src/estimate_time.py): runtime estimation for CPU constraints  

---

### 7. Before/After Comparison & Visualization
Generates full metrics and side-by-side class timelines per sample, split clearly into malware/goodware folders.

**Implementation:**  
- [compare.py](https://github.com/Mozeel-V/nebula-real/blob/main/new_src/compare.py)  

Produces directories:
- malware/before, malware/after, malware/side_by_side  
- benign/before, benign/after, benign/side_by_side  

---

## Running the Pipeline

### Train the Model

```bash
python new_src/train_supervised.py
--data data/dataset_small_2k_normalized.tsv
--vocab checkpoints/vocab_n.json
--out_dir checkpoints/run_weighted
--epochs 6 --batch_size 64 --lr 1e-4
--weight0 4.0 --weight1 1.0 --model_type nebula_cls
```

### Window-Level Evaluation (Before Attack)

```bash
python new_src/window_eval_plot.py
--data_file data/dataset_small_2k_normalized.tsv
--ckpt checkpoints/run_weighted/best.pt
--vocab checkpoints/vocab_n.json
--out_dir new_results/window_eval_before
```

### Run Strong Attack

```bash
python new_src/saliency_attack_strong.py
--in_tsv data/dataset_small_2k_normalized.tsv
--out_tsv new_results/attacks/saliency_strong.tsv
--ckpt checkpoints/run_weighted/best.pt
--vocab checkpoints/vocab_n.json
--api_candidates checkpoints/500_api_candidates.json
--only_malware
--cand_sample 30 --topk_salient 12 --iter_steps 1
```


### Window-Level Evaluation (After Attack)

```bash
python new_src/window_eval_plot.py
--data_file new_results/attacks/saliency_strong.tsv
--ckpt checkpoints/run_weighted/best.pt
--vocab checkpoints/vocab_n.json
--out_dir new_results/window_eval_after
```


### Generate Side-by-Side Comparison Plots

```bash
python new_src/compare.py
--before new_results/window_eval_before/window_eval.json
--after new_results/window_eval_after/window_eval.json
--out_dir new_results/comparisons_full
--thr <zero-fp-threshold>
--save_side_by_side
```

---

## Acknowledgements

This implementation is based on real sandbox telemetry provided for the project  
and inspired by dynamic malware analysis methodologies developed in academic research

## Contributions

Feel free to fork, raise issues, or submit PRs to improve this project!

## Author

Mozeel Vanwani

Computer Science and Engineering Undergrad 💻

Indian Institute of Technology Kharagpur 🎓

Email 📧: vanwani.mozeel@gmail.com
