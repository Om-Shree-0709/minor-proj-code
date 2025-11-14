## Smart Home Activity Anomaly Detection

This repository contains a single Jupyter notebook, `minor.ipynb`, that builds an end-to-end pipeline for detecting abnormal daily routines in a smart home using longitudinal sensor logs and a sequence modeling approach with PyTorch.

### Project Overview

- **Goal:** Learn normal activity sequences from ambient sensors and flag days that deviate from the learned routine. The anomalies simulate mild cognitive impairment (aMCI) and dementia-like behavior patterns.
- **Input:** Raw CASAS-style event logs (`data`) with timestamps, sensor IDs, status values, and optional activity labels.
- **Output:** Daily anomaly scores, ROC/PR-style diagnostics, confusion matrix, and interpretability plots (heatmaps of activity frequency).

### Dataset & Preprocessing

1. **Parsing:** Load the whitespace-delimited log, split into date, time, sensor ID, status, and activity columns.
2. **Cleaning:**
   - Drop `Sensor_ID`, `Status`, and `Time`.
   - Remove blank activities, “Respirate begin/end”, trailing “end” markers, and consecutive duplicates.
   - Trim partial days (e.g., half-day for `2011-05-23`) and drop days with <5 labeled events.
3. **Chronological Split:** Use 14% of dates for training, the next 14% for validation, and the remaining 72% for testing (by date order).
4. **Synthetic Anomalies:** Introduce domain-inspired perturbations into specific date ranges to emulate:
   - Sleep interruption bursts (`Bed_to_Toilet → Eating → Sleeping`)
   - Highly repetitive meal/relax/eating loops
   - Housekeeping obsession (Housekeeping/Wash_Dishes cycles)
   - Relaxation spikes spread through the day
   - Dementia-like erratic routines late in the timeline  
     Dates are sampled randomly per regime to keep class labels hidden from the model.

### Modeling Pipeline

- **Encoding:** Map 10 activities plus an explicit End-of-Sequence token (EoS) to integers; store as `Encoded_Activity` in each split.
- **Windowing:** Slide a window of 5 activities across each day and predict the next activity (next-event modeling). Each window → `(window, next_activity_one_hot)`.
- **Architecture:** 4-layer LSTM (`hidden_size=128`, `batch_first=True`) with a linear head and cross-entropy loss. Optimized via Adam (`lr=1e-3`, 50 epochs, batch size 10).
- **Scoring:** During validation and testing, compute the per-window prediction loss, average it per day, and treat the mean loss as that day’s anomaly score.
- **Thresholding:** Sweep 3,000 thresholds between the min/max validation losses and pick the one with the best F1 against the known injected anomalies.

### Key Results

- **Best threshold:** ~1.65 (derived from validation sweep)
- **Test metrics:** Accuracy 0.803, Precision 0.600, Recall 0.615, F1 0.608
- **ROC AUC:** 0.716 with max accuracy ~0.822 at threshold 1.78
- **Visuals:** Daily loss traces with highlighted anomalies, ROC curve, accuracy-vs-threshold curve, confusion matrix, and heatmaps showing anomalous days (outlined in red).

### Repository Layout

```
minor.ipynb   # Full pipeline: data loading, preprocessing, anomaly synthesis, LSTM training, evaluation, viz
data          # Raw sensor log (not tracked in Git; place alongside the notebook)
README.md     # Project documentation (this file)
```

### Getting Started

1. **Clone & setup**
   ```bash
   git clone <repo-url>
   cd minor-proj-code
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   pip install pandas numpy matplotlib seaborn scikit-learn torch jupyter
   ```
2. **Data:** Place the raw sensor log file at `minor proj code/data`. The notebook assumes this filename.
3. **Dependencies:**  
   `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `torch`, `torchvision` (auto-installed with PyTorch), and `jupyter`.

### Running the Notebook

1. Launch Jupyter: `jupyter notebook`.
2. Open `minor.ipynb`.
3. Execute cells sequentially:
   - **Data preparation:** Loads/cleans the log and prepares train/val/test splits.
   - **Anomaly injection:** Reruns to resample random anomaly days (results vary by RNG seed).
   - **Model training:** Trains the LSTM predictor (expect UserWarning about tensor construction).
   - **Evaluation:** Generates plots, best-threshold selection, and metrics.

> **Note:** Because anomaly days are sampled randomly, metrics may vary slightly per run. For reproducibility, fix Python’s `random.seed()` and NumPy/PyTorch seeds before sampling/training.

### Extending the Project

- Tune hyperparameters (window size, hidden units, layers, learning rate).
- Experiment with bidirectional LSTM, GRU, or transformer-based predictors.
- Replace rule-based anomaly injection with clinically validated behavioral signatures.
- Introduce additional sensor features (status values, durations) or multivariate encodings.
- Export the trained model and daily anomaly scores as standalone scripts or dashboards.

### License & Citation

If you rely on third-party datasets (e.g., CASAS smart home logs), please respect their licenses and cite them accordingly. Add an explicit license to this repo before distributing.

---

Feel free to open issues or PRs for improvements, cleaner data loaders, or deployment scripts.
