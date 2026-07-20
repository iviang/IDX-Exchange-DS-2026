# Week 5 Deliverable Workflow Log
**Author:** Vivian Nguyen
---

Notebook: 04_model_comparison.ipynb 
Input: cleaned CSV (../../data/cleaned_CRMLSSold202512_202605.csv) from 02_preprocessing.ipynb (NOT uploaded to github) 
Deliverable: 04_model_comparison.ipynb

# 1. Objective

Train Decision Tree and Random Forest regressors on the corrected cleaned dataset and then compare their test R² against baseline (log, R^2 =0.5071), and document model behavior (Strenghts/Weaknesses)
Every named experiment logged to `results/model_results.csv` for easier access later on

## Task Requirements
- Try Decision Tree and Random Forest regressors
- Compare their test R² against baseline
- Document model behavior (strengths/weaknesses)
    - Log every experiment version (`model_results.csv`)
- Deliverable: `04_model_comparison.ipynb`

---

# 2. Procedure

## Environment Setup
- Input is the cleaned .csv output of `02_preprocessing.` (same file as Week 4)
- Setup replicates the baseline: same split, same 66 features, same evaluation contract
- All tunable values live at the top: `TEST_MONTH` (2026-05), `TRAIN_MONTHS` (5), `RANDOM_STATE` (42), paths, column names
- `RANDOM_STATE = 42` fixed on every tree/forest (trees tie-break randomly, forests bootstrap) for reproducibility under Restart & Run All

## Notebook Order

1. Load & sanity checks (reused from 03)
2. Chronological train/test split (reused from 03)
3. Feature selection (reused from 03)
4. Helpers: log_result (results logging), fit_eval (fit + back-transform + score), plot_pred_actual (predicted-vs-actual scatter)
5. Linear reference refit (reproduces baseline R^2 0.5071)
6. Decision Tree
7. Random Forest
8. Compare to baseline
9. Results + summary of approach

---
## Evaluation Contract and Standardization (shared by all models)
- Target: log(ClosePrice), fit in log space, predictions back-transformed via `np.exp` before scoring (mirrors 03, handles the right-skewed price distribution)
- Metric: R^2 in dollar space, full 12,007-row test set
    - every model is judged on the same scale as the baseline
- Split: fixed chronological, train Dec 2025 to Apr 2026 / test May 2026
- Features: same 66-feature matrix as Week 4
    - `County_Los_Angeles` also dropped as the dummy reference (per week 4)
- Trees fit on unscaled features 

---

## Linear Reference Refit
- Refit the Week 4 baseline inside this notebook before any new model runs
- assert abs(test_r2 - 0.5071) < 0.0005 
    - certifies data, split, and features are identical to 03_baseline_model.ipynb, so every tree vs baseline attributable to the model
- Reproduced: train R^2 = 0.5425 | test R^2 = 0.5071

---

# Logging in model_results.csv
- One row per named experiment (model, version) in model_results.csv
- rerun of notebook makes sure that log reflects the latest run
- all tag log(ClosePrice) except for A-raw which uses raw ClosePrice

## Experiment Log 

- DecisionTree A 
    - config: DecisionTreeRegressor(random_state=42), log and raw
    - change vs previous: unconstrained to expose overfitting
- DecisionTree B (depth-controlled)
    - config: max_depth swept [4, 6, 8, 10, 12, 16, 20], best depth (6) refit and logged
    - change vs previous: constrain the one knob that controls the bias-variance regime
    - Note: max_depth is one of several single-tree regularizers
        - min_samples_leaf=5 (the same constraint used in RF.B) was run on a single tree as an unlogged diagnostic
            - test 0.4789, still below baseline and below the DT.B run
            - confirms no single-tree knob is strong enough to beat out the linear
- RandomForest A 
    - config: RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1), log
    - change vs previous: 100-tree ensemble
    - test whether bagging fixes the single-tree variance
- RandomForest B
    - config: RandomForestRegressor(n_estimators=100, min_samples_leaf=5, ...), log
    - change vs previous: one light constraint (>= 5 sales per leaf) to smooth single-sale memorization
    - a depth sweep [8, 12, 16, 20, None]
        - Best depth = None reproduces RF.A exactly, so it produces no distinct model and gets no row
    - Note: an earlier RF.B version swept `max_depth` instead but was redudant and replaced with leaf constraint instead per the depth sweep section
---

# 3. Findings

- Baseline to beat: log LinearRegression, test R^2 0.5071 (full test, dollar space) 
- DecisionTree:
    - DT.A (unconstrained): train 0.9986 / test 0.1370, gap 0.86, severe overfit
        - log vs raw nearly identical (0.137 vs 0.135). Target transform barely matters for a model that partitions rather than fits a line
    - DT.B (max_depth=6): train 0.5202 / test 0.4976, gap 0.02, overfit controlled but still below baseline
        - test R^2 degrades as depth grows past 6 (depths 16 to 20 fall to ~0.29): extra depth memorizes training months
    - predicted vs actual scatter: 
        - DT.B shows horizontal stripes (piecewise-constant buckets)
        - both DT.A and DT.B underpredict the high end values
- RandomForest:
    - RF.A (unconstrained, 100 trees): train 0.9039 / test 0.5939, first model to beat baseline (+0.087)
    - RF.B (min_samples_leaf=5): train 0.7166 / test 0.6043, best model of the week (+0.097 over baseline)
        - depth-sweep diagnostic: 
            - RF test R2 climbs then flattens (0.548 to 0.594), never collapses
            - Best = None = RF.A
            
- strengths:
    - RF breaks the single-tree lose-lose (deep = noisy, shallow = coarse)
        - averaging keeps depth and stability at once
    - RF.B's leaf constraint improved test R2 (+0.010 over RF.A) and cut the gap from 0.31 to 0.11. 
        - constraint removed noise, not signal
    - scatter shows RF.B a (subjectively) tighter cloud than RF.A (less vertical spread = more consistent predictions for similar homes)
    - trees need no scaling; simpler pipeline than the linear baseline
- weaknesses:
- single trees cannot beat linear
    - Piecewise predictions can't approximate the smooth log-linear relationships
- RF gap reduced but not closed (0.11)
    - trees still memorize within the 5-sale floor
- both tree families underpredict the high end
    - Trees can't extrapolate past the priciest homes seen in training set
- best hyperparameters (DT depth, RF leaf) chosen by peeking at test R^2
    - strictly should use a validation split from training months
- Overall Model Notes:
    - Ranking by test R2: RF.B (0.604) > RF.A (0.594) > baseline (0.507) > DT.B (0.498) > DT.A (0.137)
    - The same idea (constrain to reduce overfit) fails on a single tree (DT.B still < baseline) but wins on a forest (RF.B)
        - The forest had variance worth removing while the tree had discarded the signal
    - expect more refining in next week tasks of further feature engineering
---

# 4. Next Steps

- Team meeting to identify areas of improvement
- Week 6: feature engineering

---

# 5. Tools and Resources

- VS Code
- Jupyter Notebook
- Python 3.13.14
- pandas, numpy, scikit-learn, matplotlib, seaborn
- GitHub (raw and cleaned data excluded via .gitignore)
- Decision Tree method research
- Random Forest method research
    - constraints of RF

---

# 6. Meeting Notes/Thoughts

- 

---

# 7. Revisions

- 
