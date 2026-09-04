# Predicting California Property Close Price (Final Sales)
This repository contains Python code for data preprocessing, model development, evaluation, and prediction of the ClosePrice of single-family residential properties in California. Documentation is included to explain the methodology, assumptions, and decisions made throughout the analysis process.
The workflow is organized into notebooks that inform a deployed Streamlit app built through a separate repository (most recent version: <https://github.com/iviang/IDX-DS26-streamlit-v3>). App is deployed at: <https://idx-ds26-app-v3-o8iv5baqvzgg2rb99jfviw.streamlit.app/>

## Project Objective
The formal task of this project is to develop a machine learning model to predict the close price of any single residential property (currently for sale or not for sale) in California based on its characteristics at the time of the query.

## Repository Contents
- Data preprocessing scripts
- Model development
- Model training
- Model evaluation
- Price prediction
- Project documentation

| Notebook | Purpose |
|---|---|
| `01_exploration.ipynb` | EDA on the target and key features |
| `02_preprocessing.ipynb` | Cleaning, feature engineering, encoding, chronological split, cleaned-CSV export |
| `03_baseline_model.ipynb` | Linear Regression baseline |
| `04_model_comparison.ipynb` | Decision Tree and Random Forest |
| `05_advanced_models.ipynb` | LightGBM, XGBoost, CatBoost, and ensembles |
| `06_evaluation.ipynb` | Full metric suite (R², MAE, RMSE, MAPE, MdAPE), log-target experiment, model export |

## Dataset Source
- **Data input**: from CRMLS data files, internally access through IDX Exchange FTP. All Data files are not committed to this repository and stored locally to a git-ignored /data folder.
- **Window of Data**: June 2025 - June 2026 (062025, ..., 062026) Consecutive.
- **Target variable:** `ClosePrice` (the final agreed sale price).
- Additional geographic layer data of California School District areas of 2025-2026, accessed via geoJSON from <https://data.ca.gov/dataset/california-school-district-areas-2025-26>
- Filtered to Residential, SingleFamilyResidence, CA properties
- Note: at implementation of this project, test month is : **June 2026 (062026)**

## Preprocessing
Preprocessing is performed across all input CRMLSSold*.csv files in notebook: ../Deliverables/Week 3/**02_preprocesssing.ipynb**. The notebook outputs a cleaned CSV to a git-ignored /data folder stored locally.
1. **Load & Filter**
2. **Deduplication**
3. **Implausible Values**
4. **Missing Values**
5. **Leakage prevention**
6.  **Feature engineering (Week 6)**
7.  **Encoding**
8.  **Normalization**
9.  **Chronological train/test split**

## Models Tested
All models are scored in **dollar space** (log-target models are back-transformed with `np.exp` before scoring). Unless noted, test R^2 is on the 1/99-trimmed June test set and the baseline to beat is Linear Regression at **0.8152**. 
All most updated models and results reviewable at: <../results/**model_results.csv**>

### Linear Regression (Baseline) [../Deliverables/Week 4/**03_baseline_model.ipynb**]
- Raw Target
- Log Target
### Decision Tree [../Deliverables/Week 5/**04_model_comparison.ipynb**]
- A-Raw Target
- A-Log Target
- B-max_depth constraint
- C-min_samples_leaf constraint
### Random Forest [../Deliverables/Week 5/**04_model_comparison.ipynb**]
- A-Raw Target
- A-Log Target
- A + School District feature incorporation
- B-min_samples_leaf constraint
- C grid tuning (time series)
### Gradient Boost [../Deliverables/Week 7/**05_advanced_models.ipynb**]
- LightGBM
- LightGBM tuned
- LightGBM tuned + School District feature incorporation
- XGBoost
- XGBoost tuned
- XGBoost tuned + School District feature incorporation
- CatBoost native categoricals
- CatBoost tuned

### Ensemble [../Deliverables/Week 7/**05_advanced_models.ipynb**]
- Weighted blend (0.35 LightGBM / 0.25 XGBoost / 0.40 CatBoost)
- Ridge Stack (LightGBM+XGBoost+CatBoost)

### Log-target Experiment [../Deliverables/Week 8/**06_evaluation.ipynb**]
- LightGBM + School District feature incorporation, using log target 

## Best Results

Best results of this repository derived two models that specialize in different markets.
For the typical single family residence property at mid-market, the strongest result comes from Ensemble Stack-Ridge, which reports 90.6% accuracy in ClosePrice predictions, scored on only the 1st-99th percentile of the test-set's close prices. Outlier sales are not factored in.
To achieve prediction more accurately on the whole test-set with a focus the luxury tail accuracy, LightGBM, tuned factoring in the School District geographical layer, shows the highest accuracy at 82% through the log target of ClosePrice. 

**Mid-market (1st–99th percentile trimmed test-set):** 

| Model | Feature Set | Target | R^2 | MAE | RMSE | MdAPE | 
|---|---|---|---|---|---|---|
| **Ensemble (stack-ridge)** | tuned base models +District | raw | **0.9064** | $146,915 | $269,628 | 7.93% |
| XGBoost (B-tuned+district)| tuned +District | raw | 0.9008 | $151,510 | $277,504 | 8.48% |
| LightGBM (B-tuned+district)| tuned +District | raw | 0.8989 | $152,157 | $280,176 | 8.42% |
| CatBoost (B-tuned)| tuned, native cats | raw | 0.8936 | $157,355 | $287,343 | 8.65% |
| RandomForest (A+SchoolDistrict)| base +District | raw | 0.8814 | $158,375 | $303,464 | 7.88% |
| **LinearRegression (baseline)** | base | raw | **0.8152** | $235,407 | $378,752 | 15.65% |
| DecisionTree (C-leaf5)| base, constrained min_samples_leaf | raw | 0.8143 | $201,957 | $379,727 | 10.18% |


**Full market INCLUDING the luxury tail (untrimmed test-set):**

| Model | Feature Set | Target | R^2 | MAE | RMSE | MdAPE | 
|---|---|---|---|---|---|---|
| **LightGBM** (B-tuned+district-LOG) | tuned +District | Log | **0.8200** | $191,651 | $651,556 | 8.06% |
| Ensemble (stack4-log) | tuned base models +District| Mixed | 0.8020 | $191,993 | $683,371 | 8.35% |
| CatBoost (B-tuned) | tuned, native cats| Raw | 0.6224 | $220,118 | $943,799 | 8.87% |
| XGBoost (B-tuned) | tuned | Raw | 0.5978 | $219,937 | $974,034 | 8.80% |
| RandomForest (A+SchoolDistrict) | base +District | Raw | 0.5525 | $229,912 | $1,027,443 | 8.10% |
| DecisionTree (C-leaf5) | base, constrained min_samples_leaf  | Raw | 0.5313 | $272,226 | $1,051,561 | 10.48% |
| **LinearRegression (baseline)** | Base | Raw | 0.5226 | $309,885 | $1,061,299 | 16.01% |

Full direct results export of all models saved to .../results/*.

## Instructions (full project)
### Environment/Set Up
Notebooks requires Python 3.13.14. Install dependencies as commented in `Set Up` sections of each notebook.
- Download a minimum of 6 months of `CRMLSSold*.csv` from the IDX Exchange FTP server into `data/` (credentials are in the internal task prompt; do not commit them or the raw data).
- Download the [CA School District Areas 2025–26]<https://data.ca.gov/dataset/california-school-district-areas-2025-26> geoJSON into `data/` for the spatial-join feature.
### Run the notebooks
Notebooks must be run one by one, in complete order to create the correct artifacts necessary to inform the next.

1. `01_exploration.ipynb` — EDA (optional to re-run and is moreso informational about the data inserted)
2. `02_preprocessing.ipynb` — writes the cleaned CSV to git-ignored `data/`.
3. `03_baseline_model.ipynb` — Linear Regression; seeds `results/model_results.csv`.
4. `04_model_comparison.ipynb` — Decision Tree + Random Forest; seeds `results/model_results.csv`.
5. `05_advanced_models.ipynb` — LightGBM / XGBoost / CatBoost + ensembles; writes model bundles to `models/`; seeds `results/model_results.csv`.
6. `06_evaluation.ipynb` — full metric suite; writes `results/metrics_summary.csv` and exports the deployment bundle **`models/models.pkl`**.

Use "Restart & Run All" for each notebook so the results log reflects the latest run.

## App Deployment (Streamlit)
The most recent variation of the App is deployed at: <https://idx-ds26-app-v3-o8iv5baqvzgg2rb99jfviw.streamlit.app/>
The app estimates a property's closing price from a typed **address** plus its characteristics. The address is geocoded to latitude/longitude (Google Maps Geocoding API when a key is configured), and location features (ZIP, City, County, School District) are resolved from that.
Files that build the deployment are placed in a separate repository at : <https://github.com/iviang/IDX-DS26-streamlit-v3>

User can input property details and features: 
- Living area (sqft)
- Lot Size Unit (Acres/sqft)
- Bedrooms
- Bathrooms
- Year Built
- Stories
- Garage Spaces
- Total Parking Spaces
- View (Y/N)
- Basement (Y/N)
- Private Pool (Y/N)
- Attached Garage (Y/N)
- Fireplace (Y/N)
- New Construction (Y/N)

**Version history**

- **v1** — user manually supplies latitude, longitude, and location cluster.
- **v2** — user enters an address; geocoding + a saved KMeans model derive the location features.
- **v3 (current)** — a **router**: it takes a ballpark estimate and switches (not blends) between the two best models of this repository: **Ridge stack** for the mid-market (ballpark ≤ $6.3M) and **LightGBM-log** for the luxury tail (ballpark > $6.3M).

## Author
Vivian Nguyen (ds49)
Data Scientist Intern DS49 IDX Exchange
