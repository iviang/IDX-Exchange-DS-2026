# Week 3 Deliverable Workflow Log
**Author:** Vivian Nguyen
---

# 1. Objective

Preprocess the CRMLS sold-listings data (Dec 2025 – May 2026, single-family residences) into a cleaned dataset ready for Week 4 modeling, applying the findings from Week 2 EDA.

## Task Requirements
- Handle missing values (decide whether to drop, impute, or flag)
- Convert categorical fields to numeric (encoding)
- Normalize numerical features if needed
- Create train/test split:
    - most recent month of available data = test set
    - the X months immediately preceding = training set
    - X is not fixed — treat the training window length as a tunable choice
- Deliverable: `02_preprocessing.ipynb` + cleaned CSV (Not uploaded to public Github)

---

# 2. Procedure

## Environment Setup

- Created `02_preprocessing.ipynb`
- Raw `CRMLSSold*.csv` files stay in the local gitignored `/data` folder; the notebook loads them via a `glob` pattern, so anyone with the raw files can run it end-to-end without the data ever entering the GitHub repo

## Pipeline Order

1. Load & filter (identical restrictions to `01_exploration.ipynb`)
2. Deduplicate ListingKeys
3. Remove/repair implausible values
4. Handle missing values
5. Feature selection (drop leakage & unusable columns)
6. Encode categoricals
7. Normalize numerical features
8. Chronological train/test split
9. Export cleaned CSV

Note: Deduplication and implausible-value removal come before the imputing and normalizing to avoid contamination and skews.

---

## Load & Filter

- Same loading logic as `01_exploration.ipynb`
- Keeping the filter identical to the EDA notebook means every Week 2 finding applies to exactly this data
- `CloseDate` parsed to datetime up front 
    - as needed by both the dedup rule and the chronological split
- ensuring only data within state of California is used and cutting out accidental additions

---

## Deduplication

- Week 2 found 33 duplicated `ListingKey`s
    - most pairs agree on `ClosePrice` but differ on `CloseDate`
    - a few pairs conflict on `ClosePrice`
- first I separate and display the conflicting groups for manual inspection (only a realistic step due to the number of dupes, wouldn't be realistic if 50+) before resolving
- Resolution: keep the most recent record per `ListingKey` (sorted by `CloseDate`, tiebroken by `ModificationTimestamp`). 
    - Since duplicates come from re-reporting, the later record is the most current version the MLS published, helps to correct conflicts

---

## Implausible Values

Distinct from missing values where present but wrong.

- Zero-value structure fields (rows dropped): a single-family residence cannot have 0 bedrooms, 0 bathrooms, or 0 living area. 
    - There is no sound basis to impute them while the affected row counts are also a few amount
- ClosePrice: inspecting to determine what to keep, repair, or drop due to ambiguity
    - ClosePrice can never be imputed 
    - inspection of the 11 out-of-bounds rows (via a Close/List ratio column) showed three distinct populations, each handled on its evidence: 
        - unit errors
            - 7 rows with ratios clustering at ~1,000 (three extra zeros) 
            - 1 row at ~1/1,000,000 ($1.75 close against a $1.749M list, i.e. entered in millions) 
            repaired by the unique power-of-ten factor that lands the corrected price in a normal close-to-list band [0.80, 1.25], and 
            - flagged in ClosePrice_repaired audit column; 
        - ambiguous artifacts (DROP)
            - rows no power of ten reconciles with the list price
        - one real sale outside the bounds (KEPT)
            - an $8,000 close on an $8,000 list in Trona (ratio 1.0; a depressed market where four-figure sales are genuine)\
    - repaired targets are inferences, not observations (the in-millions row may echo the list price), so Week 4 should audit the flagged rows' residuals. 
    - I considered attempting a percentile trim was considered but 123 entries would've been removed so I opted to preserve as much as possible.
- LotSizeAcres: unit-conversion errors (rows with exactly 43,560 "acres" = 1 acre
  expressed in sq ft; max 60,113 acres ≈ 94 sq mi).
    - cross-check diagnostic against LotSizeSquareFeet column
        - 60,363 of 60,366 dual-populated rows consistent at ratio 43,560
        - No copy errors found
        - the 3 mismatches are rounding artifacts at tiny magnitudes (removed by the too-small rule anyway)
        - every checkable suspect row's sqft = the bad acres × 43,560
            - so can't repair the corrupted rows, and the heuristic repair is the
              justified approach.
    - Repair logic:
        - 0 acres → NaN (a real property isn't able to have 0 space)
        - values above 640 acres (1 sq mile) are suspected square-feet entries
            - divide by 43,560
            - then I accept only if the result lands in a plausible [0.01, 640] acre range,
            - otherwise NaN.
        - Results show:
            - 45 suspects, all 45 repaired, and the originals (2,262–43,560) are
              textbook lot sizes in sq ft, converting to 0.05–1.38 acres
            - nonzero values below 0.01 acres (~436 sq ft) → NaN (12 rows
                - too small to be a real lot, no safe interpretation of the error direction)
        - where acres is entirely unusable but LotSizeSquareFeet is populated (157 rows:
          145 from the diagnostic + the 12 too-small rows after NaN-ing)
            - attempted recovery: acres = sqft ÷ 43,560, plausibility-guarded
            - result: 0 recovered, 157 rejected — every candidate held near-zero junk,
              confirming the sq-ft column is auto-derived everywhere in this extract
            - the guard prevented injecting 157 implausible values; rule retained for
              future extracts where sqft may be directly entered
        - a LotSizeAcres_imputed flag column marks every row whose final value is
          synthetic

---

## Missing Values

The following workflow was used to determine how missing values were handled throughout the preprocessing pipeline.


- Drop Features with No Information
    - Dynamically identify columns that are 100% missing.
    - Remove these columns from the dataset since they contain no usable information.
    - This step is data-driven and automatically adapts to future CRMLSSold extracts.

- Standardize Boolean Fields
    - Convert missing values in YN indicator fields to False.
    - Standardize all Boolean representations (Y/N, True/False, 1/0) into a consistent Boolean format.
    - Drop Boolean features with a minority class frequency below **0.1%**, as they provide little to no predictive information.

- Preserve Key Predictors
    - LivingArea and BathroomsTotalInteger are considered essential predictors of housing price.
    - Rather than imputing these variables, remove rows containing missing values to avoid introducing artificial signal.
    - 31 rows were removed, resulting in minimal data loss.

- Impute Lot Size
    - LotSizeAcres is treated as a weaker predictor and contains relatively few missing values.
    - Missing values are filled using the median calculated from the training period only.
    - A companion flag (LotSizeAcres_imputed) is retained to indicate which observations were imputed.

Preventing Data Leakage

- Summary statistics used for imputation were calculated from the training data, excluding the most recent month reserved for testing.
    - Because the latest month is always designated as the test set, no information from the test period contributes to the preprocessing stats.
- Dataset-wide cleaning operations (e.g., removing invalid values, correcting unit inconsistencies, or trimming impossible observations) are permitted to use the full dataset

---

## Feature Selection & Leakage

- Dropped all price-like columns except the target
    - (`ListPrice`, `OriginalListPrice`, and anything else matching the pattern, printed for review): 
        - on a sold listing, any other price field is either listing-side information or derived from the close price
- YearBuilt dropped
    - to keep the feature set lean, per its near-zero correlation with ClosePrice
    - EDA median-price-by-year plot showed a nonlinear pattern that Pearson/Spearman understate
    - If week 4 task underperforms, I can reintroduce YearBuilt

- PostalCode kept as a raw reference column
    - 1,675 unique values is too many to one-hot
    - CountyOrParish carries the location signal for now, and keeping the raw column
    - Week 4 revisit (e.g. target encoding) without re-running preprocessing
- Note: the three size variables are strongly intercorrelated 
    - EDA: (LivingArea–Bathrooms 0.82, LivingArea–Bedrooms 0.65, Bedrooms–Bathrooms 0.61). (All kept)
    - multicollinearity mainly muddies linear-model coefficients, not predictions

---

## Encoding

- Boolean YN fields and the imputed-flag 
    - standardizing to 0/1 integers
- CountyOrParish → one-hot (~58 dummy columns): 
    - no target leakage (unlike target encoding), no false ordinality (unlike label encoding).
    - All dummies kept and seen in final .csv
    - if Week 4 fits an unregularized linear model with an intercept, one dummy can be dropped 

---

## Normalizing

- ClosePrice and LivingArea: right-skewed, log-normalize well 
    - added log1p versions (raw columns kept so Week 4 can narrow down). 
    - Post-cleaning skew results
        - ClosePrice: 12.69 → 0.51
        - LivingArea: 2.95 → 0.32
- LotSizeAcres: the 2c repair cut raw skew from EDA-era extremes to 45.78, but log1p only reached 5.69 
    - most lots are < 1 acre, where log1p(x) ≈ x is near-identity and leaves the distribution's bulk untransformed
    - Plain log reaches 2.26 → LotSizeAcres_log uses plain log
    - residual skew is structural and accepted given the weak predictive role
- Bedrooms/bathrooms: small discrete counts
    - left as-is
- Standardization (z-scoring) (TBD week 4 task?)
    - a scaler must be fit on training rows only, and the training window X is still a tunable choice there
    - The log transforms are safe now because they are row-wise (no statistics fitted on the data)

---

## Train/Test Split

- Most recent month: CRMLSSold202605.csv
- Implemented as a function, chrono_split(df, x): test = most recent month (fixed), train = the x months immediately preceding
- TRAIN_WINDOW_MONTHS:
    - single tunable variable (default 3)
- demo cell prints train/test sizes for every possible X (1–5) to make the trade-off visible
    - more months = more training data
    - the oldest months are least representative of the test month's market
- Sanity assertions confirm no train row closes on/after the test month

- Note: In week 4 we can now choose the optimal X by validating within the training data

---

## Export

- Cleaned CSV (`cleaned_CRMLSSold.csv`) is written into the gitignored `/data` folder, so the cleaned data also stays off the public repo
- The full cleaned frame (all 6 months) is exported
- train/test subsets are re-derivable from it
---

# 3. Findings

- Rows after Residential/SFR filter: 61,727 (matches EDA); geographic scope filter removed 3 rows (1 Foreign Country, 2 Other State) → 61,724
- Rows removed by dedup: 33 (matches EDA)
- Conflicting-ClosePrice duplicate groups inspected: 9
- Rows dropped for zero-value structure fields: 54 unique rows 
- ClosePrice out-of-bounds rows: 11 examined
    - 8 repaired via unit decoding (seven ÷1,000, one ×1,000,000; repaired Close/List ratios 0.886–1.001, all PASS)
    - 1 kept as-is (Trona, ratio 1.0)
    - 2 dropped as ambiguous
- LotSizeAcres:
    - 45 sq-ft entries converted (all 45 suspects; 0 unconvertible); post-repair max 450.00 acres
    - 185 zeros and 12 too-small values → NaN
    - Cross-check vs LotSizeSquareFeet: 60,363 of 60,366 dual-populated rows consistent, 0 copy errors, 3 mismatches (rounding artifacts at tiny magnitudes, removed by the too-small rule) — 
    - the sq-ft column is auto-derived and cannot repair the suspects, justifying the heuristic
    - Total NaN after repair: 1,279 (2.08%); 
        - imputed: 1,278 with median 0.1671 acres fit excluding the test month 
    - Recovery from LotSizeSquareFeet (implemented): 0 of 157 candidates passed the plausibility
        - every candidate held near-zero junk, a definitive negative result confirming the sq-ft column is auto-derived everywhere in this extract
    - Borderline row noted: ListingKey 1153265653 (726.41 → 0.0167 acres, near the plausibility floor, a genuine 726-acre parcel cannot be fully excluded)
- Columns dropped as 100% null: 8, matching the EDA list exactly — FireplacesTotal, AboveGradeFinishedArea, TaxAnnualAmount, TaxYear, ElementarySchoolDistrict, BusinessType, CoveredSpaces, MiddleOrJuniorSchoolDistrict
- Near-constant YN columns 
    - dropped: 1 WaterfrontYN (0.06% True)
    - Kept: ViewYN (57.03%), AttachedGarageYN (73.46%), FireplaceYN (72.35%), PoolPrivateYN (15.58%), NewConstructionYN (3.48%), BasementYN (2.36%)
- LotSizeAcres skewness after repair 
    - raw: 45.78, log1p: 5.69, plain log: 2.26 
- Final cleaned dataset shape: 61,604 rows × 77 columns
- Row-count reconciliation: 61,727 (SFR filter) 
    - −3 geographic scope → −33 dedup → −54 zero-structure → −2 ambiguous ClosePrice → −31 missing key predictors → 61,604
- Column reconciliation:
    - 78 raw + 2 created (ClosePrice_repaired, LotSizeAcres_imputed) = 80 → −8 fully null → −1 WaterfrontYN → −2 price-like leakage → 17 kept at feature selection → +56 dummies +3 logs +CloseMonth = 77
- Train/test at default X=3: 31,697 / 12,007 (~72/28); monthly volumes show real seasonality (Jan trough ~7.5K closings, spring ramp to ~12K) 
    - relevant to the Week 4 optimal-X experiment. 
    - 3 out-of-scope rows traced to April (1) and December (2)

---

# 4. Next Steps

- Cross-check preprocessing decisions with the team (especially the YearBuilt drop and the percentile-trim trade-off)
- Week 4: modeling 
    - experiment with X via a validation month inside the training window
    - fit scalers on train only
    - revisit YearBuilt and PostalCode if baseline models underperform

---

# 5. Tools and Resources

- VS Code
- Jupyter Notebook
- Python 3.13.14
- pandas, numpy
- GitHub (raw and cleaned data excluded via .gitignore)
- 

---

# 6. Meeting Notes/Thoughts (07/06/2026)

