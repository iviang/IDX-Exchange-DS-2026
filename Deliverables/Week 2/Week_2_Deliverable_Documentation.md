# Week 2 Deliverable Workflow Log
**Author:** Vivian Nguyen
---

# 1.	Objective
Creating basic EDA plots on the recent 6 months into pandas while filtering the data according to the task requirements.

## Task Requirements
- Explore distributions of:
    - ClosePrice (the target variable)
    - LivingArea
    - Bedrooms
    - Bathrooms
    - LotSize.
-	Restrict analysis to:
    - PropertyType = Residential 
    - PropertySubType = SingleFamilyResidence

--- 

# 2.	Procedure

## Environment Setup

- Created `01_exploration.ipynb` using VS Code
- Configured Python 3.13.14 kernel
- Installed required libraries:
  - pandas
  - matplotlib

## Data Loading

Loaded the six most recent monthly CSV files (202512–202605).

```python
csv = pd.read_csv(
    data[0],
    usecols=[
        "ClosePrice",
        "LivingArea",
        "BedroomsTotal",
        "BathroomsTotalInteger",
        "LotSizeAcres",
        "PropertyType",
        "PropertySubType",
    ],
)
```

Applied dataset restrictions per task docBedroomsTotal:

```python
csv = csv[
    (csv["PropertyType"] == "Residential")
    & (csv["PropertySubType"] == "SingleFamilyResidence")
]
```

## Data Preparation

- Filtered data according to task requirements
- Combined the six monthly DataFrames into one DataFrame using `pd.concat()`
- Verified dataset shape and preview
- Generated summary statistics
- Checked missing values
  - Deferred handling until Week 3 preprocessing
- Evaluated each variable individually for:
  - Outliers
  - Percentile distributions
BedroomsTotal
## Explored distributions
  - ClosePrice
  - LivingArea
  - Bedrooms (--> BedroomsTotal)
  - Bathrooms (--> BathroomsTotalInteger)
  - LotSize (--> LotSizeAcres)

---

# 3.	Findings
## Data Loading

- Successfully loaded **6 monthly CSV files**

## Combined Dataset
**Shape**
```
Size of combined DataFrame (rows, columns): (61727, 7)
```

---

## Missing Values
Missing values were identified in:

- LivingArea
- LotSizeAcres
- BathroomsTotalInteger

These will be addressed during Week 3 preprocessing.

```
ClosePrice                  0
PropertyType                0
LivingArea                 30
PropertySubType             0
LotSizeAcres             1083
BathroomsTotalInteger       1
BedroomsTotal               0
dtype: int64
```

---

## ClosePrice

Checking outliers and percentiles:
```
count    6.172700e+04
mean     1.340106e+06
std      7.307629e+06
min      1.750000e+00
90%      2.300000e+06
95%      3.200000e+06
99%      6.500000e+06
max      7.960000e+08
Name: ClosePrice, dtype: float64
```

- Distribution is highly right-skewed due to several extreme outliers.
- Created a secondary visualization near the 99th percentile (~$7M) to better visualize the majority of observations.
- Full visualization available in `01_exploration.ipynb` (44e60a2)

---

## LivingArea

Checking outliers and percentiles:
```
count    61697.000000
mean      2055.552918
std       1037.243701
min          0.000000
90%       3236.000000
95%       3838.000000
99%       5698.120000
max      23314.000000
Name: LivingArea, dtype: float64
```

- Distribution is generally reasonable.
- Identified a small number of unusually large and zero-valued observations.
- Full visualization available in `01_exploration.ipynb` (44e60a2)
---

## Bedrooms

Variable used:

```
BedroomsTotal
```

Checking outliers and percentiles:

```
count    61727.000000
mean         3.498437
std          0.966684
min          0.000000
90%          5.000000
95%          5.000000
99%          6.000000
max         22.000000
Name: BedroomsTotal, dtype: float64
```

Observations:

- Majority of homes contain **3–4 bedrooms**
- Full visualization available in `01_exploration.ipynb` (44e60a2)

---

## Bathrooms

Variable used:

```
BathroomsTotalInteger
```

Checking outliers and percentiles:
```
count    61726.000000
mean         2.644801
std          1.135026
min          0.000000
90%          4.000000
95%          5.000000
99%          6.000000
max         22.000000
Name: BathroomsTotalInteger, dtype: float64
```

Observations:

- Majority of homes contain **2–4 bathrooms**
- Full visualization available in `01_exploration.ipynb` (44e60a2)

---

## LotSize

Variable used:

```
LotSizeAcres
```

Checking outliers and percentiles:
```
count    60644.000000
mean        10.828569
std        544.530184
min          0.000000
90%          0.525180
95%          1.229085
99%          7.059700
max      60113.000000
Name: LotSizeAcres, dtype: float64
```

Observations:
- distribution was highly right-skewed due to several extremely large observations.
- secondary visualization focusing on values near the **99th percentile (less or equal to 8 acres)** was created for better visualization.
- Full visualization available in `01_exploration.ipynb` (44e60a2)

---

# 4.	Next Steps
- Cross-check findings with the team
- Revise EDA process based on peer methods
- Verify measurement units for consistency
- Address missing/null values
- Perform Week 3 preprocessing

---

# 5.	Tools and Resources
- VS Code
- Jupyter Notebook
- Python 3.13.14
- pandas
- matplotlib
- GitHub

---

# 6.	Code Revisions post Team Meeting (06/29/2026)

## Notebook Improvements
- Markdown headings for visual organization of code
- Removed unnecessary comments

## Read and filter the CSV

### Original implementation: 
I directly read in the files only of the columns ('ClosePrice', 'LivingArea', 'BedroomsTotal', 'BathroomsTotalInteger', 'LotSizeAcres', 'PropertyType', 'PropertySubType') as expected by the task doc with the restrictions where PropertyType = Residential and PropertySubType = SingleFamilyResidence. Loading the full dataset first provides greater flexibility for identifying additional predictors and preparing for Week 3 preprocessing.

### Reflection

While my original approach satisfied the assignment requirements, loading only seven variables limited exploration. This is helpful for identifying additional predictors and preparing for Week 3 preprocessing.


### Revision:

Removed `usecols`.

```python
csv = pd.read_csv(file)
```
Encountered:

```
DtypeWarning:
Columns (0: WaterfrontYN, 1: PostalCode) have mixed types.
```
Temporary solution:

```python
csv = pd.read_csv(file, low_memory=False)
```
Verified filtering by checking row counts before and after filtering.

---

## Dataset Overview
### Original Implementation

Only examined the filtered seven-column DataFrame.

```
Shape:
(61727, 7)
```

### Reflection

Reviewing the complete dataset provides better visibility into variable availability and potential predictors.

### Revision

Used:

```python
display(df.head(10))
df.info()
df.describe()
```

---	
## Missing Data
### Original Implementation
```python
df.isnull().sum()
```

### Reflection

Previously focused only on the seven required variables of raw counts. Examining missing values across the entire dataset will provide more context for preprocessing and identifying relationships between missingness and other variables. It would additionally be easier to take the values head on through percentages, also to organize by severity of missingness.

### Revision

- Checking for duplicates through the whole dataframe
    - 33 duplicates found through shared ListingKey values
        - Common pattern: commmon ClosePrice, different CloseDates 
- Implementing sorting by percentage of missing values
- bar chart visualization of missing values by percentages with a legend organized by severity of missingness (90-100% missing, 51-89% missing, 10–50% missing, < 10% missing)\
- Variables with 100% missing values
    - BusinessType
    - CoveredSpaces
    - MiddleOrJuniorSchoolDistrict
    - FireplacesTotal
    - TaxAnnualAmount
    - AboveGradeFinishedArea
    - TaxYear
    - ElementarySchoolDistrict

--- 
## Focused Variables

### Original Implementation
Originally this section was implemented through the loading of the .csv files directly organized with 'usecols'

### Reflection

### Revision
