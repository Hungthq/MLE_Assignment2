# EDA steps:

An effective Exploratory Data Analysis (EDA) moves from structural validation down to interaction analysis and feature prep.

1. **Inspect Structure & Data Types:** Sanity check dimensions and schema.
Check basic dimensions (`df.shape`), column names, and data types (`df.info()`). Ensure numerical features aren't encoded as strings, categorical variables are identified, and date columns are parsed as datetimes.


2. **Audit Data Quality & Missingness:** Quantify anomalies before analysis.
* **Missing values:** Calculate the percentage of missing values per column (`df.isna().mean()`). Decide whether missingness represents MCAR (completely at random) or carries structural signal.
* **Duplicates:** Check and isolate exact duplicate rows or repeated primary keys.
* **Invalid entries:** Scan for impossible physical values (e.g., negative ages, future transaction dates, extreme formatting inconsistencies).


3. **Univariate Analysis:** Evaluate one variable at a time.
* **Numerical features:** Inspect summary statistics (`mean`, `median`, `std`, `IQR`). Plot histograms or KDE plots to evaluate skewness, kurtosis, and multimodality. Use box plots to flag potential outliers.
* **Categorical features:** Compute frequency tables and cardinality counts (`df['col'].value_counts()`). Check for rare levels, dominant baseline categories, or imbalanced targets.


4. **Bivariate & Multivariate Analysis:** Identify correlations and interactions.
* **Numerical vs. Numerical:** Compute Spearman/Pearson correlation matrices to catch multicollinearity. Use scatter plots or pair plots to spot non-linear relationships.
* **Categorical vs. Numerical:** Use box plots, violin plots, or grouped aggregates (`df.groupby().agg(...)`) to check how numerical distributions shift across categories.
* **Categorical vs. Categorical:** Cross-tabulate with contingency tables (`pd.crosstab`) and use Chi-square tests to detect dependencies.
* **Target interaction:** If supervised, evaluate every feature's direct relationship against the target label.


5. **Detect Outliers & Distribution Shifts:** Assess extremes and stability.
Apply IQR boundaries ($Q_1 - 1.5 \times \text{IQR}$, $Q_3 + 1.5 \times \text{IQR}$) or z-scores ($\vert{}z\vert{} > 3$) to detect extreme values. Determine whether outliers reflect sensor/system noise to drop or legitimate high-impact edge cases to retain. If temporal, check for drift over time.


6. **Synthesize & Document Preprocessing Requirements:** Translate findings into modeling decisions.
Summarize actionable steps for the pipeline:

* Imputation strategies (median vs. mode vs. indicator column).
* Transformations required (log/Box-Cox for skewed features).
* Encodings needed (one-hot vs. target encoding vs. ordinal).
* Candidate features for pruning due to collinearity or near-zero variance.