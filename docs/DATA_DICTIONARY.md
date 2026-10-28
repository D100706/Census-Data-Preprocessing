# Data Documentation

## Source and scope
**Dataset:** Adult / Census Income  
**Source:** UCI Machine Learning Repository, dataset ID 2  
**Official page:** https://archive.ics.uci.edu/dataset/2/adult  
**Original sample count:** 48,842 instances (as listed by UCI)  
**License:** CC BY 4.0  
**Original collection:** 1994 US Census extraction.

The dataset is used to explore whether annual income is above or below a historical $50,000 threshold. It contains demographic, education, work, and capital-gain/loss attributes. The threshold is historical and should not be interpreted as a current income standard.

## Original variables (common UCI field names)
| Variable | Meaning | Typical type | Notes |
|---|---|---|---|
| age | Age in years | Integer | Implausible values outside 17–100 are set missing before imputation. |
| workclass | Employment sector/class | Categorical | `?` becomes missing, then `Unknown`. |
| fnlwgt | Census sampling weight | Integer | Retained in the clean dataset for traceability. |
| education | Highest education category | Categorical | Original category retained. |
| education_num | Education level encoded numerically | Integer | Numeric conversion is explicit. |
| marital_status | Marital status | Categorical | Whitespace trimmed. |
| occupation | Occupation category | Categorical | `?` becomes missing, then `Unknown`. |
| relationship | Relationship category | Categorical | Whitespace trimmed. |
| race | Census race category | Categorical | Sensitive demographic attribute; use responsibly. |
| sex | Recorded sex category | Categorical | Sensitive demographic attribute; use responsibly. |
| capital_gain | Capital gains | Numeric | Negative values set missing; original valid extremes retained. |
| capital_loss | Capital losses | Numeric | Negative values set missing; original valid extremes retained. |
| hours_per_week | Hours worked per week | Numeric | Values outside 1–99 set missing. |
| native_country | Country of origin | Categorical | Missing values imputed using the mode. |
| income | Historical income class | Target/category | Trailing period removed if present; values otherwise retained. |

## Engineered features
| Feature | Definition | Reason |
|---|---|---|
| has_capital_gain | 1 if capital_gain > 0, otherwise 0 | Separates no reported gain from a positive amount. |
| capital_gain_log1p | `log(1 + max(capital_gain, 0))` | Reduces positive skew while preserving zero values. |
| has_capital_loss | 1 if capital_loss > 0, otherwise 0 | Separates no reported loss from a positive loss amount. |
| capital_loss_log1p | `log(1 + max(capital_loss, 0))` | Reduces positive skew. |
| age_group | Age bands: 17–25, 26–35, 36–45, 46–55, 56–65, 66+ | Makes age segments easier to summarise. |
| work_hours_band | Part-time-low, Part-time, Standard-40, Overtime, Very-high | Creates interpretable working-hours groups. |

## Quality and transformation decisions
- Trim leading/trailing whitespace in text fields.
- Convert `?` and empty strings to missing values.
- Convert known numeric columns using `pandas.to_numeric(errors="coerce")`.
- Remove exact duplicate rows only. No person-level unique ID is available, so similar records are not assumed to be duplicates.
- Treat implausible age and weekly-hour values and negative capital amounts as invalid/missing.
- Preserve missing workclass/occupation as an explicit `Unknown` category to avoid dropping rows.
- Impute missing native country using the mode; any remaining numeric gaps use the median; remaining categorical gaps use `Unknown`.
- Do not indiscriminately delete extreme but potentially valid capital values. Log-transformed versions are added instead.

## Responsible-use note
The data is historical, US-specific, and contains demographic variables. It can encode historical social inequities. Do not use the dataset alone to make consequential decisions about real people. Document limitations in any analysis.
