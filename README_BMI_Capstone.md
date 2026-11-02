# BMI Prediction Capstone

Predict BMI from physical and lifestyle attributes using a standard supervised-learning workflow: load, explore, preprocess, split, train, evaluate, tune and interpret. Five regression models are compared: Simple Linear Regression, Multiple Linear Regression, Ridge, Lasso and Elastic Net.

## Project files

| File | Description |
|---|---|
| `BMI_Capstone.ipynb` | Full notebook with code, outputs and plots. |
| `human_physical_attributes.csv` | Dataset (5,000 rows, 11 numeric columns). Keep it in the same folder as the notebook. |

## Dataset

| Type | Columns |
|---|---|
| Features (10) | `Height_cm`, `Weight_kg`, `Shoe_Size`, `Age`, `Exercise_Hours_Per_Week`, `Daily_Calorie_Intake`, `Steps_Per_Day`, `Body_Fat_Percentage`, `Water_Intake_Liters`, `Sleep_Hours_Per_Night` |
| Target | `BMI` |

No missing values and no duplicate rows were found.

## Requirements

- Python 3.9 or newer
- `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`
- `statsmodels` is installed in the notebook but not used by the later code, so it is optional.

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## How to run

1. Put `BMI_Capstone.ipynb` and `human_physical_attributes.csv` in one folder.
2. Open the notebook in Jupyter or PyCharm.
3. Run all cells from top to bottom. `DATA_PATH` near the top of the notebook must match the CSV file name.

## Workflow in the notebook

1. **Load and inspect**: shape, `head`, `info`, summary statistics, missing values, duplicates.
2. **EDA**: boxplots, histograms with KDE, correlation heatmap.
3. **Data quality**: negative values in exercise hours (329), steps (19) and water intake (35) are impossible, so they are set to 0. Outliers are checked with the IQR rule and capped (none were found after this step).
4. **Multicollinearity check**: feature-to-feature correlation; no pair above 0.8.
5. **Split and scale**: 80/20 train-test split (`random_state=42`); `StandardScaler` fitted on training data only.
6. **Models**: Simple LR (best single feature), Multiple LR, then Ridge, Lasso and Elastic Net with default `alpha=1`.
7. **Tuning**: R² compared across `alpha` = 0.0001 to 100 for each regularised model.
8. **Comparison and interpretation**: R² table and bar chart, coefficients, features removed by Lasso, train vs. test R² and RMSE, fitted line, actual vs. predicted and residual plots.

## Key results (test set)

| Model | R² test | RMSE test |
|---|---|---|
| Lasso (tuned, alpha 0.1) | 0.9573 | 1.256 |
| Ridge (tuned, alpha 100) | 0.9571 | 1.259 |
| Elastic Net (tuned, alpha 0.01) | 0.9570 | 1.260 |
| Multiple Linear Regression | 0.9568 | 1.262 |
| Simple Linear Regression (Weight_kg) | 0.7660 | 2.939 |

- `Weight_kg` (correlation 0.86 with BMI) and `Height_cm` (-0.47) are the only useful predictors. Lasso set the other 8 coefficients to exactly 0.
- Per raw unit, BMI rises about 0.34 per kg and falls about 0.28 per cm of height.
- Train and test R² are nearly equal (about 0.958 vs. 0.957), so there is no sign of overfitting.
- The four multi-feature models perform almost identically; the gain over Simple LR comes from adding `Height_cm`.
- With strong regularization (`alpha` of 10 or more), Lasso and Elastic Net collapse to R² of about 0 because every coefficient is shrunk to zero.

## Notes and limitations

- BMI is calculated from weight and height, so those two features naturally explain almost everything. The remaining features add essentially no information in this dataset.
- The best `alpha` is chosen by test-set R², so the reported test scores are slightly optimistic. A stricter approach is to tune with `GridSearchCV` or cross-validation on the training set and use the test set only once at the end. Differences between the tuned models are tiny (0.9568 to 0.9573), so the conclusions would not change.
- Small differences in decimals can appear with other library versions.
