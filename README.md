# Traffic Congestion Predictor

A binary classifier that predicts whether traffic on major Lagos roads will be
**Normal** or **Congested**, given the road, time, and weather conditions.

Built and submitted to
the **3MTT x MIT Open Learning Universal AI Innovation Challenge**.

## Problem

Commuters on Lagos's major roads and expressways have no easy way to check
expected congestion before setting off. This project predicts congestion
from conditions known in advance — road, day, time of day, and weather — so
a commuter can decide whether to leave early or take an alternate route.

## Dataset

Initially it was a manually collected dataset of 248 rows taken across 7 days within fixed corridors using google maps which had a great imbalance of normal to congestion rate. Apython data generator was used to based on the initial dataset create more rows and bring about a balance to the data
`traffic_data_gen.csv` — 1,500 records across 8 major Lagos roads and
expressways (Third Mainland Bridge, Lekki-Epe Expressway, Ikorodu Road, and
others), each with a road, corridor, day, weather, temperature, rain
chance, and observation timestamp. The original 3-class target
(`low`/`medium`/`high`) was collapsed into a binary target: `Normal` (was
`low`) vs. `Congested` (was `medium` or `high`) — roughly a 33/67 split.

## Data leakage found and removed

Two columns in the raw data — `traffic_pattern` (a traffic-light-style
G/Y/R code) and `congestion_location` — turned out to near-perfectly encode
the target itself rather than predict it (e.g. `traffic_pattern = "R"` was
**always** `high`, `congestion_location = "none evident"` was **always**
`low`). Both were dropped before training. Their inclusion is what had
produced an initially inflated ~97–98% F1 score.

## Feature engineering: `obs_hour`

With the leaky columns removed, the remaining raw features
(`temperature`, `rain_chance`, `road`, `day`, `weather`) carried almost no
signal on their own (near-zero correlation with the target). Extracting
the **hour of day** from the observation timestamp and adding it as a
feature (`obs_hour`) revealed a strong, genuine rush-hour pattern —
07:00–09:00 and 15:00–19:00 are consistently congested, while late-morning
and late-night hours are consistently normal. This became the model's main
predictive feature.

`fixed_corridor` is a fixed 1:1 function of `road` (each road maps to
exactly one corridor), so it adds no independent signal but is kept in the
pipeline for consistency with the original schema.

## Model

**Logistic Regression** (scikit-learn `Pipeline` with a `ColumnTransformer`
for preprocessing — median imputation + scaling for numeric features,
most-frequent imputation + one-hot encoding for categorical features).
XGBoost was evaluated earlier but dropped in favor of the simpler,
equally-performing model.

**Result:** F1 = **0.86** on held-out test data — a legitimate score with
the leaky columns removed, built on real predictive signal (`obs_hour`)
rather than a restated label.

## Features used

| Type        | Columns                                    |
| ----------- | ------------------------------------------ |
| Numeric     | `temperature`, `rain_chance`, `obs_hour`   |
| Categorical | `road`, `fixed_corridor`, `day`, `weather` |

## App

A Streamlit interface (`app.py`) lets a user pick a road, day, time, and
weather and get a live prediction, including:

- Confidence-aware labeling (`Partially Congested` / `Partially Normal`
  when the model is close to a 50/50 split)
- A 24-hour forecast chart for the selected road/day/weather
- "What to prepare for" guidance based on the prediction
- A running history of recent predictions
- A one-click "use current day & time" shortcut

### Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Requires `traffic_binary_lr_pipeline.pkl` (the trained pipeline) in the
same folder as `app.py`.

## Tech stack

Python, scikit-learn, pandas, Streamlit, joblib.

## Author

Muhammad Misbahudeen — FUT Minna, Mechatronics Engineering — 3MTT/MIT OpenLearning
Cohort 1.
