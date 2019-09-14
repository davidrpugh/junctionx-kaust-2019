import joblib
import numpy as np
import pandas as pd
from sklearn import ensemble, preprocessing


MODEL = joblib.load("../models/tuned-random-forest-regression-model.pkl")


def preprocess_features(df):
    _numeric_features = ["GHI(W/m2)",
                         "mslp(hPa)",
                         "rain(mm)",
                         "rh(%)",
                         "t2(C)",
                         "td2(C)",
                         "wind_dir(Deg)",
                         "wind_speed(m/s)"]

    _ordinal_features = ["AOD",
                         "day",
                         "month",
                         "year"]

    standard_scalar = preprocessing.StandardScaler()
    Z0 = standard_scalar.fit_transform(df.loc[:, _numeric_features])
    
    ordinal_encoder = preprocessing.OrdinalEncoder()
    Z1 = ordinal_encoder.fit_transform(df.loc[:, _ordinal_features])
    transformed_features = np.hstack((Z0, Z1))
    
    return transformed_features


def feature_engineering(df):
    _dropped_cols = ["SWDIR(W/m2)", "SWDNI(W/m2)", "SWDIF(W/m2)"]

    _year = (df.index
               .year)
    _month = (df.index
                .month)
    _day = (df.index
              .dayofyear)
    _hour = (df.index
               .hour)

    features = (df.drop(_dropped_cols, axis=1, inplace=False)
                  .assign(year=_year, month=_month, day=_day, hour=_hour)
                  .groupby(["year", "month", "day", "hour"])
                  .mean()
                  .unstack(level=["hour"])
                  .reset_index(inplace=False)
                  .sort_index(axis=1)
                  .drop("year", axis=1, inplace=False))
    
    # create the proxy for our solar power target
    efficiency_factor = 0.5
    target = (features.loc[:, ["GHI(W/m2)"]]
                      .mul(efficiency_factor)
                      .shift(-1)
                      .rename(columns={"GHI(W/m2)": "target(W/m2)"}))

    # combine to create the input data
    input_data = (features.join(target)
                      .dropna(how="any", inplace=False)
                      .sort_index(axis=1))
    return input_data


if __name__ == "__main__":
    
    # load data
    neom_data = (pd.read_csv("../data/raw/neom-data.csv", parse_dates=[0])
                   .rename(columns={"Unnamed: 0": "Timestamp"})
                   .set_index("Timestamp", drop=True, inplace=False))
    
    # perform feature engineering
    input_data = feature_engineering(neom_data)
    
    # simulate online learning by sampling features from the input data
    _prng = np.random.RandomState(42)
    new_features = input_data.sample(n=1, random_state=_prng)
    
    # perform inference
    processed_features = preprocess_features(new_features)
    predictions = MODEL.predict(processed_features)
    
    # print the total solar power produced
    print(predictions.sum())