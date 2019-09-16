# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'notebooks'))
	print(os.getcwd())
except:
	pass
#%%
from IPython import get_ipython


#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model, ensemble, metrics, model_selection, preprocessing
import joblib


#%%
get_ipython().run_line_magic('matplotlib', 'inline')

#%% [markdown]
# # Predicting Solar Power Output at NEOM

#%%
neom_data = (pd.read_csv("../data/raw/neom-data.csv", parse_dates=[0])
               .rename(columns={"Unnamed: 0": "Timestamp"})
               .set_index("Timestamp", drop=True, inplace=False))


#%%
neom_data.info()


#%%
neom_data.head()


#%%
neom_data.tail()


#%%
neom_data.describe()


#%%
_ = neom_data.hist(bins=50, figsize=(20,15))


#%%
_hour = (neom_data.index
                  .hour
                  .rename("hour"))

hourly_averages = (neom_data.groupby(_hour)
                            .mean())

fig, ax = plt.subplots(1, 1)
_targets = ["GHI(W/m2)", "SWDIR(W/m2)", "SWDNI(W/m2)", "SWDIF(W/m2)"]
(hourly_averages.loc[:, _targets]
                .plot(ax=ax))
_ = ax.set_ylabel(r"$W/m^2$", rotation="horizontal")


#%%
months = (neom_data.index
                   .month
                   .rename("month"))
hours = (neom_data.index
                  .hour
                  .rename("hour"))

hourly_averages_by_month = (neom_data.groupby([months, hours])
                                     .mean())


#%%
fig, axes = plt.subplots(2, 6, sharex=True, sharey=True, figsize=(12, 6))

for month in months.unique():
    if month <= 6:
        (hourly_averages_by_month.loc[month, _targets]
                                 .plot(ax=axes[0, month - 1], legend=False))
        _ = axes[0, month - 1].set_title(month)
    else:
        (hourly_averages_by_month.loc[month, _targets]
                                 .plot(ax=axes[1, month - 7], legend=False))
        _ = axes[1, month - 7].set_title(month)
    
    if month - 1 == 0: 
        _ = axes[0, 0].set_ylabel(r"$W/m^2$")
    if month - 7 == 0: 
        _ = axes[1, 0].set_ylabel(r"$W/m^2$")
   

#%% [markdown]
# # Feature Engineering

#%%
_dropped_cols = ["SWDIR(W/m2)", "SWDNI(W/m2)", "SWDIF(W/m2)"]

_year = (neom_data.index
                  .year)
_month = (neom_data.index
                   .month)
_day = (neom_data.index
                 .dayofyear)
_hour = (neom_data.index
                  .hour)

features = (neom_data.drop(_dropped_cols, axis=1, inplace=False)
                     .assign(year=_year, month=_month, day=_day, hour=_hour)
                     .groupby(["year", "month", "day", "hour"])
                     .mean()
                     .unstack(level=["hour"])
                     .reset_index(inplace=False)
                     .sort_index(axis=1)
                     .drop("year", axis=1, inplace=False))


#%%
# want to predict the next 24 hours of "solar power"
efficiency_factor = 0.5

# square meters of solar cells required to generate 20 GW (231000 m2 will generate 7mW)
m2_of_solar_cells_required = 660000

target = (features.loc[:, ["GHI(W/m2)"]]
                  .mul(efficiency_factor)
                  .shift(-1)
                  .rename(columns={"GHI(W/m2)": "target(W/m2)"}))


#%%
input_data = (features.join(target)
                      .dropna(how="any", inplace=False)
                      .sort_index(axis=1))


#%%
input_data

#%% [markdown]
# # Train, Validation, Test Split

#%%
# use first eight years for training data...
training_data = input_data.loc[:8 * 365]

# ...next two years for validation data...
validation_data = input_data.loc[8 * 365 + 1:10 * 365 + 1]

# ...and final year for testing data!
testing_data = input_data.loc[10 * 365 + 2:]


#%%
training_data.shape


#%%
validation_data.shape


#%%
testing_data.shape

#%% [markdown]
# # Preprocessing the training and validation data

#%%
def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
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


#%%
training_features = training_data.drop("target(W/m2)", axis=1, inplace=False)
training_target = training_data.loc[:, ["target(W/m2)"]]
transformed_training_features = preprocess_features(training_features)

validation_features = validation_data.drop("target(W/m2)", axis=1, inplace=False)
validation_target = validation_data.loc[:, ["target(W/m2)"]]
transformed_validation_features = preprocess_features(validation_features)

#%% [markdown]
# # Find a few models that seem to work well
#%% [markdown]
# ## Linear Regression

#%%
# training a liner regression model
linear_regression = linear_model.LinearRegression()
linear_regression.fit(transformed_training_features, training_target)


#%%
# measure training error
_predictions = linear_regression.predict(transformed_training_features)
np.sqrt(metrics.mean_squared_error(training_target, _predictions))


#%%
# measure validation error
_predictions = linear_regression.predict(transformed_validation_features)
np.sqrt(metrics.mean_squared_error(validation_target, _predictions))


#%%
# user requests forecast for 1 January 2016 which we predict using data from 31 December 2015!
user_forecast_request = transformed_training_features[[-1], :]
user_forecast_response = linear_regression.predict(user_forecast_request)[0]
actual_values_response = training_target.values[[-1], :][0]

# this would be rendered in Tableau!
plt.plot(user_forecast_response, label="predicted")
plt.plot(actual_values_response, label="actual")
plt.legend()
plt.title("1 January 2016")
plt.ylabel("Solar Power (W/m2)")
plt.xlabel("Hour")

#%% [markdown]
# Linear regression is not bad but we an do better!
#%% [markdown]
# ## MultiTask ElasticNet Regression

#%%
# training a multi-task elastic net model
_prng = np.random.RandomState(42)
elastic_net = linear_model.MultiTaskElasticNet(random_state=_prng)
elastic_net.fit(transformed_training_features, training_target)


#%%
# measure training error
_predictions = elastic_net.predict(transformed_training_features)
np.sqrt(metrics.mean_squared_error(training_target, _predictions))


#%%
# measure validation error
_predictions = elastic_net.predict(transformed_validation_features)
np.sqrt(metrics.mean_squared_error(validation_target, _predictions))


#%%
# user requests forecast for 1 January 2016 which we predict using data from 31 December 2015!
user_forecast_request = transformed_training_features[[-1], :]
user_forecast_response = elastic_net.predict(user_forecast_request)[0]
actual_values_response = training_target.values[[-1], :][0]

# this would be rendered in Tableau!
plt.plot(user_forecast_response, label="predicted")
plt.plot(actual_values_response, label="actual")
plt.legend()
plt.title("1 January 2016")
plt.ylabel("Solar Power (W/m2)")
plt.xlabel("Hour")

#%% [markdown]
# ElasticNet is underfitting.
#%% [markdown]
# ## MultiTask Lasso Regression

#%%
# training a multi-task lasso model
_prng = np.random.RandomState(42)
lasso_regression = linear_model.MultiTaskLasso(random_state=_prng)
lasso_regression.fit(transformed_training_features, training_target)


#%%
# measure training error
_predictions = lasso_regression.predict(transformed_training_features)
np.sqrt(metrics.mean_squared_error(training_target, _predictions))


#%%
# measure validation error
_predictions = lasso_regression.predict(transformed_validation_features)
np.sqrt(metrics.mean_squared_error(validation_target, _predictions))


#%%
# user requests forecast for 1 January 2016 which we predict using data from 31 December 2015!
user_forecast_request = transformed_training_features[[-1], :]
user_forecast_response = lasso_regression.predict(user_forecast_request)[0]
actual_values_response = training_target.values[[-1], :][0]

# this would be rendered in Tableau!
plt.plot(user_forecast_response, label="predicted")
plt.plot(actual_values_response, label="actual")
plt.legend()
plt.title("1 January 2016")
plt.ylabel("Solar Power (W/m2)")
plt.xlabel("Hour")

#%% [markdown]
# Lasso Regression is underfitting.
#%% [markdown]
# ## Random Forest Regression

#%%
_prng = np.random.RandomState(42)
random_forest_regressor = ensemble.RandomForestRegressor(n_estimators=100, random_state=_prng, n_jobs=2)
random_forest_regressor.fit(transformed_training_features, training_target)


#%%
# measure training error
_predictions = random_forest_regressor.predict(transformed_training_features)
np.sqrt(metrics.mean_squared_error(training_target, _predictions))


#%%
# measure validation error
_predictions = random_forest_regressor.predict(transformed_validation_features)
np.sqrt(metrics.mean_squared_error(validation_target, _predictions))


#%%
# user requests forecast for 1 January 2016 which we predict using data from 31 December 2015!
user_forecast_request = transformed_training_features[[-1], :]
user_forecast_response = random_forest_regressor.predict(user_forecast_request)[0]
actual_values_response = training_target.values[[-1], :][0]

# this would be rendered in Tableau!
plt.plot(user_forecast_response, label="predicted")
plt.plot(actual_values_response, label="actual")
plt.legend()
plt.title("1 January 2016")
plt.ylabel("Solar Power (W/m2)")
plt.xlabel("Hour")

#%% [markdown]
# Random Forest with default parameters is over-fitting and needs to be regularized.
#%% [markdown]
# # Tuning hyper-parameters

#%%
from scipy import stats

#%% [markdown]
# ## MultiTask ElasticNet Regression

#%%
_prng = np.random.RandomState(42)

_param_distributions = {
    "l1_ratio": stats.uniform(),
    "alpha": stats.lognorm(s=1),
}

elastic_net_randomized_search = model_selection.RandomizedSearchCV(
    elastic_net,
    param_distributions=_param_distributions,
    scoring="neg_mean_squared_error",
    random_state=_prng,
    n_iter=10,
    cv=8,
    n_jobs=2,
    verbose=10
)

elastic_net_randomized_search.fit(transformed_training_features, training_target)


#%%
_ = joblib.dump(elastic_net_randomized_search.best_estimator_,
                "../models/tuned-elasticnet-regression-model.pkl")


#%%
elastic_net_randomized_search.best_estimator_


#%%
(-elastic_net_randomized_search.best_score_)**0.5


#%%
# user requests forecast for 1 January 2016 which we predict using data from 31 December 2015!
user_forecast_request = transformed_training_features[[-1], :]
user_forecast_response = elastic_net_randomized_search.predict(user_forecast_request)[0]
actual_values_response = training_target.values[[-1], :][0]

# this would be rendered in Tableau!
plt.plot(user_forecast_response, label="predicted")
plt.plot(actual_values_response, label="actual")
plt.legend()
plt.title("1 January 2017")
plt.ylabel("Solar Power (W/m2)")
plt.xlabel("Hour")


#%%
# user requests forecast for last week of 2015
user_forecast_request = transformed_training_features[-7:, :]
user_forecast_response = elastic_net_randomized_search.predict(user_forecast_request)
actual_values_response = training_target.values[-7:, :]

# this would be rendered in Tableau!
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.plot(user_forecast_response.flatten(), label="predicted")
ax.plot(actual_values_response.flatten(), label="actual")
ax.legend()
ax.set_title("Last week of December 2015")
ax.set_ylabel("Solar Power (W/m2)")
ax.set_xlabel("Hours")

#%% [markdown]
# ## MultiTask Lasso Regression

#%%
_prng = np.random.RandomState(42)

_param_distributions = {
    "alpha": stats.lognorm(s=1),
}

lasso_regression_randomized_search = model_selection.RandomizedSearchCV(
    lasso_regression,
    param_distributions=_param_distributions,
    scoring="neg_mean_squared_error",
    random_state=_prng,
    n_iter=10,
    cv=8,
    n_jobs=2,
    verbose=10
)

lasso_regression_randomized_search.fit(transformed_training_features, training_target)


#%%
_ = joblib.dump(lasso_regression_randomized_search.best_estimator_,
                "../models/tuned-lasso-regression-model.pkl")


#%%
(-lasso_regression_randomized_search.best_score_)**0.5


#%%
# user requests forecast for last week of 2015
user_forecast_request = transformed_training_features[-7:, :]
user_forecast_response = lasso_regression_randomized_search.predict(user_forecast_request)
actual_values_response = training_target.values[-7:, :]

# this would be rendered in Tableau!
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.plot(user_forecast_response.flatten(), label="predicted")
ax.plot(actual_values_response.flatten(), label="actual")
ax.legend()
ax.set_title("Last week of December 2015")
ax.set_ylabel("Solar Power (W/m2)")
ax.set_xlabel("Hours")

#%% [markdown]
# ## Random Forest Regressor

#%%
_prng = np.random.RandomState(42)

_param_distributions = {
    "n_estimators": stats.geom(p=0.01),
     "min_samples_split": stats.beta(a=1, b=99),
     "min_samples_leaf": stats.beta(a=1, b=999),
}

_cv = model_selection.TimeSeriesSplit(max_train_size=None, n_splits=5)

random_forest_randomized_search = model_selection.RandomizedSearchCV(
    random_forest_regressor,
    param_distributions=_param_distributions,
    scoring="neg_mean_squared_error",
    random_state=_prng,
    n_iter=10,
    cv=8,
    n_jobs=2,
    verbose=10
)

random_forest_randomized_search.fit(transformed_training_features, training_target)


#%%
_ = joblib.dump(random_forest_randomized_search.best_estimator_,
                "../models/tuned-random-forest-regression-model.pkl")


#%%
random_forest_randomized_search.best_estimator_


#%%
(-random_forest_randomized_search.best_score_)**0.5


#%%
# user requests forecast for 1 January 2016 which we predict using data from 31 December 2015!
user_forecast_request = transformed_training_features[[-1], :]
user_forecast_response = random_forest_randomized_search.predict(user_forecast_request)[0]
actual_values_response = training_target.values[[-1], :][0]

# this would be rendered in Tableau!
plt.plot(user_forecast_response, label="predicted")
plt.plot(actual_values_response, label="actual")
plt.legend()
plt.title("1 January 2016")
plt.ylabel("Solar Power (W/m2)")
plt.xlabel("Hour")
plt.savefig("../results/img/typical-actual-vs-predicted-solar-power.png")

#%% [markdown]
# # Assess model performance on testing data

#%%
testing_features = testing_data.drop("target(W/m2)", axis=1, inplace=False)
testing_target = testing_data.loc[:, ["target(W/m2)"]]
transformed_testing_features = preprocess_features(testing_features)


#%%
elastic_net_predictions = elastic_net_randomized_search.predict(transformed_testing_features)
np.sqrt(metrics.mean_squared_error(testing_target, elastic_net_predictions))


#%%
lasso_regression_predictions = lasso_regression_randomized_search.predict(transformed_testing_features)
np.sqrt(metrics.mean_squared_error(testing_target, lasso_regression_predictions))


#%%
# random forest wins!
random_forest_predictions = random_forest_randomized_search.predict(transformed_testing_features)
np.sqrt(metrics.mean_squared_error(testing_target, random_forest_predictions))


#%%
# user requests forecast for last week of 2018
user_forecast_request = transformed_testing_features[-7:, :]
user_forecast_response = random_forest_randomized_search.predict(user_forecast_request)
actual_values_response = testing_target.values[-7:, :]

# this would be rendered in Tableau!
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.plot(user_forecast_response.flatten(), label="predicted")
ax.plot(actual_values_response.flatten(), label="actual")
ax.legend()
ax.set_title("Last week of December 2018")
ax.set_ylabel("Solar Power (W/m2)")
ax.set_xlabel("Hours")


#%%
submission = (pd.DataFrame.from_dict({"Timestamp": pd.date_range("2018-01-01", end="2018-12-31 23:00:00", freq="H"),
                                     "Predicted Solar Power (W/m2)": random_forest_predictions.flatten(),
                                     "Actual Solar Power (W/m2)": testing_target.values.flatten()})
                          .set_index("Timestamp", inplace=False))


#%%
submission.info()


#%%
fig, ax = plt.subplots(1, 1, figsize=(20, 8))
_ = submission.plot(ax=ax)


#%%
submission.to_csv("../results/actual-vs-predicted-values-for-2018.csv", index=True)


#%%
# combine the training and validtion data
combined_training_features = pd.concat([training_features, validation_features])
transformed_combined_training_features = preprocess_features(combined_training_features)
combined_training_target = pd.concat([training_target, validation_target])

# tune a random forest regressor using CV ro avoid overfitting
_prng = np.random.RandomState(42)

_param_distributions = {
    "n_estimators": stats.geom(p=0.01),
     "min_samples_split": stats.beta(a=1, b=99),
     "min_samples_leaf": stats.beta(a=1, b=999),
}

tuned_random_forest_regressor = model_selection.RandomizedSearchCV(
    ensemble.RandomForestRegressor(n_estimators=100, random_state=_prng),
    param_distributions=_param_distributions,
    scoring="neg_mean_squared_error",
    random_state=_prng,
    n_iter=10,
    cv=8,
    n_jobs=2,
    verbose=10
)

tuned_random_forest_regressor.fit(combined_training_features, combined_training_target)


#%%
tuned_random_forest_regressor.best_estimator_


#%%
(-tuned_random_forest_regressor.best_score_)**0.5


#%%
# user requests forecast for last week of 2018
user_forecast_request = transformed_testing_features[-7:, :]
user_forecast_response = random_forest_randomized_search.predict(user_forecast_request)
actual_values_response = testing_target.values[-7:, :]

# this would be rendered in Tableau!
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.plot(user_forecast_response.flatten(), label="predicted")
ax.plot(actual_values_response.flatten(), label="actual")
ax.legend()
ax.set_title("Last week of December 2018")
ax.set_ylabel("Solar Power (W/m2)")
ax.set_xlabel("Hours")

#%% [markdown]
# # Forecasting the future of solar power at NEOM
# 
# Once the model is trained, the model can generate a new forecast for next day's solar power generation. Once actual values of solar power generation are observed, model can be automatically re-trained and improved.  Model can be retrained with weekly, monthly forecast horizons if longer forecasts are required.

#%%
incoming_features = features.loc[[4017]]
new_predictions = tuned_random_forest_regressor.predict(incoming_features)[0]
solar_power_forecast = (pd.DataFrame.from_dict({"Timestamp": pd.date_range(start="2019-01-01", end="2019-01-01 23:00:00", freq='H'),
                                                "Predicted Solar Power (W/m2)": new_predictions})
                          .set_index("Timestamp", inplace=False))


#%%
_ = solar_power_forecast.plot()


#%%



#%%
get_ipython().run_line_magic('pinfo', 'ensemble.RandomForestRegressor')


#%%



