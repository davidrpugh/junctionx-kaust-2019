#%%

# from libs.Grafana.config import Config
# from libs.Grafana.dbase import Database
import datetime
import pandas as pd
import numpy as np
import datetime
import time



import logging

import pprint
from time import time
import requests

from influxdb import InfluxDBClient
from influxdb.client import InfluxDBClientError

import datetime
import random
import time

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


class Database(object):
    def __init__(self):
        super(Database, self).__init__()
        self.logger = logging.getLogger(__name__)

        self.INFLUX_DBASE_HOST='91.121.66.197'
        self.INFLUX_DBASE_PORT=8086
        self.INFLUX_DBASE_NAME='NEOM'


        print(f"INFLUX_DBASE_HOST: {str(self.INFLUX_DBASE_HOST)}, INFLUX_DBASE_PORT: {str(self.INFLUX_DBASE_PORT)},  INFLUX_DBASE_NAME: {str(self.INFLUX_DBASE_NAME)}")
        self.client = InfluxDBClient(self.INFLUX_DBASE_HOST,self.INFLUX_DBASE_PORT, self.INFLUX_DBASE_NAME)
        self.client.switch_database(self.INFLUX_DBASE_NAME)
        self.create()
        #---------------------------------------------------------------------------------
        self.yesterday = yesterday = datetime.date.today() - datetime.timedelta(days=1)
        #Yesterday at midnight
        self.yesterday0 = datetime.datetime.combine(self.yesterday, datetime.time.min)

    def create(self):
        try :
            self.client.create_database(self.INFLUX_DBASE_NAME)
        except requests.exceptions.ConnectionError as e:
            self.logger.warning("CONNECTION ERROR %s" %e)
            self.logger.warning("try again")


    def log(self,interval, obj,seriesName,value):
        records = []

        print(interval)

        now = self.yesterday0 + datetime.timedelta(minutes=15*interval)
        print(now)

        if value != None:
            try:
                floatValue = float(value)
            except:
                floatValue = None
        if floatValue != None:
            #---------------------------------------------------------------------------------
            record = {  "time": now,
                        "measurement":seriesName,
                        "tags" : { "object" : obj },
                        "fields" : { "value" : floatValue },
                        }
            records.append(record)
        self.logger.info("writing: %s" % str(records))
        try:
            res= self.client.write_points(records) # , retention_policy=self.retention_policy)
        except requests.exceptions.ConnectionError as e:
            self.logger.warning("CONNECTION ERROR %s" %e)
            self.logger.warning("try again")
            self.create()
        #---------------------------------------------------------------------------------

        # print (res)
        # assert res

    def post(self, now, tag_dict, seriesName, value):
        records = []

        if value != None:
            try:
                floatValue = float(value)
            except:
                floatValue = None
        if floatValue != None:
            #--------------------------------------------------------------------------------- - datetime.timedelta(days=1)
            record = {  "time": now ,
                        "measurement":seriesName,
                        "tags" : tag_dict,
                        "fields" : { "value" : floatValue },
                        }
            records.append(record)
        self.logger.info("writing: %s" % str(records))
        try:
            res= self.client.write_points(records) # , retention_policy=self.retention_policy)
        except requests.exceptions.ConnectionError as e:
            self.logger.warning("CONNECTION ERROR %s" %e)
            self.logger.warning("try again")
            self.create()

    def postArray(self, tag_dict, seriesName, values):
        records = []

        if values != None:
            for row in values:
                d = list(row.values())[0]
                f = list(row.values())[1]
                record = {  "time": d ,
                            "measurement":seriesName,
                            "tags" : tag_dict,
                            "fields" : { "value" : f },
                            }
                records.append(record)
        self.logger.info("writing: %s" % str(records))
        self.logger.info(f"len ======================> {str(len(records))}" )

        try:
            res= self.client.write_points(records) # , retention_policy=self.retention_policy)
        except requests.exceptions.ConnectionError as e:
            self.logger.warning("CONNECTION ERROR %s" %e)
            self.logger.warning("try again")
            self.create()


    def __destroy__(self):
        self.client.drop_database(self.INFLUX_DBASE_NAME)






#%%
dbase = Database()  


#%%

import pyowm
#https://github.com/csparpa/pyowm
#https://pyowm.readthedocs.io/en/latest/usage-examples-v2/weather-api-usage-examples.html#owm-weather-api-version-2-5-usage-examples
owm = pyowm.OWM('68f8c0b152aa2c29c1f6123f3cdb4760')  # You MUST provide a valid API key

# Have a pro subscription? Then use:
# owm = pyowm.OWM(API_key='your-API-key', subscription_type='pro')


# Search for current weather in London (Great Britain)
observation = owm.weather_at_place('London,GB')
w = observation.get_weather()

from datetime import datetime

now = datetime.now()

#%%


#%%


# load data
#this is an example to load today's forecast. we use historic value as we don't have them
# we assume it will come from weather api , like above
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
print(predictions)


#%%
dates = pd.date_range(start="2019-09-14", end="2019-09-14 23:00:00", freq='H')
print(dates)

#%%
values = []
for i, d in enumerate(dates):
    date = pd.to_datetime(d)
    value = predictions[0, i]
    values.append({"date": date, "value":value})
print(values)
#%%

dbase.postArray(tag_dict={"ID":"Predicted Solar Power (W/m2)"}, seriesName="csv_import", values=values)


#%%
