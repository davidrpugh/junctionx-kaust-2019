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
print(w)

#%%
# Weather details
print(w.get_wind()     )             # {'speed': 4.6, 'deg': 330}
print(w.get_humidity()  )            # 87
print(w.get_temperature('celsius'))  # {'temp_max': 10.5, 'temp': 9.7, 'temp_min': 9.0}

#%%
#neom gps 
#observation = owm.weather_around_coords(28.239313, 34.736139)
#%%

#w = observation[0].get_weather()
#%%


print(w.get_reference_time(timeformat='date'))
print(w.get_clouds())
print(w.get_rain())
print(w.get_wind())
print(w.get_humidity())
print(w.get_pressure())
print(w.get_temperature(unit='celsius'))
print(w.get_sunrise_time())
print(w.get_sunset_time('iso'))

# 2019-09-13 20:09:52+00:00
# 0
# {}
# {'speed': 6.2, 'deg': 90}
# 72
# {'press': 1035, 'sea_level': None}
# {'temp': 14.03, 'temp_max': 17.0, 'temp_min': 11.11, 'temp_kf': None}
# 1568352700
# 2019-09-13 18:21:49+00
#%%
#riyad
from datetime import datetime

now = datetime.now()
print(now)
#%%

#updating humidity historian
values = []
date = now
value = w.get_humidity()
values.append({"date": date, "value":value})

#%%
values

#%%

#this will insert in influxDB the humidity 
dbase.postArray(tag_dict={"ID":"rh(%)"}, seriesName="csv_import", values=values)

#do the same with all other mertrics ...