
#%%


# from libs.Grafana.config import Config
# from libs.Grafana.dbase import Database
import datetime
import pandas as pd
import numpy as np
import datetime
import time





'''
Created on Feb 20, 2017

@author: riaps
'''
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

data_dir = './data/raw/'
fname = os.path.join(data_dir, 'neom-data.csv')

df = pd.read_csv(fname, header = 0, error_bad_lines=False)
#df = df.rename(columns={'Unnamed: 0': 'date'})
df.columns = df.columns.str.replace('Unnamed: 0','date')
df['date'] = pd.to_datetime(df['date'], format='%m/%d/%y %H:%M')
df['Day'] = df["date"].dt.dayofyear
df['Hour'] = df["date"].dt.hour
columns = [
 'mslp(hPa)',
 't2(C)',
 'td2(C)',
 'wind_speed(m/s)',
 'wind_dir(Deg)',
 'rh(%)',
 'GHI(W/m2)',
 'SWDIR(W/m2)',
 'SWDNI(W/m2)',
 'SWDIF(W/m2)',
 'rain(mm)',
 'AOD']#'date',
#%%
for i in range(1,df.shape[0]):
    print(f" date:{df.iloc[i]['date']} ")#, temp:{df[i]['t2(C)']}
    date = df.iloc[i]['date']
    for c in columns:
        dbase.post(now=date, tag_dict={"ID":c}, seriesName="csv_import", value=df.iloc[i][c])
    
#%%

values_mslp = []
for i in range(1,df.shape[0]):
    date = df.iloc[i]['date']
    value = df.iloc[i]["mslp(hPa)"]
    values_mslp.append({"date": date, "value":value})


#%%
dbase.postArray(tag_dict={"ID":"mslp(hPa)"}, seriesName="csv_import", values=values_mslp)


#%%

columns = [
 "mslp(hPa)",
 't2(C)',
 'td2(C)',
 'wind_speed(m/s)',
 'wind_dir(Deg)',
 'rh(%)',
 'GHI(W/m2)',
 'SWDIR(W/m2)',
 'SWDNI(W/m2)',
 'SWDIF(W/m2)',
 'rain(mm)',
 'AOD']
for c in columns:
    values = []
    for i in range(1,df.shape[0]):
        date = df.iloc[i]['date']
        value = df.iloc[i][c]
        values.append({"date": date, "value":value})
    print(f"inserting column {c}")
    dbase.postArray(tag_dict={"ID":c}, seriesName="csv_import", values=values)
    

#%%
