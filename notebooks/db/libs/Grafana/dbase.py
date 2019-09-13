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
        self.INFLUX_DBASE_NAME='pouet'


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
