'''
Created on Jan 29, 2018
'''
import sys
import os,signal
from os.path import join

class Config(object):
    '''
    Configuration database for RIAPS tools
    Including logging configuration
    '''

    #grafana Setup
    INFLUX_DBASE_HOST=os.getenv('INFLUX_DBASE_HOST')
    INFLUX_DBASE_PORT=os.getenv('INFLUX_DBASE_PORT')
    INFLUX_DBASE_NAME = os.getenv('INFLUX_DBASE_NAME')


    def __init__(self):
        '''
        Constructor
        '''
        pass
