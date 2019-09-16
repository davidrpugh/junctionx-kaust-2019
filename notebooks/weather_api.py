#%%

import pyowm
#https://github.com/csparpa/pyowm
#https://pyowm.readthedocs.io/en/latest/usage-examples-v2/weather-api-usage-examples.html#owm-weather-api-version-2-5-usage-examples
owm = pyowm.OWM('5f28df836a35fc458e6270eb8f01ec86')  # You MUST provide a valid API key

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
observation = owm.weather_around_coords(28.239313, 34.736139)
#%%

w = observation[0].get_weather()
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
uvi = owm.uvindex_around_coords(24.670646, 46.928074)


#%%
uvi.get_value()
#

# uvi.get_value()...
# 12.04

#%%
uvi.get_reference_time(timeformat='date')

#%%

# weather script every hours 
# prediction every day