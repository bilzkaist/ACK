import requests
import json
import openlocationcode as olc

def fetch_weather_data(latitude, longitude):
    # API key for OpenWeatherMap API
    API_KEY = "f4d2be9f95f203032489e2b8f89c063c"
    API_ENDPOINT2 = "https://api.openweathermap.org/data/2.5/onecall?lat={latitude}&lon={longitude}&exclude=hourly,daily&appid={API_KEY}"
    # API endpoint to retrieve weather data
    #API_EP = f"http://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={API_KEY}"
    #API_ENDPOINT2 = "https://api.openweathermap.org/data/2.5/onecall?lat={latitude}&lon={longitude}&exclude=hourly,daily&appid={API_KEY}"
    API_ENDPOINT = "https://api.openweathermap.org/data/2.5/weather?lat=36.3700&lon=127.3794&appid=8c22ed8f9769fae00d5e2ea4aaccf42b"
    # Make a GET request to the API endpoint
    response = requests.get(API_ENDPOINT)
    print("response.status_code = ",response.status_code)
    print("response.content = ", response.content)
    # Check if the request was successful
    if response.status_code == 200:
        # Retrieve the weather data from the response
        weather_data = json.loads(response.text)
        
        # Extract the required information from the weather data
        uv_index = weather_data["value"]
        light_sensor = weather_data["value"]
        pressure = weather_data["main"]["pressure"]
        temperature = weather_data["main"]["temp"]
        humidity = weather_data["main"]["humidity"]
        
        # Return the extracted information
        return uv_index, light_sensor, pressure, temperature, humidity
    else:
        # Return None if the request was unsuccessful
        return None




def get_latitude_longitude(plus_code):
    decoded = olc.decode(plus_code)
    latitude = decoded[0]
    longitude = decoded[1]
    return latitude, longitude

plus_code = "8Q899973+P2Q"
#latitude, longitude = get_latitude_longitude(plus_code)
latitude = 36.3700
longitude = 127.3794
print("Latitude:", latitude)
print("Longitude:", longitude)
print(fetch_weather_data(latitude, longitude))