import requests

API_KEY = "f4d2be9f95f203032489e2b8f89c063c"
API_ENDPOINT = "https://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={API_KEY}"

def get_light_intensity(latitude, longitude):
    url = API_ENDPOINT.format(latitude=latitude, longitude=longitude, API_KEY=API_KEY)
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if "weather" in data:
            weather = data["weather"][0]
            if "id" in weather:
                weather_id = weather["id"]
                if weather_id >= 200 and weather_id < 600:
                    # Rain or storm
                    return 100
                elif weather_id >= 600 and weather_id < 700:
                    # Snow
                    return 50
                elif weather_id >= 700 and weather_id < 800:
                    # Fog or mist
                    return 20
                elif weather_id == 800:
                    # Clear sky
                    return 1000
                else:
                    # Other weather conditions
                    return 500
            else:
                return None
        else:
            return None
    else:
        return None

# Example usage
latitude = 37.7749
longitude = -122.4194
light_intensity = get_light_intensity(latitude, longitude)
if light_intensity:
    print("Light intensity in lux:", light_intensity)
else:
    print("Unable to fetch light intensity.")
