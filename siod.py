import numpy as np
import folium
import pandas as pd
import plotly.graph_objects as go

path = "S:\\IOD\\Data1\\"

""" df = pd.read_csv(path + 'Accelerometer.csv')
df.index = pd.to_datetime(df['time'], unit = 'ns')

fig = go.Figure()

for axis in ['x', 'y', 'z']:
    fig.add_trace(go.Scatter(x = df.index, y = df[axis], name = axis))

fig.show()

df = pd.read_csv(path + 'Location.csv')
df.index = pd.to_datetime(df['time'], unit='ns')

fig = go.Figure()

fig.add_trace(go.Scatter(x=df.index, y=df['speed'], mode='markers',
              error_y={'type': 'data', 'array': df['speedAccuracy']}))

fig.show() """

df = pd.read_csv(path +"Location.csv")
coords = [(row.latitude, row.longitude) for _, row in df.iterrows()]

my_map = folium.Map(location=[df.latitude.mean(), df.longitude.mean()], zoom_start=16)
folium.PolyLine(coords, color="blue", weight=5.0).add_to(my_map)



# approx 
# Magnetometer readings
magnetometer = np.array([-1.875, -3.3, -39.4688])

# Barometer readings
barometer = 1013.25

# Light sensor readings
light_sensor = 800

# GPS data (longitude, latitude, speed, altitude)
gps_data = np.array([-122.35, 37.87, 0.0, 30.0])

# Threshold values for each sensor
mag_threshold = 0.5
bar_threshold = 1000
light_threshold = 500
speed_threshold = 5

# Initialize the indoor/outdoor detection result
result = "Indoor"

# Check if the magnetometer readings are above the threshold
if np.linalg.norm(magnetometer) > mag_threshold:
    result = "Outdoor"

# Check if the barometer readings are above the threshold
if barometer > bar_threshold:
    result = "Outdoor"

# Check if the light sensor readings are above the threshold
if light_sensor > light_threshold:
    result = "Outdoor"

# Check if the speed in the GPS data is above the threshold
if gps_data[2] > speed_threshold:
    result = "Outdoor"

# Print the final indoor/outdoor detection result
print("Result:", result)
