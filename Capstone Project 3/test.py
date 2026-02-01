import requests

url = 'http://localhost:9696/predict'

sample_data = {
    'X_Minimum': 42.0,
    'X_Maximum': 50.0,
    'Y_Minimum': 270900.0,
    'Y_Maximum': 270944.0,
    'Pixels_Areas': 267.0,
    'X_Perimeter': 17.0,
    'Y_Perimeter': 44.0,
    'Sum_of_Luminosity': 24220.0,
    'Minimum_of_Luminosity': 76.0,
    'Maximum_of_Luminosity': 108.0,
    'Length_of_Conveyer': 1687.0,
    'TypeOfSteel_A300': 1.0,
    'TypeOfSteel_A400': 0.0,
    'Steel_Plate_Thickness': 80.0,
    'Edges_Index': 0.1739,
    'Empty_Index': 0.25,
    'Square_Index': 0.1818,
    'Outside_X_Index': 0.0047,
    'Edges_X_Index': 0.4706,
    'Edges_Y_Index': 1.0,
    'Outside_Global_Index': 1.0,
    'Log_X_Index': 0.9031,
    'Log_Y_Index': 1.6435,
    'LogOfAreas': 2.4265,
    'Orientation_Index': 0.8182,
    'Luminosity_Index': -0.2913,
    'SigmoidOfAreas': 0.5822
}

try:
    response = requests.post(url, json=sample_data)
    response.raise_for_status()
    print("Response from API:")
    print(response.json())
except Exception as e:
    print(f"Error connecting to API: {e}")
