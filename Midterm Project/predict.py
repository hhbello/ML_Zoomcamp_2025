import requests

url = 'http://localhost:9696/predict'


strength =  {
    "cement": 462.0,
    "blast_furnace_slag": 113.5,
    "fly_ash": 45.5,
    "water": 212.0,
    "superplasticizer": 2.5,
    "coarse_aggregate": 1010.0,
    "fine_aggregate": 526.0,
    "age": 90.0
}


requests.post(url, json=strength).json()


