import requests

url = "http://127.0.0.1:5000/detect-anomalies"

data = {
    "data": [
        {"date": "2024-01-01", "meantemp": 30, "humidity": 60, "wind_speed": 5, "meanpressure": 1010},
        {"date": "2024-01-02", "meantemp": 32, "humidity": 58, "wind_speed": 4, "meanpressure": 1012},
        {"date": "2024-01-03", "meantemp": 29, "humidity": 65, "wind_speed": 3, "meanpressure": 1011},
        {"date": "2024-01-04", "meantemp": 45, "humidity": 10, "wind_speed": 0, "meanpressure": 950},
        {"date": "2024-01-05", "meantemp": 28, "humidity": 70, "wind_speed": 6, "meanpressure": 1013},
        {"date": "2024-01-06", "meantemp": 27, "humidity": 75, "wind_speed": 7, "meanpressure": 1012},
        {"date": "2024-01-07", "meantemp": 30, "humidity": 60, "wind_speed": 5, "meanpressure": 1010},
        {"date": "2024-01-08", "meantemp": 31, "humidity": 62, "wind_speed": 4, "meanpressure": 1011},
        {"date": "2024-01-09", "meantemp": 50, "humidity": 15, "wind_speed": 1, "meanpressure": 960},
        {"date": "2024-01-10", "meantemp": 29, "humidity": 64, "wind_speed": 3, "meanpressure": 1012},
        {"date": "2024-01-11", "meantemp": 30, "humidity": 90, "wind_speed": 0, "meanpressure": 1000},
        {"date": "2024-01-12", "meantemp": 31, "humidity": 85, "wind_speed": 2, "meanpressure": 1008}
    ]
}

response = requests.post(url, json=data)

print(response.json())
