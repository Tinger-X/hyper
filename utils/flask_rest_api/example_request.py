"""Perform test request"""
import pprint

import requests

DETECTION_URL = "http://localhost:5000/v1/object-detection/yolov5s"
TEST_IMAGE = r"D:\Project\Python\local\hyper-main\S2ADet\runs\detect\exp7\pad_rgb_0001_ir.jpg"

image_data = open(TEST_IMAGE, "rb").read()

response = requests.post(DETECTION_URL, files={"image": image_data}).json()

pprint.pprint(response)
