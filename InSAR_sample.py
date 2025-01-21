import os
import cv2
import struct
import numpy as np
import json
import requests
import matplotlib.pyplot as plt
import math
from io import BytesIO
from matplotlib.projections import LambertAxes
import scipy
from scipy.interpolate import interp1d
import itertools

# myTOKEN
myTOKEN = "HzVMVJiCinAmtYWPetO1zpnKHuON1c3p"



if(0):
    print("hoge")
    exit()

BASE_API_URL = "https://www.tellusxdp.com/api/traveler/v1/datasets/{}/data-search/"
dataset_id = "8836ec9a-35b5-43c3-92be-263498061e91" # PALSAR_L1.1のプロダクトID
ACCESS_TOKEN = myTOKEN
HEADERS = {"Authorization": "Bearer " + ACCESS_TOKEN}
 
url = BASE_API_URL.format(dataset_id)
 
# 東京ディズニーリゾートの座標
lat = 35.631  # 緯度
lon = 139.883  # 経度

 # リクエストボディ
r_body= {
  "intersects": {
    "type": "Polygon",
    # 検索範囲を領域で指定
    "coordinates": [
        [
            [lon, lat],
            [lon+0.01, lat],
            [lon+0.01, lat+0.01],
            [lon, lat+0.01],
            [lon, lat],
        ]
    ]
  },
  "query": {
    # 検索期間の指定
    "start_datetime": {"gte": "2006-01-01T15:00:00Z"},
    "end_datetime": {"lte": "2011-03-18T12:31:12Z"},
  }
}

# 該当するデータの情報を取得
response = requests.post(url, json=r_body, headers=HEADERS)    # SSLError
res_json = response.json()

# JSONデータを保存
file_path = 'data.json'
with open(file_path, 'w') as json_file:
    json.dump(res_json, json_file, indent=4)


print("finished")
exit()

