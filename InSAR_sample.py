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
from pathlib import Path  # ファイル有無確認のために利用

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
print(f'(緯度,経度)=({lat},{lon})   # 東京ディズニーリゾート')

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

# 結果JSONの有無を確認
file_path = Path('./data.json')
# ファイルの存在を確認
if file_path.is_file():
    print(f"'{file_path}' はファイルとして存在します。HTTPリクエスト実行せず、既存JSONファイルを読み込む")
    # JSONデータを保存
    with open(file_path, 'r', encoding='utf-8') as json_file:
        res_json = json.load(json_file)
else:
    print(f"'{file_path}' はファイルとして存在しません。HTTPリクエスト実行します")
    # 該当するデータの情報を取得
    response = requests.post(url, json=r_body, headers=HEADERS)
    res_json = response.json()
    # JSONデータを保存
    file_path = 'data.json'
    with open(file_path, 'w') as json_file:
        json.dump(res_json, json_file, indent=4)

# 以降は res_json に結果が入っている前提の処理

print(f'#--- 観測データの確認 ---')
print(res_json.keys())
print(res_json["features"][0].keys())
# 取得した観測データの数
print(len(res_json["features"]))

# 0番目の観測データのプロパティを表示
for k, v in res_json["features"][0]["properties"].items():
    print("{:<30} +      {}".format(k, v))

print(f'#--- 可干渉な観測データIDのリスト ---')
data_list = []
for data in res_json["features"]:
  prop = data["properties"]
  if (
      prop["sat:relative_orbit"] == 406  # 衛星の軌道経路
      and prop["sat:orbit_state"] == "ascending"  # 衛星の進行方向
      and prop["sar:observation_direction"] == "right"  # 電波照射方向
      and prop["view:off_nadir"] == 34.3  # オフナディア角
      and prop["tellus:sat_frame"] == 700  # 観測範囲の中心位置
      and prop["sar:polarizations"] == "HH"  # 送受信の偏波  
      and prop["sar:instrument_mode"] == "H"  # 観測モード
      ):
    data_list.append(data["id"])

for data in data_list:
  print(data)

print(f'#--- 観測データのファイル情報の確認(HTTPリクエストあり) ---')
dataset_id = "8836ec9a-35b5-43c3-92be-263498061e91" # PALSAR_L1.1のプロダクトID
data_id = data_list[0]  # 0番目の観測データID

BASE_API_URL = "https://www.tellusxdp.com/api/traveler/v1/datasets/{}/data/{}/files/"
url = BASE_API_URL.format(dataset_id, data_id)

# 0番目の観測データのファイル情報を取得
response = requests.get(url, headers=HEADERS)
res_file_json = response.json()

for res_file in res_file_json["results"]:
  print(res_file)

print(f'#--- ファイルIDの取得(HTTPリクエストあり) ---')
def get_file_info(dataset_id, data_id, pol):
    BASE_API_URL = "https://www.tellusxdp.com/api/traveler/v1/datasets/{}/data/{}/files/"
    url = BASE_API_URL.format(dataset_id, data_id)
    response = requests.get(url, headers=HEADERS)
    res_file_json = response.json()
    for i in res_file_json["results"]:
      if i["name"].startswith(f"IMG-{pol}"):
        img = {str(i["id"]): i["name"]}
      elif i["name"].startswith("LED"):
        led = {str(i["id"]): i["name"]}
    files = [img, led]
    return files
 
file_id_list = []
pol = "HH"  # 送受信時の偏波
for data_id in data_list:
  file_id_list.append(get_file_info(dataset_id, data_id, pol))

for file_id in file_id_list:
  print(file_id)

print(f'#--- データのダウンロード(HTTPリクエストあり) ---')
def save_file(dataset_id, data_id_list, files_list):
    BASE_API_URL = "https://www.tellusxdp.com/api/traveler/v1/datasets/{}/data/{}/files/{}/download-url/"
    for data_id, files_data in zip(data_id_list, files_list):
      for file_data in files_data:
        file_id =int(list(file_data.keys())[0])
        file_name = list(file_data.values())[0]
        url = BASE_API_URL.format(dataset_id, data_id, file_id)
        # ダウンロードURLの発行
        response_post = requests.post(url, headers=HEADERS)
        dl_url = response_post.json()["download_url"]
        # 現在のディレクトリ下にファイルをダウンロード
        with open(file_name, "wb") as f:
          f.write(requests.get(dl_url).content)

save_file(dataset_id, data_list, file_id_list)


print("#--------------------------------")
print("finished")
exit()

