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

BASE_API_URL = "https://www.tellusxdp.com/api/traveler/v1/datasets/{}/data-search/"
dataset_id = "8836ec9a-35b5-43c3-92be-263498061e91" # PALSAR_L1.1のプロダクトID
ACCESS_TOKEN = myTOKEN
HEADERS = {"Authorization": "Bearer " + ACCESS_TOKEN}
HEADERS_CTYPE = {"Authorization": "Bearer " + ACCESS_TOKEN, "Content-Type": "application/json",}
 
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
            [lon+0.001, lat],
            [lon+0.001, lat+0.001],
            [lon, lat+0.001],
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

r_body2= {
    "query":{
        "tellus:name":{"startsWith":""},
        "start_datetime":{"gte":"2006-01-01T00:00:00Z"},
        "end_datetime":{"lte":"2011-03-18T23:59:59Z"},
        "sat:orbit_state":{"eq":"descending"},
        "sar:observation_direction":{"eq":"right"},
        "view:off_nadir":{"gte":10.9,"lte":39.5},
        "sar:polarizations":{"eq":"HH+HV+VH+VV"}
    },
    "intersects":{
        "type":"Polygon",
        "coordinates":[
            [
                [139.86044298907964,35.61110126116198],
                [139.94879891945575,35.61110126116198],
                [139.9512063071534,35.74799218609996],
                [139.86090099836466,35.747094744617286],
                [139.86044298907964,35.672049719714366],
                [139.86044298907964,35.61110126116198]
            ]
        ]
    },
    "datasets":["8836ec9a-35b5-43c3-92be-263498061e91"],
    "sortby":[
        {"field":"properties.end_datetime","direction":"desc"}
    ],
    "only_downloadable_file":"true"
}

def get_pl(fpled,latlon):
    print(f'#--- 緯度・経度からピクセル座標への変換の関数 ---')
    fpled.seek(720+4096+0+4680+8192+9860+1620+0+1540000+4314000+345000+325000+325000+3072+511000+4370000+728000+15000+2064, 0)
    c = np.zeros(25,dtype="float64")
    d = np.zeros(25,dtype="float64")
    result = fpled.read(500)  # バイトNo.2065-2564の中を20バイト分ずつ、計500バイト分進む
    result_str = result.decode()
    arr_str = result_str.split()
    coeff_list = [float(num) for num in arr_str]
    for i in range(25):
      c[i] = coeff_list[i]
    result = fpled.read(500)  # バイトNo.2065-2564の中を20バイト分ずつ、計500バイト分進む
    result_str = result.decode()
    arr_str = result_str.split()
    coeff_list = [float(num) for num in arr_str]
    for i in range(25):
      d[i] = coeff_list[i]
    lat0 = float(fpled.read(20))  # 原点緯度(Φo)[度] 3065-3084を読み込み、20バイト分進む
    lon0 = float(fpled.read(20))  # 原点経度(Λo)[度] 3085-3104を読み込み、20バイト分進む
    phi = np.zeros(2, dtype="float64")
    lam = np.zeros(2, dtype="float64")
    phi[0] = latlon[0] - lat0
    phi[1] = latlon[1] - lat0
    lam[0] = latlon[2] - lon0
    lam[1] = latlon[3] - lon0
    pl = np.zeros(4,dtype="float64")
    for i in range(5):
      for j in range(5):
        id = i*5+j
        pl[0] += c[id]*lam[0]**(4-j) * phi[0]**(4-i)
        pl[1] += c[id]*lam[1]**(4-j) * phi[1]**(4-i)
        pl[2] += d[id]*lam[0]**(4-j) * phi[0]**(4-i)
        pl[3] += d[id]*lam[1]**(4-j) * phi[1]**(4-i)
    return pl

def gen_img(fpimg,off_col,col,off_row,row):
    print(f'#--- 強度・位相画像の出力 ---')
    # off_col,col,off_row,rowで、対象範囲を絞り込む
    cr = np.array([off_col,off_col+col,off_row,off_row+row], dtype="i4")
    # データセット（チャネル）当たりのライン数（境界を除く）
    fpimg.seek(236)
    nline = int(fpimg.read(8))
    nline = cr[3]-cr[2]  # row
    # 1ライン当たりのデータグループ（ピクセル）の数
    fpimg.seek(248)
    ncell = int(fpimg.read(8))
    # PALSAR-2の場合は、Prefixが544であることに注意
    prefix = 412
    nrec = prefix + ncell*8

    # シグナルデータレコードの前まで移動
    fpimg.seek(720)
    # SAR画像データを取得し、2次元配列に変換
    fpimg.seek(int((nrec/4)*(cr[2])*4))
    data = struct.unpack(">%s"%(int((nrec*nline)/4))+"f",fpimg.read(int(nrec*nline)))
    # 1次元配列を列数「nrec/4」で分割し、それぞれ行に変換する
    data = np.array(data).reshape(-1,int(nrec/4))
    # 全ての行に対して、列番号が「prefix/4」以降の全ての列を選択する
    data = data[:,int(prefix/4):]
    # 各行の偶数列の要素を実数部とし、奇数列の要素を虚数部として、複素数の配列を作成する
    data = data[:,::2] + 1j*data[:,1::2]
    data = data[:,cr[0]:cr[1]]

    # 位相成分と強度成分に分解
    CF = -83.0
    CF_offset = 32.0
    sigma = 10*np.log10(abs(data)) + CF - CF_offset
    phase = np.angle(data)
    # 画像の輝度を、0~255の範囲で正規化 & 8ビット符号なし整数のデータ型に変換する
    sigma = np.array(255*(sigma - np.amin(sigma))/(np.amax(sigma) - np.amin(sigma)), dtype="uint8")
    # ヒストグラムの均一化によってコントラストを向上させる
    sigma = cv2.equalizeHist(sigma)
    return sigma, phase

# 東京ディズニーリゾート周辺
latlon = np.array([35.644252, 35.621438, 139.897408, 139.871744], dtype="float64")
data_names = [
    "ALPSRP137552890",  # 観測データ0
    "ALPSRP144262890",  # 観測データ1
    "ALPSRP150972890",  # 観測データ2
    "ALPSRP191232890",  # 観測データ3
    "ALPSRP197942890",  # 観測データ4
    "ALPSRP244912890",  # 観測データ5
]
 
for data_name in data_names:
    fpimg = open(os.path.join("IMG-HH-" + data_name + "-P1.1__D"),mode='rb')
    fpled = open(os.path.join("LED-" + data_name + "-P1.1__D"),mode='rb')
    pl = np.array(np.ceil(get_pl(fpled, latlon)), dtype=np.int64)
    off_col = min(pl[0], pl[1])
    off_row = min(pl[2], pl[3])
    
    # ピクセル値を調整して強度・位相画像を出力
    rotate_count = 0
    sigma, phase = gen_img(fpimg, off_col-200, 1000, off_row-300, 1000)
    sigma = np.fliplr(sigma)  # 画像が反転しているため
    sigma = np.rot90(sigma, k=rotate_count)
    #sigma = np.rot90(sigma, k=-1)
    phase = np.fliplr(phase)  # 画像が反転しているため
    phase = np.rot90(phase, k=rotate_count)
    #phase = np.rot90(phase, k=-1)

    # 強度画像(sigma)と位相画像(phase)をそれぞれグレースケールとjetカラーマップで表示
    plt.imsave('sigma_{}.jpg'.format(data_name), sigma, cmap = "gray")
    plt.imsave('phase_{}.jpg'.format(data_name), phase, cmap = "jet")
    np.save('sigma_{}.npy'.format(data_name), sigma)
    np.save('phase_{}.npy'.format(data_name), phase)
    cv2.imshow("sigma",sigma)
    cv2.waitKey(0)
    #exit()

print("#--------------------------------")
print("finished")
cv2.destroyAllWindows()
exit()

