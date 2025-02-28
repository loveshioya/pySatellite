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


# 画像の位置合わせと干渉処理を行う関数
def coregistration(A,B,C,D):
    # 画像の位置合わせを行う
    # A, Bが強度画像、C, Dが位相画像
    py = len(A)
    px = len(A[0])
    A = np.float64(A)
    B = np.float64(B)
    # 強度画像A, Bを用いて変位量を求める
    d, etc = cv2.phaseCorrelate(B,A)
    dx, dy = d  # 画像の変位量
    print(f"画像の変位量:{d}")
    # 位相画像C, Dをその変位量でカットする
    if dx < 0 and dy >= 0:
        dx = math.ceil(dx)-1
        dy = math.ceil(dy)
        rB = B[dy:py,0:px+dx]
        rA = A[dy:py,0:px+dx]
        rD = D[dy:py,0:px+dx]
        rC = C[dy:py,0:px+dx]
    elif dx < 0 and dy < 0:
        dx = math.ceil(dx)-1
        dy = math.ceil(dy)-1
        rB = B[0:py+dy,0:px+dx]
        rA = A[0:py+dy,0:px+dx]
        rD = D[0:py+dy,0:px+dx]
        rC = C[0:py+dy,0:px+dx]
    elif dx >= 0 and dy < 0:
        dx = math.ceil(dx)
        dy = math.ceil(dy)-1
        rB = B[0:py+dy,dx:px]
        rA = A[0:py+dy,dx:px]
        rD = D[0:py+dy,dx:px]
        rC = C[0:py+dy,dx:px]
    elif dx >= 0 and dy >= 0:
        dx = math.ceil(dx)
        dy = math.ceil(dy)-1
        rB = B[dy:py,dx:px]
        rA = A[dy:py,dx:px]
        rD = D[dy:py,dx:px]
        rC = C[dy:py,dx:px]
 
    return rA, rB, rC, rD
    #return rC, rD
 
def wraptopi(delta_phase):
    # 値の範囲が、[-π~+π] – [-π~+π] = [-2π~+2π]となるので、それを[-π~+π]に戻す（ラップする）
    delta_phase = delta_phase - np.floor(delta_phase/(2*np.pi))*2*np.pi - np.pi
    return delta_phase
 
def get_ifgm(data_A, data_B, AB):
    file_name_sigma = "sigma_{}.npy"
    file_name_phase = "phase_{}.npy"
    sigma1 = np.load(file_name_sigma.format(data_A))
    sigma2 = np.load(file_name_sigma.format(data_B))
 
    phase1 = np.load(file_name_phase.format(data_A))
    phase2 = np.load(file_name_phase.format(data_B))
    print(f'{data_A}:{sigma1.shape},{phase1.shape}')
    print(f'{data_B}:{sigma2.shape},{phase2.shape}')
    min_y = min(sigma1.shape[0] , sigma2.shape[0])
    min_x = min(sigma1.shape[1] , sigma2.shape[1])
    print(f'({min_y},{min_x})')
    if(0):
       return
 
    # crop min-size
    sigma1_crop = sigma1[0:min_y , 0:min_x]
    phase1_crop = phase1[0:min_y , 0:min_x]
    sigma2_crop = sigma2[0:min_y , 0:min_x]
    phase2_crop = phase2[0:min_y , 0:min_x]
    print(f'croped:{sigma1_crop.shape},{phase1_crop.shape},{sigma2_crop.shape},{phase2_crop.shape}')
    #coreg_phase2, coreg_phase1 = coregistration(sigma1, sigma2, phase1, phase2)
    #coreg_phase2, coreg_phase1 = coregistration(sigma1_crop, sigma2_crop, phase1_crop, phase2_crop)
    coreg_sigma1, coreg_sigma2,coreg_phase1, coreg_phase2 = coregistration(sigma1_crop, sigma2_crop, phase1_crop, phase2_crop)
    diff_sigma1_2 = coreg_sigma1 - coreg_sigma2
    diff_sigma2_1 = coreg_sigma2 - coreg_sigma1
    if(1):
        cv2.imshow("coreg_sigma1",coreg_sigma1)
        cv2.imshow("coreg_sigma2",coreg_sigma2)        
        cv2.imshow("diff_sigma1_2",diff_sigma1_2)
        cv2.imshow("diff_sigma2_1",diff_sigma2_1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
       
    # 位相画像の差分を、[-π~+π]にラップする
    ifgm = wraptopi(coreg_phase2 - coreg_phase1)
 
    np.save('ifgm{}.npy'.format(AB), ifgm)
    plt.imsave('ifgm{}.jpg'.format(AB), ifgm, cmap = "jet")
    cv2.imshow("ifgm",ifgm)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

# 干渉処理の実行
combinations = list(itertools.combinations(range(len(data_names)), 2))
for combination in combinations:
    AB = str(combination[0]) + "_" + str(combination[1])
    data_A = data_names[int(combination[0])]
    data_B = data_names[int(combination[1])]
    get_ifgm(data_A, data_B, AB)



print("#--------------------------------")
print("finished")
exit()

