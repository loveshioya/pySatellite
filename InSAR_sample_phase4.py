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

def wraptopi(delta_phase):
    # 値の範囲が、[-π~+π] – [-π~+π] = [-2π~+2π]となるので、それを[-π~+π]に戻す（ラップする）
    delta_phase = delta_phase - np.floor(delta_phase/(2*np.pi))*2*np.pi - np.pi
    return delta_phase


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


# 観測中の衛星位置を取得する関数
def get_sat_pos(led,img):
  # シグナルデータレコードのセンサ取得ミリ秒を取得する
  img.seek(720+44)
  time_start = struct.unpack(">i",img.read(4))[0]/1000
  # データセットサマリレコードのレーダ波長(m)を取得する
  led.seek(720+500)
  lam = float(led.read(16)) #m
  # プラットフォーム位置データレコードのデータポイント数を取得する
  led.seek(720+4096+0+140)
  position_num = int(led.read(4))
  # プラットフォーム位置データレコードのポイント間のインターバル時間(秒=60)を取得する
  led.seek(720+4096+0+182)
  time_interval = int(float(led.read(22)))
  # プラットフォーム位置データレコードの第一ポイントの通算秒を取得する
  led.seek(720+4096+0+160)
  start_time = float(led.read(22))
  # データセットサマリレコードのシーンセンタ時刻を取得する
  led.seek(720+68)
  center_time = led.read(32)
  
  Hr = float(center_time[8:10])*3600
  Min = float(center_time[10:12])*60
  Sec = float(center_time[12:14])
  msec = float(center_time[14:17])*1e-3
  center_time = Hr+Min+Sec+msec
  time_end = time_start + (center_time - time_start)*2
  
  # SARイメージファイルディスクリプタのデータセット(チャネル)当たりのライン数を取得する
  img.seek(236)
  nline = int(img.read(8))
  
  time_obs = np.arange(time_start, time_end, (time_end - time_start)/nline)
  time_pos = np.arange(start_time, start_time+time_interval*position_num, time_interval)
  pos_ary = [];
 
  # プラットフォーム位置データレコードのデータポイント位置ベクトル(x,y,z/m)を、データポイント数分取得する。
  # データポイントとは、衛星が1枚の画像を撮影している間(約10秒間)の、衛星の観測点である。
  # その10秒間で衛星は100km程度移動するため、適切に補完を行い、画像の各ピクセルが取得されたときの衛星の位置を特定する。
  led.seek(720+4096+0+386)
  for i in range(position_num):
      for j in range(3):
          pos = float(led.read(22))
          pos_ary.append(pos)
      led.read(66)
  pos_ary = np.array(pos_ary).reshape(-1,3)
 
  fx = scipy.interpolate.interp1d(time_pos,pos_ary[:,0],kind="cubic")
  fy = scipy.interpolate.interp1d(time_pos,pos_ary[:,1],kind="cubic")
  fz = scipy.interpolate.interp1d(time_pos,pos_ary[:,2],kind="cubic")
  X = fx(time_obs)
  Y = fy(time_obs)
  Z = fz(time_obs)
  XYZ = np.zeros(nline*3).reshape(-1,3)
  XYZ[:,0] = X
  XYZ[:,1] = Y
  XYZ[:,2] = Z
 
  return XYZ

# 軌道縞を生成する関数
def get_orbit_stripe(pos1,pos2,pl,led):
    # データセットサマリレコードのレーダ波長(m)を取得する
    led.seek(720+500)
    lam = float(led.read(16))

    a = []
    b = []
    c = []
    # 設備関連データ11のピクセルとラインを緯度と経度に変換する8次多項式の係数を取得（p.3-83）
    # PALSAR-2とPALDAR-1では、設備関連データの番号が異なるので注意
    led.seek(720+4096+0+4680+8192+9860+1620+0+(1540000+4314000+345000+325000+325000+3072+511000+4370000+728000+15000+1024), 0)
    for i in range(25):
        a.append(float(led.read(20)))
    for i in range(25):
        b.append(float(led.read(20)))
    # 原点ピクセルと原点ラインを取得
    for i in range(2):
        c.append(float(led.read(20)))
    npix = abs(pl[0]-pl[1])
    nline = abs(pl[2]-pl[3])

    orb = np.zeros((nline, npix))
    for i in range(npix):
        if i % 100 == 0:
            continue
        for j in range(nline):
            px = i+pl[0]-c[0]
            ln = j+pl[2]-c[1]
            ilat = (a[0]*ln**4 + a[1]*ln**3 + a[2]*ln**2 + a[3]*ln + a[4])*px**4 + (a[5]*ln**4 + a[6]*ln**3 + a[7]*ln**2 + a[8]*ln + a[9])*px**3 + (a[10]*ln**4 + a[11]*ln**3 + a[12]*ln**2 + a[13]*ln + a[14])*px**2 + (a[15]*ln**4 + a[16]*ln**3 + a[17]*ln**2 + a[18]*ln + a[19])*px + a[20]*ln**4 + a[21]*ln**3 + a[22]*ln**2 + a[23]*ln + a[24]
            ilon = (b[0]*ln**4 + b[1]*ln**3 + b[2]*ln**2 + b[3]*ln + b[4])*px**4 + (b[5]*ln**4 + b[6]*ln**3 + b[7]*ln**2 + b[8]*ln + b[9])*px**3 + (b[10]*ln**4 + b[11]*ln**3 + b[12]*ln**2 + b[13]*ln + b[14])*px**2 + (b[15]*ln**4 + b[16]*ln**3 + b[17]*ln**2 + b[18]*ln + b[19])*px + b[20]*ln**4 + b[21]*ln**3 + b[22]*ln**2 + b[23]*ln + b[24]
            ixyz = lla2ecef(ilat*np.pi/180.0,ilon*np.pi/180.0,0)
            r1 = np.linalg.norm(pos1[j+pl[2],:] - ixyz);
            r2 = np.linalg.norm(pos2[j+pl[2],:] - ixyz);
            orb[j,i] = wraptopi(2*np.pi/lam*2*(r2-r1));
    return orb
 
# 緯度・経度高度から地球のECEF座標系(地球を基準とした直交座標系)のXYZを出力する関数
def lla2ecef(lat, lon, alt):
    # 地球の長半径
    a = 6378137.0
    # 地球の偏平率
    f = 1 / 298.257223563
    # 第一離心率の2乗(e2)
    e2 = 1 - (1 - f) * (1 - f)
    # 子午線曲率半径
    v = a / math.sqrt(1 - e2 * math.sin(lat) * math.sin(lat))
 
    x = (v + alt) * math.cos(lat) * math.cos(lon)
    y = (v + alt) * math.cos(lat) * math.sin(lon)
    z = (v * (1 - e2) + alt) * math.sin(lat)
    return np.array([x,y,z])

pos = []
fpled = []


# 軌道縞の除去
def del_orbit_stripe(ifgm, orbit_stripe, X, Y):
  # 軌道縞の画像サイズをインターフェログラムのサイズに合わせる
  orbit_stripe = orbit_stripe[0:ifgm.shape[0], 0:ifgm.shape[1]]
  
  # 軌道縞の除去
  ifgm_sub_orb = wraptopi(ifgm - orbit_stripe)  
  plt.figure()
  plt.imsave('ifgm_sub_orb{}_{}.jpg'.format(X, Y), ifgm_sub_orb, cmap = "jet")
  np.save('ifgm_sub_orb{}_{}.npy'.format(X, Y), ifgm_sub_orb)
 
def get_SAR_sub_orbit(A,B):
    plAB = np.array(np.ceil(get_pl(fpled[A], latlon)), dtype=np.int64)
    off_colAB = min(plAB[0], plAB[1])
    off_rowAB = min(plAB[2], plAB[3])
    x = 1000
    plAB = [off_colAB, off_colAB + x, off_rowAB, off_rowAB + x]
 
    # pos[i]は画像iの衛星の座標
    if(1):
       print(f'A:{A}')
       print(f'A:{B}')
       print(f'posA:{pos[0]}')
       print(f'posB:{pos[1]}')
    orbit_stripeAB = get_orbit_stripe(pos[A], pos[B], plAB, fpled[A])
    # 元画像の変換に合わせて、軌道縞も回転・反転させる
    rotate_count = 0
    orbit_stripeAB = np.fliplr(orbit_stripeAB)
    orbit_stripeAB = np.rot90(orbit_stripeAB, k=rotate_count)
    #orbit_stripeAB = np.rot90(orbit_stripeAB, k=-1)
    
 
    np.save('orbit_stripe{}_{}.npy'.format(A,B), orbit_stripeAB)
    plt.imsave('orbit_stripe{}_{}.jpg'.format(A,B), orbit_stripeAB, cmap = "jet")
    ifgmAB = np.load('ifgm{}_{}.npy'.format(A,B))
    del_orbit_stripe(ifgmAB, orbit_stripeAB, A, B)
    if(1):
       cv2.imshow("orbit_stripeAB",orbit_stripeAB)
       cv2.waitKey(0)
       cv2.destroyAllWindows()

# 軌道縞の生成と除去
select_data_names = [
data_names[0],  # ALPSRP104950700
data_names[3],  # ALPSRP212310700
data_names[5],  # ALPSRP219020700
]
for i, data_name in enumerate(select_data_names):
  fpimg = (open(os.path.join("IMG-HH-" + data_name + "-P1.1__D"),mode='rb'))
  fpled.append(open(os.path.join("LED-" + data_name + "-P1.1__D"),mode='rb'))
  pos.append(get_sat_pos(fpled[i], fpimg))
 
combinations = list(itertools.combinations(range(len(data_names)), 2))
for combination in combinations:
  get_SAR_sub_orbit(int(combination[0]), int(combination[1]))

# 地形縞の除去
def get_insar(pairs):
    if(1):
       print(f'pairs[0]:{pairs[0]}')
       print(f'pairs[1]:{pairs[1]}')
    pair1 = pairs[0]
    pair2 = pairs[1]
    ifgm_pair1 = np.load('ifgm_sub_orb{}_{}.npy'.format(pair1[0], pair1[1]))
    ifgm_pair2 = np.load('ifgm_sub_orb{}_{}.npy'.format(pair2[0], pair2[1]))

    # 地形縞の除去
    # 色彩調整用の + np.pi*2/4.0 および-ifgm_trueは画像ペアによって適宜修正
    ifgm_true = wraptopi(ifgm_pair1[0:999, 0:999] - ifgm_pair2[0:999, 0:999] + np.pi*2/4.0)
    ifgm_true = -ifgm_true
 
    plt.figure()
    plt.imsave('D-InSAR_{}_{}_{}_{}.jpg'.format(pair1[0], pair1[1], pair2[0], pair2[1]), ifgm_true, cmap = "jet")
    plt.imshow(ifgm_true, cmap = "jet")
    plt.colorbar()

# 地形縞の除去
pairs = [[0,3], [3,5]]  # +2/4 -> 全体を-
get_insar(pairs)



print("#--------------------------------")
print("finished")
exit()

