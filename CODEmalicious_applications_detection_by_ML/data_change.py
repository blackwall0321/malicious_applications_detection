
import pandas as pd
import math
from collections import Counter

def insert_zero_1(path):
  all_info_csv = pd.read_csv(path)
  all_info_csv.fillna(0,inplace=True)
  all_info_csv.to_csv(path,index=False)
  print('ocer 1')
def max_num(target):
    aar = Counter(target)
    max = 0
    for i in aar:
        if aar[i] > max:
            max = aar[i]
    return max
def process_col_3(path,col_num):
    df = pd.read_csv(path)
    data = df.values
    col_size = df.shape[1]
    #print(col_size)
    delete = []
    for key in data:
        sum = 0
        #print(key)
        for i in key[1:]:
            if int(i) > 0:
                #print(i)
                sum +=1

        if sum < col_size * col_num:
            delete.append(key[0])
    print(len(delete))
    df = pd.read_csv(path,index_col='app_name')
    for i in delete:
        print(i)
        df.drop(i,axis=0,inplace=True)
    df.to_csv(path, index=True)

def process_row_2(path,out_path,row_num):
  df = pd.read_csv(path,index_col='app_name')
  row_size = len(df)
  columns = list(df.columns)
  col_size = len(columns)
  col_true_size = (row_size-1) * row_num
  for key in df:
      if max_num(df[key]) >col_true_size:
          print(key)
          df.drop(key,axis=1,inplace=True)
  df.to_csv(out_path, index=True)
def process_all(path,out_path,row_num,col_num):
    process_row_2(path,out_path,row_num)
    process_col_3(path,col_num)

def calcMean(x, y):# 计算特征和类的平均值
  sum_x = sum(x)
  sum_y = sum(y)
  n = len(x)
  x_mean = float(sum_x + 0.0) / n
  y_mean = float(sum_y + 0.0) / n
  return x_mean, y_mean
def calcPearson(x, y):# 计算Pearson系数
  x_mean, y_mean = calcMean(x, y)
  n = len(x)
  sumTop = 0.0
  sumBottom = 0.0
  x_pow = 0.0
  y_pow = 0.0
  for i in range(n):
      sumTop += (x[i] - x_mean) * (y[i] - y_mean)
  for i in range(n):
      x_pow += math.pow(x[i] - x_mean, 2)
  for i in range(n):
      y_pow += math.pow(y[i] - y_mean, 2)
  sumBottom = math.sqrt(x_pow * y_pow)
  p = sumTop / sumBottom
  return (p)
def corr_4(path):
  df = pd.read_csv(path)
  columns_name = list(df.columns)
  columns_name.remove('safe_or_bad')
  columns_name.remove('app_name')
  #cache = columns_name.copy()
  for n in columns_name[0:int(len(columns_name) / 2)]:
      print(n,'============')
      for m in columns_name[int(len(columns_name) / 2):]:
          print(m)
          if n != m:
              p = calcPearson(df[n], df[m])
              # 筛选皮尔森相关系数大于0.7的
              if abs(p) > 0.7:
                  print(n, m)
                  # 去掉其中一个权限
                  df.drop([n],axis=1)
  #print(cache)
  #df = df[cache]
  df.to_csv(path,index=False)


csv_path = 'E:\Droid-LMAD\source_code\\table\\DSDM_1.csv'
out_path = 'E:\Droid-LMAD\source_code\\table\\DSDM_2_pro.csv'
#insert_zero_1(csv_path)

process_all(csv_path,out_path,0.995,0.08)


#corr_4(csv_path)
