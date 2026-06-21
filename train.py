import pandas as pd
import numpy as np
from datetime import date
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt 
import xgboost as xgb
import warnings
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")  
# 数据文件路径
off_train = pd.read_csv('/kaggle/input/datasets/strickailkes/o2odata/ccf_offline_stage1_train/ccf_offline_stage1_train.csv',keep_default_na=True)
off_train.columns=['user_id','merchant_id','coupon_id','discount_rate','distance','date_received','date']
off_test = pd.read_csv('/kaggle/input/datasets/strickailkes/o2odata/ccf_offline_stage1_test_revised.csv',keep_default_na=True)
off_test.columns = ['user_id','merchant_id','coupon_id','discount_rate','distance','date_received']
on_train = pd.read_csv('/kaggle/input/datasets/strickailkes/o2odata/ccf_online_stage1_train/ccf_online_stage1_train.csv',keep_default_na=True)
on_train.columns = ['user_id','merchant_id','action','coupon_id','discount_rate','date_received','date']
off_train[['user_id','merchant_id','coupon_id']]=off_train[['user_id','merchant_id','coupon_id']].astype(str)
off_test[['user_id','merchant_id','coupon_id']]=off_test[['user_id','merchant_id','coupon_id']].astype(str)
on_train[['user_id','merchant_id','coupon_id']]=on_train[['user_id','merchant_id','coupon_id']].astype(str)


print(off_train.isnull().any())
print(off_train.isnull().sum()/len(off_train))

fd_seperator = ':'

def convert_discount_rate(rate):
    """将折扣率转换为减免金额"""
    if pd.isna(rate):
        return np.nan
    if ':' in rate:
        x, y = rate.split(':')
        return float(y)  # 返回减免金额
    else:
        return 0  # 如果是直接折扣率，没有减免金额

def is_target_coupon(rate):
    """判断减免金额是否在0到50元之间"""
    rate = convert_discount_rate(rate)
    return 0 <= rate <= 50

def get_discount_rate(s):
    s = str(s)
    if s==np.nan:
        return -1
    s = s.split(fd_seperator)
    if len(s) == 1:
        return float(s[0])
    else:
        return round((1.0-float(s[1])/float(s[0])),3)

#获取是否满减（full reduction promotion）
def get_if_fd(s):
    s = str(s)
    s = s.split(fd_seperator)
    if len(s)==1:
        return 0
    else:
        return 1

#获取满减的条件
def get_full_value(s):
    s = str(s)
    s = s.split(fd_seperator)
    if len(s)==1:
        #return 'null'
        return np.nan
    else:
        return int(s[0])

#获取满减的优惠     
def get_reduction_value(s):
    s = str(s)
    s = s.split(fd_seperator)
    if len(s) == 1:
        #return 'null'
        return np.nan
    else:
        return int(s[1])

#获取日期间隔，输入内容为Date_received:Date
def get_day_gap(s):
    s = s.split(fd_seperator)
    if s[0]=='nan':
        return -1
    if s[1]=='nan':
        return -1
    else:    
        return (date(int(s[0][0:4]),int(s[0][4:6]),int(s[0][6:8])) - date(int(s[1][0:4]),int(s[1][4:6]),int(s[1][6:8]))).days


#获取Label，输入内容为Date:Date_received
def get_label(s):
    s = s.split(fd_seperator)
    if s[0]=='nan':
        return 0
    if s[1]=='nan':
        return -1
    elif (date(int(s[0][0:4]),int(s[0][4:6]),int(s[0][6:8]))-date(int(s[1][0:4]),int(s[1][4:6]),int(s[1][6:8]))).days<=15:
        return 1
    else:
        return 0

#增加折扣相关特征
def add_discount(df):
    df['if_fd']=df['discount_rate'].apply(get_if_fd)
    df['full_value']=df['discount_rate'].apply(get_full_value)
    df['reduction_value']=df['discount_rate'].apply(get_reduction_value)
    df['discount_rate']=df['discount_rate'].apply(get_discount_rate)
    df.distance=df.distance.replace('null',np.nan)
    return df

#计算日期间隔  
def add_day_gap(df):
    df['day_gap']=df['date'].astype('str') + ':' +  df['date_received'].astype('str')
    df['day_gap']=df['day_gap'].apply(get_day_gap)
    return df

#获取label
def add_label(df):
    df['label']=df['date'].astype('str') + ':' +  df['date_received'].astype('str')
    df['label']=df['label'].apply(get_label)
    return df


def is_firstlastone(x):
    if x==0:
        return 1
    elif x>0:
        return 0
    else:
        #return -1
        return np.nan

def get_day_gap_before(s):
    date_received,dates = s.split('-')
    dates = dates.split(':')
    gaps = []
    for d in dates:
        #将时间差转化为天数
        if date_received[0:4]=='nan':
            continue
        this_gap = (dt.date(int(date_received[0:4]),int(date_received[4:6]),int(date_received[6:8]))-dt.date(int(d[0:4]),int(d[4:6]),int(d[6:8]))).days
        if this_gap>0:
            gaps.append(this_gap)
    if len(gaps)==0:
        #return -1
        return np.nan
    else:
        return min(gaps)
    
def get_day_gap_after(s):
    date_received,dates = s.split('-')
    dates = dates.split(':')
    gaps = []
    for d in dates:
        if date_received[0:4]=='nan':
            continue
        this_gap = (dt.datetime(int(d[0:4]),int(d[4:6]),int(d[6:8]))-dt.datetime(int(date_received[0:4]),int(date_received[4:6]),int(date_received[6:8]))).days
        if this_gap>0:
            gaps.append(this_gap)
    if len(gaps)==0:
        return np.nan
    else:
        return min(gaps)
    
def get_date_feature(dataset):
    data = dataset.copy()
    data['date_received'] = pd.to_datetime(data['date_received'], format='%Y%m%d')
    dataset['day'] = data['date_received'].dt.day
    dataset['day_of_week'] = data['date_received'].dt.dayofweek
    dataset['day_of_week'] =  dataset['day_of_week'].astype(float)
    one_hot_encoded = pd.get_dummies(dataset['day_of_week'], prefix='day')
    dataset = pd.concat([dataset, one_hot_encoded], axis=1)
    dataset = dataset.drop(['day_of_week'], axis=1)
    return dataset

def get_merchant_feature(feature):
    merchant = feature[['merchant_id','coupon_id','distance','date_received','date']].copy()
    merchant_id = merchant[['merchant_id']].copy()
    #删除重复行数据
    merchant_id.drop_duplicates(inplace=True)
    
    new_feature_name = 'merchant_id_sales_use_coupon'
    merchant_id_sales_use_coupon = merchant[(merchant.date.notna())&(merchant.coupon_id != 'nan')][['merchant_id']].copy()
    merchant_id_sales_use_coupon[new_feature_name] = 1
    merchant_id_sales_use_coupon = merchant_id_sales_use_coupon.groupby('merchant_id').agg('sum').reset_index()
    merchant_id = merchant_id.merge(merchant_id_sales_use_coupon, on='merchant_id', how="left")
    merchant_id.merchant_id_sales_use_coupon = merchant_id.merchant_id_sales_use_coupon.fillna(0)

    new_feature_name = 'merchant_id_total_coupon'
    merchant_id_total_coupon = merchant[merchant.coupon_id!='nan'][['merchant_id']].copy()
    merchant_id_total_coupon[new_feature_name] = 1
    merchant_id_total_coupon = merchant_id_total_coupon.groupby('merchant_id').agg('sum').reset_index()
    merchant_id = merchant_id.merge(merchant_id_total_coupon, on='merchant_id', how="left")
    merchant_id.total_coupon = merchant_id.merchant_id_total_coupon.fillna(0)

    merchant_id_distance = merchant[(merchant.date.notna())&(merchant.coupon_id!='nan')&(merchant.distance.notna())][['merchant_id','distance']].copy()
    merchant_id_distance.distance=merchant_id_distance.distance.astype('int')
    merchant_id_distance = merchant_id_distance.groupby('merchant_id')['distance'].agg(['min', 'max', 'mean','std','skew']).reset_index()
    new_column_names = {'merchant_id':'merchant_id','min':'merchant_id_distance_min', 'max':'merchant_id_distance_max','mean': 'merchant_id_distance_mean','std':'merchant_id_distance_std','skew':'merchant_id_distance_skew'}
    merchant_id_distance = merchant_id_distance.rename(columns=new_column_names)
    merchant_id = merchant_id.merge(merchant_id_distance, on='merchant_id', how="left")
    merchant_id.merchant_id_sales_use_coupon = merchant_id.merchant_id_sales_use_coupon.fillna(0)
    
    first_received = merchant.groupby('merchant_id')['date_received'].transform('min')
    last_received = merchant.groupby('merchant_id')['date_received'].transform('max')
    
    # 创建标记列
    merchant_id['first_received_flag'] = (merchant['date_received'] == first_received).astype(int)
    merchant_id['last_received_flag'] = (merchant['date_received'] == last_received).astype(int)

    # 按 user_id 和 coupon_id 分组，找到每个用户第一次和最后一次领取某张优惠券的日期
    first_coupon_received = merchant.groupby(['merchant_id', 'coupon_id'])['date_received'].transform('min')
    last_coupon_received = merchant.groupby(['merchant_id', 'coupon_id'])['date_received'].transform('max')
    
    # 创建标记列，判断是否为第一次或最后一次领取某张优惠券
    merchant_id['is_first_coupon_received'] = (merchant['date_received'] == first_coupon_received).astype(int)
    merchant_id['is_last_coupon_received'] = (merchant['date_received'] == last_coupon_received).astype(int)
    
    #优惠券的使用率
    merchant_id['merchant_coupon_transfer_rate'] = merchant_id.merchant_id_sales_use_coupon.astype('float')/merchant_id.merchant_id_total_coupon
    # merchant_id['coupon_rate'] = merchant_id.merchant_id_sales_use_coupon.astype('float') / merchant_id.merchant_id_total_sales
    return merchant_id

def get_user_feature(feature):
    user = feature[['user_id','merchant_id','coupon_id','discount_rate','distance','date_received','date']]
    user_feature = user[['user_id']].copy()
    user_feature.drop_duplicates(inplace=True)
    
    new_feature_name = 'user_id_count_merchant'
    user_id_count_merchant = user[user.date.notna()][['user_id','merchant_id']].copy()
    user_id_count_merchant.drop_duplicates(inplace=True)
    user_id_count_merchant = user_id_count_merchant[['user_id']]
    user_id_count_merchant[new_feature_name] = 1
    user_id_count_merchant = user_id_count_merchant.groupby('user_id').agg('sum').reset_index()
    user_feature = user_feature.merge(user_id_count_merchant, on='user_id', how="left")
    user_feature.user_id_count_merchant = user_feature.user_id_count_merchant.fillna(0)

    user_id_distance = user[(user.date.notna())&(user.coupon_id!='nan')&(user.distance.notna())][['user_id','distance']]
    user_id_distance.distance=user_id_distance.distance.astype('int')
    user_id_distance = user_id_distance.groupby('user_id')['distance'].agg(['min', 'max', 'mean', 'median','std','skew']).reset_index()
    new_column_names = {'user_id':'user_id','min':'user_id_distance_min', 'max':'user_id_distance_max','mean': 'user_id_distance_mean','median':'user_id_distance_median','std':'user_id_distance_std','skew':'user_id_distance_skew'}
    user_id_distance = user_id_distance.rename(columns=new_column_names)
    user_feature = user_feature.merge(user_id_distance, on='user_id', how="left")
    
    new_feature_name = 'user_id_buy_use_coupon'
    user_id_buy_use_coupon = user[(user.date.notna())&(user.coupon_id!='nan')][['user_id']]
    user_id_buy_use_coupon[new_feature_name] = 1
    user_id_buy_use_coupon = user_id_buy_use_coupon.groupby('user_id').agg('sum').reset_index()
    user_feature = user_feature.merge(user_id_buy_use_coupon, on='user_id', how="left")
    user_feature.user_id_buy_use_coupon = user_feature.user_id_buy_use_coupon.fillna(0)
    
    new_feature_name = 'user_id_coupon_received'
    user_id_coupon_received = user[user.coupon_id!='nan'][['user_id']]
    user_id_coupon_received[new_feature_name] = 1
    user_id_coupon_received = user_id_coupon_received.groupby('user_id').agg('sum').reset_index()
    user_feature = user_feature.merge(user_id_coupon_received, on='user_id', how="left")
    
    user_id_day_gap = user[(user.date_received.notna())&(user.date.notna())][['user_id','date_received','date']]
    user_id_day_gap = add_day_gap(user_id_day_gap)
    user_id_day_gap = user_id_day_gap[['user_id','day_gap']]
    user_id_day_gap = user_id_day_gap.groupby('user_id')['day_gap'].agg(['min', 'max', 'mean','skew']).reset_index()
    new_column_names = {'user_id':'user_id','min':'user_id_day_gap_min', 'max':'user_id_day_gap_max','mean': 'user_id_day_gap_mean','skew':'user_id_day_gap_skew'}
    user_id_day_gap = user_id_day_gap.rename(columns=new_column_names)
    user_feature = user_feature.merge(user_id_day_gap, on='user_id', how="left")

    first_received = user.groupby('user_id')['date_received'].transform('min')
    last_received = user.groupby('user_id')['date_received'].transform('max')
    
    # 创建标记列
    user_feature['first_received_flag'] = (user['date_received'] == first_received).astype(int)
    user_feature['last_received_flag'] = (user['date_received'] == last_received).astype(int)

    # 按 user_id 和 coupon_id 分组，找到每个用户第一次和最后一次领取某张优惠券的日期
    first_coupon_received = user.groupby(['user_id', 'coupon_id'])['date_received'].transform('min')
    last_coupon_received = user.groupby(['user_id', 'coupon_id'])['date_received'].transform('max')
    
    # 创建标记列，判断是否为第一次或最后一次领取某张优惠券
    user_feature['is_first_coupon_received'] = (user['date_received'] == first_coupon_received).astype(int)
    user_feature['is_last_coupon_received'] = (user['date_received'] == last_coupon_received).astype(int)

    ###############
    user_feature['user_coupon_transfer_rate'] = user_feature.user_id_buy_use_coupon.astype('float') / user_feature.user_id_coupon_received.astype('float')
    user_feature.user_id_coupon_received = user_feature.user_id_coupon_received.fillna(0)
    return user_feature

def get_user_merchant_feature(feature):
    user_merchant_feature = feature[['user_id','merchant_id']].copy()
    user_merchant_feature.drop_duplicates(inplace=True)
    
    new_feature_name = 'user_merchant_buy_total'
    user_merchant_buy_total = feature[['user_id','merchant_id','date']].copy()
    user_merchant_buy_total = user_merchant_buy_total[user_merchant_buy_total.date.notna()][['user_id','merchant_id']]
    user_merchant_buy_total[new_feature_name] = 1
    user_merchant_buy_total = user_merchant_buy_total.groupby(['user_id','merchant_id']).agg('sum').reset_index()
    user_merchant_feature = user_merchant_feature.merge(user_merchant_buy_total, on=['user_id','merchant_id'], how="left")

    new_feature_name = 'user_merchant_buy_use_coupon'
    user_merchant_buy_use_coupon = feature[['user_id','merchant_id','date','date_received']]
    user_merchant_buy_use_coupon = user_merchant_buy_use_coupon[(user_merchant_buy_use_coupon.date.notna())&(user_merchant_buy_use_coupon.date_received.notna())][['user_id','merchant_id']]
    user_merchant_buy_use_coupon[new_feature_name] = 1
    user_merchant_buy_use_coupon = user_merchant_buy_use_coupon.groupby(['user_id','merchant_id']).agg('sum').reset_index()
    user_merchant_feature = user_merchant_feature.merge(user_merchant_buy_use_coupon, on=['user_id','merchant_id'], how="left")
    user_merchant_feature.user_merchant_buy_use_coupon = user_merchant_feature.user_merchant_buy_use_coupon.fillna(0)

    new_feature_name = 'user_merchant_any'
    user_merchant_any = feature[['user_id','merchant_id']]
    user_merchant_any[new_feature_name] = 1
    user_merchant_any = user_merchant_any.groupby(['user_id','merchant_id']).agg('sum').reset_index()
    user_merchant_feature = user_merchant_feature.merge(user_merchant_any, on=['user_id','merchant_id'], how="left")
    
    new_feature_name = 'user_merchant_buy_common'
    user_merchant_buy_common = feature[['user_id','merchant_id','date','coupon_id']]
    user_merchant_buy_common = user_merchant_buy_common[(user_merchant_buy_common.date.notna())&(user_merchant_buy_common.coupon_id=='nan')][['user_id','merchant_id']]
    user_merchant_buy_common[new_feature_name] = 1
    user_merchant_buy_common = user_merchant_buy_common.groupby(['user_id','merchant_id']).agg('sum').reset_index()
    user_merchant_feature = user_merchant_feature.merge(user_merchant_buy_common, on=['user_id','merchant_id'], how="left")
    user_merchant_feature.user_merchant_buy_common = user_merchant_feature.user_merchant_buy_common.fillna(0)

    # user_merchant_feature['user_merchant_coupon_transfer_rate'] = user_merchant_feature.user_merchant_buy_use_coupon.astype('float') / user_merchant_feature.user_merchant_received.astype('float')
    user_merchant_feature['user_merchant_coupon_buy_rate'] = user_merchant_feature.user_merchant_buy_use_coupon.astype('float') / user_merchant_feature.user_merchant_buy_total.astype('float')
    user_merchant_feature['user_merchant_rate'] = user_merchant_feature.user_merchant_buy_total.astype('float') / user_merchant_feature.user_merchant_any.astype('float')
    user_merchant_feature['user_merchant_common_buy_rate'] = user_merchant_feature.user_merchant_buy_common.astype('float') / user_merchant_feature.user_merchant_buy_total.astype('float')
    return user_merchant_feature


def get_coupon_feature(feature):
    coupon_data = feature[['coupon_id','discount_rate','date_received','date','distance']].copy()
    coupon_feature = coupon_data[['coupon_id']].copy()
    coupon_feature.drop_duplicates(inplace=True)
    
    new_feature_name = 'coupon_type_count'
    coupon_type_count = coupon_data[(coupon_data.coupon_id!='nan') & coupon_data.date_received.notna()][['coupon_id']]
    coupon_type_count[new_feature_name] = 1
    coupon_type_count = coupon_type_count.groupby('coupon_id')[new_feature_name].agg('sum').reset_index()
    coupon_feature = coupon_feature.merge(coupon_type_count, on='coupon_id', how="left")

    new_feature_name = 'coupon_type_used_count'
    coupon_type_used_count = coupon_data[(coupon_data.coupon_id!='nan') & (coupon_data.date_received.notna()) & (coupon_data.date.notna())][['coupon_id']]
    coupon_type_used_count[new_feature_name] = 1
    coupon_type_used_count = coupon_type_used_count.groupby('coupon_id')[new_feature_name].agg('sum').reset_index()
    coupon_feature = coupon_feature.merge(coupon_type_used_count, on='coupon_id', how="left")
    coupon_feature.coupon_type_used_count = coupon_feature.coupon_type_used_count.fillna(0)

    
    coupon_data = add_day_gap(coupon_data)
    new_feature_name = 'coupon_type_used_15_count'
    coupon_type_used_15_count = coupon_data[(coupon_data.coupon_id!='nan')  & (coupon_data.date.notna()) & (coupon_data.day_gap<=15) & (coupon_data.day_gap>=0)][['coupon_id']]
    coupon_type_used_15_count[new_feature_name] = 1
    coupon_type_used_15_count = coupon_type_used_15_count.groupby('coupon_id')[new_feature_name].agg('sum').reset_index()
    coupon_feature = coupon_feature.merge(coupon_type_used_15_count, on='coupon_id', how="left")
    coupon_feature.coupon_type_used_15_count = coupon_feature.coupon_type_used_15_count.fillna(0)
    
    coupon_id_distance = coupon_data[(coupon_data.date.notna())&(coupon_data.coupon_id!='nan')&(coupon_data.distance.notna())][['coupon_id','distance']].copy()
    coupon_id_distance.distance=coupon_id_distance.distance.astype('int')
    coupon_id_distance = coupon_id_distance.groupby('coupon_id')['distance'].agg(['min', 'max', 'mean']).reset_index()
    new_column_names = {'coupon_id':'coupon_id','min':'coupon_id_distance_min', 'max':'coupon_id_distance_max','mean': 'coupon_id_distance_mean'}
    coupon_id_distance = coupon_id_distance.rename(columns=new_column_names)
    coupon_feature = coupon_feature.merge(coupon_id_distance, on='coupon_id', how="left")
    ##########
    coupon_id_day_gap = coupon_data[(coupon_data.date_received.notna())&(coupon_data.date.notna())][['coupon_id','day_gap']]
    coupon_id_day_gap = coupon_id_day_gap.groupby('coupon_id')['day_gap'].agg(['min', 'max', 'mean']).reset_index()
    new_column_names = {'coupon_id':'coupon_id','min':'coupon_id_day_gap_min', 'max':'coupon_id_day_gap_max','mean': 'coupon_id_day_gap_mean'}
    coupon_id_day_gap = coupon_id_day_gap.rename(columns=new_column_names)
    coupon_feature = coupon_feature.merge(coupon_id_day_gap, on='coupon_id', how="left")
    ###########
    coupon_feature['coupon_type_used_15_count_rate'] = coupon_feature['coupon_type_used_15_count']/coupon_feature['coupon_type_count']
    coupon_feature['coupon_type_used_15_count_rate'].fillna(0, inplace=True)

    coupon_feature['coupon_type_used_count_rate'] = coupon_feature['coupon_type_used_count']/coupon_feature['coupon_type_count']
    coupon_feature['coupon_type_used_count_rate'].fillna(0, inplace=True)


    return coupon_feature


def get_user_coupon_feature(feature):
    user_coupon_data = feature[['user_id','coupon_id','date_received','date']].copy()
    user_coupon_feature = user_coupon_data[['user_id','coupon_id']].copy()
    user_coupon_feature.drop_duplicates(inplace=True)

    new_feature_name = 'user_coupon_type_count'
    user_coupon_type_count = user_coupon_data[(user_coupon_data.coupon_id!='nan') & (user_coupon_data.date_received.notna())][['user_id','coupon_id']]
    user_coupon_type_count[new_feature_name] = 1
    user_coupon_type_count = user_coupon_type_count.groupby(['user_id','coupon_id'])[new_feature_name].agg('sum').reset_index()
    user_coupon_feature = user_coupon_feature.merge(user_coupon_type_count, on=['user_id','coupon_id'], how="left")
    
    new_feature_name = 'user_used_coupon_type_count'
    user_used_coupon_type_count = user_coupon_data[(user_coupon_data.coupon_id!='nan') & (user_coupon_data.date.notna())][['user_id','coupon_id']]
    user_used_coupon_type_count[new_feature_name] = 1
    user_used_coupon_type_count = user_used_coupon_type_count.groupby(['user_id','coupon_id'])[new_feature_name].agg('sum').reset_index()
    user_coupon_feature = user_coupon_feature.merge(user_used_coupon_type_count, on=['user_id','coupon_id'], how="left")
    user_coupon_feature.user_used_coupon_type_count = user_coupon_feature.user_used_coupon_type_count.fillna(0)

    user_coupon_feature['user_used_coupon_type_count_rate'] = user_coupon_feature['user_used_coupon_type_count']/user_coupon_feature['user_coupon_type_count']
    
    user_coupon_data = add_day_gap(user_coupon_data)
    new_feature_name = 'user_used_coupon_type_used_15_count'
    user_used_coupon_type_used_15_count = user_coupon_data[(user_coupon_data.coupon_id!='nan')  & (user_coupon_data.date.notna()) & (user_coupon_data.day_gap<=15) & (user_coupon_data.day_gap>=0)][['user_id','coupon_id']]
    user_used_coupon_type_used_15_count[new_feature_name] = 1
    user_used_coupon_type_used_15_count = user_used_coupon_type_used_15_count.groupby(['user_id','coupon_id'])[new_feature_name].agg('sum').reset_index()
    user_coupon_feature = user_coupon_feature.merge(user_used_coupon_type_used_15_count, on=['user_id','coupon_id'], how="left")

    user_coupon_feature['user_used_coupon_type_used_15_count_rate'] = user_coupon_feature['user_used_coupon_type_used_15_count']/user_coupon_feature['user_coupon_type_count']

    user_coupon_day_gap = user_coupon_data[(user_coupon_data.date_received.notna())&(user_coupon_data.date.notna())][['user_id','coupon_id','day_gap']]
    user_coupon_day_gap = user_coupon_day_gap.groupby(['user_id','coupon_id'])['day_gap'].agg(['min', 'max', 'mean']).reset_index()
    new_column_names = {'user_id':'user_id','coupon_id':'coupon_id','min':'user_coupon_day_gap_min', 'max':'user_coupon_day_gap_max','mean': 'user_coupon_day_gap_mean'}
    user_coupon_day_gap = user_coupon_day_gap.rename(columns=new_column_names)
    user_coupon_feature = user_coupon_feature.merge(user_coupon_day_gap, on=['user_id','coupon_id'], how="left")
    
    user_coupon_feature.user_used_coupon_type_used_15_count = user_coupon_feature.user_used_coupon_type_used_15_count.fillna(0)
    user_coupon_feature.user_coupon_type_count = user_coupon_feature.user_coupon_type_count.fillna(0)
    return user_coupon_feature

def get_merchant_coupon_feature(feature):
    merchant_coupon_data = feature[['merchant_id','coupon_id','date_received','date']].copy()
    merchant_coupon_feature = merchant_coupon_data[['merchant_id','coupon_id']].copy()
    merchant_coupon_feature.drop_duplicates(inplace=True)

    new_feature_name = 'merchant_coupon_type_count'
    merchant_coupon_type_count = merchant_coupon_data[(merchant_coupon_data.coupon_id!='nan') & (merchant_coupon_data.date_received.notna())][['merchant_id','coupon_id']]
    merchant_coupon_type_count[new_feature_name] = 1
    merchant_coupon_type_count = merchant_coupon_type_count.groupby(['merchant_id','coupon_id'])[new_feature_name].agg('sum').reset_index()
    merchant_coupon_feature = merchant_coupon_feature.merge(merchant_coupon_type_count, on=['merchant_id','coupon_id'], how="left")
    
    new_feature_name = 'merchant_id_used_coupon_type_count'
    merchant_id_used_coupon_type_count = merchant_coupon_data[(merchant_coupon_data.coupon_id!='nan') & (merchant_coupon_data.date.notna())][['merchant_id','coupon_id']]
    merchant_id_used_coupon_type_count[new_feature_name] = 1
    merchant_id_used_coupon_type_count = merchant_id_used_coupon_type_count.groupby(['merchant_id','coupon_id'])[new_feature_name].agg('sum').reset_index()
    merchant_coupon_feature = merchant_coupon_feature.merge(merchant_id_used_coupon_type_count, on=['merchant_id','coupon_id'], how="left")
    merchant_coupon_feature.merchant_id_used_coupon_type_count = merchant_coupon_feature.merchant_id_used_coupon_type_count.fillna(0)

    merchant_coupon_feature['merchant_id_used_coupon_type_count_rate'] = merchant_coupon_feature['merchant_id_used_coupon_type_count']/merchant_coupon_feature['merchant_coupon_type_count']
    
    merchant_coupon_data = add_day_gap(merchant_coupon_data)
    new_feature_name = 'merchant_used_coupon_type_used_15_count'
    merchant_used_coupon_type_used_15_count = merchant_coupon_data[(merchant_coupon_data.coupon_id!='nan')  & (merchant_coupon_data.date.notna()) & (merchant_coupon_data.day_gap<=15) & (merchant_coupon_data.day_gap>=0)][['merchant_id','coupon_id']]
    merchant_used_coupon_type_used_15_count[new_feature_name] = 1
    merchant_used_coupon_type_used_15_count = merchant_used_coupon_type_used_15_count.groupby(['merchant_id','coupon_id'])[new_feature_name].agg('sum').reset_index()
    merchant_coupon_feature = merchant_coupon_feature.merge(merchant_used_coupon_type_used_15_count, on=['merchant_id','coupon_id'], how="left")

    merchant_coupon_feature['merchant_used_coupon_type_used_15_count_rate'] = merchant_coupon_feature['merchant_used_coupon_type_used_15_count']/merchant_coupon_feature['merchant_coupon_type_count']

    merchant_coupon_day_gap = merchant_coupon_data[(merchant_coupon_data.date_received.notna())&(merchant_coupon_data.date.notna())][['merchant_id','coupon_id','day_gap']]
    merchant_coupon_day_gap = merchant_coupon_day_gap.groupby(['merchant_id','coupon_id'])['day_gap'].agg(['min', 'max', 'mean']).reset_index()
    new_column_names = {'merchant_id':'merchant_id','coupon_id':'coupon_id','min':'merchant_coupon_day_gap_min', 'max':'merchant_coupon_day_gap_max','mean': 'merchant_coupon_day_gap_mean'}
    merchant_coupon_day_gap = merchant_coupon_day_gap.rename(columns=new_column_names)
    merchant_coupon_feature = merchant_coupon_feature.merge(merchant_coupon_day_gap, on=['merchant_id','coupon_id'], how="left")
    
    merchant_coupon_feature.merchant_used_coupon_type_used_15_count = merchant_coupon_feature.merchant_used_coupon_type_used_15_count.fillna(0)
    merchant_coupon_feature.merchant_coupon_type_count = merchant_coupon_feature.merchant_coupon_type_count.fillna(0)
    return merchant_coupon_feature
    
def get_user_merchant_coupon_feature(feature):
    user_merchant_coupon_data = feature[['user_id','merchant_id','coupon_id','date_received','date']].copy()
    user_merchant_coupon_feature = user_merchant_coupon_data[['user_id','merchant_id','coupon_id']].copy()
    user_merchant_coupon_feature.drop_duplicates(inplace=True)

    new_feature_name = 'user_merchant_coupon_type_count'
    user_merchant_coupon_type_count = user_merchant_coupon_data[(user_merchant_coupon_data.coupon_id!='nan') & (user_merchant_coupon_data.date_received.notna())][['user_id','merchant_id','coupon_id']]
    user_merchant_coupon_type_count[new_feature_name] = 1
    user_merchant_coupon_type_count = user_merchant_coupon_type_count.groupby(['user_id','merchant_id','coupon_id'])[new_feature_name].agg('sum').reset_index()
    user_merchant_coupon_feature = user_merchant_coupon_feature.merge(user_merchant_coupon_type_count, on=['user_id','merchant_id','coupon_id'], how="left")
    
    new_feature_name = 'user_merchant_id_used_coupon_type_count'
    user_merchant_id_used_coupon_type_count = user_merchant_coupon_data[(user_merchant_coupon_data.coupon_id!='nan') & (user_merchant_coupon_data.date.notna())][['user_id','merchant_id','coupon_id']]
    user_merchant_id_used_coupon_type_count[new_feature_name] = 1
    user_merchant_id_used_coupon_type_count = user_merchant_id_used_coupon_type_count.groupby(['user_id','merchant_id','coupon_id'])[new_feature_name].agg('sum').reset_index()
    user_merchant_coupon_feature = user_merchant_coupon_feature.merge(user_merchant_id_used_coupon_type_count, on=['user_id','merchant_id','coupon_id'], how="left")
    user_merchant_coupon_feature.user_merchant_id_used_coupon_type_count = user_merchant_coupon_feature.user_merchant_id_used_coupon_type_count.fillna(0)

    user_merchant_coupon_feature['user_merchant_id_used_coupon_type_count_rate'] = user_merchant_coupon_feature['user_merchant_id_used_coupon_type_count']/user_merchant_coupon_feature['user_merchant_coupon_type_count']
    
    user_merchant_coupon_data = add_day_gap(user_merchant_coupon_data)
    new_feature_name = 'user_merchant_used_coupon_type_used_15_count'
    user_merchant_used_coupon_type_used_15_count = user_merchant_coupon_data[(user_merchant_coupon_data.coupon_id!='nan')  & (user_merchant_coupon_data.date.notna()) & (user_merchant_coupon_data.day_gap<=15) & (user_merchant_coupon_data.day_gap>=0)][['user_id','merchant_id','coupon_id']]
    user_merchant_used_coupon_type_used_15_count[new_feature_name] = 1
    user_merchant_used_coupon_type_used_15_count = user_merchant_used_coupon_type_used_15_count.groupby(['user_id','merchant_id','coupon_id'])[new_feature_name].agg('sum').reset_index()
    user_merchant_coupon_feature = user_merchant_coupon_feature.merge(user_merchant_used_coupon_type_used_15_count, on=['user_id','merchant_id','coupon_id'], how="left")

    user_merchant_coupon_feature['user_merchant_used_coupon_type_used_15_count_rate'] = user_merchant_coupon_feature['user_merchant_used_coupon_type_used_15_count']/user_merchant_coupon_feature['user_merchant_coupon_type_count']

    user_merchant_coupon_day_gap = user_merchant_coupon_data[(user_merchant_coupon_data.date_received.notna())&(user_merchant_coupon_data.date.notna())][['user_id','merchant_id','coupon_id','day_gap']]
    user_merchant_coupon_day_gap = user_merchant_coupon_day_gap.groupby(['user_id','merchant_id','coupon_id'])['day_gap'].agg(['min', 'max', 'mean']).reset_index()
    new_column_names = {'user_id':'user_id','merchant_id':'merchant_id','coupon_id':'coupon_id','min':'user_merchant_coupon_day_gap_min', 'max':'user_merchant_coupon_day_gap_max','mean': 'user_merchant_coupon_day_gap_mean'}
    user_merchant_coupon_day_gap = user_merchant_coupon_day_gap.rename(columns=new_column_names)
    user_merchant_coupon_feature = user_merchant_coupon_feature.merge(user_merchant_coupon_day_gap, on=['user_id','merchant_id','coupon_id'], how="left")
    
    user_merchant_coupon_feature.user_merchant_used_coupon_type_used_15_count = user_merchant_coupon_feature.user_merchant_used_coupon_type_used_15_count.fillna(0)
    user_merchant_coupon_feature.user_merchant_coupon_type_count = user_merchant_coupon_feature.user_merchant_coupon_type_count.fillna(0)
    return user_merchant_coupon_feature
    

def get_leakage_feature(dataset):
    dataset = dataset[["user_id",'merchant_id',"coupon_id",	"discount_rate","distance",	"date_received"	]]
    dataset = dataset[dataset['coupon_id']!=np.nan]
    dataset['date_received'] = pd.to_datetime(dataset['date_received'], format='%Y%m%d')

    
    dataset['user_coupon_cnt'] = dataset['user_id'].map(dataset[dataset['coupon_id']!=np.nan] ['user_id'].value_counts())
    
    new_feature_name = 'user_received_type_coupon_cnt'
    dataset[new_feature_name] = 1
    feature = dataset.groupby(['user_id','coupon_id'])[new_feature_name].agg('sum').reset_index()
    feature.rename(columns={'user_received_type_coupon_cnt': 'user_received_type_coupon'}, inplace=True)
    dataset = pd.merge(dataset, feature, on=['user_id', 'coupon_id'])
    dataset.drop(columns=[new_feature_name], inplace=True)

    # 按用户和领取时间排序
    dataset.sort_values(by=['user_id', 'date_received'], inplace=True)
    df_reverse = dataset.sort_values(by=['user_id', 'date_received'], ascending=[True, False])
    df_reverse['coupons_after'] = df_reverse.groupby('user_id').cumcount()
    dataset = df_reverse.sort_values(by=['user_id', 'date_received'])
    dataset.sort_values(by=['user_id', 'date_received'], inplace=True)
    # 计算此次之前领取的所有优惠券数目
    dataset['coupons_before'] = dataset.groupby('user_id').cumcount()
    # 按 User_id 和 Date_received 排序
    dataset = dataset.sort_values(['user_id', 'date_received'])
    
    # 按用户和优惠券ID分组，然后按领取时间排序
    dataset.sort_values(by=['user_id', 'coupon_id', 'date_received'], inplace=True)
    # 计算此次之前领取的该优惠券数目
    dataset['Same_Coupons_Before'] = dataset.groupby(['user_id', 'coupon_id']).cumcount()
    
    # 按用户和领取时间逆序排序
    dataset.sort_values(by=['user_id', 'coupon_id', 'date_received'], ascending=[True, True, False], inplace=True)
    # 计算此次之后领取的该优惠券数目
    dataset['Same_Coupons_After'] = dataset.groupby(['user_id', 'coupon_id']).cumcount()
    # 逆序回来，恢复为按时间升序排列
    dataset.sort_values(by=['user_id', 'coupon_id', 'date_received'], ascending=[True, True, True], inplace=True)
    
    # 按 User_id 和 Date_received 排序
    dataset = dataset.sort_values(['user_id', 'date_received'])
    # 计算上一次领取的时间间隔
    dataset['days_since_last_coupon'] = dataset.groupby('user_id')['date_received'].diff().dt.days
    # 计算下一次领取的时间间隔
    dataset['days_until_next_coupon'] = dataset.groupby('user_id')['date_received'].diff(periods=-1).dt.days

    grouped = dataset.groupby(['user_id', 'merchant_id'])
    # 计算每个用户领取的该商家的优惠券数目
    coupon_counts = grouped.size().reset_index(name='cupon_count')
    dataset = pd.merge(dataset, coupon_counts, on=['user_id', 'merchant_id'])

    unique_merchants = dataset.groupby('user_id')['merchant_id'].nunique().reset_index(name='Unique_Merchant_Count')
    dataset = pd.merge(dataset, unique_merchants, on=['user_id'])

    unique_coupon_counts = dataset.groupby('user_id')['coupon_id'].nunique().reset_index(name='Unique_Coupon_Count')
    dataset = pd.merge(dataset, unique_coupon_counts, on=['user_id'])

    coupon_counts = dataset.groupby('merchant_id')['coupon_id'].count().reset_index(name='Coupon_Count')
    dataset = pd.merge(dataset, coupon_counts, on=['merchant_id'])

    specific_coupon_counts = dataset.dropna(subset=['coupon_id']).groupby(['merchant_id', 'coupon_id']).size().reset_index(name='Specific_Coupon_Count')
    dataset = pd.merge(dataset, specific_coupon_counts, on=['merchant_id','coupon_id'])

    merchant_user_counts = dataset.groupby('merchant_id')['user_id'].nunique().reset_index(name='Unique_User_Count')
    dataset = pd.merge(dataset, merchant_user_counts, on=['merchant_id'])

    first_received = dataset.groupby('user_id')['date_received'].transform('min')
    last_received = dataset.groupby('user_id')['date_received'].transform('max')
    
    # 创建标记列
    dataset['first_received_flag'] = (dataset['date_received'] == first_received).astype(int)
    dataset['last_received_flag'] = (dataset['date_received'] == last_received).astype(int)
    
    # 计算每个用户第一次和最后一次领取的优惠券数量
    first_received_count = dataset[dataset['date_received'].isin(first_received)].groupby('user_id')['coupon_id'].count()
    last_received_count = dataset[dataset['date_received'].isin(last_received)].groupby('user_id')['coupon_id'].count()
    
    # 将结果合并回原始 DataFrame
    dataset['first_received_count'] = dataset['user_id'].map(first_received_count)
    dataset['last_received_count'] = dataset['user_id'].map(last_received_count)

    ######
    first_received = dataset.groupby('merchant_id')['date_received'].transform('min')
    last_received = dataset.groupby('merchant_id')['date_received'].transform('max')
    
    # 创建标记列
    dataset['first_received_flag_mer'] = (dataset['date_received'] == first_received).astype(int)
    dataset['last_received_flag_mer'] = (dataset['date_received'] == last_received).astype(int)
    
    # 计算每个mer第一次和最后一次领取的优惠券数量
    first_received_count = dataset[dataset['date_received'].isin(first_received)].groupby('merchant_id')['coupon_id'].count()
    last_received_count = dataset[dataset['date_received'].isin(last_received)].groupby('merchant_id')['coupon_id'].count()
    
    # 将结果合并回原始 DataFrame
    dataset['first_received_count_mer'] = dataset['merchant_id'].map(first_received_count)
    dataset['last_received_count_mer'] = dataset['merchant_id'].map(last_received_count)
    
    dataset['date_received'] = dataset['date_received'].dt.strftime('%Y%m%d').astype(float)
    return dataset

def user_coupon_consumed(feature):

    scaler = MinMaxScaler()

    feature['user_consumed_cnt'] = feature['user_id'].map(feature[feature['date']!=np.nan] ['user_id'].value_counts())

    feature['user_consumed_cnt_50'] = feature['user_id'].map(feature[(feature['date']!=np.nan) & (feature['discount_rate'].apply(lambda x:0<(convert_discount_rate(x))<=50))]['user_id'].value_counts()) 
    feature['user_consumed_cnt_50'].fillna(0, inplace=True)
    feature['user_consumed_cnt_50_rate'] = feature['user_consumed_cnt_50']/feature['user_consumed_cnt']

    feature['user_consumed_cnt_200'] = feature['user_id'].map(feature[(feature['date']!=np.nan) & (feature['discount_rate'].apply(lambda x:50<(convert_discount_rate(x))<=200))]['user_id'].value_counts()) 
    feature['user_consumed_cnt_200'].fillna(0, inplace=True)
    feature['user_consumed_cnt_200_rate'] = feature['user_consumed_cnt_200']/feature['user_consumed_cnt']

    feature['user_consumed_cnt_500'] = feature['user_id'].map(feature[(feature['date']!=np.nan) & (feature['discount_rate'].apply(lambda x:200<(convert_discount_rate(x))<=500))]['user_id'].value_counts()) 
    feature['user_consumed_cnt_500'].fillna(0, inplace=True)
    feature['user_consumed_cnt_500_rate'] = feature['user_consumed_cnt_500']/feature['user_consumed_cnt']

    feature['user_consumed_cnt_others_rate'] = (feature['user_consumed_cnt']-feature['user_consumed_cnt_50']-feature['user_consumed_cnt_200']-feature['user_consumed_cnt_500'])/feature['user_consumed_cnt']
    feature['user_consumed_cnt_others_rate'].fillna(0, inplace=True)
    
    feature[['user_consumed_cnt_50_rate', 'user_consumed_cnt_200_rate', 'user_consumed_cnt_500_rate', 'user_consumed_cnt_others_rate']] = scaler.fit_transform(feature[['user_consumed_cnt_50_rate', 'user_consumed_cnt_200_rate', 'user_consumed_cnt_500_rate', 'user_consumed_cnt_others_rate']])

    return feature[['user_id', 'user_consumed_cnt_50_rate', 'user_consumed_cnt_200_rate','user_consumed_cnt_500_rate', 'user_consumed_cnt_others_rate']]

def user_merchant_used_discount_rate(feature):
    data = feature[(feature.coupon_id!='nan')  & (feature.date.notna()) & (feature.date_received.notna())][['user_id','merchant_id','coupon_id','distance','discount_rate','date_received','date']]
    data_feature=add_discount(data)
    data_feature = add_day_gap(data_feature)
    data_feature = data_feature[(data_feature.day_gap<=15) & (data_feature.day_gap>=0)][['user_id','merchant_id','coupon_id','distance','discount_rate','date_received','date','day_gap']]
    
    grouped_discount_rate = data_feature.groupby(['user_id','merchant_id'])['discount_rate'].agg(
        max_mer_discount_rate='max',
    ).reset_index()


    data_feature = data_feature.merge(grouped_discount_rate, on=['user_id','merchant_id'], how='left')

    return data_feature[['user_id','merchant_id','max_mer_discount_rate']]

def get_all_feature(dataset,feature,if_train=True):
    data_feature=add_discount(dataset)
    merchant_feature=get_merchant_feature(feature)
    data_feature=data_feature.merge(merchant_feature, on='merchant_id', how="left")
    user_feature=get_user_feature(feature)
    data_feature=data_feature.merge(user_feature, on='user_id', how="left")
    user_coupon_consumed_fearture = user_coupon_consumed(feature)
    data_feature=data_feature.merge(user_coupon_consumed_fearture, on='user_id', how="left")

    user_merchant=get_user_merchant_feature(feature)
    data_feature=data_feature.merge(user_merchant, on=['user_id','merchant_id'], how="left")
    
    coupon_feature = get_coupon_feature(feature)
    data_feature=data_feature.merge(coupon_feature, on=['coupon_id'], how="left")
    user_coupon_feature = get_user_coupon_feature(feature)
    data_feature=data_feature.merge(user_coupon_feature, on=['user_id','coupon_id'], how="left")
    merchant_coupon_feature=get_merchant_coupon_feature(feature)
    data_feature=data_feature.merge(merchant_coupon_feature, on=['merchant_id','coupon_id'], how="left")
    user_merchant_coupon_feature=get_user_merchant_coupon_feature(feature)
    data_feature=data_feature.merge(user_merchant_coupon_feature, on=['user_id','merchant_id','coupon_id'], how="left")
    leakage_feature=get_leakage_feature(dataset)    
    data_feature=data_feature.merge(leakage_feature, on=['user_id','merchant_id','coupon_id','date_received'],how='left')
    
    data_feature.drop_duplicates(inplace=True)
    if if_train==True:
        data_feature=add_label(data_feature)
    return data_feature


def myauc(valid_data):
    testgroup = valid_data.groupby(['coupon_id'])
    aucs = []
    for i in testgroup:
        coupon_df = i[1]
        if len(coupon_df['label'].unique()) < 2:
            continue
        auc = metrics.roc_auc_score(coupon_df['label'], coupon_df['pred'])
        aucs.append(auc)
    return np.average(aucs)

dftrain = off_train.copy()
dftest = off_test.copy()

dftrain = get_date_feature(dftrain)
dftest = get_date_feature(dftest)

dataset1 = dftrain[(dftrain.date_received.astype(str)>='201604014')&(dftrain.date_received.astype(str)<='20160514')]
#交叉训练集一特征：线下数据中领券和用券日期大于1月1日和小于4月13日
feature1 = dftrain[(dftrain.date.astype(str)>='20160101')&(dftrain.date.astype(str)<='20160413')|((dftrain.date.astype(str)=='null')&(dftrain.date_received.astype(str)>='20160101')&(dftrain.date_received.astype(str)<='20160413'))]

#交叉训练集二：收到券的日期大于5月15日和小于6月15日
dataset2 = dftrain[(dftrain.date_received.astype(str)>='20160515')&(dftrain.date_received.astype(str)<='20160615')]
#交叉训练集二特征：线下数据中领券和用券日期大于2月1日和小于5月14日
feature2 = dftrain[(dftrain.date.astype(str)>='20160201')&(dftrain.date.astype(str)<='20160514')|((dftrain.date.astype(str)=='null')&(dftrain.date_received.astype(str)>='20160201')&(dftrain.date_received.astype(str)<='20160514'))]


#测试集
dataset3 = dftest
#测试集特征 :线下数据中领券和用券日期大于3月15日和小于6月30日的
feature3 = dftrain[((dftrain.date.astype(str)>='20160315')&(dftrain.date.astype(str)<='20160630'))|((dftrain.date.astype(str)=='null')&(dftrain.date_received.astype(str)>='20160315')&(dftrain.date_received.astype(str)<='20160630'))]
print('数据滑窗划分完成！')
exclude_columns = ['user_id', 'coupon_id','date_received', 'day_0.0', 'day_1.0', 'day_2.0', 
                   'day_3.0', 'day_4.0', 'day_5.0', 'day_6.0', 'if_fd', 
                   'full_value', 'reduction_value','label']


dftrain1=get_all_feature(dataset1,feature1,True)
cols_to_normalize = [col for col in dftrain1.columns if col not in exclude_columns and pd.api.types.is_numeric_dtype(dftrain1[col])]
dftrain1[cols_to_normalize] = dftrain1[cols_to_normalize].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

dftrain2=get_all_feature(dataset2,feature2,True)
cols_to_normalize = [col for col in dftrain2.columns if col not in exclude_columns and pd.api.types.is_numeric_dtype(dftrain2[col])]
dftrain2[cols_to_normalize] = dftrain2[cols_to_normalize].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

dftrain=pd.concat([dftrain1,dftrain2],axis=0)
dftest=get_all_feature(dataset3,feature3,False)

cols_to_normalize = [col for col in dftest.columns if col not in exclude_columns[:-1] and pd.api.types.is_numeric_dtype(dftest[col])]
dftest[cols_to_normalize] = dftest[cols_to_normalize].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

print('特征提取完成！')
dftrain.drop(['date'], axis=1, inplace=True)
dftrain.drop(['merchant_id'], axis=1, inplace=True)
dftest.drop(['merchant_id'], axis=1, inplace=True)
dftrain = dftrain.loc[:, (dftrain != 0).any(axis=0)]

# 删除全为np.nan的列
dftrain = dftrain.dropna(axis=1, how='all')

dftrain = dftrain.drop_duplicates()
dftrain = dftrain.loc[:, ~dftrain.columns.duplicated()]
train_X= np.array(dftrain.drop(['label'], axis=1))
train_y = np.array(dftrain['label'])
test_X = dftest.copy()
a = dftrain.columns.tolist()
a.remove('label')
test_X=test_X[a]
len(dftrain.columns)

from sklearn.model_selection import cross_val_score # 交叉检验
n_splits = 5
skfolds = StratifiedKFold(n_splits=n_splits, random_state=2023,shuffle =True)
xgb_valid = np.zeros((len(train_X)))
xgb_pred = np.zeros((len(test_X)))
all_auc = 0
importance_sum = {}

num_boost_round = 3650

for train_index, valid_index in skfolds.split(train_X,train_y):
    train_data = train_X[train_index]
    train_label = train_y[train_index]
    vaild_data = train_X[valid_index]
    vaild_label = train_y[valid_index]
    params = {'booster': 'gbtree',
          'objective': 'binary:logistic',
          'eval_metric': 'auc',
          'gamma': 0.1,
          'min_child_weight': 1.1,
          'max_depth': 5,
          'lambda': 10,
          'subsample': 0.7,
          'colsample_bytree': 0.7, 
          'colsample_bylevel': 0.7,
          'eta': 0.01,
          'tree_method': 'exact',
          'seed': 0,
          }
    dtrain =  xgb.DMatrix(np.delete(train_data, [0,1], axis=1),label=train_label)
    dvalid  = xgb.DMatrix(np.delete(vaild_data, [0,1], axis=1))
    label_val=vaild_label
    valid_data_df = pd.DataFrame({'coupon_id':vaild_data[:,1],'label':label_val})
    dtest  = xgb.DMatrix(test_X.drop(['user_id', 'coupon_id'], axis=1))
    # 训练
    watchlist = [(dtrain, 'train')]
    # 2025
    model = xgb.train(params, dtrain, num_boost_round=num_boost_round ,evals=watchlist,verbose_eval=False)
    fold_importance = model.get_score(importance_type='gain')
    for key, value in fold_importance.items():
        importance_sum[key] = importance_sum.get(key, 0.0) + value
    # 预测
    predict_val = model.predict(dvalid)
    valid_data_df['pred'] = predict_val
    xgb_valid[valid_index] = predict_val
#     roc_auc =metrics.roc_auc_score(label_val, predict_val)
    roc_auc=myauc(valid_data_df)
    all_auc += roc_auc/n_splits    
#     # fpr假正率，tpr召回率，thresholds阈值，pos_label（设置正样本值）默认为None（标签数据为二分类的情况）
#     fpr, tpr, thresholds = metrics.roc_curve(label_val, predict_val, pos_label=1)
#     plt.plot(fpr, tpr, 'r')  # 绘制ROC曲线
#     axline = np.array([0.,0.2,0.4,0.6,0.8,1.0])  # 斜线参考线坐标
#     plt.plot(axline,axline,'gray',linestyle='--',alpha=0.5)
#     plt.xlim([-0.05, 1.05])  # 设置x轴刻度范围
#     plt.ylim([-0.05, 1.05])  # 设置y轴刻度范围
#     plt.xlabel('FPR')  # x轴是False Positive Rate
#     plt.ylabel('TPR')  # y轴是True Positive Rate
#     plt.title('AUC = %0.3f' % all_auc)
    print('auc: '+str(roc_auc)) 
    # 处理结果
    predict = model.predict(dtest)
    xgb_pred += predict/n_splits
else:
    print('average: '+str(all_auc))

# 输出特征重要性（按平均 gain 排序）
base_cols = dftrain.drop(['label'], axis=1).columns.tolist()
feature_cols = base_cols[2:] if len(base_cols) >= 2 else base_cols
feature_map = {f"f{i}": col for i, col in enumerate(feature_cols)}
importance_avg = {feature_map.get(k, k): v / n_splits for k, v in importance_sum.items()}
importance_df = pd.DataFrame(
    [{"feature": k, "gain": v} for k, v in importance_avg.items()]
).sort_values("gain", ascending=False)
print("\nXGBoost Feature Importance (avg gain):")
print(importance_df.head(20))

# Top20 特征的训练/测试分布对比
top_features = importance_df.head(20)["feature"].tolist()
plot_features = [f for f in top_features if f in dftrain.columns and f in dftest.columns]
if len(plot_features) < 20:
    print(f"Warning: only {len(plot_features)} features found in both train and test for plotting")

train_plot = dftrain[plot_features].copy()
test_plot = dftest[plot_features].copy()
train_plot["_set"] = "train"
test_plot["_set"] = "test"

sample_size = 20000
if len(train_plot) > sample_size:
    train_plot = train_plot.sample(sample_size, random_state=2023)
if len(test_plot) > sample_size:
    test_plot = test_plot.sample(sample_size, random_state=2023)

plot_df = pd.concat([train_plot, test_plot], axis=0, ignore_index=True)

fig, axes = plt.subplots(4, 5, figsize=(22, 16))
axes = axes.flatten()
for idx, feature in enumerate(plot_features):
    ax = axes[idx]
    sns.kdeplot(data=plot_df, x=feature, hue="_set", ax=ax, common_norm=False, fill=False)
    ax.set_title(feature)
for ax in axes[len(plot_features):]:
    ax.axis("off")
plt.tight_layout()
plt.savefig("top20_feature_distribution.png", dpi=200)
plt.show()

test_data = test_X.copy()
sumbit_xgb = off_test[['user_id', 'coupon_id', 'date_received']]
test_data['label'] = xgb_pred
sumbit_xgb = sumbit_xgb.merge(test_data[['user_id','coupon_id','date_received','label']], on=['user_id','coupon_id','date_received'])
sumbit_xgb.to_csv( 'sumbit_xgb'+str(num_boost_round)+'.csv',index=False, header=None)

