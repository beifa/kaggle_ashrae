import os
import gc
import numpy as np 
import pandas as pd
from lightgbm import LGBMRegressor
import lightgbm as lgb
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_log_error, mean_squared_error
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor
#
import catboost
from catboost import Pool, cv
from catboost import CatBoostRegressor
print(catboost.__version__)
#
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

def input_file(file):
    path = f"../input/ashrae-energy-prediction/{file}"
    if not os.path.exists(path): return path + ".gz"
    return path

#compress
def compress_dataframe(df):
    """
    не знаю надо будет попробовать, что такой подход проще чем указывать
    и проверять данные для уменьшения размера(по памяти).
    А тут пандас само это делает downcast: {"целое число", "подписано", "без знака", "плавать"}, начинает с мин и тд.
    train - mem used - 616 after compres -173
    """
    result = df.copy()
    for col in result.columns:
        col_data = result[col]
        dn = col_data.dtype.name    
        if dn == "object":
          """
          make category and return cat int      
          """
          result[col] = pd.to_numeric(col_data.astype("category").cat.codes, downcast="integer")
        elif dn == "bool":
          result[col] = col_data.astype("int8")
        elif dn.startswith("int") or (col_data.round() == col_data).all():
          result[col] = pd.to_numeric(col_data, downcast="integer")
        else:
          result[col] = pd.to_numeric(col_data, downcast='float')
    return result

#weather

def fill_by_each_id(df):
    """
    идею подглядел, можно время преобразовать в часы и сделать после смещение для выравнивания по зонам
    но можно сделать и иначе просто сделать из времени категории(на выходе одно и тоже)
    """
    offsets = [-5, 0, -7, -5, -8, 0, -5, -5, -5, -6, -7, -5, 0, -6, -5, -5]#на сколько время отклоняется для каждого сайта  
    df.timestamp = df.timestamp.astype("datetime64[ns]")
    df.timestamp = (df.timestamp- pd.to_datetime('2016-01-01')).dt.total_seconds() // 3600
    dict_offset = {site:offset for site, offset in enumerate(offsets)} #{0: -5, 1: 0,...}
    df.timestamp = df.timestamp - df.site_id.map(dict_offset)
    box = []
    for iid in df.site_id.unique(): 
        site = df[df.site_id == iid].set_index(['timestamp']).reindex(range(8784))
        site.site_id = iid #fill site id
        for col in [c for c in site.columns if c  != 'site_id']: # all col without site_id
            site[f'mark_not_fill_{col}'] = ~site[col].isna() # return not na
            site[col] = site[col].interpolate(limit_direction='both', method='linear')
            site[col] = site[col].fillna(df[col].median())
        box.append(site)# after we concat site_id : 0, 1, 15 
    df = pd.concat(box).reset_index()
    return compress_dataframe(df).set_index(["site_id", "timestamp"])

#for test

def fill_by_each_id_test(df):
    """
    df = weather
    df == 1 year train but test split for bool by two years
    year = '2016-01-01'(train), count = 8784
    year = '2017-01-01'(test), count = 8599
    year = '2018-01-01'(test), count = 3079 not all year
    ======================ps
    карочь не хватает памяти дропается, + много дубликатов но и вот дропнуть их не могу по памяти не получается такой подход и вот
    темным вечерком пришла мысля такая timestamp - 2016 на тесте начинается с 8789 и кончается 26308 и вот оно решение
    ======================
    """
    offsets = [-5, 0, -7, -5, -8, 0, -5, -5, -5, -6, -7, -5, 0, -6, -5, -5]#на сколько время отклоняется для каждого сайта  
    df.timestamp = df.timestamp.astype("datetime64[ns]")
    df.timestamp = (df.timestamp- pd.to_datetime('2016-01-01')).dt.total_seconds() // 3600
    dict_offset = {site:offset for site, offset in enumerate(offsets)} #{0: -5, 1: 0,...}
    df.timestamp = df.timestamp - df.site_id.map(dict_offset)
    #return df 
    box = []
    for iid in df.site_id.unique():
        site = df[df.site_id == iid].set_index(['timestamp']).reindex(range(8789, 26308))
        site.site_id = iid #fill site id
        for col in [c for c in site.columns if c  != 'site_id']: # all col without site_id
            site[f'mark_not_fill_{col}'] = ~site[col].isna() # return not na
            site[col] = site[col].interpolate(limit_direction='both', method='linear')
            site[col] = site[col].fillna(df[col].median())
        box.append(site)# after we concat site_id : 0, 1, 15 
    df = pd.concat(box).reset_index()
    return compress_dataframe(df).set_index(["site_id", "timestamp"])

#drop_0

def drop_0(df):
    """ 
    df = train data

    идея тут проста электричество не может быть 0, для коммерческих построек
    там полюбому даже если никто не работает или ночь или хз что тратится энергия,
    от общего числа данных эти значения составляют 0,02%.
    Альтернатива добавить сюда другие счетчики дабы исключить все"""
    df = df.reindex(df[(df.meter_reading > 0) & (df.meter >= 0)].index) #0.9073027933181969 
    return  df

#fill metadata

def fill_metadata(name):
    """
    name = name file "building_metadata.csv"

    Food sales and service (2005 - 2012 )
    сюда входят рестораны, ночные клубы, фастфуд, супермаркеты и тд.
    Я думаю что в большинстве своем это одноэтажные здания и мы установим значение на 2.
    Religious worship (< 1980)
    церкви, храмы, мечети, синагоги, дома собраний или любые другие здания,
    которые в основном служат местом религиозного поклонения значение 1
    Services (2003-2007-2012)
    Data Center, Personal Services (Health/Beauty, Dry Cleaning, etc), Repair Services (Vehicle, Shoe, Locksmith, etc) значение 1.5
    Радует не большое количество зданий 5, 3, 10    
    """
    metadata = pd.read_csv(input_file(name))
    m = metadata.primary_use == 'Food sales and service'
    metadata.loc[m, 'year_built'] = [2012, np.nan, 2005, np.nan, 2009]
    metadata.loc[m, 'floor_count'] = [2, np.nan, 1, np.nan, 2]
    #
    m2 =  metadata.primary_use == 'Religious worship'
    metadata.loc[m2, 'year_built'] = [1970, 1980, 1930]
    metadata.loc[m2, 'floor_count'] = [1, 2, 2]
    #
    m3 =  metadata.primary_use == 'Services'
    metadata.loc[m3, 'year_built'] = [2003, 2007, np.nan, 2005, np.nan, 2012, np.nan, np.nan, 2009, np.nan]
    metadata.loc[m3, 'floor_count'] = [3, 1, np.nan, 2, np.nan, 1, np.nan, 3, 2, np.nan]
    ##
    metadata['year_built'] = metadata['year_built'].interpolate()
    metadata['floor_count'] = metadata['floor_count'].interpolate(method = 'pad')
    #в начале много пропусков  мином заполним
    m = metadata['floor_count'].isnull()
    metadata.loc[m, 'floor_count'] = int(metadata['floor_count'].mean())
    metadata['year_built'] = metadata['year_built'].astype('int')
    return metadata

#####MERGE

def read_building_metadata():
    return compress_dataframe(fill_metadata('building_metadata.csv')).set_index("building_id")
    
def read_test():
    df = pd.read_csv(input_file("test.csv"), parse_dates=["timestamp"])
    df.timestamp = (df.timestamp - pd.to_datetime("2016-01-01")).dt.total_seconds() // 3600
    return compress_dataframe(df).set_index("row_id")

def mergeall(name, weather, train = True, save = False): 
    ##train --> metadata-->weather 
    weather = pd.read_csv(input_file(weather)) 
    if train:
        weather = fill_by_each_id(weather)
        data = drop_0(compress_dataframe(pd.read_csv(input_file(name))))
    else:
        weather = fill_by_each_id_test(weather)
        data = read_test()
    df = data.join(read_building_metadata(), on="building_id") \
             .join(weather, on=["site_id", "timestamp"]).fillna(-1)#.to_pickle(path + 'data_test_before_fake.pkl') # -9999
    
    #no time to explain just do it
    bad_col = [
    'mark_not_fill_cloud_coverage',
    'mark_not_fill_precip_depth_1_hr',
    'mark_not_fill_sea_level_pressure',
    'mark_not_fill_wind_direction']    
    
    df.drop(bad_col, axis = 'columns', inplace = True)    
    
    del data
    del weather
    gc.collect()

    if save:
        #Save but crush memory
        df.to_pickle(path + 'data_test_before_fake.pkl')
        return print('Saved not return df pls load instance')
    return df

#merged data
data_train = mergeall('train.csv', 'weather_train.csv')
data_test = compress_dataframe(mergeall('test.csv', 'weather_test.csv', train=False, save = False))

print('Data Merged')


def drop_fake_site(df):
    #train
    #141days, after merge data
    df = df[(df.timestamp >= 3378) | (df.site_id != 0) | (df.meter != 0)]
    return df

def make_time(idx, namefile, savename):
    """
    name = file name to load
    le train after drop 0 meter  
    maybe saved ?
    """
    temp = pd.read_csv(input_file(namefile))
    time = temp.loc[idx.index].timestamp
    del temp
    gc.collect()
    time = pd.to_datetime(time)
    df = pd.DataFrame(time, columns=['timestamp'])
    col = 'timestamp'
    
    #df['hour'] = df[col].dt.hour.astype(np.uint8)
    #df['month'] = df[col].dt.month.astype(np.uint8) - 1
    df['weekday'] = df[col].dt.weekday.astype(np.uint8)
    df['dayofyear'] = df[col].dt.dayofyear.astype(np.uint16) - 1
    #df['weekofyear'] = df[col].dt.weekofyear.astype(np.uint8) - 1
    #df['quarter'] = df[col].dt.quarter.astype(np.uint8) - 1
    #df['monthday'] = df[col].dt.day.astype(np.uint8) - 1
    
    #holiday
    #from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
    #dates_range = pd.date_range(start='2015-12-31', end='2019-01-01')
    #us_holidays = calendar().holidays(start=dates_range.min(), end=dates_range.max())
    #df['holiday'] = df[col].isin(us_holidays).astype(np.int8)
    
    del time
    gc.collect()
    df.drop('timestamp', axis = 'columns', inplace=True) #нам не нужно время по умолчанию дропаем, только  признаки
    #df.to_pickle(path + savename)    
    """
    у нас теперь есть время отдельно 
    """    
    return df

print('Make time features')


#not use(roll_features) all signs were excluded from the final model

# def roll_features(df,savename, win = 24):
#     #after merge and add time
#     #df = datamerge
#     for col in ['air_temperature', 'dew_temperature']:
#         df['rolling_back_'+ col] = df.groupby(by='site_id')[col]\
#               .rolling(window=win, min_periods=1).mean().interpolate().values.astype(np.int16)
#         # reversed rolling
#         df['rolling_forw_'+ col] = df.iloc[::-1].groupby(by='site_id')[col]\
#               .rolling(window=win, min_periods=1).mean().interpolate().values.astype(np.int16)

#         # rolling mean for same hour of the day
#         df['rolling_back_h_'+ col] = df.groupby(by=['site_id', 'hour'])[col]\
#               .rolling(window=3, min_periods=1).mean().interpolate().values.astype(np.int16)

#         df['rolling_back_h_f_' + col] = df.iloc[::-1].groupby(by=['site_id', 'hour'])[col]\
#               .rolling(window=3, min_periods=1).mean().interpolate().values.astype(np.int16)
#     #df.to_pickle(path + savename)
#     return df

def humidity(df):
    #df = data with weather  
    
    df['e'] = df['air_temperature'].apply(lambda x: 6.11 * 10**( (7.5 * x) / (237.7 + x) ))
    df['es'] = df['dew_temperature'].apply(lambda x: 6.11 * 10**( (7.5 * x) / (237.7 + x) ))
    df['rh'] =((df['es'] /  df['e'])) * 100
    #add recommends
    df['rec_air'] = pd.cut(df.air_temperature, [-50, 0, 20, 25, 60],
                           labels=['very_bad_cold', 'bad', 'Good', 'very_bad_hot'],
                           include_lowest=True)  
#     df['rec_dew'] = pd.cut(df.dew_temperature, [-50, -20, -5,  16, 20, 50],
#                            labels=['dew_very_low','dew_low', 'Good', 'dew_big', 'dew_very_big'],
#                            include_lowest=True)
    #need check rh not big 100
    #df.loc[df.rh >= 100, 'rh'] = 100
    #df['rec_rh'] = pd.cut(df.rh, [0,  20, 60, 100],labels=['small_h', 'Good', 'big_h'],  include_lowest=True)    
    return df

print('Make humidity')

def add_new_f(data_train, data_test):
    train_new = drop_fake_site(data_train)
    del data_train
    #time
    temp_train = make_time(train_new, 'train.csv', savename = 'time_train.pkl.gz')
    temp_test  = make_time(data_test, 'test.csv', savename = 'time_test.pkl.gz')
    
    train = humidity(pd.concat([train_new, temp_train], axis = 'columns'))
    test = humidity(pd.concat([data_test, temp_test], axis = 'columns'))
    del temp_train
    del temp_test
    del data_test
    gc.collect()      
    return train, test

train, test = add_new_f(data_train, data_test)

print('Merge new features')

#i need more gold :)

#add cat
for_cat = ['building_id', 'meter', 'site_id', 'primary_use', 'floor_count', 'cloud_coverage',
              'mark_not_fill_air_temperature', 'mark_not_fill_dew_temperature', 'mark_not_fill_wind_speed']

#droped in table

# drop_1 = ['air_temperature','rolling_forw_air_temperature', 'hour', 'site_id', 'rec_rh'] # 1.99
# drop_2 = ['dayofyear', 'dew_temperature', 'primary_use', 'rec_air', 'floor_count', 'air_temperature', 'wind_direction'] #1.93
# drop_3 = ['dew_temperature', 'rec_rh', 'rh', 'rolling_forw_air_temperature', 'air_temperature', 'wind_direction', 'hour']#1.95
# drop_4 = ['rh', 'air_temperature', 'dew_temperature', 'wind_speed', 'wind_direction', 'rec_air', 'dayofyear']#1.97


final_col = ['building_id', 'meter', 'timestamp', 'meter_reading', 'site_id',
             'primary_use', 'square_feet', 'year_built', 'floor_count',
             'cloud_coverage', 'dew_temperature', 'sea_level_pressure',
             'wind_direction', 'wind_speed', 'mark_not_fill_air_temperature',
             'mark_not_fill_dew_temperature', 'mark_not_fill_wind_speed', 'weekday',
             'dayofyear', 'rh', 'rec_air']

final_col_test = ['building_id', 'meter', 'timestamp', 'site_id',
                  'primary_use', 'square_feet', 'year_built', 'floor_count',
                  'cloud_coverage', 'dew_temperature', 'sea_level_pressure',
                  'wind_direction', 'wind_speed', 'mark_not_fill_air_temperature',
                  'mark_not_fill_dew_temperature', 'mark_not_fill_wind_speed', 'weekday',
                  'dayofyear', 'rh', 'rec_air']

#bad site
def find_bad_building1099(df):
    #data = df
    #3351 row
    return df[(df.building_id == 1099) & (df.meter == 2) & (df.meter_reading > 3e4)].index

def drop_make_cat(data, col, cat_col, train= True):
    #data - data
    #col = col for droped
    d = data.copy()
    
    #cat error fetures in kaggle not error in colab    ['cloud_coverage', 'floor_count']
    d.loc[:,'floor_count'] = d.loc[:,'floor_count'].astype('int')
    d.loc[:,'cloud_coverage'] = d.loc[:,'cloud_coverage'].astype('int')
    
    for c in cat_col:        
        d[c] = d[c].astype('category')         
    new_d = d.loc[:, col]
    del data
    del d
    gc.collect()
    
    if train:
        print(new_d.shape)
        idx_bad = find_bad_building1099(new_d)
        new_d = new_d.drop(idx_bad)
        print(new_d.shape)        
    
    new_d.rec_air = new_d.rec_air.cat.codes.astype("category")    
    return new_d      
    

train_new = drop_make_cat(train, final_col, for_cat, train = True)
test_new = drop_make_cat(test, final_col_test, for_cat, train= False)


del test
del train
gc.collect()

print('Droped not cool features')

#for test for small sample data
# def rand_sample(train, f = 0.25):
#     #вернет выборку рандомную, f - коеф. размера выборки
#     np.random.seed(0)
#     idx = np.argsort(train.timestamp.values, kind='stable')
#     sample_idx = np.random.choice(idx, int(len(idx) * f), replace=False) 
#     return sample_idx

# sample_train = train_new.iloc[rand_sample(train_new, f = 0.01)] #4584686 rows × 45 columns

# X = sample_train.drop('meter_reading', axis = 1)
# y = sample_train.meter_reading

# del train_new
# del test_new
# gc.collect()

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=13)
# train_new = pd.concat([X_train, y_train], axis = 1)
# test_new = X_test
# print('Sample data maked ends')

def cost(target, pred):
    return np.sqrt(mean_squared_log_error(np.expm1(target), np.clip(np.expm1(pred), 0, None)))

def cluster_id(name_model, data, test, mark, model, num_cluster, num_iters_lgb,  skip = True, param = None):  
    """
    че мы хотим, а мы хотим
    разбиваем на  сайт ид и в нем делаем кластеры  затем учим и предсказывае для каждого кластера в сайт ид
    name_model = 'lgb' or cat
    data = data
    num_clusters = int, num cluster  
    mark = feature test
    num_iters_lgb - iter lgb
    param = for lgb, model if lgb blank

    """
    idx_cat = np.where(data.drop(['meter_reading'], axis = 'columns').dtypes == 'category')[0]
    idx_cat_test = np.where(test.dtypes == 'category')[0]
    idx_cat = np.insert(idx_cat, 0, 0) # add building id
    idx_cat_test = np.insert(idx_cat_test, 0, 0) 
    #print(idx_cat)
    data.building_id = data.building_id.astype(np.int16)
    test.building_id = test.building_id.astype(np.int16)  
    df_id = pd.DataFrame(data = np.zeros(len(test)), columns = ['pred'], index=test.index)
    for id_ in data.site_id.unique(): 
        mask_id = data.site_id == id_
        mask_id_test = test.site_id == id_ 
        site_id = data[mask_id] 
        site_id_test = test[mask_id_test]   
        #we make cluster in site id 
        if skip:
            g = site_id.groupby(mark)['meter_reading'].median() 
            km = KMeans(n_clusters=num_cluster, init='random', n_init=5, max_iter=100, random_state=13, n_jobs=-1)   
            label = km.fit_predict(g.values.reshape(-1, 1)) 
            g = g.to_frame()    
            g['labels'] = label
            cluster_df = site_id[mark].map(g['labels']) #i make idx - labels
            cluster_test = site_id_test[mark].map(g['labels']) 
            for num in range(num_cluster):
                #num clusters
                idx = cluster_df[cluster_df == num].index #index cluster in id_site
                #idx_box[id_] = idx #index clusters only site_id
                #site_id.loc[cluster_df[cluster_df == 1].index]        
                X = site_id.loc[idx].drop('meter_reading', axis = 1)
                y = site_id.loc[idx]['meter_reading']      
                # learning
                # predict for cluster
                # join for predict site
                # add for predict sample train
                idx_test = cluster_test[cluster_test == num].index 
                test_cluster = site_id_test.loc[idx_test]
                
                if name_model == 'lgb':
                    train_data = lgb.Dataset(X, label=np.log1p(y))
                    #test_data = lgb.Dataset(test_cluster, label=None)                    
                    model = lgb.train(param, train_data, int(num_iters_lgb))                                       
                    df_id.loc[idx_test, 'pred'] = model.predict(test_cluster)                   
                
                if name_model == 'cat':                    
                    train_pool = Pool(
                        data=X, 
                        label=np.log1p(y), 
                        cat_features=idx_cat
                    )
                    test_pool = Pool(
                        data= test_cluster, 
                        cat_features=idx_cat_test
                    )
                    model.fit(train_pool)                    
                    df_id.loc[idx_test, 'pred'] = model.predict(test_pool)
                    
                    
                if name_model == '' :
                    model.fit(X, np.log1p(y))
                    df_id.loc[idx_test, 'pred'] = model.predict(test_cluster)
            else: 
                pass   
#                 X = site_id.drop('meter_reading', axis = 1)
#                 y = site_id['meter_reading']
#                 model.fit(X, y)            
#                 df_id.loc[site_id.index, 'pred'] = model.predict(test)
    #print(cost(np.log1p(data.meter_reading.values), df_id.pred.values)  )
    return df_id



#1 all
#2 not cat

#drop by RAM

# def data_fo_knn(data, drop_cat = False):    
#     if drop_cat:
#         idx = np.where(data.dtypes != 'category')[0]
#         data_w_cat = data.iloc[:, idx]
#         return data_w_cat
#     idx = np.where(data.dtypes == 'category')[0]
#     data_all = data.copy()
#     for col in data.iloc[:, idx]:
#         data_all[col] = data_all[col].astype(int)
#         data_all_comp = compress_dataframe(data_all)
#     return data_all_comp
        
# print('ends') 

# # data_knn = data_fo_knn(train_new, drop_cat = True)
# # data_knn_test = data_fo_knn(test_new, drop_cat = True)
    
# data_knn = data_fo_knn(train_new, drop_cat = False)
# data_knn_test = data_fo_knn(test_new, drop_cat = False)
# ###knn



#####################Model
print('Train and predict')

# smal iter(need commit),  because a very very long time 1h33min kaggle
# for  scored uncomment, iter = 1000 ~ 7h(if = 1500 not ended session in kaggle 9h), i predict fo my Pc iter > 2000 5d

num_cluster = 3
num_iters = 10 #1000
model_empty = 'no model'

param_knn = {'algorithm': 'ball_tree',
             'n_neighbors': 15, # 50 #100 # 250
             'p': 1, 
             'weights': 'distance'}
model_knn = KNeighborsRegressor(**param_knn)
# pred_knn = cluster_id('', data_knn, data_knn_test, 'building_id', model_knn, num_cluster, skip = True, param = None)
# print('ends')
pred_knn = cluster_id('', train_new, test_new, 'building_id', model_knn, num_cluster,num_iters,  skip = True, param = None)

print('Knn predict end')

####lgbm
param = {'boosting': 'dart',
         #'n_estimators': 2500,
         'bagging_fraction': 1, 
         'bagging_freq': 5,
         'cat_l2': 2,
         'cat_smooth': 2,
         'feature_fraction': 1,
         'lambda_l1': 0.7409042536819661,
         'lambda_l2': 0.2936292195158015,
         'learning_rate': 0.4900184084648123,
         'max_bin': 184,
         'max_depth': 2,#6
         'min_data_in_leaf': 20, 
         'min_gain_to_split': 1.3900000000000001,
         'min_sum_hessian_in_leaf': 0.9550109807879941, 
         'num_leaves': 123}

# param = {'bagging_fraction': 1,
#          'n_estimators': 2000,
#          'bagging_freq': 4,
#          'boosting': 'gbdt',
#          'cat_l2': 4,
#          'cat_smooth': 8,
#          'feature_fraction': 1, 
#          'lambda_l1': 0.3649958771024604,
#          'lambda_l2': 0.39118421353320576,
#          'learning_rate': 0.305125828911035,
#          'max_bin': 33,
#          'max_depth': 2,#6
#          'min_data_in_leaf': 100,
#          'min_gain_to_split': 4.46,
#          'min_sum_hessian_in_leaf': 0.0014733825324006822,
#          'num_leaves': 27}

#model = lgb.LGBMRegressor(**param, random_state=13) ### CHANGE not SKLERN
# model.fit(X_train, np.log1p(y_train))
# preds = model.predict(X_test)
# cost(np.log1p(y_test), preds) #47

pred_lgbm = cluster_id('lgb', train_new, test_new, 'building_id', model_empty, num_cluster,  num_iters,  skip = True, param = param)

print('Lgb predict end')
####cat

param = {'border_count': 237,
         'iterations': 10,#1000
         #'grow_policy': 'Lossguide',#gpu
         'l2_leaf_reg': 16,
         'learning_rate': 0.3857377455824425,
         'max_depth': 14, 
         #'max_leaves': 34,#gpu
         #'min_data_in_leaf': 75,  #gpu    
         'iterations':10,
         'random_strength': 6,
         'eval_metric': 'RMSE',
          'random_seed':13,
          'verbose':25,
          #'task_type': 'GPU',
          'od_type':'Iter',    
          'od_wait': 20 }

# param = {'border_count': 111, 
#          'grow_policy': 'Depthwise',
#          'l2_leaf_reg': 126,
#          'learning_rate': 0.30065425194784257,
#          'max_depth': 16,
#          #'max_leaves': 54,
#          'min_data_in_leaf': 90,
#          'random_strength': 10,
#          'iterations':2000,
#          'eval_metric': 'RMSE',
#          'random_seed':13,
#          'verbose':25,
#          'task_type': 'GPU',
#          'od_type':'Iter',    
#          'od_wait': 20 }


model_cat = CatBoostRegressor(**param)
# model.fit(train_pool, eval_set=validation_pool, verbose= True)
# cost(np.log1p(y_test), model.predict(validation_pool))
pred_cat = cluster_id('cat', train_new, test_new, 'building_id', model_cat, num_cluster,num_iters, skip = True, param = None)
print('Cat predict end')

####lasso
model_lasso = Lasso(alpha = 1, random_state=13)
pred_lasso = cluster_id('', train_new, test_new, 'building_id', model_lasso, num_cluster,num_iters, skip = True, param = None)
print('Lasso predict end')

#long ago in a distant galaxy :))

# pred_df = pd.concat([pred_knn, pred_lasso], axis =1).mean(axis = 1)
# pred_df = pd.concat([pred_lgbm, pred_knn, pred_lasso], axis =1).mean(axis = 1)
# pred = pred_cat * 0.6 + pred_lgbm * 0.2 + pred_df * .20
# pred = pred_cat * 0.6 + pred_df * .40
# pred = pred_cat * 0.3 + pred_lgbm * 0.3 + pred_knn * 0.2 + pred_lasso * 0.2
pred = pred_cat * 0.6 + pred_lgbm * 0.25 + pred_knn * 0.15
# pred = pred_cat * 0.45 + pred_lgbm * 0.25 + pred_knn * 0.15 + pred_lasso *0.15
# pred = pred_cat * 0.25 + pred_lgbm * 0.45 + pred_knn * 0.30
# pred = pred_cat * 0.3 + pred_lgbm * 0.55 + pred_knn * 0.15


print('Saved predict, still a bit!!')

predictions = pd.DataFrame({
    "row_id": test_new.index,
    "meter_reading": np.clip(np.expm1(pred['pred']), 0, None)
})

path = '/kaggle/working/'
predictions.to_csv(path + "submission_f_v048.csv", index=False, float_format="%.4f")
print('Script complite!')

print('Ends ........... ')

#kagle scores 1.1568