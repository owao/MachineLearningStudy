import os
from os.path import join
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn import metrics
import xgboost as xgb
import lightgbm as lgb
import seaborn as sns


#데이터 불러오기

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')
submission = pd.read_csv('./data/sample_submission.csv')

target = train['price']
del train['price']
train_len = len(train)
data = pd.concat((train, test), axis=0)


#결측치 확인 및 데이터 전처리

print(data.info())  #결측치 없음, data 컬럼만 object형

id = data['id'][train_len:]  #id 컬럼 삭제
del data['id']
data['date'] = data['date'].apply(lambda x : str(x[:4]))  #date 컬럼에서 연도 빼고 자르기
data['date'] = data['date'].apply(lambda i: i[:6]).astype(int)

print(data['condition'], data['grade'])  #condition과 grade 칼럼 비교 -> 컨디션이 좋으면 등급은 6과 6.5가 제일 많음
plt.scatter(data['condition'], data['grade'])
plt.xlabel('condition')
plt.ylabel('grade')
plt.show()

del data['sqft_living15'], data['sqft_lot15']  #sqft_living15와 sqft_lot15 칼럼 삭제(집을 재건축 해서 변동이 생겼을 가능성이 있다면 데이터가 일관적이지 않고, 면적 데이터도 이미 존재함)

skew_columns = ['bedrooms', 'sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement']
for c in skew_columns:  #로그 변환
    data[c] = np.log1p(data[c].values)


#feature engineering
data["buy-build year"] = data["date"] - data["yr_built"]  #구매 년도 - 건축 년도 컬럼 생성
data["buy-rebuild year"] = data["date"] - data["yr_renovated"]  #구매 년도 - 재건축 년도 컬럼 생성(재건축에 결측치가 없으므로)


#데이터 재분리
test = data.iloc[train_len:, :]
train = data.iloc[:train_len, :]


#averaging

gboost = GradientBoostingRegressor(random_state=2019)
xgboost = xgb.XGBRegressor(random_state=2019)
lightgbm = lgb.LGBMRegressor(random_state=2019)

gboost.fit(train, target)
xgboost.fit(train, target)
lightgbm.fit(train, target)

predictions = np.column_stack([gboost.predict(test), xgboost.predict(test), lightgbm.predict(test)])
submission = np.mean(predictions, axis=1)

result = pd.DataFrame({
    'id' : id, 
    'price' : submission
})

my_submission_path = join('./data/', 'submission.csv')
result.to_csv(my_submission_path, index=False)