import pandas as pd
from elice_utils import EliceUtils

elice_utils = EliceUtils()

"""
1. 정규화를 수행하는 함수를 구현합니다.
"""
def normal(data):
    
    data = (data-data.min()) / (data.max() - data.min())
    
    return data

# 데이터를 읽어옵니다.
titanic = pd.read_csv('./data/titanic.csv')
print('변환 전: \n',titanic['Fare'].head())

# normal 함수를 사용하여 정규화합니다.
Fare = normal(titanic['Fare'])

# 변환한 Fare 데이터를 출력합니다.
print('\n변환 후: \n',Fare.head())

