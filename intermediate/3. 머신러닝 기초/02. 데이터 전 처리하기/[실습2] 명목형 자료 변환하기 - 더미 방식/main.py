import pandas as pd
from elice_utils import EliceUtils

elice_utils = EliceUtils()
   
# 데이터를 읽어옵니다.
titanic = pd.read_csv('./data/titanic.csv')
print('변환 전: \n',titanic['Embarked'].head())

"""
1. get_dummies를 사용하여 변환합니다.
"""
dummies = pd.get_dummies(titanic[['Embarked']])

# 변환한 Embarked 데이터를 출력합니다.
print('\n변환 후: \n',dummies.head())

