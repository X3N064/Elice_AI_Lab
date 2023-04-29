import pandas as pd
import numpy as np
from elice_utils import EliceUtils

elice_utils = EliceUtils()



# 데이터를 읽어옵니다.
titanic = pd.read_csv('./data/titanic.csv')

# Cabin 변수를 제거합니다.
titanic_1 = titanic.drop(columns=['Cabin'])

# 결측값이 존재하는 샘플 제거합니다.
titanic_2 = titanic_1.dropna()

# (Age 값 - 내림 Age 값) 0 보다 크다면 소수점을 갖는 데이터로 분류합니다.
outlier = titanic_2[titanic_2['Age']-np.floor(titanic_2['Age']) > 0 ]['Age']

print('소수점을 갖는 Age 변수 이상치')
print(outlier)
print('이상치 처리 전 샘플 개수: %d' %(len(titanic_2)))
print('이상치 개수: %d' %(len(outlier)))

"""
1. 이상치를 처리합니다.
"""
titanic_3 = titanic_2[titanic_2['Age']-np.floor(titanic_2['Age']) == 0 ]
print('이상치 처리 후 샘플 개수: %d' %(len(titanic_3)))
