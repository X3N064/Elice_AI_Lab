import numpy as np
import pandas as pd

data = {
    'korean' : [50, 60, 70],
    'math' : [10, np.nan, 40]
}
df = pd.DataFrame(data, index = ['a','b','c'])
print(df, "\n")

# 각 컬럼별 데이터 개수
col_num = df.count(axis=0)
print(col_num, "\n")

# 각 행별 데이터 개수
row_num = df.count(axis=1)
print(row_num, "\n")

# 각 컬럼별 최댓값
col_max = df.max()
print(col_max, "\n")

# 각 컬럼별 최솟값
col_min = df.min()
print(col_min, "\n")

# 각 컬럼별 합계
col_sum = df.sum()
print(col_sum, "\n")

# 컬럼의 최솟값으로 NaN값 대체
math_min = df['math'].min()
df['math'] = df['math'].fillna(math_min)
print(df, "\n")

# 각 컬럼별 평균
col_avg = df.mean()
print(col_avg, "\n")
