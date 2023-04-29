import numpy as np


coffee = np.array([202,177,121,148,89,121,137,158])

"""
1. 평균계산
"""
cf_mean = np.mean(coffee)

# 소수점 둘째 자리까지 반올림하여 출력합니다. 
print("Mean :", round(cf_mean,2))
