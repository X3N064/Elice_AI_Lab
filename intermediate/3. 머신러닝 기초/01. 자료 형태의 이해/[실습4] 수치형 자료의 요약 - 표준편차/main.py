from statistics import stdev
import numpy as np

coffee = np.array([202,177,121,148,89,121,137,158])

"""
1. 표준편차 계산
"""
cf_std = stdev(coffee)

# 소수점 둘째 자리까지 반올림하여 출력합니다. 
print("Sample std.Dev : ", round(cf_std,2))
