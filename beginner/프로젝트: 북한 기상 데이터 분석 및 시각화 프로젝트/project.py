#!/usr/bin/env python
# coding: utf-8

# 
# # [Project] 북한 기상 데이터 분석 및 시각화 프로젝트
# 
# 

# ---

# ## 프로젝트 목표
# - 세계기상 통신망을 통해 입력된 북한의 기온, 기압, 바람, 강수량 등의 전문자료를 시간, 일자료로 탐색적 데이터 분석(EDA) 기법을 적용하여 시각화 프로젝트 수행

# ---

# ## 데이터 출처 
# - https://www.data.go.kr/data/15043645/fileData.do

# ---

# ## 프로젝트 개요
# 
# 이번 프로젝트에서는 Pandas 및 Matplotlib를 사용하여 기후 데이터에 탐색적 데이터 분석(EDA)을 적용합니다. 이를 통해 날씨 변수를 시간 순서대로 시각화합니다.
# 
# 탐색적 데이터 분석(EDA) 방법 및 데이터 시각화를 통하여 데이터의 분포와 값을 다양한 각도에서 관찰하며 데이터가 표현하는 현상과 다양한 패턴을 파악할 수 있습니다.

# ---

# ## 1. 데이터 읽기

# pandas를 사용하여 `project_data.csv` 데이터를 읽고 dataframe 형태로 저장해 봅시다.

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.font_manager as fm

# 엘리스 환경에서 한글 폰트를 사용하기 위한 코드입니다.
font_dirs = ['/usr/share/fonts/truetype/nanum', ]
font_files = fm.findSystemFonts(fontpaths=font_dirs)

for font_file in font_files:
    fm.fontManager.addfont(font_file)


# In[2]:


# 데이터 파일 경로 정의하기
fp = 'project_data.csv'


# In[3]:


# csv 파일 읽기
data = pd.read_csv(fp, parse_dates=['일시'], dayfirst=True, encoding='euc-kr')


# In[4]:


# 데이터 타입 확인하기
data.dtypes


# In[5]:


# 열 이름 출력하기 
data.columns.values


# In[6]:


# 행 수, 열 수 등 데이터프레임 모양 출력하기
data.shape


# In[7]:


# 데이터 확인하기, 첫 5개 데이터 출력
data.head()


# In[8]:


# 데이터 확인하기. 마지막 5개 데이터 출력
data.tail()


# ---

# ## 2. 데이터 정제

# 일반적으로 데이터 정제에서는 **결측값(missing value)** 또는 **이상치(outlier)**를 처리합니다.

# ### 2.1. 결측값 확인

# 아래 코드를 수행하여 각 변수별로 결측값이 몇개가 있는지 확인합니다.
# 
# 이번 프로젝트는 평균 기온을 월별, 지역별로 시각화합니다. 따라서 별도의 처리는 하지 않습니다.

# In[9]:


# 각 열의 빈 데이터 개수 확인하기
data[['평균기온', '평균이슬점온도', '평균현지기압', '평균풍속', '합계강수량', '평균전운량', '최고기온', '평균습도', '평균해면기압', '중하층운량', '최저기온']].isna().sum()


# ---

# ## 3. 데이터 시각화

# 평균 기온 데이터의 시각화를 수행하겠습니다.
# 
# 먼저 일자별 평균 기온을 그래프로 그려봅니다. 다음으로 일자별 평균 기온을 바탕으로 월별 평균 기온을 시각화하고 지역별로도 확인하겠습니다.

# ### 3.1. 일평균 기온 시각화

# 이번 실습에서는 `평균기온` 에 있는 데이터를 시각화합니다. 
# 
# 일자별로 처리하기 위해 `일시` 데이터를 인덱스로 만들어서 처리합니다.

# In[10]:


# '일시' 열을 인덱스로 지정하기
data.set_index('일시', inplace=True)
data.head()


# In[11]:


# 2020년 1월 1일의 데이터 확인
data.loc['20200101']


# 위와 같이 일자별로 '평균기온' 데이터를 확인하면 많은 지역의 데이터가 있습니다.
# 
# '평균기온' 데이터를 일자별로 확인하기 위해 먼저 전체 지역에서 평균한 값을 구할 것입니다.
# 
# 이를 위해 먼저 데이터들을 날짜별로 묶는 작업이 필요합니다.
# 
# 이는 resample 메서드를 사용하며, 인자로는 'D'를 넘겨주면 가능합니다.
# 
# 이후 날짜별로 묶인 데이터에 mean 메서드를 적용하면 전체 지역에서의 평균기온을 구할 수 있습니다.

# In[12]:


# 일자별로 모든 관측지역의 평균 기온을 구하기
daily_temp = round(data['평균기온'].resample('D').mean(), 2)
daily_temp.head()


# In[13]:


# 일자별 평균 기온에서 가장 낮은 기온과 가장 높은 기온을 구하기
print(daily_temp.min(),'°C')
print('')
print(daily_temp.max(),'°C')


# In[14]:


# 일자별 평균 기온에서 기온이 가장 낮은 날짜와 기온이 가장 높은 날짜 구하기
min_daily = daily_temp.loc[daily_temp == -14.33]
print(min_daily)
print(' ')
max_daily = daily_temp.loc[daily_temp == 25.61]
print(max_daily)


# 인덱스로 지정된 `일시`  데이터를 해제해서, 그래프에 일시와 평균기온을 각각 X축과 Y축 데이터로 지정합니다.

# In[15]:


# 일자별 평균 기온 데이터 인덱스 재설정 하여 '일시' 데이터를 열로 만들기
daily_temp = daily_temp.reset_index()
daily_temp.head()


# In[16]:


# 그래프 스타일 지정하기
plt.style.use('seaborn-whitegrid')

# 그래프 속성 설정하기
fig, ax = plt.subplots(1, figsize=(10, 6))

# X축, Y축 데이터 지정하기
x = daily_temp.일시
y = daily_temp.평균기온

# 그래프 데이터와 속성 설정하기
plt.plot(x, y, color = 'red',  linestyle='-', marker='o',  mfc='blue')

fig.autofmt_xdate()

# 그래프 제목 및 속성 설정하기
plt.title('Daily Mean Temperatures, North Korea', fontdict={'fontsize': 20, 'weight': 'bold'})

# X, Y축에 제목 설정하기
plt.xlabel('Day', fontdict={'fontsize': 14})
plt.ylabel('Temperature °C', fontdict={'fontsize': 14})

# 그래프안에 데이터 설명(레전드) 넣기
plt.legend(['Mean Temperature'], frameon=True, fontsize='x-large')

plt.tight_layout()


# ---

# ### 3.2. 월평균 기온 시각화

# 이번에는 `평균기온` 에 있는 데이터를 월별로 시각화합니다. 
# 
# 월별로 데이터를 묶기 위하여 resample 메서드에 'M'을 넘겨줍니다.

# In[17]:


# 월별로 모든 관측지역의 평균 기온을 구하기
monthly_data = round(data.resample('M').mean(), 2)
monthly_data.head()


# In[18]:


# 월별 평균 기온 데이터 인덱스 재설정 하여 '일시' 데이터를 열로 만들기
monthly_data.reset_index(inplace=True)
monthly_data.head()


# `일시`데이터를 보기 좋게 표시하기 위해서 `월`데이터를 만들어줍니다.

# In[19]:


# '일시'컬럼의 월명을 '월'컬럼으로 저장하기
monthly_data['월'] = monthly_data['일시'].dt.month_name().str.slice(stop=3)
monthly_data.head()


# In[20]:


# 그래프 스타일 지정하기
plt.style.use('seaborn')

# 그래프 속성 설정하기
fig, ax1 = plt.subplots(1, figsize=(10, 6))

# x, y축 데이터 설정하기
x = monthly_data['월']
y = monthly_data['평균기온']

# y축 제한 설정하기
plt.ylim(-15, 30)

# 그래프 데이터와 속성 설정하기
plt.plot(x, y, color = 'tomato',  linestyle='-', marker='o',  mfc='orange', linewidth = 3, markersize = 8)

plt.grid(axis = 'x')

# 그래프 제목 및 속성 설정하기
plt.title('Monthly Mean Temperatures, North Korea', fontdict={'fontsize': 20, 'weight': 'bold'})

# X, Y축에 제목 설정하기
plt.xlabel('Month', fontdict={'fontsize': 14})
plt.ylabel('Temperature °C', fontdict={'fontsize': 14})


# ---

# ### 3.3. 지역별 기온 시각화

# 이번에는 `평균기온` 에 있는 데이터를 지역별로 시각화합니다. 
# 
# 원하는 지역별로 그룹을 만들기 위해서 `지점`데이터를 가공해서 `도`데이터를 만들어줍니다.

# In[22]:


# 도(행정구역) 별로 데이터 열 새로 만들기
data['도'] = data['지점'].str.split('_').str[0]
data['도'] = data['도'].replace({'함경북도': '함경도', '함경남도': '함경도', 
                             '평안북도': '평안도', '평안남도': '평안도', 
                             '황해북도': '황해도', '황해남도': '황해도'})
data.head()


# In[23]:


# 도(행정구역) 별 월평균 기온 구하기
zones_monthly = data.groupby('도').resample('M').mean()
zones_monthly.head()


# In[24]:


# 도별 인덱스 재설정 하여 '도' 데이터를 열로 만들기
zones_monthly.reset_index(inplace=True)
zones_monthly.head()


# In[25]:


# 월별로 도별 평균 기온을 비교하기 위해 피봇 테이블 생성하기
zones_monthly = zones_monthly.pivot_table(values = '평균기온', index = '일시', columns = '도')
zones_monthly.head()


# 먼저 히트맵을 이용해서 지역별로 월평균 기온을 확인해보면, 여름에 더 붉은색으로 표시되는 것을 확인할 수 있습니다.

# In[26]:


# 한글 설정
plt.rc('font', family="NanumBarunGothic")
plt.rc('axes', unicode_minus=False)

# 히트맵 설정
plt.pcolor(zones_monthly, cmap='Reds')
plt.xticks(np.arange(0.5, len(zones_monthly.columns), 1), zones_monthly.columns)
plt.yticks(np.arange(0.5, len(zones_monthly.index), 1), zones_monthly.index)
plt.title('Monthly Mean Temperatures Heatmap, North Korea', fontsize=20)
plt.xlabel('Zone', fontsize=14)
plt.ylabel('Month', fontsize=14)
plt.colorbar()
plt.show()


# 마지막으로 지역별로 선그래프를 그려서 데이터를 확인합니다.
# 
# 이와 같은 방식으로 `평균습도`, `평균풍속` 데이터를 분석할 수 있습니다.

# In[27]:


# 그래프 스타일 지정하기
plt.style.use('seaborn')

# 그래프 속성 설정하기
fig, ax1 = plt.subplots(1, figsize=(15, 11))

# x, y축 데이터 설정하기
x = zones_monthly.reset_index()['일시']

y_1 = zones_monthly['강원도']
y_2 = zones_monthly['양강도']
y_3 = zones_monthly['자강도']
y_4 = zones_monthly['평안도']
y_5 = zones_monthly['함경도']
y_6 = zones_monthly['황해도']

# y축 제한 설정하기
plt.ylim(-15, 30)

# 그래프 데이터와 속성 설정하기
plt.rc('font', family="NanumBarunGothic")
plt.plot(x, y_1, '.-b', label = '강원도', linewidth = 2)
plt.plot(x, y_2, '.-g', label = '양강도', linewidth = 2)
plt.plot(x, y_3, '.-.r', label = '자강도', linewidth = 2)
plt.plot(x, y_4, '.:c', label = '평안도', linewidth = 2)
plt.plot(x, y_5, '.:m', label = '함경도', linewidth = 2)
plt.plot(x, y_6, '.:y', label = '황해도', linewidth = 2)

leg = plt.legend()
# 그래프안에 선별 데이터 설명(범주박스) 넣기
leg_lines = leg.get_lines()
leg_texts = leg.get_texts()

# 한꺼번에 선별 속성 
plt.setp(leg_lines, linewidth=4)
plt.setp(leg_texts, fontsize='x-large')

# 그래프 제목 및 속성 설정하기
plt.title('Monthly Mean Temperature per Zone, North Korea', fontdict={'fontsize': 15, 'weight': 'bold'})

# X, Y축에 제목 설정하기
plt.xlabel('Month', fontdict={'fontsize': 14})
plt.ylabel('Temperature °C', fontdict={'fontsize': 14})


# ---

# ## 퀴즈
# 마지막으로 구한 `zones_monthly`에서 `일시`와 `도`에 따른 `평균기온`이 아닌 `최고기온`으로 대체한 DataFrame을 `zones_mothly_max`에 저장하세요.
# 
# 3번째 줄의 `None` 부분에 알맞은 코드를 넣어 해결합니다.

# In[ ]:


zones_mothly_max = data.groupby('도').resample('M').mean()
zones_mothly_max.reset_index(inplace=True)
zones_mothly_max = zones_mothly_max.pivot_table(values = '최고기온', index = '일시', columns = '도')
zones_mothly_max


# ## 제출하기
# 
# 퀴즈 수행 후, 아래 코드를 실행하면 `zones_mothly_max` 변수가 저장된 `submission.pickle` 파일을 제작하여 채점을 받을 수 있습니다.
# 
# **아래 코드를 수정하면 채점이 불가능 합니다.**

# In[ ]:


import pickle

d = {'quiz': zones_mothly_max.values}

with open('submission.pickle', 'wb') as f:
    pickle.dump(d, f)


# In[ ]:


# 채점을 수행하기 위하여 로그인
import sys
sys.path.append('vendor')
from elice_challenge import check_score, upload


# In[ ]:


# 제출 파일 업로드
await upload()


# In[ ]:


# 채점 수행
await check_score()

