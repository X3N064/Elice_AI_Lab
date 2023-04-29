#!/usr/bin/env python
# coding: utf-8

# 
# # [Project] 날씨 변화에 따른 군 감영볌 정보 EDA 분석 및 데이터 시각화
# 
# 

# ---

# ## 프로젝트 목표
# - 분기별 군 감염병(폐렴, 수두, 결핵 등) 현황 정보와 날씨 데이터를 활용하여 기온 변화에 따른 군 감염병과의 EDA 분석 및 시각화 프로젝트 수행

# ---

# ## 데이터 출처 
# - https://www.data.go.kr/data/15083055/fileData.do (국방부 군 감염병 정보)
# - https://data.kma.go.kr/data/grnd/selectAwsRltmList.do?pgmNo=56 (기상청 방재기상관측 자료)

# ---

# ## 프로젝트 개요
# 
# 이번 프로젝트에서는 Pandas 및 Matplotlib를 사용하여 분기별 `군 감염병 현황 데이터`와 `기온 데이터`에 대한 탐색적 데이터 분석(EDA)를 적용하여 색인된 시간 순서로 질병과 기온 변수 레코드의 시계열 데이터를 시각화합니다.
# 
# 탐색적 데이터 분석(EDA) 방법 및 데이터 시각화를 통하여 데이터의 분포와 값을 다양한 각도에서 관찰하며 데이터가 표현하는 현상과 다양한 패턴을 파악할 수 있습니다.

# ---

# ## 데이터 준비
# 국방부는 2015년도부터 2019년도까지 분기별, 질병별로 군 감염병 정보를 제공하고 있습니다.
# 
# 따라서 기상청 기상자료개방포털에서 2015년도부터 2019년도까지의 방재기상관측 데이터의 월별, 지역별 평균기온 정보를 다운로드 받아서 사용합니다.

# ---

# ## 1. 데이터 읽기

# pandas를 사용하여 `군 감염병 정보.csv` 파일과 `월별 기온 정보.csv`파일을 읽고 dataframe 형태로 저장합니다.

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
fp1 = '군 감염병 정보.csv'
fp2 = '월별 기온 정보.csv'


# In[3]:


# csv 파일 읽기
df_disease = pd.read_csv(fp1, encoding='euc-kr') # 군 감염병 정보
df_temperature = pd.read_csv(fp2, parse_dates=['일시'], encoding='euc-kr') # 월별 기온 정보, '일시' 데이터를 datetime으로 설정


# In[4]:


# 군 감염병 정보의 행(row)과 열(column)의 개수를 확인해보기
df_disease.shape


# In[5]:


# 군 감염병 정보 확인하기, 첫 5개 데이터 출력
df_disease.head()


# In[6]:


# 월별 기온 정보의 행(row)과 열(column)의 개수를 확인해보기
df_temperature.shape


# In[7]:


# 월별 기온 정보 확인하기, 첫 5개 데이터 출력
df_temperature.head()


# ---

# ## 2. 데이터 정제

# 일반적으로 데이터 정제에서는 **결측값(missing value)** 또는 **이상치(outlier)**를 처리합니다.

# ### 2.1. 결측값 확인

# 아래 코드를 수행하여 `군 감염병 정보`의 각 변수별로 결측값이 몇개가 있는지 확인합니다.

# In[8]:


# 각 열의 빈 데이터 개수 확인하기
df_disease[['연도', '분기구분', '질병명', '현황', '비고']].isna().sum()


# 아래 코드를 수행하여 `월별 기온 정보`의 각 변수별로 결측값이 몇개가 있는지 확인합니다.

# In[9]:


# 각 열의 빈 데이터 개수 확인하기
df_temperature[['지점', '지점명', '일시', '평균기온(°C)']].isna().sum()


# 두 정보 모두 결측치가 없습니다. 
# 이번 프로젝트는 시각화를 통한 EDA 분석이기 때문에 이상치 처리는 하지 않겠습니다.

# ---

# ## 3. 데이터 시각화

# 분기별 감염병 데이터의 시각화를 수행하겠습니다.
# 
# 먼저 분기별 감염병 현황 그래프로 그려봅니다.

# ### 3.1. 분기별 감염병 현황 시각화

# 이번 실습에서는 군 감염병 정보의 `현황` 컬럼 데이터를 시각화합니다. 
# 
# 분기별로 처리하기 위해 `연도`와 `분기구분` 컬럼 데이터를 인덱스로 만들어서 처리합니다.

# In[10]:


# 데이터 프레임의 groupby 메서드를 활용하여 연도와 분기구분별로 감염병 현황의 합계를 구합니다.
df_disease2 = df_disease.groupby(['연도', '분기구분'])['현황'].sum()


# In[11]:


df_disease2


# In[12]:


# 데이터 프레임의 인덱스를 재설정하여 연도와 분기구분 컴럼을 설정합니다.
df_temp = df_disease2.reset_index()


# In[13]:


df_temp


# In[14]:


df_temp.dtypes


# In[15]:


# 분기별 시각화를 위해 연도와 분기기분 컴럼을 병합합니다.
# 위 셀에서 확인한 바와 같이, int64 타입 데이터를 문자열로 변환하여 병합하였습니다.
df_temp['분기'] = df_temp['연도'].astype(str) + '-' + df_temp['분기구분'].astype(str)


# In[16]:


df_temp


# In[17]:


# 그래프 스타일 지정하기
plt.style.use('seaborn-whitegrid')
plt.rc('font', family="NanumBarunGothic") # 한글 설정

# 그래프 속성 설정하기
fig, ax = plt.subplots(1, figsize=(10, 6))

# X축, Y축 데이터 지정하기
x = df_temp.분기
y = df_temp.현황

# 그래프 데이터와 속성 설정하기
plt.plot(x, y, color = 'red',  linestyle='-', marker='o',  mfc='blue')

fig.autofmt_xdate()

# 그래프 제목 및 속성 설정하기
plt.title('분기별 군감염병 현황', fontdict={'fontsize': 20, 'weight': 'bold'})

# X, Y축에 제목 설정하기
plt.xlabel('분기', fontdict={'fontsize': 14})
plt.ylabel('감염병 현황', fontdict={'fontsize': 14})

# 그래프안에 데이터 설명(레전드) 넣기
plt.legend(['감염병 현황'], frameon=True, fontsize='x-large')

plt.tight_layout()


# ---

# ### 3.2. 분기별 기온 시각화

# 이번에는 `월별 기온 정보` 에 있는 데이터를 분기별로 시각화합니다. 
# 
# `월별 기온 정보`를 확인해보면 각 지점별로 월평균기온 데이터를 확인할 수 있습니다.
# 
# DataFrame의 resample이라는 함수를 사용하면 기간별로 데이터를 쉽게 집계할 수 있습니다. 
# 
# 분기별로 집계하기 위해서 'Q' 파리미터를 활용하겠습니다.

# In[18]:


df_temperature


# In[19]:


# '일시' 열을 인덱스로 지정하기
df_temperature.set_index('일시', inplace=True)
df_temperature.head()


# In[20]:


# resample 메서드를 활용하여 분기별로 모든 관측지역의 평균 기온을 구하기
quarterly_data = round(df_temperature['평균기온(°C)'].resample('Q').mean(), 2)
quarterly_data.head()


# In[21]:


# 분기별 평균 기온 데이터 인덱스 재설정 하여 '일시' 데이터를 열로 만들기
quarterly_data = quarterly_data.reset_index()
quarterly_data.head()


# In[22]:


# '일시'컬럼의 데이터를 '분기'컬럼으로 저장하기
quarterly_data['분기'] = quarterly_data['일시'].dt.year.astype(str) + '-' + quarterly_data['일시'].dt.quarter.astype(str) 
quarterly_data.head()


# In[23]:


# 그래프 스타일 지정하기
plt.style.use('seaborn')
plt.rc('font', family="NanumBarunGothic") # 한글 설정

# 그래프 속성 설정하기
fig, ax1 = plt.subplots(1, figsize=(10, 6))

# x, y축 데이터 설정하기
x = quarterly_data['분기']
y = quarterly_data['평균기온(°C)']

# y축 제한 설정하기
plt.ylim(-5, 30)

# 그래프 데이터와 속성 설정하기
plt.plot(x, y, color = 'tomato',  linestyle='-', marker='o',  mfc='orange', linewidth = 3, markersize = 8)

#plt.grid(axis = 'x')
fig.autofmt_xdate()

# 그래프 제목 및 속성 설정하기
plt.title('분기별 평균 기온', fontdict={'fontsize': 20, 'weight': 'bold'})

# X, Y축에 제목 설정하기
plt.xlabel('분기', fontdict={'fontsize': 14})
plt.ylabel('평균기온', fontdict={'fontsize': 14})

plt.tight_layout()


# ---

# ### 3.3. 분기별 감염병 현황과 평균 기온 시각화

# 이번에는 두 데이터를 합쳐서 시각화해봅니다.

# In[24]:


# 먼저 분기별 평균 기온 데이터를 확인합니다.
quarterly_data


# In[25]:


# 다음으로 분기별 감염병 현황 데이터를 확인합니다.
df_temp


# 두 데이터를 확인하면, 분기 데이터를 기준으로 두 데이터를 병합할 수 있습니다.

# In[26]:


df_merge = df_temp.merge(quarterly_data, how='left', on='분기')[['분기', '현황', '평균기온(°C)']]
df_merge


# 병합된 데이터프레임을 시각화 합니다.

# In[27]:


# 데이터 설정하기
x = df_merge['분기']
y1 = df_merge['평균기온(°C)']
y2 = df_merge['현황']

plt.style.use('seaborn-white')
plt.rc('font', family="NanumBarunGothic") # 한글 설정

# 그래프 생성
fig, ax1 = plt.subplots()

# 선그래프로 평균 기온을 그려줍니다.
ax1.plot(x, y1, linestyle='--', marker='o',  mfc='blue', label='평균기온(°C)')
ax1.set_xlabel('분기')
ax1.set_ylabel('평균기온(°C)')
plt.ylim(0, 30)

# twinx 메서드를 활용하여 y축이 다른 막대 그래프를 그립니다.
ax2 = ax1.twinx()
ax2.bar(x, y2, color='deeppink', label='감염병 현황', alpha=0.7, width=0.7)
ax2.set_ylabel('감염병 현황')

# 
leg1, leg1_label = ax1.get_legend_handles_labels()
leg2, leg2_label = ax2.get_legend_handles_labels()
ax2.legend(leg1 + leg2, leg1_label + leg2_label)

fig.autofmt_xdate()


# ### 3.4. 감염병별 현황 시각화

# 감염병별 현황을 시각화하기 위해서 감염병 현황 데이터프레임 `df_disease`를 확인해보겠습니다.

# In[28]:


df_disease


# 위 데이터프레임은 연도-분기-질병으로 데이터가 나열되어 있는데, 감염병 별로 데이터를 쉽게 확인하기 위해서 `groupby`함수를 써서 질병-연도-분기 형태도 변경하겠습니다. 
# 
# 이때 `현황`데이터의 합계를 컬럼값으로 설정하였습니다.

# In[29]:


df_disease3 = df_disease.groupby(['질병명', '연도', '분기구분'])['현황'].sum()
df_disease3


# 위 데이터프레임의 인덱스를 재설정하여 아래와 같이 데이터를 만들어줍니다.

# In[30]:


df_temp2 = df_disease3.reset_index()
df_temp2


# `연도` 컬럼과 `분기구분` 컬럼 데이터를 합쳐서 `분기` 컬럼데이터를 생성하여 연도별-분기 데이터를 만들겠습니다.

# In[31]:


df_temp2['분기'] = df_temp2['연도'].astype(str) + '-' + df_temp2['분기구분'].astype(str)
df_temp2


# In[32]:


# 감염병별로 현황을 비교하기 위해 피봇 테이블 생성하기
df_disease4 = df_temp2.pivot_table(values = '현황', index = '분기', columns = '질병명')
df_disease4.head()


# 감염병별 현황을 시각화하겠습니다.

# In[33]:


# 그래프 스타일 지정하기
plt.style.use('seaborn')
plt.rc('font', family="NanumBarunGothic") # 한글 설정

# 그래프 속성 설정하기
fig, ax1 = plt.subplots(1, figsize=(15, 11))

# x, y축 데이터 설정하기
x = df_disease4.reset_index()['분기']

y_1 = df_disease4['A형 간염']
y_2 = df_disease4['결 핵']
y_3 = df_disease4['렙토스피라증']
y_4 = df_disease4['말라리아']
y_5 = df_disease4['매 독']
y_6 = df_disease4['세균성 이질']
y_7 = df_disease4['수 두']
y_8 = df_disease4['수막구균성 수막염']
y_9 = df_disease4['신증후군 출혈열']
y_10 = df_disease4['유행성 이하선염']
y_11 = df_disease4['장티푸스']
y_12 = df_disease4['파상풍']
y_13 = df_disease4['폐 렴']
y_14 = df_disease4['홍 역']

# 그래프 데이터와 속성 설정하기
plt.rc('font', family="NanumBarunGothic")
plt.plot(x, y_1, '.-b', label = 'A형 간염', linewidth = 2)
plt.plot(x, y_2, '.-g', label = '결 핵', linewidth = 2)
plt.plot(x, y_3, '.-.r', label = '렙토스피라증', linewidth = 2)
plt.plot(x, y_4, '.:c', label = '말라리아', linewidth = 2)
plt.plot(x, y_5, '.:m', label = '매 독', linewidth = 2)
plt.plot(x, y_6, '.:y', label = '세균성 이질', linewidth = 2)
plt.plot(x, y_7, '.:k', label = '수 두', linewidth = 2)
plt.plot(x, y_8, '.--b', label = '수막구균성 수막염', linewidth = 2)
plt.plot(x, y_9, '.--g', label = '신증후군 출혈열', linewidth = 2)
plt.plot(x, y_10, '.--r', label = '유행성 이하선염', linewidth = 2)
plt.plot(x, y_11, '.--c', label = '장티푸스', linewidth = 2)
plt.plot(x, y_12, '.--m', label = '파상풍', linewidth = 2)
plt.plot(x, y_13, '.--y', label = '폐 렴', linewidth = 2)
plt.plot(x, y_14, '.--k', label = '홍 역', linewidth = 2)

leg = plt.legend()
# 그래프안에 선별 데이터 설명(범주박스) 넣기
leg_lines = leg.get_lines()
leg_texts = leg.get_texts()

# 한꺼번에 선별 속성 
plt.setp(leg_lines, linewidth=2)
plt.setp(leg_texts, fontsize='x-large')

# 그래프 제목 및 속성 설정하기
plt.title('감염병별 질병 발생 현황', fontdict={'fontsize': 15, 'weight': 'bold'})

# X, Y축에 제목 설정하기
plt.xlabel('분기', fontdict={'fontsize': 14})
plt.ylabel('감염병 현황', fontdict={'fontsize': 14})


# 폐렴 데이터가 2018년도 3분기를 기준으로 많은 데이터 발생하고 있습니다.

# In[34]:


df_disease4['폐 렴']


# 마지막으로 감염병 별로 시각화를 하겠습니다.

# In[35]:


# 그래프 스타일 지정하기
plt.style.use('seaborn-whitegrid')
plt.rc('font', family="NanumBarunGothic") # 한글 설정

# 14개의 감염병을 그려주기 위해서 row=5, column=3 으로 subplot을 설정합니다.
fig, axes = plt.subplots(5, 3, figsize=(16, 16))

# x, y축 데이터 설정하기
x = df_disease4.reset_index()['분기']

y_1 = df_disease4['A형 간염']
y_2 = df_disease4['결 핵']
y_3 = df_disease4['렙토스피라증']
y_4 = df_disease4['말라리아']
y_5 = df_disease4['매 독']
y_6 = df_disease4['세균성 이질']
y_7 = df_disease4['수 두']
y_8 = df_disease4['수막구균성 수막염']
y_9 = df_disease4['신증후군 출혈열']
y_10 = df_disease4['유행성 이하선염']
y_11 = df_disease4['장티푸스']
y_12 = df_disease4['파상풍']
y_13 = df_disease4['폐 렴']
y_14 = df_disease4['홍 역']

# 그래프 데이터와 속성 설정하기
axes[0][0].plot(x, y_1, '.-b', label = 'A형 간염')
axes[0][0].legend(fontsize=15)
axes[0][0].set_title('A형 간염', size=20)

axes[0][1].plot(x, y_2, '.-g', label = '결 핵')
axes[0][1].legend(fontsize=15)
axes[0][1].set_title('결 핵', size=20)

axes[0][2].plot(x, y_3, '.-.r', label = '렙토스피라증')
axes[0][2].legend(fontsize=15)
axes[0][2].set_title('렙토스피라증', size=20)

axes[1][0].plot(x, y_4, '.:c', label = '말라리아')
axes[1][0].legend(fontsize=15)
axes[1][0].set_title('말라리아', size=20)

axes[1][1].plot(x, y_5, '.:m', label = '매 독')
axes[1][1].legend(fontsize=15)
axes[1][1].set_title('매 독', size=20)

axes[1][2].plot(x, y_6, '.:y', label = '세균성 이질')
axes[1][2].legend(fontsize=15)
axes[1][2].set_title('세균성 이질', size=20)

axes[2][0].plot(x, y_7, '.:k', label = '수 두')
axes[2][0].legend(fontsize=15)
axes[2][0].set_title('수 두', size=20)

axes[2][1].plot(x, y_8, '.--b', label = '수막구균성 수막염')
axes[2][1].legend(fontsize=15)
axes[2][1].set_title('수막구균성 수막염', size=20)

axes[2][2].plot(x, y_9, '.--g', label = '신증후군 출혈열')
axes[2][2].legend(fontsize=15)
axes[2][2].set_title('신증후군 출혈열', size=20)

axes[3][0].plot(x, y_10, '.--r', label = '유행성 이하선염')
axes[3][0].legend(fontsize=15)
axes[3][0].set_title('유행성 이하선염', size=20)

axes[3][1].plot(x, y_11, '.--c', label = '장티푸스')
axes[3][1].legend(fontsize=15)
axes[3][1].set_title('장티푸스', size=20)

axes[3][2].plot(x, y_12, '.--m', label = '파상풍')
axes[3][2].legend(fontsize=15)
axes[3][2].set_title('파상풍', size=20)

axes[4][0].plot(x, y_13, '.--y', label = '폐 렴')
axes[4][0].legend(fontsize=15)
axes[4][0].set_title('폐 렴', size=20)

axes[4][1].plot(x, y_14, '.--k', label = '홍 역')
axes[4][1].legend(fontsize=15)
axes[4][1].set_title('홍 역', size=20)

axes[4][2].axis('off') # 빈 그래프를 그리지 않습니다.

fig.autofmt_xdate(rotation=45)


# ---

# ## 퀴즈

# `A형 간염`병에 대한 현황과 평균 기온의 시각화하기 위한 데이터프레임을 df_result에 저장하세요.
# 
# 2번째 줄의 None 부분에 알맞은 코드를 넣어 해결합니다.

# In[36]:


df_temp = df_disease4.reset_index()
df_result = df_temp.merge(quarterly_data, how='left', on='분기')[['분기', 'A형 간염', '평균기온(°C)']]
df_result


# ## 제출하기
# 
# 퀴즈 수행 후, 아래 코드를 실행하면 `df_result` 변수가 저장된 `submission.pickle` 파일을 제작하여 채점을 받을 수 있습니다.
# 
# **아래 코드를 수정하면 채점이 불가능 합니다.**

# In[37]:


import pickle

with open('submission.pickle', 'wb') as f:
    pickle.dump(df_result, f)


# In[38]:


# 채점을 수행하기 위하여 로그인
import sys
sys.path.append('vendor')
from elice_challenge import check_score, upload


# In[39]:


# 제출 파일 업로드
await upload()


# In[40]:


# 채점 수행
await check_score()


# In[ ]:




