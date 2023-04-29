#!/usr/bin/env python
# coding: utf-8

# # [Project] 2차 세계대전 공중폭격 및 날씨 데이터 시계열 분석을 통한 폭격 시점 예측 프로젝트

# ## 프로젝트 목표
# 
# - 기후 변화 시계열 데이터 및 공중폭격 시점을 시각화하여 EDA 분석을 수행하고, 기후 데이터를 예측하여 실제 데이터와 비교하는 프로젝트 수행
# 

# ## 데이터 출처
# - https://www.kaggle.com/datasets/usaf/world-war-ii
# - https://www.kaggle.com/code/kanncaa1/time-series-prediction-tutorial-with-eda/

# ## 프로젝트 개요
# 
# 
# - EDA(탐색적 데이터 분석: Exploratory Data Anlysis)을 위해 데이터 전처리 및 시각화를 수행합니다.
# 
# - 날씨 데이터의 예측을 위해 시계열 정상성(Stationarity)을 체크하고 이를 만족시키기 위한 기법을 이해합니다.
# 
# - 예측한 날씨데이터와 실제데이터를 비교하여 상관성을 분석합니다.

# ### 데이터 베이스
# 
# - 이 데이터 베이스는 2차 세계대전의 종이 임무 보고서를 디지털화한 것으로써, 교전일자, 지리적 위치 등을 포함하는 데이터가 포함되어 있습니다.
# 
# - 1939년 부터 1945년 까지의 폭격 기록으로써 미국 및 호주, 뉴질랜드 및 영국의 공군 데이터가 포함되어 있습니다.    
# 
# 
#   - 본 실습에서 사용할 데이터에 대한 간략한 설명입니다.
#     - 폭격 데이터: (operations.csv)
#       - Mission Date: 작전 수행 일자
#       - Theater of Operations: 작전(군사) 지역
#       - Country: 미국처럼 작전을 수행한 나라
#       - Air Force: 5AF와 같은 공군 부대의 이름 혹은 ID
#       - Aircraft Series: B24와 같은 항공기의 모델 혹은 유형
#       - Callsign: 폭격 직전 라디오로 송출하는 메시지, 코드, 방송 등
#       - Takeoff Base: 이륙한 공항의 이름
#       - Takeoff Location: 이륙한 지역
#       - Takeoff Latitude: 이륙한 지역의 위도
#       - Takeoff Longitude: 이륙한 지역의 경도
#       - Target Country: 공격 목표 국가 (ex: 독일)
#       - Target City: 공격 목표 도시 (ex: 베를린)
#       - Target Type: 공격 목표 유형 (ex: 도시지역)
#       - Target Industry: 도시나 도회지 같은 목표 산업
#       - Target Priority: 목포물의 우선순위, 1이 가장 높은 순위
#       - Target Latitude: 공격 지점의 위도
#       - Target Longitude: 공격 지점의 경도
#    
#    
#   - Weather Condition data description:
#   
#     - Weather station location(기상관측소): (Weather Station Locations.csv)
#       - WBAN: 기상관측소의 ID
#       - NAME: 기상관측소의 이름
#       - STATE/COUNTRY ID: 국가의 약어
#       - Latitude: 관측소의 위도
#       - Longitude: 관측소의 경도
#     
#     - Weather(날씨): (Summary of Weather.csv)
#       - STA: 관측소 ID (WBAN)
#       - Date: 기온을 측정한 날짜
#       - MeanTemp: 평균 기온

# In[1]:


# 기본으로 설치되어 있지 않은 필요 라이브러리를 설치합니다.
get_ipython().system('pip install plotly chart_studio')


# In[2]:


# python3 환경에서 필요한 라이브러리를 import합니다.
import numpy as np # matrix와 선형 대수를 위한 대표적인 라이브러리
import pandas as pd # 다양한 포맷의 데이터를 일고 처리할 수 있는 라이브러리

# 데이터를 표와 그래프로 표현할 수 있는 라이브러리들
import seaborn as sns 
import matplotlib.pyplot as plt # visualization library
import chart_studio.plotly as py
from plotly.offline import init_notebook_mode, iplot # plotly offline mode
init_notebook_mode() 
import plotly.graph_objs as go # 

# 엘리스 환경에서 한글 폰트를 사용하기 위한 코드입니다.
import matplotlib.font_manager as fm
font_dirs = ['/usr/share/fonts/truetype/nanum', ]
font_files = fm.findSystemFonts(fontpaths=font_dirs)

for font_file in font_files:
    fm.fontManager.addfont(font_file)
    
# 실슴에 사용할 데이터는 input 폴더에 있습니다. input 폴더에 어떤 파일이 있는지 확인합니다.
import os
print(os.listdir("./input/"))


# In[3]:


# 공중 폭격 데이터를 로딩합니다.
aerial = pd.read_csv("./input/operations.csv", low_memory=False)

# 첫 번째 데이터는 위/경도를 포함하는 기상 관측소의 위치 데이터를 읽습니다.
weather_station_location = pd.read_csv("./input/Weather Station Locations.csv", low_memory=False)

# 두 번째로 최소, 최대, 평균 기온등을 포함하는 날씨 정보를 로딩 합니다.
weather = pd.read_csv("./input/Summary of Weather.csv", low_memory=False)


# ---

# ## 데이터 정제
# 
#   - 폭격데이터는 많은 수의 NaN 데이터가 있습니다. NaN 데이터는 데이터분석의 불확정성을 높이고 visualization(시각화) 과정에서 오류를 발생시킬 가능성을 높이기 때문에 적절하게 제거하는 과정이 필요합니다.
#   
#   - 아래의 기준에 따라서 NaN데이터를 제거합니다.
#   
#     - Country 필드에 NaN인 레코드
#     - Target longitude가 NaN 인 레코드
#     - Takeoff Longitude가 NaN인 레코드
#     - 데이터 분석에 사용하지 않는 컬럼
#    
#   - 날씨 데이터는 데이터 정제가 필요하지 않습니다.다만, 분석에 사용하지 않는 컬럼은 지웁니다.
# 
# 

# In[4]:


# country가 NaN인 데이터 삭제
aerial = aerial[pd.isna(aerial.Country)==False]
# target longitude 가 NaN 이면 삭제
aerial = aerial[pd.isna(aerial['Target Longitude'])==False]
# takeoff longitude 가 NaN 이면 삭제
aerial = aerial[pd.isna(aerial['Takeoff Longitude'])==False]

# 분석에 사용하지 않는 컬럼 삭제
drop_list = ['Mission ID','Unit ID','Target ID','Altitude (Hundreds of Feet)','Airborne Aircraft',
             'Attacking Aircraft', 'Bombing Aircraft', 'Aircraft Returned',
             'Aircraft Failed', 'Aircraft Damaged', 'Aircraft Lost',
             'High Explosives', 'High Explosives Type','Mission Type',
             'High Explosives Weight (Pounds)', 'High Explosives Weight (Tons)',
             'Incendiary Devices', 'Incendiary Devices Type',
             'Incendiary Devices Weight (Pounds)',
             'Incendiary Devices Weight (Tons)', 'Fragmentation Devices',
             'Fragmentation Devices Type', 'Fragmentation Devices Weight (Pounds)',
             'Fragmentation Devices Weight (Tons)', 'Total Weight (Pounds)',
             'Total Weight (Tons)', 'Time Over Target', 'Bomb Damage Assessment','Source ID']
aerial.drop(drop_list, axis=1,inplace = True)

aerial = aerial[ aerial.iloc[:,8]!="4248"] # 이 데이터는 삭제
aerial = aerial[ aerial.iloc[:,9]!=1355] # 이 데이터는 삭제
aerial.info()


# In[5]:


# 분석에 사용할 컬럼만 남기고 다른 데이터는 삭제
weather_station_location = weather_station_location.loc[:,["WBAN","NAME","STATE/COUNTRY ID","Latitude","Longitude"] ]
weather_station_location.info()


# In[6]:


# weather 데이터는 평균기온, 관측소 ID, 날짜 만 사용
weather = weather.loc[:,["STA","Date","MeanTemp"] ]
weather.info()

# Date Field 의 datatype 을 날짜 타입으로 변경
weather['Date'] = pd.to_datetime(weather['Date'])
weather.info()


# ---

# ## 데이터 Visualization(시각화)
# 
#  - 아래와 같은 기본적인 Visualization 통해서 데이터의 특징을 이해해 봅니다.
#  
#    - 얼마나 많은 국가를 공격 하였는가?
#    - 상위 공격 대상 국가
#    - 상위 10개의 항공기 타입
#    - 이륙기지 위치 (공격 국가)
#    - 공격 대상 위치
#    - 폭격 경로
#    - 작전(군사) 지역
#    - 기상 관측소 위치

# 아래 셀을 실행하면 공격을 가장 많이 수행한 국가를 알 수 있습니다. 

# In[7]:


# Country 필드 분석
print(aerial['Country'].value_counts())
plt.figure(figsize=(22,10))
sns.countplot(x = aerial['Country'])
plt.show()


# 아래 셀을 실행하면 가장 많이 폭격을 받은 나라를 알 수 있습니다.

# In[8]:


# 상의 10 곳의 공격 대상 국가
print(aerial['Target Country'].value_counts()[:10])
plt.figure(figsize=(22,10))
sns.countplot(x = aerial['Target Country'])
plt.xticks(rotation=90)
plt.show()


# In[9]:


# 폭격기 타입별 분석
data = aerial['Aircraft Series'].value_counts()
print(data[:10])
data = [go.Bar(
            x=data[:10].index,
            y=data[:10].values,
            hoverinfo = 'text',
            marker = dict(color = 'rgba(177, 14, 22, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
    )]

layout = dict(
    title = 'Aircraft Series',
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# - 가장 많이 사용한 비행기는 아래의 그림과 같은 A36 입니다.
# ![A36](http://image.ibb.co/b3x66c/a36.png "A36")

# 폭격을 위해 이륙한 위치를 지도에 표시하기 전에, `Takeoff Latitude`, `Takeoff Latitude` 컬럼값을 float으로 변환합니다.
# 
# 데이터형의 원활한 변경을 위해서 필드의 위경도 좌표에 들어있는 숫자가 아닌 데이터(TUNISIA)를 지워줍니다.

# In[10]:


# 폭격 데이터의 레코드를 직접 확인해 봅니다.
aerial.head()

df_loc = aerial[aerial['Takeoff Latitude'] == 'TUNISIA']
aerial.drop(df_loc.index, inplace=True)
df2 = aerial[aerial['Takeoff Latitude'] == 'TUNISIA']

aerial = aerial.astype({'Takeoff Longitude':'float'})
aerial = aerial.astype({'Takeoff Latitude':'float'})
aerial.info()


# 아래 셀을 실행하면 공격 위치를 지도에서 확인할 수 있습니다.
# 
# 마우스로 지도를 움직이고 휠로 확대/축소할 수 있습니다. 마우스 커서 위치에 따라 안내창이 표시되기도 합니다.

# In[11]:


# 공격 위치 시각화

aerial['color'] = ""
aerial.loc[aerial['Country'] == 'USA', 'color'] = "rgb(0,116,217)"
aerial.loc[aerial['Country'] == 'GREAT BRITAIN', 'color'] = "rgb(255,65,54)" 
aerial.loc[aerial['Country'] == 'NEW ZEALAND', 'color'] = "rgb(133,20,75)"
aerial.loc[aerial['Country'] == 'SOUTH AFRICA', 'color'] = "rgb(255,133,27)"
data = [dict(
    type='scattergeo',
    lon = aerial['Takeoff Longitude'],
    lat = aerial['Takeoff Latitude'],
    hoverinfo = 'text',
    text = "Country: " + aerial.Country + " Takeoff Location: "+aerial["Takeoff Location"]+" Takeoff Base: " + aerial['Takeoff Base'],
    mode = 'markers',
    marker=dict(
        sizemode = 'area',
        sizeref = 1,
        size= 10 ,
        line = dict(width=1,color = "white"),
        color = aerial["color"],
        opacity = 0.7),
)]
layout = dict(
    title = 'Countries Take Off Bases ',
    hovermode='closest',
    geo = dict(showframe=False, showland=True, showcoastlines=True, showcountries=True,
               countrywidth=1, projection=dict(type='mercator'),
              landcolor = 'rgb(217, 217, 217)',
              subunitwidth=1,
              showlakes = True,
              lakecolor = 'rgb(255, 255, 255)',
              countrycolor="rgb(5, 5, 5)")
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# 이제 폭격 경로를 시각화해보겠습니다.

# In[12]:


# 폭격 경로 시각화 

# trace1
airports = [ dict(
        type = 'scattergeo',
        lon = aerial['Takeoff Longitude'],
        lat = aerial['Takeoff Latitude'],
        hoverinfo = 'text',
        text = "Country: " + aerial.Country + " Takeoff Location: "+aerial["Takeoff Location"]+" Takeoff Base: " + aerial['Takeoff Base'],
        mode = 'markers',
        marker = dict( 
            size=5, 
            color = aerial["color"],
            line = dict(
                width=1,
                color = "white"
            )
        ))]
# trace2
targets = [ dict(
        type = 'scattergeo',
        lon = aerial['Target Longitude'],
        lat = aerial['Target Latitude'],
        hoverinfo = 'text',
        text = "Target Country: "+aerial["Target Country"]+" Target City: "+aerial["Target City"],
        mode = 'markers',
        marker = dict( 
            size=1, 
            color = "red",
            line = dict(
                width=0.5,
                color = "red"
            )
        ))]
        
# trace3
flight_paths = []
for i in range( len( aerial['Target Longitude'] ) ):
    flight_paths.append(
        dict(
            type = 'scattergeo',
            lon = [ aerial.iloc[i,9], aerial.iloc[i,16] ],
            lat = [ aerial.iloc[i,8], aerial.iloc[i,15] ],
            mode = 'lines',
            line = dict(
                width = 0.7,
                color = 'black',
            ),
            opacity = 0.6,
        )
    )
    
layout = dict(
    title = 'Bombing Paths from Attacker Country to Target ',
    hovermode='closest',
    geo = dict(showframe=False, showland=True, showcoastlines=True, showcountries=True,
               countrywidth=1, projection=dict(type='mercator'),
              landcolor = 'rgb(217, 217, 217)',
              subunitwidth=1,
              showlakes = True,
              lakecolor = 'rgb(255, 255, 255)',
              countrycolor="rgb(5, 5, 5)")
)
    
fig = dict( data=flight_paths + airports+targets, layout=layout )
iplot( fig )


#   - 위의 결과에서 보듯이 대부분의 폭격은 지중해 연안에서 수행되었습니다.
#   
#   - 작전지역의 약어는 아래와 같습니다.
#   
#     - ETO: 유럽 작전 지역(European Theater of Operations)
#     - PTO: 태평양 작전 지역(Pasific Theater of Operations)
#     - MTO: 지중해 작전 지역(Mediterranean Theater of Operations)
#     - CBI: 버마 작전 지역(China-Burma-India Theater of Operations)
#     - EAST AFRICA: 동아프리카 작전 지역(East Africa Theater of Operations)
#     
#     ![유럽지도](http://image.ibb.co/bYvFzx/mto.png "유럽지도")

# 작전지역의 빈도를 그래프로 확인해봅니다.

# In[13]:


#Theater of Operations
print(aerial['Theater of Operations'].value_counts())
plt.figure(figsize=(22,10))
sns.countplot(x=aerial['Theater of Operations'])
plt.show()


# 기상관측소의 위치를 시각화해보겠습니다. 
# 
# 아래 셀을 실행하면 지도에서 기상관측소의 위치를 확인할 수 있습니다.

# In[14]:


# 기상관측소 위치

data = [dict(
    type='scattergeo',
    lon = weather_station_location.Longitude,
    lat = weather_station_location.Latitude,
    hoverinfo = 'text',
    text = "Name: " + weather_station_location.NAME + " Country: " + weather_station_location["STATE/COUNTRY ID"],
    mode = 'markers',
    marker=dict(
        sizemode = 'area',
        sizeref = 1,
        size= 8 ,
        line = dict(width=1,color = "white"),
        color = "blue",
        opacity = 0.7),
)]
layout = dict(
    title = 'Weather Station Locations ',
    hovermode='closest',
    geo = dict(showframe=False, showland=True, showcoastlines=True, showcountries=True,
               countrywidth=1, projection=dict(type='mercator'),
              landcolor = 'rgb(217, 217, 217)',
              subunitwidth=1,
              showlakes = True,
              lakecolor = 'rgb(255, 255, 255)',
              countrycolor="rgb(5, 5, 5)")
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)


#  - 2차 세계 대전 중 미군은 BURMA (KATHI city)를 공격했습니다.
#  - 이 지역에서 가장 가까운 기상 관측소는 `BINDUKURI`입니다.
#  - 이곳의 온도 변화를 확인해 보려고 합니다.
#  - 일자별 온도를 확인해 보기 전에, weather의 Date 컬럼의 데이터 타입을 날짜로 변경합니다.

# In[15]:


weather['Date'] = pd.to_datetime(weather['Date'])
weather_station_id = weather_station_location[weather_station_location['NAME'] == "BINDUKURI"]['WBAN']
print("BINDUKURI 관측소의 ID:", weather_station_id.values)

# STA == 32907 인 레코드를 선별합니다.
weather_bin = weather[weather['STA'] == 32907]

plt.figure(figsize=(22,10))
plt.plot(weather_bin.Date,weather_bin.MeanTemp)
plt.title("Mean Temperature of Bindukuri Area")
plt.xlabel("Date")
plt.ylabel("Mean Temperature")
plt.grid('on')
plt.show()


#   -  1943년 부터 1945년 사이의 BINDUKURI의 온도 변화는 위와 같습니다.
#   - 온도는 12도에서 32도 사이에서 변하고 있으며, 여름의 평균 기온이 겨울보다 높습니다.

# 아래 셀을 실행하면 `BINDUKURI`의 평균온도와 폭격일자의 기온을 비교할 수 있는 그래프를 그려줍니다.

# In[16]:


aerial = pd.read_csv("./input/operations.csv", low_memory=False)
aerial["year"] = [ each.split("/")[2] for each in aerial["Mission Date"]]
aerial["month"] = [ each.split("/")[0] for each in aerial["Mission Date"]]
aerial = aerial[aerial["year"]>="1943"]
aerial = aerial[aerial["month"]>="8"]

aerial["Mission Date"] = pd.to_datetime(aerial["Mission Date"])

attack = "USA"
target = "BURMA"
city = "KATHA"

aerial_war = aerial[aerial.Country == attack]
aerial_war = aerial_war[aerial_war["Target Country"] == target]
aerial_war = aerial_war[aerial_war["Target City"] == city]

liste = []
aa = []
for each in aerial_war["Mission Date"]:
    dummy = weather_bin[weather_bin.Date == each]
    liste.append(dummy["MeanTemp"].values)
aerial_war["dene"] = liste
for each in aerial_war.dene.values:
    aa.append(each[0])

# Create a trace
trace = go.Scatter(
    x = weather_bin.Date,
    mode = "lines",
    y = weather_bin.MeanTemp,
    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
    name = "Mean Temperature"
)
trace1 = go.Scatter(
    x = aerial_war["Mission Date"],
    mode = "markers",
    y = aa,
    marker = dict(color = 'rgba(16, 0, 200, 1)'),
    name = "Bombing temperature"
)
layout = dict(title = 'Mean Temperature --- Bombing Dates and Mean Temperature at this Date')
data = [trace,trace1]

fig = dict(data = data, layout = layout)
iplot(fig)


#  - 녹색 선은 `BINDUKURI`에서 측정한 평균 온도입니다.
#  - 파란점는 폭격 날짜와 폭격 날의 온도입니다.
#  - 데이터에서 알 수 있듯이 미국은 주로 온도가 높은 날 폭격을 했습니다.
#   - 문제는 [우리가 미래의 날씨를 예측할 수 있고, 이 예측에 따라 폭격이 이루어질지 여부를 예상 할 수 있는가] 입니다. 
#   - 이 질문에 답하기 위해서는 먼저 시계열 예측을 시작해야합니다.

# ---

# ### ARIMA 시계열 예측 기법이란?
# 
#   - 날씨를 예측하기 위하여 ARIMA 시계열 예측 기법을 사용하려고 합니다. 
#   - 전통적으로 시계열 데이터 분석은 AR(Autoregressive), MA(Moving average), ARMA(Autoregressive Moving average), ARIMA(Autoregressive Integrated Moving average) 모델 등을 활용해 불규칙적인 시계열 데이터에 규칙성을 부여하는 방식을 활용해왔습니다. 최근 딥러닝 방법이 발전함에 따라, 시계열 특성을 반영할 수 있는 RNN, LSTM과 같은 모델들을 시계열 예측에 활용하여 뛰어난 성능을 보여주고 있습니다. ARIMA에 대한 자세한 내용은 출처를 참고하시기 바랍니다.
#                                                                                          [※ 출처: https://dong-guri.tistory.com/9]
#   - ARIMA(AutoRegressive Integrated Moving Average) 기법에 대해서는 아래의 순서로 설명하겠습니다.
#     - 시계열이란?
#     - 시계열의 정상성이란?
#     - 시계열의 정상성 만들기?
#     - 시계열예측
# 
# 
# 
# #### 시계열(TIme Series)이란?
# 
#   - 시계열은 일정한 시간 간격으로 수집된 데이터를 뜻합니다. 즉 시간에 의존적인 데이터입니다.
#   - 대부분의 시계열 데이터는 계절적 경향성을 보입니다. 예를 들어, 아이스크림 판매 데이터는 여름에 많이 발생하기 때문에 계절성 있다고 할 수 있습니다. 반면에 1년동안 매일 한번씩 주사위를 던진다고 가정했을때의 데이터는 계절과 상관없이 결과값을 누적하기 때문에 계절성 추세가 없다고 할 수 있습니다.
# 
# 
# #### 시계열의 정상성(Stationarity of a Time Series)이란?
#   - 먼저 `정상성`이란 일정하여 늘 한결같은 성질을 뜻합니다. 즉, 정상성을 가진 시계열은 과거와 현재와 미래의 데이터가 모두 항상 안정적인, 일정한 분포를 가진것이 특징입니다.
#   - 시계열의 정상성 판단을 위한 3가지 기본 기준이 있습니다.
#     - 일정한 평균(constant mean)
#     - 일정한 분산(constant variance)
#     - 시간에 독립적인 자기공분산(autocovariance). 자기공분산이란 시계열과 지연(lagged) 시계열의 공분산입니다. 
#   - 전통적으로 시계열 분석은 현재 시점까지의 시계열 데이터 특성이 시간이 지나도 그대로 유지될 것이라고 가정하고 있습니다. 즉 시간과 관계 없이 평균과 분산이 불변해야 하고, 두 개의 시점 간의 공분산이 다른 시점과는 무관해야 한다고 합니다.
#   
#   
# - 수학적으로 정의한다면 다음의 세 가지 조건을 만족해야 합니다.
#   - 임의의 t에 대해서 E(Xt)=μ
#   - 임의의 t에 대해서 Var(Xt)<∞
#   - 임의의 t, h에 대하여 Cov(Xt+h,Xt)=γ(h)
#     - 3번에서 γ(h)는 자기공분산 함수 (Autocovariance Function, ACVF)라 불리우는데, 공분산이 시점 t에 의존하지 않고, 시간의 차이인 h에만 의존함을 의미합니다.
# 
#   - 예를 들어, A 주식이 정상성을 띤다고 가정합시다. 이 주가의 평균이 3천원이고, 임의의 시점을 t = 2020년 1월 3일이라 치면
# 
#     - 2020년 1월 3일의 주가의 평균이 μ=3천원이고
#     - 2020년 1월 3일의 주가의 분산이 유한하며 (= 극단치로 터지지 않고)
#     - 2020년 1월 3일의 주가와, 그로부터 5일 뒤인 2020년 1월 8일의 주가 간의 공분산이 γ(5)임을 의미합니다.
# 
#   - 정상성이 중요한 이유는 시계열의 평균과 분산이 일정해야 시계열 값을 예측할 수 있기 때문입니다. 자세한 내용은 출처를 참고해주시기 바랍니다.
#                                                      [※ 출처: https://assaeunji.github.io/statistics/2021-08-08-stationarity/]

# ---

# 데이터 시각화를 통해서 계절성을 확인해보겠습니다.

# In[17]:


# Bindikuri 지역의 평균 온도
plt.figure(figsize=(22,10))
plt.plot(weather_bin.Date,weather_bin.MeanTemp)
plt.title("Mean Temperature of Bindukuri Area")
plt.xlabel("Date")
plt.ylabel("Mean Temperature")
plt.grid('on')
plt.show()

# 날씨 시계열 데이터 표시 및 저장
timeSeries = weather_bin.loc[:, ["Date","MeanTemp"]]
timeSeries.index = timeSeries.Date
ts = timeSeries.drop("Date",axis=1)
print(ts)


#   - 위의 데이터에서 볼 수 있듯이 시계열에는 계절적인 변화가 있습니다. 매년 여름에는 평균 기온이 더 높고 겨울에는 평균 기온이 더 낮습니다.

# ---

#   - 다음으로 정상성을 체크해보겠습니다. 다음과 같은 방법으로 정상성을 확인할 수 있습니다.
#   
#     - Plotting Rolling Statistics: 윈도우를 6으로한 이동평균(rolling mean with windows size 6 : 주변 6개의 깂의 평균)과 이동분산(rolling variance)값을 구합니다.
#     - Dickey-Fuller Test: 테스트 결과값은 `Test Statistic(검정통계량)`과 `Critical Values for difference confidence levels(차이신뢰수준 임계값)`으로 구성되어 있습니다.  `Test Statics`값이 `Critical Values`보다 작으면 `정상성`이라고 판단합니다.

# In[18]:


# adfuller library: ADF 검정 테스트 관련 파이썬 패키지
from statsmodels.tsa.stattools import adfuller

# ADF 검정 함수 정의
def check_adfuller(ts):
    # Dickey-Fuller test
    result = adfuller(ts, autolag='AIC')
    print('Test statistic: ' , result[0])
    print('p-value: '  ,result[1])
    print('Critical Values:' ,result[4])
    
# 이동평균 및 이동분산 그래프 시각화 함수 정의
def check_mean_std(ts):
    #Rolling statistics
    rolmean = ts.rolling(6).mean()
    rolstd = ts.rolling(6).std()
    plt.figure(figsize=(22,10))   
    orig = plt.plot(ts, color='red',label='Original')
    mean = plt.plot(rolmean, color='black', label='Rolling Mean')
    std = plt.plot(rolstd, color='green', label = 'Rolling Std')
    plt.xlabel("Date")
    plt.ylabel("Mean Temperature")
    plt.title('Rolling Mean & Standard Deviation')
    plt.legend()
    plt.grid('on')
    plt.show()
    
# 정상성(stationary) 체크: 이동평균, 이동분산(표준편차), ADF검정
check_mean_std(ts)
check_adfuller(ts.MeanTemp)


#   - 정상성인지 판단하기 위한 첫번째 기준이 일정한 평균입니다. 그래프의 검은선을 보면 일정하지 않기 때문에 정상성이 아님을 알수 있습니다. 
#   - 두번째 기준인 일정한 분산(녹색선)을 확인해보면 비교적 일정한 것으로 판단할 수 있습니다. (정상성임)
#   - 마지막 세번째 기준을 판단하기 위하여 검정통계값을 확인해보면 `Test staticstic`값이 `Critical value` 값보다 크기 때문에 정상성이 아닙니다.
#     - test statistic(검정통계) = -1.4는 1% critical values(임계값) = -3.439229783394421, 5% 임계값: -2.86545894814762, 10% 임계값 -2.5688568756191392보다 항상 큽니다.

# 다음 셀에서 시계열 정상성으로 만들어 보겠습니다.

# ---

# #### 시계열 정상성 만들기(Make a Time Series Stationary)? 
#   - 위 셀에서 확인한바와 같이, 2가지 조건이 정상성 조건에 부합하지 않습니다.
#     - 추세(Trend): 시간 경과에 따른 평균, 즉 일정한 이동평균 값이 필요합니다.
#     - 계절성(Seasonality): 특점 시점의 variations(변동). 즉 일정한 variations가 필요핪니다. 
#     
#     
#   - 첫번째로 추세에 대한 문제(일정한 평균: constant mean)를 해결합니다.
#     - 가장 많이 사용되는 방법은 moving average방법입니다.
#     - Moving average: 'n'개 샘플의 평균을 구합니다. 'n'은 window 크기입니다. 

# In[19]:


# Moving average method
window_size = 6
moving_avg = ts.rolling(window_size).mean()
plt.figure(figsize=(22,10))
plt.plot(ts, color = "red",label = "Original")
plt.plot(moving_avg, color='black', label = "moving_avg_mean")
plt.title("Mean Temperature of Bindukuri Area")
plt.xlabel("Date")
plt.ylabel("Mean Temperature")
plt.legend()
plt.grid('on')
plt.show()


# 위 그래에서 붉은선은 원래 평균기온값이고 검은선은 6개의 샘플로 구한 `이동평균`값입니다.

# In[20]:


ts_moving_avg_diff = ts - moving_avg
ts_moving_avg_diff.dropna(inplace=True) # 윈도우 크기가 6이기 때문에 처음 6개 데이터는 NaN 값임

# 정상성(stationary) 체크: 이동평균, 이동분산(표준편차), ADF검정
check_mean_std(ts_moving_avg_diff)
check_adfuller(ts_moving_avg_diff.MeanTemp)


#   - 첫번째 조건(일정한 평균, constant mean): 이동평균(검은색선)를 보면 평균은 일정하게 보입니다.(정상성 조건 충족)
#   - 두번째 조건(일정한 분산, constant variance): 이동분산(녹색선) 또한 일정하게 보입니다.(정상정 조건 충족)
#   - 세번째 조건(ADF 검정 만족, 자기공분산) : ADF검정 결과, 1%의 임계값 보다 작으므로 99%의 신뢰도로 정상성이라고 할 수 있습니다.(정상성 조건 충족)
# 
# 
# - 우리는 시계열 정상성을 확인했습니다. 그러나 추세와 계절성을 피하기 위하여 한가지 방법을 더 살펴보겠습니다.
# 
#   - 차분기법(differencing method): 가장 일반적인 방법 중 하나이며, 시계열과 이동 시계열의 차이를 이용하는 것입니다. 

# In[21]:


# 차분기법: differencing method
ts_diff = ts - ts.shift() # shift 메서드는 시계열 데이터의 데이터나 인덱스를 원하는 만큼 이동시키고 기본값은 1입니다.
plt.figure(figsize=(22,10))
plt.plot(ts_diff)
plt.title("Differencing method") 
plt.xlabel("Date")
plt.ylabel("Differencing Mean Temperature")
plt.grid('on')
plt.show()


# In[22]:


ts_diff.dropna(inplace=True) # 데이터를 shift했기때문에 NaN 값이 생깁니다.

# 정상성(stationary) 체크: 이동평균, 이동분산(표준편차), ADF검정
check_mean_std(ts_diff)
check_adfuller(ts_diff.MeanTemp)


#   - 첫번째 조건(일정한 평균, constant mean): 이동평균(검은색선)를 보면 평균은 일정하게 보입니다.(정상성 조건 충족)
#   - 두번째 조건(일정한 분산, constant variance): 이동분산(녹색선) 또한 일정하게 보입니다.(정상정 조건 충족)
#   - 세번째 조건(ADF 검정 만족, 자기공분산) : ADF검정 결과, t-test는 1%의 임계값보다 작으므로 99%의 신뢰도로 정상성이라고 할 수있습니다.(정상성 조건 충족)

# ---

# #### 시계열 예측(Forecasting a Time Series)
# 
# 
#   - 추세와 계절성 조건을 확인하기 위해서 위해 이동평균(moving average)과 차분(differencing) 기법을 활용했습니다.
#   - 시계열 예측을 위해 차분기법의 결과인 ts_diff 시계열 데이터를 사용할 예정입니다.
#   - 시계열 예측을 위해 사용할 ARIMA(Auto-Regressive Integrated Moving Averages) 기법은 아래와 같이 구성되어 있습니다.
#     - AR:Auto-Regressive (P): 자동 회귀 분석으로 종속 변수의 시차를 이용합니다. 예를 들어 p가 3이면 x(t-1), x(t-2), x(t-3)를 사용하여 x(t)를 예측합니다.
#     - I: Integrated (d): 비계절성 차분 계수입니다. 1차 차분이 포함된 정도이며 이번 프로젝트에서는 d=0을 사용합니다.
#     - MA: Moving Averages (q): 이동평균 부분의 차수입니다.
#     
#   - 위에서 설명한 (p,d,q)는 ARIMA 모델의 파라미터입니다.
#   - p,d,q 파라미터를 정하기 위해 ACF와 PACF 그래프를 이용합니다.
#     - Autocorrelation Function (ACF): 자기상관함수란 시차에 따른 일련의 자기상관을 의미하며, 시차가 커질수록 ACF는 0에 가까워집니다. 정상 시계열은 상대적으로 빠르게 0에 수렴하며, 비정상 시계열은 천천히 감소하거나 큰 양의 값을 가집니다.
#     - Partial Autocorrelation Function (PACF): 편자기상관함수란 시차에 따른 일련의 편자기상관이며, 시차가 다른 두 시계열 데이터 간의 순수한 상호 연관성을 나타냅니다. 
#     
#               [출처: https://leedakyeong.tistory.com/entry/ARIMA란-ARIMA-분석기법-AR-MA-ACF-PACF-정상성이란 [슈퍼짱짱:티스토리]]
#     

# In[23]:


# ACF and PACF 
from statsmodels.tsa.stattools import acf, pacf

lag_acf = acf(ts_diff, nlags=20, fft=False)
lag_pacf = pacf(ts_diff, nlags=20, method='ols')

# ACF
plt.figure(figsize=(22,10))
plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')
plt.grid('on')

# PACF
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.grid('on')
plt.tight_layout()


#   - 위 그래프에서 두개의 점선은 신뢰구간입니다. 이 라인을 참고하여 'p'와 'q'값을 결정합니다. (좌측이 ACF, 우측이 PACF 그래프입니다.)
#     - p 선택: PACF 차트가 처음으로 신뢰 상한 구간을 교차하는 지연 값입니다. p=1
#     - q 선택: ACF 차트가 처음으로 신뢰 상한 구간을 교차하는 지연 값입니다. q=1
#     
#     
#   - 이제 (1,0,1)을 ARIMA 모델 파라메터로 사용하여 예측을 수행하겠습니다.
#     - 예측함수(model_fit.prediect)에 파라미터로 전달하는 시작-종료 인덱스는 예측할 기간을 입력합니다.
#     
#     
#   - 1944년 6월 25일부터 1945년 5월 31일 기간의 날씨를 예측하겠습니다.

# In[24]:


import statsmodels.api as sm
from datetime import datetime

# ts는 하루의 주기를 가지는 데이터임
ts = ts.to_period('D')

# fit model
model = sm.tsa.arima.ARIMA(ts, order=(1,0,1))
model_fit = model.fit()

# predict
start_index = datetime(1944, 6, 25)
end_index = datetime(1945, 5, 31)
forecast = model_fit.predict(start=start_index, end=end_index)

# visualization
plt.figure(figsize=(22,10))
plt.plot(weather_bin.Date,weather_bin.MeanTemp,label = "original")
plt.plot(forecast,label = "predicted")
plt.title("Time Series Forecast")
plt.xlabel("Date")
plt.ylabel("Mean Temperature")
plt.legend()
plt.grid('on')
plt.show()


# ---

# ## 퀴즈

# 모든 날짜에 대한 예측 및 시각화를 수행하고 평균제곱오차(MSE, Mean Squared Error)를 error에 저장하세요.
#   - MSE란 오차(error)를 제곱한 값의 평균입니다. 오차란 알고리즘이 예측한 값과 실제 정답과의 차이를 의미합니다. 즉, MSE 값은 작을수록 알고리즘의 성능이 좋다고 할 수 있습니다.

# ### 지시사항
# 
# 1. ARIMA 모델을 활용하여 모든 날짜에 대한 예측 값을 도출합니다. 
#     - 모델의 하이퍼 파라미터는 `order=(1,0,1)` 로 설정합니다.
# 
# 
# 2. 예측값을 활용하여 MSE 오차를 구해 `error` 변수에 저장합니다. 
#     - MSE는 sklearn의 metrics.mean_squared_error 를 활용하여 구할 수 있습니다. 
#     
# 
# 3. 시각화 결과를 확인합니다.

# In[25]:


from sklearn.metrics import mean_squared_error

# fit model
model2 = sm.tsa.arima.ARIMA(ts, order=(1,0,1)) # (ARMA) = (1,0,1)
model_fit2 = model2.fit()
forecast2 = model_fit2.predict()
error = mean_squared_error(ts, forecast2)
print("error: " ,error)

# visualization
plt.figure(figsize=(22,10))
plt.plot(weather_bin.Date,weather_bin.MeanTemp,label = "original")
plt.plot(forecast2,label = "predicted")
plt.title("Time Series Forecast")
plt.xlabel("Date")
plt.ylabel("Mean Temperature")
plt.legend()
plt.grid('on')
plt.savefig('graph.png')

plt.show()


# ## 제출하기
# 
# 퀴즈 수행 후, 아래 코드를 실행하면 `error` 변수가 저장된 `submission.pickle` 파일을 제작하여 채점을 받을 수 있습니다.
# 
# **아래 코드를 수정하면 채점이 불가능 합니다.**

# In[26]:


import pickle

df_result = pd.DataFrame([error], columns=["error"])

with open('submission.pickle', 'wb') as f:
    pickle.dump(df_result, f)


# In[27]:


# 채점을 수행하기 위하여 로그인
import sys
sys.path.append('vendor')
from elice_challenge import check_score, upload


# In[28]:


# 제출 파일 업로드
await upload()


# In[29]:


# 채점 수행
await check_score()


# In[ ]:




