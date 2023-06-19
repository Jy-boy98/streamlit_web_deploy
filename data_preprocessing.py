import streamlit as st
import numpy as np
from numpy import mean
import pandas as pd
import argparse
from parse import compile
from pathlib import Path
from datetime import date
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from mlxtend.frequent_patterns import fpgrowth
import xgboost
from xgboost import plot_importance
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pyswarm import pso
import GPUtil
warnings.simplefilter("ignore")

def preprocessing_data():

    path = './raw_data.csv'
    df_raw = pd.read_csv(path)

    print(df_raw.head())
    print(df_raw.info())


    # In[109]:

    st.subheader("원본 데이터")
    st.write(df_raw.describe())


    # ## 데이터 전처리
    # 
    # ### 자주 사용하는 변수 정의
    # 
    # ### MOLD_IN_TOP, MOLD_IN_BOT, MOLD_OUT_TOP, MOLD_ OUT_BOT 같은 금형값 컬럼이나 IN_RADIUS, OUT_RADIUS 같은 곡률값 컬럼처럼 코드 내에 반복하여 사용될 값은 변수로 정의하여 사용성을 높이고 불필요한 에러 발생을 줄인다.

    # In[110]:


    # 금형조합과 곡률
    CORES = ['MOLD_IN_TOP','MOLD_IN_BOT','MOLD_OUT_TOP','MOLD_OUT_BOT']
    RADIUS = ['IN_RADIUS','OUT_RADIUS']
    EQUIP_INFO = ['EQUIP_ID','MOLD_POS']
    POWERS = ['POWER1','POWER2','POWER3','POWER4','POWER5']


    # ## 렌즈 분류 및 카테고리 열 추가
    # ### 렌즈의 도수를 기준으로 렌즈의 종류를 근시, 무도수(미용), 원시로 나눈다.
    # 
    # ### ◦ 근시(MYOPIA) : 도수가 0보다 작을 때
    # ### ◦ 무도수(BEAUTY) : 도수가 0일 때
    # ### ◦ 원시(HYPER) : 도수가 0보다 클 때
    # 
    # #### df_raw에 CATEGORY 열을 추가한 뒤 기본값으로 MYOPIA를 넣는다. 그후 REAL_POWER 값이 0과 같으면 BEAUTY, 0보다 크면 HYPER로 값을 변경한다.
    # #### df_raw.info()를 실행해보면 CATEGORY 열이 생성되고 데이터가 추가된 것을 확인할 수 있다.

    # In[111]:


    df_raw['CATEGORY']='MYOPIA'
    df_raw['CATEGORY'][df_raw['REAL_POWER']==0] = 'BEAUTY'
    df_raw['CATEGORY'][df_raw['REAL_POWER']>0] = 'HYPER'

    st.subheader("카테고리")
    st.write(df_raw['CATEGORY'])


    # ### 누락 데이터 처리 함수 정의
    # 
    # ####  입력된 데이터셋(df_before)에서 NULL이거나 공백인 데이터의 값을 np.NaN으로 수정하고, MOLD_IN_TOP, MOLD_IN_BOT, MOLD_ OUT_TOP, MOLD_ OUT_BOT 값이 np.NaN인 행을 삭제한 데이터셋(df_after)를 반환한다.

    # In[112]:


    def preprocessing_missing(df_before):
        type = 'Missing Value'
        print(f'Type of preprocessing: {type}')
        print(f'The shape of data before preprocessing: {df_before.shape}')
        
        tmp = np.where((df_before.values == 'NULL')|(df_before.values == ''), np.NaN, df_before.values)
        df_after = pd.DataFrame(data=tmp, columns=df_before.columns)
        df_after = df_after.dropna(how='any', subset=CORES+RADIUS)

        print(f'The shape of data after preprocessing: {df_after.shape}\n')
    
        return df_after


    # ### 데이터 형변환 처리 함수 정의
    # #### 열(column)의 이름을 모두 대문자로 변환하고 각 데이터의 형을 통일시킨다.
    # #### IN_RADIUS와 OUT_RADIUS의 양수/음수 부호를 없애고 절대값으로 수정한다. 이를 통해 부호 기입 실수로 인한 에러를 줄일 수 있다.

    # In[113]:


    def preprocessing_conversion(df_before):
        type = 'Data Type Conversion'
        print(f'Type of preprocessing: {type}')
        print(f'The shape of data before preprocessing: {df_before.shape}')
        
        # 컬럼명을 모두 대문자로 변환
        df_before.columns = [col.upper() for col in df_before.columns]
        print(df_before.columns)

        # 컬럼의 데이터형 통일
        col_type = {
            'MOLD_POS': 'int',
            'REAL_POWER':'float32',
            'OUT_RADIUS': 'float32',
            'IN_RADIUS': 'float32',
            'POWER': 'float32',
            'POWER1': 'float32',
            'POWER2': 'float32',
            'POWER3': 'float32',
            'POWER4': 'float32',
            'POWER5': 'float32'
        }

        df_after = df_before.astype(col_type)
        df_after['MFG_DT'] = pd.to_datetime(df_before['MFG_DT'])

        # IN_RADIUS와 OUT_RADIUS의 값의 부호 삭제
        df_after['IN_RADIUS'] = abs(df_after['IN_RADIUS'])
        df_after['OUT_RADIUS'] = abs(df_after['OUT_RADIUS'])

        # CORES 대문자로 변환
        df_after[CORES] = df_after[CORES].applymap(lambda x:x.upper())

        print(f'The shape of data after preprocessing: {df_after.shape}\n')

        return df_after


    # ### 지정 날짜 기준 데이터 추출 함수 정의
    # 
    # #### 너무 오래된 공정 데이터는 최근의 경향성을 반영하지 못하기 때문에 모델의 정확성을 높이기 위해 2020년 1월 1일 이후 데이터만 분석에 활용한다.

    # In[114]:


    def preprocessing_date(df_before):
        start_time = st.slider("Date select", value = datetime(2020, 1, 1), min_value=datetime(2020, 1, 1), max_value=datetime(2021, 4, 25), format = "YY/MM/DD")
        type = 'Removing Irrelevant Data in terms of Date'
        print(f'Type of preprocessing: {type}')
        print(f'The shape of data before preprocessing: {df_before.shape}')
        df_after = df_before[df_before['MFG_DT'] >= np.datetime64(start_time)]
        print(f'The shape of data after preprocessing: {df_after.shape}\n')
    
        return df_after


    # ### 이상치 처리 함수 정의
    # 
    # #### WP_VALUE는 POWER1, POWER2, POWER3, POWER4, POWER5의 평균을 도수기준표에 넣어 결정하는 값으로 최종적으로 결정된 도수인 REAL_POWER와 일치해야한다.
    # 
    # #### 위 조건을 만족하지 않는 데이터를 제거한 뒤 IN_RADIUS와 OUT_RADIUS를 기준으로 Isolation Forest 알고리즘을 사용하여 이상치를 제거한다.
    # 
    # #### IN_RADIUS와 OUT_RADIUS를 기준으로 한 산점도를 그려 이상치 제거가 잘 되었는지 확인한다.

    # In[115]:


    def preprocessing_outlier(df_before):
        
        type = 'Removing Outliers'
        print(f'Type of preprocessing: {type}')
        print(f'The shape of data before preprocessing: {df_before.shape}')
    
        # 조건1 : REAL_POWER 값과 POWER 값이 같은 경우
        mask1 = (df_before['REAL_POWER'] == df_before['POWER'])
        # 조건2 : POWER1, POWER2, POWER3, POWER4, POWER5의 평균과 POWER의 오차가 0.5보다 큰 경우
        mask2 = abs(mean(df_before[POWERS], axis=1) - df_before['POWER']) < 0.5
        # 조건1과 조건2를 모두 만족하는 경우만 추출
        df_after = df_before[mask1 & mask2]

        
        # IsolationForest 적용 전의 데이터 시각화
        fig = plt.figure(figsize=(16,16))
        ax_before = fig.add_subplot(1, 1, 1)
        df_after.plot.scatter(
        x='IN_RADIUS',
        y='OUT_RADIUS',
        ax=ax_before
    )
    
        # IsolationForest 적용 후의 데이터 시각화
        results = IsolationForest(random_state=0).fit_predict(df_after[RADIUS])
        df_after = df_after[results == 1]
        df_after.plot.scatter(
            x='IN_RADIUS',
            y='OUT_RADIUS',
            ax=ax_before,
            color='black'
    )
        st.pyplot(fig)
        (f'The shape of data after preprocessing: {df_after.shape}\n')
    
        return df_after


    # ### 데이터 전처리 실행
    # 
    # #### 앞서 정의한 전처리 프로세스를 순차적으로 실행하는 파이프라인을 구성하고 최종 전처리 데이터를 뽑아내는 실행 함수를 정의한다.
    # 
    # #### 이후의 과정은 근시(MYOPIA) 데이터만으로 진행한다. 무도수(BEAUTY)와 원시(HYPER) 렌즈에 대해서도 동일한 방식을 적용하여 AI 모델을 생성할 수 있다.

    # In[116]:


    # 근시 데이터만 추출
    st.subheader("근시 데이터 추출")
    df_raw = df_raw[df_raw['CATEGORY']=='MYOPIA']
    st.write(df_raw)


    # In[117]:


    # 누락값 처리 함수 실행
    df_after = preprocessing_missing(df_raw)
    df_after = df_after.drop(columns=['CP','AX'])
    st.subheader("누락값 처리한 데이터")
    st.write(df_after)



