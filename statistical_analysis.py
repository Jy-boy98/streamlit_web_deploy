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
import time
warnings.simplefilter("ignore")

def analysis_data():
        
    path = './raw_data.csv'
    df_raw = pd.read_csv(path)

    print(df_raw.head())
    print(df_raw.info())


    # In[109]:


    df_raw.describe()


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

    df_raw.info()


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
        
        type = 'Removing Irrelevant Data in terms of Date'
        print(f'Type of preprocessing: {type}')
        print(f'The shape of data before preprocessing: {df_before.shape}')
        df_after = df_before[df_before['MFG_DT'] >= np.datetime64('2020-01-01')]
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
        plt.show()
        
        print(f'The shape of data after preprocessing: {df_after.shape}\n')
    
        return df_after


    # ### 데이터 전처리 실행
    # 
    # #### 앞서 정의한 전처리 프로세스를 순차적으로 실행하는 파이프라인을 구성하고 최종 전처리 데이터를 뽑아내는 실행 함수를 정의한다.
    # 
    # #### 이후의 과정은 근시(MYOPIA) 데이터만으로 진행한다. 무도수(BEAUTY)와 원시(HYPER) 렌즈에 대해서도 동일한 방식을 적용하여 AI 모델을 생성할 수 있다.

    # In[116]:


    # 근시 데이터만 추출
    df_raw = df_raw[df_raw['CATEGORY']=='MYOPIA']


    # In[117]:


    # 누락값 처리 함수 실행
    df_after = preprocessing_missing(df_raw)
    df_after = df_after.drop(columns=['CP','AX'])


    # In[118]:


    # 형 변환 처리 함수 실행
    df_after = preprocessing_conversion(df_after)


    # In[119]:


    # 2020년 이후 데이터 추출 함수
    df_after = preprocessing_date(df_after)


    # ####  아래의 산점도에서 파란색 점은 Isolation Forest를 이용한 이상치 처리 전의 데이터이고 검정색 점은 이상치 처리 후의 데이터이다. preprocessing_outlier 함수에서 Isolation Forest 함수의 속성값을 수정하여 출력결과를 조정할 수 있다.

    # In[120]:


    df_after = preprocessing_outlier(df_after)


    # In[121]:


    df_after.info()


    # In[122]:


    df_after.describe()


    # ### corr 함수를 통한 변수 간 상관관계 파악
    # 
    # #### 상관관계 분석을 통해 두 변수 간의 선형적인 관계 존재 여부를 확인할 수 있다.
    # 
    # #### 분석에 사용된 상관 계수는 보편적으로 사용되는 피어슨 상관 계수로서 -1과 1 사이의 값을 가진다. 상관 계수의 절대값이 1에 가까울수록 변수 간의 선형관계가 강하다고 볼 수 있으며, 값이 양수일 때는 양적인 선형관계, 값이 음수일 때는 음적인 선형관계를 가진다.
    # 
    # #### 상관관계 분석은 비선형적인 관계를 확인하기 어렵지만, 변수 간의 선형관계를 정량적으로 확인할 수 있다는 장점이 있다. 상관계수의 절댓값이 0.1과 0.3 사이이면 약한 선형관계, 0.3과 0.7 사이이면 뚜렷한 선형관계, 0.7 이상인 경우 강한 선형관계를 나타낸다고 이해할 수 있다.
    # 
    # #### 결과의 이미지는 변수 간의 상관계수를 시각화한 히트 맵으로써, 각 칸의 값은 해당되는 X축, Y축의 변수 간의 상관계수이며, 우측의 범례처럼 상관계수 값마다 다른 색을 가진다.

    # In[123]:


    #plt.subplots(figsize=(25,25))
    #sns.heatmap(data = df_after.corr(), linewidths=0.1, annot=True, fmt ='.4f', cmap='Blues')


    # ### 단계 5 XGBoost 모델 학습
    # 
    # ### ENCODER는 XGBoost 학습과 예측에 동일한 컬럼을 가진 X 데이터를 생성하기 위해 사용한다.
    # ### N_ESTIMATORS 와 같이 상수를 사용하는 XGBoost 모델 파라미터도 변수로 정의한다.
    # ### prepare_data 는 모델 학습에 필요한 학습 데이터셋과 검증 데이터셋을 생성한다.
    # 

    # In[124]:


    # Onehot encoder
    ENCODER = None
    # XGBoost
    N_ESTIMATORS = None
    ETA = 0.25
    SUB_SAMPLE = 0.9

    N_ESTIMATORS = st.number_input("Number of Estimators", min_value=100, max_value=10000, value=6000, step=100)

    # In[125]:


    def prepare_data(df, numeric_cols, onehot_cols, y_col):
        x_cols = numeric_cols + onehot_cols
        df = df.dropna(
            axis=0,
            how='any',
            subset=x_cols+y_col
        ) 
        data_X = df[x_cols]
        data_y = df[y_col]
    
        print('\nSplitting data into train set and test set...')
        X_train, X_test, y_train, y_test = train_test_split(data_X,data_y,test_size=.3,random_state=42)
    
        print(f'The shape of X_train, y_train: X_train.shape, y_train.shape')
        print(f'The shape of X_test, y_test: X_test.shape, y_test.shape')
    
        bounds = df.groupby(['REAL_POWER'])[numeric_cols].agg(['min', 'max'])
        
        return X_train, X_test, y_train, y_test, bounds


    # In[181]:


    def transform_cols(df, encoder):
        if encoder is None:
            print('ENCODER fit_transform')
            encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
            onehot = encoder.fit_transform(df[EQUIP_INFO])
        else:
            print('ENCODER transform')
            onehot = encoder.transform(df[EQUIP_INFO])
    
        transformed_onehot_cols = [f"{col}_{category}" for col, categories in zip(EQUIP_INFO, encoder.categories_) for category in categories]
        df = np.hstack((df[RADIUS].to_numpy(),onehot))
        df = pd.DataFrame(df, columns=RADIUS+transformed_onehot_cols)
    
        return df, encoder


    # ### 학습 데이터와 평가 데이터 분리
    # 
    # #### 먼저 전처리 데이터(df_prep) 중 근시 데이터만 추출한다.
    # #### numeric_cols와 onehot_cols를 지정해준 후에 df_prep 데이터셋을 학습용 데이터셋과 평가용 데이터셋으로 분리한다.
    # 
    # #### ◦ X 데이터 : IN_RADIUS, OUT_RADIUS, EQUIP_ID, MOLD_POS
    # #### ◦ Y 데이터 : X데이터를 기준으로 묶었을 때의 POWER1, POWER2, POWER3, POWER4, POWER5의 값

    # In[182]:


    df_prep = df_after.reset_index().melt(
        id_vars=['index','SALE_CD','REAL_POWER']+RADIUS+EQUIP_INFO,
        value_vars=POWERS,
        value_name='y'
    )


    # ####  POWER1, POWER2, POWER3, POWER4, POWER5 열을 variable이라는 새로운 열로 통합하고 각 값을 새로운 y열에 넣었기 때문에 데이터셋의 크기가 변화한 것을 확인할 수 있다.

    # In[183]:

    df_prep


    # In[184]:


    df_prep.info()


    # In[185]:


    X_train, X_test, y_train, y_test, bounds = prepare_data(
        df = df_prep,
        numeric_cols = RADIUS,
        onehot_cols = EQUIP_INFO,
        y_col=['y']
    )


    # ### XGBoost Regressor 호출 및 파라미터 설정
    # 
    # #### n_estimators : 부스팅 단계를 수행하는 트리의 개수
    # #### objective : 목적함수
    # #### ◦ reg:squarederror : 선형회귀
    # #### ◦ binary:logistic : 이진회귀분류
    # #### ◦ count:poisson : 포아송회귀
    # 
    # #### eta : 부스팅 스탭마다 가중치를 부여하여 과적합을 방지
    # #### tree_method : 모델에 사용할 트리 방식
    # #### ◦ hist 히스토그램 방식
    # #### ◦ gpu_hist 연산에 GPU를 사용하는 히스토그램 방식
    # 
    # #### gpu_id : 사용할 특정 GPU를 지정, 기본값 0
    # #### subsample : 사용할 훈련 세트의 샘플 비율

    # In[186]:
    st.subheader("생산 조건 입력")
    c1, c2 = st.columns([2,2]) # 각 열 간의 간격 조절
    with c1:
        examples_number = int(st.number_input('샘플을 만들 개수 :'))
    df_sample = df_raw.groupby(['CATEGORY']+EQUIP_INFO+['REAL_POWER','MOLD_IN_TOP']).count().reset_index()
    df_sample = df_sample[df_sample['POWER1'] > 10].sort_values(by=['POWER1'], ascending=False)
    df_sample = df_sample[df_sample['CATEGORY']=='MYOPIA']
    df_sample.reset_index(drop=True, inplace=True)
    st.write(df_sample.iloc[0:examples_number])


        # #### 예시 보기 중에서 원하는 예시의 index를 입력한다.

        # In[147]:

    with c2:
        idx_sample = st.number_input('샘플데이터 중 원하는 데이터의 인덱스를 입력해주세요: ')
    idx_sample = int(idx_sample)
    equipment_id = df_sample.iloc[idx_sample]['EQUIP_ID']
    mold_position = df_sample.iloc[idx_sample]['MOLD_POS']
    mold_position = int(mold_position)
    target_power = df_sample.iloc[idx_sample]['REAL_POWER']
    target_power = float(target_power)
    core_group = df_sample.iloc[idx_sample]['MOLD_IN_TOP'][:-9]

        # 선택한 조건 확인 출력
    st.write(f'equipment_id: {equipment_id},     mold_position: {mold_position},     target_power: {target_power},     core_group: {core_group}')


    if target_power < 0:
        category = 'MYOPIA'
    elif target_power == 0:
        category = 'BEAUTY'
    else:
        category = 'HYPER'

    if st.button("Train"):
        # 결과를 출력할 빈 상자 생성
        result_text = st.empty()

        if len(GPUtil.getGPUs()) < 1:
            regressor = xgboost.XGBRegressor(
                n_estimators=N_ESTIMATORS,
                objective='reg:squarederror',
                eta=ETA,
                tree_method='hist',
                gpu_id=0,
                subsample=SUB_SAMPLE
            )
        else:
            regressor = xgboost.XGBRegressor(
                n_estimators=N_ESTIMATORS,
                objective='reg:squarederror',
                eta=ETA,
                tree_method='gpu_hist',
                gpu_id=0,
                subsample=SUB_SAMPLE
            )


    # ###  X_train과 X_test 열의 원핫 인코딩(One-hot encoding)
    # 
    # #### 출력결과를 통해 X_train 인코딩에는 fit_transform 함수가 사용되고 X_test 인코딩에는 transform 함수가 사용된 것을 확인할 수 있다.
    # #### DataFrame.columns를 이용해 학습데이터가 제대로 인코딩 되었는지 확인한다.

    # In[187]:


        X_train, ENCODER = transform_cols(X_train, ENCODER)
        X_test, ENCODER = transform_cols(X_test, ENCODER)


        # In[188]:


        X_train.columns
        print(X_train.shape[1])
        print(X_train.index.shape[0])


        # ### XGBoost Regressor 모델 학습
        # 
        # #### 렌즈의 결정도수는 0.25 단위로 증가하기 때문에 mae(Mean absolute error, 평균 절대 오차)가 0.125보다 작아야만 모델의 도수적중률이 확보된다. 학습 모델과 검증 모델 모두 mae는 0.079 이하, score는 0.99 이상으로 나타났다

        # In[134]:


        history = regressor.fit(
            X = X_train,
            y = y_train,
            eval_set = [(X_train, y_train), (X_test, y_test)],
            eval_metric = 'mae',
            early_stopping_rounds = N_ESTIMATORS/10,
            verbose = False
        )
        learning_progress = st.empty()

        # Create the figure and axes for the plot
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.set_yscale('log')

        # Initialize an empty list to store the mae values
        train_loss = []
        val_loss = []

        # Train the model and update the plot with each epoch
        for i, loss in enumerate(regressor.evals_result()['validation_0']['mae']):
            train_loss.append(regressor.evals_result()['validation_0']['mae'][i])
            val_loss.append(regressor.evals_result()['validation_1']['mae'][i])

            # Clear the previous plot
            ax.clear()

            # Plot the training and validation losses
            ax.plot(train_loss, label='Training Loss')
            ax.plot(val_loss, label='Validation Loss')
            ax.axvline(i, color='gray', label='Current Epoch')

            # Set the labels and title
            ax.set_xlabel('Number of Trees  ')
            ax.set_ylabel('Loss(log)')
            ax.set_title('Learning process')

            # Update the plot in the Streamlit app
            learning_progress.pyplot(fig)

            # Wait for a short time before proceeding to the next epoch
            # to allow the plot to be updated in the Streamlit app
            time.sleep(0.01)

        st.subheader("xgboost로 학습한 값")
        st.write(f'Score: train({regressor.score(X_train, y_train)}),test({regressor.score(X_test, y_test)})')

        from sklearn.tree import DecisionTreeRegressor
        from sklearn.metrics import mean_absolute_error

        # 의사결정나무 모델 초기화
        model = DecisionTreeRegressor()

        # 학습 데이터셋으로 모델 학습
        model.fit(X_train, y_train)

        # 학습 데이터셋에 대한 예측 결과
        train_predictions = model.predict(X_train)
        # 검증 데이터셋에 대한 예측 결과
        val_predictions = model.predict(X_test)

        # 학습 데이터셋과 검증 데이터셋에 대한 MAE 계산
        train_mae = mean_absolute_error(y_train, train_predictions)
        val_mae = mean_absolute_error(y_test, val_predictions)

        # 결과 출력
        st.subheader("의사결정나무로 학습한 값")
        st.write(f"Train MAE: {train_mae}")
        st.write(f"Validation MAE: {val_mae}")

        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_absolute_error

        # 랜덤 포레스트 회귀 모델을 초기화합니다.
        model = RandomForestRegressor()

        # 모델을 학습합니다.
        model.fit(X_train, y_train)

        # 학습 데이터셋에 대한 예측을 수행합니다.
        train_predictions = model.predict(X_train)

        # 검증 데이터셋에 대한 예측을 수행합니다.
        val_predictions = model.predict(X_test)

        # 학습 데이터셋과 검증 데이터셋에 대한 평균 절대 오차(MAE)를 계산합니다.
        train_mae = mean_absolute_error(y_train, train_predictions)
        val_mae = mean_absolute_error(y_test, val_predictions)

        # 결과를 출력합니다.
        st.subheader("랜덤포레스트로 학습한 값")
        st.write(f"Train MAE: {train_mae}")
        st.write(f"Validation MAE: {val_mae}")

        # In[136]:

        # ### 모델 평가
        # 
        # #### 학습된 모델에서 각 변수의 중요도를 확인해보면 아래와 같으며, 중요도가 높다고 평가된 변수는 IN_RADIUS와 OUT_RADIUS였다.

        # In[137]:


        # 모델 평가 
        #  학습된 모델에서 각 변수의 중요도를 확인해보면 아래와 같으며, 중요도가 높다고 
        # 평가된 변수는 IN_RADIUS와 OUT_RADIUS였다.

        st.subheader("모델 평가")
        fig, ax = plt.subplots(figsize=(10,10))
        plot_importance(regressor, ax = ax)
        st.pyplot(fig)


        # #### 학습 데이터셋과 검증 데이터셋의 학습률을 확인해보면 아래와 같다.
        # 
        # #### 두 경우의 손실률 차이가 매우 적어서 vadilation 값에 log를 취한 뒤 비교해보면 학습 데이터셋의 정확도가 더 높게 나오는 것을 확인할 수 있다. best_ntree_limit 속성을 이용해 모델의 최적 분류기 개수를 확인할 수 있는데, 본 모델의 경우 n_estimator가 6000일 때 best_ntree_limit도 6000이 나왔기 때문에 n_estimator의 크기를 늘릴 경우 모델의 정확도가 더 높아진다고 판단할 수 있다.
        # 
        # #### 모델의 score가 이미 0.99를 넘겼기 때문에 n_estimator를 늘릴 필요가 없다고 판단하여 보정 없이 학습된 모델을 사용한다.

        # In[139]:


        # 모델의 학습률과 최소 트리 수 확인
        xgb_val = regressor.evals_result()

        fig, ax = plt.subplots(figsize=(10, 7))
        plt.yscale('log')
        plt.text(N_ESTIMATORS, 0.09, 'Minimum mae \n= '+str(min(xgb_val['validation_0']['mae'])))
        plt.plot(xgb_val['validation_0']['mae'], label='Training loss')
        plt.plot(xgb_val['validation_1']['mae'], label='Validation loss')
        plt.axvline(regressor.best_ntree_limit, color='gray', label='Optimal tree number')
        plt.xlabel('Number of trees')
        plt.ylabel('Loss(log)')
        plt.legend()

        st.subheader("모델의 학습률과 최소 트리 수 확인")
        st.pyplot(fig)


        # ### 목표 도수를 생산하기 위한 최적 금형조합 도출
        # 
        # #### 결과 데이터프레임 생성
        # #### 분석결과를 담을 데이터프레임 구조를 지정한다.

        # In[140]:


        # 결과를 담을 데이터프레임 생성
        cols_result = EQUIP_INFO+CORES+RADIUS+['RANK_INDEX','RANK_INDEX_1','RANK_INDEX_2','PREDICTED_POWER',
        'PREDICTED_POWER2','ERROR']
        df_result = pd.DataFrame(columns=cols_result)
        df_result = df_result.astype(
            {
                'IN_RADIUS':'float',
                'OUT_RADIUS':'float',
                'RANK_INDEX':'float',
                'RANK_INDEX_1':'float',
                'RANK_INDEX_2':'float',
                'PREDICTED_POWER':'float',
                'PREDICTED_POWER2':'float',
                'ERROR':'float'
            }
        )
        df_result.dtypes


        # ### 생산조건 입력
        # #### 최적 공정조건을 찾기 위한 제약조건을 입력한다.
        # #### ◦ 설비(equipment_id)
        # #### ◦ 금형 위치(mold_position)
        # #### ◦ 목표 도수(target_power)
        # #### ◦ 금형 조합(core_group)
        # 
        # #### 이때 category를 목표 도수가 0보다 작으면 근시렌즈(MYOPIA)로, 0과 같으면 미용렌즈(BEAUTY)로 지정한다.
        # 
        # #### 실제 적용된 사례에서는 사용자가 UI를 통해 입력조건을 넣게 되어있지만, 가이드북에서는 입력 도중 발생할 수 있는 에러를 방지하고 학습자의 편의성을 높이기 위해 10건의 예시를 제공한다.

        # In[141]:


        # #### 10건 모두 최소 하나 이상의 목표도수를 만족하는 금형조합을 추천했으며 추천된 5개의 금형조합이 모두 일치한 사례도 있었다. XGBoost에서 예측된 도수를 산술계산하여 최종 예측도수를 결정하기 때문에, 모델에서 발생한 실제 오차는 0.25보다 작을 수 있으며 따라서 숙련된 사용자의 경우 동일한 금형조합을 사용하여 목표도수에 맞는 렌즈를 생산할 수 있을 것으로 예상된다.
        # 
        # #### 이와 같은 오차가 발생하는 원인은 (1)숙련 작업자와 비숙련 작업자의 데이터를 분리하지 못하고 모델 학습에 일괄 적용했으며 (2)데이터 품질처리와 전처리 과정에서 과반수 이상의 데이터가 유실되었기 때문으로 볼 수 있다.
        # 
        # #### 50개의 추천 결과 중 작업자에 따른 차이로 보기 어려운 오차 0.5 이상의 추천 조합은 모두 4개로 8%의 비율을 차지했다.

        # ### 연관규칙 알고리즘을 통한 금형조합 묶음
        # 
        # #### 금형조합은 IN 몰드의 금형 2개와 OUT 몰드의 금형 2개, 총 4개의 금형으로 이루어진다. 하나의 금형조합에 포함되는 금형은 모두 동일한 금형군 이름을 가져야만 한다.
        # 
        # #### 연관규칙 알고리즘은 입력조건을 만족하면서 동일한 금형군에 속한 금형조합의 평균 IN_RADIUS와 OUT_RADIUS를 구한다.
        # 
        # #### 연관규칙 적용을 위한 데이터 전처리를 진행한다.

        # In[148]:


        # 금형 이름에서 공백(‘ ‘)을 밑줄(‘_’)로 변환
        df_pre_assoc = df_raw.copy();

        df_pre_assoc = df_pre_assoc[df_pre_assoc['MOLD_IN_TOP'].str.contains(core_group)]

        #for name in CORES:
        #   df_pre_assoc[name] = df_pre_assoc[name].str.replace('', '_')

        print(df_pre_assoc[CORES][:5])


        # In[149]:


        # one hot encoding을 실행한다.
        df_pre_assoc = pd.get_dummies(df_pre_assoc, columns=CORES)
        print(df_pre_assoc.columns[16:], df_pre_assoc.shape)


        # #### 연관규칙 알고리즘을 위한 데이터셋을 추출한다.

        # In[150]:


        # 전처리 데이터에서 CATEGORY, EQUIP_ID, MOLD_POS, REAL_POWER가 입력조건과 일치하는 데이터만 추출
        df_assoc = df_pre_assoc.copy()
        assoc_pos = mold_position

        df_assoc = df_assoc[(df_assoc['CATEGORY']==category) &
                    (df_assoc['EQUIP_ID']==equipment_id) &
                    (df_assoc['MOLD_POS']==mold_position) &
                    (df_assoc['REAL_POWER']==target_power)]


        # In[151]:


        # 생산이력이 없으면 입력조건과 CATEGORY가 일치하는 모든 데이터를 대상으로 연관규칙을 실행
        if df_assoc.shape[0] == 0:
            print(f'No history, Searching within All')
            df_assoc = df_pre_assoc.copy()
            df_assoc = df_assoc[(df_assoc['CATEGORY']==category) & (df_assoc['REAL_POWER']==target_power)]
            assoc_pos = 0
        # 생산조건과 일치하는 이력이 없는 경우의 데이터셋 설정


        # In[152]:


        df_assoc.shape


        # In[153]:


        # One-hot 인코딩 결과 int(0,1)에서 bool(False, True)로 변환
        # df.iloc[:,16] : One-hot encoding 열만 잘라냄
        df_ = df_assoc.copy()
        df_ex = df_.iloc[:, 16:].replace({1: True, 0: False})

        print(df_ex.info(verbose=True))


        # #### 연관규칙 알고리즘 실행한다.

        # In[154]:


        # 연관규칙 알고리즘
        assoc_result = fpgrowth(df_ex, min_support=0.005, use_colnames=True)

        # 연관규칙 도출 결과 중 [‘sets’]==4인 경우,
        # 즉 4개의 금형이 조합을 이루는 유효한 값을 추려냄
        assoc_result['sets'] = assoc_result['itemsets'].apply(lambda x: len(x))
        print(assoc_result.sort_values(by=['sets'], ascending=False))

        assoc_result = assoc_result[assoc_result['sets']==4]
        print(assoc_result)
        #연관규칙 알고리즘으로 금형조합 sets=4 인 데이터셋만 추출


        # ### 연관규칙 결과 테이블 만들기
        # 
        # #### 데이터셋의 컬럼명 복원 : 원핫 인코딩을 위해 수정했던 데이터셋의 컬럼명을 원래의 형태로 복원한다.

        # In[155]:


        temp_df = pd.DataFrame(columns=CORES+['SUPPORT'])
        temp_list = [list(x) for x in assoc_result['itemsets']]

        for i, dset in enumerate(temp_list):
            mold_in_bot = None  # mold_in_bot 변수를 먼저 정의합니다
            mold_in_top = None  # mold_in_top 변수를 그 다음에 정의합니다
            mold_out_top = None
            mold_out_bot = None
            score = None
            
            for name in dset:
                if 'MOLD_IN_TOP_' in name:
                    p1 = compile('MOLD_IN_TOP_{}')
                    mold_in_top = p1.parse(name)[0]
                elif 'MOLD_IN_BOT_' in name:
                    p2 = compile('MOLD_IN_BOT_{}')
                    mold_in_bot = p2.parse(name)[0]
                elif 'MOLD_OUT_TOP_' in name:
                    p3 = compile('MOLD_OUT_TOP_{}')
                    mold_out_top = p3.parse(name)[0]
                elif 'MOLD_OUT_BOT_' in name:
                    p4 = compile('MOLD_OUT_BOT_{}')
                    mold_out_bot = p4.parse(name)[0] 
                score = assoc_result['support'].iloc[i]
            
            temp_df.loc[i] = [mold_in_top, mold_in_bot, mold_out_top, mold_out_bot, score]
            temp_df = temp_df.sort_values(by='SUPPORT', ascending=False)
            
        assoc_result = temp_df

        # 금형이름에서 ‘_’를 ‘ ‘로 변환
        for name in CORES:
            assoc_result[name] = assoc_result[name].str.replace('_', '')
            
        st.subheader("연관규칙 결과 테이블")
        st.write(assoc_result)


        # #### 비교군 데이터셋 생성 : 원천 데이터(df_raw) 중 EQUIP_ID와 MOLD_POS가 일치하는 비교군 데이터를 추출한다.

        # In[156]:


        df_temp_raw = df_raw.copy()
        if assoc_pos != 0:
            df_temp_raw = df_temp_raw[(df_temp_raw['EQUIP_ID']==equipment_id) & (df_temp_raw['MOLD_POS']==assoc_pos)]


        # #### 중복 결과데이터 삭제(1) : 비교군 데이터에서 연관규칙 결과와 일치하는 데이터를 모두 찾고 중복되는 값을 제거한다.

        # In[157]:


        df_temp_result = pd.DataFrame(columns=df_temp_raw.columns)
        df_temp_result = df_temp_result.astype(
        {
            'REAL_POWER':'float',
            'IN_RADIUS':'float',
            'OUT_RADIUS':'float',
            'POWER':'float',
            'POWER1':'float',
            'POWER2':'float',
            'POWER3':'float',
            'POWER4':'float',
            'POWER5':'float'
        }
        )


        # In[158]:


        for i in range(assoc_result.shape[0]):
        # 연관규칙 결과와 일치하는 결과를 모두 찾기
            mold_in_top = assoc_result.loc[i,]['MOLD_IN_TOP']
            mold_in_bot = assoc_result.loc[i,]['MOLD_IN_BOT']
            mold_out_top = assoc_result.loc[i,]['MOLD_OUT_TOP']
            mold_out_bot = assoc_result.loc[i,]['MOLD_OUT_BOT']
        
            df_temp = df_temp_raw[(df_temp_raw['MOLD_IN_TOP']==mold_in_top) &                 (df_temp_raw['MOLD_IN_BOT']==mold_in_bot) &                 (df_temp_raw['MOLD_OUT_TOP']==mold_out_top) &                 (df_temp_raw['MOLD_OUT_BOT']==mold_out_bot)
                        ]
            df_temp_result = pd.concat([df_temp_result, df_temp])
        
            # 중복 데이터 제거
        df_temp_result = df_temp_result.drop(['MFG_DT','SALE_CD','REAL_POWER','POWER','CATEGORY'], axis=1)

        df_temp_result = df_temp_result[EQUIP_INFO+CORES+RADIUS+POWERS]
        df_temp_result = df_temp_result.drop_duplicates(CORES+RADIUS)
        df_temp_result.reset_index(inplace=True, drop=True)


        # In[160]:


        df_temp_result[EQUIP_INFO+CORES+POWERS]


        # ####  P 컬럼 생성 : POWER1, POWER2, POWER3, POWER4, POWER5열을 pivot 하고 POWER1, POWER2, POWER3, POWER4, POWER5 값을 통합한 컬럼의 이름을 P로 지정한다.

        # In[161]:


        tmp = df_temp_result.melt(
            id_vars=EQUIP_INFO+CORES,
            value_vars=POWERS,
            value_name='P'
        ).astype({'P':'float'})

        tmp


        # ### P값이 중복되는 데이터 삭제 : EQUIP_ID, MOLD_POS, core group이 같은 데이터를 묶고 나머지 값(P 외)의 중위수를 찾는다

        # In[162]:


        tmp = tmp.groupby(
            EQUIP_INFO+CORES,
            as_index=False,
            sort=False
        ).median()


        # ###  중복 결과데이터 삭제(2) : EQUIP_ID, MOLD_POS, CORES가 일치하는 데이터끼리 묶고 나머지 값의 중위수를 찾는다.

        # In[163]:


        df_temp_result = df_temp_result.groupby(
            EQUIP_INFO+CORES,
            as_index=False,
            sort=False
        ).median()

        df_result = pd.concat([df_result, df_temp_result])


        # ###  RANK_INDEX_1 컬럼값 추가 : df_result의 ‘RANK_INDEX_1’열에 POWER와 target_power의 오차의 절대값을 추가한다.

        # In[164]:


        df_result['RANK_INDEX_1'] = np.abs(tmp['P'].values - target_power)
        df_result


        # ###  연관규칙 결과가 존재하지 않을 때 : 생산이력이 전무한 경우에는 XGBoost 모델이 해당 예시를 학습하지 못했으므로 프로세스를 종료한다

        # In[165]:


        if df_result.shape[0]==0:
            print('입력조건과 일치하는 생산이력이 존재하지 않습니다.')


        # ### 연관규칙 결과와 XGBoost 예측 결과에서 최적조건 찾기
        # 
        # ### 예측도수 변환에 필요한 변수 및 함수 정의
        # 
        # #### POWER_VALUES는 0.25 또는 0.5 단위로 나누어진 도수의 기준표이다. 0부터 –6까지는 0.25 단위로 작아지고, -6부터 –10까지는 0.5 단위로 작아진다.
        # #### 생산된 렌즈는 5차례에 걸쳐 도수를 측정하는데 그 측정값이 POWER1 ~ POWER5 열의 값이며, POWER_VALUES의 값 중에서 측정도수의 평균(POWER 1 ~ POWER 5)과 오차가 가장 작은 값이 최종도수로 결정된다.
        # #### 예를 들어 측정도수의 평균이 –1.82일 때, -1.75와의 오차는 0.07이고 –2.00과의 오차는 0.18이므로 최종 결정도수는 –1.75이다.

        # In[166]:


        # POWER 할당 함수
        POWER_VALUES = np.array([-0.25, -0.5, -0.75,
                -1.0, -1.25, -1.5, -1.75,
                -2.0, -2.25, -2.5, -2.75,
                -3.0, -3.25, -3.5, -3.75,
                -4.0, -4.25, -4.5, -4.75,
                -5.0, -5.25, -5.5, -5.75,
                -6.0, -6.5,
                -7.0, -7.5,
                -8.0, -8.5,
                -9.0, -9.5,
                -10.0])

        def power_allocate(value):
            return POWER_VALUES[np.argmin(abs(POWER_VALUES - value))]

        def vectorize(value):
            return np.vectorize(power_allocate)(value)


        # ### XGBoost Regressor를 이용한 도수 예측
        # 
        # #### 연관규칙 결과 데이터 df_result에서 IN_RADIUS와 OUT_RADIUS의 값을 부호를 없앤 뒤, IN_RADIUS, OUT_RADIUS, EQUIP_ID, MODL_POS를 XGBoost 회귀모델에 넣고 REAL_POWER를 예측한다

        # In[168]:


        # 생산조건(설비, 금형위치)에 따른 REAL_POWER 예측
        df_result[RADIUS] = abs(df_result[RADIUS])

        X = df_result[RADIUS+EQUIP_INFO]
        X.loc[:,'EQUIP_ID'] = equipment_id
        X.loc[:,'MOLD_POS'] = mold_position

        X, ENCODER = transform_cols(X,ENCODER)


        # In[169]:


        # XGBoost Regressor 모델을 이용한 REAL_POWER 예측
        df_result['PREDICTED_POWER'] = regressor.predict(X)
        df_result['PREDICTED_POWER2'] = df_result['PREDICTED_POWER'].apply(vectorize)


        # ### XGBoost 예측결과 내 우선순위 도출
        # 
        # #### 예측 도수(PREDICTED_POWER)와 목표 도수(target_power)의 오차를 RANK_INDEX_2에 저장한다.
        # F
        # #### 입력조건(equipment_id, mold_postion)과 일치하는 데이터가 있는 경우 최종 우선순위(RANK_INDEX)는 RANK_INDEX_1과 RANK_INDEX_2의 합이 되고, 일치하는 입력조건이 없는 경우는 RANK_INDEX_2를 최종 우선순위로 사용한다.
        # 
        # #### 최종 우선순위가 높은 순으로 최대 5개의 최적조건을 도출한다.

        # In[170]:


        df_result['RANK_INDEX_2'] = np.abs(df_result['PREDICTED_POWER'].values - target_power)
        if all(df_result['EQUIP_ID'] == equipment_id) & all(df_result['MOLD_POS'] == mold_position):
            df_result['RANK_INDEX'] = df_result['RANK_INDEX_1'] + df_result['RANK_INDEX_2']
        else:
            df_result['RANK_INDEX'] = df_result['RANK_INDEX_2']
        df_result = df_result.sort_values(by='RANK_INDEX').iloc[:5] # 5개까지

        df_result['EQUIP_ID'] = equipment_id
        df_result['MOLD_POS'] = mold_position


        # ### 결과 출력
        # 
        # #### 사용자가 지정한 목표도수(target_power)와 예측한 결정도수(PREDECTED_POWER2)의 오차를 ERROR에 담은 후 결과테이블을 출력한다

        # In[171]:


        df_result['ERROR'] = df_result['PREDICTED_POWER2'] - target_power


        # In[173]:


        print('\n최종 금형조합 추천 결과')
        showing_cols = EQUIP_INFO+CORES+RADIUS+['PREDICTED_POWER','PREDICTED_POWER2','ERROR']
        st.subheader("최종 금형조합 추천 결과")
        st.write(df_result[showing_cols])