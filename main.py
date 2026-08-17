import streamlit as st
from home_page import home_page
from preprocessing_page import preprocessing_page

def project_description_page():
    st.title("생산시스템구축실무 프로젝트")
    st.subheader("생산공정 최적화를 위한 콘택트렌즈의 도수와 금형조합")
    
    st.markdown("---")

    st.header("프로젝트 개요")
    st.write("""
    콘택트렌즈 생산 공정 데이터를 활용하여
    생산 조건과 렌즈 도수 간의 관계를 분석하고,
    머신러닝 모델을 통해 목표 도수에 적합한
    생산 조건을 도출하는 프로젝트입니다.
    """)

    st.header("프로젝트 목표")
    st.write("""
    - 생산 데이터 전처리 및 이상치 제거
    - 주요 생산 변수 간 상관관계 분석
    - XGBoost 기반 렌즈 도수 예측 모델 구축
    - 목표 도수에 적합한 금형 조합 및 생산 조건 추천
    """)

    st.header("주요 데이터")
    st.write("""
    분석에 사용되는 주요 변수는 다음과 같습니다.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("생산 조건")
        st.write("""
        - EQUIP_ID : 생산 설비
        - MOLD_POS : 금형 위치
        - MOLD_IN_TOP
        - MOLD_IN_BOT
        - MOLD_OUT_TOP
        - MOLD_OUT_BOT
        """)

    with col2:
        st.subheader("렌즈 데이터")
        st.write("""
        - IN_RADIUS : 내부 곡률
        - OUT_RADIUS : 외부 곡률
        - REAL_POWER : 최종 렌즈 도수
        - POWER1 ~ POWER5 : 측정 도수
        """)

    st.header("분석 프로세스")

    st.write("""
    ① 원본 데이터 확인  
    ↓  
    ② 결측값 처리  
    ↓  
    ③ 데이터 형 변환  
    ↓  
    ④ 이상치 제거  
    ↓  
    ⑤ 변수 간 상관관계 분석  
    ↓  
    ⑥ XGBoost 모델 학습  
    ↓  
    ⑦ 목표 도수 예측  
    ↓  
    ⑧ 최적 금형 조합 추천
    """)

    st.header("사용 기술")

    st.write("""
    Python / Streamlit / Pandas / NumPy / Scikit-learn / XGBoost / Matplotlib / Seaborn
    """)

    st.header("프로젝트 결과")
    st.write("""
    생산 데이터를 기반으로 렌즈 생산 조건과 도수 간 관계를 분석하고,
    머신러닝 모델을 활용하여 목표 렌즈 도수에 적합한 생산 조건을
    예측할 수 있는 데이터 분석 시스템을 구현합니다.
    """)

def main():
    st.sidebar.title("Contents")
    
    # 페이지 선택을 위한 라디오 버튼
    page = st.sidebar.radio("Select a page", ("프로젝트 설명","CSV Input", "Preprocessing / Visualization / Analyzing"))
    
    if page == "프로젝트 설명":
        project_description_page()
    elif page == "CSV Input":
        home_page()
    elif page == "Preprocessing / Visualization / Analyzing":
        preprocessing_page()

if __name__ == '__main__':
    main()
