import streamlit as st
from data_preprocessing import preprocessing_data
from data_visualization import visualization_data   
from statistical_analysis import analysis_data

def preprocessing_page():
    st.header("Preprocessing / Visualization / Analyzing Page")
    
    # 하위 페이지 선택을 위한 라디오 버튼
    page = st.radio("Select a Process",("데이터 전처리", "데이터 시각화", "통계적 분석"))
    
    if page == "데이터 전처리":
        preprocessing_data()
    elif page == "데이터 시각화":
        visualization_data()
    elif page == "통계적 분석":
        analysis_data()
    