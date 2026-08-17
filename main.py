import streamlit as st
from home_page import home_page
from preprocessing_page import preprocessing_page

def project_description_page():

    st.set_page_config(
        page_title="Lens Process Optimization",
        page_icon="📊",
        layout="wide"
    )

    st.title("Lens Process Optimization Dashboard")
    st.caption("생산공정 데이터를 활용한 콘택트렌즈 도수 예측 및 최적 금형 조합 추천")

    st.markdown("---")

    # 핵심 요약 카드
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("분석 데이터", "162K+ rows")

    with c2:
        st.metric("주요 변수", "44 Features")

    with c3:
        st.metric("예측 모델", "XGBoost")

    with c4:
        st.metric("최적화 목표", "Power Prediction")

    st.markdown("### Project Overview")

    st.write("""
    본 프로젝트는 콘택트렌즈 생산 공정 데이터를 기반으로
    설비, 금형 위치, 금형 조합, 곡률값 등의 생산 조건과
    최종 렌즈 도수 간의 관계를 분석하는 데이터 기반 생산 최적화 프로젝트입니다.

    전처리 및 이상치 제거 후 머신러닝 모델을 학습하고,
    사용자가 원하는 목표 도수에 적합한 생산 조건과 금형 조합을 추천합니다.
    """)

    st.markdown("### Analysis Pipeline")

    p1, p2, p3, p4 = st.columns(4)

    with p1:
        st.info("① Data Preprocessing\n\n결측값 / 형변환 / 이상치 제거")

    with p2:
        st.info("② Data Analysis\n\n상관관계 및 생산 변수 분석")

    with p3:
        st.info("③ Model Training\n\nXGBoost 기반 도수 예측")

    with p4:
        st.info("④ Optimization\n\n최적 금형 조합 추천")

    st.markdown("### Tech Stack")

    st.code(
        "Python | Streamlit | Pandas | NumPy | Scikit-learn | XGBoost | Matplotlib | Seaborn",
        language=None
    )

    st.markdown("---")

    st.success(
        "왼쪽 사이드바에서 CSV Input 또는 Preprocessing / Visualization / Analyzing 메뉴를 선택해 분석을 시작하세요."
    )

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
