import streamlit as st
from home_page import home_page
from preprocessing_page import preprocessing_page

def project_description_page():
    st.title("생산시스템구축실무 프로젝트")
    st.subheader("생산공정 최적화를 위한 콘택트렌즈의 도수와 금형조합")

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
