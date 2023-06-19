import streamlit as st
import pandas as pd

def home_page():
    st.header("CSV 업로드")
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("CSV file uploaded successfully!")

        st.dataframe(df)  # CSV 파일의 내용을 표시

        # 컬럼 수와 행 수를 보여줌
        num_columns = len(df.columns)
        num_rows = len(df)
        st.write(f"Number of Columns: {num_columns}", text_size=18)
        st.write(f"Number of Rows: {num_rows}", text_size=18)

if __name__ == '__main__':
    home_page()