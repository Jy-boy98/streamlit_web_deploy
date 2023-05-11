import streamlit as st
import numpy as np
import altair as alt
import pandas as pd

# 유투브 예제
""" view = [100,150,30]
st.write('## youtube view')
view
st.write('## bar chart')
st.bar_chart(view)
import pandas as pd
sview = pd.Series(view)
sview """

# 3일차
""" st.header('st.button')
if st.button('say hello'): # say hello 라는 버튼을 누르면 아래 문장 실행
    st.write('Why hello there') # 실제 화면에 문장 출력
else:
    st.write('Goodbye') # 초기 설정된 문장 """

# 4일차
st.header('st.write')

## Example 1
st.write('Hello, *world!* :sunglasses:')
## Example 2
st.write(1234)
##Example 3
df = pd.DataFrame({'first column' : [1,2,3,4], 'second column' : [10,20,30,40]})
st.write(df)
##Example 4
st.write('Below is a Dataframe', df, 'Above is a dataframe')
##Example 5
df2 = pd.DataFrame(np.random.randn(200,3), columns = ['a', 'b', 'c'])
c = alt.Chart(df2).mark_circle().encode(x = 'a', y = 'b', size = 'c', tooltip = ['a', 'b', 'c'])
st.write(c)