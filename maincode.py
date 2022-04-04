import streamlit as st
import numpy as np
import pandas as pd

st.header("Web App Developed by ShahirCheSoh")
readme = st.checkbox("More Information")
if readme:
    st.write("""
        This is a web app demo using [streamlit](https://streamlit.io/) library. It is hosted on [heroku](https://www.heroku.com/). You may get the codes via [github](https://github.com/richieyuyongpoh/myfirstapp)
        """)
    st.write ("Contact me at LinkedIn:")
    st.write("<a href='https://www.linkedin.com/in/shahir-chesoh-1392b1204/'>ShahirCheSoh </a>", unsafe_allow_html=True)

st.write("The datasets is from [Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)")

data = pd.read_csv(r'https://raw.githubusercontent.com/Shahirrrr/Health/main/heart.csv')
X = data[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']]
y = data.target
X
y
