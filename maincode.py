import streamlit as st
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

st.header("Welcome to my Streamlit [Machine Learning]")
st.write("In this WebApp, I will demonstrate 3 types of Machine Learning which are k-Nearest Neighbors, Support Vector Machine (SVM) and Random Forest)
readme = st.checkbox("More Information")
if readme:
    st.write("""
        This is a web app demo using [streamlit](https://streamlit.io/) library. It is hosted on [heroku](https://www.heroku.com/). You may get the codes via [github](https://github.com/Shahirrrr/Health)
        """)
    st.write ("Contact me at LinkedIn:")
    st.write("<a href='https://www.linkedin.com/in/shahir-chesoh-1392b1204/'>ShahirCheSoh </a>", unsafe_allow_html=True)

st.write("The datasets is from [Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)")

data = pd.read_csv(r'https://raw.githubusercontent.com/Shahirrrr/Health/main/heart.csv')
X = data[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']]
y = data.target

classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ('KNN','SVM','Random Forest')
)

test_data_ratio = st.sidebar.slider('Select testing size or ratio', 
                                    min_value= 0.10, 
                                    max_value = 0.50,
                                    value=0.2)
random_state = st.sidebar.slider('Select random state', 1, 9999,value=1234)
st.write("## 1: Summary (X variables)")

if len(X)==0:
   st.write("<font color='Aquamarine'>Note: Predictors @ X variables have not been selected.</font>", unsafe_allow_html=True)
else:
   st.write('Shape of predictors @ X variables :', X.shape)
   st.write('Summary of predictors @ X variables:', pd.DataFrame(X).describe())

st.write("## 2: Summary (y variable)")

if len(y)==0:
   st.write("<font color='Aquamarine'>Note: Label @ y variable has not been selected.</font>", unsafe_allow_html=True)
elif len(np.unique(y)) <5:
     st.write('Number of classes:', len(np.unique(y)))

else: 
   st.write("<font color='red'>Warning: System detects an unusual number of unique classes. Please make sure that the label @ y is a categorical variable. Ignore this warning message if you are sure that the y is a categorical variable.</font>", unsafe_allow_html=True)
   st.write('Number of classes:', len(np.unique(y)))

def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0,value=1.0)
        params['C'] = C
    elif clf_name == 'KNN':
        K = st.sidebar.slider('K', 1, 15,value=5)
        params['K'] = K
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15,value=5)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100,value=10)
        params['n_estimators'] = n_estimators
    return params

params = add_parameter_ui(classifier_name)

def get_classifier(clf_name, params):
    clf = None
    if clf_name == 'SVM':
        clf = SVC(C=params['C'])
    elif clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    else:
        clf = clf = RandomForestClassifier(n_estimators=params['n_estimators'], 
            max_depth=params['max_depth'], random_state=random_state)
    return clf

clf = get_classifier(classifier_name, params)


st.write("## 3: Classification Report")

if len(X)!=0 and len(y)!=0: 

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_data_ratio, random_state=random_state)
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)    

  clf.fit(X_train_scaled, y_train)
  y_pred = clf.predict(X_test_scaled)

  st.write('Classifier:',classifier_name)
  st.write('Classification report:')
  report = classification_report(y_test, y_pred,output_dict=True)
  df = pd.DataFrame(report).transpose()
  st.write(df)

else: 
   st.write("<font color='Aquamarine'>Note: No classification report generated.</font>", unsafe_allow_html=True)
        
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
