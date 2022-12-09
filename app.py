import numpy as np 
import pandas as pd 
import joblib
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import streamlit as st
from PIL import Image


st.set_option('deprecation.showPyplotGlobalUse', False)
# Load  model a 
model = joblib.load(open("model-v1.joblib","rb"))

def data_preprocessor(df):
    """this function preprocess the user input
        return type: pandas dataframe
    """
    df.wine_type = df.wine_type.map({'Working at Computer':0, 'Standing Up, Walking and Going up-down stairs':1, 'Standing':2 , 'Walking':3 ,'Going Up\Down Stairs':4 ,'Walking and Talking with Someone':5 ,'Talking while Standing': 6})
    return df

def visualize_confidence_level(prediction_proba):
    """
    this function uses matplotlib to create inference bar chart rendered with streamlit in real-time 
    return type : matplotlib bar chart  
    """
    data = (prediction_proba[0]*100).round(2)
    grad_percentage = pd.DataFrame(data = data,columns = ['Percentage'],index = ['Class 0','Class 1','Class 2','Class 3','Class 4','Class 5','Class 6'])
    ax = grad_percentage.plot(kind='barh', figsize=(7, 4), color='#722f37', zorder=10, width=0.5)
    ax.legend().set_visible(False)
    ax.set_xlim(xmin=0, xmax=100)
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

    ax.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")
    
    vals = ax.get_xticks()
    for tick in vals:
        ax.axvline(x=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)

    ax.set_xlabel(" Percentage(%) Confidence Level", labelpad=2, weight='bold', size=12)
    ax.set_ylabel("Human Activities", labelpad=10, weight='bold', size=12)
    ax.set_title('Prediction Confidence Level ', fontdict=None, loc='center', pad=None, weight='bold')

    st.pyplot()
    return

st.write("""
# Human Activity Prediction ML Web-App 
This app predicts the ** Human Activity **  using **features extracted from accelerometer time signals** input via the **side panel**.  
""")

#read in wine image and render with streamlit
image = Image.open('har.jfif')
st.image(image, caption='Human activities',use_column_width=True)

st.sidebar.header('User Input Parameters') #user input parameter collection with streamlit side bar


def get_user_input():
    """
    this function is used to get user input using sidebar slider and selectbox 
    return type : pandas dataframe

    """
    #wine_type = st.sidebar.selectbox("Select Wine type",("white", "red"))
    x_mean = st.sidebar.slider('x_mean', -3.0, 2.69, 1.08)
    y_mean = st.sidebar.slider('y_mean', -5.99, 1.92, -0.81)
    z_mean = st.sidebar.slider('z_mean', -3.88, 2.53, -0.48)
    x_std = st.sidebar.slider('x_std',-0.80, 13.32, -0.76)
    y_std = st.sidebar.slider('y_std', -0.68, 12.42, -0.66)
    z_std = st.sidebar.slider('z_std', -0.786, 17.17, -0.66)
    x_peak_count = st.sidebar.slider('x_peak_count', -2.63, 1.66, 1.023)
    y_peak_count = st.sidebar.slider('y_peak_count', -2.39, 1.55, 0.829)
    z_peak_count = st.sidebar.slider('z_peak_count', -2.52, 1.81, -0.02)
    x_skewness  = st.sidebar.slider('x_skewness', -11.16, 11.69, 0.3)
    y_skewness  = st.sidebar.slider('y_skewness', -17.23, 17.27, -0.351)
    z_skewness  = st.sidebar.slider('z_skewness', -15.68, 10.98, -0.042)
    x_kurtosis  = st.sidebar.slider('x_kurtosis', -0.642, 26.26, -0.349)
    y_kurtosis  = st.sidebar.slider('y_kurtosis', -0.473, 28.44, -0.276)
    z_kurtosis  = st.sidebar.slider('z_kurtosis', -0.559, 34.76, -0.361)
    x_energy  = st.sidebar.slider('x_energy', -2.806, 3.10, 1.08)
    y_energy  = st.sidebar.slider('y_energy', -5.365, 2.01, -0.827)
    z_energy  = st.sidebar.slider('z_energy', -3.649, 2.70, -0.512)

    
    features = {
            'x_mean': x_mean,
            'y_mean': y_mean,
            'z_mean': z_mean,
            
            'x_std': x_std,
            'y_std': y_std,
            'z_std': z_std,
            
            'x_peak_count': x_peak_count,
            'y_peak_count': y_peak_count,
            'z_peak_count': z_peak_count,
            
            'x_skewness': x_skewness,
            'y_skewness': y_skewness,
            'z_skewness': z_skewness,
            
            'x_kurtosis': x_kurtosis,
            'y_kurtosis': y_kurtosis,
            'z_kurtosis': z_kurtosis,
            
            'x_energy': x_energy,
            'y_energy': y_energy,
            'z_energy': z_energy,

            }
    data = pd.DataFrame(features,index=[0])

    return data

user_input_df = get_user_input()

st.subheader('User Input parameters')
st.write(user_input_df)

prediction = model.predict(user_input_df)

prediction_proba = model.predict_proba(user_input_df)

visualize_confidence_level(prediction_proba)