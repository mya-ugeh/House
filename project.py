import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import warnings
warnings.filterwarnings('ignore')
import pickle
from sklearn.linear_model import LinearRegression
import streamlit as st
import joblib
import time



data = pd.read_csv('USA_Housing.csv')

model = joblib.load(open('house.pkl', 'rb'))

import base64
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('house.jpg') 

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html=True)

with st.form('my_form', clear_on_submit=True,):
    st.header("HOUSE PREDICTION")
    income = st.number_input('Area Income')
    age = st.number_input('Area House Age')
    room = st.number_input('Number of Rooms')
    pop = st.number_input('Area population')
    submitted = st.form_submit_button("PREDICT")
    if (income and age and room and pop):
        if submitted:
            with st.spinner(text='In progress'):
                time.sleep(3)
                st.success('Done')
            st.write("Your Inputted Data:")
            input_var = pd.DataFrame([{'Avg. Area Income' : income,	'Avg. Area House Age' : age,	'Avg. Area Number of Rooms' : room,	'Area Population' : pop}])
            st.write(input_var) 
            time.sleep(2)
            tab1, tab2 = st.tabs(["Prediction Pane", "Intepretation Pane"])
            with tab1:
                if submitted:
                    st.markdown("<br>", unsafe_allow_html= True)
                    prediction = model.predict(input_var)
                    st.write("Predicted Profit is :", prediction)
                else:
                    st.write('Pls press the predict button for prediction')
            with tab2:
                st.subheader('Model Interpretation')
                st.write(f"Profit = {model.intercept_.round(2)} + {model.coef_[0].round(2)} R&D Spend + {model.coef_[1].round(2)} Administration + {model.coef_[2].round(2)} Marketing Spend")

                st.markdown("<br>", unsafe_allow_html= True)

                st.markdown(f"- The expected Profit for a startup is {model.intercept_}")

                st.markdown(f"- For every additional 1 dollar spent on R&D Spend, the expected profit is expected to increase by ${model.coef_[0].round(2)}  ")

                st.markdown(f"- For every additional 1 dollar spent on Administration Expense, the expected profit is expected to decrease by ${model.coef_[1].round(2)}  ")

                st.markdown(f"- For every additional 1 dollar spent on Marketting Expense, the expected profit is expected to increase by ${model.coef_[2].round(2)}")


# modal = Modal("Demo Modal",key=2)
# open_modal = st.button("Name")
# if open_modal:
#     modal.open()

# if modal.is_open():
#     with modal.container():
#         st.write("Text goes here")

#         html_string = '''
#         <h1>HTML string in RED</h1>

#         <script language="javascript">
#           document.querySelector("h1").style.color = "red";
#         </script>
#         '''
#         components.html(html_string)

#         st.write("Some fancy text")
#         value = st.checkbox("Check me")
#         st.write(f"Checkbox checked: {value}")