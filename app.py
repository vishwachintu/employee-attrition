#life of an employee at an organization.
#Author: Vishwa - Drpinnacle

#import libraries
import numpy as np
import pandas as pd
import pandas_profiling
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import xlrd
import io

#import libraries from streamlit 
import streamlit as st
from streamlit_pandas_profiling import st_profile_report

#machinlearning libraries
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import accuracy_score,classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

#set page configuration and this can be initiated only once
st.set_page_config(layout="wide", page_title='DrLoE App')

#Tittle, Header, sub header and any other..... 
st.title("DrLoE(Life of an Employee) APP")

st.subheader(
    """
    APP to Predict when your employee will leave the company
""")

#initiate side bar for naviagtion
#Add the expander to provide some information about the app
st.sidebar.header("DrLoE APP")
with st.sidebar.expander("About the DrLoE App", expanded=True):
     st.write("""
        This interactive people management App was built by Vishwa(DrPinnacle) using Streamlit.
     """)

#Rating for the app Create a user feedback section to collect comments and ratings from users
with st.sidebar.form(key='columns_in_form',clear_on_submit=True): #set clear_on_submit=True so that the form will be reset/cleared once it's submitted
    st.write('Please help us improve!')
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;} </style>', unsafe_allow_html=True) #Make horizontal radio buttons
    rating=st.radio("Please rate the app",('1','2','3','4','5'),index=4)    #Use radio buttons for ratings
    text=st.text_input(label='Please leave your feedback here') #Collect user feedback
    submitted = st.form_submit_button('Submit')
    if submitted:
      st.write('Thanks for your feedback!')
      st.markdown('Your Rating:')
      st.markdown(rating)
      st.markdown('Your Feedback:')
      st.markdown(text)

with st.sidebar.expander("About the DrLoE App", expanded=True):
     st.write("""
        This interactive people management App was built by Vishwa(DrPinnacle) using Streamlit.
     """)

#User Choice
user_choices = ['Online Predictions', 'Batch Predictions']
selected_choice = st.selectbox("Please select your choice:", user_choices)
#for online predictions
if selected_choice is not None:
    if selected_choice == 'Online Predictions':
    # st.write('You selected:', user_choices)
        st.write(" Please fill in the below form and remember that teh employee will be unknow")

    #defining for online predictions
        def user_input_online():
            Employee_name = st.text_input("Please enter employee name")
            Satisfaction_level = st.number_input("Satisfaction level",min_value=0.00, max_value= 0.99,value=0.5)
            Last_evaluation = st.number_input("Last Evaluation",min_value=0.00, max_value= 0.99,value=0.5)
            number_project =st.number_input('Number of project',min_value=0, max_value= 10,value=5)
            average_montly_hours = st.number_input('The average montly hours',min_value=0.00, max_value= 1000.00,value=300.00)
            time_spend_company  = st.number_input('Time spend in company',min_value=0, max_value= 20,value=5)
            Work_accident =st.selectbox('Work accident',(0, 1))
            promotion_last_5years = st.selectbox('Promotion last 5 years',(0, 1))
            dept = st.selectbox('Department',("sales","technical","support","IT","hr","accounting","marketing","product_mng","randD","mangement"))
            Salary =  st.selectbox('Salary Level ',("low","medium","high"))
            
            ### Dictionaries of Input
            input_user= {"Satisfaction_level":Satisfaction_level ,"Last_evaluation":Last_evaluation, "number_project":number_project,"average_montly_hours":average_montly_hours,"time_spend_company":time_spend_company,"Work_accident":Work_accident,"promotion_last_5years":promotion_last_5years, "dept":dept,"Salary":Salary}
                    
            ### Converting to a Dataframes
            input_user =pd.DataFrame(input_user,index=[0])
            return input_user

        input_value = user_input_online()                               

        # print(input_value.info())
                
        # Label Encoding will be used for columns with 2 or less unique values

        ## Encoding The  Categorical Variables
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder
        le1= LabelEncoder()

        le1_count = 0
        for col in input_value.columns:
            if input_value[col].dtypes == 'object':
                le1.fit(input_value[col])
                input_value[col] = le1.transform(input_value[col])
                le1_count += 1


        print('{} columns were label encoded.'.format(le1_count))

        if st.button("Predict"):
            Prediction = rforest.predict(input_value)
            if Prediction == 0:
                result = pd.DataFrame({"Churn":Prediction,"Info":"The Employee will not Leave the Ogarnization"})
            else:
                result = pd.DataFrame({"Churn":Prediction,"Info":"The Employee wil Leave the Ogarnization"})                      
            
            st.write("""
            # The Result of the Classification:
            """)
            st.write("Attrition : ")  
            st.dataframe(result)

    #For Batch Predictions
    elif selected_choice == 'Batch Predictions':
        st.write('Please upload your dataset in the form csv')
        source_data = st.file_uploader("Upload/select source (.csv) data", type=['csv'], accept_multiple_files=True)
        for source_data in source_data:
            bytes_data = source_data.read()
            st.write("filename:", source_data.name)
            st.write(bytes_data)

    #defining for batch predictions
        # def user_input_batch():



# df = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/train.csv")
# pr = df.profile_report()

# st_profile_report(pr)