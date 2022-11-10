# life of an employee at an organization.
# Author: Vishwa, Raghu - Drpinnacle

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import xlrd
import io
import joblib
from category_encoders import one_hot
from io import StringIO

# import libraries from streamlit
import streamlit as st

# machinlearning libraries
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# set page configuration and this can be initiated only once
st.set_page_config(layout="wide", page_title="DrLoE App")

# Tittle, Header, sub header and any other.....
st.title("DrLoE(Life of an Employee) APP")

st.subheader(
    """
    APP to Predict when your employee will leave the company
"""
)

# initiate side bar for naviagtion
# Add the expander to provide some information about the app
st.sidebar.header("DrLoE APP")
with st.sidebar.expander("About the DrLoE App", expanded=True):
    st.write(
        """
        This interactive people management App was built by Vishwa, Raghu(DrPinnacle) using Streamlit.
     """
    )

# Rating for the app Create a user feedback section to collect comments and ratings from users
with st.sidebar.form(
    key="columns_in_form", clear_on_submit=True
):  # set clear_on_submit=True so that the form will be reset/cleared once it's submitted
    st.write("Please help us improve!")
    st.write(
        "<style>div.row-widget.stRadio > div{flex-direction:row;} </style>",
        unsafe_allow_html=True,
    )  # Make horizontal radio buttons
    rating = st.radio(
        "Please rate the app", ("1", "2", "3", "4", "5"), index=4
    )  # Use radio buttons for ratings
    text = st.text_input(
        label="Please leave your feedback here"
    )  # Collect user feedback
    submitted = st.form_submit_button("Submit")
    if submitted:
        st.write("Thanks for your feedback!")
        st.markdown("Your Rating:")
        st.markdown(rating)
        st.markdown("Your Feedback:")
        st.markdown(text)

with st.sidebar.expander("Contact Us", expanded=True):
    st.write(
        """
        info@drpinnacle.com
     """
    )

# User Choice
user_choices = ["Online Predictions", "Batch Predictions"]
selected_choice = st.selectbox("Please select your choice:", user_choices)


def cleanData(df, cat_columns):

    """
    Clean the dataframe, arrange the dataframe into
    model required format
    Input:   df(dataframe)
            cat_columns: categorical columns(['salary','dept'])
            ohe_fl: onehotEncode model
            model: predictive model
    Output: preduction result
    """

    ce_ohe = joblib.load("clean_ohe.pkl")
    model = joblib.load("emp_rf.pkl")

    loaded_ce_dummies = ce_ohe.transform(df[cat_columns])

    data_other_cols = df.drop(columns=cat_columns)
    # # #Concatenate the two dataframes
    pred_df = pd.concat([loaded_ce_dummies, data_other_cols], axis=1)

    pred_res = model.predict(pred_df.values)

    return pred_res[0]


def cleanDataBatch(df, cat_columns):

    """
    Clean the dataframe, arrange the dataframe into
    model required format
    Input:   df(dataframe)
            cat_columns: categorical columns(['salary','dept'])
            ohe_fl: onehotEncode model
            model: predictive model
    Output: preduction result
    """

    ce_ohe = joblib.load("clean_ohe.pkl")
    model = joblib.load("emp_rf.pkl")

    loaded_ce_dummies = ce_ohe.transform(df[cat_columns])

    data_other_cols = df.drop(columns=cat_columns)
    # # #Concatenate the two dataframes
    pred_df = pd.concat([loaded_ce_dummies, data_other_cols], axis=1)
    pred_res = model.predict(pred_df.values)
    return pred_res


# for online predictions
if selected_choice is not None:
    if selected_choice == "Online Predictions":
        st.write(
            " Please fill in the below form and remember that teh employee will be unknow"
        )
        # defining for online predictions
        def user_input_online():
            Employee_name = st.text_input("Please enter employee name")
            Employee_ID = st.text_input("Please enter employee ID")
            Satisfaction_level = st.number_input(
                "Satisfaction level", min_value=0.00, max_value=0.99, value=0.5
            )
            Last_evaluation = st.number_input(
                "Last Evaluation", min_value=0.00, max_value=0.99, value=0.5
            )
            number_project = st.number_input(
                "Number of project", min_value=0, max_value=10, value=5
            )
            average_montly_hours = st.number_input(
                "The average montly hours",
                min_value=0.00,
                max_value=1000.00,
                value=300.00,
            )
            time_spend_company = st.number_input(
                "Time spend in company", min_value=0, max_value=20, value=5
            )
            Work_accident = st.selectbox("Work accident", (0, 1))
            promotion_last_5years = st.selectbox("Promotion last 5 years", (0, 1))
            dept = st.selectbox(
                "Department",
                (
                    "sales",
                    "technical",
                    "support",
                    "IT",
                    "hr",
                    "accounting",
                    "marketing",
                    "product_mng",
                    "randD",
                    "mangement",
                ),
            )
            salary = st.selectbox("Salary Level ", ("low", "medium", "high"))

            ### Dictionaries of Input
            input_user = {
                "Satisfaction_level": Satisfaction_level,
                "Last_evaluation": Last_evaluation,
                "number_project": number_project,
                "average_montly_hours": average_montly_hours,
                "time_spend_company": time_spend_company,
                "Work_accident": Work_accident,
                "promotion_last_5years": promotion_last_5years,
                "dept": dept,
                "salary": salary,
            }

            ### Converting to a Dataframes
            input_user = pd.DataFrame(input_user, index=[0])

            Prediction = cleanData(input_user, ["salary", "dept"])
            return Prediction,Employee_ID

        pred_result, Employee_ID = user_input_online()

        if st.button("Predict"):
            if pred_result == 0:
                result = st.write(f'{Employee_ID} will not leave the organization')
                #result = {"Info": "The Employee will not Leave the Ogarnization"}
            else:
               result = st.write(f'{Employee_ID} will leave the organization')
                #result = {"Info": "The Employee wil Leave the Ogarnization"}
            #st.write("Attrition : ")
            st.write(result)

    # For Batch Predictions
    elif selected_choice == "Batch Predictions":
        st.markdown("please make sure you have collected **_right_ _data_ _points_** to run the batch predicctions")
        st.write("Please upload your dataset in the form csv")
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:

            # To read file as bytes:
            bytes_data = uploaded_file.getvalue()

            # To convert to a string based IO:
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))

            # To read file as string:
            string_data = stringio.read()

            # Can be used wherever a "file-like" object is accepted:
            dataframe = pd.read_csv(uploaded_file, index_col=0)
            pred_result = cleanDataBatch(dataframe, ["salary", "dept"])
            result_df = pd.DataFrame()
            result_df["result"] = pred_result

            @st.cache
            def convert_df(df):
                # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode("utf-8")

            csv = convert_df(result_df)
            file_name = uploaded_file.name.split(".")[0] + "_result" + ".csv"
            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name=file_name,
                mime="text/csv",
            )
