EMPLOYEE ATTRITION WEB APPLICATION USING PYTHON AND STREAMLIT SHARING.


Goal:
 To Predict The Future Employee Who Would Tend To Leave The Company.

Employee attrition is defined as the natural process by which employees leave the workforce – for example, through resignation for personal reasons or retirement – and are not immediately replaced.

Employee attrition occurs when the size of your workforce diminishes over time due to unavoidable factors such as employee resignation for personal or professional reasons.
Employees are leaving the workforce faster than they are hired, and it is often outside the employer’s control. For example, let’s say that you have opened a new office designated as the Sales Hub for your company. Every salesperson must work out of this office – but a few employees cannot relocate and choose to leave the company. This is a typical reason for employee attrition.
But there are other reasons for attrition as well, including the lack of professional growth, a hostile work environment, or declining confidence in the company’s market value. Weak leadership is another factor that often drives attrition among employees.

Step by Step process in Building the App:
1)Build the Machine Learning Model In Python
2)Build the Web app using Streamlit

1) Building the Machine Learning Algorithm:
The following Steps were taking in building the machine Learning Algorithm.
	a. Data  Pre-processing
The data was downloaded and the imported into the python Environment with aid of the NumPy and pandas libraries. The existing employee dataset and the employee who have left dataset was collected. The two dataset was concatenated to from and single data set, the Employee who have left was assign a label value 1(Target variable Churn), while the existing employee was assigned a label value 0(Target variable Churn)Next, we check for missing values, no missing values was found. 

	b. Exploratory Data Analysis
We perform EDA to give us insight into the dataset. Numerical Descriptive data analysis was performed and Visualization. We also check for the distribution of all the features in the dataset, we found that all of the features are not distributed normally, as thus we decided to use the Non parametric machine learning Algorithm in modeling the dataset

	c. Splitting the data into train and test set
The training dataset was used in training the different machine learning algorithm considered while the test set will be used in Evaluation of the model performance.

	d. Training the Data
The following Non parametric machine learning Algorithm was used
SVM, Random Forest, Decision Tree, K nearest Neighbor Algorithm

	e. Evaluation
The choice of the model was based on the accuracy score and the Kohen kappa value after evaluating the model. Since the data is an unbalanced dataset major weight was given to the Cohen kappa statistics
f)	Prediction
The Prediction is based on input collected by the user of the machine learning algorithm which was made possible by Streamlit.



2) Building the web app using Streamlit
The Streamlit library was implement inside the machine learning code to build the web application. The major function used are 
st.header() for creating header inside the web application
st.subheader() for creating sub header inside of the web application
st.write() for writing text information inside the dataset
st.image() for putting images inside the web application
st.dataframes() for putting table inside   the web application
if st.button() for creating web button where user can click in the web page.

The requirement.txt file should contain a list all the dependencies (Packages) required to run the python code successfully.
For instance, for this project our requirement.txt contains the following

numpy==1.18.5

pandas==1.1.3

matplotlib==3.3.2

seaborn==0.10.1

streamlit==0.69.2

Pillow==7.2.0

scikit-learn==0.23.2

xlrd >= 1.0.0

## Usage
- Clone my repository.
- Open CMD/terminal in working directory.
- Run following command.
  ```
  pip install -r requirements.txt
  ```
- `app.py` is the main Python file of Streamlit Web-Application. 
- To run app, write following command in CMD. or use any IDE.
  ```
  streamlit run app.py
  ```
- For more explanation of this project see the tutorial on Machine Learning Hub YouTube channel.

# Hurray, Your App is Up and Running. 
## Just follow☝️ me and Star⭐ my repository 

