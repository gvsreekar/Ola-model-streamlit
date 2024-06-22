import numpy as np 
import pandas as pd 
import joblib 
from custom_transformers import column_names
import streamlit as st

loaded_model = joblib.load('rfc_model_final.joblib')

def ola_churn_prediction(input_data:pd.DataFrame):
    pred = loaded_model.predict(input_data)
    if pred[0]==1:
        return 'The driver with the entered details will churn.'
    else:
        return 'The driver with the entered details will stay in OLA.'
    
    
def main():
    
    st.title('Ola drivers churn predictor:car:')
    #st.image('Ola_logo.png',use_column_width=True)
    st.write("**Enter the details of driver to predict whether he/she would stay or churn**")
    
    output = ""
    
    # Creating 2 columns
    col1,col2 = st.columns(2)
    
    with col1:
        # Take details in the input
        Age = st.number_input("Enter the age of the Driver", min_value=18.0, max_value=65.0, step=1.0, value=25.0)
        gender_options = [0.0,1.0]
        Gender = st.selectbox("Select the gender of the Driver from the dropdown (0 : Male, 1: Female)",gender_options)
        city_options = ['C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11',
                     'C12','C13','C14','C15','C16','C17','C18','C19','C20','C21','C22',
                     'C23','C24','C25','C26','C27','C28','C29']
        City = st.selectbox("Select the city from the dropdown", city_options)
        Education_Level = st.number_input("Enter the education level of the Driver (0 for 10+ ,1 for 12+ ,2 for graduate)",min_value=0.0,max_value=2.0,step=1.0)
        Income = st.slider("Enter monthly income of Driver",min_value=0.0,max_value=3000000.0,value=30000.0)
    
    with col2:
        jd_options = [1,2,3,4,5]
        Joining_designation = st.selectbox("Select the joining designation of Driver",jd_options)
        Grade = st.selectbox("Select grade of the Driver",jd_options)
        tbv = st.slider("Enter total business value of the Driver",min_value=-300000,max_value=3000000,value=100000)
        rating = st.selectbox("Select the rating of the Driver",jd_options)
        increased = [0.0,1.0]
        income_increased = st.selectbox("Has the income of Driver increased (0.0 for No and 1.0 for yes)",increased)
        rating_increased = st.selectbox("Has the rating of Driver improved (0.0 for No and 1.0 for yes)",increased)
        
    
    if st.button("Submit",use_container_width=10):
        data = {
            
            'Age':Age,
            'Gender':Gender,
            'City':City,
            'Education_Level':Education_Level,
            'Income':Income,
            'Joining Designation':Joining_designation,
            'Grade':Grade,
            'Total Business Value':tbv,
            'Quarterly Rating':rating,
            'income_increased':income_increased,
            'rating_increased':rating_increased   
        }
        
        input_data = pd.DataFrame([data])
        output = ola_churn_prediction(input_data)
    
    st.success(output)
    
if __name__=='__main__':
    main()
