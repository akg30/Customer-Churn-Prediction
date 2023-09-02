# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 16:44:47 2023

@author: Hp
"""

import numpy as np
import streamlit as st
import pickle as pk

loaded_model=pk.load(open('C:\customer churn ml/trained_model.sav','rb'))

def churn_pred(input_data_churn):
    input_data_array_churn=np.asarray(input_data_churn)
    input_data_array_new_churn=input_data_array_churn.reshape(1,-1)
    prediction_churn=loaded_model.predict(input_data_array_new_churn)
    if prediction_churn==0:
        return 'Customer Is Not Going To Churn'
    else:
        return 'Customer Is Going To Churn'

def main():
    st.title('Customer Churn Predition Using Machine Learning')
    creditscore=st.number_input('Enter Credit Score Of Person')
    age=st.number_input('Enter Age Of Person')
    balance=st.number_input('Enter Account Balance Of Person')
    tenure=st.number_input('Enter Tenure Of Person')
    numofproducts=st.number_input('Enter Number Of Bank Products Use By Person')
    st.warning('Choose 0 For No And 1 For Yes For Next Two Questions') 
    hascrcard=st.radio('Choose Whether Person Has Credit Card Not',[0,1])
    isactivemember=st.radio('Choose Whether Person Is Active Member Not',[0,1])
    estimatedsalary=st.number_input('Enter Estimated Salary Of Person')
    churn_predict=' '
    if st.button('Click Here To Know Whether A Person Is Going To Churn Or Not'):
        churn_predict=churn_pred([creditscore,age,balance,tenure,numofproducts,hascrcard,isactivemember,estimatedsalary])
    st.success(churn_predict)
    st.markdown('##### Exploratory Data Analysis Done And Machine Learning Model Deployed By "Anubhav Kumar Gupta"')
    
if __name__=='__main__':
    main()
