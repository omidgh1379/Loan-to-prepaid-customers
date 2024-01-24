import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def generate_phone_num():    
    phone_number = []
    
    fitst_digit = ['912' , '913' ,'914' , '915' , '991' , '992']
    
    for i in range(1000):
        
        first_tree = np.random.choice(fitst_digit)
        remaing = ''.join(str(np.random.randint(0 , 9)) for _ in range(7))
            
        full = first_tree + remaing
        
        
        phone_number.append(full)  
        phone_num = [int(i) for i in phone_number]
        
    return phone_num


phone_num = generate_phone_num()

def generate_data():

    num_records=1000

    average_duration_call_day = np.random.exponential(10, num_records)
    average_data_usage_mg = np.random.exponential(500, num_records)
    sum_money_spending_toman = np.random.exponential(1000, num_records)
    

    # Create a DataFrame
    data_ex = pd.DataFrame({
    'Average_Duration_Call': average_duration_call_day,
    'Average_Data_Usage': average_data_usage_mg,
    'Money_Spending': sum_money_spending_toman ,
    'phone num' : phone_num
    })

    data_ex = data_ex.set_index('phone num')
    
    return data_ex
#generate the data

data_ex = generate_data()
#load the model from joblib

loaded_model = joblib.load('C:\\Users\\omid\\Desktop\\streamlib\\logistic_regression.pkl')




def filter_good_customers(data):
    X = data[['Average_Duration_Call', 'Average_Data_Usage', 'Money_Spending']]
    scaler = MinMaxScaler()

    X = scaler.fit_transform(X)
    
    probabilities = loaded_model.predict_proba(X)[:, 1]
    return data[probabilities >= 0.99].index
    
def test():
    good_cus_phone_num = list(filter_good_customers(data_ex))

    df = pd.DataFrame(good_cus_phone_num , columns = ['phone_num'] )
    return df

