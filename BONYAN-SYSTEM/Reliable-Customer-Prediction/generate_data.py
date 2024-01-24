"""
"""
import numpy as np
import pandas as pd


def generate_phone_num():
    
    np.random.RandomState(3)
    phone_number = []
    
    fitst_digit = ['912' , '913' ,'914' , '915' , '991' , '992']

    for _ in range(1000):       
        first_tree = np.random.choice(fitst_digit)
        remaing = ''.join(str(np.random.randint(0 , 9)) for _ in range(7))
            
        full = first_tree + remaing
        phone_number.append(full)
        
        
    return phone_number


def generate_data():
    np.random.RandomState(3)
    phone_num = generate_phone_num()
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



data = generate_data()
data.to_csv('C:\\Users\\omid\\Desktop\\streamlib\\bonyan_ml\\data.csv' , index = True)