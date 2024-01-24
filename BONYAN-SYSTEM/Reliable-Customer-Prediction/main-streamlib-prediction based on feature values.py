import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px 

np.random.seed(50)

def generate_data():

    num_records=1000

    average_duration_call_day = np.random.exponential(10, num_records)
    average_data_usage_mg = np.random.exponential(500, num_records)
    sum_money_spending_toman = np.random.exponential(1000, num_records)



    # Create a DataFrame
    data_ex = pd.DataFrame({
    'Average Duration Call': average_duration_call_day,
    'Average Data Usage': average_data_usage_mg,
    'Sum of Money Spending': sum_money_spending_toman 
    })
    
    return data_ex

    # Display the DataFrame
data_ex = generate_data()

def classify_cutomers(df):
    
    a  = df['Average Duration Call']
    b  = df['Average Data Usage']
    c  = df['Sum of Money Spending']
    
    if (a > 15 or b > 700) and c > 1500:
        return 'good customer'
    
    elif (a > 15 or c > 1500) and b > 700:
        return 'good customer'
    
    elif (b > 700 or c > 1500) and a > 15:
        return 'good customer'
    
    else:
        return 'bad customers'



def label_customers(df):
    
    if df['classification'] == 'good customer':
        return 1
    
    else:
        return 0 
    

data_ex['classification'] = data_ex.apply(classify_cutomers  , axis = 1 ) 
data_ex['label'] = data_ex.apply(label_customers , axis = 1)


import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , confusion_matrix , precision_score , recall_score , f1_score
from imblearn.over_sampling import SMOTE

X = data_ex.iloc[: , :-2]
y = data_ex.iloc[: , -1]


scaler = MinMaxScaler()

X = scaler.fit_transform(X)

X_train , X_test , y_train , y_test = train_test_split(X , y ,test_size = 0.2 , random_state= 42 )
# Create an instance of SMOTE
smote = SMOTE()

# Generate the synthetic samples using SMOTE
X_train, y_train = smote.fit_resample(X_train, y_train)

from sklearn.linear_model import LogisticRegression
Lo = LogisticRegression()


Lo.fit(X_train , y_train )
import streamlit as st

def predict_active_user(features):
    # Predict the calorie value based on the features
    sample = scaler.transform([features])
    predicted_calories = Lo.predict_proba(sample)
    return predicted_calories[0][1]



def main():
   
    st.title("Loan prediction")

    # Add input widgets for the user to enter the four features
    duration_call = st.slider("avg duration call per day: ", min_value=0, max_value=200)
    data = st.slider("data usage per day (mg): " , min_value=0 , max_value = 5000)
    sum_money = st.slider("sum money spending (toman);", min_value=0, max_value=20000)
    

    # Add a button to initiate the prediction
    if st.button("Predict"):
        # Call the predict_fruit_calories function with the entered features
        features = [duration_call , data , sum_money ]
        predicted_loan =  predict_active_user(features)

        # Display the predicted calorie value
        st.write("Predicted user:", predicted_loan)


        
if __name__ == "__main__":
    main()