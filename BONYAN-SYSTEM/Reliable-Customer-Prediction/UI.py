from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import pandas as pd

import joblib

loaded_model = joblib.load(r"C:\Users\USER\Desktop\BONYAN-20240124T100228Z-001\BONYAN\streamlib\bonyan_ml\logistic_regression.pkl")


def main():   
    
    st.title("Good Customer Phone Numbers")

    # Read the data
    data =  pd.read_csv(r'C:\Users\USER\Desktop\BONYAN-20240124T100228Z-001\BONYAN\streamlib\bonyan_ml\data.csv')
    # Filter good customer phone numbers
    threshold = st.slider("set the threshhold: ", min_value=0.0, max_value=1.0 , step = 0.01)
    
    
    # Display the filtered phone numbers


    def filter_good_customers(data):
        X = data[['Average_Duration_Call', 'Average_Data_Usage', 'Money_Spending']]
        scaler = MinMaxScaler()

        X = scaler.fit_transform(X)

        probabilities = loaded_model.predict_proba(X)[:, 1]

        df = data[probabilities >= threshold]
        return df
    
    good_customer_phone_numbers = filter_good_customers(data)

    if len(good_customer_phone_numbers) > 0:
        st.subheader("Phone numbers of being a good customer:")
        st.write(good_customer_phone_numbers)
        st.subheader("number of good customer :")
        st.write(len(good_customer_phone_numbers))
        st.subheader("percentile of whole dataset :")
        st.write(len(good_customer_phone_numbers) / len(data) )
    else:
        st.subheader("No phone numbers found with probability >= 0.9 of being a good customer.")


    def download_csv():
        
        csv = good_customer_phone_numbers.to_csv(index=True)
        return csv

    if st.button('Download CSV'):
        csv = download_csv()

        st.download_button(
            label="Download CSV File",
            data=csv,
            file_name='your_dataset.csv',
            mime='text/csv')