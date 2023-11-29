
import streamlit as st
import pickle
from sklearn.metrics import mean_squared_error as mse

with open('model.plk','rb') as linear_streamlit:
    model = pickle.load(linear_streamlit)

st.header("Linear Rgression")

st.sidebar.header("This is a webapp")

experience = st.sidebar.slider("select the exp ",5,6,7,8,9,10)
education = st.sidebar.slider("select the education ",1,2,3,4)

st.write("Test values are",experience," ",education)

test_predict = model.predict([[experience,education]])

st.write(f"The predicted salary is: {test_predict[0]}")