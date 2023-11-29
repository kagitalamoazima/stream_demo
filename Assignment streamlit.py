import warnings
warnings.filterwarnings("ignore")
warnings.resetwarnings()

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle
import random
from sklearn.linear_model import LinearRegression

df=pd.DataFrame(columns=['exp','degree','pay'])

for i in range(25):
    Exp = random.randint(5,10)
    Degree = random.choice(["High_school","Bachelor's","Master's","PhD"])
    Pay = random.randint(50000,100000)
    df.loc[len(df.index)] = [Exp,Degree,Pay] 

df.head()

df.to_csv("practice_dataset.csv")

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
scaler = MinMaxScaler()

x=df.drop('pay',axis=1)
y=df.pay
x

df_num = x.select_dtypes('number')
df_num_scaled = scaler.fit_transform(df_num)

# Scaling the numerical column to decrease the difference in the column.
x_num_scaled = pd.DataFrame(df_num_scaled,columns=df_num.columns,index=df_num.index)

x_cat = x.select_dtypes('object')
x_cat_scaled = label_encoder.fit_transform(x_cat)
x_cat_encoded=pd.DataFrame(x_cat_scaled,columns=x_cat.columns,index=x_cat.index)

X = pd.concat([df_num, x_cat_encoded], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train,y_train)

train_pred =  model.predict(X_train)

test_pred = model.predict(X_test)

print(f"The train mse: {train_pred}")

print(f"The test mse: {test_pred}")

#save the model
with open('model.plk','wb') as linear_streamlit:
    pickle.dump(model,linear_streamlit)