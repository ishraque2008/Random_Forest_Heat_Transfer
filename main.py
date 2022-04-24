# Created By : Ishraque Zaman Borshon
# Oklahoma State University
# 04/23/2022

# Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import pandas as pd
from sklearn.preprocessing import  LabelEncoder
import xgboost as xgb
import numpy as np
st.header("Heat Loss Prediction App")
st.text_input("Enter your Name: ", key="name")

# Importing the dataset
dataset = pd.read_csv('Nu_Correlation_Physical_Based.csv')
X = dataset.iloc[:, 1:-2].values
y = dataset.iloc[:, 15:17].values


def Random_Forest():
    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # # Feature Scaling
    # from sklearn.preprocessing import StandardScaler
    # sc_X = StandardScaler()
    # X_train = sc_X.fit_transform(X_train)
    # X_test = sc_X.transform(X_test)
    # sc_y = StandardScaler()
    # y_train = sc_y.fit_transform(y_train)

    #    Fitting Random Forest Regression to the dataset
    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators=10, random_state=30)

    # from sklearn.model_selection
    # regressor = Regression()
    # regressor.fit(X_train, y_train)
    regressor.fit(X_train, y_train)

    # print('R2 value of the model', regressor.score(X_test, y_test))
    # prediction = regressor.predict([[-15,9.81,0.5,0.3,0.6,300,773,536.5,0.001863933,0.7,0.00004765,0.5,0.19635,3.444845]])
    # conv_loss = prediction[0][0]
    # nu_number = prediction[0][1]
    # print(conv_loss)
    # print(nu_number)


    input_g = 9.81
    imput_length = 0.5
    input_massflow = 0.3
    input_density = 0.6
    input_Ta = 300
    input_Phi = st.slider('Tilt angle(Degree)', 0, max(dataset["Tilt"]), 1)
    input_surface_temp = st.slider('Receiver Surface Temperature (K)', 520, max(dataset["Ts"]), 1)
    input_avg_temp = (input_Ta+input_surface_temp)/2
    input_beta = 1/input_avg_temp
    input_pr = 0.7
    input_vel = 0.00004765
    input_aperture_dia = 0.5
    input_aperture_area = 0.19635
    input_tube_area = 3.444845

    encoder = LabelEncoder()
    # encoder.classes_ = np.load('classes.npy', allow_pickle=True)
    # # load model
    best_xgboost_model = xgb.XGBRegressor()
    if st.button('Make Prediction'):
        # input_species = encoder.transform(np.expand_dims(inp_species, -1))
        inputs = np.expand_dims(
            [input_Phi, input_g, imput_length, input_massflow, input_density, input_Ta, input_surface_temp,
             input_avg_temp,input_beta, input_pr, input_vel, input_aperture_dia, input_aperture_area, input_tube_area  ], 0)
        prediction = best_xgboost_model.predict(inputs)
        conv_loss = prediction[0][0]
        nu_number = prediction[0][1]
        print("final pred", np.squeeze(prediction, -1))
        st.write(f"Your Convective heat loss is: {conv_loss} W/m^2")  #
        st.write(f"Your Nusselt Number is: {nu_number} ")

if __name__ == '__main__':
    Random_Forest()
