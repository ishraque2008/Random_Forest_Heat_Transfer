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
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import numpy as np

st.header("CSP Receiver Heat Loss Prediction Software")
st.write('Author: Ishraque Zaman Borshon')
st.write('Source: \"Borshon, I. Z., 2021. Study of Upward facing cavity receiver for Scheffler\'s Concentrator. '
         'M.Tech. Thesis, Indian Institute of Technology, Bombay.\" ')
name = st.text_input("Enter your Name: ", key="name")



# Importing the dataset
dataset = pd.read_csv('Nu_Correlation_Physical_Based_12_removed.csv')
X = dataset.iloc[:, 1:15].values
y = dataset.iloc[:, 15:18].values

if st.checkbox('Show dataframe'):
    dataset

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

    st.write('')
    input_g = 9.81
    input_length = 0.5
    input_massflow = st.slider('Mass Flow Rate (Kg/s ',min_value= min(dataset["Mass Flow Rate"]),
                               max_value =max(dataset["Mass Flow Rate"]))
    input_density = 0.6
    input_Ta = 300
    input_Phi = st.slider('Tilt angle(Degree)', -90, 0, 0)
    input_surface_temp = st.slider('Receiver Surface Temperature (K)', min_value= 520.,
                                   max_value =max(dataset["Ts"]))
    input_avg_temp = (input_Ta + input_surface_temp) / 2
    input_beta = 1 / input_avg_temp
    input_pr = 0.7
    input_vel = 0.00004765
    input_aperture_dia = 0.5
    input_aperture_area = 0.19635
    input_tube_area = 3.444845

    st.write('Length of the receiver ', input_length, 'm')
    st.write('Aperture of receiver ', input_aperture_dia, 'm')
    st.write('Density of Air ', input_density, 'kg/m^3')
    st.write('Ambient Temperature ', input_Ta, 'K')
    st.write('Prandtl Number ', input_pr)


    if st.button('Make Prediction'):
        # input_species = encoder.transform(np.expand_dims(inp_species, -1))
        inputs = np.expand_dims(
            [input_Phi, input_g, input_length, input_massflow, input_density, input_Ta, input_surface_temp,
             input_avg_temp, input_beta, input_pr, input_vel, input_aperture_dia, input_aperture_area, input_tube_area],
            0)
        prediction = regressor.predict(inputs)
        conv_loss = prediction[0][0]
        nu_number = prediction[0][1]
        h = prediction[0][2]
        # print("final pred", np.squeeze(prediction, -1))
        st.write(f"Convective heat loss is: {np.round(conv_loss)} W")  #
        st.write(f"Nusselt Number is: {np.round(nu_number)} ")
        st.write(f"Convective heat transfer coefficient is: {np.round(h)} W/m^2 ")
        st.write(f'Hey {name}, Best of luck on your CSP Receiver designing!')
        st.write('I hope the software was helpful.')


if __name__ == '__main__':
    Random_Forest()
