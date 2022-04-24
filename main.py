# Created By : Ishraque Zaman Borshon
# Oklahoma State University
# 04/23/2022

# Random Forest Regression

# Importing the libraries
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Nu_Correlation_Physical_Based.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values


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

    print('R2 value of the model', regressor.score(X_test, y_test))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    Random_Forest()
