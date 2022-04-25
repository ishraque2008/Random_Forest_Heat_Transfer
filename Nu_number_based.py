# Author: : Ishraque Zaman Borshon
# Oklahoma State University
# 04/23/2022

# Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

# Importing the dataset
dataset = pd.read_csv('Nusselt_12removed.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values


def Random_Forest():
    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    #    Fitting Random Forest Regression to the dataset
    from sklearn.ensemble import RandomForestRegressor

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=15)

    # # Feature Scaling
    # from sklearn.preprocessing import StandardScaler
    # sc = StandardScaler()
    # X_train = sc.fit_transform(X_train)
    # X_test = sc.transform(X_test)
    # y_train = sc.fit_transform(y_train)

    # Fitting and Parameter Tuning
    # Creating a dictionary to contain parameters with maximum accuracy
    max_parameter_dictionary = {}

    '''    # Varying number of trees in the forest
    accuracy_list = []
    scores = []
    j = 0
    accuracy_old = 0
    treelist = range(1, 150)
    for i in treelist:
        regressor = RandomForestRegressor(n_estimators=i, random_state=30)
        regressor.fit(X_train, y_train)
        # accuracy = regressor.score(X_train, y_train)
        score = cross_val_score(regressor, X, y, cv=5)
        score = np.mean(score)
        scores.append(score)
    plt.plot(treelist, scores)
    plt.title('Variation in number of trees in forest')
    plt.xlabel('Number of trees in forest')
    # plt.title('Score for leaves')
    # plt.xlabel('Leaves')
    plt.ylabel('Score')
    plt.show()
    print(max(scores))'''

    # accuracy = regressor.score(X_test, y_test)
    #     accuracy_list.append(accuracy)
    #     if accuracy > accuracy_old:
    #         accuracy_old = accuracy
    #         j = i
    # max_parameter_dictionary['n_estimators'] = j

    # plt.plot(treelist,accuracy_list)
    # plt.title('Variation in number of trees in forest')
    # plt.xlabel('Number of trees in forest')
    # plt.ylabel('Score')
    # plt.savefig('Number_of_Trees.jpg')
    # plt.show()

    '''scores = []
    j = 0
    accuracy_old = 0
    depthlist = range(1, 50)
    for i in depthlist:
        regressor = RandomForestRegressor(n_estimators=20,max_depth=i, random_state=30)
        regressor.fit(X_train, y_train)
        # accuracy = regressor.score(X_train, y_train)
        score = cross_val_score(regressor, X, y, cv=5)
        score = np.mean(score)
        scores.append(score)
    plt.plot(depthlist, scores)
    plt.title('Variation in depth of each tree in forest')
    plt.xlabel('Depth of trees in forest')
    plt.ylabel('Score')
    plt.savefig('CV_Depth_of_Tree.jpg')
    plt.show()
    print(max(scores))'''

    '''scores = []
    j = 0

    accuracy_old = 0
    leaf_nodes_max = range(2, 40)
    for i in leaf_nodes_max:
        regressor = RandomForestRegressor(n_estimators=20,max_depth=10, max_leaf_nodes =i, random_state=30)
        regressor.fit(X_train, y_train)
        # accuracy = regressor.score(X_train, y_train)
        score = cross_val_score(regressor, X, y, cv=5)
        score = np.mean(score)
        scores.append(score)
    plt.plot(leaf_nodes_max,scores)
    plt.title('Variation in number of leaf nodes in trees')
    plt.xlabel('Leaf nodes of trees')
    plt.ylabel('Score')
    plt.savefig('CV Leaf_nodes_of_Trees.jpg')
    plt.show()
    print(max(scores))


    # Varying depth of trees in the forest
    accuracy_list = []
    j = 0
    accuracy_old = 0
    depthlist = range(1, 50)
    for i in depthlist:
        regressor = RandomForestRegressor(n_estimators=max_parameter_dictionary['n_estimators'],
                                          max_depth=i, random_state=30)
        regressor.fit(X_train, y_train)
        accuracy = regressor.score(X_train, y_train)
        # accuracy = regressor.score(X_test, y_test)
        accuracy_list.append(accuracy)
        if accuracy > accuracy_old:
            accuracy_old = accuracy
            j = i
    max_parameter_dictionary['max_depth'] = j

    # plt.plot(depthlist,accuracy_list)
    # plt.title('Variation in depth of each tree in forest')
    # plt.xlabel('Depth of trees in forest')
    # plt.ylabel('Score')
    # plt.savefig('Depth_of_Tree.jpg')
    # plt.show()
    #
    #

    # Varying leaf nodes of trees in the forest
    accuracy_list = []
    j = 0
    accuracy_old = 0
    leaf_nodes_max = range(2, 40)
    for i in leaf_nodes_max:
        regressor = RandomForestRegressor(n_estimators=max_parameter_dictionary['n_estimators'],
                                          max_depth=max_parameter_dictionary['max_depth'],
                                          max_leaf_nodes=i, random_state=30)
        regressor.fit(X_train, y_train)
        accuracy = regressor.score(X_train, y_train)
        # accuracy = regressor.score(X_test, y_test)
        accuracy_list.append(accuracy)
        if accuracy > accuracy_old:
            accuracy_old = accuracy
            j = i
    max_parameter_dictionary['max_leaf_nodes'] = j

    # plt.plot(leaf_nodes_max,accuracy_list)
    # plt.title('Variation in number of leaf nodes in trees')
    # plt.xlabel('Leaf nodes of trees')
    # plt.ylabel('Score')
    # plt.savefig('Leaf_nodes_of_Trees.jpg')
    # plt.show()
    #
    accuracy_list = []
    # # Varying max samples split of trees in the forest
    j = 0
    accuracy_old = 0
    max_samples_list = range(1, 56)
    for i in max_samples_list:
        regressor = RandomForestRegressor(n_estimators=max_parameter_dictionary['n_estimators'],
                                          max_depth=max_parameter_dictionary['max_depth'],
                                          max_leaf_nodes=max_parameter_dictionary['max_leaf_nodes'],
                                          max_samples=i, random_state=30)
        regressor.fit(X_train, y_train)
        accuracy = regressor.score(X_train, y_train)
        # accuracy = regressor.score(X_test, y_test)
        accuracy_list.append(accuracy)
        if accuracy > accuracy_old:
            accuracy_old = accuracy
            j = i
    max_parameter_dictionary['max_samples'] = j
    # plt.plot(max_samples_list, accuracy_list)
    # plt.title('Variation in number of leaf nodes in trees')
    # plt.xlabel('Maximum samples')
    # plt.ylabel('Score')
    # plt.savefig('Maximum_samples_split_of_Trees.jpg')
    # plt.show()



    regressor = RandomForestRegressor(n_estimators=max_parameter_dictionary['n_estimators'],
                                      max_depth=max_parameter_dictionary['max_depth'],
                                      max_leaf_nodes=max_parameter_dictionary['max_leaf_nodes'],random_state=30,
                                      max_samples=20)'''

    # Taking output from tuning data for fitting the model
    regressor = RandomForestRegressor(n_estimators=20, random_state=30, max_depth=8, max_leaf_nodes=20, max_samples=55)
    # regressor.fit(X, y)
    regressor.fit(X_train, y_train)
    score = regressor.score(X_train, y_train)
    print(score)
    score = regressor.score(X_test, y_test)
    print(score)
    regressor.fit(X, y)

    # accuracy_list = []
    # for i in range(1, 200):
    #     for j in range(1, 20):
    #         for k in range(2, 55):
    #             for l in range(1, 10):
    #                 regressor = RandomForestRegressor(n_estimators=i, max_depth=j, max_leaf_nodes=k,
    #                                                   max_samples=l, random_state=30)
    #                 regressor.fit(X_train, y_train)
    #                 score = cross_val_score(regressor, X, y, cv=5)
    #                 accuracy_list.append(np.mean(score))
    #
    # plt.plot(accuracy_list)

    # Finding Scores of the fit
    accuracy = regressor.score(X_test, y_test)
    score = cross_val_score(regressor, X, y, cv=5)
    print("Mean cross validation score is ", np.mean(score))
    print("Maximum cross validation score is ", np.max(score))
    print("Minimum cross validation score is ", np.min(score))
    print(accuracy)
    print(max_parameter_dictionary.items())

    # Cross validation
    # scores = []

    # for i in range(2, 50):
    #     bp_model = RandomForestRegressor(max_samples=i)
    #     score = cross_val_score(bp_model, X, y, cv=5)
    #     score = np.mean(score)
    #     scores.append(score)
    # plt.plot(range(2, 50), scores)
    # plt.title('Score for leaves')
    # plt.xlabel('Max Samples')
    # # plt.title('Score for leaves')
    # # plt.xlabel('Leaves')
    # plt.ylabel('Score')
    # plt.show()
    # print(max(scores))

    # Reversing back the scaled features
    # print(sc_X.inverse_transform(X_test) )
    # print(sc_X.inverse_transform(y_test))
    # print('R2 value of the model', regressor.score(X_test, y_test))

    # input_g = 9.81
    # input_length = 0.5
    # input_massflow = st.slider('Mass Flow Rate (Kg/s ', min_value=min(dataset["Mass Flow Rate"]),
    #                            max_value=max(dataset["Mass Flow Rate"]))
    # input_density = 0.6
    # input_Ta = 300
    # input_Phi = st.slider('Tilt angle(Degree)', -90, 0, 0)
    # input_surface_temp = st.slider('Receiver Surface Temperature (K)', min_value=520.,
    #                                max_value=max(dataset["Ts"]))
    # input_avg_temp = (input_Ta + input_surface_temp) / 2
    # input_beta = 1 / input_avg_temp
    # input_pr = 0.7
    # input_vel = 0.00004765
    # input_aperture_dia = 0.5
    # input_aperture_area = 0.19635
    # input_tube_area = 3.444845

    # Offline Testing with missing 12 data
    #
    input_series = [[369349789.6,	-30,	2.91,	0.8],
                    [236444735.9,	-90	,1.91,	0.8],
                    [264595264.6	,-90,	2.08,	0.8],
                    [264595264.6,	-30	,2.08	,0.8],
                    [312641035,	-30,	2.41,	0.8],
                    [204873729.8,	-15	,1.74,	0.8],
                    [333305641.8,	-45,	2.58,	0.8],
                    [369349789.6,	-15,	2.91,	0.8],
                    [312641035	,-15,	2.41,	0.8],
                    [385161470.2	,0	,3.08,	0.8],
                    [264595264.6	,-60,	2.08,	0.8],
                    [236444735.9,	-60	1.91	,0.8]
                    ]

        # [[-90, 9.81, 0.5, 0.3, 0.6, 300, 523, 411.5, 0.002430134, 0.7, 0.00004765, 0.5, 0.19635, 3.444845],
        #             [-75, 9.81, 0.5, 0.3, 0.6, 300, 573, 436.5, 0.002290951, 0.7, 0.00004765, 0.5, 0.19635, 3.444845],
        #             [-90, 9.81, 0.5, 0.3, 0.6, 300, 573, 436.5, 0.002290951, 0.7, 0.00004765, 0.5, 0.19635, 3.444845],
        #             [-75, 9.81, 0.5, 0.3, 0.6, 300, 623, 461.5, 0.002166847, 0.7, 0.00004765, 0.5, 0.19635, 3.444845],
        #             [-45, 9.81, 0.5, 0.3, 0.6, 300, 673, 486.5, 0.002055498, 0.7, 0.00004765, 0.5, 0.19635, 3.444845],
        #             [-15, 9.81, 0.5, 0.3, 0.6, 300, 723, 511.5, 0.001955034, 0.7, 0.00004765, 0.5, 0.19635, 3.444845],
        #             [-90, 9.81, 0.5, 0.3, 0.6, 300, 723, 511.5, 0.001955034, 0.7, 0.00004765, 0.5, 0.19635, 3.444845],
        #             [-45, 9.81, 0.5, 0.3, 0.6, 300, 773, 536.5, 0.001863933, 0.7, 0.00004765, 0.5, 0.19635, 3.444845],
        #             [-45, 9.81, 0.5, 0.3, 0.6, 300, 823, 561.5, 0.001780944, 0.7, 0.00004765, 0.5, 0.19635, 3.444845],
        #             [-30, 9.81, 0.5, 0.3, 0.6, 300, 873, 586.5, 0.00170503, 0.7, 0.00004765, 0.5, 0.19635, 3.444845],
        #             [0, 9.81, 0.5, 0.3, 0.6, 300, 923, 611.5, 0.001635323, 0.7, 0.00004765, 0.5, 0.19635, 3.444845],
        #             [-60, 9.81, 0.5, 0.3, 0.6, 300, 923, 611.5, 0.001635323, 0.7, 0.00004765, 0.5, 0.19635, 3.444845]
        #             ]
    nu_list = []

    for i in input_series:
        inputs = np.expand_dims(i, 0)  # ,0)

        prediction = (regressor.predict(inputs))
        conv_loss = prediction[0][0]
        nu_number = prediction[0][1]
        h = prediction[0][2]
        nu_list.append(nu_number)
    #
    for i in nu_list:
        print(i)
    # input_1 = np.expand_dims(
    #     [-90,9.81,0.5,	0.3,0.6,300	,523,	411.5,	0.002430134,	0.7	,0.00004765	,0.5,	0.19635,3.444845],
    #     0)
    # prediction = regressor.predict(input_1)
    # # prediction = (regressor.predict([-90,9.81,0.5,	0.3,0.6,300	,523,	411.5,	0.002430134,	0.7	,0.00004765	,0.5,	0.19635,3.444845],0))
    # conv_loss = prediction[0][0]
    # nu_number = prediction[0][1]
    # print(nu_number)


if __name__ == '__main__':
    Random_Forest()
