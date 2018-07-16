# Spyder Python 3.6 

print("hello world")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset= pd.read_csv('Data.csv')
X= dataset.iloc[:, :-1].values
x = pd.DataFrame(X)
Y= dataset.iloc[:,3].values
y = pd.DataFrame(Y)

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy='mean', axis= 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_x = LabelEncoder()
X[:, 0] = label_x.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

label_y = LabelEncoder()
Y = label_y.fit_transform(Y)


from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train , Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_train)
