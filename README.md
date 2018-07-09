# Data-cleaning-in-Machine-learning

Basics of processing the Machine Learning Dataset

The Objective of this project was to process the data before applying to any machine learning model and Spyder Python 3.6 is used 

###Code contains information about reading Dataset and applying basics methods to it 

Import important libraries 
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```
Dataset is loaded with segregating dependent and independent columns
```
dataset= pd.read_csv('Data.csv')
X= dataset.iloc[:, :-1].values  [ Considering all rows and columns except last column ]
x = pd.DataFrame(X)
Y= dataset.iloc[:,3].values [Considering all rows and only last column ] 
y = pd.DataFrame(Y)
```

Imputer is for filling the empty fields in the data 
```
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy='mean', axis= 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
```

LabelEncoder is for converting categorical string into int and OneHotEncoder [Dummy encoding] for nullifying the int value  
```
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_x = LabelEncoder()
X[:, 0] = label_x.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
label_y = LabelEncoder()
Y = label_y.fit_transform(Y)
```

Splitting the data into train and test data

from sklearn.cross_validation import train_test_split 
However the "cross_validation" name is now deprecated and was replaced by "model_selection" inside the new anaconda versions.
Therefore you might get a warning or even an error if you run this line of code above.

To avoid this, you just need to replace:
from sklearn.cross_validation import train_test_split 
by
from sklearn.model_selection import train_test_split 
test_size should be 0.25 to 0.3 or 0.4 max
```
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train , Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
```
Feature scaling 
Basically works on euclidean distance  
standardisation and normalization 
xstand=x-mean(x)/standard deviation(x)
xnorm=x-min(x)/max(x)-min(x) 

```
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_train)
```
