import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import matplotlib.pyplot as plt


# Load the data
filename = '2002_final.csv'
data = pd.read_csv(filename, encoding='ISO-8859-1')

# Dropping rows with missing 'ArrDelay' (target) values
data = data.dropna(subset=['ArrDelay'])

#Dropping less useful columns
data = data.drop(['Cancelled', 'Diverted','TaxiIn', 'AirTime', 'Year','UniqueCarrier','TailNum', 'ActualElapsedTime', 'CRSElapsedTime'], axis=1)

# Remove records with missing values
data = data.dropna()

# Encode categorical variables
encoder = OrdinalEncoder()
data[['Origin', 'Dest']] = encoder.fit_transform(data[['Origin', 'Dest']])

# Split the data into training and testing sets
X = data.drop('ArrDelay', axis=1)
y = data['ArrDelay']

# Making target variable binary where value>0 shows delayed (1) and <0 shows not delayed (0)
y = y.astype('int')
y = np.where(y>0,1,0)

# Splitting the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Creating & Training the Decision Tree Model
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train,y_train)
dt_prediction=dt_model.predict(X_test)

# Find AUC and ROC for Decision Tree model
dt_fpr, dt_tpr, threshold =roc_curve(y_test,dt_prediction)
auc_dt = auc(dt_fpr, dt_tpr)
print('Area under the curve for Decision Tree = ',auc_dt)

#Creating & Training the Logistic Regression Model
lr_model = LogisticRegression(multi_class='auto', solver='lbfgs', max_iter=5000)
lr_model.fit(X_train,y_train)
X_test = X_test.astype(float)
lr_prediction=lr_model.predict(X_test)

# Find AUC and ROC for Logistic Regression model
lr_fpr, lr_tpr, threshold =roc_curve(y_test,lr_prediction)
auc_lr = auc(lr_fpr, lr_tpr)
print('Area under the curve for Logistic Regression = ',auc_lr)

# Plotting the ROC for both the models
plt.plot(lr_fpr, lr_tpr,label='Logistic Regression (AUC=%0.2f)' % auc_lr)
plt.plot(dt_fpr, dt_tpr, linestyle='-', label='Decision Tree (AUC=%0.2f)' % auc_dt)

# Plotting the baseline
plt.plot([0, 1], [0, 1], linestyle='--', color='Green', label='Baseline')

plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
plt.show()


