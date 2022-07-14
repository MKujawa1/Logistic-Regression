import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
%matplotlib inline 

#Load data
ad_data = pd.read_csv('advertising.csv')

ad_data.head()

ad_data.info()

ad_data.describe()

#Histogram of age
sns.set_style('whitegrid')
ad_data['Age'].hist(bins=30)
plt.xlabel('Age')

sns.jointplot(x = 'Age',y = 'Area Income', data = ad_data)

sns.jointplot(x = 'Age',y = 'Daily Time Spent on Site',data = ad_data, color = 'red', kind = 'hex')

sns.jointplot(x = 'Daily Time Spent on Site', y = 'Daily Internet Usage',data = ad_data,color = 'green')

sns.pairplot(data = ad_data, hue = 'Clicked on Ad',palette='bwr')

from sklearn.model_selection import train_test_split

# Split data
X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Male']]
y = ad_data['Clicked on Ad']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from sklearn.linear_model import LogisticRegression

# Init and fit model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Model predictions and evaluations
predictions = log_reg.predict(X_test)

from sklearn.metrics import classification_report

print(classification_report(y_test,predictions))