import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Get the Data
df = pd.read_csv('KNN_Project_Data')
df.head()


# Exploratory Data Analysis (EDA)
# Create a pairplot with the hue indicated by the TARGET CLASS column
sns.pairplot(df, hue='TARGET CLASS', palette='coolwarm')


# Standardize The Variables
scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS', axis=1))

# Transform teh features to a scaled version
scaled_features = scaler.transform(df.drop('TARGET CLASS', axis=1))

# Convert the scaled features to a dataframe
df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])
df_feat.head()


# Train Test Split
# Split the data into a training set and testing set
X = df_feat
y = df['TARGET CLASS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# Using KNN
knn = KNeighborsClassifier(n_neighbors=1)

# Fit KNN model to the training data
knn.fit(X_train, y_train)


# Predictions and Evaluations
# Predict values using KNN model and X_test
pred = knn.predict(X_test)

# Create report
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))


# Choose a K Value
# Train various KNN models with different K values, keep track of error_rate with a list
error_rate = []

for i in range(1, 60):

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

# Create a plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, 60), error_rate, color='blue', linestyle='--', marker='o', markerfacecolor='red', markersize='10')
plt.title('Error Rate vs K')
plt.xlabel('K')
plt.ylabel('Error Rate')


# Retrain With A New K Value
# Re-do the report with the best K value
knn = KNeighborsClassifier(n_neighbors=30)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)
print(confusion_matrix(y_test, pred))
print('\n')
print(classification_report(y_test, pred))
