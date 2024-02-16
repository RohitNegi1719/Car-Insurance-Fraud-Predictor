import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix
import pickle

df = pd.read_csv('insurance_claims.csv')

#Cleaning the data
#print(df.info()) #There appears to be no null values except in _c39  there are no values

#drop the _c39
df.drop('_c39',axis=1,inplace=True)

df.rename(columns = {'capital-gains':'capital_gains', 'capital-loss':'capital_loss'}, inplace = True)

#Although there are no null values in dataset but
# in features like  collision_type,policy_report_available,property_damage there are '?' to indicate missing data

# removing the '?' with np.nan. It is a special value in the NumPy library that represents a missing or undefined value.
# It stands for "Not a Number" and is used to denote undefined or unrepresentable values in numerical computations.
# np.nan is often used to represent missing values in data because it can be handled in a consistent way by various
# data analysis and machine learning libraries.

df.replace('?', np.nan, inplace = True)
#print(df.info())  #in features like  collision_type,policy_report_available,property_damage there are null values
#using the fillna() method to replace missing values in three columns (collision_type, property_damage,
# and police_report_available) with the mode (most frequent value) of each respective column.

df['collision_type'] = df['collision_type'].fillna(df['collision_type'].mode()[0])
df['property_damage'] = df['property_damage'].fillna(df['property_damage'].mode()[0])
df['police_report_available'] = df['police_report_available'].fillna(df['police_report_available'].mode()[0])
#print(df.info())  #No null values


#print(list(df.columns))  #printing the columns name
#['months_as_customer', 'age', 'policy_number', 'policy_bind_date', 'policy_state', 'policy_csl', 'policy_deductable',
# 'policy_annual_premium', 'umbrella_limit', 'insured_zip', 'insured_sex', 'insured_education_level',
# 'insured_occupation','insured_hobbies', 'insured_relationship', 'capital-gains', 'capital-loss', 'incident_date',
# 'incident_type', 'collision_type', 'incident_severity', 'authorities_contacted', 'incident_state', 'incident_city',
# 'incident_location', 'incident_hour_of_the_day', 'number_of_vehicles_involved', 'property_damage', 'bodily_injuries',
# 'witnesses', 'police_report_available', 'total_claim_amount', 'injury_claim', 'property_claim', 'vehicle_claim',
# 'auto_make', 'auto_model', 'auto_year', 'fraud_reported']

#Now, there are some columns which do not affect our fraud detection such as policy number,policy state,insured hobbies,
#policy bind date,policy state,incidentlocation,incident date,incident state,incident city,auto make,auto model.
# These are just extra information
# auto year etc .  So,we can drop these columns to make the model more efficient
#Also,the total claim amount columns is the sum of injury claim , property claim and vehicle claim .So ,i will drop it .

df.drop(['policy_number','policy_bind_date','policy_state','insured_zip','incident_location','incident_date','incident_state',
             'incident_city','insured_hobbies','total_claim_amount','auto_make','auto_model','auto_year'],axis=1,inplace=True)

# I could have dropped 'incident_hour_of_the_day' , but it can be an important factor because the freqency of accidents
# can depend on the time of the day which governs factors such as , traffic,visibility etc .


#print(df.info())  # we are left with 25 columns now

#A correlation matrix heat map is a graphical representation of the correlation matrix, which shows the pairwise
# correlation coefficients between different variables in a dataset. The correlation matrix contains the correlation
# coefficients, which measure the strength and direction of the linear relationship between two variables.

# Compute the correlation matrix
corr = df.corr(numeric_only=True)
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))
# Set up the matplotlib figure
fig, ax = plt.subplots(figsize=(10, 10))
# Generate the heatmap with seaborn
sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=True, fmt='.2f', square=True, ax=ax)
# Rotate the tick labels for better readability
plt.xticks(rotation=90)
plt.yticks(rotation=0)
# Show the plot
plt.show()

# age : month as customer  = 0.92 , vehicle claim : injury claim : 0.72 , vehicle claim : property claim :0.73
# Age and month as customer are highly correlated.In this case, one of the variables can be dropped as it does not provide any additional
# information that is not already captured by the other variable. This helps in reducing the number of variables and
# simplifying the model without losing any significant information. On the other hand, the claims are different amounts
# and are individually important. So,i will not drop them.Between months as customer and age , age is important as old people
#and young people tends to not involved in insurance fraud

df.drop(['months_as_customer'],axis =1,inplace = True)
#print(dframe.info())
df['fraud_reported'] = df['fraud_reported'].apply(lambda x:1 if 'Y' in x else 0)
"""
#Random Forest
# Split into X and y
X = df.drop('fraud_reported', axis=1)
y = df['fraud_reported']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=52)

# Define the column transformer
cat_cols = df.select_dtypes(include=['object']).columns
cat_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer(transformers=[('cat', cat_transformer, cat_cols)], remainder='passthrough')

# Define the random forest classifier
rfc = RandomForestClassifier(n_estimators=100, random_state=42)

# Define the pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', rfc)])

# Fit the pipeline
pipeline.fit(X_train, y_train)

# Save the pipeline as a pickle file
with open('Fraud_Predictor.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

# Predict on test data
y_pred = pipeline.predict(X_test)
rfc_test_acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy of Random Forest Classifier is : {rfc_test_acc}")

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:\n', conf_matrix)
# Print precision, recall, and F1 score
classification_rep = classification_report(y_test, y_pred)
print('Classification Report:\n', classification_rep)
"""
#SVM
# Split into X and y
X = df.drop('fraud_reported', axis=1)
y = df['fraud_reported']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=52)

# Define the column transformer
cat_cols = df.select_dtypes(include=['object']).columns
cat_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer(transformers=[('cat', cat_transformer, cat_cols)], remainder='passthrough')

# Define the SVM classifier
svm = SVC(kernel='linear', C=1.0, random_state=42)

# Define the pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('scaler', StandardScaler()), ('classifier', svm)])

# Fit the pipeline
pipeline.fit(X_train, y_train)

# Save the pipeline as a pickle file
with open('Fraud_Predictor_SVM.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

# Predict on test data
y_pred = pipeline.predict(X_test)
svm_test_acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy of SVM Classifier is : {svm_test_acc}")

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:\n', conf_matrix)

# Print precision, recall, and F1 score
classification_rep = classification_report(y_test, y_pred)
print('Classification Report:\n', classification_rep)


""" 
#KNN
X = dframe.drop('fraud_reported', axis = 1)
y = dframe['fraud_reported']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=52)
#Getting the object type
cat_cols = dframe.select_dtypes(include=['object']).columns
#print(cat_cols)
# Define the column transformer
cat_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
col_transformer = ColumnTransformer(transformers=[('cat', cat_transformer, ['policy_csl', 'insured_sex', 'insured_education_level',
       'insured_occupation', 'insured_relationship', 'incident_type','collision_type', 'incident_severity', 'authorities_contacted',
       'property_damage', 'police_report_available'])], remainder='passthrough')

# Define the KNN classifier
knn = KNeighborsClassifier(n_neighbors=35)
# Define the pipeline
pipeline = Pipeline(steps=[('col_transformer', col_transformer), ('knn', knn)])
# Fit the pipeline
pipeline.fit(X_train, y_train)

dframe.to_csv("newdataset.csv", index=False)
pickle.dump(pipeline,open('Fraud_Predictor.pkl','wb'))
# Predict on test data
y_pred = pipeline.predict(X_test)
knn_test_acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy of KNN is : {knn_test_acc}")

# calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:\n', conf_matrix)

print('Classification Report:\n',classification_report(y_test, y_pred))
""" 















