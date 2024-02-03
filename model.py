import numpy as np  
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score

#Used to calc error
def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0  #ensure no 0 division
    return np.mean(diff) * 100

def mape_scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    return smape(y, y_pred)

#Read file
data = pd.read_csv('Data.csv')

data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace = True)

#Calculating moving avergaes
data['SMA_10'] = data['Close_A'].rolling(window=10).mean()

##Calculating RSI
delta = data["Close_A"].diff()

gain = delta.clip(lower = 0)
loss = -delta.clip(upper = 0)

avg_gain = gain.rolling(window = 14, min_periods =14).mean()
avg_loss = loss.rolling(window = 14, min_periods = 14).mean()

rs = avg_gain/avg_loss

data['RSI'] = 100 - (100/ (1 + rs))

#regulization
columns_to_standardize = ['Close_A', 'Open_A', 'High_A', 'Low_A', 'Volumes_A','Change_A', 'Price_S', 'Change_S', 
                          'Price_T', 'Change_T','SMA_10', 'RSI']

scaler = MinMaxScaler()

for col in columns_to_standardize:
    standardized = scaler.fit_transform(data[[col]])
    data[col + '_Standardized'] = pd.DataFrame(standardized, index=data.index)

X = data[['Open_A_Standardized', 'High_A_Standardized', 'Low_A_Standardized', 'Volumes_A_Standardized','Change_A_Standardized', 
          'Price_S_Standardized', 'Change_S_Standardized', 'Price_T_Standardized', 'Change_T_Standardized', 
          'SMA_10_Standardized', 'RSI_Standardized' ]]
Y = data['Close_A_Standardized']
    
#drop non-regularized data
columns_to_drop = ['Close_A', 'Open_A', 'High_A', 'Low_A', 'Volumes_A', 'Change_A', 'Price_S', 'Change_S', 
                   'Price_T', 'Change_T', 'SMA_10', 'RSI']

data = data.drop(columns=columns_to_drop, axis=1)

# Splitting the dataset (70% training, 15% validation, 15% testing)
total_data_points = len(data)
train_size = int(total_data_points * 0.70)
validation_size = int(total_data_points * 0.15)
full_train_size = int(total_data_points * 0.85)
test_size = total_data_points - train_size - validation_size

train_data = data.iloc[:train_size]
validation_data = data.iloc[train_size:train_size+validation_size]
full_train_data = data.iloc[:full_train_size] #cv + original training set
test_data = data.iloc[train_size+validation_size:]

#training Data
X_train = train_data.drop('Close_A_Standardized', axis=1)  #drops close_A
y_train = train_data['Close_A_Standardized']

#cv set
X_validation = validation_data.drop('Close_A_Standardized', axis=1)
y_validation = validation_data['Close_A_Standardized']

#test
X_test = test_data.drop('Close_A_Standardized', axis=1)    
y_test = test_data['Close_A_Standardized']  

#imputer, replaces NaN values with acc values so training can ocur
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

model = LinearRegression()

model.fit(X_train_imputed, y_train)

#use cv to test
y_validation_pred = model.predict(X_validation)

plt.figure(figsize=(10, 6))
plt.plot(y_validation.index, y_validation.values, label='Actual', color='blue', marker='o')
plt.plot(y_validation.index, y_validation_pred, label='Predicted', color='red', linestyle='--', marker='x')
plt.title('Comparison of Actual and Predicted Values on Validation Set')
plt.xlabel('Date')
plt.ylabel('Close Price (x200 USD)')  
plt.legend()
plt.show()

X_train_full = full_train_data.drop('Close_A_Standardized', axis = 1)
y_train_full = full_train_data['Close_A_Standardized']

#Use imputation to remove NaN values
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X_train_full_imputed = imputer.fit_transform(X_train_full)
X_test_imputed = imputer.transform(X_test)


#test the full model
model_full = LinearRegression()
model_full.fit(X_train_full_imputed, y_train_full)

y_pred_full = model_full.predict(X_test_imputed)

plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test.values, label='Actual', color='blue', marker='o')
plt.plot(y_test.index, y_pred_full, label='Predicted', color='red', linestyle='--', marker='x')
plt.title('Comparison of Actual and Predicted Values on Test Set')
plt.xlabel('Date')
plt.ylabel('Close Price (x200 USD)') 
plt.legend()
plt.show()

#Calc error for basline performance
baseline_pred = y_train.mean()
baseline_preds = np.full_like(y_test, fill_value=baseline_pred)
baseline_preds = np.full_like(y_test, fill_value=baseline_pred)
baseline_smape = smape(y_test, baseline_preds)

#Calc training error error
y_train_pred = model.predict(X_train_imputed)
training_smape = smape(y_train, y_train_pred)

#Calc cv error
cv_scores = cross_val_score(model, X_train_imputed, y_train, cv=5, scoring=mape_scorer)
cv_smape = np.mean(cv_scores)

print(f"\n\n\nBaseline SMAPE: {baseline_smape:.2f}%, Training SMAPE: {training_smape:.2f}%, CV SMAPE: {cv_smape:.2f}%")