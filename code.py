# --------------
# Import Libraries
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Code starts here
df = pd.read_csv(path)

print(df.head())

print(df.shape)

df.columns = df.columns.str.lower()
df.columns = df.columns.str.replace(' ', '_')

df.fillna(np.nan, inplace=True)



# Code ends here


# --------------
from sklearn.model_selection import train_test_split
df.set_index(keys='serial_number',inplace=True,drop=True)


# Code starts

df['established_date'] = pd.to_datetime(df['established_date'])
df['acquired_date'] = pd.to_datetime(df['established_date'])

y = df['2016_deposits']
X = df.iloc[:, :-1]

X_train,X_val,y_train,y_val = train_test_split(X, y, test_size=0.25, random_state=3)
print(X_train.shape, X_val.shape)

# Code ends here


# --------------
# time_col = X_train.select_dtypes(exclude=[np.number,'O']).columns
time_col = ['established_date', 'acquired_date']

# Code starts here
new_col_name = []
for time_col in time_col:
    new_col_name.append('since_'+time_col)
for col_name in new_col_name:
    X_train[col_name] = (pd.datetime.now() - X_train[time_col])/np.timedelta64(1,'Y')
    X_val[col_name] = (pd.datetime.now() - X_val[time_col])/np.timedelta64(1,'Y')
print(X_train[new_col_name].tail())
X_train.drop(columns=['established_date', 'acquired_date'], inplace=True)
X_val.drop(columns=['established_date', 'acquired_date'], inplace=True)
print(X_train.shape, X_val.shape)
# Code ends here


# --------------
from sklearn.preprocessing import LabelEncoder
cat = X_train.select_dtypes(include='O').columns.tolist()

# Code starts here
le = LabelEncoder()
for col in cat:
  df[col] = le.fit_transform(df[col])

X_train_temp = pd.get_dummies(X_train, columns=cat)
X_val_temp = pd.get_dummies(X_val, columns=cat)

print(X_train_temp.shape, X_val_temp.shape)

# Code ends here


# --------------
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Code starts here
dt = DecisionTreeRegressor(random_state=5)

dt.fit(X_train,y_train)

accuracy = dt.score(X_val,y_val)

y_pred = dt.predict(X_val)

rmse = np.sqrt(mean_squared_error(y_val,y_pred))


# --------------
from xgboost import XGBRegressor


# Code starts here

xgb = XGBRegressor(max_depth = 50, learning_rate=0.83, n_estimators=100)
xgb.fit(X_train, y_train)

accuracy  = xgb.score(X_val, y_val)

print(f'Accuracy - {accuracy}')

y_pred = xgb.predict(X_val)

rmse = np.sqrt(mean_squared_error(y_val, y_pred))

print(f'RMSE - {rmse}')
# Code ends here


