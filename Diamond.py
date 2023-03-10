"""
Diamond Price Prediction 

"""

__date__ = "2023-01-16"
__author__ = "Thomas Mullan"


# %% --------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error,r2_score, accuracy_score, mean_absolute_error, explained_variance_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xg
from sklearn.pipeline import Pipeline

# %% --------------------------------------------------------------------------
# Generate Random State 
# -----------------------------------------------------------------------------
rng = np.random.RandomState(123)

# %% --------------------------------------------------------------------------
# Load Data 
# -----------------------------------------------------------------------------
# Read the text file into a pandas DataFrame
df = pd.read_csv(
    r'C:/mle06/Machine Learning/Code/Thomas/Diamond Project/diamonds.csv'
    )
# Print the first 5 rows of the DataFrame
print(df.head())
#df.info()
#df.describe()
#Remove unnamed:0 column
df = df.drop("Unnamed: 0", axis=1)

#df.shape
#Check for null values
null_columns=df.columns[df.isnull().any()]
df.isnull().sum()

#Drop x, y, z values = 0
df.drop(df[df["x"]==0].index)
df.drop(df[df["y"]==0].index)
df.drop(df[df["z"]==0].index)


# %% --------------------------------------------------------------------------
# Data Visualisations 
# -----------------------------------------------------------------------------
#Histogram plot of features 
df.hist(bins=10,figsize=(9,7),grid=False)
#Pair Plot
sns.pairplot(df, hue="cut")
#Correlation Plot w/ heatmap 
corrmat= df.corr()
f, ax = plt.subplots(figsize=(12,12))
sns.heatmap(corrmat,cmap="Pastel2",annot=True)

# %% --------------------------------------------------------------------------
# One Hot Encode Cat Varibles  
# -----------------------------------------------------------------------------
df['cut'] = df['cut'].map({'Ideal':1, 'Premium':2, 'Very Good':3, 'Good':4, 'Fair':5}).astype(int)

dummy = pd.get_dummies(df["color"], prefix="C_", drop_first=True)
df = pd.concat([df, dummy], axis=1)
df = df.drop("color", axis=1)

dummy1 = pd.get_dummies(df["clarity"], prefix="Cl_", drop_first=True)
df = pd.concat([df, dummy1], axis=1)
df = df.drop("clarity", axis=1)


# %% --------------------------------------------------------------------------
# Split Data 
# -----------------------------------------------------------------------------
df_train, df_test = train_test_split(df, test_size=0.2, random_state=rng)

X_train = df_train.drop("price", axis=1)
X_test = df_test.drop("price", axis=1)
y_train = df_train["price"]
y_test = df_test["price"]

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

#%% Models
pipeline_lr=Pipeline([("scalar1",StandardScaler()),
                     ("lr_classifier",LinearRegression())])

pipeline_dt=Pipeline([("scalar2",StandardScaler()),
                     ("dt_classifier",DecisionTreeRegressor())])

pipeline_rf=Pipeline([("scalar3",StandardScaler()),
                     ("rf_classifier",RandomForestRegressor())])


pipeline_kn=Pipeline([("scalar4",StandardScaler()),
                     ("rf_classifier",KNeighborsRegressor())])


pipeline_xgb=Pipeline([("scalar5",StandardScaler()),
                     ("rf_classifier",xg.XGBRegressor())])

# List of all the pipelines
pipelines = [pipeline_lr, pipeline_dt, pipeline_rf, pipeline_kn, pipeline_xgb]

# Dictionary of pipelines and model types for ease of reference
pipe_dict = {0: "LinearRegression", 1: "DecisionTree", 2: "RandomForest",3: "KNeighbors", 4: "XGBRegressor"}

# Fit the pipelines
for pipe in pipelines:
    pipe.fit(X_train, y_train)
    
cv_results_rms = []
for i, model in enumerate(pipelines):
    cv_score = cross_val_score(model, X_train,y_train,scoring="explained_variance", cv=10)
    cv_results_rms.append(cv_score)
    print("%s: %f " % (pipe_dict[i], cv_score.mean()))


# %% --------------------------------------------------------------------------
# Evaluate predicted data 
# -----------------------------------------------------------------------------
y_pred = pipeline_xgb.predict(X_test)
print(f"R2 : {r2_score(y_test, y_pred)}")
print(f"MAE : {mean_absolute_error(y_test, y_pred)}")
print(f"MSE : {mean_squared_error(y_test, y_pred)}")