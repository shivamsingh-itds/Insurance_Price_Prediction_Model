# Importing manipulation libaries 
import numpy as np
import pandas as pd

# Importing Visualization libaries
import seaborn as sns
import matplotlib.pyplot as plt

# Filtering out warnings
import warnings
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(level = logging.INFO,
                    filename = 'model.log',
                    filemode = 'w',
                    format = '%(asctime)s - %(levelname)s - %(message)s',
                    force = True)
                    
# 3. Import OrderedDict()
from collections import OrderedDict

#Import scikrit learn libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,MinMaxScaler

# Step 1 : Data Ingestion 
def data_ingestion():
    return pd.read_csv(r'C:\15Days15Project\InsurancePricePredictionModel\Data\insurance.csv')


# step 2 : Data Exploration
# checking Descriptive stats
# (numerical stats , categorical stas , data info )
def descriptive_stats():
    
    # segregate numerical columns and categorical columns
    numerical_col = df.select_dtypes(exclude = "object").columns
    categorical_col = df.select_dtypes(include = "object").columns

    # Checking Stats: Numerical Columns
    # Checking Stats: Numerical Columns
    num_stats = []
    cat_stats = []
    data_info = []

    for i in numerical_col:
        Q1 = df[i].quantile(0.25)
        Q3 = df[i].quantile(0.75)
        IQR = Q3 - Q1
        LWR = Q1 - 1.5 * IQR
        UWR = Q3 + 1.5 * IQR

        outlier_count = len(df[(df[i] < LWR) | (df[i] > UWR)])
        outlier_percentage = (outlier_count / len(df)) * 100

        numericalstats = OrderedDict({
            "Feature": i,
            "Mean": df[i].mean(),
            "Median": df[i].median(),
            "Minimum": df[i].min(),
            "Maximum": df[i].max(),
            "Q1": Q1,
            "Q3": Q3,
            "IQR": IQR,
            "LWR": LWR,
            "UWR": UWR,
            "Outlier Count": outlier_count,
            "Outlier Percentage": outlier_percentage,
            "Standard Deviation": df[i].std(),
            "Variance": df[i].var(),
            "Skewness": df[i].skew(),
            "Kurtosis": df[i].kurtosis()
        })
        num_stats.append(numericalstats)

    # Create DataFrame AFTER the loop completes
    numerical_stats_report = pd.DataFrame(num_stats)

    # Checking for Categorical columns
    for i in categorical_col:
        # Handle case where mode() might return empty Series
        mode_val = df[i].mode()
        mode_val = mode_val[0] if not mode_val.empty else None

        cat_stats1 = OrderedDict({
            "Feature": i,
            "Unique Values": df[i].nunique(),
            "Value Counts": df[i].value_counts().to_dict(),
            "Mode": mode_val
        })
        cat_stats.append(cat_stats1)

    categorical_stats_report = pd.DataFrame(cat_stats)

    # Checking dataset information
    for i in df.columns:
        data_info1 = OrderedDict({
            "Feature": i,
            "Data Type": str(df[i].dtype),
            "Missing_Values": df[i].isnull().sum(),
            "Unique_Values": df[i].nunique(),
            "Value_Counts": df[i].value_counts().to_dict()
        })
        data_info.append(data_info1)

    data_info_report = pd.DataFrame(data_info)

    return numerical_stats_report, categorical_stats_report, data_info_report




# step 3 : Data Preprocessing
def data_preprocessing(df):
    X = df.drop(columns = 'charges',axis = 1)
    y = df['charges']

  # Split the Dataset into train and test
    X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                  test_size = 0.3,
                                              random_state = 10)
    for i in df.select_dtypes(include = "object").columns:
        le = LabelEncoder()
        X_train[i] = le.fit_transform(X_train[i])  # Seen Data
        X_test[i] = le.transform(X_test[i])        # Unseen Data
     # Using Normalization Technique
      
    sc = MinMaxScaler()
    X_train = sc.fit_transform(X_train)          # Seen Data
    X_test = sc.transform(X_test)                # Unseen Data
    return X_train,X_test,y_train,y_test



#step 4 : Model Building
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
def data_model(df):
    
    model_comparison = []

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(),
        "Random Forest Regressor": RandomForestRegressor()
    }

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2score = r2_score(y_test, y_pred)

        model_comparison.append({
            "Model Name": model_name,
            "R2 Score": r2score
        })

    model_comparison = pd.DataFrame(model_comparison)
    return model_comparison


# Function calling
df = data_ingestion()
numerical_stats_report,categorical_stats_report,data_info_report = descriptive_stats()
X_train ,X_test , y_train ,y_test = data_preprocessing(df)
model_comparsion = data_model(df)

# Testing
print(df)
print(numerical_stats_report)
print(categorical_stats_report)
print(data_info_report)
print(X_train)
print(model_comparsion)
