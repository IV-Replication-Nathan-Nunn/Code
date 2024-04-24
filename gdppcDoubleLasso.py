import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
def doubleLasso(df, target_column, independent_vars):
    # Extracting the independent variables (X) and the dependent variable (y) from the DataFrame
    X = df[independent_vars].values
    y = df[target_column].values

    # Scaling features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # First Lasso: Variable selection
    lasso1 = LassoCV(cv=10, random_state=42).fit(X_train, y_train)
    selected_features = np.where(lasso1.coef_ != 0)[0]

    # Extracting the selected features for the second Lasso
    X_train_selected = X_train[:, selected_features]
    X_test_selected = X_test[:, selected_features]

    # Second Lasso: Estimation using selected features
    lasso2 = LassoCV(cv=10, random_state=42).fit(X_train_selected, y_train)

    # Printing selected features and their coefficients
    print(f"Selected Features: {independent_vars[selected_features]}")
    print(f"Coefficients: {lasso2.coef_}")

    # Optionally, using statsmodels for detailed statistics
    X_train_sm = sm.add_constant(X_train_selected)  # Adding a constant for statsmodels
    model_sm = sm.OLS(y_train, X_train_sm).fit()

    # Printing model summary from statsmodels
    print(model_sm.summary())

    # Evaluation Metrics
    y_pred_train = lasso2.predict(X_train_selected)
    y_pred_test = lasso2.predict(X_test_selected)

    print(f"Training R^2: {r2_score(y_train, y_pred_train)}")
    print(f"Test R^2: {r2_score(y_test, y_pred_test)}")
    print(f"Training MSE: {mean_squared_error(y_train, y_pred_train)}")
    print(f"Test MSE: {mean_squared_error(y_test, y_pred_test)}")
    
data = 'C:/vscode2/ECON470/IV_replication_CooperMuery-main/mergedReplicationData2.csv'
#Target: 2021 GDP
target_column = 'YR2021'
independent_vars = np.array(['ln_export_area', 'longitude', 'rain_min', 'humid_max', 'low_temp', 'ln_coastline_area', 
                            'island_dum', 'islam', 'legor_fr', 'region_n', 'ln_avg_gold_pop', 'ln_avg_oil_pop', 
                            'ln_avg_all_diamonds_pop'] + [f'colony{i}' for i in range(8)])
df = pd.read_csv(data, usecols=['YR2021', 'ln_export_area', 'longitude', 'rain_min', 'humid_max', 'low_temp', 'ln_coastline_area', 
                            'island_dum', 'islam', 'legor_fr', 'region_n', 'ln_avg_gold_pop', 'ln_avg_oil_pop', 
                            'ln_avg_all_diamonds_pop'] + [f'colony{i}' for i in range(8)])
doubleLasso(df,target_column,independent_vars)

#Target: 2002 GDP
target_column = 'YR2002'
df = pd.read_csv(data, usecols=['YR2002', 'ln_export_area', 'longitude', 'rain_min', 'humid_max', 'low_temp', 'ln_coastline_area', 
                            'island_dum', 'islam', 'legor_fr', 'region_n', 'ln_avg_gold_pop', 'ln_avg_oil_pop', 
                            'ln_avg_all_diamonds_pop'] + [f'colony{i}' for i in range(8)])
doubleLasso(df,target_column,independent_vars)

#Target: 2003 GDP
target_column = 'YR2003'
df = pd.read_csv(data, usecols=['YR2003', 'ln_export_area', 'longitude', 'rain_min', 'humid_max', 'low_temp', 'ln_coastline_area', 
                            'island_dum', 'islam', 'legor_fr', 'region_n', 'ln_avg_gold_pop', 'ln_avg_oil_pop', 
                            'ln_avg_all_diamonds_pop'] + [f'colony{i}' for i in range(8)])
doubleLasso(df,target_column,independent_vars)

#Target: 2004 GDP
target_column = 'YR2004'
df = pd.read_csv(data, usecols=['YR2004', 'ln_export_area', 'longitude', 'rain_min', 'humid_max', 'low_temp', 'ln_coastline_area', 
                            'island_dum', 'islam', 'legor_fr', 'region_n', 'ln_avg_gold_pop', 'ln_avg_oil_pop', 
                            'ln_avg_all_diamonds_pop'] + [f'colony{i}' for i in range(8)])
doubleLasso(df,target_column,independent_vars)

#Target: 2005 GDP
target_column = 'YR2005'
df = pd.read_csv(data, usecols=['YR2005', 'ln_export_area', 'longitude', 'rain_min', 'humid_max', 'low_temp', 'ln_coastline_area', 
                            'island_dum', 'islam', 'legor_fr', 'region_n', 'ln_avg_gold_pop', 'ln_avg_oil_pop', 
                            'ln_avg_all_diamonds_pop'] + [f'colony{i}' for i in range(8)])
doubleLasso(df,target_column,independent_vars)

#Target: 2006 GDP
target_column = 'YR2006'
df = pd.read_csv(data, usecols=['YR2006', 'ln_export_area', 'longitude', 'rain_min', 'humid_max', 'low_temp', 'ln_coastline_area', 
                            'island_dum', 'islam', 'legor_fr', 'region_n', 'ln_avg_gold_pop', 'ln_avg_oil_pop', 
                            'ln_avg_all_diamonds_pop'] + [f'colony{i}' for i in range(8)])
doubleLasso(df,target_column,independent_vars)