import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

def save_regression_results_to_csv(results, data, columns, filename):
    """
    Save regression results to a CSV file.

    Parameters:
    results (statsmodels.regression.linear_model.RegressionResultsWrapper): The regression results.
    data (pandas.DataFrame): The DataFrame to save.
    columns (list of str): The columns of the DataFrame to save.
    filename (str): The name of the file to save the results to.
    """
    # Convert the regression results to a DataFrame
    results_df = pd.DataFrame({
        'coefficients': results.params,
        'standard errors': results.bse,
        't-values': results.tvalues,
        'p-values': results.pvalues,
        'conf_int_lower': results.conf_int().iloc[:, 0],
        'conf_int_upper': results.conf_int().iloc[:, 1]
    })

    # Add the specified columns from the original DataFrame
    for column in columns:
        results_df[column] = data[column]

    # Save the DataFrame to a CSV file
    results_df.to_csv(filename)

    # Save the DataFrame to a CSV file
    results_df.to_csv(filename)
# Load and clean data
def load_data(file_name):
    current_dir = os.getcwd()
    os.chdir(current_dir)
    data = pd.read_csv(file_name)
    # Drop rows with NaN values
    data.fillna('0')
    return data

# OLS Regression
def perform_ols(data, independent_vars):
    # Defining the model
    independent_vars = ['ln_export_area'] + [f'colony{i}' for i in range(8)]
    X = data[independent_vars]
    y = data['ln_maddison_pcgdp2000']
    X = sm.add_constant(X) # Adding a constant term

    model1 = sm.OLS(y, X, hasconst=False).fit()
    
    independent_vars = ['ln_export_area','abs_latitude', 'longitude', 'rain_min', 'humid_max', 'low_temp', 'ln_coastline_area'] + [f'colony{i}' for i in range(8)]
    X = data[independent_vars]
    #X = sm.add_constant(X) # Adding a constant term
    model2 = sm.OLS(y, X, hasconst=False).fit()
    
    independent_vars = ['ln_export_area','abs_latitude', 'longitude', 'rain_min', 'humid_max', 'low_temp', 'ln_coastline_area', 
                        'island_dum', 'islam', 'legor_fr', 'region_n'] + [f'colony{i}' for i in range(8)]
    X = data[independent_vars]
    X = sm.add_constant(X) # Adding a constant term
    model3 = sm.OLS(y, X).fit()
    
    independent_vars = ['ln_export_area','abs_latitude', 'longitude', 'rain_min', 'humid_max', 'low_temp', 'ln_coastline_area', 
                        'island_dum', 'islam', 'legor_fr', 'region_n', 'ln_avg_gold_pop', 'ln_avg_oil_pop', 
                        'ln_avg_all_diamonds_pop'] + [f'colony{i}' for i in range(8)]
    X = data[independent_vars]
    X = sm.add_constant(X) # Adding a constant term
    model4 = sm.OLS(y, X).fit()
    
    return model1, model2, model3, model4

# Instrumental Variable Regression - First Stage
def perform_iv_first_stage(data):
    # Independent instrument variables
    instruments = ['atlantic_distance_minimum', 'indian_distance_minimum', 
                   'saharan_distance_minimum', 'red_sea_distance_minimum']
    X = data[instruments]
    y = data['ln_export_area']
    X = sm.add_constant(X) # Adding a constant term

    first_stage = sm.OLS(y, X).fit()
    data['predicted_ln_export_area'] = first_stage.predict(X) # Adding predicted values
    return first_stage, data

# Instrumental Variable Regression - Second Stage
def perform_iv_second_stage(data):
    X = data[['predicted_ln_export_area']]
    y = data['ln_maddison_pcgdp2000']
    X = sm.add_constant(X) # Adding a constant term

    second_stage = sm.OLS(y, X).fit()
    return second_stage

# Plotting scatter plots
def plot_scatter(data, x_var, y_var, title, file_name):
    plt.figure(figsize=(8, 6))
    sns.regplot(x=x_var, y=y_var, data=data)
    plt.title(title)
    plt.savefig(file_name)

# Saving regression summary and summary statistics
def print_summary(model, data, independent_vars):
    summary = model.summary().as_text()
    summary_stats = data[independent_vars].describe().to_string()
    print(summary + '\n\nSummary Statistics:\n' + summary_stats)
        

def run_analysis(entry_filename):
    """Run the analysis with the selected CSV file."""
    filename = entry_filename
    if os.path.exists(filename):
        data = load_data(filename)
        
        independent_vars = ['ln_export_area', 'abs_latitude', 'longitude', 'rain_min', 'humid_max', 'low_temp', 'ln_coastline_area', 
                        'island_dum', 'islam', 'legor_fr', 'region_n', 'ln_avg_gold_pop', 'ln_avg_oil_pop', 
                        'ln_avg_all_diamonds_pop'] + [f'colony{i}' for i in range(8)]

        # Perform OLS regression
        ols_results1, ols_results2, ols_results3, ols_results4 = perform_ols(data, independent_vars)
        print_summary(ols_results1, data, ols_results1.model.exog_names[1:-8])
        print_summary(ols_results2, data, ols_results2.model.exog_names[1:-8])
        print_summary(ols_results3, data, ols_results3.model.exog_names[1:-8])
        print_summary(ols_results4, data, ols_results4.model.exog_names[1:-8])

        # Perform IV regression
        iv_first_stage_results, data = perform_iv_first_stage(data)
        iv_second_stage_results = perform_iv_second_stage(data)
        
        print_summary(iv_first_stage_results, data, ['ln_export_area', 'longitude', 'rain_min', 'humid_max', 'low_temp', 'ln_coastline_area', 'island_dum', 'islam', 'legor_fr', 'region_n', 'ln_avg_gold_pop', 'ln_avg_oil_pop', 'ln_avg_all_diamonds_pop'])
        save_regression_results_to_csv(iv_second_stage_results, data, ['predicted_ln_export_area'], 'IV_Second_Stage_Summary.csv')

        # Plot and save scatter plots
        plot_scatter(data, 'ln_export_area', 'ln_maddison_pcgdp2000', 'ln_export_area vs ln_maddison_pcgdp2000', 'ScatterPlot1.jpg')
        plot_scatter(data, 'ln_pop_dens_1400', 'ln_export_area', 'ln_pop_dens_1400 vs ln_export_area', 'ScatterPlot2.jpg')
        plot_scatter(data, 'ln_export_area', 'ethnic_fractionalization', 'ln_export_area vs Ethnic Fractionalization', 'ScatterPlot3.jpg')


def main():
    # Set up the GUI
    entry_filename = "mergedReplicationData2.csv"
    run_analysis(entry_filename)

if __name__ == "__main__":
    main()
    
