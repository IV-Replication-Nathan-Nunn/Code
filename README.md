# Project_ECON_470_IV
ECON 470 IV group project
Project Title: Instrumental Variable Analysis for Economic Impact Study

Description:

This
 project aims to investigate the impact of historical slave trade activities on contemporary economic performance. Utilizing an Instrumental Variable (IV) analysis, we focus on leveraging geographical distance as an instrument to assess the causal relationship
 between the number of slaves sent from African regions and the economic performance indicators of those regions today. The analysis incorporates a double lasso approach for variable selection, enhancing the robustness and validity of our causal inferences.


Objectives:
To establish a causal link between historical slave trade activities and current economic outcomes.
To utilize geographical distance as an instrumental variable in addressing endogeneity issues.
To apply a double lasso technique for effective variable selection in the context of IV analysis.

Methodology:
Instrumental Variable Analysis: We employ distance as an instrumental variable for the number of slaves sent, aiming to isolate the exogenous variations influencing economic performance.
Double Lasso Selection: This technique is applied to select relevant control variables and mitigate potential bias in the causal estimation.

Dependencies:
Python 3.x: The primary programming language used for analysis.
Libraries: pandas, numpy, statsmodels, sklearn, matplotlib, seaborn.
pandas and numpy for data manipulation,
statsmodels for statistical modeling,
sklearn for machine learning applications, particularly the Lasso regression,
matplotlib and seaborn for data visualization.

Installation:

Ensure
 Python 3.x is installed on your machine. Clone this repository to your local machine. Install required dependencies using piprn
 matplotlib seaborn 


Data:

The analysis utilizes historical data on the slave trade, including:

Number of slaves sent from African regions,
Economic performance indicators of these regions in the contemporary period,
Geographical data for instrumental variable analysis.

How to Run the Analysis:
Prepare the Environment: Ensure all dependencies are installed.
Load the Data: Use pandas to load the dataset into your analysis environment.
Run the IV Analysis: Execute the IV analysis script, which performs the double lasso selection followed by the instrumental variable estimation.
Review Results: Analyze the output for causal inferences and insights into the impact of the slave trade on economic performance.
