# churn_prediction

The Project Customer churn prediction is developed using Python 3.6.
We provided two files regarding the project.
 File named "finalnotebook.ipynb" is a Jupyter notebook.It provides step by step procedure and visualizations at each run of the cell.
 
The entire code is divided into three steps:
   i) Data set merging: We merged the three csv files customer_data.csv ,internet_data.csv, churn_data.csv to form final dataset for further proceesing.
   a) After that we convert the categorical values into numeric values, so our ML algorithms can process the data.
   b)  We also splitted the data into test and train sets.
   
 ii) Data Visualisation: To decide which features of the data to include in our predictive churn model, we have examined the correlation between churn and each customer feature. So we created correlation matrix.
 
 iii) Classification model: 
    We used logistic regression and decision tree classification methods to predict churn.
    Logistic regression is one of the more basic classification algorithms in a data scientist’s toolkit. It is used to predict a category or group based on an observation. Logistic regression is usually used for binary classification (1 or 0, win or lose, true or false). The output of logistic regression is a probability, which will always be a value between 0 and 1. 
    
  A decision tree is a supervised learning method that makes a prediction by learning simple decision rules from the explanatory variables. 
  Decision trees have the following advantages:

  Trees can be visualised, which makes them easy to interpret
  They can handle numerical and categorical data
  We can easily validate the model using statistical tests


