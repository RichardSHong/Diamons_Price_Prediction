# Diamons_Price_Prediction

Title: Diamond Price Prediction

----------------------------------------------------------------------

README CONTENTS
0.1 Group Information
0.2 Use
0.3 Data information
0.4 Folder Contents


----------------------------------------------------------------------
0.1 Group Information

This project was completed by Himanshu Jat, Richard Hong and 
Manoj Venkatachalaiah for DSCI 631, Winter 2021. 


----------------------------------------------------------------------
0.2 Use

Because this project was completed by using data taken from publicly available websites,
it has all of the same free use protections as that data. In other 
words, our work can be used for research but cannot be used for 
things such as profit or identifying individuals.

Anyone can use the code without specifically citing the authors. 


----------------------------------------------------------------------
0.3 
The dataset was acquired from Kaggle. 
Here's the link to the dataset: https://www.kaggle.com/shivam2503/diamonds

----------------------------------------------------------------------
0.4 Folder Contents

dsci631_group_project_group5.ipynb	   Project code

diamonds.csv				   Dataset used in project

----------------------------------------------------------------------




DSCI 631, Applied Machine Learning
Title : Diamond Price Prediction
Team members: Himanshu Jat, Richard Hong, Manoj Venkatachalaiah
----------------------------------------------------------------------------------------------------------------------------------
Project Overview
Our project is divided into two stages:
1) Regression : Where we have predicted the retail price of diamonds by making use of different features in the dataset
2) Classification-then-regression: Where we have classifed each instance into a cluster based on several features and then predicted the retail price of diamonds in each cluster.
Data description
Attributes and description:
price(target attribute): price in US dollars (\$326--\\$18,823)
carat: weight of the diamond (0.2--5.01)
cut: quality of the cut (Fair, Good, Very Good, Premium, Ideal)
color: diamond colour, from D (best) to J (worst)
clarity: a measurement of how clear the diamond is (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))
x: length in mm (0--10.74)
y: width in mm (0--58.9)
z: depth in mm (0--31.8)
depth: total depth percentage = z / mean(x, y) = 2 * z / (x + y) (43--79)
table: width of top of diamond relative to widest point (43--95)
Algorithms used:
1) Linear Regression
2) DecisionTreeRegressor
3) RandomForestRegressor
4) KNeighborsRegressor
5) XGBRegressor
The above algorithms are a mixture of regression and tree based regression algorithms. They were picked because our dataset has a good split between numerical and categorical features and we believe we will get good results using said algorithms.
Linear Regression
What is Linear Regression?
Linear regression is a basic and commonly used type of predictive analysis. The overall idea of regression is to examine two things: (1) does a set of predictor variables do a good job in predicting an outcome (dependent) variable? (2) Which variables in particular are significant predictors of the outcome variable, and in what way do they–indicated by the magnitude and sign of the beta estimates–impact the outcome variable? These regression estimates are used to explain the relationship between one dependent variable and one or more independent variables. The simplest form of the regression equation with one dependent and one independent variable is defined by the formula y = c + b*x, where y = estimated dependent variable score, c = constant, b = regression coefficient, and x = score on the independent variable.
First, the regression might be used to identify the strength of the effect that the independent variable(s) have on a dependent variable. Typical questions are what is the strength of relationship between dose and effect, sales and marketing spending, or age and income.
Second, it can be used to forecast effects or impact of changes. That is, the regression analysis helps us to understand how much the dependent variable changes with a change in one or more independent variables. A typical question is, “how much additional sales income do I get for each additional $1000 spent on marketing?”
Third, regression analysis predicts trends and future values. The regression analysis can be used to get point estimates. A typical question is, “what will the price of gold be in 6 months?”
Decision Tree Regressor
Decision tree builds regression or classification models in the form of a tree structure. It breaks down a dataset into smaller and smaller subsets while at the same time an associated decision tree is incrementally developed. The final result is a tree with decision nodes and leaf nodes. A decision node (e.g., Outlook) has two or more branches (e.g., Sunny, Overcast and Rainy), each representing values for the attribute tested. Leaf node (e.g., Hours Played) represents a decision on the numerical target. The topmost decision node in a tree which corresponds to the best predictor called root node. Decision trees can handle both categorical and numerical data.
Random Forest Regressor
A Random Forest is an ensemble technique capable of performing both regression and classification tasks with the use of multiple decision trees and a technique called Bootstrap and Aggregation, commonly known as bagging. The basic idea behind this is to combine multiple decision trees in determining the final output rather than relying on individual decision trees. Random Forest has multiple decision trees as base learning models. We randomly perform row sampling and feature sampling from the dataset forming sample datasets for every model. This part is called Bootstrap.
We need to approach the Random Forest regression technique like any other machine learning technique
Design a specific question or data and get the source to determine the required data.
Make sure the data is in an accessible format else convert it to the required format.
Specify all noticeable anomalies and missing data points that may be required to achieve the required data.
Create a machine learning model
Set the baseline model that you want to achieve
Train the data machine learning model.
Provide an insight into the model with test data
Now compare the performance metrics of both the test data and the predicted data from the model.
If it doesn’t satisfy your expectations, you can try improving your model accordingly or dating your data or use another data modeling technique.
At this stage you interpret the data you have gained and report accordingly.
K-Neighbors Regressor
k-nearest neighbors It is called a lazy learning algorithm because it doesn’t have a specialized training phase. It doesn’t assume anything about the underlying data because is a non-parametric learning algorithm. Since most of data doesn’t follow a theoretical assumption that’s a useful feature.K-Nearest Neighbors biggest advantage is that the algorithm can make predictions without training, this way new data can be added. It’s biggest disadvantage the difficult for the algorithm to calculate distance with high dimensional data.
Applications
A few examples can be:
Collect financial characteristics to compare people with similar financial features to a database, in order to do Credit Ratings.
Classify the people that can be potential voter to one party or another, in order to predict politics.
Pattern recognition for detect handwriting, image recognition and video recognition.
K-Nearest Neighbors (knn) has a theory you should know about.
First, K-Nearest Neighbors simply calculates the distance of a new data point to all other training data points. It can be any type of distance.
Second, selects the K-Nearest data points, where K can be any integer.
Third, it assigns the data point to the class to which the majority of the K data points belong.
XBG Regressor
The results of the regression problems are continuous or real values. Some commonly used regression algorithms are Linear Regression and Decision Trees. There are several metrics involved in regression like root-mean-squared error (RMSE) and mean-squared-error (MAE). These are some key members for XGBoost models, each plays their important roles.
RMSE: It is the square root of mean squared error (MSE).
MAE: It is an absolute sum of actual and predicted differences, but it lacks mathematically, that’s why it is rarely used, as compared to other metrics.
XGBoost is a powerful approach for building supervised regression models. The validity of this statement can be inferred by knowing about its (XGBoost) objective function and base learners.
The objective function contains loss function and a regularization term. It tells about the difference between actual values and predicted values, i.e how far the model results are from the real values. The most common loss functions in XGBoost for regression problems is reg:linear, and that for binary classification is reg:logistics.
Ensemble learning involves training and combining individual models (known as base learners) to get a single prediction, and XGBoost is one of the ensemble learning methods. XGBoost expects to have the base learners which are uniformly bad at the remainder so that when all the predictions are combined, bad predictions cancels out and better one sums up to form final good predictions.
