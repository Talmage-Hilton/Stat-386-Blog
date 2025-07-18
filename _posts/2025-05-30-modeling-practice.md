---
layout: post
title:  "Bilingual Modeling: A Code-First Guide to 13 Statistical Models"
date: 2025-05-30
description: A fun data science project I did in both Python and R
image: "/assets/img/R_python.avif"
display_image: false  # change this to true to display the image below the banner 
---
<p class="intro"><span class="dropcap">T</span>his post will be a review of a fun data science project I did to practice statistical modeling in both Python and R.</p>
<p class="intro">Cover image source: <a href="https://www.datacamp.com/blog/python-vs-r-for-data-science-whats-the-difference">datacamp</a></p>


## Introduction

About two years ago I began learning Python, but I haven't used it very much since then. Being a statistics student, we use R almost exclusively in my classes. R is a wonderful tool and it's the first programming language I learned, so it will always hold a special place in my heart, but I know the data science industry primarily relies on Python. So, as an aspiring data scientist, I wanted to relearn Python and do it in the best way I know how - by doing some statistical modeling!

This post will dive into the data I modeled, how to code the models up in both Python and R, and the results from each. My [last post]("https://talmage-hilton.github.io/Stat-386-Blog/blog/model-explanations/") dove deeply into all the nitty-gritty details of these models. I would highly recommend reading it prior to jumping into this post, especially if you do not already have a strong understanding of how these models function. After that, please come here again, sit back, grab a refreshing beverage, and enjoy reading about my project!

Some of the models in this post, such as linear regression, are deterministic. This means that there is no randomness and they follow a fixed procedure each time. Linear regression, for example, finds the beta coefficients in the same way each time (using linear algebra), meaning that the model will always be the same assuming the inputs are the same. Random forests, however, use only a random subset of the data and predictors when fitting each tree, thus leading to slightly different results with each run. The nature of the “Results” section for each model below will depend on whether or not the model is deterministic. If it is deterministic, both Python and R will yield the exact same results. If it is random, then I will explore the differences in the results between the two models. For the linear models, I will include the summary output from either Python, R, or both. For the machine learning models, since such a summary output is not available, I will just compare the in-sample vs. out-of-sample [RMSE](“https://coralogix.com/ai-blog/root-mean-square-error-rmse-the-cornerstone-for-evaluating-regression-models/”) (for regression) or [Accuracy](“https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall”) & [AUC](“https://www.geeksforgeeks.org/machine-learning/auc-roc-curve/”) (for classification).

I want to make it very clear from the beginning that this post is not a comparison or competition between Python and R to see which coding language is better. There are so many situations where R is the easier option (such as statistical tests and EDA), and others where Python is better (such as deep learning and web development). Wherever results differ between the two languages, I include both. If I include results from only Python or only R, it is not because I think it’s better than the other language; I just want to give you, the reader, plenty of exposure to both languages. This post is simply to show you how versatile both coding languages are and give you plenty of resources to be able to begin statistical modeling on your own in whatever language you please.

A final note before we hop into the analysis: in an ideal world, I would also compare the computational time between Python and R for each model. However, both datasets were small enough that each model ran virtually instantly in both languages. The primary one that did not was BART, but I could only run it in R (for reasons I will explain later), so there would be no comparison to do anyway.


## The Data

I used two different datasets for this project, both found on [Kaggle]("https://www.kaggle.com/"). The first dataset measures students' exam scores based on various predictor variables. This dataset has a continuous response variable (exam_score) and 1000 observations. The continuous predictor variables are age, study_hours_per_day, social_media_hours, netflix_hours, part_time_job, attendance_percentage, sleep_hours, exercise_frequency, mental_health_rating, extracurricular_participation. The categorical predictor variables are gender, diet_quality, and internet_quality. To finalize the data for analysis, I used [one-hot encoding](“https://www.educative.io/blog/one-hot-encoding”) on the categorical variables — gender, diet_quality, and internet_quality — transforming them into binary indicator columns. This led to the following columns being created: gender_Male and gender_Other (gender_Female is the baseline), diet_quality_Good and diet_quality_Poor (diet_quality_Fair is the baseline), internet_quality_Good and internet_quality_Poor (internet_quality_Average is the baseline). I synthetically added new data to get up to 10,000 rows just to give the models some more data from which to learn.

Here are a few EDA plots of this dataset so you can start getting a little bit of intuition about the data we'll be working with:

<figure>
	<img src="{{site.url}}/{{site.baseurl}}/assets/img/modeling_eda_histograms.png" alt="" style="width: 700px; height=auto;"> 
	<figcaption>Histograms of numeric variables</figcaption>
    <figcaption>Image Source: <a href="https://www.r-project.org/about.html">R</a></figcaption>
</figure>

<figure>
	<img src="{{site.url}}/{{site.baseurl}}/assets/img/modeling_eda_scatterplots.png" alt="" style="width: 700px; height=auto;"> 
	<figcaption>Scatterplots of numeric variable relationships</figcaption>
    <figcaption>Image Source: <a href="https://www.r-project.org/about.html">R</a></figcaption>
</figure>

<figure>
	<img src="{{site.url}}/{{site.baseurl}}/assets/img/modeling_eda_correlation_heatmap.png" alt="" style="width: 700px; height=auto;"> 
	<figcaption>Correlation heatmap of numeric variables</figcaption>
    <figcaption>Image Source: <a href="https://www.r-project.org/about.html">R</a></figcaption>
</figure>

The second dataset measures whether or not a social media user clicked on an advertisement. Clearly this is a binary response variable. The response variable is Purchased, the continuous predictors are Age and EstimatedSalary, and the categorical predictor is Gender. Just as before, I used one-hot encoding on the Gender variable. Here is a bit of EDA for that dataset as well:

<figure>
	<img src="{{site.url}}/{{site.baseurl}}/assets/img/modeling_eda_scatterplot2.jpeg" alt="" style="width: 700px; height=auto;"> 
	<figcaption>Scatterplot of numeric variables</figcaption>
    <figcaption>Image Source: <a href="https://www.r-project.org/about.html">R</a></figcaption>
</figure>

<figure>
	<img src="{{site.url}}/{{site.baseurl}}/assets/img/modeling_eda_bar_chart.jpeg" alt="" style="width: 700px; height=auto;"> 
	<figcaption>Bar Charts of categorical variables</figcaption>
    <figcaption>Image Source: <a href="https://www.r-project.org/about.html">R</a></figcaption>
</figure>

I used the advertisements dataset for Logistic Regression, as it requires a binary response. I used the exam scores dataset for all the other linear models. Finally, I used both datasets for all the machine learning models to display their flexibility.



## The Models

I used six linear models and seven machine learning models. The linear models are OLS (Linear Regression), LASSO, Polynomial Regression, Nautral Splines, GAM, and Logistic Regression. The machine learning models I used are K Nearest Neighbors, Support Vector Machines, Decision Tree (CART), Random Forest, Boosting, BART, and Neural Networks.


## Linear Models


### Linear Regression

Linear Regression is a deterministic model, so the results are identical between R and Python. As long as you are implementing the model correctly, it will always lead to the same beta coefficients, critical values, p-values, etc.

#### Python Code

I will supply you with two Python options. They lead to the same results, but just use slightly different functions and have slightly different syntax.

{%- highlight python -%}
# Python Option 1

import statsmodels.formula.api as smf

# Fit the linear regression model using formula syntax
model = smf.ols(
    formula='exam_score ~ age + study_hours_per_day + social_media_hours + netflix_hours + part_time_job + attendance_percentage + sleep_hours + exercise_frequency + mental_health_rating + extracurricular_participation + gender_Male + gender_Other + diet_quality_Good + diet_quality_Poor + internet_quality_Good + internet_quality_Poor',
    data=df
).fit()

# Print the model summary
print(model.summary())
{%- endhighlight -%}

{%- highlight python -%}
# Python Option 2

import statsmodels.api as sm

# Define X and y
X = df[['age', 'study_hours_per_day', 'social_media_hours', 'netflix_hours', 'part_time_job',
        'attendance_percentage', 'sleep_hours', 'exercise_frequency', 'mental_health_rating',
        'extracurricular_participation', 'gender_Male', 'gender_Other', 'diet_quality_Good',
        'diet_quality_Poor', 'internet_quality_Good', 'internet_quality_Poor']]  # predictors
X = sm.add_constant(X)  # adds the intercept term
y = df['exam_score']     # response

# Fit the model
model = sm.OLS(y, X).fit()

# Summary output
print(model.summary())
{%- endhighlight -%}



#### R Code



#### Results

The following table is the R summary output from the regression model:

*insert R summary output including top part with R^2*




### LASSO

#### Python Code

{%- highlight python -%}
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Define predictors and response
X = df[['age', 'study_hours_per_day', 'social_media_hours', 'netflix_hours',
        'part_time_job', 'attendance_percentage', 'sleep_hours',
        'exercise_frequency', 'mental_health_rating', 'extracurricular_participation',
        'gender_Male', 'gender_Other', 'diet_quality_Good', 'diet_quality_Poor',
        'internet_quality_Good', 'internet_quality_Poor']]
y = df['exam_score']

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Fit Lasso
lasso = Lasso(alpha=0.1)  # regularization parameter of 0.1

# Fit the model
lasso.fit(X_train_scaled, y_train)

# Create DataFrame of coefficients
coef_df = pd.DataFrame({
    'Variable': X.columns,
    'Coefficient': lasso.coef_
})

# Display
print(coef_df)
{%- endhighlight -%}

#### R Code

#### Results


### Polynomial Regression

#### Python Code

I will supply you with two Python options again.

{%- highlight python -%}
# Python Option 1

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm

# Define your predictors and response
X = df[['age', 'study_hours_per_day', 'social_media_hours', 'netflix_hours',
        'part_time_job', 'attendance_percentage', 'sleep_hours',
        'exercise_frequency', 'mental_health_rating', 'extracurricular_participation',
        'gender_Male', 'gender_Other', 'diet_quality_Good', 'diet_quality_Poor',
        'internet_quality_Good', 'internet_quality_Poor']]
y = df['exam_score']

# Define the transformer
poly_transformer = ColumnTransformer(transformers=[
    ('age^2', PolynomialFeatures(degree=2, include_bias=False), ['age']),
    ('study^3', PolynomialFeatures(degree=3, include_bias=False), ['study_hours_per_day'])
], remainder='passthrough')

# Apply the transformation
X_poly = poly_transformer.fit_transform(X)

# Get feature names for the transformed columns
# (this part is optional but makes the summary more readable)
age_features = poly_transformer.named_transformers_['age^2'].get_feature_names_out(['age'])
study_features = poly_transformer.named_transformers_['study^3'].get_feature_names_out(['study_hours_per_day'])
original_features = X.columns.drop(['age', 'study_hours_per_day'])
feature_names = list(age_features) + list(study_features) + list(original_features)

# Create DataFrame and reset index
X_poly_df = pd.DataFrame(X_poly, columns=feature_names)
X_poly_df = sm.add_constant(X_poly_df)  # add intercept
X_poly_df = X_poly_df.reset_index(drop=True)
y = y.reset_index(drop=True)

# Fit the model
model = sm.OLS(y, X_poly_df).fit()

# Summary
print(model.summary())
{%- endhighlight -%}

{%- highlight python -%}
# Python Option 2

import statsmodels.api as sm

# Manually create polynomial terms
df['age^2'] = df['age'] ** 2
df['study_hours^2'] = df['study_hours_per_day'] ** 2
df['study_hours^3'] = df['study_hours_per_day'] ** 3

# Prepare predictors (including polynomial terms)
X = df[['age', 'age^2', 'study_hours_per_day', 'study_hours^2', 
        'study_hours^3', 'social_media_hours', 'netflix_hours', 'part_time_job',
        'attendance_percentage', 'sleep_hours', 'exercise_frequency',
        'mental_health_rating', 'extracurricular_participation',
        'gender_Male', 'gender_Other', 'diet_quality_Good', 'diet_quality_Poor',
        'internet_quality_Good', 'internet_quality_Poor']]

# Add constant intercept
X = sm.add_constant(X)

# Response variable
y = df['exam_score']

# Fit the model
model = sm.OLS(y, X).fit()

# Print summary (like R's summary())
print(model.summary())
{%- endhighlight -%}

#### R Code

#### Results


### Nautral Splines

#### Python Code

{%- highlight python -%}
from patsy import dmatrix
import statsmodels.api as sm
import pandas as pd

# Create spline basis for 'age'
age_spline = dmatrix("cr(age, df=4)", data=df, return_type='dataframe')  # cr does natural spline, bs does B-spline

# Create spline basis for 'study_hours_per_day'
study_spline = dmatrix("cr(study_hours_per_day, df=3)", data=df, return_type='dataframe')

# Combine with other predictors
X = df[['social_media_hours', 'netflix_hours', 'part_time_job', 'attendance_percentage',
        'sleep_hours', 'exercise_frequency', 'mental_health_rating',
        'extracurricular_participation', 'gender_Male', 'gender_Other',
        'diet_quality_Good', 'diet_quality_Poor', 'internet_quality_Good', 'internet_quality_Poor']]  # your other predictors
X = pd.concat([age_spline, study_spline, X], axis=1)

# Add intercept
X = sm.add_constant(X)

# Response variable
y = df['exam_score']

# Fit OLS model
model = sm.OLS(y, X).fit()
print(model.summary())
{%- endhighlight -%}

#### R Code

#### Results


### GAM

#### Python Code

{%- highlight python -%}
from statsmodels.gam.api import GLMGam, BSplines
from statsmodels.genmod.families import Gaussian
import numpy as np

# Define spline bases for 'age' and 'study_hours_per_day'
from patsy import dmatrix
age_spline = dmatrix("bs(age, df=4, include_intercept=False)", data=df)  # define B-spline basis
study_spline = dmatrix("bs(study_hours_per_day, df=3, include_intercept=False)", data=df)  # define B-spline basis

# Create full feature matrix
X_splines = np.hstack([age_spline, study_spline])
X_other = df[['social_media_hours', 'netflix_hours', 'part_time_job',
              'attendance_percentage', 'sleep_hours', 'exercise_frequency',
              'mental_health_rating', 'extracurricular_participation',
              'gender_Male', 'gender_Other', 'diet_quality_Good',
              'diet_quality_Poor', 'internet_quality_Good', 'internet_quality_Poor']].to_numpy()

# Combine everything
X = np.hstack([X_splines, X_other])
y = df['exam_score'].to_numpy()

# Fit the GAM model
model = sm.GLM(y, X, family=Gaussian()).fit()
print(model.summary())
{%- endhighlight -%}

#### R Code

#### Results


### Logistic Regression

#### Python Code

{%- highlight python -%}
import statsmodels.api as sm

# X: DataFrame of predictors (make sure to add a constant)
X = sm.add_constant(ads[['Gender', 'Age', 'EstimatedSalary']])
y = ads['Purchased']

# Fit logistic regression
model = sm.Logit(y, X).fit()

# Print summary
print(model.summary())
{%- endhighlight -%}

#### R Code

#### Results


## Machine Learning Models

For each of the Machine Learning Models, I will supply the code for the regression setting (continuous response variable) and the classification setting (binary response variable).


### K Nearest Neighbors

#### Python Code

{%- highlight python -%}
# Continuous Response

from sklearn.neighbors import KNeighborsRegressor  # For regression (predicting continuous y)

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd

X = df[['age', 'study_hours_per_day', 'social_media_hours', 'netflix_hours',
        'part_time_job', 'attendance_percentage', 'sleep_hours',
        'exercise_frequency', 'mental_health_rating', 'extracurricular_participation',
        'gender_Male', 'gender_Other', 'diet_quality_Good', 'diet_quality_Poor',
        'internet_quality_Good', 'internet_quality_Poor']]  # make sure these are numeric
y = df['exam_score']

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

# Fit KNN model
knn = KNeighborsRegressor(n_neighbors=5)  # k=5 neighbors, adjust as you want
knn.fit(X_train, y_train)

# Predictions on training and testing sets
y_train_pred = knn.predict(X_train)
y_test_pred = knn.predict(X_test)

# Calculate RMSE for training set (in-sample)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))

# Calculate RMSE for testing set (out-of-sample)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(f'In-sample RMSE: {rmse_train:.4f}')
print(f'Out-of-sample RMSE: {rmse_test:.4f}')
{%- endhighlight -%}

{%- highlight python -%}
# Binary Response

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import pandas as pd
import numpy as np

# Define features and target
X = ads[['Gender', 'Age', 'EstimatedSalary']]  # Ensure all features are numeric
y = ads['Purchased']

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

# Fit KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predict class labels
y_train_pred = knn.predict(X_train)
y_test_pred = knn.predict(X_test)

# Predict probabilities for AUC
y_train_proba = knn.predict_proba(X_train)[:, 1]
y_test_proba = knn.predict_proba(X_test)[:, 1]

# Accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# AUC
train_auc = roc_auc_score(y_train, y_train_proba)
test_auc = roc_auc_score(y_test, y_test_proba)

# Output results
print(f'In-sample Accuracy: {train_accuracy:.4f}')
print(f'In-sample AUC: {train_auc:.4f}')
print(f'Out-of-sample Accuracy: {test_accuracy:.4f}')
print(f'Out-of-sample AUC: {test_auc:.4f}')

# Optional: Detailed classification metrics
print("\nClassification Report (Test Data):")
print(classification_report(y_test, y_test_pred))  # recall is sensitivity (true positive rate)
{%- endhighlight -%}



#### R Code

#### Results


### Support Vector Machines

#### Python Code

{%- highlight python -%}
# Continuous Response

from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

X = df[['age', 'study_hours_per_day', 'social_media_hours', 'netflix_hours',
        'part_time_job', 'attendance_percentage', 'sleep_hours',
        'exercise_frequency', 'mental_health_rating', 'extracurricular_participation',
        'gender_Male', 'gender_Other', 'diet_quality_Good', 'diet_quality_Poor',
        'internet_quality_Good', 'internet_quality_Poor']]  # make sure these are numeric
y = df['exam_score']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

# Train SVR model
svr_model = SVR(kernel='rbf')  # You can also try 'linear' or 'poly'
svr_model.fit(X_train, y_train)

# Predict
y_pred_train = svr_model.predict(X_train)
y_pred_test = svr_model.predict(X_test)

# Evaluate RMSE for training set (in-sample)
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))

# Evaluate RMSE for testing set (out-of-sample)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f"In-sample RMSE: {rmse_train:.4f}")
print(f"Out-of-sample RMSE: {rmse_test:.4f}")
{%- endhighlight -%}

{%- highlight python -%}
# Binary Response

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Split features and target
X = ads[['Gender', 'Age', 'EstimatedSalary']]  # Ensure all features are numeric
y = ads['Purchased']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

# Train SVM (enable probability estimates)
svm_classifier = SVC(kernel='rbf', probability=True)
svm_classifier.fit(X_train, y_train)

# Predictions
y_train_pred = svm_classifier.predict(X_train)
y_test_pred = svm_classifier.predict(X_test)

# Probabilities (needed for AUC)
y_train_proba = svm_classifier.predict_proba(X_train)[:, 1]
y_test_proba = svm_classifier.predict_proba(X_test)[:, 1]

# Accuracy
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

# AUC
train_auc = roc_auc_score(y_train, y_train_proba)
test_auc = roc_auc_score(y_test, y_test_proba)

# Results
print(f"In-sample Accuracy: {train_acc:.4f}")
print(f"In-sample AUC: {train_auc:.4f}")
print(f"Out-of-sample Accuracy: {test_acc:.4f}")
print(f"Out-of-sample AUC: {test_auc:.4f}")

# Confusion Matrix and Classification Report
print("\nClassification Report - Testing Data:")
print(classification_report(y_test, y_test_pred))
{%- endhighlight -%}

#### R Code

#### Results


### Decision Tree (CART)

#### Python Code

{%- highlight python -%}
# Continuous Response

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Features and target for regression
X = df[['age', 'study_hours_per_day', 'social_media_hours', 'netflix_hours',
        'part_time_job', 'attendance_percentage', 'sleep_hours',
        'exercise_frequency', 'mental_health_rating', 'extracurricular_participation',
        'gender_Male', 'gender_Other', 'diet_quality_Good', 'diet_quality_Poor',
        'internet_quality_Good', 'internet_quality_Poor']]  # make sure these are numeric
y = df['exam_score']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

# Fit CART model (regression tree)
reg_tree = DecisionTreeRegressor(max_depth=3, random_state=123)  # change depth here
reg_tree.fit(X_train, y_train)

# Predict
y_train_pred = reg_tree.predict(X_train)
y_test_pred = reg_tree.predict(X_test)

# Evaluation
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(f"Regression Tree - In-sample RMSE: {rmse_train:.4f}")
print(f"Regression Tree - Out-of-sample RMSE: {rmse_test:.4f}")


# Plot decision tree

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))
plot_tree(reg_tree, feature_names=X.columns, filled=True)
plt.show()
{%- endhighlight -%}

{%- highlight python -%}
# Binary Response

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# Features and target for classification
X = ads[['Gender', 'Age', 'EstimatedSalary']]  # Ensure all features are numeric
y = ads['Purchased']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

# Fit CART model (classification tree)
clf_tree = DecisionTreeClassifier(max_depth=3, random_state=123)  # change depth here
clf_tree.fit(X_train, y_train)

# Predict
y_train_pred = clf_tree.predict(X_train)
y_test_pred = clf_tree.predict(X_test)
y_train_proba = clf_tree.predict_proba(X_train)[:, 1]
y_test_proba = clf_tree.predict_proba(X_test)[:, 1]

# Evaluation
acc_train = accuracy_score(y_train, y_train_pred)
acc_test = accuracy_score(y_test, y_test_pred)
auc_train = roc_auc_score(y_train, y_train_proba)
auc_test = roc_auc_score(y_test, y_test_proba)

print(f"In-sample Accuracy: {acc_train:.4f}")
print(f"In-sample AUC: {auc_train:.4f}")
print(f"Out-of-sample Accuracy: {acc_test:.4f}")
print(f"Out-of-sample AUC: {auc_test:.4f}")


# Plot decision tree

from sklearn.tree import DecisionTreeRegressor, plot_tree

plt.figure(figsize=(20, 10))
plot_tree(clf_tree, feature_names=X.columns, filled=True)
plt.show()
{%- endhighlight -%}

#### R Code

#### Results


### Random Forest

#### Python Code

{%- highlight python -%}
# Continuous Response

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# X and y should be numeric
X = df[['age', 'study_hours_per_day', 'social_media_hours', 'netflix_hours',
        'part_time_job', 'attendance_percentage', 'sleep_hours',
        'exercise_frequency', 'mental_health_rating', 'extracurricular_participation',
        'gender_Male', 'gender_Other', 'diet_quality_Good', 'diet_quality_Poor',
        'internet_quality_Good', 'internet_quality_Poor']]  # make sure these are numeric
y = df['exam_score']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit random forest regressor
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)

# Predictions
y_train_pred = rf_reg.predict(X_train)
y_test_pred = rf_reg.predict(X_test)

# Metrics
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

print("In-sample RMSE:", rmse_train)
print("Out-of-sample RMSE:", rmse_test)
{%- endhighlight -%}

{%- highlight python -%}
# Binary Response

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# X and y should be prepared ahead of time
X = ads[['Gender', 'Age', 'EstimatedSalary']]  # Ensure all features are numeric
y = ads['Purchased']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit random forest classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

# Predictions
y_train_pred = rf_clf.predict(X_train)
y_test_pred = rf_clf.predict(X_test)

# Probabilities for AUC
y_train_proba = rf_clf.predict_proba(X_train)[:, 1]
y_test_proba = rf_clf.predict_proba(X_test)[:, 1]

# Metrics
print("In-sample Accuracy:", accuracy_score(y_train, y_train_pred))
print("In-sample AUC:", roc_auc_score(y_train, y_train_proba))
print("Out-of-sample Accuracy:", accuracy_score(y_test, y_test_pred))
print("Out-of-sample AUC:", roc_auc_score(y_test, y_test_proba))
print("\nClassification Report (OOS):")
print(classification_report(y_test, y_test_pred))
{%- endhighlight -%}

#### R Code

#### Results


### Boosting

#### Python Code

{%- highlight python -%}
# Continuous Response

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# X and y should be numeric
X = df[['age', 'study_hours_per_day', 'social_media_hours', 'netflix_hours',
        'part_time_job', 'attendance_percentage', 'sleep_hours',
        'exercise_frequency', 'mental_health_rating', 'extracurricular_participation',
        'gender_Male', 'gender_Other', 'diet_quality_Good', 'diet_quality_Poor',
        'internet_quality_Good', 'internet_quality_Poor']]  # make sure these are numeric
y = df['exam_score']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit random forest regressor
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)

# Predictions
y_train_pred = rf_reg.predict(X_train)
y_test_pred = rf_reg.predict(X_test)

# Metrics
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

print("In-sample RMSE:", rmse_train)
print("Out-of-sample RMSE:", rmse_test)
{%- endhighlight -%}

{%- highlight python -%}
# Binary Response

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# Prepare X and y (numeric predictors and binary response)
X = ads[['Gender', 'Age', 'EstimatedSalary']]  # Ensure all features are numeric
y = ads['Purchased']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit Gradient Boosting Classifier
gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_clf.fit(X_train, y_train)

# Predictions
y_train_pred = gb_clf.predict(X_train)
y_test_pred = gb_clf.predict(X_test)

# Probabilities for AUC
y_train_proba = gb_clf.predict_proba(X_train)[:, 1]
y_test_proba = gb_clf.predict_proba(X_test)[:, 1]

# Metrics
print("In-sample Accuracy:", accuracy_score(y_train, y_train_pred))
print("In-sample AUC:", roc_auc_score(y_train, y_train_proba))
print("Out-of-sample Accuracy:", accuracy_score(y_test, y_test_pred))
print("Out-of-sample AUC:", roc_auc_score(y_test, y_test_proba))
print("\nClassification Report (OOS):")
print(classification_report(y_test, y_test_pred))
{%- endhighlight -%}

#### R Code

#### Results


### BART

BART is a wonderful tool for data anlysis. It combines the flexibility of random forests with the strengths of Bayesian statistics. However, I was not able to ever get it to run in Python. There are a multiple packages in R to run BART (some are better than others), but I was never able to find one that actually worked in Python. There were many that ChatGPT told me to try, but I was unsuccessful to say the least. For all the other models, I include both R and Python code. For this one, I will only include R code. There is a way to connect Python to R in order to run R code in the Python environment, but what's the use in that if I have to write the code in R anyway? I will just run it in R. Sure, it may run faster in Python, but with the time it takes to connect R to Python, run it, and do that whole process, it's probably not worth it.

#### Python Code

{%- highlight python -%}
# This is my best attempt, but I could never figure it out

import pandas as pd
import numpy as np
from rpy2.robjects import pandas2ri
import rpy2.robjects as ro

# Activate pandas <-> R DataFrame conversion
pandas2ri.activate()

# Define which columns are predictors and which is response
X = df.drop(columns="exam_score")
y = df["exam_score"]

# Push to R
ro.globalenv["X"] = pandas2ri.py2rpy(X)
ro.globalenv["y"] = pandas2ri.py2rpy(y)

# Load BART and run the model in R
ro.r(
# Prepare your data: X numeric matrix, y numeric vector
X <- df[, c('age', 'study_hours_per_day', 'social_media_hours', 'netflix_hours',
            'part_time_job', 'attendance_percentage', 'sleep_hours',
            'exercise_frequency', 'mental_health_rating', 'extracurricular_participation',
            'genderMale', 'genderOther', 'diet_qualityGood', 'diet_qualityPoor',
            'internet_qualityGood', 'internet_qualityPoor')]
y <- df$exam_score

# Train-test split (70%-30%)
set.seed(42)
train_idx <- createDataPartition(y, p = 0.7, list = FALSE)
X_train <- as.matrix(X[train_idx, ])
y_train <- y[train_idx]
X_test <- as.matrix(X[-train_idx, ])
y_test <- y[-train_idx]

# Fit BART model
set.seed(123)
bart_model <- wbart(x.train = X_train, y.train = y_train, x.test = X_test)

# --- In-sample predictions ---
yhat_train_mean <- colMeans(bart_model$yhat.train)
rmse_train <- sqrt(mean((y_train - yhat_train_mean)^2))

# --- Out-of-sample predictions ---
yhat_test_mean <- colMeans(bart_model$yhat.test)
rmse_test <- sqrt(mean((y_test - yhat_test_mean)^2))
)

# Get predictions back into Python
rmse_train = np.array(ro.r('rmse_train'))
rmse_test = np.array(ro.r('rmse_test'))
preds_mean = preds.mean(axis=0)  # Average across posterior samples
{%- endhighlight -%}


#### R Code

#### Results


### Neural Network

#### Python Code

{%- highlight python -%}
# Continuous Response

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Example data
X = df[['age', 'study_hours_per_day', 'social_media_hours', 'netflix_hours',
        'part_time_job', 'attendance_percentage', 'sleep_hours',
        'exercise_frequency', 'mental_health_rating', 'extracurricular_participation',
        'gender_Male', 'gender_Other', 'diet_quality_Good', 'diet_quality_Poor',
        'internet_quality_Good', 'internet_quality_Poor']]  # make sure these are numeric
y = df['exam_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Neural network regressor
nn_reg = MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
nn_reg.fit(X_train, y_train)

# Predictions
y_train_pred = nn_reg.predict(X_train)
y_test_pred = nn_reg.predict(X_test)

# Evaluation
print("Train RMSE:", np.sqrt(mean_squared_error(y_train, y_train_pred)))
print("Test RMSE:", np.sqrt(mean_squared_error(y_test, y_test_pred)))
{%- endhighlight -%}

{%- highlight python -%}
# Binary Response

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# Define X and y
X = ads[['Gender', 'Age', 'EstimatedSalary']]  # Ensure all features are numeric
y = ads['Purchased']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

# Neural network classifier
nn_clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
nn_clf.fit(X_train, y_train)

# Predictions
y_train_pred = nn_clf.predict(X_train)
y_train_proba = nn_clf.predict_proba(X_train)[:, 1]
y_test_pred = nn_clf.predict(X_test)
y_test_proba = nn_clf.predict_proba(X_test)[:, 1]

# Evaluation
print("Train Accuracy:", accuracy_score(y_train, y_train_pred))
print("Train AUC:", roc_auc_score(y_train, y_train_proba))
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print("Test AUC:", roc_auc_score(y_test, y_test_proba))
{%- endhighlight -%}

#### R Code

#### Results



## Conclusion

