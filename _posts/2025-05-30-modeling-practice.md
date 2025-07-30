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
    formula='exam_score ~ age + study_hours_per_day + social_media_hours +
    netflix_hours + part_time_job + attendance_percentage + sleep_hours +
    exercise_frequency + mental_health_rating + extracurricular_participation +
    gender_Male + gender_Other + diet_quality_Good + diet_quality_Poor +
    internet_quality_Good + internet_quality_Poor',
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

{%- highlight python -%}
model <- lm(exam_score ~ age + study_hours_per_day + social_media_hours +
netflix_hours + part_time_job + attendance_percentage + sleep_hours +
exercise_frequency + mental_health_rating + extracurricular_participation +
genderMale + genderOther + diet_qualityGood + diet_qualityPoor +
internet_qualityGood + internet_qualityPoor,
data = df)

summary(model)
{%- endhighlight -%}


#### Results

The following table is the R summary output from the regression model:

<figure>
	<img src="{{site.url}}/{{site.baseurl}}/assets/img/linear_regression.png" alt="" style="width: 700px; height=auto;"> 
	<figcaption>ROC Curve for regression setting</figcaption>
    <figcaption>Image Source: <a href="https://www.r-project.org/about.html">R</a></figcaption>
</figure>




### LASSO

LASSO uses bootstrapping to obtain coefficients, which is a random process. Thus, we get different results each time we run the model. In Python, LASSO removed the age, gender_Male, diet_quality_Poor, and internet_quality_Poor variables. In R, LASSO removed the extracurricular_participation, genderMale, and internet_qualityPoor variables.

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

{%- highlight python -%}
library(glmnet)
library(caret)
library(dplyr)

# Define predictors and response
X <- df %>%
  select(
    age, study_hours_per_day, social_media_hours, netflix_hours,
    part_time_job, attendance_percentage, sleep_hours,
    exercise_frequency, mental_health_rating, extracurricular_participation,
    genderMale, genderOther, diet_qualityGood, diet_qualityPoor,
    internet_qualityGood, internet_qualityPoor
  ) %>% as.matrix()

y <- df$exam_score

# Train-test split (80/20)
set.seed(123)
train_index <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[train_index, ]
X_test <- X[-train_index, ]
y_train <- y[train_index]
y_test <- y[-train_index]

# Scale predictors (standardize)
scaler <- preProcess(X_train, method = c("center", "scale"))
X_train_scaled <- predict(scaler, X_train)

# Fit LASSO model (alpha = 1 for LASSO, lambda = 0.1)
lasso_model <- glmnet(X_train_scaled, y_train, alpha = 1, lambda = 0.1)

# Extract coefficients
coef(lasso_model)
{%- endhighlight -%}

#### Results

For an easy way to compare LASSO in Python and R, I will take the coefficients that are left in the model from each, run a linear regression model using them, and see how they compare. We can see from the outputs below that the Adjusted R^2 for the models in Python and R are identical. Even with the differences in which variables were removed, the fit is the same. I’m including the output from both coding languages not only to illustrate the differences in the values, but also to show you the format of the summary output in both Python and R. Python includes some metrics that R does not, and R includes some nice features for statisticians (like significance of the p-values) that Python does not. Python’s summary output is first, followed by R’s.

<figure>
	<img src="{{site.url}}/{{site.baseurl}}/assets/img/lasso_python.png" alt="" style="width: 700px; height=auto;"> 
	<figcaption>ROC Curve for regression setting</figcaption>
    <figcaption>Image Source: <a href="https://www.python.org/">Python</a></figcaption>
</figure>

<figure>
	<img src="{{site.url}}/{{site.baseurl}}/assets/img/lasso_r.png" alt="" style="width: 700px; height=auto;"> 
	<figcaption>ROC Curve for regression setting</figcaption>
    <figcaption>Image Source: <a href="https://www.r-project.org/about.html">R</a></figcaption>
</figure>




### Polynomial Regression

Polynomial Regression is deterministic, so we will get the same results in Python and R (and any coding language, for that matter) regardless of how many times we run it. For illustration purposes, I decided to put a squared term on age and a cubic term on study_hours_per_day. These polynomial terms do not reflect the true nature between those predictors and the response - this is simply to show how you would perform polynomial regression if the relationship was, in fact, not linear. Because this relationship isn’t actually what the data follows, it makes sense that we see a lower R^2 in this model compared to linear regression.

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

I will supply you with two R options as well.

{%- highlight python -%}
# R Option 1

model <- lm(exam_score ~ age + I(age^2) + study_hours_per_day + I(study_hours_per_day^2) +
I(study_hours_per_day^3) + social_media_hours + netflix_hours + part_time_job +
attendance_percentage + sleep_hours + exercise_frequency + mental_health_rating +
extracurricular_participation + genderMale + genderOther + diet_qualityGood +
diet_qualityPoor + internet_qualityGood + internet_qualityPoor,
data = df)

summary(model)
{%- endhighlight -%}

{%- highlight python -%}
# R Option 2

# The poly() function also works

model <- lm(exam_score ~ poly(age, 2, raw=TRUE) + poly(study_hours_per_day, 3, raw=TRUE) +
social_media_hours + netflix_hours + part_time_job + attendance_percentage + sleep_hours +
exercise_frequency + mental_health_rating + extracurricular_participation + genderMale +
genderOther + diet_qualityGood + diet_qualityPoor + internet_qualityGood + internet_qualityPoor,
data = df)

summary(model)
{%- endhighlight -%}

#### Results

Below is the summary output from Python:

<figure>
	<img src="{{site.url}}/{{site.baseurl}}/assets/img/polynomial_regression.png" alt="" style="width: 700px; height=auto;"> 
	<figcaption>ROC Curve for regression setting</figcaption>
    <figcaption>Image Source: <a href="https://www.python.org/">Python</a></figcaption>
</figure>



### Nautral Splines

Natural Splines are deterministic, so we theoretically will get the same results every time we run it. I say “theoretically” because different libraries handle natural splines slightly differently. For this implementation, I used the statsmodels library in Python and the splines library in R. Some generate a regression spline basis while others generate a B-spline-based natural spline basis, some don’t include an intercept term while others do, and some have different definitions of the spline basis (including different scaling, transformations, and constraints).

I could’ve used different libraries or finagled these two implementations to match more closely, but I wanted to actually leave it in how I initially did it. This is a good illustration of how different libraries work, but also of how both are completely valid ways of approaching a problem. Similarly, when you are given a problem, there are infinitely many ways of finding an answer. Some may be better than others, but as long as you are using a valid approach and can justify it, then that’s enough.

I decided to use a natural cubic spline with four degrees of freedom for age, meaning that four basis functions were used to represent the smooth, flexible relationship between age and the response. I also decided to use a natural cubic spline with three degrees of freedom for study_hours_per_day. As in polynomial regression, these are not accurate to the true relationships.


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
        'diet_quality_Good', 'diet_quality_Poor', 'internet_quality_Good',
        'internet_quality_Poor']]  # your other predictors
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

{%- highlight python -%}
# Load required package
library(splines)

# Define the model formula with natural splines
model <- lm(
  exam_score ~ ns(age, df = 4) + ns(study_hours_per_day, df = 3) +
    social_media_hours + netflix_hours + part_time_job + attendance_percentage +
    sleep_hours + exercise_frequency + mental_health_rating +
    extracurricular_participation + genderMale + genderOther +
    diet_qualityGood + diet_qualityPoor + internet_qualityGood + internet_qualityPoor,
  data = df
)

# Summary of the model
summary(model)
{%- endhighlight -%}

#### Results

Below is the summary output from both Python (first) and R (second).

<figure>
	<img src="{{site.url}}/{{site.baseurl}}/assets/img/natural_splines_python.png" alt="" style="width: 700px; height=auto;"> 
	<figcaption>ROC Curve for regression setting</figcaption>
    <figcaption>Image Source: <a href="https://www.python.org/">Python</a></figcaption>
</figure>

<figure>
	<img src="{{site.url}}/{{site.baseurl}}/assets/img/natural_splines_r.png" alt="" style="width: 700px; height=auto;"> 
	<figcaption>ROC Curve for regression setting</figcaption>
    <figcaption>Image Source: <a href="https://www.r-project.org/about.html">R</a></figcaption>
</figure>



### GAM

Generalized Additive Models (GAMs) are deterministic models, so we would expect to see the same results in Python and R. However, we experience the same thing here as we did with natural splines. The statsmodels library in Python and the mgcv library in R handle GAMs differently, so it will not be possible to get identical results. The results in R will always be the same as each other, and the results in Python will be the same from run to run, but the results in R will not match those in Python.

#### Python Code

In Python, I defined cubic splines with four degrees of freedom on age and three degrees of freedom on study_hours_per_day. In R, the work is done for me in choosing the basis function and smoothness penalty, but the degrees of freedom on age and study_hours_per_day is still four and three, respectively.

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

{%- highlight python -%}
library(mgcv)

# Fit GAM model
gam_model <- gam(
  exam_score ~ s(age, k = 4) + s(study_hours_per_day, k = 3) +
    social_media_hours + netflix_hours + part_time_job + attendance_percentage +
    sleep_hours + exercise_frequency + mental_health_rating +
    extracurricular_participation + genderMale + genderOther +
    diet_qualityGood + diet_qualityPoor + internet_qualityGood + internet_qualityPoor,
  data = df,
  method = "REML"  # recommended smoothing parameter estimation method
)

# Summary of the GAM model
summary(gam_model)
{%- endhighlight -%}

#### Results

Below are the results from Python and R:

<figure>
	<img src="{{site.url}}/{{site.baseurl}}/assets/img/gam_python.png" alt="" style="width: 700px; height=auto;"> 
	<figcaption>ROC Curve for regression setting</figcaption>
    <figcaption>Image Source: <a href="https://www.python.org/">Python</a></figcaption>
</figure>

<figure>
	<img src="{{site.url}}/{{site.baseurl}}/assets/img/gam_r.png" alt="" style="width: 700px; height=auto;"> 
	<figcaption>ROC Curve for regression setting</figcaption>
    <figcaption>Image Source: <a href="https://www.r-project.org/about.html">R</a></figcaption>
</figure>


### Logistic Regression

Logistic Regression works very similarly to traditional linear regression, except that it models the log odds of the response variable. Regardless, just like linear regression, logistic regression is a deterministic model. Logistic regression is a bit more “basic” (for lack of a better word) than any of the non-linear linear models, so we don’t have to worry about different libraries implementing logistic regression differently. The process is straightforward enough that any library *should* yield identical results regardless of coding language.

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

{%- highlight python -%}
# Make Purchased a factor
ads$Purchased <- as.factor(ads$Purchased)

model <- glm(Purchased ~ Gender + Age + EstimatedSalary, data=ads, family=binomial)
summary(model)
{%- endhighlight -%}

#### Results

Below is the summary output from Python followed by R:

<figure>
	<img src="{{site.url}}/{{site.baseurl}}/assets/img/logistic_regression_python.png" alt="" style="width: 700px; height=auto;"> 
	<figcaption>ROC Curve for regression setting</figcaption>
    <figcaption>Image Source: <a href="https://www.python.org/">Python</a></figcaption>
</figure>

<figure>
	<img src="{{site.url}}/{{site.baseurl}}/assets/img/logistic_regression_r.png" alt="" style="width: 700px; height=auto;"> 
	<figcaption>ROC Curve for regression setting</figcaption>
    <figcaption>Image Source: <a href="https://www.r-project.org/about.html">R</a></figcaption>
</figure>



## Machine Learning Models

For each of the Machine Learning Models, I will supply the code for the regression setting (continuous response variable) and the classification setting (binary response variable).


### K Nearest Neighbors

K Nearest Neighbors (KNN) is the first of the machine learning models we will discuss in this post. Even though machine learning models tend to be thought of as superior over linear models, this doesn’t mean they rely on super fancy techniques or crazy computing power. In fact, KNN is still a deterministic model. Once the number of neighbors (k) is specified, the results will be consistent each time the model is run.

#### Python Code

You may notice in this implementation that most of the lines of code aren’t even to run the actual model, and that is exactly right. The packages and libraries we use do almost all of the work, so most of what we have to do (from a coding standpoint) is prep the data and calculate metrics to be able to compare different models or even the same model but with different hyperparameters. This principle will remain consistent in all the machine learning models. The majority of the code is to prep the data to put into the model and then to compute metrics. But the actual running of the model is at most a few lines of code.

I also include the ROC curve plots for this model in the classification setting (it doesn’t make sense in the regression setting). I won’t include the ROC plot for each following model; I just wanted to show you what it looks like and how you could do it yourself.


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

{%- highlight python -%}
# Continuous Response

# Load required packages
library(FNN)       # For KNN regression

library(caret)     # For data splitting and preprocessing
library(dplyr)

# Ensure the categorical variables are numeric
# (Assuming you've already transformed them into: genderMale, genderOther, etc.)
X <- df %>%
  select(age, study_hours_per_day, social_media_hours, netflix_hours,
         part_time_job, attendance_percentage, sleep_hours,
         exercise_frequency, mental_health_rating, extracurricular_participation,
         genderMale, genderOther, diet_qualityGood, diet_qualityPoor,
         internet_qualityGood, internet_qualityPoor)

y <- df$exam_score

# Train/Test split (70/30)
set.seed(123)
train_index <- createDataPartition(y, p = 0.7, list = FALSE)
X_train <- X[train_index, ]
X_test  <- X[-train_index, ]
y_train <- y[train_index]
y_test  <- y[-train_index]

# Scale the predictors
X_train_scaled <- scale(X_train)
X_test_scaled  <- scale(X_test, center = attr(X_train_scaled, "scaled:center"),
                                  scale = attr(X_train_scaled, "scaled:scale"))

# Fit KNN model (k = 5)
knn_pred_train <- knn.reg(train = X_train_scaled, test = X_train_scaled, y = y_train, k = 5)$pred
knn_pred_test  <- knn.reg(train = X_train_scaled, test = X_test_scaled,  y = y_train, k = 5)$pred

# Calculate RMSE and R-squared
rmse_train <- sqrt(mean((y_train - knn_pred_train)^2))
rmse_test <- sqrt(mean((y_test - knn_pred_test)^2))

# Output results
cat(sprintf("In-sample RMSE: %.4f\n", rmse_train))
cat(sprintf("Out-of-sample RMSE: %.4f\n", rmse_test))
{%- endhighlight -%}

{%- highlight python -%}
# Binary Response

# Load required packages
library(FNN)
library(caret)
library(pROC)
library(dplyr)

# Ensure 'Purchased' is a factor (0/1)
ads$Purchased <- as.factor(ads$Purchased)

# Select predictors and target
X <- ads %>% select(Gender, Age, EstimatedSalary)
y <- ads$Purchased

# Split into training and testing
set.seed(123)
train_index <- createDataPartition(y, p = 0.7, list = FALSE)
X_train <- X[train_index, ]
X_test <- X[-train_index, ]
y_train <- y[train_index]
y_test <- y[-train_index]

# Standardize predictors
X_train_scaled <- scale(X_train)
X_test_scaled <- scale(X_test, center = attr(X_train_scaled, "scaled:center"),
                                  scale = attr(X_train_scaled, "scaled:scale"))

### --- In-Sample Predictions --- ###
knn_train <- knn(train = X_train_scaled, test = X_train_scaled, cl = y_train, k = 5, prob = TRUE)
train_preds <- knn_train
train_probs <- ifelse(train_preds == "1", attr(knn_train, "prob"), 1 - attr(knn_train, "prob"))
train_accuracy <- mean(train_preds == y_train)
train_auc <- auc(roc(as.numeric(as.character(y_train)), train_probs))

### --- Out-of-Sample Predictions --- ###
knn_test <- knn(train = X_train_scaled, test = X_test_scaled, cl = y_train, k = 5, prob = TRUE)
test_preds <- knn_test
test_probs <- ifelse(test_preds == "1", attr(knn_test, "prob"), 1 - attr(knn_test, "prob"))
test_accuracy <- mean(test_preds == y_test)
test_auc <- auc(roc(as.numeric(as.character(y_test)), test_probs))

### --- Print Results --- ###
cat(sprintf("In-sample Accuracy: %.4f\n", train_accuracy))
cat(sprintf("In-sample AUC: %.4f\n", train_auc))
cat(sprintf("Out-of-sample Accuracy: %.4f\n", test_accuracy))
cat(sprintf("Out-of-sample AUC: %.4f\n", test_auc))

# Optional: Print confusion matrix for test set
confusionMatrix(test_preds, y_test, positive = "1")
{%- endhighlight -%}

#### Results

We will begin with the regression setting (the Exam Scores dataset). Python had slightly better metrics, but the difference was quite minimal (only a difference of about 0.15% out-of-sample). This is likely due just to Python getting a more favorable training/testing dataset split. Below are the results from Python, and then R:

{%- highlight python -%}
# Regression Setting - Python Results
In-sample RMSE: 2.0005
Out-of-sample RMSE: 2.7387

# Regression Setting - R Results
In-sample RMSE: 2.0256
Out-of-sample RMSE: 2.8813
{%- endhighlight -%}

For the classification setting (the Social Media Ads dataset), we see that R had better metrics. To reiterate what I wrote in the Introduction, however, the point of this post is not to decide if Python is better than R or vice versa. I am nowhere near smart enough to know how to answer that, nor would I be naive enough to even try because they both have so many strengths in different areas. Rather, this post is simply to teach you how to deploy certain models in both languages, and then you can choose how to implement them yourself. Below are the results from Python and R in the classification setting:

<figure>
	<img src="{{site.url}}/{{site.baseurl}}/assets/img/knn_python.png" alt="" style="width: 500px; height=auto;"> 
	<figcaption>ROC Curve for regression setting</figcaption>
    <figcaption>Image Source: <a href="https://www.python.org/">Python</a></figcaption>
</figure>

<figure>
	<img src="{{site.url}}/{{site.baseurl}}/assets/img/knn_r.png" alt="" style="width: 600px; height=auto;"> 
	<figcaption>ROC Curve for binary setting</figcaption>
    <figcaption>Image Source: <a href="https://www.r-project.org/about.html">R</a></figcaption>
</figure>

<figure>
	<img src="{{site.url}}/{{site.baseurl}}/assets/img/roc_regression.png" alt="" style="width: 700px; height=auto;"> 
	<figcaption>ROC Curve for regression setting</figcaption>
    <figcaption>Image Source: <a href="https://www.python.org/">Python</a></figcaption>
</figure>

<figure>
	<img src="{{site.url}}/{{site.baseurl}}/assets/img/roc_binary.png" alt="" style="width: 700px; height=auto;"> 
	<figcaption>ROC Curve for binary setting</figcaption>
    <figcaption>Image Source: <a href="https://www.r-project.org/about.html">R</a></figcaption>
</figure>


### Support Vector Machines

Support Vector Machines (SVM) are also deterministic models, which makes sense because they are very similar to linear regression in that they try to find the best hyperplane that matches the data. We see very different results between Python and R, but that is not because the models are being computed any differently.

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

{%- highlight python -%}
# Continuous Response

# Load necessary libraries
library(e1071)     # For SVR
library(caret)     # For train/test split and evaluation
library(dplyr)

# Define predictors and response
X <- df %>% select(age, study_hours_per_day, social_media_hours, netflix_hours,
                   part_time_job, attendance_percentage, sleep_hours,
                   exercise_frequency, mental_health_rating, extracurricular_participation,
                   genderMale, genderOther, diet_qualityGood, diet_qualityPoor,
                   internet_qualityGood, internet_qualityPoor)

y <- df$exam_score

# Train/test split
set.seed(123)
train_index <- createDataPartition(y, p = 0.7, list = FALSE)
X_train <- X[train_index, ]
X_test  <- X[-train_index, ]
y_train <- y[train_index]
y_test  <- y[-train_index]

# Combine X and y for training since svm() needs formula interface or combined data
train_data <- cbind(X_train, exam_score = y_train)
test_data  <- cbind(X_test,  exam_score = y_test)

# Fit SVR model (default is radial basis kernel)
svr_model <- svm(exam_score ~ ., data = train_data, kernel = "radial")

# Predict
y_pred_train <- predict(svr_model, newdata = X_train)
y_pred_test  <- predict(svr_model, newdata = X_test)

# Compute RMSE
rmse_train <- sqrt(mean((y_train - y_pred_train)^2))
rmse_test  <- sqrt(mean((y_test - y_pred_test)^2))

# Output results
cat(sprintf("In-sample RMSE: %.4f\n", rmse_train))
cat(sprintf("Out-of-sample RMSE: %.4f\n", rmse_test))
{%- endhighlight -%}

{%- highlight python -%}
# Binary Response

# Load necessary libraries
library(e1071)       # For svm()
library(caret)       # For train/test split
library(pROC)        # For AUC calculation

# Prepare data (make sure Gender is numeric or already dummy coded)
X <- ads[, c("Gender", "Age", "EstimatedSalary")]
y <- ads$Purchased

# Ensure Gender is numeric (if it's a factor)
if (is.factor(X$Gender)) {
  X$Gender <- as.numeric(X$Gender)  # or use model.matrix to one-hot encode if needed
}

# Train/test split
set.seed(123)
train_index <- createDataPartition(y, p = 0.7, list = FALSE)
X_train <- X[train_index, ]
X_test  <- X[-train_index, ]
y_train <- y[train_index]
y_test  <- y[-train_index]

# Combine X and y for svm() training
train_data <- cbind(X_train, Purchased = as.factor(y_train))  # Response must be factor for classification
test_data  <- cbind(X_test, Purchased = as.factor(y_test))

# Train SVM with radial kernel and probability estimates
svm_model <- svm(Purchased ~ ., data = train_data, kernel = "radial", probability = TRUE)

# Predictions (class labels)
y_train_pred <- predict(svm_model, newdata = X_train)
y_test_pred  <- predict(svm_model, newdata = X_test)

# Probabilities
y_train_prob <- attr(predict(svm_model, newdata = X_train, probability = TRUE), "probabilities")[, 2]
y_test_prob  <- attr(predict(svm_model, newdata = X_test, probability = TRUE), "probabilities")[, 2]

# Accuracy
train_acc <- mean(y_train_pred == y_train)
test_acc  <- mean(y_test_pred == y_test)

# AUC
train_auc <- auc(y_train, y_train_prob)
test_auc  <- auc(y_test, y_test_prob)

# Output results
cat(sprintf("In-sample Accuracy: %.4f\n", train_acc))
cat(sprintf("In-sample AUC: %.4f\n", train_auc))
cat(sprintf("Out-of-sample Accuracy: %.4f\n", test_acc))
cat(sprintf("Out-of-sample AUC: %.4f\n", test_auc))

cat("\nConfusion Matrix - Testing Data:\n")
print(confusionMatrix(y_test_pred, as.factor(y_test)))
{%- endhighlight -%}

#### Results

The main difference between the scikit-learn library in Python and e1071 library in R is that e1071 automatically scales features by default, but scikit-learn does not. Because of this, we are getting pretty different results. I should have handled this more carefully when I ran the model. I didn’t notice this error until I began writing this post, and I thought about fixing it, but wanted to leave this in as a cautionary example! It’s important to remember that data science is not a life or death type of industry. For the most part, if you make a mistake implementing a model, nobody will die. There may be financial or social implications, but it’s okay to not be perfect. However, you should do your absolute best to minimize mistakes and correct them before they lead to anything worse. Not scaling features is a pretty benevolent mistake to make, but it could have huge consequences depending on the context. So next time you are learning something new, and especially the next time you make a mistake, forgive yourself, remember you’re human, and create ways for you to not make the mistake again the next time. Anyway, here are the (incorrect) results from Python and the (better) results from R in the regression setting:

{%- highlight python -%}
# Regression Setting - Python Results
In-sample RMSE: 13.5663
Out-of-sample RMSE: 13.4650

# Regression Setting - R Results
In-sample RMSE: 2.7143
Out-of-sample RMSE: 3.0696
{%- endhighlight -%}

For the binary setting, we see the same thing where the model in R is greatly outperforming the model in Python. If I had manually scaled the features, the results would be much closer, and Python may even be outperforming R. However, here are the results I got:

<figure>
	<img src="{{site.url}}/{{site.baseurl}}/assets/img/svm_python.png" alt="" style="width: 700px; height=auto;"> 
	<figcaption>ROC Curve for regression setting</figcaption>
    <figcaption>Image Source: <a href="https://www.python.org/">Python</a></figcaption>
</figure>

<figure>
	<img src="{{site.url}}/{{site.baseurl}}/assets/img/svm_r.png" alt="" style="width: 700px; height=auto;"> 
	<figcaption>ROC Curve for binary setting</figcaption>
    <figcaption>Image Source: <a href="https://www.r-project.org/about.html">R</a></figcaption>
</figure>



### Decision Tree (CART)

CART is another deterministic model. Assuming you use the same loss function, hyperparameters, and library, you will get the same tree every time. Using different libraries may lead to slightly different results, but it is always very close, as we will see shortly.

#### Python Code

The only hyperparameter I chose for this implementation was a max depth of 3. This means that the tree can’t be split more than three times, helping avoid overfitting.

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

{%- highlight python -%}
# Continuous Response

# Load necessary libraries
library(rpart)
library(rpart.plot)
library(Metrics)  # for RMSE
set.seed(123)

# Ensure your data frame is numeric where needed
# If your df is not already loaded, load it here

# Define features and target
features <- c('age', 'study_hours_per_day', 'social_media_hours', 'netflix_hours',
              'part_time_job', 'attendance_percentage', 'sleep_hours',
              'exercise_frequency', 'mental_health_rating', 'extracurricular_participation',
              'genderMale', 'genderOther', 'diet_qualityGood', 'diet_qualityPoor',
              'internet_qualityGood', 'internet_qualityPoor')

X <- df[, features]
y <- df$exam_score

# Combine features and target into one data frame
data <- cbind(X, exam_score = y)

# Split into training and testing sets (70/30)
sample_idx <- sample(seq_len(nrow(data)), size = 0.7 * nrow(data))
train_data <- data[sample_idx, ]
test_data  <- data[-sample_idx, ]

# Fit regression tree
reg_tree <- rpart(exam_score ~ ., data = train_data, method = "anova", control = rpart.control(maxdepth = 3))

# Predictions
y_train_pred <- predict(reg_tree, newdata = train_data)
y_test_pred <- predict(reg_tree, newdata = test_data)

# RMSE
rmse_train <- rmse(train_data$exam_score, y_train_pred)
rmse_test <- rmse(test_data$exam_score, y_test_pred)

cat(sprintf("Regression Tree - In-sample RMSE: %.4f\n", rmse_train))
cat(sprintf("Regression Tree - Out-of-sample RMSE: %.4f\n", rmse_test))

# Plot the tree
rpart.plot(reg_tree, main = "Regression Tree", extra = 101, type = 2, under = TRUE, faclen = 0)
{%- endhighlight -%}

{%- highlight python -%}
# Binary Response

# Load necessary libraries
library(rpart)
library(rpart.plot)
library(caret)      # for createDataPartition
library(pROC)       # for AUC

# Assume your data frame is called ads
# Make sure 'Gender' and other features are numeric or factors as needed

# Example: Convert Gender to factor if not already
ads$Gender <- as.factor(ads$Gender)
ads$Purchased <- as.factor(ads$Purchased)  # Target as factor

# Define features and target
X <- ads[, c("Gender", "Age", "EstimatedSalary")]
y <- ads$Purchased

# Train/test split: 70% train, 30% test
set.seed(123)
train_idx <- createDataPartition(y, p = 0.7, list = FALSE)
train_data <- ads[train_idx, ]
test_data <- ads[-train_idx, ]

# Fit classification tree with max depth = 3
# In rpart, control max depth via maxdepth parameter
fit <- rpart(Purchased ~ Gender + Age + EstimatedSalary,
             data = train_data,
             method = "class",
             control = rpart.control(maxdepth = 3))

# Predict class labels on train and test
train_pred_class <- predict(fit, newdata = train_data, type = "class")
test_pred_class <- predict(fit, newdata = test_data, type = "class")

# Predict probabilities for class "1" (assuming positive class is "1")
train_pred_prob <- predict(fit, newdata = train_data, type = "prob")[, "1"]
test_pred_prob <- predict(fit, newdata = test_data, type = "prob")[, "1"]

# Calculate accuracy
acc_train <- mean(train_pred_class == train_data$Purchased)
acc_test <- mean(test_pred_class == test_data$Purchased)

# Calculate AUC
auc_train <- roc(train_data$Purchased, train_pred_prob)$auc
auc_test <- roc(test_data$Purchased, test_pred_prob)$auc

cat(sprintf("In-sample Accuracy: %.4f\n", acc_train))
cat(sprintf("In-sample AUC: %.4f\n", auc_train))
cat(sprintf("Out-of-sample Accuracy: %.4f\n", acc_test))
cat(sprintf("Out-of-sample AUC: %.4f\n", auc_test))

# Plot the classification tree
rpart.plot(fit, main = "Classification Tree", type = 3, extra = 104, fallen.leaves = TRUE)
{%- endhighlight -%}

#### Results

For both the continuous and binary response variable settings, we see nearly identical results. Below are the results for the regression setting:

{%- highlight python -%}
# Regression Setting - Python Results
In-sample RMSE: 9.4565
Out-of-sample RMSE: 9.4325

# Regression Setting - R Results
In-sample RMSE: 9.4259
Out-of-sample RMSE: 9.9073
{%- endhighlight -%}

<figure>
	<img src="{{site.url}}/{{site.baseurl}}/assets/img/cart_regression_python_regression.png" alt="" style="width: 700px; height=auto;"> 
	<figcaption>Regression Tree in regression setting - Python</figcaption>
    <figcaption>Image Source: <a href="https://www.python.org/">Python</a></figcaption>
</figure>

<figure>
	<img src="{{site.url}}/{{site.baseurl}}/assets/img/cart_regression_r_regression.png" alt="" style="width: 700px; height=auto;"> 
	<figcaption>Regression Tree in regression setting - R</figcaption>
    <figcaption>Image Source: <a href="https://www.r-project.org/about.html">R</a></figcaption>
</figure>

Below are the results for the classification setting:

{%- highlight python -%}
# Classification Setting - Python Results
In-sample Accuracy: 0.9286
Out-of-sample Accuracy: 0.8917
In-sample AUC: 0.9756
Out-of-sample AUC: 0.9156

# Classification Setting - R Results
In-sample Accuracy: 0.9253
Out-of-sample Accuracy: 0.8992
In-sample AUC: 0.9290
Out-of-sample AUC: 0.9097
{%- endhighlight -%}

<figure>
	<img src="{{site.url}}/{{site.baseurl}}/assets/img/cart_regression_python_binary.png" alt="" style="width: 700px; height=auto;"> 
	<figcaption>Regression Tree in binary setting - Python</figcaption>
    <figcaption>Image Source: <a href="https://www.python.org/">Python</a></figcaption>
</figure>

<figure>
	<img src="{{site.url}}/{{site.baseurl}}/assets/img/cart_regression_r_binary.png" alt="" style="width: 700px; height=auto;"> 
	<figcaption>Regression Tree in binary setting - R</figcaption>
    <figcaption>Image Source: <a href="https://www.r-project.org/about.html">R</a></figcaption>
</figure>




### Random Forest

As I said in the Introduction, random forests are not deterministic because they only use a subset of the data and a subset of the predictor variables to fit each tree. I’ve always thought the name “random forest” is very cute. The “random” part refers to the random subsets of the data and predictors used to fit each tree. The “forest” part refers to the many trees that are grown and averaged across to get the prediction. This side note holds nothing of value. I just wanted it to be known that I really like the name random forest.

#### Python Code

Everything I did here is pretty much straight out of the box, with the exception of only having 100 trees grown. I did this mostly just to keep computation time down, though having more trees would’ve taken only milliseconds longer. These datasets are pretty small and simple, so any differences in computation time were negligible.

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

{%- highlight python -%}
# Continuous Response

# Load necessary libraries
library(randomForest)
library(caret)  # for data splitting

# Assume your data.frame is called df and all variables are numeric as in Python

# Features and target
X <- df[, c('age', 'study_hours_per_day', 'social_media_hours', 'netflix_hours',
            'part_time_job', 'attendance_percentage', 'sleep_hours',
            'exercise_frequency', 'mental_health_rating', 'extracurricular_participation',
            'genderMale', 'genderOther', 'diet_qualityGood', 'diet_qualityPoor',
            'internet_qualityGood', 'internet_qualityPoor')]

y <- df$exam_score

# Combine X and y for caret's createDataPartition
data_rf <- data.frame(X, exam_score = y)

# Split data (70% train, 30% test)
set.seed(123)
train_index <- createDataPartition(data_rf$exam_score, p = 0.7, list = FALSE)
train_data <- data_rf[train_index, ]
test_data <- data_rf[-train_index, ]

# Fit random forest regressor
rf_model <- randomForest(exam_score ~ ., data = train_data, ntree = 100, importance = TRUE)

# Predict on train and test
train_pred <- predict(rf_model, newdata = train_data)
test_pred <- predict(rf_model, newdata = test_data)

# Calculate RMSE
rmse <- function(actual, predicted) {
  sqrt(mean((actual - predicted)^2))
}

rmse_train <- rmse(train_data$exam_score, train_pred)
rmse_test <- rmse(test_data$exam_score, test_pred)

cat("In-sample RMSE:", round(rmse_train, 4), "\n")
cat("Out-of-sample RMSE:", round(rmse_test, 4), "\n")
{%- endhighlight -%}

{%- highlight python -%}
# Binary Response

# Load libraries
library(randomForest)
library(caret)
library(pROC)  # for AUC

# Assume your dataframe is ads and variables are numeric/factors as needed

# Features and target
X <- ads[, c('Gender', 'Age', 'EstimatedSalary')]  # Make sure Gender is numeric/factor
y <- ads$Purchased  # binary factor or 0/1

# Combine X and y for splitting
data_rf <- data.frame(X, Purchased = y)

# Train-test split (70%-30%)
set.seed(123)
train_index <- createDataPartition(data_rf$Purchased, p = 0.7, list = FALSE)
train_data <- data_rf[train_index, ]
test_data <- data_rf[-train_index, ]

# Train random forest classifier
rf_model <- randomForest(Purchased ~ ., data = train_data, ntree = 100)

# Predict classes
train_pred <- predict(rf_model, train_data)
test_pred <- predict(rf_model, test_data)

# Predict probabilities (needed for AUC)
train_proba <- predict(rf_model, train_data, type = "prob")[, 2]
test_proba <- predict(rf_model, test_data, type = "prob")[, 2]

# Accuracy
acc_train <- mean(train_pred == train_data$Purchased)
acc_test <- mean(test_pred == test_data$Purchased)

# AUC
auc_train <- roc(train_data$Purchased, train_proba)$auc
auc_test <- roc(test_data$Purchased, test_proba)$auc

cat("In-sample Accuracy:", round(acc_train, 4), "\n")
cat("In-sample AUC:", round(auc_train, 4), "\n")
cat("Out-of-sample Accuracy:", round(acc_test, 4), "\n")
cat("Out-of-sample AUC:", round(auc_test, 4), "\n\n")

# Classification report (precision, recall, F1)
conf_mat <- confusionMatrix(test_pred, test_data$Purchased, positive = "1")
print(conf_mat)
{%- endhighlight -%}

#### Results

Below are the regression setting results:

{%- highlight python -%}
# Regression Setting - Python Results
In-sample RMSE: 1.1742
Out-of-sample RMSE: 3.0872

# Regression Setting - R Results
In-sample RMSE: 1.302
Out-of-sample RMSE: 2.906
{%- endhighlight -%}


Below are the classification setting results:

<figure>
	<img src="{{site.url}}/{{site.baseurl}}/assets/img/random_forest_python.png" alt="" style="width: 700px; height=auto;"> 
	<figcaption>ROC Curve for regression setting</figcaption>
    <figcaption>Image Source: <a href="https://www.python.org/">Python</a></figcaption>
</figure>

<figure>
	<img src="{{site.url}}/{{site.baseurl}}/assets/img/random_forest_r.png" alt="" style="width: 700px; height=auto;"> 
	<figcaption>ROC Curve for binary setting</figcaption>
    <figcaption>Image Source: <a href="https://www.r-project.org/about.html">R</a></figcaption>
</figure>




### Boosting

Boosting is very similar to random forests, except that each tree is fit on the residuals of the trees before it, not on a new random subset of the data each time. Because of this, boosted models are actually deterministic in theory. The weak learners (decision trees) are added in a fixed pattern and the learners fix mistakes in a deterministic manner. However, in practice, the only way to make it deterministic is to control all randomness introduced by the coding libraries (setting the seed, subsampling, turning the deterministic option on, etc.).

#### Python Code

Again, the only hyperparameter I changed was setting the number of trees to be 100. That goes for both the regression setting and classification setting.

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

{%- highlight python -%}
# Continuous Response

# Load libraries
library(xgboost)
library(caret)
library(Metrics)  # for rmse calculation

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

# Convert to xgb.DMatrix (efficient data structure for xgboost)
dtrain <- xgb.DMatrix(data = X_train, label = y_train)
dtest <- xgb.DMatrix(data = X_test, label = y_test)

# Set parameters for regression
params <- list(
  objective = "reg:squarederror",  # regression task with squared error loss
  eval_metric = "rmse"
)

# Train boosted trees model
set.seed(123)
xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 100,               # number of boosting rounds (trees)
  watchlist = list(train = dtrain),
  verbose = 0
)

# Predict
y_train_pred <- predict(xgb_model, dtrain)
y_test_pred <- predict(xgb_model, dtest)

# Compute RMSE
rmse_train <- rmse(y_train, y_train_pred)
rmse_test <- rmse(y_test, y_test_pred)

cat("In-sample RMSE:", round(rmse_train, 4), "\n")
cat("Out-of-sample RMSE:", round(rmse_test, 4), "\n")
{%- endhighlight -%}

{%- highlight python -%}
# Binary Response

# Load libraries
library(xgboost)
library(caret)
library(pROC)

# Prepare your data
X <- ads[, c('Gender', 'Age', 'EstimatedSalary')]  # numeric predictors only
y <- as.numeric(as.character(ads$Purchased))  # binary response (0/1 or factor with two levels)

# If 'Gender' is a factor, convert to numeric (e.g., one-hot encoding or numeric encoding)
# For simplicity, let's convert factor to numeric:
if (is.factor(X$Gender)) {
  X$Gender <- as.numeric(as.factor(X$Gender))
}

# Train-test split (70%-30%)
set.seed(123)
train_idx <- createDataPartition(y, p = 0.7, list = FALSE)
X_train <- as.matrix(X[train_idx, ])
y_train <- y[train_idx]
X_test <- as.matrix(X[-train_idx, ])
y_test <- y[-train_idx]

# Convert to xgb.DMatrix
dtrain <- xgb.DMatrix(data = X_train, label = y_train)
dtest <- xgb.DMatrix(data = X_test, label = y_test)

# Set parameters for binary classification
params <- list(
  objective = "binary:logistic",  # binary classification with logistic loss
  eval_metric = "auc",
  max_depth = 3,
  eta = 0.1  # learning rate
)

# Train the model
set.seed(123)
xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 100,
  watchlist = list(train = dtrain),
  verbose = 0
)

# Predict probabilities
y_train_proba <- predict(xgb_model, dtrain)
y_test_proba <- predict(xgb_model, dtest)

# Convert probabilities to class labels using 0.5 cutoff
y_train_pred <- ifelse(y_train_proba > 0.5, 1, 0)
y_test_pred <- ifelse(y_test_proba > 0.5, 1, 0)

# Calculate accuracy
acc_train <- mean(y_train_pred == y_train)
acc_test <- mean(y_test_pred == y_test)

# Calculate AUC
auc_train <- roc(y_train, y_train_proba)$auc
auc_test <- roc(y_test, y_test_proba)$auc

cat("In-sample Accuracy:", round(acc_train, 4), "\n")
cat("In-sample AUC:", round(auc_train, 4), "\n")
cat("Out-of-sample Accuracy:", round(acc_test, 4), "\n")
cat("Out-of-sample AUC:", round(auc_test, 4), "\n")

# Optional: Confusion matrix for test set
table(Predicted = y_test_pred, Actual = y_test)
{%- endhighlight -%}

#### Results

Below are the results from the regression setting:

{%- highlight python -%}
# Regression Setting - Python Results
In-sample RMSE: 4.6946
Out-of-sample RMSE: 4.9761

# Regression Setting - R Results
In-sample RMSE: 1.7777
Out-of-sample RMSE: 3.5823
{%- endhighlight -%}

Below are the results from the classification setting:

<figure>
	<img src="{{site.url}}/{{site.baseurl}}/assets/img/boosting_python.png" alt="" style="width: 700px; height=auto;"> 
	<figcaption>ROC Curve for regression setting</figcaption>
    <figcaption>Image Source: <a href="https://www.python.org/">Python</a></figcaption>
</figure>

<figure>
	<img src="{{site.url}}/{{site.baseurl}}/assets/img/boosting_r.png" alt="" style="width: 700px; height=auto;"> 
	<figcaption>ROC Curve for binary setting</figcaption>
    <figcaption>Image Source: <a href="https://www.r-project.org/about.html">R</a></figcaption>
</figure>



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

{%- highlight python -%}
# Continuous Response

# Load library
library(BART)

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
cat("In-sample RMSE:", round(rmse_train, 4), "\n")

# --- Out-of-sample predictions ---
yhat_test_mean <- colMeans(bart_model$yhat.test)
rmse_test <- sqrt(mean((y_test - yhat_test_mean)^2))
cat("Out-of-sample RMSE:", round(rmse_test, 4), "\n")

# Optionally, credible intervals
ci_lower <- apply(bart_model$yhat.test, 2, quantile, probs = 0.025)
ci_upper <- apply(bart_model$yhat.test, 2, quantile, probs = 0.975)
{%- endhighlight -%}

{%- highlight python -%}
# Binary Response

# Load libraries
library(BART)
library(caret)
library(pROC)

# Prepare the data
X <- ads[, c('Gender', 'Age', 'EstimatedSalary')]
y <- as.numeric(as.character(ads$Purchased))  # must be binary: 0 or 1

# Convert Gender to numeric if it's a factor
if (is.factor(X$Gender)) {
  X$Gender <- as.numeric(as.factor(X$Gender))
}

# Split into training and testing sets
set.seed(123)
train_idx <- createDataPartition(y, p = 0.7, list = FALSE)
X_train <- as.matrix(X[train_idx, ])
y_train <- y[train_idx]
X_test <- as.matrix(X[-train_idx, ])
y_test <- y[-train_idx]

# Fit BART for binary response
set.seed(123)
bart_model <- pbart(x.train = X_train, y.train = y_train, x.test = X_test)

# Predicted probabilities
y_train_proba <- colMeans(bart_model$prob.train)
y_test_proba <- colMeans(bart_model$prob.test)

# Predicted class labels (using 0.5 threshold)
y_train_pred <- ifelse(y_train_proba > 0.5, 1, 0)
y_test_pred <- ifelse(y_test_proba > 0.5, 1, 0)

# Accuracy
acc_train <- mean(y_train_pred == y_train)
acc_test <- mean(y_test_pred == y_test)

# AUC
auc_train <- roc(y_train, y_train_proba)$auc
auc_test <- roc(y_test, y_test_proba)$auc

# Output results
cat("In-sample Accuracy:", round(acc_train, 4), "\n")
cat("In-sample AUC:", round(auc_train, 4), "\n")
cat("Out-of-sample Accuracy:", round(acc_test, 4), "\n")
cat("Out-of-sample AUC:", round(auc_test, 4), "\n")

# Optional: Confusion matrix
cat("\nConfusion Matrix (OOS):\n")
print(table(Predicted = y_test_pred, Actual = y_test))
{%- endhighlight -%}

#### Results

Below are the results from the regression setting:

{%- highlight python -%}
# Regression Setting - Python Results
None :(

# Regression Setting - R Results
In-sample RMSE: 2.8416
Out-of-sample RMSE: 3.6197
{%- endhighlight -%}

Below are the results from the classification setting:

<figure>
	<img src="{{site.url}}/{{site.baseurl}}/assets/img/bart_r.png" alt="" style="width: 700px; height=auto;"> 
	<figcaption>ROC Curve for regression setting</figcaption>
    <figcaption>Image Source: <a href="https://www.r-project.org/about.html">R</a></figcaption>
</figure>




### Neural Network

Neural Networks are definitely not a deterministic model. In fact, they might be the farthest thing from them. Neural networks are "black boxes," which is just a saying in the statistical world that means we have no idea what they're really doing internally to give us the results. We understand the general process, we understand the math behind them, but we aren't able to see into neural nets while they run to understand what exactly they are doing. However, that doesn't mean they aren't extremely powerful. I hope you learn as much as you can about neural networks, these models, and data science in general. Never be satisfied with not understanding.

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

{%- highlight python -%}
# Continuous Response

# Load necessary libraries
library(nnet)     # for neural network
library(caret)    # for train/test split and preprocessing
library(Metrics)  # for RMSE
library(dplyr)

# Prepare the data
X <- df %>%
  select(age, study_hours_per_day, social_media_hours, netflix_hours,
         part_time_job, attendance_percentage, sleep_hours,
         exercise_frequency, mental_health_rating, extracurricular_participation,
         genderMale, genderOther, diet_qualityGood, diet_qualityPoor,
         internet_qualityGood, internet_qualityPoor)

y <- df$exam_score

# Train/test split
set.seed(123)
train_index <- createDataPartition(y, p = 0.7, list = FALSE)
X_train <- X[train_index, ]
X_test  <- X[-train_index, ]
y_train <- y[train_index]
y_test  <- y[-train_index]

# Standardize the predictors (recommended for neural nets)
preproc <- preProcess(X_train, method = c("center", "scale"))
X_train_scaled <- predict(preproc, X_train)
X_test_scaled  <- predict(preproc, X_test)

# Fit neural network
set.seed(123)
nn_model <- nnet(
  x = as.matrix(X_train_scaled),
  y = y_train,
  size = 10,         # number of hidden units
  linout = TRUE,     # for regression
  maxit = 500        # max iterations
)

# Predictions
y_train_pred <- predict(nn_model, as.matrix(X_train_scaled))
y_test_pred  <- predict(nn_model, as.matrix(X_test_scaled))

# Evaluation
train_rmse <- rmse(y_train, y_train_pred)
test_rmse  <- rmse(y_test, y_test_pred)
train_r2   <- 1 - sum((y_train - y_train_pred)^2) / sum((y_train - mean(y_train))^2)
test_r2    <- 1 - sum((y_test - y_test_pred)^2) / sum((y_test - mean(y_test))^2)

cat("Train RMSE:", round(train_rmse, 4), "\n")
cat("Test RMSE:", round(test_rmse, 4), "\n")
{%- endhighlight -%}

{%- highlight python -%}
# Binary Response

# Load necessary libraries
library(nnet)     # for neural network modeling
library(caret)    # for data partitioning
library(pROC)     # for AUC
library(dplyr)

# Define X and y
X <- ads %>%
  select(Gender, Age, EstimatedSalary)

# Ensure Gender is numeric
if (is.factor(X$Gender)) {
  X$Gender <- as.numeric(as.factor(X$Gender))
}

y <- as.numeric(as.character(ads$Purchased))  # Binary response (0/1)

# Train-test split
set.seed(123)
train_index <- createDataPartition(y, p = 0.7, list = FALSE)
X_train <- X[train_index, ]
X_test  <- X[-train_index, ]
y_train <- y[train_index]
y_test  <- y[-train_index]

# Standardize features (important for neural nets)
preproc <- preProcess(X_train, method = c("center", "scale"))
X_train_scaled <- predict(preproc, X_train)
X_test_scaled  <- predict(preproc, X_test)

# Fit neural network classifier
set.seed(42)
nn_clf <- nnet(
  x = as.matrix(X_train_scaled),
  y = y_train,
  size = 10,          # number of hidden units
  linout = FALSE,     # FALSE = classification
  maxit = 500,        # max iterations
  trace = FALSE       # suppress output
)

# Predictions
y_train_proba <- predict(nn_clf, as.matrix(X_train_scaled), type = "raw")
y_test_proba  <- predict(nn_clf, as.matrix(X_test_scaled), type = "raw")

# Convert probabilities to class labels using 0.5 cutoff
y_train_pred <- ifelse(y_train_proba > 0.5, 1, 0)
y_test_pred  <- ifelse(y_test_proba > 0.5, 1, 0)

# Evaluation
acc_train <- mean(y_train_pred == y_train)
acc_test  <- mean(y_test_pred == y_test)
auc_train <- roc(y_train, y_train_proba)$auc
auc_test  <- roc(y_test, y_test_proba)$auc

cat("Train Accuracy:", round(acc_train, 4), "\n")
cat("Train AUC:", round(auc_train, 4), "\n")
cat("Test Accuracy:", round(acc_test, 4), "\n")
cat("Test AUC:", round(auc_test, 4), "\n")
{%- endhighlight -%}

#### Results

Below are the results from the regression setting:

{%- highlight python -%}
# Regression Setting - Python Results
In-sample RMSE: 5.7950
Out-of-sample RMSE: 5.7248

# Regression Setting - R Results
In-sample RMSE: 4.6167
Out-of-sample RMSE: 5.0804
{%- endhighlight -%}

Below are the results from the classification setting:

<figure>
	<img src="{{site.url}}/{{site.baseurl}}/assets/img/neural_network_python.png" alt="" style="width: 700px; height=auto;"> 
	<figcaption>ROC Curve for regression setting</figcaption>
    <figcaption>Image Source: <a href="https://www.python.org/">Python</a></figcaption>
</figure>

<figure>
	<img src="{{site.url}}/{{site.baseurl}}/assets/img/neural_network_r.png" alt="" style="width: 700px; height=auto;"> 
	<figcaption>ROC Curve for binary setting</figcaption>
    <figcaption>Image Source: <a href="https://www.r-project.org/about.html">R</a></figcaption>
</figure>



## Conclusion

All in all, I am extremely proud of this project and blog post. It took months to finish and, while I could’ve gone very deeply into every facet of the data or models, I really like how it turned out. Hopefully you enjoyed it as well! Of course, that’s not to say that there’s nothing I would change; there were plenty of shortcomings.

The first shortcoming is one that every data scientist has felt at some time or another: not enough data! I would’ve loved for the datasets to be much larger to give the models more to learn from. While that would’ve made the computation time much longer, it would’ve really stretched some of these models and made their strengths or weaknesses more apparent. Along with that, I wish I would’ve compared computational times between R and Python. It would’ve been very interesting to learn which language works faster with which models.

If I were to do this project again, I would’ve manually changed the baseline group for the categorical variables. As they were, the baseline groups were gender_Female, diet_quality_Fair, and internet_quality_Average. I wish I would’ve made diet_quality_Poor and internet_quality_Poor the baseline groups. That would make more logical sense to me and would make the interpretations easier and more understandable. This is a small change, but if interpretation is important to you, this could make a huge difference.

I could have also fine-tuned the hyperparameters in the machine learning models much more - or at all. The only hyperparameter I modified was just the number of trees grown, and that was only to limit computation time. If I actually wanted to push these models, I could’ve spent hours tuning the hyperparameters, but I wanted you to see how well the models do out of the box.

The last shortcoming I’ll discuss is quite a big one. I wish I would’ve used more metrics to assess performance of the models. R^2, RMSE, Accuracy, and AUC are great metrics, don’t get me wrong, but there are dozens of others that I could’ve used as well. It would’ve required a couple more libraries, or a few more lines of code, but it’s never a bad thing to have more ways to measure how well a model fits the data. I recently worked at a corporate internship and I can confidently say that company executives could not care less about how things are done. They tend to only care about the results. They love their performance numbers, so it is always a good idea to include more performance metrics just in case.

It’s always important to look back on your work and realize what you could improve for next time. Some things we may not be able to change, like the data we receive. But some things we absolutely can change, like modifying things for clarity and fine-tuning hyperparameters. Remember to always review your work and look for ways you can be a better data scientist. However, it’s also important to look back on your work and think of what went well! For example, I really liked the 13 models I used. Sure, I could’ve used more, but I think 13 was more than sufficient. I really liked learning how to model more in Python, since I have mostly only used R before. I also really liked the format and content of the blog post. I feel very proud of this project and am so happy I can share it with you!

To reiterate, this project was never intended to be a fight between R and Python to see which coding language is the best. If you walk away from this post thinking that Python is better than R, or vice versa, then I failed. The point of this post was to show you how easy it can be to deploy statistical models in either language, not to prove that one is better or worse than the other. Please reach out to me if you detected any bias for or against either coding language in this post.

I hope you enjoyed reading more about these models in action! My next post will be an analysis of Formula 1. I recently watched F1: The Movie (twice because it was so good) and fell in love with the sport. If this interests you, please watch out for it in the coming weeks! Thank you again for reading. Please reach out to me for any ideas, critiques, or questions!