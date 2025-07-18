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

#### Python Code

#### R Code

#### Results




### LASSO

#### Python Code

#### R Code

#### Results


### Polynomial Regression

#### Python Code

#### R Code

#### Results


### Nautral Splines

#### Python Code

#### R Code

#### Results


### GAM

#### Python Code

#### R Code

#### Results


### Logistic Regression

#### Python Code

#### R Code

#### Results


## Machine Learning Models


### K Nearest Neighbors

#### Python Code

#### R Code

#### Results


### Support Vector Machines

#### Python Code

#### R Code

#### Results


### Decision Tree (CART)

#### Python Code

#### R Code

#### Results


### Random Forest

#### Python Code

#### R Code

#### Results


### Boosting

#### Python Code

#### R Code

#### Results


### BART

BART is a wonderful tool for data anlysis. It combines the flexibility of random forests with the strengths of Bayesian statistics. However, I was not able to ever get it to run in Python. There are a multiple packages in R to run BART (some are better than others), but I was never able to find one that actually worked in Python. There were many that ChatGPT told me to try, but I was unsuccessful to say the least. For all the other models, I include both R and Python code. For this one, I will only include R code. There is a way to connect Python to R in order to run R code in the Python environment, but what's the use in that if I have to write the code in R anyway? I will just run it in R. Sure, it may run faster in Python, but with the time it takes to connect R to Python, run it, and do that whole process, it's probably not worth it.

#### Python Code

#### R Code

#### Results


### Neural Network

#### Python Code

#### R Code

#### Results



## Conclusion

