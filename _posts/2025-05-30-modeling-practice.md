---
layout: post
title:  "Modeling Project in Python & R"
date: 2025-05-30
description: A fun data science project I did in both Python and R
image: "/assets/img/R_python.avif"
display_image: false  # change this to true to display the image below the banner 
---
<p class="intro"><span class="dropcap">T</span>his post will be a review of a fun data science project I did to practice statistical modeling in both Python and R.</p>
<p class="intro">Cover image source: <a href="https://www.datacamp.com/blog/python-vs-r-for-data-science-whats-the-difference">datacamp</a></p>


## Introduction

After over a year without posting, we are finally back with another data science blog post! The past year has been the busiest but best year. I got married, started my master's degree in statistics at BYU, and have been neck deep in learning everything I can about data science!

About two years ago I began learning Python, but I haven't used it very much since then. Being a statistics student, we use R almost exclusively in my classes. R is a wonderful tool and it's the first programming language I learned, so it will always hold a special place in my heart, but I know the data science industry primarily relies on Python. So, as an aspiring data scientist, I wanted to relearn Python and do it in the best way I know how - by doing some statistical modeling!

This post will dive into the data I modeled, how to code the models up, and the performance of each. My [last post]("https://talmage-hilton.github.io/Stat-386-Blog/blog/model-explanations/") dove deeply into all the nitty-gritty details of these models. I would highly recommend reading it prior to jumping into this post, especially if you do not already have a strong understanding of how these models function. After that, please come here again, sit back, grab a refreshing beverage, and enjoy reading about my project!


## The Data

I used two different datasets for this project, both found on [Kaggle]("https://www.kaggle.com/"). The first dataset measures students' exam scores based on various predictor variables. This dataset has a continuous response variable (exam_score) and 1000 observations. The continuous predictor variables are age, study_hours_per_day, social_media_hours, netflix_hours, part_time_job, attendance_percentage, sleep_hours, exercise_frequency, mental_health_rating, extracurricular_participation. The categorical predictor variables are gender, diet_quality, and internet_quality. I synthetically added new data to get up to 10,000 rows just to give the models some more data from which to learn.

The second dataset measures whether or not a social media user clicked on an advertisement. Clearly this is a binary response variable. The response variable is Purchased, the continuous predictors are Age and EstimatedSalary, and the categorical predictor is Gender.

I used the advertisements dataset for Logistic Regression, as it requires a binary response. I used the exam scores dataset for all the other linear models. Finally, I used both datasets for all the machine learning models to display their flexibility.


## The Models

I used six linear models and seven machine learning models. The linear models are OLS (Linear Regression), LASSO, Polynomial Regression, Nautral Splines, GAM, and Logistic Regression. The machine learning models I used are K Nearest Neighbors, Support Vector Machines, Decision Tree (CART), Random Forest, Boosting, BART, and Neural Networks.


## Linear Models


### Linear Regression

#### Code



#### Strengths & Weaknesses




### LASSO

#### Code

#### Strengths & Weaknesses


### Polynomial Regression

#### Code

#### Strengths & Weaknesses


### Nautral Splines

#### Code

#### Strengths & Weaknesses


### GAM

#### Code

#### Strengths & Weaknesses


### Logistic Regression

#### Code

#### Strengths & Weaknesses


## Machine Learning Models


### K Nearest Neighbors

#### Code

#### Strengths & Weaknesses


### Support Vector Machines

#### Code

#### Strengths & Weaknesses


### Decision Tree (CART)

#### Code

#### Strengths & Weaknesses


### Random Forest

#### Code

#### Strengths & Weaknesses


### Boosting

#### Code

#### Strengths & Weaknesses


### BART

BART is a wonderful tool for data anlysis. It combines the flexibility of random forests with the strengths of Bayesian statistics. However, I was not able to ever get it to run in Python. There are a multiple packages in R to run BART (some are better than others), but I was never able to find one that actually worked in Python. There were many that ChatGPT told me to try, but I was unsuccessful to say the least. For all the other models, I include both R and Python code. For this one, I will only include R code. There is a way to connect Python to R in order to run R code in the Python environment, but what's the use in that if I have to write the code in R anyway? I will just run it in R. Sure, it may run faster in Python, but with the time it takes to connect R to Python, run it, and do that whole process, it's probably not worth it.

#### Code

#### Strengths & Weaknesses


### Neural Network

#### Code

#### Strengths & Weaknesses



## Conclusion

