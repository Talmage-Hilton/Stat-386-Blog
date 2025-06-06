---
layout: post
title:  "Explaining Statistical Models"
date: 2025-05-29
description: Explanations of the 13 statistical models I used in my data science project
image: "/assets/img/datascience.jpg"
display_image: false  # change this to true to display the image below the banner 
---
<p class="intro"><span class="dropcap">T</span>his post will explain the 13 (6 linear, 7 machine learning) models I used in my data science project.</p>
<p class="intro">Cover image source: <a href="https://www.informationweek.com/data-management/how-it-supports-the-data-science-operation">Information Weekly</a></p>


### Introduction

My absolute favorite part of data science involves statistical modeling. Getting a dataset, having some questions of interest, building a model, and answering those questions is extremely fun and rewarding. I love that every dataset is different and so each one presents its own unique challenges and insights.

Obviously statistical modeling is nothing without the models. If you're reading this post, I'm sure you have experience with modeling. In this post, we will be going over 13 different statistical models, six of which are linear models, with the other seven being machine learning models. Knowing how to implement these models is great, but if the math and all the work going on under the hood is lost, you will not be as effective as a data scientist. A good data scientist will be able to receive data and a problem, know how to find and assess the issues, find a model that fits the data, and answers the questions of interest.

Some questions of interest deal with individual variable significance, some with relative variable significance, some with interpretation, some purely with prediction, or any combination of those. The models below all have their strengths and weaknesses depending on the situation. What follows is an explanation of those strengths and weaknesses, as well as a fundamental mathematical explanation of how the models function.


### Linear Models


#### Linear Regression

Put simply, linear regression is a model that estimates the relationship between a response variable and one or more predictor variables. That isn't super helpful because any of these models could be described in this way, so let's discuss how linear regression works, why it's called linear regression, and how it works. Linear regression works by finding the line or hyperplane that minimizes the sum of squared differences between the observed and predicted values. If that sounds insane, don't worry! Everyone feels like this when they're first taught it. Essentially, linear regression is going to fit a straight line (technically it’s a plane, but it’s easier to imagine a line) through the cloud of all the data points. If you imagine your data as a cloud of points, you can see how there are infinitely many lines you could draw. Where should the line start and end? How steep or shallow should it be? To answer this, we measure the vertical difference between each point and the line (called a residual), add them all up, and choose the line that makes that sum as small as possible. Hopefully that makes more sense and seems much less daunting. The math required to do this requires linear algebra, which can be tricky, but the general idea is quite straightforward. Below is a simple visualization I made of how linear regression works in practice:

<figure style="text-align: center;">
  <div style="display: flex; gap: 10px; justify-content: center;">
    <img src="{{site.url}}/{{site.baseurl}}/assets/img/reg1.png" alt="" style="width: 30%;">
    <img src="{{site.url}}/{{site.baseurl}}/assets/img/reg2.png" alt="" style="width: 30%;">
    <img src="{{site.url}}/{{site.baseurl}}/assets/img/reg3.png" alt="" style="width: 30%;">
  </div>
  <figcaption style="margin-top: 0.5em;">
    The hyperplane drawn through the cloud of data points in linear regression<br>
    Image Source: <a href="https://www.r-project.org/about.html">R</a>
  </figcaption>
</figure>

Linear regression requires the relationship between the response and the predictors to be linear, independence between observations, and normality and equal variance in the residuals (which we will call the LINE assumptions). Linear regression, as the name would suggest, assumes a relationship between the predictors and the response that is linear in the beta coefficients. This means that a change in the response variable is proportional to a change in the predictors.

The linear regression model is as follows:

{% raw %}
$$
y_i = \beta_0 + \beta_1 \x_{i1} + \beta_2 \x_{i2} + \dots + \beta_p \x_{ip} + \varepsilon_i
$$
{% endraw %}

- \( y_i \) is the value of the response variable for observation \( i \)  
- \( x_{ij} \) is the value of predictor \( j \) for observation \( i \)  
- \( \beta_0 \) is the intercept term (the average value of \( y \) when all predictors are 0)  
- \( \beta_j \) is the slope coefficient for predictor \( x_j \)  
- \( \varepsilon_i \) is the error term for observation \( i \), capturing the noise/unexplained variation

We can see in the construction of the model that the average change in y (the response) changes proportionally to x (the predictors) changing. In other words, everything is added together, and that's why it's called linear regression. Specifically, if we wanted to learn the relationship between one specific predictor and the response, we could explain it as follows: "Holding all the other predictors constant, as x_j increases by one unit, y increases by beta_j units, on average." And here is where we begin to learn the strengths of OLS linear regression. It is so simple to interpret and understand how each predictor impacts the model. If you change a predictor by x, it changes y by beta. Other strengths are that it is very computationally efficient and takes just moments to run on a machine, you can easily plot and visualize the response's nature, and you can perform inference and prediction with it. It does, however, have some weaknesses. For example, it doesn't allow for a categorical response variable, it doesn't do any variable selection, it doesn't allow for more predictors than there are observations, and it requires strict adherence to the model assumptions in order to be valid.

Other notes are that if the relationship between the response and a predictor is not linear (curved or wiggly), then a different model would be appropriate. Those models will be discussed shortly.


#### LASSO




#### Polynomial Regression




#### Nautral Splines




#### GAM




#### Logistic Regression




### Machine Learning Models


#### K Nearest Neighbors




#### Support Vector Machines




#### Decision Tree (CART)




#### Random Forest




#### Boosting




#### BART




#### Neural Network




### Conclusion

