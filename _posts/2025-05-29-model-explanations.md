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
y_i = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \dots + \beta_p x_{ip} + \varepsilon_i
$$

<ul>
  <li>\( y_i \) is the value of the response variable for observation \( i \)</li>
  <li>\( x_{ij} \) is the value of predictor \( j \) for observation \( i \)</li>
  <li>\( \beta_0 \) is the intercept term (the average value of \( y \) when all predictors are 0)</li>
  <li>\( \beta_j \) is the slope coefficient for predictor \( x_j \)</li>
  <li>\( \varepsilon_i \) is the error term for observation \( i \), capturing the noise/unexplained variation</li>
</ul>
{% endraw %}

We can see in the construction of the model that the average change in y (the response) changes proportionally to x (the predictors) changing. In other words, everything is added together, and that's why it's called linear regression. Here is where we begin to learn the strengths of OLS linear regression. It is so simple to interpret and understand how each predictor impacts the model. If you change a predictor by x, it changes y by beta. Other strengths are that it is very computationally efficient and takes just moments to run on a machine, you can easily plot and visualize the response's nature, and you can perform inference and prediction with it. It does, however, have some weaknesses. For example, it doesn't allow for a categorical response variable, it doesn't do any variable selection, it doesn't allow for more predictors than there are observations, and it requires strict adherence to the model assumptions in order to be valid.

Other notes are that if the relationship between the response and a predictor is not linear (curved or wiggly), then a different model would be appropriate. Those models will be discussed shortly.


#### LASSO

Now that we understand linear regression, it makes all the rest of the linear models much easier to understand. LASSO (Least Absolute Shrinkage and Selection Operator) models are a type of linear regression that add a penalty which can shrink some beta coefficients all the way to zero. In effect, it performs variable selection to lower the amount of predictor variables in the model. LASSO works by solving the following optimization problem:

{% raw %}
$$
\hat{\beta} = \arg\min_{\beta} \left\{ \sum_{i=1}^{n} \left( y_i - \beta_0 - \sum_{j=1}^{p} \beta_j x_{ij} \right)^2 + \lambda \sum_{j=1}^{p} |\beta_j| \right\}
$$

<ul>
  <li>The first term: \( \sum \left( y_i - \hat{y}_i \right)^2 \) is just like linear regression (minimize residual sum of squares)</li>
  <li>The second term: \( \lambda \sum_{j=1}^{p} |\beta_j| \) is the LASSO penalty</li>
  <li>\( \lambda \): the tuning parameter that controls how strong the penalty is</li>
  <li>When \( \lambda = 0 \), you get standard linear regression (no penalty)</li>
  <li>When \( \lambda \) increases, the penalty becomes stronger, eventually shrinking some coefficients to exactly zero</li>
</ul>
{% endraw %}

Why would this be useful? As I explained a moment ago, traditional linear regression does not allow for there to be more predictors than there are observations. LASSO overcomes this by removing some predictor variables (called regularization) so that linear regression can be run. There also may be a problem with [multicollinearity]("https://www.investopedia.com/terms/m/multicollinearity.asp#:~:text=Multicollinearity%20is%20a%20statistical%20concept,coefficient%20is%20%2B%2F%2D%201.0."), which is when some predictor variables are highly correlated with each other (a person's height and their weight, for example). This can lead to erroneous results, but LASSO will remove some of the highly correlated variables in order to overcome the multicollinearity problem. Finally, sometimes we just want fewer predictor variables in the model. Maybe we're given a dataset with dozens or even hundreds of predictors. In order to only fit the data with the variables of most importance, we can employ LASSO. This is called variable selection. A word of caution with this, however, is that sometimes LASSO will remove a variable that we do care about. If we have a dataset about car crashes, and we want to see how car size impacts the damage of a car crash, we need to be careful if LASSO removes the car size variable, because then we will lose the information we care about. As with any of these models, LASSO should not be blindly employed with no checks or balances.

There are other variable selection/regularization models and techniques out there, but LASSO is typically seen as the best (and my personal favorite) option. It is fast to implement, simple to understand if you have an understanding of traditional linear regression, only requires the linearity assumption, has standard coefficient definitions (like in linear regression), has individual variable significance (using [bootstrapping]("https://www.datacamp.com/tutorial/bootstrapping")), relative variable importance, and you can plot and visualize the response. Weaknesses include a more complicated model than linear regression, you have to be careful of the variables it's removing, and it can require bootstrapping depending on the research questions. Overall, LASSO is a very important tool that can always be used just to see what variables may be more/less important to the model.



#### Polynomial Regression

Yes, polynomial regression is a linear model. Don't worry if you’re questioning everything you’ve been taught or questioning why you started studying statistics; I promise it will make more sense in a few minutes. Recall that everything was added together in the linear regression formula. That summing up of the effects of each predictor is what makes it a linear model. In polynomial regression, we are still adding up all the terms, it's just that we are modeling the relationship between the response and the predictors as an n^th-degree polynomial. A degree-3 model would look like this:

{% raw %}
$$
y_i = \beta_0 + \beta_1 x_i + \beta_2 x_i^2 + \beta_3 x_i^3 + \varepsilon_i
$$

<ul>
  <li>\( \beta_0, \beta_1, \beta_2, \beta_3 \) are the coefficients to estimate</li>
  <li>\( x_i, x_i^2, x_i^3 \) are transformed versions of the original predictor \( x_i \)</li>
  <li>\( \varepsilon_i \) is the usual error term</li>
</ul>
{% endraw %}

I stated earlier that linear regression cannot handle wiggly or curvy relationships between the response and predictors. Polynomial regression overcomes this because you can add polynomial terms to fit that curvature. Once you've included the degree of the polynomial for the predictor, the rest is completed just like traditional linear regression.

Strengths of polynomial regression are that it can handle both linear and non-linear relationships, retains individual and relative variable significance, can plot the response, is fast to run, and you can perform inference and prediction with it. Weaknesses are that it is somewhat complicated, you have to be careful about the degree of the polynomial you choose, it requires the LINE assumptions, interpreting the polynomial terms can be convoluted, and extrapolation would be a very bad idea. Polynomial terms shoot off to positive/negative infinity beyond the boundaries of the data, so if you wanted to predict the response given a value of a predictor, you will get very poor results. I created a simple example to show that good fits to the data can still lead to very bad predictions:

<figure>
	<img src="{{site.url}}/{{site.baseurl}}/assets/img/poly_fit.png" alt="" style="width: 500px; height=auto;"> 
	<figcaption>Predictions shoot off to neg/pos infinity beyond the range of the data in Polynomial Regression</figcaption>
    <figcaption>Image Source: <a href="https://www.r-project.org/about.html">R</a></figcaption>
</figure>


#### Natural Splines

Natural splines are very similar to polynomial regression, with a few notable exceptions. In short, natural splines:

- Break the range of the predictor variable into sections at knots
- Fit cubic polynomials in each section
- Ensure the pieces are smoothly connected (differentiable at knots)
- Add a natural constraint where the function becomes linear beyond the boundary knots

A mathematical explanation is now provided for a natural spline with K internal knots:

{% raw %}
$$
f(x) = \beta_0 + \beta_1 x + \sum_{j=1}^{K} \theta_j N_j(x)
$$

<ul>
  <li>\( \beta_0 \) is the intercept term</li>
  <li>\( \beta_1 x \) is the linear term</li>
  <li>\( N_j(x) \) are basis functions built from transformations involving the knots</li>
  <li>\( \theta_j \) are the coefficients of the spline terms</li>
</ul>
{% endraw %}

One standard way to construct the natural spline basis is the [truncated power basis]("https://documentation.sas.com/doc/en/pgmsascdc/9.4_3.4/statug/statug_introcom_sect022.htm") with natural constraints. We will define the truncated function as follows:

{% raw %}
$$
(x - \xi_j)^3_+ = 
\begin{cases}
(x - \xi_j)^3 & \text{if } x > \xi_j \\
0 & \text{otherwise}
\end{cases}
$$

<ul>
  <li>\( \xi_j \) are the internal knots</li>
</ul>
{% endraw %}

Then the full spline function using the truncated power basis (with natural constraints applied to reduce the degrees of freedom) looks like:

{% raw %}
$$
f(x) = \beta_0 + \beta_1 x + \sum_{j=1}^{K-2} \theta_j d_j(x)
$$

<ul>
  <li>\( d_j(x) \) are constructed from the truncated cubic functions in a special way that enforces the natural spline conditions (linearity beyond boundary knots)</li>
</ul>
{% endraw %}

These basis functions ensure that:

- The function is continuous
- The first and second derivatives are continuous
- The function is linear outside the boundary knots (natural condition)

By breaking the range into sections, we can get very accurate overall fits to the data. The constraint overcomes the extrapolation issue from polynomial regression. Instead of the predictions shooting off to positive/negative infinity beyond the range of the data, natural splines continue on linearly. You still absolutely have to be careful when extrapolating, especially if you choose many knots, but the predictions will be much more stable regardless.

<figure>
	<img src="{{site.url}}/{{site.baseurl}}/assets/img/natural_spline.png" alt="" style="width: 500px; height=auto;"> 
	<figcaption>Manually placed knots in a Natural Splines model</figcaption>
    <figcaption>Image Source: <a href="https://www.r-project.org/about.html">R</a></figcaption>
</figure>

My preferred method to implement natural splines in R is the 'splines' package. It will place default knots if you do not specify knot locations. These default placements are often not great. I like to fit the natural splines model just to the predictor of interest and the response, then plot the two against each other, overlay the spline with the knots, and use cross-validation (by minimizing some criteria of choice) or subjective graphical approaches to find the best fit. Then I do this for any other non-linear relationships there may be and fit the whole model to all the data, using the knots I found.

Strengths of natural splines are that it handles non-linear data very well, is still fairly simple to understand, fast to implement, and handles prediction much better than polynomial regression. Weaknesses are that knots must be selected carefully, the model is prone to overfitting and underfitting, requires the LINE assumptions, does not have the standard coefficient definitions, and requires some work to find individual variable significance.



#### GAM

Generalized Additive Models are my personal favorite of the "non-linear" linear models. In a very general sense, GAMs are a flexible extension of linear regression that allows for non-linear relationships in the data. GAMs are convenient because they don't pick the best method in a one-size-fits-all way, but rather let you choose the smoothing method — and then the GAM chooses how smooth it should be based on the data. The most common smoothing methods are splines, LOESS, or kernel smoothers. The general form of a GAM is:

{% raw %}
$$
g(\mathbb{E}[Y]) = \beta_0 + f_1(X_1) + f_2(X_2) + \dots + f_p(X_p)
$$

<ul>
  <li>\( \mathbb{E}[Y] \) is the expected value of the response \( Y \)</li>
  <li>\( g(\cdot) \) is a link function (similar to GLMs: identity for regression, logit for binary classification, etc.)</li>
  <li>\( \beta_0 \) is the intercept</li>
  <li>\( f_j(X_j) \) is a smooth, flexible function of predictor \( X_j \) (these functions are usually splines, LOESS, or kernel smoothers)</li>
  <li>Each \( f_j(X_j) \) is estimated from the data, subject to smoothness constraints (to avoid overfitting)</li>
</ul>
{% endraw %}

The GAM balances goodness of fit and smoothness to decide how much to smooth. It minimizes a penalized loss function:

{% raw %}
$$
\text{Loss} = \text{Residual Sum of Squares} + \lambda \int \left[ f_j''(x) \right]^2 \, dx
$$

<ul>
  <li>The second derivative \( f_j''(x) \) measures how curvy the function is</li>
  <li>\( \lambda \) is the smoothing parameter</li>
  <li>\( \lambda \) is automatically chosen using methods like Generalized Cross-Validation or Restricted Maximum Likelihood</li>
</ul>
{% endraw %}

Strengths of GAMs are that they allow for non-linearity in data, are quick to implement, have individual and relative variable significance, and handle inference and prediction well. Weaknesses are that they still require the LINE assumptions, do not have the standard coefficient definitions, and it's fairly complicated to understand how exactly they are working.



#### Logistic Regression

Logistic regression is the final linear model we will discuss. The key difference between logistic and linear regression is that logistic regression allows for a binary (yes/no or 0/1) response variable. It does this by modeling the probability of the response being a success (yes or 1). Mathematically, this looks like the following:

{% raw %}
$$
\log\left(\dfrac{p_i}{1 - p_i}\right) = \beta_0 + \beta_1 x_{i1} + \dots + \beta_p x_{ip}
$$

<ul>
  <li>\( p_i \) is the probability that \( y_i = 1 \)</li>
  <li>\( \dfrac{p_i}{1 - p_i} \) is the odds</li>
  <li>The log of the odds is called the logit function</li>
</ul>
{% endraw %}

It looks very similar to the linear regression formula except that we are modeling the log odds of the response instead of the response directly. This makes it linear in the log odds, not in the probability itself. What is going on under the hood is a little different as well: instead of using ordinary least squares, it uses maximum likelihood estimation [MLE]("https://medium.com/data-science/probability-concepts-explained-maximum-likelihood-estimation-c7b4342fdbb1") to find the set of beta values that maximize the probability of observing the data that we have.

Interpreting the coefficients takes a bit more work, but is pretty straightforward once you get the hang of it. [This article]("https://www.displayr.com/how-to-interpret-logistic-regression-coefficients/") by Displayr does a great job of explaining how to interpret coefficients in logistic regression.

The last thing I want to highlight is that logistic regression can be combined with any of the previous models (besides traditional linear regression) to model a binary response instead of a continuous one. In most R functions, you can just specify the family="binomial" argument to do this.

Strengths of logistic regression are that it handles a binary response, is fast and simple to understand, has individual and relative variable significance, and can handle inference and prediction well. Weaknesses are that it requires linearity in the log odds (monotonic in x vs. y) and also independence, requires some work to interpret coefficients, and has all the other weaknesses as traditional linear regression.



### Machine Learning Models

Machine learning models differ from linear models in that they don't rely on an additive structure, they have fewer assumptions, often less interpretability (black box nature), more flexibility, and tend to be used more for prediction and performance rather than explanation and inference. Both linear and machine learning models have their place, it just depends on what your goals are.


#### K Nearest Neighbors

K Nearest Neighbors (KNN) is the first machine learning model we will discuss. KNN works by choosing a number of neighbors (k) to consider. Then, for a new observation, it calculates the distance (usually Euclidean) to all points in the dataset from that observation, selects the k points in the training set that are closest, then assigns the class that occurs the most frequently among the neighbors. KNN can be used to predict a binary or continuous response. For example, we could predict that a new observation is either a cat or a dog (binary) based on some features, or we could predict a house's price (continuous) based on some features. That latter example, about using KNN to predict house prices, is extremely common in practice. Housing prices could be calculated by taking nearby houses with the same number of bedrooms, bathrooms, floors, with similar square footage, age, etc., and then averaging the prices of those homes to predict the price of the new house of interest.

The mathematical way KNN works is as follows:

Let's say you want to predict the class of a new data point \( x_0 \).

1. Compute distance between \( x_0 \) and all training points \( x_1, x_2, \dots, x_n \).

2. Pick the \( k \) points with the **smallest distances**.

3. For classification:

\[
\hat{y}_0 = \text{mode} \left\{ y_{(1)}, y_{(2)}, \dots, y_{(k)} \right\}
\]

4. For regression:

\[
\hat{y}_0 = \frac{1}{k} \sum_{i=1}^{k} 1^k y_{(i)}
\]

where \( y_{(i)} \) is the outcome of the \( i \)-th nearest neighbor.

Strengths of KNN are that it is very simple and intuitive, requires no assumptions, can handle a continuous or binary response, has relative variable significance, and can be optimized easily by choosing different values of k. Weaknesses are that it does not regularize or do any variable selection, does not have a smooth fit, and has no standard coefficient definitions nor individual variable significance.



#### Support Vector Machines




#### Decision Tree (CART)




#### Random Forest




#### Boosting




#### BART




#### Neural Network




### Conclusion

