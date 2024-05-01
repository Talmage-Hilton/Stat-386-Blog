---
layout: post
title:  "Creating a Streamlit Web App"
date: 2024-04-30
description: A simple explanation of how I created a Streamlit Web Application
image: "/assets/img/web-app.jpg"
display_image: false  # change this to true to display the image below the banner 
---
<p class="intro"><span class="dropcap">T</span>his post will explore the creation of a Streamlit Web Application. Streamlit is a very intuitive tool to build and deploy data applications as simply as possible. Cover image source: <a href="https://www.koombea.com/blog/website-vs-web-application/">koombea</a></p>


### Introduction

[Streamlit]("https://streamlit.io/") is a free and open-source framework designed to quickly build and share data web applications. What is a web application, you might ask? From [TechTarget]("https://www.techtarget.com/searchsoftwarequality/definition/Web-application-Web-app"), a web application is an interactive program stored on a remote server that is delivered over the internet through a browser interface.

<figure>
	<img src="{{site.url}}/{{site.baseurl}}/assets/img/streamlit-logo.jpeg" alt="" style="width: 500px; height=auto;"> 
	<figcaption>Streamlit is an open-source framework for deploying web apps</figcaption>
    <figcaption>Image Source: <a href="https://www.analyticsvidhya.com/blog/2021/06/build-web-app-instantly-for-machine-learning-using-streamlit/">Analytics Vidhya</a></figcaption>
</figure>

The best thing about Streamlit is that it requires less code than other web app tools. In fact, it is such little code that you can have your app up and running within minutes! After reading this blog post, you can have a working web app to display whatever kind of information you want.


### Documentation

To kick things off, what will be of most help in building your own web app is the [documentation](https://docs.streamlit.io/) that Streamlit already provides. While I will be going into some specific options that I used in my own Streamlit App, the Documentation supplies everything else you may desire.


### Essentials

As stated previously, Streamlit is extremely simple and quick to use, but there still are a few things that you need to know before getting started.

The first is that Streamlit uses [Python]("https://www.python.org/"). Chances are you're already quite familiar with Python if you're reading this blog post, but if not, here is a great [tutorial video]("https://www.youtube.com/watch?v=kqtD5dpn9C8") from [Programming with Mosh]("https://www.youtube.com/@programmingwithmosh") (the rest of the videos on his profile are also excellent).

Another essential thing to know is that you must use a .py file, not a .ipynb file. [Jupyter/IPython Notebooks]("https://ipython.org/notebook.html") are very helpful to see code and output on the fly, but for making a Streamlit App, you need to use a .py file.

The last thing to know is that you will need a [Streamlit Community Cloud]("https://streamlit.io/cloud") account to deploy your app. You can sign up and connect your GitHub account to your Community Cloud account. I highly recommend connecting your GitHub account, as that will make everything much easier in the future!


### Simple Example

Using the documentation in the [API Reference]("https://docs.streamlit.io/develop/api-reference") section on Streamlit's website, we see how this can be done. I created this web app based on my EPL data that I curated and performed EDA on. If you have not read those blog posts, here is the [data curation post]("https://talmage-hilton.github.io/Stat-386-Blog/blog/data-curation/") and [EDA post]("https://talmage-hilton.github.io/Stat-386-Blog/blog/epl-eda/"). The following code creates a [Title]("https://docs.streamlit.io/develop/api-reference/text/st.title") and [Writing]("https://docs.streamlit.io/develop/api-reference/write-magic/st.write"):

{%- highlight python -%}
st.title('English Premier League Players')
st.write("Welcome to a web app to explore information about the top 250 English Premier League players! You may interact with everything on the app. Look around, stay a while, and enjoy!")
{%- endhighlight -%}

The next code chunk creates a [Header]("https://docs.streamlit.io/develop/api-reference/text/st.header"), and [Caption]("https://docs.streamlit.io/develop/api-reference/text/st.caption"):

{%- highlight python -%}
st.header("Description")
st.caption('On the left sidebar, you can select a single player and see their stats. On the first tab labeled "Player vs. Player" you can compare two players to each others based on the stat you choose. Under the second tab labeled "Choose Your Own Adventure" you can choose any two variables to see how they are related. Finally, under the third tab labeled "Correlation Matrix" you can see the correlation between each of the numeric variables.')
st.caption("For more information about this data, please check out my [GitHub Respository]('https://github.com/Talmage-Hilton/data-curation-project'), as well as my [Data Science Blog]('https://talmage-hilton.github.io/Stat-386-Blog/').")
{%- endhighlight -%}


##### Running the App

Here again, we turn to Streamlit's Documentation, under the [Running your app]("https://docs.streamlit.io/develop/concepts/architecture/run-your-app") section, to learn how to do this. To deploy the app locally (not on the internet), navigate to your terminal on your local device. Make sure the environment you're using is the environment you're writing the Python file in. Then simply type the following into the terminal, while changing "your_script" to the name of your Python file:

{%- highlight python -%}
streamlit run your_script.py
{%- endhighlight -%}

This all results in an app that looks like this:

<figure>
	<img src="{{site.url}}/{{site.baseurl}}/assets/img/streamlit-ss.png" alt="" style="width: 500px; height=auto;"> 
	<figcaption>A simple introduction to a Streamlit App</figcaption>
</figure>

Not bad for just writing a few sentences in a Python file! Now, though, we'll really get to see what this app can do.