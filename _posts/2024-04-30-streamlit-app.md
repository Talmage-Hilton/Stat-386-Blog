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

To kick things off, what will be of most help in building your own web app is the [documentation]("https://docs.streamlit.io/") that Streamlit already provides. While I will be going into some specific options that I used in my own Streamlit App, the Documentation supplies everything else you may desire.


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

The next code chunk creates a [Header]("https://docs.streamlit.io/develop/api-reference/text/st.header") and a [Caption]("https://docs.streamlit.io/develop/api-reference/text/st.caption"):

{%- highlight python -%}
st.header("Description")
st.caption('On the left sidebar, you can select a single player and see their stats. On the first tab labeled "Player vs. Player" you can compare two players to each others based on the stat you choose. Under the second tab labeled "Choose Your Own Adventure" you can choose any two variables to see how they are related. Finally, under the third tab labeled "Correlation Matrix" you can see the correlation between each of the numeric variables.')
st.caption("For more information about this data, please check out my [GitHub Respository]('https://github.com/Talmage-Hilton/data-curation-project'), as well as my [Data Science Blog]('https://talmage-hilton.github.io/Stat-386-Blog/').")
{%- endhighlight -%}


##### Running the App Locally

Here again, we turn to Streamlit's Documentation, under the [Running your app]("https://docs.streamlit.io/develop/concepts/architecture/run-your-app") section, to learn how to do this. To deploy the app locally (not on the internet), navigate to your terminal on your local device. Make sure the environment you're using is the environment you're writing the Python file in. Then simply type the following into the terminal, while changing "your_script" to the name of your Python file:

{%- highlight python -%}
streamlit run your_script.py
{%- endhighlight -%}

This all results in an app that looks like this:

<figure>
	<img src="{{site.url}}/{{site.baseurl}}/assets/img/streamlit-ss.png" alt="" style="width: 1000px; height=auto;"> 
	<figcaption>A simple introduction to a Streamlit App</figcaption>
</figure>

Not bad for just writing a few sentences in a Python file! Now, though, we'll really get to see what this app can do.


### Utilizing Streamlit

After reading in my data to my Python file, loading any libraries necessary (I will be using [pandas]("https://pandas.pydata.org/"), [plotly.express]("https://plotly.com/python/plotly-express/"), and [streamlit]("https://pypi.org/project/streamlit/") for this web app), and creating a dictionary to rename some of the variables in my data set, let's say I wanted to create some tabs on my web app. On the first tab I want to be able to compare two players to each other to see how they stack up in different categories. On the second tab I want the user to plot any two variables against each other and see how they're connected. On the third tab I simply want to display a correlation matrix of the continuous variables. To create the tabs first, I use this code:

{%- highlight python -%}
tab1, tab2, tab3 = st.tabs(['Player vs. Player', 'Choose Your Own Adventure', 'Correlation Matrix'])
{%- endhighlight -%}

Now I want to create a way for comparing any two players the user selects against each other in a certain category. I will be using [selectbox]("https://docs.streamlit.io/develop/api-reference/widgets/st.selectbox") a lot throughout the rest of the code. Let's take a look at one way to do this:

{%- highlight python -%}
with tab1:
    player1 = st.selectbox('Select First Player', df['name'])
    player2 = st.selectbox('Select Second Player', df['name'])
    numeric_variables = ['skill', 'pot', 'etv']
    variable = st.selectbox('Select Variable to Compare', [variable_names[key] for key in numeric_variables])

    column_name = reverse_variable_names[variable]

    # Plot comparison
    player1_value = df[df['name'] == player1][column_name].values[0]
    player2_value = df[df['name'] == player2][column_name].values[0]

    data = {
        'Player': [player1, player2],
        variable: [player1_value, player2_value]
    }
    fig_comparison = px.bar(data, x='Player', y=variable, color='Player', labels={'Player': 'Player', variable: variable}, title=f'{player1} vs {player2} - {variable} Comparison')
    st.plotly_chart(fig_comparison)
{%- endhighlight -%}

The second tab is really where we have some fun. This is where the user can choose any two variables to plot against each other. If both variables chosen are numeric, then it will create a scatterplot. If the x-axis is chosen to be a categorical variable, then you must choose a numeric variable as the y-axis, and a bar chart appears. The height of the bars will be the average (mean) measure of the numeric variable for each category of the x-axis. This is how my code turned out:

{%- highlight python -%}
with tab2:
    x_variable = st.selectbox('Select X-Axis Variable', [variable_names[key] for key in variable_names.keys()])
    
    # Filter numeric variables for y_variable selectbox
    numeric_variables = ['Skill Level', 'Potential Level', 'Estimated Transfer Value']
    y_variable = st.selectbox('Select Y-Axis Variable', numeric_variables)
    
    if x_variable in ['Specific Position', 'General Position']:  # Check if x-axis variable is categorical
        if y_variable:  # Check if y-axis variable is selected
            x_column = reverse_variable_names[x_variable]  # Map x_variable to column name in the DataFrame
            y_column = reverse_variable_names[y_variable]  # Map y_variable to column name in the DataFrame
            chart_data = df.groupby(x_column)[y_column].mean().reset_index()
            fig = px.bar(chart_data, x=x_column, y=y_column, labels={x_column: x_variable, y_column: y_variable}, title=f'{y_variable} by {x_variable}')
            st.plotly_chart(fig)
        else:
            st.write("Please select a numeric variable for the Y-Axis.")
    else:
        if x_variable != y_variable:  # Ensure x and y variables are not the same
            x_label = x_variable
            y_label = y_variable
            x_column = reverse_variable_names[x_variable]  # Map x_variable to column name in the DataFrame
            y_column = reverse_variable_names[y_variable]  # Map y_variable to column name in the DataFrame
            fig = px.scatter(df, x=x_column, y=y_column, hover_data=['name', 'etv', 'pot', 'general_position'], labels={x_column: x_label, y_column: y_label})
            st.plotly_chart(fig)
        else:
            st.write("Please select different variables for X-Axis and Y-Axis.")
{%- endhighlight -%}

Finally, a correlation matrix. I went over making this in my [EDA blog post]("https://talmage-hilton.github.io/Stat-386-Blog/blog/epl-eda/"), so I will simply provide the code here:

{%- highlight python -%}
with tab3:
    df_copy = df.iloc[:, [4, 5, 6]].copy()
    # Calculate correlation matrix
    corr_matrix = df_copy.corr()
    # Create heatmap
    fig_cor = px.imshow(corr_matrix,
                        labels=dict(x="Variables", y="Variables", color="Correlation"),
                        x=corr_matrix.columns,
                        y=corr_matrix.columns,
                        color_continuous_scale='YlGnBu',
                        zmin=-1, zmax=1)
    # Add annotations
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            fig_cor.add_annotation(x=i, y=j, text=f"{corr_matrix.iloc[i, j]:.2f}",
                                   showarrow=False, font=dict(color='white'))
    # Update layout
    fig_cor.update_layout(title='Correlation Matrix', xaxis_title='', yaxis_title='')
    st.plotly_chart(fig_cor)
{%- endhighlight -%}

I feel like my app is still lacking something. Let's add a [sidebar]("https://docs.streamlit.io/develop/api-reference/layout/st.sidebar") where we can choose a single player to get an overview of their information.

{%- highlight python -%}
with st.sidebar:
    selected_player = st.sidebar.selectbox('Select a Player', df['name'])

    player_stats = df[df['name'] == selected_player].squeeze()

    st.write(f"## {selected_player}'s Stats")
    st.write(f"**Position:** {player_stats['general_position']}")
    st.write(f"**Skill:** {player_stats['skill']}")
    st.write(f"**Potential:** {player_stats['pot']}")
    st.write(f"**ETV:** {player_stats['etv']}")
{%- endhighlight -%}

This all culminates in a final product that looks like this:

<figure>
	<img src="{{site.url}}/{{site.baseurl}}/assets/img/final-streamlit-ss.png" alt="" style="width: 1000px; height=auto;"> 
	<figcaption>A finalized Streamlit Web App</figcaption>
</figure>

Of course, there are dozens of other tools Streamlit offers. [Latex]("https://docs.streamlit.io/develop/api-reference/text/st.latex"), [Dataframes]("https://docs.streamlit.io/develop/api-reference/data/st.dataframe"), [Submit Buttons]("https://docs.streamlit.io/develop/api-reference/execution-flow/st.form_submit_button"), [Check Boxes]("https://docs.streamlit.io/develop/api-reference/widgets/st.checkbox"), and [Sliders]("https://docs.streamlit.io/develop/api-reference/widgets/st.slider") just begin to scratch the surface. Couple this with all the functionality of [plotly]("https://plotly.com/"), if you can dream it, you can turn it into a spectacular web app!


### Deploying the App

First, we need to create a GitHub repo which will house all the files we need to deploy the web app. If you are unfamiliar with creating a GitHub repo, please review the following [explanation]("https://docs.github.com/en/repositories/creating-and-managing-repositories/quickstart-for-repositories") from GitHub Docs.

In this repo, you will need to include your Python (.py) file, your data (if you are reading it in from a location on your local device), and a [requirements.txt]("https://www.freecodecamp.org/news/python-requirementstxt-explained/") file. In this `requirements.txt` file, you just need to put the names of the libraries used. For my example, that is just `pandas`, `plotly`, and `streamlit`.

To deploy the app on the internet (no longer just locally), you will need to use the Streamlit Community Cloud account you created earlier. After going into your account, click "New app" in the top right of the screen. Then find the GitHub repository you just created. Ensure that it is using the correct branch and that the "Main file path" is listed as the .py file that includes all the Streamlit code. In just a few moments, your app will be ready to use by anyone, anywhere in the world!


### Final Thoughts

I love Streamlit. I was astounded when I first started using it and realized just how simple it was to use. It was very intuitive and I had a working app within minutes. There are a lot of things you can do with Streamlit. This blog post and my app only take advantage of a few of them.

I hope that this blog post helped and that you were able to gain a valuable new skill! If you have any questions, please feel free to reach out to my on my socials. Good luck coding!