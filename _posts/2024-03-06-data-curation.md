---
layout: post
title:  "Curating Premier League Data"
date: 2024-03-06
description: A look into how I gathered information on the top 250 players in the English Premier League
image: "/assets/img/ball2.webp"
display_image: false  # change this to true to display the image below the banner 
---
<p class="intro"><span class="dropcap">T</span>his post will dive into how I created a data set of EPL data using web scraping via Selenium. I gathered data on individual players including their name, skill level, potential, position, and estimated transfer value. Follow along this article if you want to curate a similar data set!</p>


## Introduction

I am a huge fan of the English Premier League (also called EPL or "The Prem"), which is widely recognized as the [top league](https://www.globalfootballrankings.com/) for soccer (or “football” if you’re from outside the US) in the world. One of the biggest events that happens each year in the EPL isn’t actually a game; rather, it is the transfer window (in other sports, this may be known as the trade deadline or trade window). One of the biggest talking points throughout the season is which players will get transferred to where.

<figure>
	<img src="{{site.url}}/{{site.baseurl}}/assets/img/prem-poster.jpg" alt=""> 
	<figcaption>The Prem is also the most viewed soccer league in the world</figcaption>
</figure>


### Why This Data Set?

My obsession with EPL has led me to wonder about how much the top players’ Estimated Transfer Values (ETVs) are (how much money players are worth), and also what other variables are correlated with ETV. Keep in mind the date that this article is being written (March 6, 2024). I want to find the answers to the following questions:

1. What is the average ETV of the top 250 players in the league (measured by ETV)?
2. What’s the correlation between skill and ETV?
3. What are the most popular positions of the top 250 players in the league?


### Finding the Source

To find answers to these questions, I turned to [FootballTransfers](https://www.footballtransfers.com/us), a public, multi-language transfers source in the soccer world. They deliver accurate, algorithmically-driven valuations of players all over the world.

<figure>
	<img src="{{site.url}}/{{site.baseurl}}/assets/img/football-transfers.png" alt=""> 
	<figcaption>FootballTransfers delivers valuations of players, transfer news, and articles about the soccer world</figcaption>
</figure>

This post will not dive into the specific answers to these questions of interest; that will be done in a later post. Rather, it will focus on how I was able to gather data from FootballTransfers to perform an EDA in the future.


## Data Collection

### Tools

<figure>
	<img src="{{site.url}}/{{site.baseurl}}/assets/img/selenium.webp" alt=""> 
</figure>

I used Python and Selenium to gather this data. While Selenium can be used on several different browsers, I used the Chrome WebDriver. For a review of Selenium, including what it is and how it works, consider reading [this article](https://www.browserstack.com/guide/selenium-webdriver-tutorial#:~:text=in%20Selenium%204%3F-,What%20is%20Selenium%20WebDriver%3F,language%20to%20create%20test%20scripts.) from BrowserStack.


### Step 1: Find the Data

We need to first navigate to where the Premier League players are on the FootballTransers website. After clicking on the "Players" tab at the top of the page, you will change the "Leagues & Cups" filter to "Premier League."

<figure>
	<img src="{{site.url}}/{{site.baseurl}}/assets/img/prem-players.png" alt=""> 
	<figcaption>Players > Filters > Leagues & Cups > Premier League will yield this table</figcaption>
</figure>

This will show us 21 pages of players, with 25 on each, ranked by their estimated transfer value (highest first).


### Step 2: Inspect Page

The next step is learn how the page is set up, so that we can begin scraping data from it.

<figure>
	<img src="{{site.url}}/{{site.baseurl}}/assets/img/player-table.png" alt=""> 
	<figcaption>This is the container for all the players' information</figcaption>
</figure>

Here we can see the part of the page that contains the information for all players. This will be necessary later to create an object from which we can iteratively extract data player-by-player.

<figure>
	<img src="{{site.url}}/{{site.baseurl}}/assets/img/player-info.png" alt=""> 
	<figcaption>This holds the player name and position</figcaption>
</figure>

Here we can see the part of the page that contains the player's name and position. Just as before, we will use this shortly to extract these variables.

<figure>
	<img src="{{site.url}}/{{site.baseurl}}/assets/img/skill-pot.png" alt=""> 
	<figcaption>This has the player's skill and potential</figcaption>
</figure>

This is what will allow us to extract the player's skill level and potential. It is the second to last variable which we will be extracting.

<figure>
	<img src="{{site.url}}/{{site.baseurl}}/assets/img/player-etv.png" alt=""> 
	<figcaption>This has the player's ETV</figcaption>
</figure>

Finally, the last variable which we will scrape from this website is the player's Estimated Transfer Value (ETV).


### Step 3: Find Information for One Player

Of course, there are several different ways data can be scraped from the web. You can use Selenium, [BeautifulSoup](https://beautiful-soup-4.readthedocs.io/en/latest/), [Requests](https://pypi.org/project/requests/), or an [API](https://aws.amazon.com/what-is/api/#:~:text=API%20stands%20for%20Application%20Programming,other%20using%20requests%20and%20responses.). Moreover, there isn't just one way to find the information on the page. The images above show different classes that contain the necessary information, but most data is held in multiple locations. Use the approach that works best for you!

A recommendation I have for web scraping is to first experiment by trying to just find the information for one element--in this case, a player. Find the container that just holds one player, and see if you can extract his name, position, skill, potential, and ETV.

Here is some example code which allows us to find a player's potential:

{%- highlight python -%}
player_pot = players.find_element(By.XPATH, ".//div[contains(@class, 'table-skill')]").text.split('\n')[1]
{%- endhighlight -%}

The container that holds this player's information is named "players." From this container, we are finding the first "div" tag that has a class with the name "table-skill." After we have located the HTML, we convert it to text. Read [this article](https://www.w3schools.com/html/) for a review of HTML.

However, keep in mind from the above image that this particular tag and class contains both the skill level and potential of a player. This is why we split the text and take the second element in that list. After saving it as an object called "player_pot" we can check to make sure it gives us what we want:

{%- highlight python -%}
player_pot
#=> 100.0
{%- endhighlight -%}

Sure enough, our code is correct! You can use this as a model to find the other variables for the player. Once you find the correct tag and class, you can scrape whatever you need!


### Step 4: Build for Loop

Now that we have the code to find all the variables for one player, the last step is to build a [for loop](https://www.w3schools.com/python/python_for_loops.asp) to iteratively find the information for all players at once.

########Do I put my for loop here or just explain it conceptually?


### Ethics

FootballTransfer is a public site that does not contain proprietary information. On their Social Media sites, they say that people are free to use information from their company as they please, as long as they do not receive any monetary gain from it.


### Conclusion

<figure>
	<img src="{{site.url}}/{{site.baseurl}}/assets/img/prem-trophy.png" alt=""> 
	<figcaption>This EPL trophy-one of the most coveted cups in the soccer world</figcaption>
</figure>

This blog post reviewed how I curated a data set of the top 250 EPL players, containing their names, skill levels, potentials, positions, and ETVs. Using Selenium, we were able to build a web scraper that extracted all our desired variables from the webpage.

Stay tuned for my next post when I will perform an exploratory EDA on this data! Now that we have the data set, we will be able to answer the questions of interest!