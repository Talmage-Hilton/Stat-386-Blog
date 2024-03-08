---
layout: post
title:  "Unveiling The Beautiful Game"
date: 2024-03-06
description: A look into how I gathered information on the top 250 players in the English Premier League
image: "/assets/img/ball2.webp"
display_image: false  # change this to true to display the image below the banner 
---
<p class="intro"><span class="dropcap">T</span>his post will dive into how I created a data set of EPL data using web scraping via Selenium. I gathered data on individual players including their name, skill level, potential, position, and estimated transfer value. Follow along this article if you want to curate a similar data set!</p>


## Introduction

Soccer (or "football" if you prefer that) is affectionately known as "The Beautiful Game." I am a huge fan of the English Premier League (also called EPL or "The Prem"), which is widely recognized as [soccer's top league](https://www.globalfootballrankings.com/) in the world. One of the biggest events that happens each year in the EPL isn’t actually a game; rather, it is the transfer window (in other sports, this may be known as the trade deadline or trade window). One of the biggest talking points throughout the season is which players will get transferred to where.

<figure>
	<img src="{{site.url}}/{{site.baseurl}}/assets/img/prem-poster.jpg" alt="" style="width: 800px; height=auto;"> 
	<figcaption>The Prem is also the most viewed soccer league in the world</figcaption>
</figure>


### Why This Data Set?

My obsession with EPL has led me to wonder about how much the top players’ Estimated Transfer Values (ETVs) are (how much money players are worth), and also what other variables are correlated with ETV. Keep in mind that this article was written on March 6, 2024, so the information about the players is current to that day. I want to find the answers to the following questions:

1. What is the average ETV of the top 250 players in the league (measured by ETV)?
2. What’s the correlation between skill and ETV?
3. What are the most popular positions of the top 250 players in the league?


### Finding the Source

To find answers to these questions, I turned to [FootballTransfers](https://www.footballtransfers.com/us), a public, multi-language transfers source in the soccer world. They deliver accurate, algorithmically-driven valuations of players all over the world.

<figure>
	<img src="{{site.url}}/{{site.baseurl}}/assets/img/football-transfers.png" alt="" style="width: 1000px; height=auto;"> 
	<figcaption>FootballTransfers delivers valuations of players, transfer news, and articles about the soccer world</figcaption>
</figure>

This post will not dive into the specific answers to these questions of interest; that will be done in a later post. Rather, it will focus on how I was able to gather data from FootballTransfers to perform an EDA in the future.


## Data Collection

### Tools

<figure>
	<img src="{{site.url}}/{{site.baseurl}}/assets/img/selenium.webp" alt="" style="width: 500px; height=auto;"> 
</figure>

I used Python and Selenium to gather this data. While Selenium can be used on several different browsers, I used the Chrome WebDriver. For a review of Selenium, including what it is and how it works, consider reading [this article](https://www.browserstack.com/guide/selenium-webdriver-tutorial#:~:text=in%20Selenium%204%3F-,What%20is%20Selenium%20WebDriver%3F,language%20to%20create%20test%20scripts.) from BrowserStack.


### Step 1: Find the Data

We need to first navigate to where the Premier League players are on the FootballTransers website. After clicking on the "Players" tab at the top of the page, you will change the "Leagues & Cups" filter to "Premier League."

<figure>
	<img src="{{site.url}}/{{site.baseurl}}/assets/img/prem-players.png" alt="" style="width: 1000px; height=auto;"> 
	<figcaption>Players > Filters > Leagues & Cups > Premier League will yield this table</figcaption>
</figure>

This will show us 21 pages of players, with 25 on each, ranked by their estimated transfer value.


### Step 2: Inspect Page

The next step is learn how the page is set up, so that we can begin scraping data from it.

<figure>
	<img src="{{site.url}}/{{site.baseurl}}/assets/img/player-table.png" alt="" style="width: 1000px; height=auto;"> 
	<figcaption>This is the container for all the players' information</figcaption>
</figure>

Here we can see the part of the page that contains the information for all players. This will be necessary later to create an object from which we can iteratively extract data player-by-player.

<figure>
	<img src="{{site.url}}/{{site.baseurl}}/assets/img/player-info.png" alt="" style="width: 1000px; height=auto;"> 
	<figcaption>This holds the player name and position</figcaption>
</figure>

Here we can see the part of the page that contains the player's name and position. Just as before, we will use this shortly to extract these variables.

<figure>
	<img src="{{site.url}}/{{site.baseurl}}/assets/img/skill-pot.png" alt="" style="width: 1000px; height=auto;"> 
	<figcaption>This holds the player's skill and potential</figcaption>
</figure>

This is what will allow us to extract the player's skill level and potential. It is the second to last variable which we will be extracting.

<figure>
	<img src="{{site.url}}/{{site.baseurl}}/assets/img/player-etv.png" alt="" style="width: 1000px; height=auto;"> 
	<figcaption>This holds the player's ETV</figcaption>
</figure>

Finally, the last variable which we will scrape from this website is the player's (ETV).


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

Sure enough, this is Erling Haaland's potential! This example code can easily be modified to find Haaland's other variables. Once you find the correct tag and class, you can scrape whatever you need!


### Step 4: Pagination

Another ingredient needed before actually building the scraper is making sure we know how to click to the next page once we have scraped all the data on the current page. This is called [pagination](https://www.techtarget.com/whatis/definition/pagination).

<figure>
	<img src="{{site.url}}/{{site.baseurl}}/assets/img/pagination.png" alt="" style="width: 1000px; height=auto;"> 
	<figcaption>This holds the "next page" button</figcaption>
</figure>

The above image shows the HTML code that contains the "next page" button. Fortunately for us, the way to implement this into code is the exact same as before! Using the Selenium pattern, you can access the button with the following code:

{%- highlight python -%}
next_button = driver.find_element(By.XPATH, ".//button[contains(@class, 'pagination_next_button')]")
{%- endhighlight -%}

The only other thing we need to do is actually click that button. This can be done simply with the `click()` command, which is part of the Selenium WebDriver library:

{%- highlight python -%}
next_button.click() # Goes to next page
{%- endhighlight -%}

Remember that I am only interested in the first 250 players from this website. There is information on nearly 525 players, so we will only be looping through the first 10 pages.


### Step 5: Load Packages

The last step is small and simple, but vital. We must make sure we have all necessary libraries and packages to scrape this data and save it as a data set. The critical ones are [Pandas](https://pandas.pydata.org/), [Selenium](https://selenium-python.readthedocs.io/), and [ChromeDriverManager](https://pypi.org/project/webdriver-manager/).

{%- highlight python -%}
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
{%- endhighlight -%}

While these are the only ones you *technically* need, you may still find these helpful:

{%- highlight python -%}
from selenium.common.exceptions import StaleElementReferenceException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
{%- endhighlight -%}

The first import handles [stale elements](https://www.softwaretestingmaterial.com/stale-element-reference-exception-selenium-webdriver/). When you go from one page to the next, an element may become stale. This means that the HTML that held an element (say, the player's ETV) on one page might not be the same HTML on the next page. If this happens, you can use a `try` and `except` statement with this import. This will allow us to continue scraping even if we come across this error.

The second import handles [wait times](https://selenium-python.readthedocs.io/waits.html). Sometimes after turning to the next page, it takes a moment for it to load. This can be due to how big the website is or your internet speed. You can create a `wait` statement, where the scraper will wait a specified amount of time before throwing an error if the webpage takes some time to load.

The final import handles [expected conditions](https://www.selenium.dev/selenium/docs/api/java/org/openqa/selenium/support/ui/ExpectedConditions.html). This can be helpful in tandem with the wait time. You can tell the scraper to wait until the conditions are what you expect, and then begin scraping.

There may be several other imports that are of use in a problem such as this, but these are some common ones that you may find necessary.


### Step 6: Build the Scraper

Now that we have the code to find all the variables for one player, the last step is to build a [for loop](https://www.w3schools.com/python/python_for_loops.asp) to iteratively find the information for all players at once.

You will want to start by initializing empty lists in which you can store the gathered data. Then, create a `while` loop. Recall that we will continue looping until we have information on 250 players. Since there are 25 players per page, we will only go through the first 10 pages.

After setting the container that has all players in it, you will write your `for` loop. This `for` loop will find the data for all five desired variables for one player. The code to do this is nearly identical as what we already wrote.

{%- highlight python -%}
for player in all_players:
	potential = player.find_element(By.XPATH, ".//div[contains(@class, 'table-skill')]").text.split('\n')[1]
	# other variables here
{%- endhighlight -%}

The only things that have changed are what we save the variable as, and the name of the container from which we find the information (in this case, `player`). The rest of the `for` loop will contain the other lines from finding one player's information.

Once you append the information to the lists we created at the beginning, the next step is to go to the next page, using the pagination code from before. Once you do this, you are done!

Keep in mind that you may have to include ways to handle wait times, expected conditions, or stale elements in your web scraper (as described above). Aside from this, though, you are done! You have successfully built your web scraper to curate Premier League data. Give yourself a pat on the back!


### Ethics

FootballTransfer is a public site that does not contain proprietary information. On their Social Media sites, they say that people are free to use information from their company as they please, as long as they do not receive any monetary gain from it.


### Conclusion

<figure>
	<img src="{{site.url}}/{{site.baseurl}}/assets/img/prem-trophy.jpg" alt="" style="width: 500px; height=auto;"> 
	<figcaption>This EPL trophy is one of the most coveted cups in soccer</figcaption>
</figure>

This blog post reviewed how I curated a data set of the top 250 EPL players, containing their names, skill levels, potentials, positions, and ETVs. Using Selenium, we were able to build a web scraper that extracted all our desired variables from the webpage.

Stay tuned for my next post when I will perform an exploratory EDA on this data! Now that we have the data set, we will be able to answer the questions of interest!