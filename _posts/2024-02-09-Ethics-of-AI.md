---
layout: post
title:  "The Dangers of AI"
date: 2024-02-09
description: A look into the potential issues with using AI 
image: "/assets/img/AI.jpg"
display_image: false  # change this to true to display the image below the banner 
---
<p class="intro"><span class="dropcap">T</span>his post will dive into the dangers of using AI in data science. We will look at a brief history of AI, potential issues in its deployment, and why that is pertinent to data science.</p>


### A History of AI

AI was birthed in 1950, when British mathematician, Alan Turing, created a test of machine intelligence. If you've seen the film "The Imitation Game" then you are familiar with this story. Turing, however, was greatly weighed down by the technology of his time. Computers needed to greatly develop before there could be any hope of true artifical intelligence.

The following two decades would see a huge level of advancement in artifical intelligence. Programming languages, speeches, robots, and even vehicles were all invented in the name of AI. Though rudimentary compared to today's machines, the public learned from these creations what the future could look like.

<figure>
	<img src="{{site.url}}/{{site.baseurl}}/assets/img/alan_turing.jpg" alt=""> 
	<figcaption>Alan Turing working on The Imitation Game</figcaption>
</figure>

After a huge boom in the 1980s, the public was gradually given more and more access to AI machines. Windows released a speech recognition software in the early 2000s, shortly thereafter the first Roomba (an autonomous vacuum) was unveiled, social media companies began using AI as part of their algorithms, and companies like Apple and Google rolled out their virtual assistants starting in 2011. Most recently, of course, has been the invention of ChatGPT and Dall-E. For a more complete explanation of AI's past, read the following [history from Tableau](https://www.tableau.com/data-insights/ai/history#:~:text=Birth%20of%20AI%3A%201950%2D1956&text=into%20popular%20use.-,Dates%20of%20note%3A,ever%20learn%20the%20game%20independently.).

While these inventions undoutedbly contributed to making life easier in many ways, there are things we need to be very careful about with AI.


### Potential Dangers of AI

The most important thing to remember is that, although AI is built to gather data and produce output completely on its own, it still was built by an imperfect human. Human tendencies and biases go into every machine, algorithm, and AI, no matter how objective it may seem at first.

These are some of the potential dangers of AI:
* Discrimination
* Harmful feedback loops
* Privacy violations
* Lack of transparency
* Security vulnerabilities
* Job displacement

The biggest problem with Big Data is that AI is often forced to use proxy data to make decisions in the name of fairness, ease, and affordability. Companies want to make things as quick and cheap as possible, so most of them turn to artifical intelligence. Whether it be finding people at which to target their ads, deciding the next promotion, or firing the next employee, AI can be unethically used to unfairly impact people that just happened to fail in one of the proxy areas the AI regarded as important.


### The Connection to Data Science

All of the dangers present in AI are especially important for data scientists to be aware of. Sometimes it is the data scientists themselves who use AI to create algorithms. Let's look at one example of harmful feedback loops that university students may find intriguing.

In 1983 US News created a ranking system for all universities in the country. While no harm was intended, many problems arose in the persuant years. Universities that received a high rating received more and more applications every year (it is obvious why students would want to go to a "prestigious" university). Not only was the quantity greater, so was the caliber of applicants. These universities could tighten their requirements, accept better students, and climb even higher in the rankings.

Universities that received a low rating, however, felt the opposite effect. They received fewer applicants, requiring the university to accept students they wouldn't have previously. They dropped further in the rankings as a result.

The issue wasn't in the universities themselves, it was that the algorithm developed did not accurately reflect how good the school actually was. They used seemingly random variables to arbitrarily grade how good the university was. Worse yet, the algorithm had a nasty feedback loop. When the universities it ranked highly continued to do even better, it took that to mean it was performing correctly. When the low-ranked universities continued to drop, it was the same.

<figure>
	<img src="{{site.url}}/{{site.baseurl}}/assets/img/byu.jpg" alt=""> 
	<figcaption>Brigham Young University - ranked #115 out of 439 universities</figcaption>
</figure>

Whether it is data scientists or not who create the algorithm in the first place, they almost certainly will be tasked with running analysis on data that, in one form or another, has been influenced by AI. AI is so ubiquitous today that it is rare to find data untouched by it.


### Conclusion

AI is a wonderful tool that allows for quicker and cheaper data collection, modeling, and analyzing. It does, however, present a great risk to the ethics of Data Science.

As Data Scientists, we must be extremely aware of how AI can be used. When we are creating models and algorithms, we have to be sure we are using it correctly. When we are analyzing data that's used AI, we should find out how that data may have been tainted by it.

The future of AI looks bright as long as we ensure our correct usage of it.