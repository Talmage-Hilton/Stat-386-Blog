---
layout: post
title:  "Creating a Streamlit Web App"
date: 2024-04-30
description: A simple explanation of how I created a Streamlit Web Application
image: "/assets/img/dangerous-algorithm.jpg"
display_image: false  # change this to true to display the image below the banner 
---
<p class="intro"><span class="dropcap">T</span>his post will be a summary of my thoughts on Cathy O'Neil's book *Weapons of Math Destruction*. It will be in a Q&A format.</p>
<p class="intro">Cover image source: <a href="https://sloanreview.mit.edu/article/coming-to-grips-with-dangerous-algorithms/">MIT Sloan Management Review</a></p>


### Introduction

[Cathy O'Neil]("https://cathyoneil.org/"), in her own words, is "a writer, bluegrass fiddler, and algorithmic auditor." She earned a Ph.D. in math from Harvard and has worked as a math professor, hedge fund quant, and a data scientist. She has spent a lifetime studying mathematical models and algorithms. After working in the private sector, she noticed many areas in which models and algorithms unfairly target certain groups of people.

<figure>
	<img src="{{site.url}}/{{site.baseurl}}/assets/img/cathy.webp" alt="" style="width: 500px; height=auto;"> 
	<figcaption>Cathy O'Neil - author of Weapons of Math Destruction</figcaption>
    <figcaption>Image Source: <a href="https://www.theguardian.com/science/2022/apr/09/cathy-oneil-big-tech-makes-use-of-shame-to-profit-from-our-interactions">The Guardian</a></figcaption>
</figure>

Rather than continuing on with her discomforting knowledge, she decided to do something about it. This resulted in her authorship of *Weapons of Math Destruction*, a book that seeks to bring attention to the discriminatory nature of certain algorithms companies use.

In short, a [Weapon of Math Destruction]("https://en.wikipedia.org/wiki/Weapons_of_Math_Destruction") (WMD) is a mathematical algorithm that takes human traits and quantifies them. This is done for anything from insurance to credit score to university acceptance. However noble their intentions are, they have damaging effects and result in the perpetuation of bias, discrimination, and destruction against certain groups of people, especially minorities.

*Weapons of Math Destruction* was written by O'Neil in 2016. This blog post will be my thoughts and response to the book. It will be in a Question and Answer format. A series of 9 questions will be posed, and I will provide my answer to each. Please keep in mind that these are simply my own thoughts and do not represent the ideas of any other organizations.

<figure>
	<img src="{{site.url}}/{{site.baseurl}}/assets/img/wmd.webp" alt="" style="width: 500px; height=auto;"> 
	<figcaption><em>Weapons of Math Destruction</em> - published in 2016</figcaption>
    <figcaption>Image Source: <a href="https://www.amazon.com/Weapons-Math-Destruction-Increases-Inequality/dp/0553418815">Amazon</a></figcaption>
</figure>


### 1. What is the primary argument or thesis of the book? How does the author support this argument throughout the text?

The primary argument of the book is that WMDs are very dangerous and very widespread. People are imperfect and they make imperfect models. Even models that seem perfectly objective and unprejudiced were designed by people who are subjective and have prejudices. The purpose of this book is to describe the harm that WMDs present.

The author primarily supports this thesis by using several different real-world examples. Some of these examples include the [LSI-R]("https://www.assessments.com/purchase/detail.asp?SKU=5212") (a questionnaire given to prisoners to assess their level of risk, but it masks racism with questions and technology), credit ratings, predatory advertisements that target poorer demographics, personality quizzes for job hiring, value-added models, insurance provider models, and even social media.

A more specific example was the [US News’s ranking of American universities]("https://www.usnews.com/best-colleges/rankings/national-universities"). Essentially, US News used a lot of proxy data to create their first list of the best universities in the country. While they did use some hard facts, they used other variables that were simpler, cheaper, or seemingly more ethical. The bigger issue, however, was the model itself, not its inputs. The algorithm had a poor feedback loop. When a university received a high ranking, more people would want to go there (because of its prestige), resulting in more applicants who had higher test scores, GPA, and a likelihood of success after college. This boosted the university even more and resulted in even higher rankings the next year. However, this unfortunately had the opposite effect in the other direction. Universities that received a low rating received fewer applicants, forcing them to accept students that they previously wouldn’t have had to. They continued to get worse and worse ratings as time went on. They were victims of a WMD.

WMDs are characterized by their opacity, scale, and damage. The author does a great job of explaining this through examples, explanations, and quotations from other experts in the field of data ethics.



### 2. What were the three most significant insights you gained from this book? How do they relate to contemporary issues in data science?

The first insight I gained is that it's important to not separate technical models from real people. Data or computer scientists that make these algorithms often have very good intentions. They typically are being told by their boss to make a cheaper or more efficient method to do something. The problem comes when cheapness and efficiency becomes more important than the people affected by the model. That is super important today in data science.

Another insight I gained is that proxies don't represent real, complicated life. This hearkens back to the US News example described earlier. You can use proxy data like [ACT scores]("https://www.act.org/content/act/en/products-and-services/the-act/scores.html") to measure how good a college is, but that doesn't mean those students are actually the smartest. Similarly, a zip code in which a person lives doesn’t mean that they have all the same attributes and demographics as someone else in that zip code. Real life is much more complicated than what a few variables can tell us. Data Scientists have to be very careful that the models they use or develop actually capture real life, not just aspects of it.

The third insight is that WMDs target everyone, not just the poor. Along with that, [Big Data]("https://www.oracle.com/big-data/what-is-big-data/#:~:text=What%20exactly%20is%20big%20data,especially%20from%20new%20data%20sources.") codifies the past, it doesn't invent the future. It reaffirms what we've done and makes us do that even more. It removes the chances of improvement or growth in a lot of ways. It’s extremely important to realize that you can’t use Big Data to perfectly predict what a person will do. Yes, a similar person may be likely to default on a loan because another similar person defaulted a decade ago, but that doesn’t mean this new similar person is guaranteed to default every time. That’s the issue with Big Data if it’s not used correctly. As Data Scientists who use Big Data, we have to be careful to not make people fall victim to it.



### 3. Which ethical concerns highlighted by the author resonated most with you? Why?

The ethical concern that resonated with me the most was the constant theme of money. Whether it’s companies, agencies, governments, or schools, the motivating factor in creating a WMD is to save money and, thus, have more of it. Money is obviously vital, but it becomes a significant issue when that becomes more important than fairness. It reminds me of a quote from [Hugh Nibley](""https://rsc.byu.edu/author/nibley-hugh-w), a religious scholar. He once said, “The more important wealth is, the less important it is how one gets it.”

I first heard this quote on my [LDS mission]("https://www.churchofjesuschrist.org/callings/missionary?lang=eng") and it instantly made a huge impact on me. I served in a place where a lot of people struggled financially. It was hard to see the people I loved so much constantly have trials simply because of the amount of money in their bank account. I made a mental vow to always make sure I make my money in ethical ways. The people in my mission deserved at least that. My future family deserved at least that. This is why the topic of money resonates so much with me.



### 4. Does the author propose any solutions to the issues they address? Do you agree with these solutions? Why or why not?

The biggest overall solution she proposed was simply to bring light to WMDs. Most people do not know of their existence. The ones that do (big companies, governments, schools, etc.) don't want people to find out. Newspapers, journal companies, influencers, and others with a large outreach can discuss WMDs and their harm. In one example, an article was written about companies who make their employees work terrible hours and only give them a couple days' notice when they do so. Within weeks, all of the companies announced that they would be changing their scheduling practices. This is great news and I think this is the simplest and best way to put an end to WMDs. The thing about which companies care the most is their public image.

Another solution proposed in the book is to get rid of the data that causes the WMDs to result in a feedback loop. The big issue with WMDs is that they value efficiency, so much so that they forfeit fairness. In the example of the police WMD, the author proposes getting rid of the “efficient” variables (antisocial behavior—data that causes police to arrest more and more poor and minority people) and thus have a more equal system. In short, the author is saying we should be okay sacrificing a little efficiency in the name of fairness.

There are other solutions the author proposes, such as reconsidering the objectives of the model, creating more all-around equality in the model, and algorithmic audits. The last solution I’ll discuss at depth is getting to the modelers. If the people who create the model don’t see the issue with WMDs, we will never be able to put an end to them. If we can somehow get them to see the possible misuses and misinterpretations that their model can create, they will be more likely to make fair models.



### 5. How does the author view the role of technology in society? Is it a tool for empowerment, a source of concern, or a bit of both?

The author definitely views technology as a source of concern. Of course, she views it as a great tool that could be used for empowerment if utilized correctly, but the evidence points to abuse much more than proper use. The introduction of the book is full of her thoughts on technology.

I should refrain from speaking in absolutes. Technology can be used to empower the rich and privileged, but that is the whole point of the book. Technology (models) can be used to create huge variations between the fortunate and the less fortunate. While it may empower some and help them receive more money, laud, and fame, it hinders the poor and unfortunate and essentially entirely prevents them from moving forward.



### 6. Can you think of a current real-world example that reflects the themes or concerns of the book? Describe and analyze this example in relation to the book's arguments.

The whole book is just a documentation of real-world examples of WMDs, but I tried to think of something else that is similar to a WMD. This led me to think about [Apple]("https://www.googleadservices.com/pagead/aclk?sa=L&ai=DChcSEwjOzI7T8fmFAxWXhsIIHTnDDZIYABAAGgJqZg&ase=2&gclid=Cj0KCQjw_-GxBhC1ARIsADGgDjtX4p0OWef2oZQlIVnwMTlwRcErSMNg9wiOoxusBi-zMz3J3SAACOUaAlFKEALw_wcB&ohost=www.google.com&cid=CAESVuD2CVJtISK-h3J8en3BZPQyeWHQ3kBTpnIcNQuwwyf9wHhrr4aGrfKAIMs9KonwDR_dWglAYNj38izBek2A-5IA7CDlDhF1dg4PTbIfRYOg-FVtizdQ&sig=AOD64_2Am4bB-qQBr1A__CTW_MTeFnFrzQ&q&nis=4&adurl&ved=2ahUKEwiPjIbT8fmFAxXaLUQIHc5YDD0Q0Qx6BAgPEAE"), the technology company that makes the iPhone. However, Apple makes much more than just the iPhone. Some of their products are iPads, Apple Watches, Macs, AirPods, Vision Pros, chargers, and cords, to name just a few. This has created a new term–the [Apple Ecosystem]("https://en.wikipedia.org/wiki/Apple_ecosystem").

The Apple Ecosystem refers to being locked into only using Apple products. Once you get an iPhone, for example, you want to get the best headphones that are integrated with the software in the most efficient ways. Thus, you’ll get AirPods. If you want to be able to see notifications and have access to your phone even when you don’t have your phone, you’ll then purchase an Apple Watch. You need an iCloud account to store all your data. You want a Macbook to seamlessly switch between your phone, tablet, watch, etc. In short, Apple locks you in and makes you want to keep purchasing their products, and only their products. And once you have, it is nearly impossible to get out of the Ecosystem. It would require so much work to transfer all your data, pictures, videos, and apps to another operating system.

This can be seen almost like a negative feedback loop. Once you’re in the model, you keep spiraling and buying new things, spending more money, getting more and more locked in. I must note that I do not think the Apple Ecosystem is bad. Besides a Mac, I have almost every other product Apple has ever produced and I really enjoy them. I’m part of the feedback loop.



### 7. While many readers often finish these books with heightened concerns or discouragement about ethical issues in data science, it's also important to identify areas of hope and potential for positive change. What were some hopeful or optimistic insights, examples, or suggestions presented in the book you read? How might these pave the way for a more ethically-informed future in the realm of data science?

One area that gives me a lot of hope is the fact that the government has made a lot of laws throughout the years to try to minimize the effect of online WMDs. While there haven’t been many repercussions for the companies that use them, and even though it has been relatively easy for those companies to find the loopholes in the law, it is definitely a good start.

I think this is great because it means that the top organizations know about WMDs. The government is the ultimate power in the US, so hopefully this can have a trickle-down effect throughout the rest of the country. Hopefully companies will start to notice the negative effects their models have. Regardless of all that, the fact that the government knows about WMDs means that the majority of the US population can as well. If we keep up with the news and see what the government has done to try and get rid of these negative effects, the general public can see it as well. We can hope that the future generations of Data Scientists grow up with this already in mind, which will lead to a much more ethical future.



### 8. On a scale of 1 to 10, how would you rate this book in terms of its clarity, insights, and overall impact on your understanding of data science ethics? Please provide reasons for your rating. What elements of the book contributed most to this score, and were there areas where you felt the author could have improved or expanded upon?

This book gets a 10 out of 10 from me. It is very clear, very instructive, includes many specific examples, with detailed explanations of them all, and painted a really simple picture of data ethics in a way that made me want to learn more about them and respect them. I don’t do a lot of reading outside of school textbooks, but I found myself wanting to read this book in my free time. That is evidence enough for me.

I really appreciated just how many examples there were and, specifically, the fact that the author included examples from different fields. If she only had focused on credit rating WMDs, it would have been good and eye-opening, but I wouldn’t have wanted to read it. But she included examples from nearly every part of life, and it really grew my understanding of the world in which I live.

I think the only area in which the author could have improved is to go a bit more in-depth into the examples. Sometimes she would include five examples in a single chapter. As I just mentioned, I loved this, but I would’ve rather seen two or three examples per chapter with more explanations behind them.



### 9. Considering the insights, writing style, and overall impact of the book, would you recommend it to others, and if so, to whom specifically (e.g., fellow students, professionals in data science, general readers)? What are the main reasons for your recommendation or lack thereof?

I would absolutely recommend this book to others. It's written very simply and comfortably, not using huge words or concepts that are difficult to understand. It's almost like reading an article online, where its purpose is to be as easily understood as possible. The examples used were really profound and she described them perfectly. She also went into more details when it could've been confusing. Overall, it is a really good book.

I would recommend this book to anyone who is a statistics major, computer science major, or any other major that could ever develop a model like the ones seen in the book. Both of my parents are also statisticians, so I want them to read it as well. They aren’t the ones making models or algorithms, but I still would like to hear their insights. It simply is a good self-check to see where our priorities lie. Do I care more about money and efficiency than I do about people? It’s always a good reminder of what matters most.



### Final Thoughts

In my opinion, it's okay to be part of a feedback loop as long as you are aware of it. Not all feedback loops are bad. In fact, some are even good! The issue comes when you are unaware that you're in a loop, or even worse, if you're not even aware of the loop in the first place. It is important to know about the existence of these things, so that you can avoid its traps. That is one reason I am making this blog post-to bring to light their existence and destructive nature.

While there certainly are a lot of things to be aware of when it comes to WMDs, it is also important to remember how great technology can be when used correctly. It can bring us together, allow for close and intimate contact from anywhere in the world, spread messages of hope, and give people the education needed to find more freedom. Yes, we must be wary of technology, but we can't be afraid of it. We should all strive to use it in as many good ways as possible.

The world is hard and scary enough as it is. The least we can do is use what we have for some good.