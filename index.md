---
layout: default
title: "⚡️ Load forecasting and peak shaving with neural networks"
nav_order: 0
permalink: /
---

# An electric utility’s 3-part guide to peak shaving with neural networks.

*By [Kevin McElwee](https://www.kmcelwee.com)*

Increasing the accuracy of day-ahead load forecasting can save utilities 
tens of thousands of dollars. Below, I've proposed statistical solutions to key questions asked
by utilities when implementing peak shaving strategies with storage.

This research was conducted in conjunction with the [Open Modeling Framework (OMF)](https://omf.coop/).
These blog posts were originally posted in [Towards Data Science.](https://towardsdatascience.com/an-electric-utilitys-3-part-guide-to-peak-shaving-with-neural-networks-de5c7752d946)
View more projects on [my professional site](https://www.kmcelwee.com/).

## Question 1: What’s tomorrow’s load?

### [Using neural nets to predict tomorrow’s electric consumption](simple-load-forecasting.html)
The smallest error can lose utilities thousands of dollars in a single day. Neural networks can help make sure that doesn’t happen.

**Main takeaways**

* To get any kind of useful energy consumption forecast, simple machine learning isn’t appropriate. Deep learning, however, can get us the accuracy we need.
* Given historical load and temperature data, a straightforward neural network can give a 24-hour forecast with about 97 percent accuracy.

### [Predict daily electric consumption with neural networks.](day-long-load-forecasting.html)
How a simple three-dimensional structure reduces error, outcompetes more complex models, and doubles savings.

**Main takeaways:**

* A day-long approach to load forecasting is more accurate than an hour-by-hour approach.
* Despite having only a 1 MAPE difference in error between the two approaches, tests showed the method doubling our savings when peak shaving due to less eratic output.


## Question 2: Is tomorrow's peak the monthly peak?

### [How short-term forecasting with neural nets can inform long-term decisions](predict-monthly-peak.html)

Electric utilities can detect monthly peaks with only a three-day forecast.

**Main takeaways:**

* Making peak shaving dispatches every day can be costly.
* Multi-day forecasts can help us dramatically reduce the number of dispatches per month without the risk of missing the monthly peak.
* Utilities would need to set their own priorities, but for the most part, they can dispatch only once every week or so while only missing one peak every few years.


## Question 3: How should we implement a peak-shaving strategy given the uncertainty?

### [Peak shaving with unreliable forecasts](calculate-uncertainty.html)
How one 19th-century physics equation can increase electric utilities’ savings by more than 60 percent.

**Main takeaways:**

* Because there’re inevitably errors in our forecast, the “optimal” dispatch solution for the forecast won’t necessarily be the best dispatch in practice.
* The heat equation can be used to spread out our dispatches (e.g. if our predicted forecast would suggest dispatching 500kW at 12pm, the equation might return 150kW at 11am, 200kW at 12pm, and 150kW at 1pm.)
* This simple approach can save a lot of money. Savings in one region of Texas was increased by more than 60%.
* The equation requires two constants as inputs, but they shouldn’t be hard for a utility to optimize.

