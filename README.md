# ufc-predictor

For our final project, we built a UFC fight predictor. We compared various ML models including Logistic Regression, AdaBoost, Decision Tree, K-nearest neighbours, XgBoost, and Random Forest. We also implemented Logistic Regression from scratch by writing a function that did gradient descent and we applied a sigmoid function on it.

We had two datasets. Our first dataset was scraped from ufcstats.com. We scraped stats of all past fights (such as strikes landed, takedowns, date, duration), and we also scraped individual fighter data (height, weight, reach, record). Our second dataset was used for backtesting. It consisted of betting odds from 2010, and this was downloaded from kaggle.

# Experimental Results

Our models performed very well on the training set. Fully tuned decision trees were at 75%, and knn was also at a similar level. However, Logistic Regression, the boosting algorithms, and random forest were all over 85% accuracy.

However, the issue with this is that the model assumes it has fight data available before the fight has taken place, so this metric is not really any useful. Essentially, the models learned to associate certain features with a win or a loss.

To truly test our models, we had to backtest them. We created a dataframe that was ordered by date, and for each date, had fighter stats as of that time period, which were calculated by iterating chronologically. For backtesting, we did not use the decision tree and knn, as they were significantly worse.
Our backtested models performed around 53-54% accuracy. We created a voting classifier that included all the models used for backtesting, and while adjusting the classification threshold, it reached as high as 59%.

But even accuracy is not as important as actual money made from betting. Using the odds, we simulated 100 dollar bets made by each model, and we calculated the long term return for all the fights we had odds data for. We had reached returns as high as 54%. The model with the best return was a Logistic Regression with l1 regularization, and a very high classification boundary of 0.9.

# Learnings

We think the biggest learning point was that scraping and cleaning data, and getting it all set up was extremely time consuming. We spent 2⁄3 weeks on just preparing our data to build our models around. We had to do a lot of date manipulation, since they didn’t quite scrape correctly, and getting the fight stats
combined with the individual fighter stats was also tricky. At one point we had messed up our data so badly that we decided to just re-scrape the whole thing, which took a long time and brought us back to square one .There was a lot of time between when we had our data scraped and when we got to actually writing our models. In future projects, I think we will all budget more time for data cleaning.

Another learning point was that it truly is very hard to beat the sports bookies. I think MMA is still a new sport so there is more potential than other sports, but the counter to this is that it has very limited data. Additionally, it seems like there is only one promotion (the UFC) that keeps track of data, we could not find any other promotion’s data. Even though we were building the predictor for the UFC, most of the UFC fighters first fight in other promotions, and having their data would probably help us in making a better model.

Finally, an issue we faced was how to set up the features. Each fight had the number of strikes thrown, but we could not store a fighter’s total strikes, just because some fighters have more fights than others, and this would be a biased statistic. We decided to go at the per minute route, and divided every stat by the time of a fight, and aggregated it over all the fights for final predictions. We did this because the UFC itself compares statistics like strikes landed per minute. But, the biggest problem here is prediction of how many strikes or takedowns a fighter would land, and by using an aggregated measure, we are assuming that a fighter will perform similarly to his average performance. I think this approach is slightly flawed because there are honestly not enough data points to assume that the average we have is a fighter’s true expected output


# How to run the code


Libraries - numpy, statsmodels, pandas, sklearn, datetime, pickle

If you just want to see the predictor in action, all you need to run is the notebook titled ‘predict.ipynb’. All the relevant data will be in the folder.
Now if you want to run the scraper and the cleaning, run the files in the following order: 
ALL THIS ESPECIALLY THE SCRAPING WILL TAKE A VERY LONG TIME TO RUN (2-4 hours)

fighter_scrape.ipynb -> fight_scrape.ipynb -> data_cleaning.ipynb -> data_cleaning2.ipynb -> combine_fighters_fights_data.ipynb -> cleaning_continued.ipynb -> cleaning4.ipynb


If you want to see the creation of features, modelling, and backtesting, run in the following order:

#SOME OF THE GRID SEARCH WAS RUN OVERNIGHT, THE MODELLING NOTEBOOK WILL TAKE A WHILE
Feature_engineering.ipynb -> modelling.ipynb -> backtesting.ipynb
