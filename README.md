## Describe here your project


1.	Data set description.
The chosen data set is from twitter about sentiment analysis. It contains 1.6 million data entries as training data and about 500 entries as test data. The data file has 6 fields:

0 - the polarity of the tweet (0- negative, 2- neutral, 4- positive)
1 -  the id of the tweet (0000)
2 - the date of the tweet (Sat May 16 23:58:44 UTC 2009)
3 - the query (e.g. BigData; If there is no query, then this value is NO_QUERY)
4 - the user that tweeted (yuhe)
5 - the text of the tweet (BigData is cool)
	
	In our case, we only need column 0, 1, 5
The data link is shown below:
http://help.sentiment140.com/for-students

You can also get the data directly from AWS:
Training Data: s3://yuhe201809/TermProject/trainingdata.csv
Test Data: s3://yuhe201809/TermProject/testdata.csv


2.	Research questions and Learning model.

The goal of analyzing the data is to find the relation between tweets’ contents and polarities (sentiments). Thus, we could predict tweet’s sentiment given tweets’ content.

The first step is to process data and build reference wordlists for different polarities (negative/ neutral/ positive wordlists), then we could transfer the text of tweet into index of polarity wordlists and calculate the score for each tweet (for each word, 1 for positive, 0 for neutral, -1 for negative, still need to be discussed). Then we get a value pair (length, score). 

Then we could use K-means method to build a 3-cluster model, which classify the training data into 3 categories: negative, neutral, positive.  If time allowed, we could implement SVM model also. 


# How to run  

Run the task 1 by submitting the task to spark-submit. 


```python

spark-submit main_task1.py 

```



```python

spark-submit main_task2.py 

```



```python

spark-submit main_task3.py 

```



