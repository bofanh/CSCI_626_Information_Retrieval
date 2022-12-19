# Amazon Product Recommendation System


Amazon shopping has become a daily goods shopping tool for most online shoppers today in our life. We create a natural language processing system that gives customers better recommendations even if customers don't know what they want exactly. In this project, we Preprocess the dataset to make information retrieval easier, Create an Amazon product recommendation system, Generate a ranked output of recommended products using collaborative filtering.

## Data

[link](http://jmcauley.ucsd.edu/data/amazon/index_2014.html)

## Introduction

In this project we want to create a personalized search system that will recommend customers products based on product reviews. We think it is important thatIt creates a seamless way for customers to find products that they need without any information on the product. And we expect results that the personalized search system will return a ranked list of reviews that contain the keywords that were specified by the customer. For instance, If a customer searches for “good cleaning supplies”, the IR system will return a ranked list of reviews that each contain the terms “good cleaning supplies”. Each review will be associated with a product ID and ranked by rating.

## Approach
Step 1: Creating a Dataframe. Randomly sampled 1,000 objects from ‘prep,csv’, We generated a dataframe based off of ‘prep.csv’ that only contained the attributes asin and reviewText.

Step 2: Text Processing. Tokenizing the attribute reviewText
Removing the stopwords from reviewText
Removing punctuation marks from reviewText
Casting all words in reviewText to lowercase
Lemmatizing reviewText

Step 3: Term Vectorization. Vectorized the corpus of reviewTexts to calculate the tf-idfs[2]

Step 4: Essential Functions. We defined two essential functions:
A function that generates a corpus of reviewTexts
A function to calculate the cosine similarity between two objects

Step 5: Cosine Similarity Rank. Generated a dataframe containing the cosine similarities with their associated asin.

## Experiments and Results

User Query defines a function to retrieve a search query. defined a function to  concatenate the query to the first column of the dataframe. Query is tokenized,  stopwords are removed, punctuations are removed, casted to lowercase, and lemmatized. Finally we will compute a list of product asins that sorted from high similarity to low.

## Conclusions

We were able to recommend products based on a user query and product reviews. The longer the query the better. Some short queries may not produce an output because the data sample (1,000 objects) lowers the range of possible recommendations. Some asin values may not link to a product because that product no longer exists.
In the future work, we will test our recommendation system with a larger sample size/ or entire dataset. And Implement the Word2Vec[3] algorithm by using the CBOW[4] model in (example of a neural network) by using tensorflow/pytorch framework.

