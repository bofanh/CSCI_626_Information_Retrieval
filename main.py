# B
# import packges
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import wordpunct_tokenize
import nltk
from nltk.corpus import stopwords
# nltk.download('wordnet')
# nltk.download('omw-1.4')
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity
import time

import missingno as msno
import math

import pickle

english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']





def quary():
    quary_search = str(input('Enter your search key words:'))
    print(quary_search)

    return quary_search

def loaddata(numbers = 1000):
    filename = 'data/prep.csv'
    df = pd.read_csv(filename, index_col='Unnamed: 0')
    df.head()
    df = df.sample(n = numbers)
    # read df
    df = df[['asin','reviewText']]

    # print(df.head())
    return df


# load entire dataset
def loaddataall(): 
    filename = 'data/prep.csv'
    df = pd.read_csv(filename, index_col='Unnamed: 0')
    df.head()
    # df = df.sample(n = numbers)
    # read df
    df = df[['asin','reviewText']]

    # print(df.head())
    return df

def addquarytodf(search,df2):
    df1 = pd.DataFrame({'asin':'000000', 'reviewText':search}, index=[0])
    # print(df1)
    df3 = pd.concat([df1,df2])
    # print(df3.head())
    return df3

def clean_data(df):
    # apply tokenization

    df['tokens'] =  df["reviewText"].apply(word_tokenize)

    # apply remove stopwords 
    df["removeStopwords"] = df["tokens"].apply(lambda x: ([word for word in x if word not in (stopwords.words())]))

    # remove punkt

    df["removePunk"] = df["removeStopwords"].apply(lambda x: ([word for word in x if word not in (english_punctuations)]))

    # lower case
    df["lowStr"] = df["removePunk"].apply(lambda x: ([word.lower() for word in x]))

    #Lemmatization

    wnl = nltk.WordNetLemmatizer()
    df["lem"] = df["lowStr"].apply(lambda x: ([wnl.lemmatize(word) for word in x]))

    # clean df
    df = df[['asin', 'lem']]
    df["lem"] = df["lem"].apply(lambda x: ','.join(map(str, x)))

    return df



# functions 
def create_corpus(df):
    corpus = []
    for i in df:
        corpus.append(i)
    return corpus
    

def cor2vec(corpus,df):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    Xarr = X.toarray()
    Xarr
    featurearr = vectorizer.get_feature_names_out()
    featurearr = featurearr.tolist()

    df = pd.DataFrame(data=Xarr, columns=featurearr, index=df['asin'])
    return df
# calculate cosine_similarity
def cs(i,j,df, col):
  doc1 = df.iloc[i]/df.max(axis=0)
  doc2 = df.iloc[j]/df.max(axis=0)

  docAB = doc1.mul(doc2, axis=0)
  numerator = docAB.sum() # inner prodoct

  powdf = pd.DataFrame([2]*col)
  powdf = powdf.T
  doc1square = doc1.pow(powdf)
  doc2square = doc2.pow(powdf)
  sum1 = doc1square.sum(axis = 1)
  sum2 = doc2square.sum(axis = 1)
  sqrtsum1 = math.sqrt(sum1)
  sqrtsum2 = math.sqrt(sum2)
  denominator = sqrtsum1 * sqrtsum2 # product of A vec and B vec
  if denominator == 0:
    cosSim = 0
  else:
    cosSim= numerator/denominator
  # print('cosSim(doc',i+1,',','doc',j+1,') is')
  # print(" ") 
  return cosSim

# get sim rank result
def simRank(df, df_index, rangN = 30,col=2):
    result = []
    for i in range(0,rangN):
        result.append(cs(0,i,df,col))
    df_result = pd.DataFrame(data=result,columns=['CosSim'], index = df_index)
    df_result = df_result[df_result.CosSim > 0]
    df_result = df_result.sort_values(by=['CosSim'],ascending=False)
    df_result = df_result.iloc[1: , :]

    return df_result


def sumsim(df_result):
    df_result =df_result.reset_index()
    df_result = df_result.groupby(by=['asin']).sum()
    df_result.sort_values('CosSim', ascending=False)

    df_result1 =df_result.reset_index()

    print( df_result1['asin'].nunique())
    return df_result

    


def get_corpus():
    df = loaddataall()
    df = clean_data(df)
    corpus = create_corpus(df["lem"])
    with open("data/corpusfull", "wb") as fp:   #Pickling
        pickle.dump(corpus, fp)
    
    print('done')



def main():
    start = time.time()
    search = quary()
    datasize = 1000
    df = loaddata(datasize)
    df = addquarytodf(search,df)
    df = clean_data(df)
    corpus = create_corpus(df["lem"])
    row, col = df.shape
    df2 = cor2vec(corpus,df)
    df_index = df['asin']
    df_result = simRank(df2, df_index, rangN = datasize+1, col=col)
    result = sumsim(df_result)

    # print(corpus)
    # print(df.head())
    print(result.head(10))


    print(time.time() - start, 'sec')




if __name__ == "__main__":
    # main()
    get_corpus()