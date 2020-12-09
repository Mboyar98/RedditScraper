# %%% Step 1. installing the PRAW and scraping/organizing the data
# Installing and importing preliminaries 
import praw
import pandas as pd

reddit = praw.Reddit(client_id ='U8c9omfn7d9Uhw',
                     client_secret ='gMOGcH8A_ZXJXwv8tOlX6zUdUQsY2w',
                     user_agent ='Appforpython by u/DummyAccountforXavi')

# Creating the empty lists
TopPosts = []
Upvotes = []
Comments = []
for submission in reddit.subreddit("pics").top('all'):
    TopPosts.append(submission.title)
    Upvotes.append(submission.score)
    Comments.append(submission.num_comments)    
TopPosts
Upvotes
Comments

df = pd.DataFrame({'PostName': TopPosts, 
                   'Upvotes': Upvotes, 
                   'Comments': Comments})
df.iloc[22]

# %%% Setting up the Data
# Importing more preliminaries
import numpy as np
import math
import statistics 
from sklearn                         import tree
from sklearn                         import linear_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model            import LinearRegression
from sklearn.neighbors               import KNeighborsClassifier
from sklearn.datasets                import load_iris
from sklearn.model_selection         import train_test_split
from sklearn.naive_bayes             import GaussianNB
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix

pd.set_option('display.max_rows',     20)
pd.set_option('display.max_columns',  20)
pd.set_option('display.width',       800)
pd.set_option('display.max_colwidth', 20)

# Counting the number of words used in each title
mylist = df.PostName
num_words_in_title = [len(sentence.split()) for sentence in mylist]
df['Number of Words'] = num_words_in_title

df.reset_index()
df['ML_group']    = np.random.randint(100,size = df.shape[0])
df                = df.sort_values(by='ML_group')
inx_train         = df.ML_group<80                     
inx_valid         = (df.ML_group>=80)&(df.ML_group<90)
inx_test          = (df.ML_group>=90)


# %%% TVT - SPLIT
Y_train   = df.Upvotes[inx_train].to_list()
Y_valid   = df.Upvotes[inx_valid].to_list()
Y_test    = df.Upvotes[inx_test].to_list()

X_train   = df.loc[inx_train, ['Comments', 'Number of Words']]
X_valid   = df.loc[inx_valid, ['Comments', 'Number of Words']]
X_test    = df.loc[inx_test, ['Comments', 'Number of Words']]

X_train.shape[0]+X_valid.shape[0]+X_test.shape[0]


# %%% 5. Linear regression and Mean Squared Error
clf = LinearRegression().fit(X_train, Y_train)
Upvotes_pred = clf.predict(X_test)

df['N_Upvotes_reg'] = np.concatenate(
        [
                clf.predict(X_train),
                clf.predict(X_valid),
                clf.predict(X_test)
        ]
        ).round().astype(int)

df['is_equal']  = df.Upvotes == df.N_Upvotes_reg

mean_squared_error(Y_test, Upvotes_pred)

math.sqrt(mean_squared_error(Y_test, Upvotes_pred))
statistics.stdev(Upvotes)

