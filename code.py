#!/usr/bin/env python
# coding: utf-8
# In[60]:


import numpy as np
import pandas as pd 
# pandas library is attatched with a zip file in submission.
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from string import punctuation
import re
stop = stopwords.words('english')


# # Basic cleaning
# remove punctuations and numbers

# In[61]:


df_positive = pd.read_table("rt-polarity.pos",sep='\n',header = None,dtype=str,encoding='latin-1')
df_negative = pd.read_table("rt-polarity.neg",sep='\n',header = None,dtype=str,encoding='latin-1')
df_positive[1]='1'
df_negative[1]='0'

def remove_punctuations(sentence):
    for p in punctuation:
        sentence = sentence.replace(p,'')
    return sentence
df_positive[0]=df_positive[0].apply(remove_punctuations)
df_negative[0]=df_negative[0].apply(remove_punctuations)

def remove_digits(sentence):
    sentence = ''.join([i for i in sentence if not i.isdigit()])
    return sentence

df_positive[0]=df_positive[0].apply(remove_digits)
df_negative[0]=df_negative[0].apply(remove_digits)

df_positive[0].dropna(inplace = True)
df_positive[1].dropna(inplace = True)
df_negative[0].dropna(inplace = True)
df_negative[1].dropna(inplace = True)
df_train_pos = df_positive[:4264]
df_test_pos = df_positive[4264:]

df_train_neg = df_negative[:4265]
df_test_neg = df_negative[4265:]

df_train = pd.concat([df_train_pos,df_train_neg])
# till 8529
df_test = pd.concat([df_test_pos,df_test_neg])
# 8529 till end
df_mixed = pd.concat([df_train,df_test])


# In[62]:


def train(train_data,train_result):
    lr_model = LogisticRegression(penalty = 'l2', max_iter=200, C=1,random_state = 1)
    lr_model_train = lr_model.fit(train_data, train_result)
    lr_model_test = lr_model.predict(train_data)
    lr_model_score = accuracy_score(train_result,lr_model_test)
    return lr_model_score


# In[63]:


def validation(train_data,train_result):
    lr_model = LogisticRegression(penalty = 'l2', max_iter=200, C=1,random_state = 1)
    score = cross_val_score(lr_model,train_data,train_result,cv=5)
    return score.mean()


# In[64]:


def test(train_data,train_result,test_data,test_result):
    lr_model = LogisticRegression(penalty = 'l2', max_iter=200, C=1,random_state = 1)
    lr_model_train = lr_model.fit(train_data, train_result)
    lr_model_test = lr_model.predict(test_data)
    lr_model_score = accuracy_score(test_result,lr_model_test)
    return lr_model_score


# # 1.Unigram/Bigram

# **unigram**

# In[65]:


df_mixed_1 = df_mixed.copy(deep = True)
df_mixed_1[0] = df_mixed_1[0].apply(word_tokenize)
df_mixed_1[0]=df_mixed_1[0].astype("str")
cv=CountVectorizer(lowercase = True,ngram_range=(1,1))
cv_mixed_1 = cv.fit_transform(df_mixed_1[0])


# In[66]:


cv_train_1 = cv_mixed_1[:8529]
cv_test_1 = cv_mixed_1[8529:]

sen_train = df_mixed_1[:8529][1]
sen_test = df_mixed_1[8529:][1]

train_score = train(cv_train_1,sen_train)
validation_score = validation(cv_train_1,sen_train)
test_score = test(cv_train_1,sen_train,cv_test_1,sen_test)
print("The performance of model unigram")
print("training accuracy: ",train_score)
print("5-fold cross valdation average accuracy: ",validation_score)
print("testing accuracy: ",test_score)


# **bigram**

# In[67]:


cv=CountVectorizer(lowercase = True,ngram_range=(2,2))
cv_mixed_1_b = cv.fit_transform(df_mixed_1[0])
cv_train_1_b = cv_mixed_1_b[:8529]
cv_test_1_b = cv_mixed_1_b[8529:]

sen_train = df_mixed_1[:8529][1]
sen_test = df_mixed_1[8529:][1]

train_score = train(cv_train_1_b,sen_train)
validation_score = validation(cv_train_1_b,sen_train)
test_score = test(cv_train_1_b,sen_train,cv_test_1_b,sen_test)
print("The performance of model bigram")
print("training accuracy: ",train_score)
print("5-fold cross valdation average accuracy: ",validation_score)
print("testing accuracy: ",test_score)


# # 2.Remove Stop words, Unigram/Bigram

# **remove stop words, unigram**

# In[68]:


df_mixed_2 = df_mixed.copy(deep = True)


tokenizer = ToktokTokenizer()
def remove_stopwords(line):
    words = tokenizer.tokenize(line)
    words = [word.strip() for word in words]
    cleaned_words = [word for word in words if word not in stop]
    
    return ' '.join(cleaned_words)
df_mixed_2[0] = df_mixed_2[0].apply(remove_stopwords)    


# In[69]:


cv=CountVectorizer(lowercase = True,ngram_range=(1,1))
cv_mixed_2 = cv.fit_transform(df_mixed_2[0])

cv_train_2 = cv_mixed_2[:8529]
cv_test_2 = cv_mixed_2[8529:]

sen_train = df_mixed_2[:8529][1]
sen_test = df_mixed_2[8529:][1]

train_score = train(cv_train_2,sen_train)
validation_score = validation(cv_train_2,sen_train)
test_score = test(cv_train_2,sen_train,cv_test_2,sen_test)
print("The performance of model stopword+unigram")
print("training accuracy: ",train_score)
print("5-fold cross valdation average accuracy: ",validation_score)
print("testing accuracy: ",test_score)


# **remove stopwords, bigram**

# In[70]:


cv=CountVectorizer(lowercase = True,ngram_range=(2,2))
cv_mixed_2_bigram = cv.fit_transform(df_mixed_2[0])


cv_train_2_bigram = cv_mixed_2_bigram[:8529]
cv_test_2_bigram = cv_mixed_2_bigram[8529:]

train_score = train(cv_train_2_bigram,sen_train)
validation_score = validation(cv_train_2_bigram,sen_train)
test_score = test(cv_train_2_bigram,sen_train,cv_test_2_bigram,sen_test)
print("The performance of model stopword+bigram")
print("training accuracy: ",train_score)
print("5-fold cross valdation average accuracy: ",validation_score)
print("testing accuracy: ",test_score)


# # 3.Remove Stop Words, Lemmatization, Unigram/Bigram

# **remove stop words, lemmatization, unigram**

# In[71]:


df_mixed_3 = df_mixed.copy(deep = True)

tokenizer = ToktokTokenizer()
df_mixed_3[0]=df_mixed_3[0].apply(remove_stopwords)

tokenizer = ToktokTokenizer()
lemmatizer = WordNetLemmatizer()
def lemma(line):
    words = tokenizer.tokenize(line)
    lemma_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemma_words)
df_mixed_3[0] = df_mixed_3[0].apply(lemma)


# In[72]:


cv=CountVectorizer(lowercase = True,ngram_range=(1,1))
cv_mixed_3 = cv.fit_transform(df_mixed_3[0])

cv_train_3 = cv_mixed_3[:8529]
cv_test_3 = cv_mixed_3[8529:]

sen_train = df_mixed_3[:8529][1]
sen_test = df_mixed_3[8529:][1]

train_score = train(cv_train_3,sen_train)
validation_score = validation(cv_train_3,sen_train)
test_score = test(cv_train_3,sen_train,cv_test_3,sen_test)
print("The performance of model stopword+lemmatization+unigram")
print("training accuracy: ",train_score)
print("5-fold cross valdation average accuracy: ",validation_score)
print("testing accuracy: ",test_score)


# **remove stopwords, lemmatization, bigram**

# In[73]:


cv=CountVectorizer(lowercase = True,ngram_range=(2,2))
cv_mixed_3_b = cv.fit_transform(df_mixed_3[0])

cv_train_3_b = cv_mixed_3_b[:8529]
cv_test_3_b = cv_mixed_3_b[8529:]

sen_train = df_mixed_3[:8529][1]
sen_test = df_mixed_3[8529:][1]

train_score = train(cv_train_3_b,sen_train)
validation_score = validation(cv_train_3_b,sen_train)
test_score = test(cv_train_3_b,sen_train,cv_test_3_b,sen_test)
print("The performance of model stopword+lemmatization+bigram")
print("training accuracy: ",train_score)
print("5-fold cross valdation average accuracy: ",validation_score)
print("testing accuracy: ",test_score)


# # 4. Remove Stop Words, Stemming, Unigram/Bigram

# **remove stopwords, stemming, unigram**

# In[74]:


df_mixed_4 = df_mixed.copy(deep = True)

tokenizer = ToktokTokenizer()
df_mixed_4[0]=df_mixed_4[0].apply(remove_stopwords)

tokenizer = ToktokTokenizer()
stemmer = PorterStemmer()
def stem(line):
    words = tokenizer.tokenize(line)
    stem_words = [stemmer.stem(word) for word in words]
    return ' '.join(stem_words)
df_mixed_4[0] = df_mixed_4[0].apply(stem)


# In[75]:


cv=CountVectorizer(lowercase = True,ngram_range=(1,1))
cv_mixed_4 = cv.fit_transform(df_mixed_4[0])

cv_train_4 = cv_mixed_4[:8529]
cv_test_4 = cv_mixed_4[8529:]

sen_train = df_mixed_4[:8529][1]
sen_test = df_mixed_4[8529:][1]

train_score = train(cv_train_4,sen_train)
validation_score = validation(cv_train_4,sen_train)
test_score = test(cv_train_4,sen_train,cv_test_4,sen_test)
print("The performance of model stopwords+stemming+unigram")
print("training accuracy: ",train_score)
print("5-fold cross valdation average accuracy: ",validation_score)
print("testing accuracy: ",test_score)


# **remove stopwords, stemming, bigram**

# In[76]:


cv=CountVectorizer(lowercase = True,ngram_range=(2,2))
cv_mixed_4_b = cv.fit_transform(df_mixed_4[0])

cv_train_4_b = cv_mixed_4_b[:8529]
cv_test_4_b = cv_mixed_4_b[8529:]

sen_train = df_mixed_4[:8529][1]
sen_test = df_mixed_4[8529:][1]

train_score = train(cv_train_4_b,sen_train)
validation_score = validation(cv_train_4_b,sen_train)
test_score = test(cv_train_4_b,sen_train,cv_test_4_b,sen_test)
print("The performance of model stopwords+stemming+bigram")
print("training accuracy: ",train_score)
print("5-fold cross valdation average accuracy: ",validation_score)
print("testing accuracy: ",test_score)


# # 5. Stemming, Unigram/Bigram

# **stemming, unigram**

# In[77]:


df_mixed_5 = df_mixed.copy(deep = True)

tokenizer = ToktokTokenizer()
stemmer = PorterStemmer()
def stem(line):
    words = tokenizer.tokenize(line)
    stem_words = [stemmer.stem(word) for word in words]
    return ' '.join(stem_words)
df_mixed_5[0] = df_mixed_5[0].apply(stem)


# In[78]:


cv=CountVectorizer(lowercase = True,ngram_range=(1,1))
cv_mixed_5 = cv.fit_transform(df_mixed_5[0])

cv_train_5 = cv_mixed_5[:8529]
cv_test_5 = cv_mixed_5[8529:]

sen_train = df_mixed_5[:8529][1]
sen_test = df_mixed_5[8529:][1]

train_score = train(cv_train_5,sen_train)
validation_score = validation(cv_train_5,sen_train)
test_score = test(cv_train_5,sen_train,cv_test_5,sen_test)
print("The performance of model stemming+unigram")
print("training accuracy: ",train_score)
print("5-fold cross valdation average accuracy: ",validation_score)
print("testing accuracy: ",test_score)


# **stemming, bigram**

# In[79]:


cv=CountVectorizer(lowercase = True,ngram_range=(2,2))
cv_mixed_5_b = cv.fit_transform(df_mixed_5[0])

cv_train_5_b = cv_mixed_5_b[:8529]
cv_test_5_b = cv_mixed_5_b[8529:]

sen_train = df_mixed_5[:8529][1]
sen_test = df_mixed_5[8529:][1]

train_score = train(cv_train_5_b,sen_train)
validation_score = validation(cv_train_5_b,sen_train)
test_score = test(cv_train_5_b,sen_train,cv_test_5_b,sen_test)
print("The performance of model stemming+bigram")
print("training accuracy: ",train_score)
print("5-fold cross valdation average accuracy: ",validation_score)
print("testing accuracy: ",test_score)


# # 6. Lemmatization, Unigram/Bigram

# **lemmatization, unigram**

# In[80]:


df_mixed_6 = df_mixed.copy(deep = True)


tokenizer = ToktokTokenizer()
lemmatizer = WordNetLemmatizer()
def lemma(line):
    words = tokenizer.tokenize(line)
    lemma_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemma_words)
df_mixed_6[0] = df_mixed_6[0].apply(lemma)


# In[81]:


cv=CountVectorizer(lowercase = True,ngram_range=(1,1))
cv_mixed_6 = cv.fit_transform(df_mixed_6[0])
cv_train_6 = cv_mixed_6[:8529]
cv_test_6 = cv_mixed_6[8529:]

sen_train = df_mixed_6[:8529][1]
sen_test = df_mixed_6[8529:][1]

train_score = train(cv_train_6,sen_train)
validation_score = validation(cv_train_6,sen_train)
test_score = test(cv_train_6,sen_train,cv_test_6,sen_test)
print("The performance of model lemmatization+unigram")
print("training accuracy: ",train_score)
print("5-fold cross valdation average accuracy: ",validation_score)
print("testing accuracy: ",test_score)


# **lemmatization, bigram**

# In[82]:


cv=CountVectorizer(lowercase = True,ngram_range=(2,2))
cv_mixed_6_b = cv.fit_transform(df_mixed_6[0])

cv_train_6_b = cv_mixed_6_b[:8529]
cv_test_6_b = cv_mixed_6_b[8529:]

sen_train = df_mixed_6[:8529][1]
sen_test = df_mixed_6[8529:][1]

train_score = train(cv_train_6_b,sen_train)
validation_score = validation(cv_train_6_b,sen_train)
test_score = test(cv_train_6_b,sen_train,cv_test_6_b,sen_test)
print("The performance of model lemmatization+bigram")
print("training accuracy: ",train_score)
print("5-fold cross valdation average accuracy: ",validation_score)
print("testing accuracy: ",test_score)

