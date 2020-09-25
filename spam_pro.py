# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 00:56:40 2020

@author: Koray
"""

import pandas as pd 

data=pd.read_csv("spam_dataset.csv")

data=data.drop(['Unnamed: 0'],axis=1)

#%% Text ve Label datalarımızı alıyoruz

text=data.iloc[:,1:2]

y=data.iloc[:,2:]

#%% Porter Stemmer ve Stop Words ile Veri Önişleme 

import re 

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

stopwords=set(stopwords.words('english'))

#stopwords ve yapılan işlemler için tüm text satırlarında olması için for döngüsü oluşturduk

data_s=[]
for i in range(5171):
    data_new=re.sub('[^a-zA-Z]',' ',data['text'][i])
    data_new=data_new.lower()
    data_new=data_new.split()
    data_new=[ps.stem(voc) for voc in data_new if not voc in stopwords]
    data_new=' '.join(data_new)
    data_s.append(data_new)
    
#%% Count Vectorizer ile 0-1'lerdan olusturduk 

from sklearn.feature_extraction.text import CountVectorizer

c_vec=CountVectorizer(max_features=1000)

c_vec.fit(data_s)

x=c_vec.transform(data_s).toarray()

#%% Train ve Test Split

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=0)

#%% Logistic Regression

from sklearn.linear_model import LogisticRegression

l_reg=LogisticRegression()

l_reg.fit(x_train,y_train)


#%% Model Sonuçlarımız

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

y_pred=l_reg.predict(x_test)

cm=confusion_matrix(y_test,y_pred)

c_repo=classification_report(y_test,y_pred)

acc_sco=accuracy_score(y_test,y_pred)*100

print("confusion_matrix:",cm)
print("classifaction_report:",c_repo)
print("accuracy_score:",acc_sco)

#%% Seaborn ile heatmap

import seaborn as sns

sns.heatmap(confusion_matrix(y_test,y_pred),annot=True)







































    
    










