
"""
Created on Thu Jan 25 15:55:37 2018

@author: hhn

This is to demonstrate the use of word2vec/doc2vec/xgboost in gensim and xgboost libraries
The sample is prepared for Liverpool LWLOL hackathon, 26/01/18, using airbnb listings data for London

"""

import pandas as pd
import numpy as np
import gensim


listings = pd.read_csv('/home/hhn/working/data-fun/hackathon-lml/listings.csv',header=0)

# see some stats
listings.describe()

# this example is only for the listing data, one can get better word2vec model by using reviews data, or pretrained model (as in the R example)

listings['tokenized_names'] =   [str(name).lower().strip(' ').split(' ') for name in listings['name']]
w2vmodel = gensim.models.Word2Vec(listings['tokenized_names'] , size=100, window=4, min_count=3, workers=10)

# see there are how many words 
len(w2vmodel.wv.vocab)

# let's have a look at the most similar words of some important words
typical_words = ['flat','chinatown','museum','garden','tube','cozy']
for w in typical_words:
    print(w2vmodel.wv.most_similar(w))

# below result is very interesting for such short texts. It works quite okay for adjective and popular nouns, but not very well for rare token (chinatown, for example only pull the context words instead of similar words)
#[('apartment', 0.9248437881469727), ('apt', 0.8039981126785278), ('flat,', 0.7627623677253723), ('maisonette', 0.7051661014556885), ('cottage', 0.6833502650260925), ('flat!', 0.6709610223770142), ('flat.', 0.6697197556495667), ('townhouse', 0.6551421880722046), ('house', 0.6537779569625854), ('appartment', 0.6390859484672546)]
#[('walk,', 0.988723874092102), ('exec', 0.9885861873626709), ('(sleeps', 0.9883356094360352), ('apartment-', 0.9882869124412537), ('portered', 0.9880273342132568), ('heights', 0.9876649379730225), ('slg', 0.9873733520507812), ('glenthurston', 0.9869656562805176), ('apple', 0.9865999221801758), ('altitude', 0.9858288764953613)]
#[('british', 0.9919556975364685), ('metro,', 0.9802559614181519), ('week', 0.979275107383728), ('jubilee', 0.9781695604324341), ('queensway', 0.9779500365257263), ('each', 0.9772701263427734), ('barking', 0.9765881896018982), ('northern', 0.9764289259910583), ('britishmuseum', 0.9762008190155029), ('stratford!', 0.9759805202484131)]
#[('terrace', 0.854571521282196), ('charming', 0.803985059261322), ('garden,', 0.7881942987442017), ('garden!', 0.7853189706802368), ('gorgeous', 0.7832192182540894), ('balcony', 0.7674139738082886), ('deck', 0.7486076354980469), ('fulham', 0.7480264902114868), ('beautiful', 0.7466801404953003), ('jacuzzi', 0.7462540864944458)]
#[('station', 0.950753927230835), ('heathrow', 0.8684635162353516), ('tube!', 0.855481743812561), ('train', 0.852750301361084), ('underground', 0.8362025022506714), ('*15', 0.8336499929428101), ('metro*', 0.8331146836280823), ('line', 0.8306269645690918), ('everywhere', 0.8299165964126587), ('excel', 0.8119915723800659)]
#[('cosy', 0.9623124599456787), ('nice', 0.9365255236625671), ('comfortable', 0.8978907465934753), ('comfy', 0.8938279151916504), ('cute', 0.8818572163581848), ('bright', 0.8736275434494019), ('small', 0.8704944849014282), ('cheap', 0.8649131059646606), ('clean', 0.8555126190185547), ('lovely', 0.847968578338623)]
# now try some clustering
from sklearn.cluster import KMeans
num_of_clusters = 20
clusters = KMeans( n_clusters = num_of_clusters,  random_state =1 ).fit_predict( w2vmodel.wv.syn0)
df = pd.DataFrame({ 'token':w2vmodel.wv.index2word, 'cluster':clusters })

for k in range(num_of_clusters):
    print(k,df[df.cluster==k]['token'].str.cat(sep=','))
    


#now save your time by doing doc2vec (instead of aggregating words)

from gensim.models.doc2vec import LabeledSentence
labeled_sentences = [LabeledSentence(name, [' '.join(name)]) for i,name in enumerate(listings['tokenized_names'])]
    
d2vmodel = gensim.models.Doc2Vec(labeled_sentences , size=100, window=3, min_count=5, workers=10)    

# okay, training done, now try to find similar sentences
sentence = 'double room near kensington'
sentence_vector = d2vmodel.infer_vector(sentence.split())
d2vmodel.similar_by_vector(sentence_vector,topn=20)


########## now try to predict accomodation type from accomodation title
# 1. prepare titles vectors and labels
title_vectors = np.array([d2vmodel.infer_vector(tokenized_names) for tokenized_names in listings['tokenized_names']])
labels =  np.array(listings['room_type'])

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
encoded_labels = le.fit_transform(labels)

# 2. split into train and test
from sklearn.model_selection import train_test_split

title_train,title_test, title_vectors_train,title_vectors_test,encoded_labels_train,encoded_labels_test = train_test_split( np.array(listings['name']), title_vectors,encoded_labels,stratify = encoded_labels, train_size = 0.8,random_state = 1)

# 3. prepare for xgboost classifier
import xgboost as xgb
training_all =  xgb.DMatrix(data= title_vectors_train, label=encoded_labels_train)
testing_all =  xgb.DMatrix(data= title_vectors_test, label=encoded_labels_test)
param = {'eta':0.007, 'max_depth':5,'nthread':15,'num_class':len(np.unique(le.classes_)),'seed':1,'objective':'multi:softprob','eval_metric':['merror','mlogloss'],'subsample':0.8, 'debug_verbose': 2}
trained_xgb = xgb.train(param, training_all, num_boost_round=1000, early_stopping_rounds=10 ,verbose_eval=True,evals=[(training_all,'train'),(testing_all,'test')])

test_preds = trained_xgb.predict(testing_all)
topk = 1
predicted_labels = le.inverse_transform(np.argsort(test_preds)[:,-topk:])
predicted_proba = np.sort(test_preds)[:,-topk:]

from sklearn.metrics import classification_report
print(classification_report(encoded_labels_test,np.argsort(test_preds)[:,-1:]))

# 4. try the base line: using random forest
from sklearn.ensemble import *
rf = RandomForestClassifier(n_estimators=500,n_jobs=-1,random_state=1)
rf.fit(title_vectors_train,encoded_labels_train)
test_preds = rf.predict(title_vectors_test)
print(classification_report(encoded_labels_test,test_preds))
# Note that the results from RandomForest sufffer imbalance classes problem. So it is a good idea to upsample a bit for the shared accomodation

# the score is lower than the R version. This error mainly comes from our vectors quality (we trained the vectors using our own corpus, which is very limited -- and a lots of noise) 
# so I think it would be better use a pretrained vector, either from Google word2vec or Glove as in the R version
