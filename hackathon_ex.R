# This example is to demonstrate the use of word embeddings (using pretrained models) and xgboost in R
# It is recommended to use a package like caret to tune parameters and evaluating
# This is prepared for the LWLow Hackathon event (Liv-GDSL, 26/01/18), using airbnb data (listings, London)

library(data.table)
library(tidytext)
library(ggplot2)
library(caret)

setwd('/home/hhn/working/data-fun/hackathon-lml')
# download the trained vectors here: http://nlp.stanford.edu/data/glove.6B.zip
glove_word_vectors <- fread('glove.6B.100d.txt')

airbnb_listings <- fread('listings.csv')
tokens_by_names <- unnest_tokens(airbnb_listings,input='name',output = 'token',drop = F)
tokens_by_names[,count:=.N,by=token][order(-count)]
tokens_by_names<-bind_tf_idf(tokens_by_names,term = token,document=id,n=count)
tokens_by_names<- tokens_by_names[,.(id,name,token,idf,tf_idf,room_type,price)]
tokens_by_names <- merge(tokens_by_names,glove_word_vectors,by.x='token',by.y='V1')
tokens_by_names.weighted <-tokens_by_names[,.SD*tf_idf,.SDcols=c(8:107)]

tokens_by_names.weighted$id<- tokens_by_names$id
tokens_by_names.weighted$name<- tokens_by_names$name
tokens_by_names.weighted$room_type<- as.numeric(as.factor(tokens_by_names$room_type))-1
tokens_by_names.weighted$price<- tokens_by_names$price



name_vectors <-  tokens_by_names.weighted[,lapply(.SD,sum),by=.(id,name,room_type,price),.SDcols=c(1:100),]
set.seed(101)

train_idx <- createDataPartition(name_vectors$room_type, p = 0.8, times = 1)
train_data <- name_vectors[train_idx$Resample1,]
test_data <- name_vectors[-train_idx$Resample1,]

# randomForest is a good baseline as it needs less parameter tuning
library(randomForest)
rf.model <- randomForest(x=train_data[,c(5:104)],y=as.factor(train_data$room_type),xtest=test_data[,c(5:104)],ytest = as.factor(test_data$room_type),do.trace=50,keep.forest=TRUE)
test <- test_data[,.(name,room_type,price)]
test$pred <- predict(rf.model,test_data[,c(5:104)])
confusion.rf <- confusionMatrix(factor(test$pred),as.factor(test_data$room_type),mode="everything")

# svm is a very good classifier for text data
library(e1071)
svm.model <- svm(x=as.matrix(train_data[,c(5:104)]),y=as.factor(train_data$room_type))
test <- test_data[,.(name,room_type,price)]
test$pred <- predict(svm.model,as.matrix(test_data[,c(5:104)]))
confusion.svm <- confusionMatrix(factor(test$pred),as.factor(test_data$room_type),mode="everything")


# finally try xgboost
library(xgboost)
# convert from normal data table to DMatrix
train_data.dmatrix <- xgb.DMatrix(data=as.matrix(train_data[,c(5:104)]),label=train_data$room_type)
test_data.dmatrix <- xgb.DMatrix(data=as.matrix(test_data[,c(5:104)]),label=test_data$room_type) 
# create a watchlist to for each round evaluation
watchlist <- list(test = test_data.dmatrix, train = train_data.dmatrix)
# try crossvalidating first
# model <- xgb.cv(data=train_data.dmatrix,metrics=list("merror","mlogloss"), nfold=10, objective = "multi:softprob", subsample=0.75,colsample_bytree=0.75,max_depth=8, watchlist=watchlist, eta = 0.00075, nthread = 15, verbose = 1, nround = 1000, num_class=length(unique(train_data$room_type)))
# and then train using full training dataset
model <- xgb.train(data=train_data.dmatrix,metrics=list("merror","mlogloss"),  objective = "multi:softprob", subsample=0.7,colsample_bytree=0.7, watchlist=watchlist, eta = 0.01, nthread = 15, verbose = 1, nround = 1000, num_class=length(unique(train_data$room_type)))
test <- test_data[,.(name,room_type,price)]
test$pred <- max.col(predict(model,test_data.dmatrix,reshape=T))-1
confusion.xgb <- confusionMatrix(factor(test$pred),factor(test$room_type),mode="everything")
