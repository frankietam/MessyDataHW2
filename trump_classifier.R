###################################################
### MD_ML HW2 
###
### Trump_classifier
### Yeonji Jung, Frankie Tam, Ofer Chen
############################################

############################################
#A) load libraries and trump data

##loading relevant libraries
library(tidyverse)
library(lubridate)
library(plyr)
library(dplyr)
library(naivebayes)

##loading the data
trump <- read.table("trump_data.tsv", header=F, sep="\t", col.names = c("actual","time","post"))
trump <- as.tibble(trump)
trump$time <- as.character(as.character(trump$time))
trump$post <- as.character(as.character(trump$post))

trump_test <- read.table("trump_hidden_test_set.tsv", header=F, sep="\t", col.names = c("actual","time","post"))
trump_test <- as.tibble(trump_test)

############################################
#B) Clean and organize the data

##determine features to use
##! We'd like to use @, #, !, "thank you" as the feature. 
##! Trump uses "I" and @ in his tweets more often than his staff while his staff use # more often than Trump. 
##! Likewise, his staff uses ! and "thank you" more often than Trump. 
##! We also hope to look at the timing of the day the tweet was posted 
##! as a two cateogry of either morning (5am to 10am) or early afternoon (10am to 5pm).
##! This is based on what David Robinsonhe showed with the analysis of the hour of the day 

##! To sum up: @, #, !, "thank you", "I",  the hour of the day.

##clean the missing data which does not have any content of the post
trump<- filter(trump, !trump$post == "")

##make all of letters lowercase
trump$post <- tolower(trump$post)

##add the column of timing and limit the data written during 5am to 5pm to fulfill our interest
trump$time <- ymd_hms(trump$time, quiet = T, tz="EST")
trump <- trump %>%
  mutate(hour=format(as.POSIXct(trump$time, format="%Y-%m-%d %H:%M"), format="%H") )
trump$hour <- as.numeric(as.character(trump$hour))
trump<- filter(trump, between(trump$hour, 5, 17))

trump <- trump %>%
  mutate(timing= between(hour, 5, 9)+1L) 
trump$timing <- as.factor(as.integer(trump$timing))
levels(trump$timing) <- c("afternoon", "morning")

##add the column of whether each feature occurs in the post or not
trump$at <- 
  (1:nrow(trump) %in% c(sapply("@", grep, trump$post, fixed = TRUE)))+0
trump$hash <- 
  (1:nrow(trump) %in% c(sapply("#", grep, trump$post, fixed = TRUE)))+0
trump$excl <- 
  (1:nrow(trump) %in% c(sapply("!", grep, trump$post, fixed = TRUE)))+0
trump$thx <- 
  (1:nrow(trump) %in% c(sapply("thank you", grep, trump$post, fixed = TRUE)))+0
trump$self <- 
  (1:nrow(trump) %in% c(sapply("i ", grep, trump$post, fixed = TRUE)))+0
  
#choose only necessary columns 
trump_clean <- select(trump, actual, timing, at, hash, excl, thx, self)

trump_clean$at<- as.factor(as.numeric(trump_clean$at))
trump_clean$hash<-as.factor(as.numeric(trump_clean$hash))
trump_clean$excl<-as.factor(as.numeric(trump_clean$excl))
trump_clean$thx<-as.factor(as.numeric(trump_clean$thx))
trump_clean$self<- as.factor(as.numeric(trump_clean$self))

############################################
#C) Divide data into a training set and a test set

##create a training set (80% of data) and test set (20% of data) 
index <- sample(2,nrow(trump_clean), replace=TRUE, prob=c(0.8,0.2))
trainData <- trump_clean[index==1,]
testData <- trump_clean[index==2,]

##implement the Naive Bayes classifier
trainClass <- naive_bayes(actual ~ timing+at+hash+thx+self, data = trainData, prior = NULL)

############################################
#D) Select a single model and save the options we used

##fit the model on the training data
trainClass

##apply the fitted model to generate predictions on the test dataset
testPredict <- predict(trainClass, testData, type="class")

##evaluate the model 
cm <- as.matrix(table(actual = testData$actual, prediction = testData$prediction))
n = sum(cm) # number of instances
nc = nrow(cm) # number of classes
diag = diag(cm) # number of correctly classified instances per class 
rowsums = apply(cm, 1, sum) # number of instances per class
colsums = apply(cm, 2, sum) # number of predictions per class
p = rowsums / n # distribution of instances over the actual classes
q = colsums / n # distribution of instances over the predicted classes
accuracy = sum(diag) / n 
precision = diag / colsums 
recall = diag / rowsums 
f1 = 2 * precision * recall / (precision + recall) 
eval <- data.frame(precision, recall, f1) 

##save the predictions as a csv file
testData$prediction <- testPredict

finaloutput <- select(testData, actual, prediction) %>%
  mutate(actual = mapvalues(actual, from=c("Staff", "Trump"), to=c("0", "1"))) %>%
  mutate(prediction = mapvalues(prediction, from=c("Staff", "Trump"), to=c("0", "1")))

write.csv(finaloutput, file="predictions.csv")


