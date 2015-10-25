Practical Machine Learning - Course Project

For this project, we are given data from accelerometers on the belt, forearm, arm, and dumbell of 6 research study participants. Our training data consists of accelerometer data and a label identifying the quality of the activity the participant was doing. Our testing data consists of accelerometer data without the identifying label. Our goal is to predict the labels for the test set observations.

Below is the code I used when creating the model, estimating the out-of-sample error, and making predictions. I also include a description of each step of the process.

Load Data

data<-read.csv("./pml-training.csv", sep=",", header=TRUE)
test<-read.csv("./pml-testing.csv", sep=",", header=TRUE)

library(caret)
set.seed(123)

# training and cross-validation set
inTrain<-createDataPartition(y=data$classe, p=0.6, list=FALSE)
myTraining<-data[inTrain,]
myTesting<-data[-inTrain,]
dim(myTraining); dim(myTesting)
# 11776,160; # 7846, 160


Data Pre-processing

Transform 1: remove near zero predictors

nzv<-nearZeroVar(myTraining)
myTraining2<-myTraining[, -nzv]
# 11776, 106


Transform 2: remove missing value of predictor more than 60%

thresholdNA<-0.6
checkNA<-function(col){
    (   sum(is.na(col))>=(thresholdNA*length(col))   )
}
lotNAs<-sapply(myTraining2, checkNA)
# lotNAs
myTraining3<-myTraining2[, !lotNAs]
dim(myTraining3)
# 11776, 59

Transform3: remove first 1-6 columns

myTraining4<-myTraining3[,-c(1,2,3,4,5,6)]
dim(myTraining4)
# 11776, 53

Transform 4: Apply the same transformation to myTesting dataset
clean1 <- colnames(myTraining4)
myTesting2<-myTesting[clean1]
dim(myTesting2)
# 7846 53

Transform 5: Remove classe, apply the same transformation to test dataset
# Last column is classe
clean2 <- colnames(myTraining4[, -53])
test2<-test[clean2]
dim(test2)
# 20,52

any(is.na(myTraining4))


Model Building


# Fit decision tree======================================
library(rattle)
library(e1071)
library(rpart.plot)

modFit_tree<-train(classe~., method="rpart", data=myTraining4)
print(modFit_tree$finalModel)
plot(modFit_tree$finalModel, uniform=TRUE)

fancyRpartPlot(modFit_tree$finalModel)
# fancyRpartPlot(modFit_tree$finalModel,cex=.5,under.cex=1,shadow.offset=0)

predict_tree=predict(modFit_tree, myTesting2)
confusionMatrix(myTesting2$classe, predict_tree)


# Fit Random Forest====================================
library(randomForest)

modelFit_rf2<-randomForest(classe~., data=myTraining4)

predict_rf2=predict(modelFit_rf2, myTesting2)
confusionMatrix(myTesting2$classe, predict_rf2)


Re-training the Selected Model

Random Forest Model is 99.43% accuracy when testing on cross-validation dataset





Create Prediction File

prediction<-predict(modelFit_rf2, test2)
prediction

pml_write_files=function(x){
    n=length(x)
    for(i in 1:n){
        filename=paste0("problem_id_", i, ".txt")
        write.table(x[i], file=filename, quote=FALSE, row.names=FALSE, col.names=FALSE)
    }
}
pml_write_files(prediction)
