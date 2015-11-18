#setwd("/Users/QQL/Desktop/2T HardDrive Backup/Info3-Online_Test Learning/Coursera 8 Practical Machine Learning")
#setwd("./Project")
setwd("C:/Users/liqi/Desktop/Kaggle")
setwd("./Coursera Jawbone")
data<-read.csv("./pml-training.csv", sep=",", header=TRUE)
test<-read.csv("./pml-testing.csv", sep=",", header=TRUE)

library(caret)
set.seed(123)

# training and cross-validation set
# Method 1 is using createDataPartition; # Method 2 is stritify split
inTrain<-createDataPartition(y=data$classe, p=0.6, list=FALSE)
myTraining<-data[inTrain,]
myTesting<-data[-inTrain,]
dim(myTraining); dim(myTesting); dim(test)
# 11776,160; # 7846, 160  # 20, 160

# Data Exploration



# Transform1: remove near zero predictors
nzv<-nearZeroVar(myTraining)
myTraining2<-myTraining[, -nzv]
# 11776, 107


# Transform2 remove missing value of predictor more than 60%
thresholdNA<-0.6
checkNA<-function(col){
    (   sum(is.na(col))>=(thresholdNA*length(col))   )
}
lotNAs<-sapply(myTraining2, checkNA)
# lotNAs
myTraining3<-myTraining2[, !lotNAs]
dim(myTraining3)
# 11776, 59

# Transform3 remove first 1-6 columns. They are index, user_name, time_stamps
myTraining4<-myTraining3[,-c(1,2,3,4,5,6)]
dim(myTraining4)
# 11776, 53

# Testset transform: Apply the same transformation from training dataset to myTesting dataset
clean1 <- colnames(myTraining4)
myTesting2<-myTesting[clean1]
dim(myTesting2)
# 7846 53

# Prediction dataset transform: Remove classe, apply the same transformation from training dataset to prediction dataset
# Last column is classe
clean2 <- colnames(myTraining4[, -53])
test2<-test[clean2]
dim(test2)
# 20,52

any(is.na(myTraining4))

# Ensure myTraining and testing dataset has the same type of data 
# (since it come from separate file)
for (i in 1:length(test2) ) {
    for(j in 1:length(myTraining4)) {
        if( length( grep(names(myTraining4[i]), names(test2)[j]) ) ==1)  {
            class(test2[j]) <- class(myTraining4[i])
        }      
    }      
}


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
# accuracy is 50.69%

# Fit Random Forest
library(randomForest)

# too slow # not working
# modelFit_rf<-train(classe~., method="rf", data=myTraining4, prox=TRUE)

modelFit_rf2<-randomForest(classe~., data=myTraining4)

predict_rf2=predict(modelFit_rf2, myTesting2)
confusionMatrix(myTesting2$classe, predict_rf2)
# Model is 99.43% accuracy when testing on cross-validation dataset



# Generate output files
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
#1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
#B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
# All 20 test data shows correct


