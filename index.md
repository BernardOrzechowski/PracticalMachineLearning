# Practical Machine Learning: Course project writeup.  

We will use the Human Activity Recognition data found at http://groupware.les.inf.puc-rio.br/har to build a prediction model. The goal will be to estimate the "classe" variable that describes how well an activity was performed.

## Loading and preprocessing the data.  

**Step 1. Load the data.**  

First we set the locale to English and seed to 1 to be able to reproduce the results.

```r
Sys.setlocale("LC_ALL", "English")
```

```
## [1] "LC_COLLATE=English_United States.1252;LC_CTYPE=English_United States.1252;LC_MONETARY=English_United States.1252;LC_NUMERIC=C;LC_TIME=English_United States.1252"
```

```r
set.seed(1)
```


Lets start with loading the raw training and test data. Assumtion: they arealready in the project folder. 

```r
rawTrainingData <- read.csv(file = "pml-training.csv", header = TRUE, sep = ",", 
    row.names = NULL, na.strings = c("NA", ""), stringsAsFactors = FALSE)

rawTestData <- read.csv(file = "pml-testing.csv", header = TRUE, sep = ",", 
    row.names = NULL, na.strings = c("NA", "", " "), stringsAsFactors = FALSE)
```



**Step 2. Process/transform the data into a format suitable for further analysis.**

As shown below we have made the follwoing steps to tidy the data set  

1. all columns with NA values will be removed. Analysis has shown that there are either 0 or 19216 NAs values in particular columns. Because 19216 is nearly 100% of the size of the training data set we will remove those columns completely.  
2. remove additionally the columns that don't seem to have any relationships to accelerator data: user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp, new_window, num_window . They could possibly predict very well data in the training data set, but would  probably fail to predict data in the test set (the relationship is rather artificial, real prediciton model shouldn't be build upon them).  
3. Do the same (as in point 2) operation on test data set.  
4. use the function nearZeroVar to show that the remaining columns shouldn't be automaticly removed.  


```r
# check if there are NAs
countNAs <- apply(rawTrainingData, 2, function(x) {
    sum(is.na(x))
})
# how many NAs are in particular columns(show only distinct values)
unique(countNAs)
```

```
## [1]     0 19216
```

```r


# check length of the raw training data set
length(rawTrainingData$classe)
```

```
## [1] 19622
```

```r


# Therefore we can remove the colums which have at least one NA
rawTrainingData <- rawTrainingData[, -which(names(rawTrainingData) %in% names(countNAs)[countNAs > 
    0])]

# remove additionally the columns that dont seem to have any relationships
# to accelerator data
# user_name,raw_timestamp_part_1,raw_timestamp_part_2,cvtd_timestamp,new_window,num_window
rawTrainingData <- rawTrainingData[, -which(names(rawTrainingData) %in% c("user_name", 
    "raw_timestamp_part_1", "raw_timestamp_part_2", "cvtd_timestamp", "new_window", 
    "num_window", "X"))]

# do the same on the test data set
rawTestData <- rawTestData[, -which(names(rawTestData) %in% names(countNAs)[countNAs > 
    0])]
rawTestData <- rawTestData[, -which(names(rawTestData) %in% c("user_name", "raw_timestamp_part_1", 
    "raw_timestamp_part_2", "cvtd_timestamp", "new_window", "num_window", "X"))]

# load the caret library
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
# We should not remove automatically any of the remaining columns, the
# nearZeroVar fuction showss it
nsv <- nearZeroVar(rawTrainingData, saveMetrics = TRUE)
nsv
```

```
##                      freqRatio percentUnique zeroVar   nzv
## roll_belt                1.102       6.77811   FALSE FALSE
## pitch_belt               1.036       9.37723   FALSE FALSE
## yaw_belt                 1.058       9.97350   FALSE FALSE
## total_accel_belt         1.063       0.14779   FALSE FALSE
## gyros_belt_x             1.059       0.71348   FALSE FALSE
## gyros_belt_y             1.144       0.35165   FALSE FALSE
## gyros_belt_z             1.066       0.86128   FALSE FALSE
## accel_belt_x             1.055       0.83580   FALSE FALSE
## accel_belt_y             1.114       0.72877   FALSE FALSE
## accel_belt_z             1.079       1.52380   FALSE FALSE
## magnet_belt_x            1.090       1.66650   FALSE FALSE
## magnet_belt_y            1.100       1.51870   FALSE FALSE
## magnet_belt_z            1.006       2.32902   FALSE FALSE
## roll_arm                52.338      13.52563   FALSE FALSE
## pitch_arm               87.256      15.73234   FALSE FALSE
## yaw_arm                 33.029      14.65702   FALSE FALSE
## total_accel_arm          1.025       0.33636   FALSE FALSE
## gyros_arm_x              1.016       3.27693   FALSE FALSE
## gyros_arm_y              1.454       1.91622   FALSE FALSE
## gyros_arm_z              1.111       1.26389   FALSE FALSE
## accel_arm_x              1.017       3.95984   FALSE FALSE
## accel_arm_y              1.140       2.73672   FALSE FALSE
## accel_arm_z              1.128       4.03629   FALSE FALSE
## magnet_arm_x             1.000       6.82397   FALSE FALSE
## magnet_arm_y             1.057       4.44399   FALSE FALSE
## magnet_arm_z             1.036       6.44685   FALSE FALSE
## roll_dumbbell            1.022      83.78351   FALSE FALSE
## pitch_dumbbell           2.277      81.22516   FALSE FALSE
## yaw_dumbbell             1.132      83.14137   FALSE FALSE
## total_accel_dumbbell     1.073       0.21914   FALSE FALSE
## gyros_dumbbell_x         1.003       1.22821   FALSE FALSE
## gyros_dumbbell_y         1.265       1.41678   FALSE FALSE
## gyros_dumbbell_z         1.060       1.04984   FALSE FALSE
## accel_dumbbell_x         1.018       2.16594   FALSE FALSE
## accel_dumbbell_y         1.053       2.37489   FALSE FALSE
## accel_dumbbell_z         1.133       2.08949   FALSE FALSE
## magnet_dumbbell_x        1.098       5.74865   FALSE FALSE
## magnet_dumbbell_y        1.198       4.30129   FALSE FALSE
## magnet_dumbbell_z        1.021       3.44511   FALSE FALSE
## roll_forearm            11.589      11.08959   FALSE FALSE
## pitch_forearm           65.983      14.85577   FALSE FALSE
## yaw_forearm             15.323      10.14677   FALSE FALSE
## total_accel_forearm      1.129       0.35674   FALSE FALSE
## gyros_forearm_x          1.059       1.51870   FALSE FALSE
## gyros_forearm_y          1.037       3.77637   FALSE FALSE
## gyros_forearm_z          1.123       1.56457   FALSE FALSE
## accel_forearm_x          1.126       4.04648   FALSE FALSE
## accel_forearm_y          1.059       5.11161   FALSE FALSE
## accel_forearm_z          1.006       2.95587   FALSE FALSE
## magnet_forearm_x         1.012       7.76679   FALSE FALSE
## magnet_forearm_y         1.247       9.54031   FALSE FALSE
## magnet_forearm_z         1.000       8.57711   FALSE FALSE
## classe                   1.470       0.02548   FALSE FALSE
```

```r
# no columns that should be removed
nsvColumns <- rownames(nsv[nsv[, "zeroVar"] > 0, ])
nsvColumns
```

```
## character(0)
```

```r

```


We have now 53 variables (including classe). We will try to predict classe based on the others variables.  

**Step 3. Build the prediction model.**  

Because of high memory usage and to test the final model before applying to the test set provided in this assignment we will take only a subset from the original data to build the prediction model. From the original 19622 samples a subset counting 15000 samples will be used choosen randomly from the training data set. We will then use it to perfomr cross validation of the model (4 fold). The rest of 4622 samples will serve as a validation set (build from the training set, put aside) to assess the expected out of sample error. Because we are trying to solve a categorization problem with a lot of variables, we will use the random forest approach.  



```r
# make model

# Because of high memory usage we have to take only a subset from the
# original data to build the prediciton model.

trainingIndex <- sample(nrow(rawTrainingData), 15000)
# create the training set
trainingSet <- rawTrainingData[trainingIndex, ]
# create the validation set to assess the expected out of sample error
validationSet <- rawTrainingData[-trainingIndex, ]
# factor the classe variable
trainingSet$classe <- as.factor(trainingSet$classe)
validationSet$classe <- as.factor(validationSet$classe)
# training options, cross validation 4 fold
trainCtrl <- trainControl(method = "cv", number = 4)
# define the model using the sampled 15000 samples using random forest.
fittedModel <- train(classe ~ ., trControl = trainCtrl, data = trainingSet, 
    method = "rf")
```

```
## Loading required package: randomForest
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

```r
# display the model details
fittedModel
```

```
## Random Forest 
## 
## 15000 samples
##    52 predictors
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (4 fold) 
## 
## Summary of sample sizes: 11250, 11250, 11249, 11251 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa  Accuracy SD  Kappa SD
##   2     1         1      0.001        0.001   
##   30    1         1      0.002        0.003   
##   50    1         1      0.001        0.001   
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 2.
```

```r
# get the final model from the model object
finalModel <- fittedModel$finalModel
```



**Step 4. Predict classe variable in test data.**  


We will predict the classe variables on the test data (20 samples) and assign them to the answers vector that will be used to build files that will serve as input for the Course Project Submission part. 20 files each containing only one letter (A, B, C, D or E) will be created.


```r
# create the vector with predictions for test samples.
answers <- (as.character(predict(finalModel, newdata = rawTestData)))
answers
```

```
##  [1] "B" "A" "B" "A" "A" "E" "D" "B" "A" "A" "B" "C" "B" "A" "E" "E" "A"
## [18] "B" "B" "B"
```

```r

# this function is taken from the assignment site
pml_write_files = function(x) {
    n = length(x)
    for (i in 1:n) {
        filename = paste0("problem_id_", i, ".txt")
        write.table(x[i], file = filename, quote = FALSE, row.names = FALSE, 
            col.names = FALSE)
    }
}

# generate the 20 files
pml_write_files(answers)
```


The generated model predicted correctly 20 of 20 values in the provided test data set. 

**Step 5. Prediction model considerations**  

Lets check how well the model works on the training data set (19622 samples).  

First the confusion matrix showing us the prediction capabilities on the training data set and the validation test set:



```r
confusionMatrix(trainingSet$classe, predict(finalModel, trainingSet))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 4265    0    0    0    0
##          B    0 2926    0    0    0
##          C    0    0 2564    0    0
##          D    0    0    0 2449    0
##          E    0    0    0    0 2796
## 
## Overall Statistics
##                                 
##                Accuracy : 1     
##                  95% CI : (1, 1)
##     No Information Rate : 0.284 
##     P-Value [Acc > NIR] : <2e-16
##                                 
##                   Kappa : 1     
##  Mcnemar's Test P-Value : NA    
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    1.000    1.000    1.000    1.000
## Specificity             1.000    1.000    1.000    1.000    1.000
## Pos Pred Value          1.000    1.000    1.000    1.000    1.000
## Neg Pred Value          1.000    1.000    1.000    1.000    1.000
## Prevalence              0.284    0.195    0.171    0.163    0.186
## Detection Rate          0.284    0.195    0.171    0.163    0.186
## Detection Prevalence    0.284    0.195    0.171    0.163    0.186
## Balanced Accuracy       1.000    1.000    1.000    1.000    1.000
```

```r
confusionMatrix(validationSet$classe, predict(finalModel, validationSet))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1315    0    0    0    0
##          B    3  864    4    0    0
##          C    0    1  856    1    0
##          D    0    0   11  756    0
##          E    0    0    0    2  809
## 
## Overall Statistics
##                                         
##                Accuracy : 0.995         
##                  95% CI : (0.993, 0.997)
##     No Information Rate : 0.285         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.994         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.998    0.999    0.983    0.996    1.000
## Specificity             1.000    0.998    0.999    0.997    0.999
## Pos Pred Value          1.000    0.992    0.998    0.986    0.998
## Neg Pred Value          0.999    1.000    0.996    0.999    1.000
## Prevalence              0.285    0.187    0.188    0.164    0.175
## Detection Rate          0.285    0.187    0.185    0.164    0.175
## Detection Prevalence    0.285    0.188    0.186    0.166    0.175
## Balanced Accuracy       0.999    0.998    0.991    0.997    1.000
```

```r

```


It's clear that on the training set this model performed very well (Accuracy = 1). On the validation set the Accuracy = 0.9952. It could be a sign ov overfittingm but overwfitting hasn't been observed on the the assignment test data set, as the model predicted correctly 20 of 20 test samples. Therefore the expeced out of sample error is 1 - 0.9952 = 0.0048.

