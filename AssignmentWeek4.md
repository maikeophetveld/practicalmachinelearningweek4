# Quantified self movement: Identify correct and incorrect barbell lifts from data

Currently, a large amount concerning personal activity is collected. Typical about this data is that it is recorded how much of a particular activity a person does, but it is rarely quantified how well the activity is performed. In this project, it is the goal to predict in in which way barbell lifts are performed, using data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. 



## Data Processing
Data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants is used in this project. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset)". Download and load the data.

```r
temp <- tempfile()
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv",temp)
traindata_orig <- read.csv(temp, na.strings = c("NA", "#DIV/0!", ""))

temp <- tempfile()
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv",temp)
validationdata_orig <- read.csv(temp, na.strings = c("NA", "#DIV/0!", ""))
```

There are 5 ways in which the barbell lifts are performed, the variable classe contains this information. This is the variable we want to predict.

```r
prop.table(table(traindata_orig$classe))
```

```
## 
##         A         B         C         D         E 
## 0.2843747 0.1935073 0.1743961 0.1638977 0.1838243
```

Before any analysis is performed, the data is cleaned. First, the columns that are not required for predictions are deleted (columns 1-7), and columns with missing values are deleted as well.


```r
nzv <- nearZeroVar(traindata_orig, saveMetrics=TRUE)
traindata <- traindata_orig[,nzv$nzv==FALSE]

temp_train <- traindata
for(i in 1:length(traindata)) {
    if( sum( is.na( traindata[, i] ) ) /nrow(traindata) >= .7) {
        for(j in 1:length(temp_train)) {
            if( length( grep(names(traindata[i]), names(temp_train)[j]) ) == 1)  {
                temp_train <- temp_train[ , -j]
            }   
        } 
    }
}

# Set back to the original variable name
traindata <- temp_train
rm(temp_train)

clean1 <- colnames(traindata)
clean2 <- colnames(traindata[, -59])                  # remove the classe column
validationdata <- validationdata_orig[clean2]         # allow only variables in myTesting that are also in myTraining

dim(traindata)
```

```
## [1] 19622    59
```

After this, the training data is partitioned in a testing and training dataset. This helps for cross-validation and selecting the best model for the predictions.

```r
set.seed(312567)
inTraining  <- createDataPartition(y=traindata$classe, p=0.70, list=FALSE)
training    <- traindata[inTraining,]
test        <- traindata[-inTraining,]
dim(training)
```

```
## [1] 13737    59
```

## Prediction models

### Build models
3 models are evaluated for predicting the classes of barbell lifting. These are gradient boosting, random forests and decision trees. Cross-validation with K=3 is implemented.


```r
fitControl <- trainControl(method='cv', number = 3)
model_gbm <- train(  classe ~ ., training, trControl=fitControl,  method='gbm' )
```

```
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.4545
##      2        1.3255             nan     0.1000    0.3096
##      3        1.1321             nan     0.1000    0.2383
##      4        0.9848             nan     0.1000    0.1967
##      5        0.8634             nan     0.1000    0.1648
##      6        0.7622             nan     0.1000    0.1393
##      7        0.6763             nan     0.1000    0.1229
##      8        0.6010             nan     0.1000    0.1085
##      9        0.5348             nan     0.1000    0.0886
##     10        0.4796             nan     0.1000    0.0803
##     20        0.1661             nan     0.1000    0.0280
##     40        0.0236             nan     0.1000    0.0035
##     60        0.0039             nan     0.1000    0.0006
##     80        0.0007             nan     0.1000    0.0001
##    100        0.0002             nan     0.1000    0.0000
##    120        0.0001             nan     0.1000    0.0000
##    140        0.0000             nan     0.1000    0.0000
##    150        0.0000             nan     0.1000    0.0000
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.7757
##      2        1.1446             nan     0.1000    0.4620
##      3        0.8699             nan     0.1000    0.3214
##      4        0.6789             nan     0.1000    0.2376
##      5        0.5376             nan     0.1000    0.1815
##      6        0.4295             nan     0.1000    0.1413
##      7        0.3453             nan     0.1000    0.1114
##      8        0.2787             nan     0.1000    0.0887
##      9        0.2257             nan     0.1000    0.0709
##     10        0.1832             nan     0.1000    0.0572
##     20        0.0241             nan     0.1000    0.0072
##     40        0.0006             nan     0.1000    0.0001
##     60        0.0001             nan     0.1000    0.0000
##     80        0.0000             nan     0.1000    0.0000
##    100        0.0000             nan     0.1000   -0.0000
##    120        0.0000             nan     0.1000   -0.0000
##    140        0.0000             nan     0.1000   -0.0000
##    150        0.0000             nan     0.1000   -0.0000
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.7751
##      2        1.1446             nan     0.1000    0.4620
##      3        0.8699             nan     0.1000    0.3212
##      4        0.6789             nan     0.1000    0.2377
##      5        0.5377             nan     0.1000    0.1814
##      6        0.4297             nan     0.1000    0.1414
##      7        0.3454             nan     0.1000    0.1116
##      8        0.2788             nan     0.1000    0.0887
##      9        0.2258             nan     0.1000    0.0710
##     10        0.1833             nan     0.1000    0.0572
##     20        0.0241             nan     0.1000    0.0073
##     40        0.0005             nan     0.1000    0.0001
##     60        0.0000             nan     0.1000    0.0000
##     80        0.0000             nan     0.1000    0.0000
##    100        0.0000             nan     0.1000   -0.0000
##    120        0.0000             nan     0.1000   -0.0000
##    140        0.0000             nan     0.1000   -0.0000
##    150        0.0000             nan     0.1000   -0.0000
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.4569
##      2        1.3236             nan     0.1000    0.3124
##      3        1.1295             nan     0.1000    0.2413
##      4        0.9803             nan     0.1000    0.1961
##      5        0.8594             nan     0.1000    0.1576
##      6        0.7607             nan     0.1000    0.1440
##      7        0.6727             nan     0.1000    0.1217
##      8        0.5983             nan     0.1000    0.1076
##      9        0.5326             nan     0.1000    0.0886
##     10        0.4777             nan     0.1000    0.0841
##     20        0.1643             nan     0.1000    0.0259
##     40        0.0236             nan     0.1000    0.0037
##     60        0.0038             nan     0.1000    0.0006
##     80        0.0007             nan     0.1000    0.0001
##    100        0.0002             nan     0.1000    0.0000
##    120        0.0000             nan     0.1000    0.0000
##    140        0.0000             nan     0.1000    0.0000
##    150        0.0000             nan     0.1000    0.0000
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.7767
##      2        1.1448             nan     0.1000    0.4612
##      3        0.8701             nan     0.1000    0.3208
##      4        0.6791             nan     0.1000    0.2379
##      5        0.5377             nan     0.1000    0.1815
##      6        0.4297             nan     0.1000    0.1412
##      7        0.3454             nan     0.1000    0.1115
##      8        0.2788             nan     0.1000    0.0889
##      9        0.2258             nan     0.1000    0.0709
##     10        0.1833             nan     0.1000    0.0572
##     20        0.0241             nan     0.1000    0.0073
##     40        0.0004             nan     0.1000    0.0001
##     60        0.0000             nan     0.1000    0.0000
##     80        0.0000             nan     0.1000    0.0000
##    100        0.0000             nan     0.1000    0.0000
##    120        0.0000             nan     0.1000    0.0000
##    140        0.0000             nan     0.1000    0.0000
##    150        0.0000             nan     0.1000    0.0000
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.7783
##      2        1.1446             nan     0.1000    0.4615
##      3        0.8698             nan     0.1000    0.3212
##      4        0.6788             nan     0.1000    0.2369
##      5        0.5375             nan     0.1000    0.1812
##      6        0.4295             nan     0.1000    0.1412
##      7        0.3452             nan     0.1000    0.1114
##      8        0.2787             nan     0.1000    0.0888
##      9        0.2257             nan     0.1000    0.0711
##     10        0.1832             nan     0.1000    0.0572
##     20        0.0241             nan     0.1000    0.0073
##     40        0.0004             nan     0.1000    0.0001
##     60        0.0000             nan     0.1000    0.0000
##     80        0.0000             nan     0.1000    0.0000
##    100        0.0000             nan     0.1000    0.0000
##    120        0.0000             nan     0.1000    0.0000
##    140        0.0000             nan     0.1000    0.0000
##    150        0.0000             nan     0.1000    0.0000
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.4518
##      2        1.3258             nan     0.1000    0.3113
##      3        1.1326             nan     0.1000    0.2392
##      4        0.9847             nan     0.1000    0.1971
##      5        0.8631             nan     0.1000    0.1581
##      6        0.7645             nan     0.1000    0.1447
##      7        0.6763             nan     0.1000    0.1221
##      8        0.6010             nan     0.1000    0.1081
##      9        0.5352             nan     0.1000    0.0890
##     10        0.4798             nan     0.1000    0.0846
##     20        0.1652             nan     0.1000    0.0262
##     40        0.0236             nan     0.1000    0.0034
##     60        0.0041             nan     0.1000    0.0006
##     80        0.0010             nan     0.1000    0.0001
##    100        0.0003             nan     0.1000    0.0000
##    120        0.0001             nan     0.1000    0.0000
##    140        0.0000             nan     0.1000    0.0000
##    150        0.0000             nan     0.1000    0.0000
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.7747
##      2        1.1447             nan     0.1000    0.4631
##      3        0.8699             nan     0.1000    0.3210
##      4        0.6790             nan     0.1000    0.2369
##      5        0.5378             nan     0.1000    0.1812
##      6        0.4298             nan     0.1000    0.1413
##      7        0.3455             nan     0.1000    0.1115
##      8        0.2790             nan     0.1000    0.0885
##      9        0.2260             nan     0.1000    0.0710
##     10        0.1835             nan     0.1000    0.0572
##     20        0.0241             nan     0.1000    0.0072
##     40        0.0006             nan     0.1000    0.0002
##     60        0.0000             nan     0.1000    0.0000
##     80        0.0000             nan     0.1000    0.0000
##    100        0.0000             nan     0.1000    0.0000
##    120        0.0000             nan     0.1000    0.0000
##    140        0.0000             nan     0.1000   -0.0000
##    150        0.0000             nan     0.1000   -0.0000
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.7762
##      2        1.1448             nan     0.1000    0.4618
##      3        0.8700             nan     0.1000    0.3215
##      4        0.6790             nan     0.1000    0.2373
##      5        0.5377             nan     0.1000    0.1811
##      6        0.4297             nan     0.1000    0.1413
##      7        0.3454             nan     0.1000    0.1114
##      8        0.2789             nan     0.1000    0.0887
##      9        0.2259             nan     0.1000    0.0712
##     10        0.1833             nan     0.1000    0.0572
##     20        0.0241             nan     0.1000    0.0073
##     40        0.0005             nan     0.1000    0.0001
##     60        0.0000             nan     0.1000    0.0000
##     80        0.0000             nan     0.1000   -0.0000
##    100        0.0000             nan     0.1000   -0.0000
##    120        0.0000             nan     0.1000   -0.0000
##    140        0.0000             nan     0.1000   -0.0000
##    150        0.0000             nan     0.1000   -0.0000
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.4474
##      2        1.3268             nan     0.1000    0.3128
##      3        1.1331             nan     0.1000    0.2420
##      4        0.9837             nan     0.1000    0.1971
##      5        0.8622             nan     0.1000    0.1590
##      6        0.7632             nan     0.1000    0.1446
##      7        0.6747             nan     0.1000    0.1227
##      8        0.6002             nan     0.1000    0.1078
##      9        0.5344             nan     0.1000    0.0891
##     10        0.4791             nan     0.1000    0.0843
##     20        0.1649             nan     0.1000    0.0261
##     40        0.0235             nan     0.1000    0.0034
##     50        0.0095             nan     0.1000    0.0015
```

```r
save(model_gbm, file='./ModelFitGBM.RData')

model_dectree <- train(  classe ~ ., training,  trControl=fitControl,  method='rpart' )
save(model_dectree, file='./ModelFitDecisonTree.RData')

model_rf <- train(   classe ~ ., training,  trControl=fitControl,  method='rf',  ntree=100 )
save(model_rf, file='./ModelFitRF.RData')
```

### Model Evaluation


```r
pred_dectree <- predict(model_dectree, newdata=test)
cm_dectree <- confusionMatrix(pred_dectree, test$classe)
pred_gbm <- predict(model_gbm, newdata=test)
cm_gbm <- confusionMatrix(pred_gbm, test$classe)
pred_rf <- predict(model_rf, newdata=test)
cm_rf <- confusionMatrix(pred_rf, test$classe)
AccuracyResults <- data.frame(
  Model = c('DecTree', 'GBM', 'RF'),
  Accuracy = rbind(cm_dectree$overall[1], cm_gbm$overall[1], cm_rf$overall[1])
)
print(AccuracyResults)
```

```
##     Model  Accuracy
## 1 DecTree 0.6616822
## 2     GBM 0.9991504
## 3      RF 0.9993203
```

Random forest and Gradient boosting give good results, the decision tree model performs significantly worse. Because the accuracy is already high, we will not investigate a model that is a combination of these three model types.

### Generate predictions
A set of predictions is generated for the validation data with the random forest model, this data is evaluated in the Coursera course online. As expected, the results are accurate.


```r
pred_validation <- predict(model_rf, newdata=validationdata)
pred_validation
```

```
##  [1] A A A A A A A A A A A A A A A A A A A A
## Levels: A B C D E
```

