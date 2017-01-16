
#read Trial promo data
library(DMwR)
Promodata = read.csv("/Users/Suki/KE/2 semester/CA/ISBA/Workshop 2B／trialPromoResults_new.csv")
table(Promodata$decision)
Promodata = Promodata[,2:11]
Promodata$decision <- ifelse(Promodata$decision == 'None', 0,1)
Promodata$decision = as.factor(Promodata$decision)


set.seed(46)

#check target proportion
print(prop.table(table(Promodata$decision)))

#handle imbalance using Smote
Promodata <- SMOTE(decision ~ ., Promodata, perc.over = 200,k=50)
print(prop.table(table(Promodata$decision)))
Promodata$sex = as.factor(Promodata$sex)
Promodata$mstatus = as.factor(Promodata$mstatus)
Promodata$occupation = as.factor(Promodata$occupation)
Promodata$education = as.factor(Promodata$education)

#split into trainng and testing
library(rpart)
library(caTools)
split = sample.split(Promodata$decision, SplitRatio = 0.75)
train.data = subset(Promodata, split==TRUE)
test.data = subset(Promodata, split==FALSE)

print(prop.table(table(train.data$decision)))
print(prop.table(table(test.data$decision)))

#Cross validation function
library(caret)
ctrl = trainControl(method = "cv", number = 10)
#decision tree model

tree = train(decision ~ ., data = train.data, method = "rpart", trControl = ctrl)
pred_tree = predict(tree, newdata = test.data)#,type="class")
 
cf_tree = table(pred_tree, test.data$decision)
accuracy_tree = sum(diag(cf_tree))/sum(cf_tree)

library(e1071)
tune.out_1 <- tune(svm, decision~., data = train.data, kernel="radial", 
                   ranges = list(gamma=c(0.1,0.5,1,2,4), 
                                 cost = c(0.1,1,10,100,1000)
                   ))
pred_svm = predict(tune.out_1$best.model, newdata = test.data, type = "class")
cf_svm = table(pred_svm, test.data$decision)
accuracy_svm = sum(diag(cf_svm))/sum(cf_svm)


#logistic regression
logistic = train(decision~., data = train.data, method = "LogitBoost",trControl = ctrl )

pred_logistic = predict(logistic, newdata = test.data)#,type = "response")
cf_logistic = table(pred_logistic, test.data$decision)
accuracy_logistic = sum(diag(cf_logistic))/sum(cf_logistic)

#Random Forest

library(randomForest)
random_forest = train(decision~., data = train.data, method = "ranger",trControl = ctrl )
pred_rf = predict(random_forest, test.data)
cf_rf = table(pred_rf, test.data$decision)
accuracy_rf = sum(diag(cf_rf))/sum(cf_rf)

#Neural Network
library(nnet)
train.data$sex = as.numeric(train.data$sex)

train.data$mstatus = as.numeric(train.data$mstatus)
train.data$occupation = as.numeric(train.data$occupation)
train.data$education = as.numeric(train.data$education)


test.data$sex = as.numeric(test.data$sex)
test.data$mstatus = as.numeric(test.data$mstatus)
test.data$occupation = as.numeric(test.data$occupation)
test.data$education = as.numeric(test.data$education)


train.data$decision = as.factor(train.data$decision)
test.data$decision = as.factor(test.data$decision)

neural = train(x=train.data, y=train.data$decision, method = "nnet",trControl = ctrl )
pred_neural = predict(neural, newdata = test.data)
cf_neural = table(pred_neural, test.data$decision)
accuracy_neural = sum(diag(cf_neural))/sum(cf_neural)

#KNN


knn_model = kNN(decision~., train.data,test.data, norm = TRUE, k=10)
cf_knn = table(knn_model,test.data$decision)
accuracy_knn = sum(diag(cf_knn))/sum(cf_knn)

#treebag 
library(caret)
ctrl = trainControl(method = "cv", number = 5)
tb = train(decision~., data = train.data, method = "treebag",trControl = ctrl )
pred_tree_bag = predict(tb$finalModel,test.data,type="class")
cf_tree_bag = table(pred_tree_bag,test.data$decision)
accuracy_tree_bag = sum(diag(cf_tree_bag))/sum(cf_tree_bag)

#read actual data with 4000 records
actual_data = read.csv("/Users/Suki/KE/2 semester/CA/ISBA/Workshop 2B／custdatabase.csv")
actual_data_copy = actual_data
actual_data = actual_data[,2:11]

actual_data$decision <- ifelse(actual_data$decision == 'None', 0,1)
actual_data$decision= as.factor(actual_data$decision)

#predict actual data
pred_actual_tree = predict(tree, newdata = actual_data)#, type = "class")
cf_actual_tree = table(pred_actual_tree, actual_data$decision)
accuracy_actual_tree = sum(diag(cf_actual_tree))/sum(cf_actual_tree)

pred_svm_actual = predict(tune.out_1$best.model, newdata = actual_data, type = "class")
cf_svm_actual = table(pred_svm_actual, actual_data$decision)
accuracy_svm_actual = sum(diag(cf_svm_actual))/sum(cf_svm_actual)

pred_logistic_actual = predict(logistic, newdata = actual_data)#, type = "response")
cf_logistic_actual = table(pred_logistic_actual, actual_data$decision)
accuracy_logistic_actual = sum(diag(cf_logistic_actual))/sum(cf_logistic_actual)

pred_rf_actual = predict(random_forest, newdata = actual_data)#, type = "class")
cf_rf_actual = table(pred_rf_actual, actual_data$decision)
accuracy_rf_actual = sum(diag(cf_rf_actual))/sum(cf_rf_actual)

actual_data$sex = as.numeric(actual_data$sex)

actual_data$mstatus = as.numeric(actual_data$mstatus)
actual_data$occupation = as.numeric(actual_data$occupation)
actual_data$education = as.numeric(actual_data$education)
actual_data$decision = as.factor(actual_data$decision)

pred_neural_actual = predict(neural, newdata = actual_data)#, type = "class")
cf_neural_actual = table(pred_neural_actual, actual_data$decision)
accuracy_neural_actual = sum(diag(cf_neural_actual))/sum(cf_neural_actual)

pred_tree_bag_actual = predict(tb, newdata = actual_data)#, type = "class")
cf_tree_bag_actual = table(pred_tree_bag_actual, actual_data$decision)
accuracy_tree_bag_actual = sum(diag(cf_tree_bag_actual))/sum(cf_tree_bag_actual)

knn_model1 = kNN(decision~.,train.data,actual_data,norm = TRUE, k=5)
table(knn_model1,actual_data$decision)

#take ensemble

df = cbind(pred_actual_tree,pred_logistic_actual,pred_neural_actual)
#voting
decision = ifelse(rowSums(df)>4,2,1)
df = cbind(df,decision,actual_data_copy$decision)
df = data.frame(df)
colnames(df)<-c("tree","logistic","neural","final","actual")
new_data = cbind(actual_data_copy, df$final)#actual_data_copy$decision)
new_data$`df$final` = ifelse(new_data$`df$final` == "2",1,0) 
data_model2 = subset(new_data, `df$final`!='0')
data_model2  = data_model2[,1:11]
data_model2_copy = data_model2
data_model2 = data_model2[,2:11]

#create model 2, to predict A/B 

library(DMwR)
Promodata_AB = read.csv("/Users/Suki/KE/2 semester/CA/ISBA/Workshop 2B／trialPromoResults_AB.csv")
Promodata_AB = Promodata_AB[,2:11]
Promodata_AB$decision = as.factor(droplevels(Promodata_AB$decision))
table(Promodata_AB$decision)
#check target proportion
print(prop.table(table(Promodata_AB$decision)))

#split into training and testing
split = sample.split(Promodata_AB$decision, SplitRatio = 0.75)
train.data = subset(Promodata_AB, split==TRUE)
test.data = subset(Promodata_AB, split==FALSE)

print(prop.table(table(train.data$decision)))
print(prop.table(table(test.data$decision)))

#svm
library(e1071)
tune.out <- tune(svm, decision~., data = train.data, kernel="radial", 
                 ranges = list(gamma=c(0.1,0.5,1,2,4), 
                               cost = c(0.1,1,10,100,1000)
                 ))
pred_svm_AB = predict(tune.out$best.model, newdata = test.data, type = "class")
cf_svm_AB = table(pred_svm_AB, test.data$decision)
accuracy_svm_AB = sum(diag(cf_svm_AB))/sum(cf_svm_AB)

#Random Forest
library(randomForest)
set.seed(46)
train.data$decision = as.factor(train.data$decision)
random_forest_model = train(decision~., data = train.data, method = "ranger",trControl = ctrl)
pred_rf_model = predict(random_forest_model, newdata = test.data, type = "raw")
cf_rf_1 = table(pred_rf_model, test.data$decision)
accuracy_rf_1 = sum(diag(cf_rf_1))/sum(cf_rf_1)

tree_ab = train(decision~., data=train.data, method ="rpart", trControl = ctrl)
pred_tree_ab = predict(tree_ab, newdata = test.data)
cf_tree_ab = table(pred_tree_ab,test.data$decision)
accuracy_tree_ab = sum(diag(cf_tree_ab))/sum(cf_tree_ab)


#nnet_ab = train(decision~.,data = train.data, method = "nnet",trControl = ctrl)
nnet_ab = nnet(decision~.,data=train.data, size=7,maxit = 2000)
pred_nnet_ab = predict(nnet_ab, newdata = test.data, type = "class")
cf_nnet_ab = table(pred_nnet_ab,test.data$decision)
accuracy_nnet_ab = sum(diag(cf_nnet_ab))/sum(cf_nnet_ab)

train.data$sex = as.numeric(train.data$sex)

train.data$mstatus = as.numeric(train.data$mstatus)
train.data$occupation = as.numeric(train.data$occupation)
train.data$education = as.numeric(train.data$education)


test.data$sex = as.numeric(test.data$sex)
test.data$mstatus = as.numeric(test.data$mstatus)
test.data$occupation = as.numeric(test.data$occupation)
test.data$education = as.numeric(test.data$education)

nnet_ab_2 = nnet(decision~.,data = train.data,size=7,maxit=1500)
pred_nnet_ab_2 = predict(nnet_ab_2,newdata = test.data, type = "class")

#Pass BUY records to model 2
pred_tree_ab_actual = predict(tree_ab, newdata = data_model2, type = "raw")
cf_tree_ab_actual = table(pred_tree_ab_actual,data_model2$decision)

pred_rf_ab_actual = predict(random_forest_model,newdata = data_model2, type = "raw")

model_ab = cbind(data_model2_copy, pred_tree_ab_actual)
write.csv(model_ab, file = "Final model.csv")
