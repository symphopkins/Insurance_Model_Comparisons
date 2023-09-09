#############################################
#                                           #
# Author:     Symphony Hopkins              #
# Date:       Date                          #
# Subject:    Final Project                 #
# Class:      DSCI 502                      #
# Section:    01W                           #         
# Instructor: Juan Munoz Robayo             #
# File Name:  FinalProject_Hopkins_Symphony.R
#                                           # 
#############################################


########################
# 1.  Data Preparation #
########################

#     a.  Load the dataset insurance.csv into memory.
#         Answer: See code.
insurance <- read.csv("~/Documents/Maryville_University/DSCI_512/Final_Project/insurance.csv")
View(insurance)

#     b.  In the data frame, transform the variable charges by seting
#         insurance$charges = log(insurance$charges). Do not transform
#         it outside of the data frame.
#         Answer: See code.

insurance$charges = log(insurance$charges)

#     c.  Using the data set from 1.b, use the model.matrix() function
#         to create another data set that uses dummy variables in place
#         of categorical variables. Verify that the first column only has
#         ones (1) as values, and then discard the column only after
#         verifying it has only ones as values.
#         Answer: See code.

#creating dummy data set
insurance_dummies <- model.matrix(~., data = insurance)

#verifying that the first column has only has ones as values
all(insurance_dummies[, 1] == 1)

#discarding the first column (intercept) and converting insurance_dummies to dataframe
insurance_dummies <- insurance_dummies[, -1]
insurance_dummies <- as.data.frame(insurance_dummies)

#     d.  Use the sample() function with set.seed equal to 1 to generate
#         row indexes for your training and tests sets, with 2/3 of the
#         row indexes for your training set and 1/3 for your test set. Do
#         not use any method other than the sample() function for
#         splitting your data.
#         Answer: See code.

#setting seed for reproducibility
set.seed(1)

#generating train and test sets
index <- sample(1:nrow(insurance), (2/3)*nrow(insurance))


#     e.  Create a training and test data set from the data set created in
#         1.b using the training and test row indexes created in 1.d.
#         Unless otherwise stated, only use the training and test
#         data sets created in this step.
#         Answer: See code.

train_1 <- insurance[index,]
test_1 <- insurance[-index,]

#     f.  Create a training and test data set from data set created in 1.c
#         using the training and test row indexes created in 1.d
#         Answer: See code.

train_2 <- insurance_dummies[index,]
test_2 <- insurance_dummies[-index,]

#################################################
# 2.  Build a multiple linear regression model. #
#################################################

#     a.  Perform multiple linear regression with charges as the
#         response and the predictors are age, sex, bmi, children,
#         smoker, and region. Print out the results using the
#         summary() function. Use the training data set created in
#         step 1.e to train your model.
#         Answer: See code.

insurance_lm_1 <- lm(charges ~., data=train_1)
summary(insurance_lm_1)

#     b.  Is there a relationship between the predictors and the
#         response? 
#         Answer: With a significance level of 0.05, we observed that all of the  
#         predictors displayed statistical significance, as evidenced by their 
#         p-values being less than the set threshold. Additionally, the Adjusted 
#         R-squared value of 0.7819 indicates a moderate-to-strong correlation 
#         between the predictor variables and the response. 

#     c.  Does sex have a statistically significant relationship to the
#         response?
#         Answer: Sex has a p-value of 0.027847, making it less than our 
#         significance level, so it is statistically significant. (Note: 
#         Since the training data set from step 1.e. does not have the 
#         categorical variables converted to numerical variables, the columns 
#         were automatically dummified when they were passed through the lm() 
#         function; hence why, the column for “sex” says “sexmale”.)

#     d.  Perform best subset selection using the stepAIC() function
#         from the MASS library, choose best model based on AIC. For
#         the "direction" parameter in the stepAIC() method, set
#         direction="backward"
#         Answer: All predictors were included in the best subset selection.

#importing library
library(MASS)

#performing backward stepwise selection
insurance_stepAIC <- stepAIC(insurance_lm_1, direction='backward')
insurance_stepAIC

#     e.  Compute the test error of the best model in #2d based on AIC
#         using LOOCV using trainControl() and train() from the caret
#         library. Report the MSE by squaring the reported RMSE.
#         Answer: The MSE is 0.1833684.

#importing library
library(caret)

#performing LOOCV -- from the stepwise selection, all predictors were 
#included in the best subset
train_control <- trainControl(method = "LOOCV")
insurance_LOOCV <- train(charges ~ ., data = train_1, method = "lm",
                    trControl = train_control)
insurance_LOOCV

#calculating MSE from RMSE
(insurance_LOOCV$results$RMSE)^2

#     f.  Calculate the test error of the best model in #2d based on AIC
#         using 10-fold Cross-Validation. Use train and trainControl
#         from the caret library. Refer to model selected in #2d based
#         on AIC. Report the MSE.
#         Answer: The MSE is 0.1792372.

#performing 10-fold cross validation
train_control_10CV <- trainControl(method='CV', number=10)
insurance_10CV <- train(charges ~ ., data = train_1, method = "lm",
                         trControl = train_control_10CV)
insurance_10CV

#calculating MSE from RMSE
(insurance_10CV$results$RMSE)^2

#     g.  Calculate and report the test MSE using the best model from 
#         2.d and the test data set from step 1.e.
#         Answer: The MSE is 0.231291. 

y_pred = predict(insurance_stepAIC, newdata=test_1)
mlr_mse <- mean((y_pred-test_1$charges)^2)
mlr_mse

#     h.  Compare the test MSE calculated in step 2.f using 10-fold
#         cross-validation with the test MSE calculated in step 2.g.
#         How similar are they?
#         Answer: The MSE from step 2.f. was lower compared to the 
#         MSE from step 2.g., which means the model's performance was
#         (slightly) better when evaluated using 10-fold Cross-Validation.


######################################
# 3.  Build a regression tree model. #
######################################

#     a.  Build a regression tree model using function tree(), where
#         charges is the response and the predictors are age, sex, bmi,
#         children, smoker, and region.
#         Answer: See code. 

#importing library
library(tree)

#building tree model
insurance_tree <- tree(charges ~., data = train_1)
summary(insurance_tree)

#     b.  Find the optimal tree by using cross-validation and display
#         the results in a graphic. Report the best size.
#         Answer: The best size is 3.

insurance_tree_cv <- cv.tree(insurance_tree)
plot(insurance_tree_cv$size, insurance_tree_cv$dev, type = 'b')


#     c.  Justify the number you picked for the optimal tree with
#         regard to the principle of variance-bias trade-off.
#         Answer: We want a model with a low variance and a low
#         bias, however, there is a trade-off whenever we lower
#         the bias or variance, so we need to find a balance. Choosing
#         a size of 3 would help achieve a balance compared to choosing
#         a size of 5 (the size with the lowest test error as seen in plot)
#         because it would be complex enough to capture intricate patterns
#         of the data, but not too complex to where we would have a high
#         variance. With this size, we would (hopefully) achieve low variance
#         and low bias.

#     d.  Prune the tree using the optimal size found in 3.b
#         Answer: See code. 

insurance_pruned <- prune.tree(insurance_tree, best = 3)

#     e.  Plot the best tree model and give labels.
#         Answer: See code. 

plot(insurance_pruned)
text(insurance_pruned, pretty=0)

#     f.  Calculate the test MSE for the best model.
#         Answer: The MSE is 0.7035004.

y_pred_tree <-predict(insurance_pruned, newdata = test_1)
tree_mse <- mean((y_pred_tree-test_1$charges)^2)
tree_mse

####################################
# 4.  Build a random forest model. #
####################################

#     a.  Build a random forest model using function randomForest(),
#         where charges is the response and the predictors are age, sex,
#         bmi, children, smoker, and region.
#         Answer: See code. 

#importing library
library(randomForest)

#building random forest model
insurance_rf <- randomForest(charges ~., data = train_1)
insurance_rf

#     b.  Compute the test error using the test data set.
#         Answer: The MSE is 0.1797157.

y_pred_rf = predict(insurance_rf, newdata=test_1)
rf_mse <- mean((y_pred_rf-test_1$charges)^2)
rf_mse 

#     c.  Extract variable importance measure using the importance()
#         function.
#         Answer: See code. 

importance(insurance_rf)

#     d.  Plot the variable importance using the function, varImpPlot().
#         Which are the top 3 important predictors in this model?
#         Answer: The top 3 important predictors are smoker, age, and bmi.

varImpPlot(insurance_rf)


############################################
# 5.  Build a support vector machine model #
############################################

#     a.  The response is charges and the predictors are age, sex, bmi,
#         children, smoker, and region. Please use the svm() function
#         with radial kernel and gamma=5 and cost = 50.
#         Answer: See code. 

#importing library
library(e1071)

#building svm 
insurance_svm <- svm(charges ~., data=train_1, kernel='radial',
                     gamma=5, cost=50)
insurance_svm

#     b.  Perform a grid search to find the best model with potential
#         cost: 1, 10, 50, 100 and potential gamma: 1,3 and 5 and
#         potential kernel: "linear","polynomial","radial" and
#         "sigmoid". And use the training set created in step 1.e.
#         Answer: See code. (Note: We kept receiving the following error: 
#         WARNING: reaching max number of iterations, which means the optimizer 
#         did not converge in the given number of iterations. Since it is 
#         not an error, it is not a significant problem but it may or may not 
#         have affected the results: https://stackoverflow.com/questions
#         /34245785/weird-error-message-when-tuning-svm-with-polynomial-
#         kernel-warning-reaching-m) 

insurance_grid_search <- tune(svm, charges ~., data=train_1, ranges=list(kernel=
                              c('linear','polynomial','radial','sigmoid'), cost=
                              c(1, 10, 50, 100),gamma=c(1, 3, 5)))

#     c.  Print out the model results. What are the best model
#         parameters?
#         Answer: The best model parameters: kernel=polynomial, cost=1,
#         and gamma=1.

print(insurance_grid_search)

#     d.  Forecast charges using the test dataset and the best model
#         found in c).
#         Answer: See code. 

y_pred_gs <- predict(insurance_grid_search$best.model, newdata=test_1)

#     e.  Compute the MSE (Mean Squared Error) on the test data.
#         Answer: The MSE is 0.1719588.

svm_mse <- mean((y_pred_gs-test_1$charges)^2)
svm_mse



#############################################
# 6.  Perform the k-means cluster analysis. #
#############################################

#     a.  Use the training data set created in step 1.f and standardize
#         the inputs using the scale() function.
#         Answer: See code. 

scaled_train_2 <- scale(train_2)

#     b.  Convert the standardized inputs to a data frame using the
#         as.data.frame() function.
#         Answer: See code. 

scaled_train_2 <- as.data.frame(scaled_train_2)


#     c.  Determine the optimal number of clusters, and use the
#         gap_stat method and set iter.max=20. Justify your answer.
#         It may take longer running time since it uses a large dataset.
#         Answer: The optimal number of clusters is 5 as determined by
#         the vertical line on the plot.

#importing library
library('cluster')
library('factoextra')

#finding optimal number of clusters
insurance_clust <- fviz_nbclust(scaled_train_2, kmeans, method='gap_stat',
                                iter.max=20)
insurance_clust

#     d.  Perform k-means clustering using the optimal number of
#         clusters found in step 6.c. Set parameter nstart = 25
#         Answer: See code. 

insurance_clust_opt <- kmeans(scaled_train_2, 5, nstart=25)

#     e.  Visualize the clusters in different colors, setting parameter
#         geom="point"
#         Answer: See code. 

fviz_cluster(insurance_clust_opt, data=scaled_train_2, geom='point')


######################################
# 7.  Build a neural networks model. #
######################################

#     a.  Using the training data set created in step 1.f, create a 
#         neural network model where the response is charges and the
#         predictors are age, sexmale, bmi, children, smokeryes, 
#         regionnorthwest, regionsoutheast, and regionsouthwest.
#         Please use 1 hidden layer with 1 neuron. Do not scale
#         the data.
#         Answer: See code. 

#importing library
library(neuralnet)

#building neural network
insurance_nn <- neuralnet(charges ~., data=train_2, hidden=c(1))

#     b.  Plot the neural network.
#         Answer: See code. 

plot(insurance_nn)

#     c.  Forecast the charges in the test dataset.
#         Answer: See code. 

y_pred_nn <- compute(insurance_nn, test_2)

#     d.  Compute test error (MSE).
#         Answer: The MSE is 0.8737059.

nn_mse <- mean((y_pred_nn$net.result-test_2$charges)^2)
nn_mse

################################
# 8.  Putting it all together. #
################################

#     a.  For predicting insurance charges, your supervisor asks you to
#         choose the best model among the multiple regression,
#         regression tree, random forest, support vector machine, and
#         neural network models. Compare the test MSEs of the models
#         generated in steps 2.g, 3.f, 4.b, 5.e, and 7.d. Display the names
#         for these types of these models, using these labels:
#         "Multiple Linear Regression", "Regression Tree", "Random Forest", 
#         "Support Vector Machine", and "Neural Network" and their
#         corresponding test MSEs in a data.frame. Label the column in your
#         data frame with the labels as "Model.Type", and label the column
#         with the test MSEs as "Test.MSE" and round the data in this
#         column to 4 decimal places. Present the formatted data to your
#         supervisor and recommend which model is best and why.
#         Answer: We would recommend the Support Vector Machine model 
#         because it has the lowest MSE with a value of 0.1720. In this case, 
#         we used MSE as the determining metric, but in general, when choosing 
#         the best model, we should consider using multiple evaluation metrics 
#         to capture all aspects of the models. 

#creating dataframe
Model.Type <- c("Multiple Linear Regression", "Regression Tree", "Random Forest", 
                "Support Vector Machine","Neural Network")
Test.MSE <- c(mlr_mse, tree_mse, rf_mse, svm_mse, nn_mse)
models_mse <- data.frame(Model.Type, Test.MSE)
models_mse$Test.MSE <- round(models_mse$Test.MSE, 4)
View(models_mse)

#     b.  Another supervisor from the sales department has requested
#         your help to create a predictive model that his sales
#         representatives can use to explain to clients what the potential
#         costs could be for different kinds of customers, and they need
#         an easy and visual way of explaining it. What model would
#         you recommend, and what are the benefits and disadvantages
#         of your recommended model compared to other models?
#         Answer: The best model for visualization is the regression tree. 
#         Some of the benefits of using a regression tree is that they easier 
#         to explain compared to other models (e.g., linear regression); 
#         and they can easily handle categorical predictors. Some of the 
#         disadvantages of regression trees is that they don’t have the same 
#         level of predictive accuracy as other models; and they can be very 
#         non-robust. 

#     c.  The supervisor from the sales department likes your regression
#         tree model. But she says that the sales people say the numbers
#         in it are way too low and suggests that maybe the numbers
#         on the leaf nodes predicting charges are log transformations
#         of the actual charges. You realize that in step 1.b of this
#         project that you had indeed transformed charges using the log
#         function. And now you realize that you need to reverse the
#         transformation in your final output. The solution you have
#         is to reverse the log transformation of the variables in 
#         the regression tree model you created and redisplay the result.
#         Follow these steps:
#
#         i.   Copy your pruned tree model to a new variable.
#              Answer: See code. 

insurance_pruned_copy <- insurance_pruned

#         ii.  In your new variable, find the data.frame named
#              "frame" and reverse the log transformation on the
#              data.frame column yval using the exp() function.
#              (If the copy of your pruned tree model is named 
#              copy_of_my_pruned_tree, then the data frame is
#              accessed as copy_of_my_pruned_tree$frame, and it
#              works just like a normal data frame.).
#               Answer: See code. 

insurance_pruned_copy$frame$yval <- exp(insurance_pruned_copy$frame$yval)

#         iii. After you reverse the log transform on the yval
#              column, then replot the tree with labels.
#              Answer: See code. 

plot(insurance_pruned_copy)
text(insurance_pruned_copy, pretty=0)

#End Assignment