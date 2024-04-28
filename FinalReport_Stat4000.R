student_performance_math_file_path <- "C:/Users/13342/Downloads/STAT 4000/student+performance/student/student-mat.csv"
student_performance_port_file_path <- "C:/Users/13342/Downloads/STAT 4000/student+performance/student/student-por.csv"
wine_quality_red_file_path <- "C:/Users/13342/Downloads/STAT 4000/wine+quality/winequality-red.csv"

############################################################################
# Multiple Linear Regression
############################################################################
# Load necessary libraries
library(MASS)
library(ISLR2)

# Read the CSV file into R
data <- read.csv(student_performance_math_file_path, header = TRUE, sep = ";")

# Filter out non-quantitative variables
quantitative_data <- data[, sapply(data, is.numeric)]

# Fit a multiple linear regression model
lm.fit <- lm(G3 ~ ., data = quantitative_data)

# Display the summary of the regression model
summary(lm.fit)

coefficients <- coef(lm.fit)
significance <- summary(lm.fit)$coefficients[, 4]

# Filter significant predictors (p-value < 0.05)
significant_predictors <- names(coefficients)[significance < 0.05]

cat("Significant predictors and their coefficients:\n")
print(coefficients[significant_predictors])

############################################################################
# Naive Bayes
############################################################################
# Load the required libraries
library(e1071)

# Read your student performance data from the CSV file
student_data <- read.csv(student_performance_math_file_path, sep = ";", header = FALSE)

# Fit a Naive Bayes model using all predictors except the target variable
nb.fit <- naiveBayes(G3 ~ ., data = data)

# Display the Naive Bayes model summary
nb.fit

# Make predictions using the Naive Bayes model
nb.class <- predict(nb.fit, data)

# G3 is the target variable, create a confusion matrix to assess model performance
table(nb.class, data$G3)

# Compute accuracy rate of the Naive Bayes model
mean(nb.class == data$G3)

# Generate estimates of the probability that each observation belongs to a particular class
nb.preds <- predict(nb.fit, data, type = "raw")
head(nb.preds)

############################################################################
# Regression Tree
############################################################################
# Load necessary library
library(tree)
data <- read.csv(wine_quality_red_file_path, sep = ";")

# Check the structure of your data
str(data)

data <- na.omit(data)
# Set seed and create training set
set.seed(1)
train <- sample(1:nrow(data), nrow(data) / 2)

# Fit regression tree to training data
tree_data <- tree(quality ~ ., data, subset = train)

# Summary of regression tree
summary(tree_data)

# Plot regression tree
plot(tree_data)
text(tree_data, pretty = 0)

# Use cv.tree() to determine if pruning improves performance
cv_data <- cv.tree(tree_data)
plot(cv_data$size, cv_data$dev, type = "b")

# Prune the tree
prune_data <- prune.tree(tree_data, best = 5)
plot(prune_data)
text(prune_data, pretty = 0)

# Make predictions on the test set using unpruned tree
yhat <- predict(tree_data, newdata = data[-train, ])
test_labels <- data[-train, "quality"]
plot(yhat, test_labels)
abline(0, 1)
mean((yhat - test_labels)^2)

# Make predictions on the test set using pruned tree
yhat <- predict(prune_data, newdata = data[-train, ])
test_labels <- data[-train, "quality"]
plot(yhat, test_labels)
abline(0, 1)
mean((yhat - test_labels)^2)

###########################################################################
# Random Forest
###########################################################################
library(randomForest)

# Set seed for reproducibility
set.seed(1)

# Set the target variable
target_variable <- "quality"

# Set the training set
train <- sample(1:nrow(data), nrow(data) / 2)

# Extract predictors and target variable
predictors <- setdiff(names(data), target_variable)

# Perform random forest
rf_model <- randomForest(x = data[predictors], y = data[[target_variable]], subset = train, mtry = 6, importance = TRUE)

# Summary of random forest model
rf_model

# Make predictions on the test set using random forest model
predictions_rf <- predict(rf_model, newdata = data[-train, ])

# Plot predictions against true values
plot(predictions_rf, data[-train, target_variable])
abline(0, 1)

# Calculate test set MSE for random forest model
mse_rf <- mean((predictions_rf - data[-train, target_variable])^2)
mse_rf

# View variable importance
importance(rf_model)

# Plot variable importance
varImpPlot(rf_model)

###########################################################################
# Boosting
###########################################################################
library(gbm)

# Set seed for reproducibility
set.seed(1)

# Fit boosted regression trees
boost_model <- gbm(quality ~ ., data = data[train, ], distribution = "gaussian", n.trees = 5000, interaction.depth = 4)

# Summary of boosted model
summary(boost_model)

# Make predictions on the test set using boosted model
yhat_boost <- predict(boost_model, newdata = data[-train, ], n.trees = 5000)

# Calculate test set MSE for boosted model
mean((yhat_boost - test_labels)^2)

# Perform boosting with a different shrinkage parameter
boost_model <- gbm(quality ~ ., data = data[train, ], distribution = "gaussian", n.trees = 5000, interaction.depth = 4, shrinkage = 0.2, verbose = FALSE)
yhat_boost <- predict(boost_model, newdata = data[-train, ], n.trees = 5000)
mean((yhat_boost - test_labels)^2)

###########################################################################
# Logistic Regression
###########################################################################
install.packages("glmnet")
library(e1071)
library(dplyr)
library(ggplot2)

data_math=read.csv(student_performance_math_file_path,sep=";",header=TRUE)
data_por=read.csv(student_performance_port_file_path,sep=";",header=TRUE)


d3=merge(data_math,data_por,by=c("school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet"))
print(nrow(d3)) # 382 students

data <- bind_rows(data_math, data_por)

plot(data$G1, data$G3, xlab = "1st Semester Grades", ylab = "Final Grade", xlim = c(0,20), ylim = c(0,25))
plot(data$G2, data$G3, xlab = "2nd Semester Grades", ylab = "Final Grade", xlim = c(0,20), ylim = c(0,25))

#plots the relationship between attending a nursery and passing 

# Create a new column to indicate if the final grade is above or below 10
data$grade_category <- ifelse(data$G3 >= 10, "Pass", "Fail")

# Plot the relationship between 'nursery' and 'G3' with color based on grade category
ggplot(data, aes(x = nursery, y = G3, color = grade_category)) +
  geom_point() +
  scale_color_manual(values = c("blue", "red")) +  # Set colors for categories
  labs(x = "Attended Nursery", y = "Final Grade (G3)") +
  theme_minimal()

# Convert 'G3' to a binary outcome
data$G3_binary <- ifelse(data$G3 >= 10, 1, 0)

# Fit logistic regression model
model_nursery <- glm(G3_binary ~ nursery, data = data, family = "binomial")

# View the summary of the model
summary(model_nursery)

# Plot the relationship between 'internet' and 'G3'
ggplot(data, aes(x = internet, y = G3, color = grade_category)) +
  geom_point() +
  scale_color_manual(values = c("blue", "red")) +  # Set colors for categories
  labs(x = "Access to Internet", y = "Final Grade (G3)") +
  theme_minimal()

model_internet <- glm(G3_binary ~ internet, data = data, family = "binomial")

summary(model_internet)

# Plot the relationship between 'higher' and 'G3'
ggplot(data, aes(x = higher, y = G3, color = grade_category)) +
  geom_point() +
  scale_color_manual(values = c("blue", "red")) +  # Set colors for categories
  labs(x = "Seeking Higher Education", y = "Final Grade (G3)") +
  theme_minimal()

model_higher <- glm(G3_binary ~ higher, data = data, family = "binomial")

summary(model_higher)

###########################################################################
# Multiple Logistic Regression
###########################################################################

# Perform multiple logistic regression
model <- glm(G3_binary ~ ., data = data, family = binomial)

# View the summary of the model
summary(model)

###########################################################################
# Lasso
###########################################################################

library(glmnet)

# Fit Lasso regression model
lasso_model <- cv.glmnet(as.matrix(data[, -which(names(data) %in% c("G3_binary", "G3"))]), data$G3_binary, family = "binomial", alpha = 1)

# Print the optimal lambda value selected by cross-validation
print(lasso_model$lambda.min)

# Plot the cross-validated deviance curve
plot(lasso_model)

# Get the coefficients of the Lasso model
coef(lasso_model, s = "lambda.min")

###########################################################################
# Ridge
###########################################################################

# Fit Ridge regression model
ridge_model <- cv.glmnet(as.matrix(data[, -which(names(data) %in% c("G3_binary", "G3"))]), data$G3_binary, family = "binomial", alpha = 0)

# Print the optimal lambda value selected by cross-validation
print(ridge_model$lambda.min)

# Plot the cross-validated deviance curve
plot(ridge_model)

# Get the coefficients of the Ridge model
coef(ridge_model, s = "lambda.min")

############################################################################ 
# QDA
############################################################################
student_mat=read.csv(student_performance_math_file_path,sep=";",header=TRUE)

student_mat$G3[student_mat$G3 %in% c(4, 5)] <- 4
student_mat$G3[student_mat$G3 %in% c(19, 20)] <- 19
table(student_mat$G3)
set.seed(123)
train_indices <- sample(1:nrow(student_mat), 0.8 * nrow(student_mat))
train_data <- student_mat[train_indices, ]
test_data <- student_mat[-train_indices, ]
qda_fit <- qda(G3 ~ G1 + G2, data = train_data)
qda_pred <- predict(qda_fit, newdata = test_data)
conf_matrix <- table(qda_pred$class, test_data$G3)
plot(qda_pred$class, test_data$G3, xlab = "Predicted Class", ylab = "Actual Class", main = "Predicted vs. Actual Classes")

############################################################################ 
# KNN
############################################################################
library(class)
attach(student_mat)
train.X <- cbind(G1, G2)[train_indices, ]
test.X <- cbind(G1, G2)[-train_indices, ]
train.G3 <- G3[train_indices]
set.seed(1)
k <- 1
knn.pred <- knn(train.X, test.X, train.G3, k = k)
table(knn.pred, G3[-train_indices])
mean(knn.pred == G3[-train_indices])
k <- 3
knn.pred <- knn(train.X, test.X, train.G3, k = k)
table(knn.pred, G3[-train_indices])
mean(knn.pred == G3[-train_indices])
k <- 5
knn.pred <- knn(train.X, test.X, train.G3, k = k)
table(knn.pred, G3[-train_indices])
mean(knn.pred == G3[-train_indices])
detach(student_mat)

############################################################################ 
# Forward and Backward Step Selection
############################################################################
names(student_mat)
library(leaps)
regfit.fwd <- regsubsets(absences ~ ., data = student_mat, nvmax = 19, method = "forward")
summary(regfit.fwd)
regfit.bwd <- regsubsets(absences ~ ., data = student_mat, nvmax = 19, method = "backward")
summary(regfit.bwd)

