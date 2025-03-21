# --------------------------- LIBRARIES --------------------------------------
library(tidyverse)
library(caret)
library(glmnet)
library(ranger)
library(class)
library(pROC)
library(ggplot2)
library(ROSE) 

# --------------------------- DATA PREPROCESSING -----------------------------------

# Load dataset
data <- diabetes_dataset
# Drop the 'gender' column
data <- data[, !names(data) %in% c("gender")]

sink("model_output.txt")
# Convert categorical variables to factors
data <- data %>%
  mutate(
    smoking_history = factor(smoking_history, labels = c("never", "No Info", "current", "former", "ever", "no current")),
    hypertension = as.factor(hypertension),
    heart_disease = as.factor(heart_disease),
    diabetes = as.factor(diabetes)
  )

# Handle missing values 
data <- na.omit(data)

# Split data into training (80%) and testing (20%)
set.seed(123)
trainIndex <- createDataPartition(data$diabetes, p = 0.8, list = FALSE)
train_data <- data[trainIndex, ]
test_data <- data[-trainIndex, ]

# --------------------------- CLASS DISTRIBUTION  --------------------

# Plot class distribution to check for class imbalance
ggplot(train_data, aes(x = diabetes, fill = diabetes)) +
  geom_bar() +
  labs(title = "Class Distribution Before ROSE", x = "Diabetes Status", y = "Count") +
  theme_minimal()

# High class imbalance observed

# Get the count of minority class to balance the data
minority_count <- sum(train_data$diabetes == "1")
#minority_count

# --------------------------- APPLY ROSE TO BALANCE THE DATA --------------------

# Apply ROSE to balance the training data
train_data_rose <- ovun.sample(diabetes ~ ., data = train_data, method = "both", p = 0.5, N = minority_count*2)$data

# --------------------------- CLASS DISTRIBUTION AFTER ROSE --------------------

# Plot class distribution after applying ROSE
ggplot(train_data_rose, aes(x = diabetes, fill = diabetes)) +
  geom_bar() +
  labs(title = "Class Distribution After ROSE", x = "Diabetes Status", y = "Count") +
  theme_minimal()

# Export the ROSE training data to a CSV file
write.csv(train_data_rose, "train_data_rose.csv", row.names = FALSE)

# Export test data to a CSV file
write.csv(test_data, "test_data.csv", row.names = FALSE)

cat(" --------------------------- LASSO REGRESSION --------------------------------------\n")

set.seed(123)
tune_grid_lasso <- expand.grid(alpha = 1, lambda = seq(0.0001, 1, length = 10))
lasso_model <- train(
  diabetes ~ ., data = train_data_rose, method = "glmnet",
  family = "binomial", tuneGrid = tune_grid_lasso,
  preProcess = c("zv", "center", "scale"),
  trControl = trainControl(method = "cv", number = 10)
)

# Best model coefficients
coef(lasso_model$finalModel, lasso_model$bestTune$lambda)

# Predictions and evaluation
lasso_pred <- predict(lasso_model, test_data)
lasso_cm <- confusionMatrix(lasso_pred, test_data$diabetes)

print(lasso_cm)

cat("\n --------------------------- RANDOM FOREST ----------------------------------------\n")

set.seed(42)
# Ensure valid factor levels for classification
train_data_rose$diabetes <- factor(make.names(train_data_rose$diabetes))
test_data$diabetes <- factor(make.names(test_data$diabetes))

tune_grid_rf <- expand.grid(
  mtry = c(2, 4, 6, 8), 
  splitrule = "gini",  
  min.node.size = c(1, 5, 10)
)

# Train Random Forest model
rf_model <- train(
  diabetes ~ ., data = train_data_rose, method = "ranger",
  trControl = trainControl(method = "cv", number = 5, classProbs = TRUE),
  tuneGrid = tune_grid_rf, 
  num.trees = 200, importance = "impurity"
)

# Feature importance
rf_importance <- varImp(rf_model)
plot(rf_importance)

# Predictions and evaluation
rf_pred <- predict(rf_model, test_data)
rf_cm <- confusionMatrix(rf_pred, test_data$diabetes)
print(rf_cm)

cat("\n --------------------------- K-NEAREST NEIGHBORS ----------------------------------\n")

set.seed(123)
tune_grid_knn <- expand.grid(k = seq(3, 15, by = 2))
knn_model <- train(
  diabetes ~ ., data = train_data_rose, method = "knn",
  preProcess = c("center", "scale"),
  trControl = trainControl(method = "cv", number = 10),
  tuneGrid = tune_grid_knn
)

# Predictions and evaluation
knn_pred <- predict(knn_model, test_data)
knn_cm <- confusionMatrix(knn_pred, test_data$diabetes)
print(knn_cm)

cat("\n --------------------------- LOGISTIC REGRESSION ----------------------------------\n")

set.seed(123)
log_model <- train(
  diabetes ~ age + hypertension + heart_disease + smoking_history + bmi + HbA1c_level + blood_glucose_level,
  data = train_data_rose, method = "glm", family = "binomial",
  trControl = trainControl(method = "cv", number = 10)
)

# Predictions and evaluation
log_pred <- predict(log_model, test_data)
log_cm <- confusionMatrix(log_pred, test_data$diabetes)
print(log_cm)

cat("\n --------------------------- MODEL PERFORMANCE COMPARISON -------------------------\n")

results <- data.frame(
  Model = c("LASSO", "Random Forest", "KNN", "Logistic Regression"),
  Accuracy = c(lasso_cm$overall["Accuracy"], rf_cm$overall["Accuracy"], knn_cm$overall["Accuracy"], log_cm$overall["Accuracy"]),
  Precision = c(lasso_cm$byClass["Precision"], rf_cm$byClass["Precision"], knn_cm$byClass["Precision"], log_cm$byClass["Precision"]),
  Recall = c(lasso_cm$byClass["Recall"], rf_cm$byClass["Recall"], knn_cm$byClass["Recall"], log_cm$byClass["Recall"]),
  F1_Score = c(lasso_cm$byClass["F1"], rf_cm$byClass["F1"], knn_cm$byClass["F1"], log_cm$byClass["F1"])
)

print(results)

# Plot ROC curves
plot_roc <- function(model, test_data, model_name) {
  pred_probs <- predict(model, test_data, type = "prob")
  roc_curve <- roc(test_data$diabetes, pred_probs[, 2])
  ggroc(roc_curve) +
    ggtitle(paste("ROC Curve -", model_name)) +
    theme_minimal()
}

plot_roc(lasso_model, test_data, "LASSO Regression")
plot_roc(rf_model, test_data, "Random Forest")
plot_roc(knn_model, test_data, "KNN")
plot_roc(log_model, test_data, "Logistic Regression")

sink()
