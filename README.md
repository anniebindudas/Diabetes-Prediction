# Diabetes-Prediction

This project involves building machine learning models to predict the likelihood of a person developing diabetes based on various health factors. The dataset contains various health metrics such as age, BMI, blood glucose levels, and smoking history, which are used to train and evaluate different classification models.

### Techniques and Models Used:

Data Preprocessing:

 * Handled missing values by omitting rows with missing data.
 * Categorical variables were converted into factors for proper modeling.
 * Applied ROSE (Random Over-Sampling Examples) to address class imbalance in the dataset.

### Model Training & Evaluation

 * LASSO Regression: Feature selection and classification using L1 regularization
 * Random Forest: Ensemble learning with multiple decision trees
 * KNN: Classification based on nearest neighbors
 * Logistic Regression: Baseline binary classification

### Performance Metrics
 * Accuracy, Precision, Recall, and F1-score computed for all models
 * Confusion matrices generated
 * ROC curves plotted to compare model performance
