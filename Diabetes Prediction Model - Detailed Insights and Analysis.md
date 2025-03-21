# Detailed Insights and Analysis

### Overview
This project aims to predict the likelihood of a person developing diabetes based on several health-related factors. The dataset consists of health metrics, such as age, BMI (Body Mass Index), blood glucose levels, and smoking history, and is used to train and evaluate multiple machine learning models. Accurate diabetes prediction can help in early diagnosis and improve patient care.

### Data Preprocessing
* Missing Data: There were no missing values in the dataset, so no imputation or deletion was necessary.
* Categorical Variables: Categorical variables like smoking_history, hypertension, and heart_disease were converted into factors to allow proper modeling.
* Class Imbalance: Given that class 1 had only 6,800 rows, I applied ROSE (Random Over-Sampling Examples) to balance the dataset. The minority_count of 6,800 was used to generate an equal number of rows for class 0, resulting in a balanced dataset of 13,600 rows. This ensured that both classes had equal representation in the training data.

### Model Training
Four different machine learning models were trained to predict diabetes:
* LASSO Regression: This model used L1 regularization for feature selection, which is particularly effective when dealing with high-dimensional data.
* Random Forest: An ensemble method using multiple decision trees, known for handling both numerical and categorical data effectively.
* K-Nearest Neighbors (KNN): A simple and interpretable classification algorithm based on the nearest neighbors.
* Logistic Regression: A traditional binary classification model, providing a baseline for comparison with other models.

### Model Evaluation

The models were evaluated based on the following metrics:
* Accuracy: The proportion of correct predictions
* Precision: The proportion of true positives among all predicted positives
* Recall: The proportion of true positives among all actual positives
* F1 Score: The harmonic mean of precision and recall, providing a balanced metric

  ![image](https://github.com/user-attachments/assets/08a83c50-df1b-49ac-bea5-1b17983beebf)

  
![image](https://github.com/user-attachments/assets/d2c035c8-311a-4317-9b0c-cc79f7189836)
