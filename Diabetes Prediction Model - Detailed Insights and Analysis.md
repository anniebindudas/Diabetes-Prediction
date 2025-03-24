# **Detailed Insights and Analysis**

### Overview
This project aims to predict the likelihood of a person developing diabetes based on several health-related factors. The dataset consists of health metrics, such as age, BMI (Body Mass Index), blood glucose levels, and smoking history, and is used to train and evaluate multiple machine learning models. Accurate diabetes prediction can help in early diagnosis and improve patient care.

### Data Preprocessing
* Missing Data: There were no missing values in the dataset, so no imputation or deletion was necessary.
* Categorical Variables: Categorical variables like smoking_history, hypertension, and heart_disease were converted into factors to allow proper modeling.
* Class Imbalance: To address the class imbalance in the dataset, where the minority class had only 6,800 instances out of 100,000 rows, I applied the ROSE (Random Over-Sampling Examples) technique. This method generated synthetic samples for the minority class and downsampled the majority class, resulting in a more balanced dataset. This helped improve model performance by ensuring both classes were adequately represented.

### Model Training
Four different machine learning models were trained to predict diabetes:
* LASSO Regression: This model used L1 regularization for feature selection, which is particularly effective when dealing with high-dimensional data.
* Random Forest: An ensemble method using multiple decision trees, known for handling both numerical and categorical data effectively.
* K-Nearest Neighbors (KNN): A simple and interpretable classification algorithm based on the nearest neighbors.
* Logistic Regression: A traditional binary classification model, providing a baseline for comparison with other models.

#### *Feature Importance (LASSO Regression)*

To enhance model interpretability and identify the most relevant predictors of diabetes, I applied LASSO (Least Absolute Shrinkage and Selection Operator) regression. LASSO performs feature selection by shrinking less important coefficients toward zero, reducing complexity while maintaining predictive performance.

<img src="https://github.com/user-attachments/assets/3e1b0636-319d-4ff4-bf35-b5bb1718cdb3" width="500"/>

The feature importance plot highlights the most influential variables in predicting diabetes:
* HbA1c Level (Hemoglobin A1c Level): The strongest predictor, reflecting long-term blood sugar control over 2-3 months. Higher values are strongly correlated with diabetes.
* Blood Glucose Level: Another critical factor in diabetes diagnosis, representing current blood sugar levels.
* Age & BMI: Older age and higher BMI contribute significantly to diabetes risk.
* Other Factors (Hypertension, Smoking History, Heart Disease): These had minimal influence in the LASSO model, suggesting they may not be strong independent predictors in this dataset.

By selecting only the most relevant features, LASSO improved model efficiency, reduced overfitting, and enhanced generalization. This refined feature set was then used for further model training and evaluation.

### Model Evaluation

The models were evaluated based on the following metrics:
* Accuracy: The proportion of correct predictions
* Precision: The proportion of true positives among all predicted positives
* Recall: The proportion of true positives among all actual positives
* F1 Score: The harmonic mean of precision and recall, providing a balanced metric

<img src="https://github.com/user-attachments/assets/31fc1264-d536-43fc-94a2-7b138f7045d5" width="500"/>

Best Model: The Random Forest model achieved the highest accuracy (91.07%) and F1 Score (94.92%), indicating its strong performance for this classification task.




### Key Insights:
* The minority class (class 1) was significantly underrepresented with only 6,800 rows out of 100,000, so applying ROSE to balance the classes was crucial.
* Random Forest emerged as the most effective model, with the best performance in terms of accuracy, precision, recall, and F1 score.
* The LASSO regression model, while less accurate than Random Forest, demonstrated its ability to handle feature selection and performed reasonably well.

### Conclusion 
Healthcare providers can use predictive models like the ones built in this project to make more informed decisions. By analyzing health data such as blood glucose levels, BMI, and smoking history, healthcare professionals can identify individuals who may be at risk for diabetes before the onset of severe symptoms. By identifying individuals at risk for diabetes earlier, public health initiatives can be better targeted, improving outcomes for those affected by diabetes. 

