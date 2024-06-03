## Final Project Submission

Please fill out:
* Student name: Deon Durrant
* Student pace: self paced / part time / full time
* Scheduled project review date/time: 
* Instructor name: Mark  Bardour
* Blog post URL:https://medium.com/@durrantdeon/basics-of-ensemble-methods-beb0b2aa250d


# Business and Data Understanding

## Problem Statement

SyriaTel, a major telecommunications provider, is interested in minimizing resources expended on customers who are likely to terminate their services, a phenomenon known as churn.  Customers' churn  intentions maybe predictive by identifying and isolating patterns hidden in the data. 

## Data Understanding 

To tackle the business problem  I will diligently search for a classifier to  unveil predictive patterns indicating customer inclination to discontinue doing business with SyriaTel. This  constitutes a binary classification endeavor.
 
**Classifiers**

Various classifiers and evaluation metrics will be used to objectively select the most suitable model for addressing the business problem. The following  classifiers will be utilized:  

*  Logistic Regression, 
* Decision Tree
* KNeighbors Classifier
* Random Forest  Classifier

 **Evaluation Metrics**
 
 To assess model performance objectively, the following evaluation metrics will be considered:
* Precision
* Recall
* Accuracy
* F1 score
* AUC Score
* Confusion Matrices


# Exploratory Data Analysis 
- Examine and sanitize the data
- Handle missing data appropriately
- Investigate data attributes
- Rectify typos, inconsistencies in capitalization, and naming conventions
- Analyze the distribution of variables 


# Summary of data structure 
- 3333 entries, 21 columns 
- dtypes: bool(1), float64(8), int64(8), object(4)
- no missing data
- Columns
    - State                   
    - Account length            
    - Area code                 
    - phone number              
    - international plan        
    - voice mail plan           
    - number vmail messages     
    - total day minutes         
    - total day calls           
    - total day charge          
    - total eve minutes        
    - total eve calls           
    - total eve charge          
    - total night minutes       
    - total night calls         
    - total night charge        
    - total intl minutes        
    - total intl calls          
    - total intl charge         
    - customer service calls    
    - Churn* 
    
  **Churn = the number of subscribers that leave the company** 

**To ensure a robust response to the business query and deliver practical recommendations:**

* Prioritize analysis of customer behavior and their tendency to continue as subscribers of Syriatel.

* Delve into  the data features for predictive patterns that can inform  resource allocation decisions. 

* Investigate telephone usage patterns and customer churn correlations.  

* Assess market segmentation to align with the company's goals of enhancing subscriber retention


# Churn Distribution Analysis
churn_count=df['churn'].value_counts()
churn_count
```


    0    2850
    1     483
    Name: churn, dtype: int64





    
![png](output_16_0.png)
    


**Churn Distribution Analysis**
* 483 instances labeled as "1"  (churned) is significantly lower than those labeled as "0" (2850) (not churned), this indicating a class imbalance in the churn variable
* SyriaTel has a churn rate of 14.49% 

Business Implication 

* negatively impact Monthly Recurring Revenue (MRR)
* indicative of  dissatisfaction with the  service
* failure to renew subscription (subscription model)

Modeling Implication 

* This imbalance can affect the performance of machine learning models
* The business problems require classifiers, these are sensitive to class distribution.

To address the imbalance issues resampling such as SMOTE will be considered


# Area Code Distribution Analysis

    
![png](output_18_0.png)
    


**Area Code Distribution Analysis**
- 49.7% almost half of the customers are from 415 area code
- 25% of the customers are in area code 510 and another 25% in area code 408


# Boxplot to see which area code has the highest churn


    
![png](output_20_1.png)
    
**Customer Service Calls Churn Analysis**
* Customers who did not churn tend to have fewer customer service calls (median 1 to 2 calls)  across  area codes 408, 415, and 510.
* Customers who churned show a higher number of customer service calls (median 2 to 3 calls) in all area codes. 
* Customers who make more service calls are more likely to churn
* Variability is evident in not churn and churn customers(length of whiskers)
* More outliers in the churned group especially in the 408 area code

Business implication 
* Mitigate churn utilizing  proactive intervention strategies
* Opportunity to improve customer support 


# International Subscribers  Churn  Distribution




    international plan  churn
    no                  0        2664
                        1         346
    yes                 0         186
                        1         137
    dtype: int64




    
![png](output_23_1.png)
    


**International Subscribers  Churn  Distribution**

* Out of the 3333 subscribers only 346 has an international plan about 10.38%

* 39.56% of international plans subscribers terminate the service 



# Customer Loyalty Churn Analysis




    
![png](output_26_0.png)
    


**Customer Loyalty Churn Analysis**
* Customer Loyalty= Account length,  how long customers has been with SyriaTel
* high risk groups identified for customer duration between 51- 150
* account lengths decline significantly after 150 days




# Voice Mail Subscribers  Churn  Distribution





    voice mail plan  churn
    no               0        2008
    yes              0         842
    no               1         403
    yes              1          80
    dtype: int64



# Visualize voice mail plan and churn status 





    
![png](output_29_1.png)
    


**Voice Mail Subscribers  Churn  Distribution**

- For customers without a voice mail plan: Churn percentage = 16.73%

- For customers with a voice mail plan: Churn percentage = 8.67%

**Usage Churn  Analysis**







    total day charge    101864.17
    churn                  483.00
    dtype: float64




    Churn counts:
    0    2850
    1     483
    Name: churn, dtype: int64
    Total Day Charge by churn status:
    churn
    0    84874.20
    1    16989.97
    Name: total day charge, dtype: float64
    Total Evening Charge by churn status:
    churn
    0    48218.89
    1     8720.55
    Name: total eve charge, dtype: float64
    Total Night Charge by churn status:
    churn
    0    25667.31
    1     4460.76
    Name: total night charge, dtype: float64
    



# Compare churn and telephone charges across different times of the day.



    
![png](output_34_1.png)
    


**Day Usage Churn**

- Bimodal distribution for both churned and non-churned customers
- Churned customers have two distinctive peaks indicating different customer behavior within the subset
- Overlapping of churn and non-churn indicating the subsets may share similar caharecteristics
- Day rate = 0.17

**Evening Usage Churn**

- Non-churned has a peak that is slightly left compared to the peak of the distribution for churned customers .
- Non-churned has a tail that extends further to the right, suggesting higher total evening charge for some non-churned customers.
- Churned distribution tapers off more quickly, indicating that higher total evening charges are less common for churned customers.
- Evening  rate = 0.085

**Night Usage Churn**

- High degree of overlapping suggesting feature is not distinguishing churn and not-churn
- This variable may not be a strong predictor of churn
-  Day rate = 0.045



# Features correlation analysis
# Visualize the correlation between variables



    
![png](output_36_0.png)
    


**Features Correlation Analysis**
* The correlation values for **churn** seems to fall  below 0.3 for all features. 
* Multiple variables are uncorrelated 

The following are perfectly correlated due to the direct relationship between charge and usage minutes
* Total intl charge and total intl minutes
* Total night charge and total night minutes
* Total eve charge and total eve minutes
* Total day charge and total day minutes




    

# Preprocessing

Prepare data for modeling

## Create training and testing sets

- Define the predictor and target variables
- Perform a standard train-test split. 
- Assign 30% to the test set 
- Set random_state ensuring  that the split is reproducible


**Evaluate class imbalance**

    Train:
    
    0    1993
    1     340
    Name: churn, dtype: int64
    Test:
    
    0    857
    1    143
    Name: churn, dtype: int64
    

**Class Imbalance evaluation**
- In the training set the proportion  of churned customers to not churned  is 340 to 1993
- In the testing  set the proportion  of churned customers to not churned  is 143 to 857
- To address class imbalance Synthetic Minority Oversampling is used 

# Address class imbalance using SMOTE 

    0    1993
    1     340
    Name: churn, dtype: int64
    
    
    1    1993
    0    1993
    Name: churn, dtype: int64
    

**Scaling**
 * Transform the numerical features of the dataset to a similar scale 

## 2. Modeling

##   Model Development 1: Logistic Regression

Build and evaluate a baseline Logistic Regression



# Evaluation of the models
The following metrics will be used to evaluate the classifiers for both the training and test sets.
 - Precision: measures how accurate the positive predictions are.
 - Recall:  measures the model's ability to find all the relevant cases (positive cases)
 - Accuracy: 
     - measures the overall correctness of the model
     - most common metric for classification,  providing a solid holistic view of the overall performance of our model.
 - F1 score:  
     - Harmonic Mean of Precision and Recall, providing a single measure of efficacy
    - This means that the F1 score cannot be high without both precision and recall also being high. 
    - High model's F1 score indicates the  model is doing well all around
    
- AUC Score: reflects the model's ability to distinguish between classes.
 
**AUC ROC curve:**

 -  Plots the true positive rate (sensitivity) on the y-axis against the false positive rate (1 - specificity) on the x-axis.
 -  ROC curve evaluates how well the model discriminates between classes
 -  Instrumental in comparing the effectiveness of different models

   **Evaluate the Logistic Regression predictive performance**
  
   



# Snapshot of comprehensive metrics


    Classification Report (Training Set):
                   precision    recall  f1-score   support
    
               0       0.87      0.98      0.92      1993
               1       0.54      0.13      0.21       340
    
        accuracy                           0.86      2333
       macro avg       0.71      0.56      0.57      2333
    weighted avg       0.82      0.86      0.82      2333
    
    Classification Report (Testing Set):
                   precision    recall  f1-score   support
    
               0       0.88      0.98      0.93       857
               1       0.59      0.17      0.26       143
    
        accuracy                           0.86      1000
       macro avg       0.73      0.57      0.59      1000
    weighted avg       0.83      0.86      0.83      1000
    
    Confusion Matrix:
     [[840  17]
     [119  24]]
    

     

# Model Development 2: DecisionTreeClassifier



**Evaluate the DecisionTreeClassifier predictive performance**

    
# Snapshot of comprehensive metrics


    Classification Report (Training Set):
                   precision    recall  f1-score   support
    
               0       1.00      0.91      0.95      2183
               1       0.43      0.98      0.60       150
    
        accuracy                           0.92      2333
       macro avg       0.72      0.95      0.78      2333
    weighted avg       0.96      0.92      0.93      2333
    
    Classification Report (Testing Set):
                   precision    recall  f1-score   support
    
               0       0.91      0.98      0.94       857
               1       0.75      0.39      0.51       143
    
        accuracy                           0.89      1000
       macro avg       0.83      0.68      0.73      1000
    weighted avg       0.88      0.89      0.88      1000
    
    Confusion Matrix:
     [[838  19]
     [ 87  56]]
    

Model Comparison
Compare to the logistic regression model the decision tree has higher accuracy, lower AUC and lower false negatives . 

# Model Development 3: KNN


**Evaluate the KNN predictive performance**


# Snapshot of comprehensive metrics


    Classification Report (Training Set):
                   precision    recall  f1-score   support
    
               0       0.91      0.99      0.95      1993
               1       0.85      0.42      0.56       340
    
        accuracy                           0.90      2333
       macro avg       0.88      0.70      0.75      2333
    weighted avg       0.90      0.90      0.89      2333
    
    Classification Report (Testing Set):
                   precision    recall  f1-score   support
    
               0       0.90      0.98      0.94       857
               1       0.73      0.36      0.48       143
    
        accuracy                           0.89      1000
       macro avg       0.81      0.67      0.71      1000
    weighted avg       0.88      0.89      0.87      1000
    
    Confusion Matrix:
     [[838  19]
     [ 92  51]]
    

# Model Development 4: Random Forest 
Random forest ensemble method


# **Evaluate the Random Forest predictive performance**



# Snapshot of comprehensive metrics


    Classification Report (Training Set):
                   precision    recall  f1-score   support
    
               0       0.90      1.00      0.95      1993
               1       1.00      0.38      0.55       340
    
        accuracy                           0.91      2333
       macro avg       0.95      0.69      0.75      2333
    weighted avg       0.92      0.91      0.89      2333
    
    Classification Report (Testing Set):
                   precision    recall  f1-score   support
    
               0       0.90      0.99      0.94       857
               1       0.88      0.31      0.46       143
    
        accuracy                           0.90      1000
       macro avg       0.89      0.65      0.70      1000
    weighted avg       0.89      0.90      0.87      1000
    
    Confusion Matrix:
     [[851   6]
     [ 98  45]]
    
    

# Analysis of  models performances and final model selection
 

**Model Comparison**

This section provides a comparison of four different classification models utilizing different evaluation metrics.


                   
| Model                   | Precision | Recall   | F1 Score | Accuracy | AUC Score    |
|------------------- -----|-----------|----------|----------|----------|--------------|
| **Logistic Regression** | 0.56     | 0.17      | 0.26    | 0.86    | **0.7453**   |
| **KNN**                 | 0.70     | **0.41**      | **0.51**    | 0.89    | 0.6882   |    
| **Decision Tree**       | 0.75     | 0.39      | **0.51**    | 0.858    |0.6847    |
| **Random Forest**       | **0.88**     | 0.31      | 0.46    | **0.90**    |0.6538   |
  


* ***Precision***: Random Forest has the highest precision (0.88), indicating that it has the lowest rate of false positives among the models.
* ***Recall***: KNN has the highest recall (0.41), indicating that it has the highest rate of true positives among the models.
* ***F1 Score***: KNN and Decision Tree have the highest F1 score (0.51), which balances precision and recall, indicating their overall performance.
* ***Accuracy***: Random Forest and KNN have the highest accuracy (0.90 and 0.89 respectively), indicating the overall correctness of the predictions.
* ***AUC Score***: Logistic Regression has the highest AUC score (0.7453), indicating its ability to distinguish between positive and negative classes effectively based on the ROC curve.


 
 ##  Confusion Matrices Model Performance 

| Model           | True Negatives (TN) | False Positives (FP) | False Negatives (FN) | True Positives (TP) |
|-----------------|---------------------|----------------------|----------------------|---------------------|
| Logistic Regression | 838             | 19                   | 119                  | 24                  |
| KNN             | 832                 | 25                   | **85**               | **58**                  |
| Decision Tree   | 838                 | 19                   | 87                   | 56                  |
| Random Forest   | **851**             | **6**                 | 98                | 45                  |





From the chart, it is evident that:
- KNN and Random Forest appears to be the better performers  
- KNN achieving the highest TP accurately predict "Churn'  and the lowest FN where the model incorrectly predicts 'No Churn', but in reality, the customer does churn.  
- Random Forest achieves the highest TN predicts 'No Churn' and the lowest FP, predicts 'Churn' but customer does not churn

These comparisons assists in determining the appropriate model based on the specific requirements of sensitivity(true positive rate)  and specificity(true negative rate)  needed for this business problem.

# Create visualization comparing models confusion matrices




    
![png](output_79_0.png)
    


# Model Selection and Tuning 

**Random Forest** was chosen for  several reasons based on an analysis of model performances:

- It achieved  a high recall, the highest precision, F1, and accuracy scores among the models evaluated.
- It had the second-highest AUC score out of the four models considered.
- Although Random Forest did not excel the most in identifying positive predictions, as shown in the confusion matrix, it ranked second in this area and had the second-highest number of true negatives 
* The ability to handle highly-correlated variables
*  Easy to evaluate variable importance or contribution to model
* Reduce risk of overfitting compare to other models

**Hyperparameter Tuning** 

- GridSearchCV: to obtain the best parameters for the model 



# Baseline random forest parameters
rf.get_params()
```




    {'bootstrap': True,
     'ccp_alpha': 0.0,
     'class_weight': None,
     'criterion': 'gini',
     'max_depth': 5,
     'max_features': 'auto',
     'max_leaf_nodes': None,
     'max_samples': None,
     'min_impurity_decrease': 0.0,
     'min_impurity_split': None,
     'min_samples_leaf': 1,
     'min_samples_split': 2,
     'min_weight_fraction_leaf': 0.0,
     'n_estimators': 100,
     'n_jobs': None,
     'oob_score': False,
     'random_state': 1,
     'verbose': 0,
     'warm_start': False}




```python
rf_param_grid = {
    "n_estimators": [10, 30, 100],
    "criterion": ["gini", "entropy"],
    "max_depth": [None, 2, 6, 10],
    "min_samples_split": [5, 10],
    "min_samples_leaf": [3, 6],
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap': [True, False],
    'random_state':[42]
}

`

    Testing Accuracy: 90.40%
    
    Optimal Parameters: {'bootstrap': False, 'criterion': 'gini', 'max_depth': None, 'max_features': 'auto', 'min_samples_leaf': 3, 'min_samples_split': 5, 'n_estimators': 100, 'random_state': 42}
    

# Update the model with optimal parameters 
model_rf = RandomForestClassifier(criterion= 'gini', max_depth= None, min_samples_leaf= 3, 
                                  min_samples_split= 5, n_estimators= 100, bootstrap=False, max_features= 'auto', random_state= 42)

# Update Random Forest snapshot of comprehensive metrics



# Calculate evaluation metrics with confusion matrices 


    Classification Report (Training Set):
                   precision    recall  f1-score   support
    
               0       0.97      1.00      0.98      1993
               1       1.00      0.81      0.90       340
    
        accuracy                           0.97      2333
       macro avg       0.98      0.91      0.94      2333
    weighted avg       0.97      0.97      0.97      2333
    
    Classification Report (Testing Set):
                   precision    recall  f1-score   support
    
               0       0.92      0.98      0.95       857
               1       0.80      0.49      0.61       143
    
        accuracy                           0.91      1000
       macro avg       0.86      0.73      0.78      1000
    weighted avg       0.90      0.91      0.90      1000
    
    Confusion Matrix:
     [[839  18]
     [ 73  70]]
    


# Plot the Confusion Matrix



    
![png](output_88_0.png)
    

# Evalute using the AUC ROC curve



# Plot ROC curve

    


    
![png](output_90_1.png)
    


# Model Improvement Analysis 


## Model Performance Comparison

| Metric    | rf   | model_rf | Difference (%)  |
|-----------|------|----------|-----------------|
| Precision | 0.88 | 0.80     | -9.09%          |
| Recall    | 0.31 | 0.49     | +58.06%         |
| F1-score  | 0.46 | 0.61     | +32.61%         |
| Accuracy  | 90%  | 91%      | +1.11%          |
|AUC Score  |0.65  |  0.73    | +12.31%           |



**Precision**
* The rf model has a higher precision (0.88) compared to model_rf (0.80)indicting it's better at ensuring that when it predicts a positive result or churn, it is likely correct. 
*  model_rf shows a decrease in precision by 9.09%, which suggest it has a higher tendency to classify not-churn(negatives) cases as churn(positives), compared to rf.

**Recall**
* The recall of model_rf is 0.49, 58.06% higher than that of rf at 0.31. 
* model_rf is better at identifying actual positive cases overall. 
* It does not miss as many positives/churn as rf, making it a preferable model for the SyriaTel business problem 

**F1-Score**
* F1-score combines precision and recall into a single metric by taking their harmonic mean. 
* model_rf has a higher F1-score (0.61) than rf (0.46),  32.61% improvement reflecting a better balance between precision and recall. 
* improvement suggests that model_rf is more robust in balancing  between missing churns and maintaining reasonable accuracy in those predictions.

**Accuracy**
* model_rf shows a slight improvement (91%) over rf (90%), with a marginal increase of 1.11%.
* This suggests that model_rf makes correct predictions on a slightly higher percentage of the total dataset.

**AUC Score**
* AUC score improved by approximately 12.31%. 
* significant model  improvement, suggesting model_rf effectively enhances predictive performance.

## Confusion Matrix Comparison

| Model        | True Negatives (TN) | False Positives (FP) | False Negatives (FN) | True Positives (TP) |
|--------------|---------------------|----------------------|----------------------|---------------------|
| **rf**       | 851                 | 6                    | 98                   | 45                  |
| **model_rf** | 839                 | 18                   | 73                   | 70                  |


* TN are slightly lower at 839, indicating a slight drop in correctly identifying negative cases.
* FP increase to 18, suggesting this model is more aggressive in predicting positives but  with associated errors 
* FN are reduced to 73, indicating better performance in catching positive cases compared to rf.
* Significantly higher TP at 70, suggesting better effectiveness at identifying positive cases.


**Feature importance**
*  Examine the importance of each feature in the  model. 
* Check the feature_importances_ attribute of the trained model 





feature_importances


    area code_408             0.004551
    area code_510             0.004961
    area code_415             0.008124
    voice mail plan_no        0.026859
    voice mail plan_yes       0.028602
    total night charge        0.061461
    international plan_no     0.063774
    total night minutes       0.064528
    international plan_yes    0.065110
    total intl charge         0.067571
    total intl minutes        0.074507
    total eve minutes         0.097512
    total eve charge          0.100384
    total day charge          0.164814
    total day minutes         0.167243
    dtype: float64





# Visualize ranked feature importances 





    
![png](output_94_1.png)
    



# Analysis of the feature importance revealed the following:

**Daytime  usage**
- Total Day Minutes( 0.167243) highest feature importance in the model, suggesting that the number of minutes customers use during the day is highly predictive of customer propensity to churn. 
- Total Day Charge(0.164814) is ranked second .
- The amount of usage and the associated charges during the day are significant predictors of customer behavior.
-  Customers who use the service heavily during the day are crucial for the model's predictions
- Day rate = 0.17

**Evening time usage** 
- Total Evening Charge(0.100384)  is ranked third 
- Associated Total Evening Minutes (0.097512 ) follows.
- Evening time usage is highly important indicating  another critical factor in predicting customer behavior. 
- These features are slightly less important than daytime usage but still play a major predictor of customer churn.
- Evening rate= 0.085


**International usage and plan** 
- Total International Minutes (0.074507) and Total International Charge (0.067571) are also important, further emphasizing the relevance of usage in predicting customer behavior.
- International Plan Yes (0.065110) and International Plan No (0.063774) ranking indicates that subscribers opting for or against this add-on service has significant impact on churn. 
-  10.38% of subscribers utilize international plan 
- 39.56% of international plans subscribers terminate the service indicating distinctive pattern of churn behavior compared to those without the plan.


**Night time usage** 
- Total Night Minutes (0.064528) and Total Night Charge (0.061461)  as indicators of night time usage are less important than day time usage, evening and international usage.
- Night rate = 0.045

**Voice mail Plan** 
- Voice Mail Plan Yes (0.028602) and Voice Mail Plan No (0.026859) importance is ranked low but subscribers participation in this plan or not  still influences customer behavior 

**Geographical location**
- The least important features of the model are the area codes:Area Code 415 (0.008124), Area Code 510 (0.004961), and Area Code 408 (0.004551) 
- Geographical location has minimal impact on customer propensity to churn 

# Predictive Recommendations

Based on the above analysis the following predictive recommendation are proposed:

1. Concentrated Usage Analysis

    - Accurate and meticulous data collection for daytime usage, the highest ranking importance
    - Employ detailed analysis of usage patterns, including  rate structure for different times of day. Currently evening rate is  approximately 50% cheaper than day rate while night rate is 74% cheaper
    - Recognize customers are inclined to use the service more during the daytime making daytime usage the best predictor of subscribers intentions to terminate service.
    - Allocate resources to track and analyze daytime usage patterns to optimize identification of high-risk customers and employ mitigation strategies to ensure retention.


2. Customer Segmentation

    - Plan types impacts  prediction, voicemail plans to a lesser extent. Understanding how different plans influence customer behavior can help tailor retention strategies.
    - Utilize insights on plan types and usage patterns to segment customers effectively and target high-risk groups with tailored interventions
    

3. Invest in Subscription Analytics

    - Acquire dashboards to monitor high-importance features in real-time, allowing for proactive measures to prevent churn
    

4. Survey customers at the point of cancel

    - Utilize churn survey to gather data to understand why customers cancel their accounts.
    - Survey can identify issues and improve SyriaTel product



