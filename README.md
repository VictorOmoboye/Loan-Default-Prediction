# LOAN DEFAULT PREDICTION 
## Leveraging Advance Machine Learning to Predict Loan Default Risk for Smarter Lending Decisions
![image](https://github.com/user-attachments/assets/b923a5bf-5a92-483d-98e3-596c8b4b4b98)
## INTRODUCTION
**LoanAnalytics Inc.**, a leading financial services provider, seeks to enhance its loan evaluation process by implementing an innovative loan default prediction model. With rising economic uncertainties and increasing default rates, traditional methods relying on static borrower data are no longer sufficient. This project aims to leverage machine learning to predict default risks more accurately, enabling faster loan approvals, improved risk management, and minimized financial losses. By integrating dynamic borrower data with advanced algorithms, LoanAnalytics Inc. can make smarter, data-driven lending decisions and maintain a competitive edge in the industry.

![image](https://github.com/user-attachments/assets/e301c6c2-951b-46b2-a432-d044ccb23193)

## PROBLEM STATEMENT
**LoanAnalytics Inc.** seeks to implement an automated loan default prediction model that can identify high-risk borrowers early in the loan approval process. The model should
help the company:
- **Predict Borrower Default Risk:** Develop a machine learning algorithm to accurately predict the likelihood of borrowers defaulting on loans, leveraging historical and dynamic borrower data.
  
- **Improve Loan Approval Efficiency:** Streamline the loan approval process by incorporating predictive insights, reducing reliance on manual evaluations, and expediting decision-making.
  
- **Minimize Financial Losses:** Identify high-risk borrowers early to enable preventive measures, such as adjusted interest rates or collateral requirements, reducing default-related financial risks.
  
## AIM OF THE PROJECT
- **Enhance Default Risk Prediction:** Build a reliable machine learning model to accurately predict loan defaults using historical and dynamic borrower data.  

- **Streamline Loan Processing:** Automate and optimize the loan approval workflow to improve efficiency and reduce manual intervention.  

- **Mitigate Financial Risks:** Proactively identify high-risk borrowers to minimize potential losses and maintain profitability.  

- **Enable Data-Driven Decisions:** Leverage predictive insights to refine lending strategies and tailor loan terms for different borrower segments.  

- **Foster Competitive Advantage:** Utilize advanced analytics to stay ahead in the lending industry by improving risk management and customer satisfaction.  

## METHODOLOGY
- **STEP 1: Data Cleaning:**  
  - Handle missing values using appropriate imputation techniques.  
  - Remove duplicate records and irrelevant columns that do not contribute to prediction.  
  - Identify and correct anomalies in the dataset to ensure data quality.  

- **STEP 2: Exploratory Data Analysis (EDA):**  
  - Visualize feature distributions, relationships, and correlations using plots like histograms and heatmaps.  
  - Identify patterns, trends, and anomalies that may influence loan defaults.  
  - Formulate hypotheses to guide feature engineering and model selection.  

- **STEP 3: Data Preprocessing:**  
  - Scale or normalize numerical features and encode categorical variables for compatibility with machine learning models.  
  - Split the data into training, validation, and test sets to ensure robust evaluation.  

- **STEP 4: Model Training:**  
  - Select and train machine learning models such as Logistic Regression, Random Forest, or Gradient Boosting.  
  - Conduct hyperparameter tuning and k-fold cross-validation for model improvement.  
  - Experiment with multiple algorithms and compare their performance.  

- **STEP 5: Model Evaluation:**  
  - Assess model performance using metrics like accuracy, precision, recall, F1-score, and ROC-AUC.  
  - Analyze performance across subsets (e.g., borrower income levels) and perform error analysis.  
  - Compare results to a baseline model to measure improvements.  

- **STEP 6: Model Optimization:**  
  - Fine-tune hyperparameters using techniques like Grid Search or Random Search.  
  - Apply regularization or ensemble methods to address overfitting and enhance performance.  
  - Refine feature selection and ensure the model generalizes well to unseen data.

  
## LIBRARIES
- **NumPy:** It provides support for large, multi-dimensional arrays and matrices, along with mathematical functions to operate on these arrays.
- **Pandas:** It offers data structures (like DataFrames) for handling and analyzing structured data, particularly for data manipulation and cleaning.
- **Matplotlib.pyplot:** A plotting library used for creating static, interactive, and animated visualizations in Python.
- **Seaborn:** Built on top of Matplotlib, Seaborn simplifies the creation of informative and attractive statistical graphics.
  
## EXPECTED OUTCOME
- **Accurate Default Predictions:** A machine learning model capable of predicting loan default risk with high precision and reliability.  

- **Streamlined Loan Approvals:** Faster and more efficient loan approval processes through predictive insights and automation.  

- **Reduced Financial Losses:** Early identification of high-risk borrowers to mitigate defaults and improve profitability.  

- **Data-Driven Lending Strategies:** Insights into borrower behavior and risk factors to refine lending policies and tailor loan terms.  

- **Competitive Advantage:** Enhanced decision-making and risk management, positioning LoanAnalytics Inc. as a leader in the lending industry.  

### Exploratory Data Analysis (EDA)
#### Numerical Data
During the EDA process on numerical data, univariate and bivariate analyses revealed that the distributions of **Age**, **Income**, **Loan Amount**, **Credit Score**, **Months Employed**, **Interest Rate**, and **DTI Ratio** are uniform, with no outliers detected. The correlation analysis showed moderate relationships among the numerical features, with **Interest Rate** exhibiting a positive correlation of **0.13** with loan defaults, while **Age** displayed a negative correlation of **-0.17** with defaults.
![image](https://github.com/user-attachments/assets/a6521e03-5228-48bd-bf9a-9bcb982f99f4)

#### Categorical Data
During the EDA process on categorical data, univariate analysis revealed that the distributions of **Education** and **Marital Status** are uniform. Bivariate analysis of **Education** against loan defaults indicated that customers with a **high school education** have the highest likelihood of defaulting, followed by those with a **bachelor's degree**, **master's degree**, and then **PhD holders**, who exhibit the lowest default risk.
![image](https://github.com/user-attachments/assets/3a67bdfa-c1c0-4f82-93a2-888052032af4)

### Data Preprocessing 
During the data preprocessing phase, categorical variables were converted to numerical features using **Label Encoding**. The dataset was then split into **80% training** and **20% testing** using the `train_test_split()` function from the **sklearn.model_selection** library. Additionally, feature scaling was applied to ensure consistency in the dataset for model training.
![image](https://github.com/user-attachments/assets/3f145897-1204-4a66-b11c-523aca004c2f)

### Model Training
During the model training phase, **Logistic Regression** was initially used to train the dataset, but it yielded unsatisfactory results after evaluation. To improve performance, the **Min-Max Scaler** was applied, as the dataset had a uniform distribution. Following this optimization, additional supervised machine learning models, including **DecisionTreeClassifier**, **SGDClassifier**, and **RandomForestClassifier**, were explored to enhance predictive accuracy and performance.

![image](https://github.com/user-attachments/assets/90de9038-cacb-437c-91ed-6568b0399096)

### Model Evaluation
























## Still under contruction. To be fully updated soon.......
