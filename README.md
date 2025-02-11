# House Price Prediction

This project aims to predict house prices based on various features such as location, size, number of rooms, and other related factors. The dataset used for this project contains information about housing prices, and the goal is to predict the price of a house based on its attributes.

## Libraries Used

The following libraries are used for data preprocessing, analysis, and building machine learning models:

- **Numpy**: For numerical operations.
- **Pandas**: For data manipulation and analysis.
- **Matplotlib**: For data visualization.
- **Seaborn**: For statistical visualization.
- **Scikit-learn**: For machine learning algorithms and model evaluation.

## Models Implemented

The following machine learning models were applied to predict house prices:

- **Linear Regression**: A fundamental algorithm used for regression tasks, which models the relationship between the target variable and one or more predictor variables.
- **Random Forest**: An ensemble learning method that creates multiple decision trees and merges them together to get a more accurate and stable prediction.

## Model Comparison

After evaluating multiple models, the **Random Forest** model was found to be the best for this task.

### Best Model: Random Forest

- **Model Accuracy**: 81.63%

The **Random Forest Model** outperformed the **Linear Regression** model due to its ability to handle non-linear relationships and its robustness to overfitting.

## Dataset Information

The dataset used in this project is **housing.csv**, which contains various features of houses such as:

- **LotArea**: Lot size in square feet.
- **OverallQual**: Overall material and finish quality.
- **OverallCond**: Overall condition rating.
- **YearBuilt**: Year the house was built.
- **GrLivArea**: Above ground living area in square feet.
- **GarageCars**: Number of cars the garage can hold.
- **1stFlrSF**: First floor square feet.
- **TotRmsAbvGrd**: Total rooms above grade (does not include bathrooms).
- **Fireplaces**: Number of fireplaces.
- **Price**: The target variable representing the house price (in dollars).

### Dataset Source:
The dataset is publicly available on [Kaggle](https://www.kaggle.com/).

## Steps Involved in the Analysis

1. **Data Loading**: Load the dataset into a Pandas DataFrame for further analysis.
2. **Data Preprocessing**:
   - Handle missing values and outliers.
   - Encode categorical variables (if any).
   - Feature scaling for models that require it.
   - Split the dataset into training and test sets.
3. **Exploratory Data Analysis (EDA)**:
   - Visualize the distribution of features like lot area, living area, and price.
   - Identify any correlations between features and the target variable (house price).
   - Use **Seaborn** and **Matplotlib** to create plots for better understanding.
4. **Model Training and Evaluation**:
   - Train both **Linear Regression** and **Random Forest** models.
   - Evaluate model performance using metrics like **Mean Squared Error (MSE)** and **R-squared (R²)**.
5. **Model Performance**:
   - The **Random Forest** model achieved the best performance, offering more accurate predictions compared to the **Linear Regression** model.
6. **Model Tuning**:
   - Use techniques like **GridSearchCV** or **RandomizedSearchCV** to fine-tune hyperparameters for the Random Forest model.
7. **Visualization**:
   - Visualize model predictions vs. actual house prices.
   - Plot feature importance using the Random Forest model to understand the key predictors of house prices.

## Installation

To run the project, you’ll need to install the following dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Running the Analysis
To run the analysis, execute the Python script in your preferred Python environment.

```bash
python house_price_prediction.py
```

This will:
1. Load the housing dataset.
2. Perform data preprocessing and exploratory data analysis.
3. Train both Linear Regression and Random Forest models.
4. Evaluate the models and display results.

## Results
- **Linear Regression Accuracy**: 67.08%
- **Random Forest Accuracy**: 81.63%

The **Random Forest model** performed the best, achieving the highest prediction accuracy.

## Conclusion
This project demonstrates the process of predicting house prices using machine learning algorithms. The Random Forest model provided the most accurate results and was selected as the final model. The analysis highlights the importance of feature selection and model tuning in improving prediction accuracy.
