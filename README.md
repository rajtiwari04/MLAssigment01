# MLAssigment01: A Comprehensive Study of Linear Regression Models

This repository contains a comprehensive analysis and implementation of various linear regression models. The goal is to explore, build, and compare Simple Linear Regression, Multiple Linear Regression, Polynomial Regression, and regularized models (Ridge and Lasso) on the PIMA Indians Diabetes Dataset.

## Project Overview

The project is divided across three Jupyter notebooks, each focusing on a different aspect of the regression analysis:

1.  **`Untitled0.ipynb`**: Focuses on Exploratory Data Analysis (EDA) and a foundational implementation of Simple Linear Regression from scratch.
2.  **`Untitled4.ipynb`**: Implements Multiple Linear and Polynomial Regression models using the Scikit-learn library.
3.  **`ridge&lasso.ipynb`**: Applies Ridge (L2) and Lasso (L1) regularization techniques to the models built in the previous notebook to control overfitting and perform feature selection.

## Dataset

The project utilizes the **PIMA Indians Diabetes Dataset** (`diabetes.csv`). It consists of several medical predictor variables and one target variable, `Outcome`.

-   **Features**: `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`
-   **Target**: `Outcome` (0 or 1, indicating the presence of diabetes)

While the target variable is binary, this project uses regression models to predict a continuous value as a demonstrative exercise.

## Methodology and Implementations

### 1. Exploratory Data Analysis (EDA)

*   **Data Loading & Inspection**: The dataset is loaded into a Pandas DataFrame and initial checks for shape, data types, and missing values are performed. No missing values were found.
*   **Summary Statistics**: Descriptive statistics (`.describe()`) are computed to understand the central tendency and dispersion of each feature.
*   **Visualization**:
    *   **Boxplots and Histograms**: Plotted to visualize the distribution of key features like `BMI`, `Age`, `Glucose`, etc.
    *   **Correlation Heatmap**: A heatmap is generated to visualize the correlation between all features, helping identify potential predictors for the target variable. `Glucose`, `BMI`, and `Age` show a notable correlation with `Outcome`.

### 2. Simple Linear Regression

*   **Implementation**: A Simple Linear Regression model is implemented from scratch to predict `Outcome` using a single feature, `Glucose`.
*   **Method**: The slope (`m`) and intercept (`b`) are calculated using the formulas:
    *   `m = Σ((xi - x̄)(yi - ȳ)) / Σ((xi - x̄)²)`
    *   `b = ȳ - m * x̄`
*   **Result**: The model establishes a basic linear relationship between blood glucose levels and the diabetes outcome.

### 3. Multiple Linear Regression

*   **Implementation**: A Multiple Linear Regression model is built using `scikit-learn`.
*   **Features**: `Glucose`, `BMI`, `Age`
*   **Evaluation**: The model is trained on 80% of the data and tested on the remaining 20%. Performance is evaluated using Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² Score.

### 4. Polynomial Regression

*   **Implementation**: To capture non-linear relationships, a Polynomial Regression model (degree 2) is applied.
*   **Feature**: `Glucose`
*   **Evaluation**: The model's performance is compared against the linear models. Results show a decrease in RMSE and an increase in the R² score, indicating a better fit to the data's non-linear patterns.

### 5. Regularization: Ridge (L2) and Lasso (L1)

To prevent overfitting and improve generalization, Ridge and Lasso regularization techniques are applied to both the Multiple Linear and Polynomial Regression models. Features are scaled using `StandardScaler` before training.

*   **Ridge Regression**: Penalizes the model by adding the "squared magnitude" of the coefficient as a penalty term. It shrinks large coefficients but does not set them to zero.
*   **Lasso Regression**: Penalizes the model by adding the "absolute value of the magnitude" of the coefficient. This method can shrink some coefficients to exactly zero, effectively performing feature selection. In the polynomial model, Lasso identified the squared term of `Glucose` as the most significant feature, simplifying the model.

## Results Summary

The performance of each model was evaluated on the test set. The key metrics are summarized below:

| Model                       | Features            | RMSE     | R² Score |
| --------------------------- | ------------------- | -------- | -------- |
| Multiple Linear             | `Glucose`, `BMI`, `Age` | 0.4310   | 0.1908   |
| Polynomial (Degree 2)       | `Glucose`           | 0.4152   | 0.2491   |
| Ridge (on Multiple Linear)  | `Glucose`, `BMI`, `Age` | 0.4310   | 0.1909   |
| Lasso (on Multiple Linear)  | `Glucose`, `BMI`, `Age` | 0.4293   | 0.1972   |
| Ridge (on Polynomial)       | `Glucose`           | 0.4150   | 0.2497   |
| **Lasso (on Polynomial)**   | **`Glucose`**       | **0.4144** | **0.2521** |

## Observations & Conclusion

*   The Polynomial Regression model provided a better fit than the Multiple Linear Regression model, suggesting a non-linear relationship between `Glucose` and diabetes `Outcome`.
*   Lasso regularization demonstrated its feature selection capability by shrinking less important coefficients to zero, particularly in the polynomial model.
*   Both Ridge and Lasso offered slight performance improvements, with the Lasso-regularized Polynomial model achieving the lowest RMSE and the highest R² score.
*   This assignment provided practical experience in the end-to-end workflow of building, evaluating, and comparing various regression models, highlighting the importance of model selection, diagnostics, and regularization.


