# Titanic Survival Prediction: Model Comparison

This project aims to compare the prediction accuracy of different machine learning models in predicting the survival of passengers on the Titanic. By evaluating various algorithms, we identify which approach best captures the patterns within the survival data.

## 📁 Project Structure

The project is organized into the following directory structure within the Titanic folder:

```
Titanic/
├── SavedModel/
│   ├── RandomForestModel.joblib
│   └── UserDefineModel.joblib
├── TrainingCode/
│   └── (Python scripts for preprocessing and training)
└── TrainingData/
    ├── train.csv
    └── test.csv
```

- **SavedModel/**: Contains the serialized, pre-trained models ready for deployment.
- **TrainingCode/**: Contains the Python scripts (.py or .ipynb) used for data preprocessing, feature engineering, and model training.
- **TrainingData/**: Stores the original datasets used for the project.

## 📊 Model Performance Comparison

We compared two distinct models based on their accuracy scores. The Random Forest Model currently serves as our primary performer.

| Model | Accuracy | Filename |
|-------|----------|----------|
| Random Forest Model | 81.93% | RandomForestModel.joblib |
| User Defined Model | 76.52% | UserDefineModel.joblib |

## 🛠️ Technical Details

### 1. Training Environment
- **Platform**: Kaggle
- **Language**: Python 3.x
- **Key Libraries**: Scikit-Learn, Pandas, NumPy, Joblib

### 2. Feature Engineering
Both models were trained using the following features:

- **Features (X)**: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
- **Target Label (y)**: Survived (0 = No, 1 = Yes)

### 3. Preprocessing Steps
- **Sex**: Encoded as numerical values (Male: 0, Female: 1)
- **Age/Fare**: Missing values were handled via median imputation
- **Embarked**: Port of embarkation was encoded numerically (S: 0, C: 1, Q: 2)

## 💻 How to Load the Saved Models

You can use the following Python snippet to load the models from the SavedModel folder:

```python
import joblib

# Load the Random Forest Model
rf_model = joblib.load('SavedModel/RandomForestModel.joblib')

# Load the User Defined Model
ud_model = joblib.load('SavedModel/UserDefineModel.joblib')

# Make predictions
# predictions = rf_model.predict(X_test)
```
