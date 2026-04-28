# Import libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import joblib

def preprocess_dataset(data_frame):
    data_frame = data_frame.copy()
    data_frame.Sex = data_frame.Sex.map({'male': 0, 'female': 1})
    data_frame.Embarked = data_frame.Embarked.map({'S': 0, 'C': 1, 'Q': 2})
    
    # Handle missing values (Impute with median instead of dropping rows)
    data_frame['Age'] = data_frame['Age'].fillna(data_frame['Age'].median())
    data_frame['Fare'] = data_frame['Fare'].fillna(data_frame['Fare'].median())
    
    # Drop irrelevant columns
    data_frame = data_frame.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'])
    return data_frame

# 1. Load data
train_df = pd.read_csv('/kaggle/input/datasets/lancecheng666888/titanic/train.csv')
test_df = pd.read_csv('/kaggle/input/datasets/lancecheng666888/titanic/test.csv')

# 2. Process data
train_data = preprocess_dataset(train_df)
test_data = preprocess_dataset(test_df)

# 3. Prepare features and target
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X_train = train_data[features]
y_train = train_data['Survived']
X_test = test_data[features]

# 4. Initialize and train the model (using the full training set)
# n_estimators: Number of trees in the forest
# max_depth: Maximum depth of the tree to prevent overfitting
model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
model.fit(X_train, y_train)

# 5. Estimate accuracy using 5-fold cross-validation
scores = cross_val_score(model, X_train, y_train, cv=5)

print(f"Estimated average accuracy: {scores.mean():.2%}")
print(f"Accuracy Standard Deviation (Stability): {scores.std():.2%}")

# 6. Perform predictions
predictions = model.predict(X_test)

# 7. Format the results (Output in Kaggle submission format)
submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': predictions
})

# 8. Save the submission file
submission.to_csv('my_prediction.csv', index=False)
print("Prediction complete! Results saved to my_prediction.csv")

# 9. Save the trained model (typically saved as .joblib or .pkl)
model_filename = 'RandomForestModel.joblib'
joblib.dump(model, model_filename)

print(f"Model successfully saved as: {model_filename}")

# --- Supplement: How to use the model if downloaded from GitHub ---
# loaded_model = joblib.load('RandomForestModel.joblib')
# result = loaded_model.predict(new_data)