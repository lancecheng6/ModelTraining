# Import fundamental libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from sklearn.model_selection import KFold
import joblib

# Set plot resolution
plt.rcParams['figure.dpi'] = 180

def preprocess_dataset(data_frame):
    # Create a copy to avoid modifying the original dataset
    data_frame = data_frame.copy()

    # Drop rows with missing values in critical columns
    data_frame = data_frame.dropna(subset=['Age', 'Sex', 'Embarked', 'Fare'])

    # Map 'Sex' from strings to integers (male: 0, female: 1)
    data_frame.Sex = data_frame.Sex.map({'male': 0, 'female': 1})

    # Map 'Embarked' from strings to integers (S: 0, C: 1, Q: 2)
    data_frame.Embarked = data_frame.Embarked.map({'S': 0, 'C': 1, 'Q': 2})
    
    # Drop unnecessary columns
    data_frame = data_frame.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'])
    
    return data_frame

# Alias keras.layers as L for convenience
L = keras.layers

def build_model() -> keras.Sequential:
    # Define the model architecture
    model = keras.Sequential([
        # Use Input layer to define the input shape (7 features) explicitly
        keras.Input(shape=(7,)),
        # Hidden layer 1: 24 neurons, ReLU activation
        L.Dense(24, activation='relu', name='hidden_layer1'),
        # Hidden layer 2: 12 neurons, ReLU activation
        L.Dense(12, activation='relu', name='hidden_layer2'),
        # Output layer: 1 neuron with Sigmoid activation for binary classification (0 to 1)
        L.Dense(1, activation='sigmoid', name='output_layer')
    ])
    # Compile the model with parameters for binary classification
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# --- Part 1: Direct Training with train.csv ---
df = pd.read_csv('/kaggle/input/datasets/lancecheng666888/titanic/train.csv')

# Preprocess and separate features and labels
train_data_full = preprocess_dataset(df)
train_labels = train_data_full.pop('Survived')
train_features = train_data_full

model = build_model()

# Train the model
model.fit(train_features.values, train_labels.values, epochs=50)

# Save the model (filenames usually end in .joblib, .pkl, or .keras)
model_filename = 'UserDefineModel.joblib'
joblib.dump(model, model_filename)

print(f"Model successfully saved as: {model_filename}")

# --- Note: How to use the saved model in the future ---
# loaded_model = joblib.load('UserDefineModel.joblib')
# result = loaded_model.predict(new_data)


# --- Part 2: Model Evaluation using K-Fold Cross-Validation ---
# Process the original dataset again for K-Fold
train_data_kfold = preprocess_dataset(df)

# Lists to store accuracy and training history
accuracy_list = []
history_list = []

# Initialize KFold object with 5 splits
kfold = KFold(n_splits=5)

for train_index, test_index in kfold.split(train_data_kfold):

    # Split data into training and testing sets for this fold
    # Using .copy() to prevent SettingWithCopy warnings
    train = train_data_kfold.iloc[train_index].copy()
    test =  train_data_kfold.iloc[test_index].copy()

    # Separate labels from features
    train_label = train.pop('Survived')
    test_label = test.pop('Survived')

    # Build a fresh model for each fold
    model = build_model()
    
    # Train the model
    # validation_split=0.2 uses 20% of the training fold for validation
    history = model.fit(train.values,
                        train_label.values,
                        validation_split=0.2,
                        verbose=0,
                        epochs=100)

    # Evaluate the model on the test fold
    loss, accuracy = model.evaluate(test.values, test_label.values)
    accuracy_list.append(accuracy)
    history_list.append(history)

print("K-Fold Accuracy List:", accuracy_list)

# Visualize the training process using Matplotlib
plt.figure()
plt.subplots(figsize=(10,9))
for index, his in enumerate(history_list):
    plt.subplot(3, 2, index + 1)
    plt.plot(his.history['accuracy'], label='accuracy')
    plt.plot(his.history['val_accuracy'], label='val_accuracy')
    plt.legend()
    plt.title(f'K-Split {index}') # Graph title
plt.tight_layout()
plt.show()


# --- Part 3: Prediction using test.csv ---
# Load test data
raw_test_df = pd.read_csv('/kaggle/input/datasets/lancecheng666888/titanic/test.csv')

# Preprocess the test set using the same logic as training
test_df = preprocess_dataset(raw_test_df)

# Call predict() for the first 10 rows (returns probabilities)
predictions = model.predict(test_df[:10].values)

# Convert probabilities to classes (threshold = 0.5)
# Use astype(int) to convert boolean to 0 or 1
predicted_classes = (predictions > 0.5).astype(int)

print("Predictions for first 10 rows:")
print(predicted_classes)