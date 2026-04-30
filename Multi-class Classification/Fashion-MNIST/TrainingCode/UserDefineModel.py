# Import dependencies
import tensorflow as tf
from tensorflow import keras
import joblib

# 1. Load dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
               'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 2. Data normalization
train_images_norm = train_images / 255.0
test_images_norm = test_images / 255.0

L = keras.layers

# 3. Build the model
model = keras.Sequential([
    L.Flatten(input_shape=(28, 28)),
    L.Dense(128, activation=tf.nn.relu),
    L.Dense(10, activation=tf.nn.softmax)
])

# 4. Compile the model
model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. Train the model
model.fit(train_images_norm,
          train_labels,
          epochs=5,               # Total 5 training epochs
          validation_split=0.2)   # Use 20% of data for validation

# 6. Evaluate the model
test_loss, test_acc = model.evaluate(test_images_norm, test_labels)
print(f'Test accuracy: {test_acc}')

# 7. Model prediction
predictions = model.predict(test_images_norm)

# 8. Save the trained model
model_filename = 'UserDefineModel.joblib'
joblib.dump(model, model_filename)

print(f"Model successfully saved as: {model_filename}")

# --- Supplement: How to use the model if downloaded from GitHub ---
# loaded_model = joblib.load('UserDefineModel.joblib')
# result = loaded_model.predict(new_data)