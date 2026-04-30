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
conv_model = keras.Sequential([
    # Convolutional layer with 32 filters (3x3), output dimension: [28-3+1, 28-3+1, 32]
    L.Conv2D(input_shape=(28, 28, 1), filters=32, kernel_size=3, strides=1),
    # Max pooling layer
    L.MaxPool2D(pool_size=2, strides=2),
    # Convolutional layer
    L.Conv2D(filters=64, kernel_size=3, strides=1),
    # Max pooling layer
    L.MaxPool2D(pool_size=2, strides=2),
    L.Flatten(),
    L.Dense(256, activation=tf.nn.relu),
    L.Dense(10, activation=tf.nn.softmax)
])

# 4. Compile the model
conv_model.compile(optimizer=tf.optimizers.Adam(),
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])

conv_model.summary()

# 5. Reshape input
# Since Conv2D requires a 3D tensor input [height, width, channels]
# Convert data from [28, 28] to [28, 28, 1] (adding the grayscale channel)
train_images_reshape = train_images_norm.reshape([-1, 28, 28, 1])
test_images_reshape = test_images_norm.reshape([-1, 28, 28, 1])

# 6. Train the model
conv_model.fit(train_images_reshape,
               train_labels,
               epochs=5,               # Total 5 training epochs
               validation_split=0.2)   # Use 20% of data for validation

# 7. Evaluate the model
test_loss, test_acc = conv_model.evaluate(test_images_reshape, test_labels)
print(f'Conv model test accuracy: {test_acc}')

# 8. Save the trained model
model_filename = 'UserDefineCNNModel.joblib'
joblib.dump(conv_model, model_filename)

print(f"Model successfully saved as: {model_filename}")

# --- Supplement: How to use the model if downloaded from GitHub ---
# loaded_model = joblib.load('UserDefineCNNModel.joblib')
# result = loaded_model.predict(new_data)