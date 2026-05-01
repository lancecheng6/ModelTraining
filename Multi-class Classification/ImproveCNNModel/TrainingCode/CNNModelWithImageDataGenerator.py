# First run the following commands (Download dataset & Install dependencies)
#!wget https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
#!tar -xvf flower_photos.tgz 
#!mkdir data && mv flower_photos data/flower_photos
#!pip install pillow

# 1. Import dependencies
import os, shutil
import pathlib
import random
import IPython.display as display
import tensorflow as tf
import matplotlib.pyplot as plt
import shutil
import joblib
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras

# 2. Define functions
def crate_sub_dataset(images, labels, dataset_path):
    for index, image_path in enumerate(images):
        image_label = labels[index]

        # Create the label dataset directory
        target_dir = os.path.join(dataset_path, image_label)
        pathlib.Path(target_dir).mkdir(parents=True, exist_ok=True)

        # Copy the image file to the target folder
        target_path = os.path.join(target_dir, pathlib.Path(image_path).name)
        shutil.copyfile(image_path, target_path)
    return dataset_path

def visualize_keras_history(history):
    plt.figure()
    # Set subplot size
    plt.subplots(figsize=(10,9))

    plt.subplot(2, 2, 1)
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.legend()
    plt.title('train and validate accuracy') # Add chart title

    plt.subplot(2, 2, 2)
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.title('train and validate loss') # Add chart title

    plt.show()

# 3. Get image paths
data_root = pathlib.Path('/kaggle/working/data/flower_photos/')
all_image_paths = list(data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]

# all_image_paths is an array containing all image file paths
# ['/kaggle/working/data/flower_photos/roses/5061135742_2870a7b691_n.jpg',
#  '/kaggle/working/data/flower_photos/dandelion/5613466853_e476bb080e.jpg',
#  '/kaggle/working/data/flower_photos/tulips/14053292975_fdc1093571_n.jpg',
#  '/kaggle/working/data/flower_photos/dandelion/2634666217_d5ef87c9f7_m.jpg',
#  '/kaggle/working/data/flower_photos/dandelion/5598591979_ed9af1b3e9_n.jpg']

# 4. Shuffle the data order
random.shuffle(all_image_paths)

#print(len(all_image_paths))  # 3670

# 5. Parse class names from the image paths
all_image_labels = [pathlib.Path(path).parent.name for path in all_image_paths]

# 6. Use train_test_split method to split the dataset
train_x, valid_x, train_y, valid_y = train_test_split(all_image_paths,
                                                      all_image_labels,
                                                      train_size = 0.8,
                                                      random_state = 104)

# 7. Delete existing files
shutil.rmtree('/kaggle/working/data/flower-set/train', ignore_errors=True)
shutil.rmtree('/kaggle/working/data/flower-set/valid', ignore_errors=True)

train_dataset_path = crate_sub_dataset(train_x, train_y, '/kaggle/working/data/flower-set/train')
valid_dataset_path = crate_sub_dataset(valid_x, valid_y, '/kaggle/working/data/flower-set/valid')

# 8. Create two image generators; the rescale attribute normalizes the image tensors by multiplying by 1/255
train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)

# 9. Use the image generators to read images from the directory
train_generator = train_datagen.flow_from_directory(
    directory=train_dataset_path, # Directory for data reading
    target_size=(192, 192),    # Tensor dimensions; all images will be resized to this size
    batch_size=100)            # Batch size

valid_generator = valid_datagen.flow_from_directory(
    directory=valid_dataset_path,
    target_size=(192, 192),
    batch_size=100)
    
plt.rcParams['figure.dpi'] = 120

L = keras.layers

# 10. Build and compile the model
base_model = keras.Sequential([
    L.Conv2D(input_shape=(192, 192, 3), filters=32, kernel_size=5, strides=1),
    L.MaxPool2D(pool_size=2, strides=2),
    L.Conv2D(filters=64, kernel_size=3, strides=1),
    L.MaxPool2D(pool_size=2, strides=2),
    L.Flatten(),
    L.Dense(128, activation=tf.nn.relu),
    L.Dense(5, activation=tf.nn.softmax)
])
base_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=5e-4),
    loss=tf.losses.CategoricalCrossentropy(),
    metrics=['accuracy'])
base_model.summary()

tf_board_path = '/kaggle/working/tf_dir/base_model'
shutil.rmtree(tf_board_path, ignore_errors=True)

# 11. Train the model
history = base_model.fit(
      train_generator,                 # Training batch generator
      steps_per_epoch=30,              # Number of training batches
      epochs=50,                       # Total training epochs
      validation_data=valid_generator, # Validation batch generator
      validation_steps=8,              # Number of validation batches
      verbose=1,
#     callbacks=[keras.callbacks.TensorBoard(tf_board_path)],
)

# 12. Visualize training process accuracy
plt.rcParams['figure.dpi'] = 120

visualize_keras_history(history)

# 13. Save the trained model
model_filename = 'CNNModelWithImageDataGenerator.joblib'
joblib.dump(base_model, model_filename)

print(f"Model successfully saved as: {model_filename}")

# --- Supplement: How to use the model if downloaded from GitHub ---
# loaded_model = joblib.load('CNNModelWithImageDataGenerator.joblib')
# result = loaded_model.predict(new_data)

#del base_model