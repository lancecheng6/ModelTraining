# First run the following commands (Download dataset & Install dependencies)
#!wget https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
#!tar -xvf flower_photos.tgz 
#!mkdir data && mv flower_photos data/flower_photos
#!pip install pillow

# 1. Import dependencies
import os, shutil
import pathlib
import IPython.display as display
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib
import json
from tensorflow import keras
from tensorflow.keras.applications import VGG16
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

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

# 4. Parse class names from the image paths
all_image_labels = [pathlib.Path(path).parent.name for path in all_image_paths]

# 5. Use train_test_split method to split the dataset
train_x, valid_x, train_y, valid_y = train_test_split(all_image_paths,
                                                      all_image_labels,
                                                      train_size = 0.8,
                                                      random_state = 104)

train_dataset_path = crate_sub_dataset(train_x, train_y, '/kaggle/working/data/flower-set/train')
valid_dataset_path = crate_sub_dataset(valid_x, valid_y, '/kaggle/working/data/flower-set/valid')

# 6. Initialize image data generator
random_data_gen = ImageDataGenerator(
    rescale=1./255,               # Normalization
    rotation_range=40,            # Range for random rotations
    width_shift_range=0.2,        # Range for random horizontal shifts (ratio of total width)
    height_shift_range=0.2,       # Range for random vertical shifts (ratio of total height)
    shear_range=0.2,              # Range for shear transformation
    zoom_range=0.2,               # Range for random zoom
    brightness_range=(0.6, 1.2),  # Range for brightness adjustment
    horizontal_flip=True          # Whether to randomly flip images horizontally
)

valid_data_gen = ImageDataGenerator(rescale=1./255)

train_aug_gen = random_data_gen.flow_from_directory( # This function takes images from folders and feeds to ImageDataGenerator
        train_dataset_path,
        target_size=(192, 192),
        batch_size=100,
        class_mode='categorical'
)

valid_aug_gen = valid_data_gen.flow_from_directory(
        valid_dataset_path,
        target_size=(192, 192),
        batch_size=100,
        class_mode='categorical',
        shuffle=False)

# 7. Build the VGG16 Convolutional Base
vgg_base = VGG16(weights='imagenet',      # Load ImageNet weights; if None, weights are initialized randomly
                 include_top=False,       # Whether to include the fully-connected layer at the top of the network
                 input_shape=(192,192,3)) # Input image dimensions
vgg_base.summary()


# 8. Freeze all layers in the convolutional base
# Note: After modifying the "trainable" property, the model must be recompiled for changes to take effect
for layer in vgg_base.layers:
    layer.trainable = False

L = keras.layers
# 9. Create a new model
feature_extract_model = keras.Sequential([
    vgg_base,    # Use the convolutional base as the first layer (adds all layers from VGG16 base)
    L.Flatten(), # Flatten the upper features into a 1D tensor
    L.Dense(64, activation=tf.nn.relu),
    L.Dropout(0.5),
    L.Dense(5, activation=tf.nn.softmax)
])

feature_extract_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=2e-4),
    loss=tf.losses.CategoricalCrossentropy(),
    metrics=['accuracy'])

feature_extract_model.summary()
        
tf_board_path = '/kaggle/working/tf_dir/vgg_feature_extract'
shutil.rmtree(tf_board_path, ignore_errors=True)

# 10. Start training
vgg_feature_extract_hist = feature_extract_model.fit(
      train_aug_gen,                   # Training batch generator
      steps_per_epoch=30,              # Number of training steps per epoch
      epochs=50,                       # Total number of training epochs
      validation_data=valid_aug_gen,   # Validation batch generator
      validation_steps=8,              # Number of validation steps
      verbose=1,
      callbacks=[keras.callbacks.TensorBoard(tf_board_path)]
)    
#del feature_extract_model

# 11. Visualize training accuracy
visualize_keras_history(vgg_feature_extract_hist)


# 12. Save the trained model
model_filename = 'TransferLearningUsingVGG16AndUserDefineClassifier.joblib'
joblib.dump(feature_extract_model, model_filename)

print(f"Model successfully saved as: {model_filename}")

# --- Supplement: How to use the model if downloaded from GitHub ---
# loaded_model = joblib.load('TransferLearningUsingVGG16AndUserDefineClassifier.joblib')
# result = loaded_model.predict(new_data)

# 13. Save image classification indices
# Create directory
save_path = pathlib.Path('/kaggle/working/data/outputs/flower_recognizer/vgg16')
save_path.mkdir(parents=True, exist_ok=True)

with open(os.path.join(save_path, 'label2idx.json'), 'w') as f:
    f.write(json.dumps(train_aug_gen.class_indices))

#del feature_extract_model