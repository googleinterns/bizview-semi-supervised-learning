import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import SGD, RMSprop
import numpy as np

tf.enable_eager_execution()

# Read in tfrecord files and construct dataset from it
def create_dataset(filepath):
    raw_image_dataset = tf.data.TFRecordDataset(filepath)

    # Create a dictionary describing the features.
    image_feature_description = {
        'label': tf.io.FixedLenFeature([], tf.int64),
        'image': tf.io.FixedLenFeature([], tf.string),
    }

    def _parse_image_function(example_proto):
      # Parse the input tf.Example proto using the dictionary above.
      return tf.io.parse_single_example(example_proto, image_feature_description)

    parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
    
    image, label = [], []
    
    for image_features in parsed_image_dataset:
        label.append(image_features['label'])
        image.append(tf.image.decode_image(image_features['image']).numpy())
    
    return np.asarray(image), np.asarray(label)


# Callback function to check test set accuracy and loss for every training epoch.
class test_callback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        image, label = self.test_data
        loss, acc = self.model.evaluate(image, label, verbose=0)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))


# Create transfer learning model from pretrained Resnet50 model.
def create_model(image_size, num_classes, lr):
    input_tensor = Input(shape=(image_size, image_size, 3))
    # Create the base pre-trained model
    base_model = ResNet50(weights='imagenet', include_top=False)

    # Add fully connected layers on top of the model
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # First: train only the top layers (which were randomly initialized) Thus, freeze all convolutional ResNet50 layers
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=RMSprop(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


# Modify the training layers for the tuning phase
def tune_model(model, trainable_layer, lr):
    # Train the top 1 convolution block. Therefore, freezing the first trainable_layer layers and unfreeze the rest:
    for layer in model.layers[:trainable_layer]:
        layer.trainable = False
    for layer in model.layers[trainable_layer:]:
        layer.trainable = True
    
    model.compile(optimizer=SGD(lr=lr, momentum=0.8), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


# Visualize layer names and layer indices to see how many layers
def visualize_model(model):
    for i, layer in enumerate(model.layers):
        print(i, layer.name)