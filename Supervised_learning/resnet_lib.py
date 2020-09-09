from resnet50_mod import ResNet50_mod
from streetview_dataset.OCR_lib import reshape_embeddings
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import SGD, RMSprop
import numpy as np

tf.enable_eager_execution()



# Choose the correct parsing features based on if there is word embeddings in the dataset.
# Input(word_embeddings: bool), Output(dictionary)
def image_feature(word_embeddings):
    # Create a dictionary describing the features.
    if word_embeddings:
        image_feature_description = {
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
            'texts': tf.io.FixedLenFeature([], tf.string),
            'embeddings': tf.io.VarLenFeature(dtype=tf.float32)
        }
    else:
        image_feature_description = {
            'label': tf.io.FixedLenFeature([], tf.int64),
            'image': tf.io.FixedLenFeature([], tf.string),
        }
        
    return image_feature_description


# Read in tfrecord files and construct dataset from it
# Input(filepath: string), Output(tuple of two numpy arrays)
def create_dataset(filepath, word_embeddings=False):
    raw_image_dataset = tf.data.TFRecordDataset(filepath)
    image_feature_description = image_feature(word_embeddings)
    
    def _parse_image_function(example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, image_feature_description)

    parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
    
    image, label = [], []
    
    # The word_embeddings boolean determine if there is word_embedding data within the tfrecord file. The data preprocessing for two situations are different.
    if word_embeddings:
        for image_features in parsed_image_dataset:
            img = tf.image.decode_image(image_features['image'])
            # Casting the raw image data from the range of 0 ~ 255 to a normalized range of -1.0 ~ 1.0.
            img = tf.cast(img, tf.float32) * (2.0 / 255) - 1.0
            embeddings = reshape_embeddings(image_features['embeddings'].values)
            in_data = tf.concat([img, embeddings], axis=2).numpy()
            
            label.append(image_features['label'])
            image.append(in_data)
    else:
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
# Inputs(image_size: int, num_classes: int, lr: float). Outputs(model: keras Model class)
def create_model(image_size, num_classes, lr):
    # Add input layer and data preprocessing to model
    input_tensor = Input(shape=image_size)
    
    # Add pretrained imagenet weights on ResNet50 if the input images are 3 channels. Otherwise, load the modified resnet50 model.
    if image_size[-1] == 3:
        x = tf.cast(input_tensor, tf.float32)
        x = preprocess_input(x)
        base_model = ResNet50(weights='imagenet', include_top=False)(x)
        # First: train only the top layers (which were randomly initialized) Thus, freeze all convolutional ResNet50 layers
        for layer in base_model.layers:
            layer.trainable = False
    else:
        base_model = ResNet50_mod(input_shape=image_size, weights=None, include_top=False)(input_tensor)
    
    # Add fully connected layers on top of the model
    x = GlobalAveragePooling2D()(base_model)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input_tensor, outputs=predictions)

    model.compile(optimizer=RMSprop(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


# Modify the training layers for the tuning phase
# Inputs(model: keras Model class, trainable_layer: int, lr: float), Outputs(model: keras Model class)
def tune_model(model, trainable_layer, lr):
    # Train the top 1 convolution block. Therefore, freezing the first trainable_layer layers and unfreeze the rest:
    for layer in model.layers[:trainable_layer]:
        layer.trainable = False
    for layer in model.layers[trainable_layer:]:
        layer.trainable = True
    
    model.compile(optimizer=SGD(lr=lr, momentum=0.8), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


# Visualize layer names and layer indices to see how many layers
# Input(model: keras Model class)
def visualize_model(model):
    for i, layer in enumerate(model.layers):
        print(i, layer.name)