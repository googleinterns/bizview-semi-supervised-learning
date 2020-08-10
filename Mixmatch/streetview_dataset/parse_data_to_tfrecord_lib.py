import numpy as np
import tensorflow as tf
import itertools
import os  # used for directory operations
import io
from PIL import Image  # used to read images from directory

tf.enable_eager_execution()

# Global constants
# Information from input tfrecord files
SOURCE_ID = 'image/source_id'
BBOX_CONFIDENCE = 'image/object/bbox/confidence'
BBOX_XMIN = 'image/object/bbox/xmin'
BBOX_YMIN = 'image/object/bbox/ymin'
BBOX_XMAX = 'image/object/bbox/xmax'
BBOX_YMAX = 'image/object/bbox/ymax'

# confidence threshold for determine as neg/pos examples
CONF_THRESHOLD = {'neg': 0.1, 'pos': 0.9}
OUTPUT_IMAGE_SIZE = (64, 64)


# Reads tfrecords and parse the labels and data needed for the new dataset.
def read_tfrecord(file_path):
    raw_image_dataset = tf.data.TFRecordDataset(file_path)

    # Create a dictionary describing the features.
    image_feature_description = {
        SOURCE_ID: tf.io.FixedLenFeature([], tf.string),
        BBOX_CONFIDENCE: tf.io.VarLenFeature(tf.float32),
        BBOX_XMIN: tf.io.VarLenFeature(tf.float32),
        BBOX_YMIN: tf.io.VarLenFeature(tf.float32),
        BBOX_XMAX: tf.io.VarLenFeature(tf.float32),
        BBOX_YMAX: tf.io.VarLenFeature(tf.float32),
    }

    # Parse the input tf.Example proto using the dictionary above.
    def _parse_image_function(example_proto):
        return tf.io.parse_single_example(example_proto, image_feature_description)

    parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
    return parsed_image_dataset


# Parse and cleanup the labels to a more straigtforward format.
def parse_labels(image_features):
    # the format of image_features['image/source_id'] is 'cns/path/to/image_file_name.jpg'
    img_name = str(image_features[SOURCE_ID].numpy()).split('/')[-1][:-1]
    confidence = tf.sparse_tensor_to_dense(image_features[BBOX_CONFIDENCE], default_value=0).numpy()
    xmin = tf.sparse_tensor_to_dense(image_features[BBOX_XMIN], default_value=0).numpy()
    ymin = tf.sparse_tensor_to_dense(image_features[BBOX_YMIN], default_value=0).numpy()
    xmax = tf.sparse_tensor_to_dense(image_features[BBOX_XMAX], default_value=0).numpy()
    ymax = tf.sparse_tensor_to_dense(image_features[BBOX_YMAX], default_value=0).numpy()
    
    bbox = np.vstack((xmin, ymin, xmax, ymax)) # Left, Top, Right, Bottom
    
    return img_name, confidence, bbox


# Transform raw image data and label into a tfexample format.
def img_to_example(img, label):
    imgByteArr = io.BytesIO()
    img.save(imgByteArr, format='JPEG')
    imgByteArr = imgByteArr.getvalue()

    example = tf.train.Example(features=tf.train.Features(feature={
        "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[imgByteArr])),
        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))}))
    
    return example


# Write all images into the test TFrecord file.
def write_tfrecord_from_images(image_folder_path, label, writer):
    for img_name in os.listdir(image_folder_path):
        img_path = os.path.join(image_folder_path, img_name)
        
        try:
            img = Image.open(img_path, "r")
        except Exception as e:
            print(e)
            print(img_path + " is not valid")
            continue
            
        # Exclude all non RGB images
        if len(img.getbands()) != 3:
            continue

        img = img.resize(OUTPUT_IMAGE_SIZE)

        example = img_to_example(img, label)
        writer.write(example.SerializeToString())

# Striped out only the maximum confidence bbox of a image. Function is called in generate_tfexamples_from_detections().
def strip_top_confidence_bbox(confidence, bbox):
    target = []
            
    if confidence.size > 0 and max(confidence) > CONF_THRESHOLD['pos']:
        pos = np.argmax(confidence)
        target.append({'label':1, 'bbox':bbox[:, pos]})
    elif confidence.size == 0 or (confidence.size > 0 and max(confidence) < CONF_THRESHOLD['neg']):
        target.append({'label':0})
    
    return target


# Striped out ALL bbox where confidence is over threshold. Function is called in generate_tfexamples_from_detections().
def strip_all_qualified_bbox(confidence, bbox):
    target = []
    if confidence.size > 0:
        for i in range(confidence.size):
            if confidence[i] > CONF_THRESHOLD['pos'] or confidence[i] < CONF_THRESHOLD['neg']:
                target.append({'label':int(round(confidence[i])), 'bbox':bbox[:, i]})
    
    return target


# Strip the bboxes from the parsed_image_dataset that are over threshold and write it into TFrecord file.
def generate_tfexamples_from_detections(parsed_image_dataset, folder_path, include_top_camera, only_keep_top_confidence):
    examples = {0:[], 1:[]}
    
    for image_features in parsed_image_dataset:
        img_name, confidence, bbox = parse_labels(image_features)
        # The format fo the image_name is XXXXXX_Y.jpg, the Y represents the view of the image.
        view = img_name.split('.')[0][-1]
        keep_image = include_top_camera or (view != '5' and view != '0')
        
        if img_name and keep_image:
            img_path = os.path.join(folder_path, img_name)
            
            if only_keep_top_confidence:
                target = strip_top_confidence_bbox(confidence, bbox)
            else:
                target = strip_all_qualified_bbox(confidence, bbox)
            
            if not target:
                continue

            try:
                img = Image.open(img_path, "r")
            except Exception as e:
                print(e)
                print(img_path + " is not valid")
                continue

            # Exclude all non RGB images
            if len(img.getbands()) != 3:
                continue

            for t in target:
                crop_img = img
                if 'bbox' in t:
                    crop_img = crop_img.crop(t['bbox'])
                crop_img = crop_img.resize(OUTPUT_IMAGE_SIZE)
                example = img_to_example(crop_img, t['label'])
                examples[t['label']].append(example)
                
    return examples

# Write positive and negative tfexamples to tfrecord using the writer. A balance boolean parameter can decide to balance the pos and neg examples count.
def write_tfexample_to_tfrecord(positive_examples, negative_examples, balance, writer):
    take = float('inf')
    num_pos = len(positive_examples)
    num_neg = len(negative_examples)
    
    if balance:    
        take = min(num_pos, num_neg)
    
    for i in range(min(num_pos, take)):
        writer.write(positive_examples[i].SerializeToString())
    
    for i in range(min(num_neg, take)):
        writer.write(negative_examples[i].SerializeToString())


# Write tfrecords in batches of input record files.
def batch_read_write_tfrecords(file_range, input_record_path, input_img_path, writer, detection_property):
    include_top_camera = detection_property['include_top_camera']
    only_keep_top_confidence = detection_property['only_keep_top_confidence']
    balance = detection_property['balance']
    pos_examples, neg_examples = [], []
    
    for i in range(file_range[0], file_range[1]):
        file_name = "./streetlearn_detections_tfexample-" + str(i).zfill(5) + "-of-01000.tfrecord"
        parsed_image_dataset = read_tfrecord(os.path.join(input_record_path, file_name))
        examples = generate_tfexamples_from_detections(parsed_image_dataset, input_img_path, include_top_camera, only_keep_top_confidence)
        neg_examples.extend(examples[0])
        pos_examples.extend(examples[1])
    
    write_tfexample_to_tfrecord(pos_examples, neg_examples, balance, writer)


# Filter the dataset with images lower than the threshold, and store the image names in a list. These images will be handpicked to be used as negative examples in the test set.
def filter_image(parsed_image_dataset, folder_path, threshold):
    res = []
    
    for image_features in parsed_image_dataset:
        img_name, confidence, bbox = parse_labels(image_features)

        if img_name:
            target = []
            if confidence.size == 0 or (confidence.size > 0 and max(confidence) < threshold):
                res.append(img_name)
    return res


