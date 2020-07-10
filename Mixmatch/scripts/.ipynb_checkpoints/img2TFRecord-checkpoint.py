import os  # used for directory operations
import io
import tensorflow as tf
from PIL import Image  # used to read images from directory

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def create_TFRecord(cwd, recordPath, sets):
    # the number of classes of images
    keys = [name for name in os.listdir(cwd) if os.path.isdir(os.path.join(cwd, name))]
    values = range(len(keys))
    classes = dict(zip(keys, values))

    # name format of the tfrecord files
    recordFileName = "OIDv6-" + sets + ".tfrecord"
    # tfrecord file writer
    writer = tf.io.TFRecordWriter(recordPath + recordFileName)

    print("Creating " + sets + " set tfrecord file")
    for name, label in classes.items():
        print(name, label)
        class_path = os.path.join(cwd, name)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            try:
                img = Image.open(img_path, "r")
                
                # Exclude all non RGB images
                if len(img.getbands()) != 3:
                    continue
                img = img.resize((1024, 1024))
                # img = open(img_path, 'rb').read()

                # width, height = img.size
                imgByteArr = io.BytesIO()
                img.save(imgByteArr, format='JPEG')
                imgByteArr = imgByteArr.getvalue()
                # img = img.tobytes()
                example = tf.train.Example(features=tf.train.Features(feature={
                    # 'height': _int64_feature(width),
                    # 'width': _int64_feature(height),
                    # 'depth': _int64_feature(depth),
                    "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[imgByteArr])),
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))}))
                writer.write(example.SerializeToString())
            except:
                print(img_path + " is not valid")

    writer.close()

# dataset file path
cwd = "../OIDv6"
# tfrecord file path
recordPath = "../ML_DATA/"

sets = ['train', 'test']
for s in sets:
    create_TFRecord(os.path.join(cwd,s), recordPath, s)

