from google.cloud import vision
import spacy
import io
import numpy as np
from PIL import Image

IMAGE_SIZE = 64
EMBEDDING_SIZE = 300

# Read image files as byte array.
# Input(path: String), Output(byte array)
def read_image_bytes(path):
    img = Image.open(path, "r")
    imgByteArr = io.BytesIO()
    img.save(imgByteArr, format='JPEG')
    
    return imgByteArr.getvalue()

# Covert opened image to byte array.
# Input(img: PIL image file), Output(byte array)
def image_to_bytes(img):
    imgByteArr = io.BytesIO()
    img.save(imgByteArr, format='JPEG')
    
    return imgByteArr.getvalue()

# This function detects texts in the image using OCR from Google Cloud Vision API.
# Input(img: PIL image file), Output(texts: String)
def detect_text(img):
    client = vision.ImageAnnotatorClient()

    content = image_to_bytes(img)
    image = vision.types.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations

    if response.error.message:
        print(response.error.message)
        return None
    
    if texts:
        texts = texts[0].description
    return texts

# Convert string to vector using spacy package.
# Input(text: String), Output(numpy array)
def word_to_vec(text):
    nlp = spacy.load("en_core_web_lg")
    # process a sentence using the model
    doc = nlp(text)
    
    res = []
    for token in doc:
        if token.has_vector:
            res.append(token.vector)

    return np.array(res)


# Reshpae the embeddings to fit the size of a image channel.
# Input(embeddings: tf.tensor), Output(result: tf.tensor)
def reshape_embeddings(embeddings):
    # EMBEDDING_MAX is the longest embedding size that can fit a channel, for image_size:64 and embeddings_size:300:
    # the embedding_max is 64*64 // 300 * 300 = 13 * 300 = 3900
    EMBEDDING_MAX = IMAGE_SIZE**2 // EMBEDDING_SIZE * EMBEDDING_SIZE
    if tf.shape(embeddings) > EMBEDDING_MAX:
        embeddings = tf.slice(embeddings, [0], [EMBEDDING_MAX])
    zero_padding = tf.zeros([IMAGE_SIZE**2] - tf.shape(embeddings), dtype=tf.float32)
    embeddings_padded = tf.concat([embeddings, zero_padding], 0)
    result = tf.reshape(embeddings_padded, [IMAGE_SIZE, IMAGE_SIZE, 1])
    
    return result

# A pipeline function to do OCR on image, and convert the texts to word embeddings.
# Input(img: PIL image file), Output(texts: String, embeddings: numpy array)
def get_embeddings_from_image(img):
    texts = detect_text(img)
    
    if texts:
        embeddings = word_to_vec(texts).flatten()
    else:
        texts = " "
        embeddings = np.array([0.0])
    
    return texts, embeddings

