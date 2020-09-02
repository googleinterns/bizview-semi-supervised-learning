from google.cloud import vision
import spacy
import io
import numpy as np
from PIL import Image

MAX_EMBEDDINGS = 13
OUTPUT_SHAPE = 64

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

# Reshape 1D array to a square matrix
# Input(embeddings: )
def reshape_embeddings(embeddings):
    filter_embeddings = np.copy(embeddings[: min(MAX_EMBEDDINGS, len(embeddings))])
    filter_embeddings.resize((OUTPUT_SHAPE, OUTPUT_SHAPE, 1))
    
    return filter_embeddings

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

