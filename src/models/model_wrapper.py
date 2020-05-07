from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os



class ModelWrapper:
    """
    This class should be used to load and invoke the serialized model and any other required
    model artifacts for pre/post-processing.
    """

    def __init__(self):
        """
        Load the model + required pre-processing artifacts from disk. Loading from disk is slow,
        so this is done in `__init__` rather than loading from disk on every call to `predict`.

        Use paths relative to the project root directory.

        Tensorflow example:

            self.model = load_model("models/model.h5")

        Pickle example:

            with open('models/tokenizer.pickle', 'rb') as handle:
                self.tokenizer = pickle.load(handle)
        """
        # REPLACE ME - add your loading logic
        self.model = load_model('../models/bike_classification_model.model')

    def predict(self,data):
        """
        Returns model predictions.
        """
        # Add any required pre/post-processing steps here.
        image = load_img(data, target_size=(224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)
        image = np.array(image, dtype="float32")
        image = np.expand_dims(image, axis=0)
        

        [response] = self.model.predict(image)
        return 'Mountain: ' + str(response[0]) + ', Road: ' + str(response[1])