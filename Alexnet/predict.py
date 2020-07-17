#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 18:45:05 2020

@author: sudhanshukumar
"""

import numpy as np
from keras.models import load_model
from keras.preprocessing import image

class dogcat:
    def __init__(self,filename):
        self.filename =filename


    def predictiondogcat(self):
        # load model
        model = load_model('model.h5')

        # summarize model
        #model.summary()
        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = model.predict(test_image)

        if result[0][0] == 1:
            prediction = 'baby'
            return [{"image": prediction}]
        elif result[0][1] == 1:
            prediction = 'cat'
            return [{"image": prediction}]
        elif result[0][2] == 1:
            prediction = 'flower'
            return [{"image": prediction}]
        elif result[0][3] == 1:
            prediction = 'girl'
            return [{"image": prediction}]
        elif result[0][4] == 1:
            prediction = 'map'
            return [{"image": prediction}]
        elif result[0][5] == 1:
            prediction = 'mumma'
            return [{"image": prediction}]
        elif result[0][6] == 1:
            prediction = 'orange'
            return [{"image": prediction}]
        elif result[0][7] == 1:
            prediction = 'sea'
            return [{"image": prediction}]
        elif result[0][8] == 1:
            prediction = 'tajmahal'
            return [{"image": prediction}]

