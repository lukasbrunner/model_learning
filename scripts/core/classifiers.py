#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
(c) 2023 under a MIT License (https://mit-license.org)

Authors:
- Lukas Brunner || l.brunner@univie.ac.at

Abstract:

"""
import tensorflow.keras as keras

def deep_classifier(input_shape, output_shape):
    classifier = keras.Sequential()

    classifier.add(keras.layers.Conv2D(
        filters=128,  # depth of the network (=number of kernels)
        kernel_size=(5, 5),  # size of the kernels 
        strides=(2, 2),  # steps in which the kernels are moved
        padding='same',  # wether to pad zeros at the edges
        activation='relu', 
        input_shape=input_shape
    ))
    classifier.add(keras.layers.MaxPool2D(pool_size=4))

    classifier.add(keras.layers.Conv2D(
        filters=64, kernel_size=3, 
        strides=1, padding='same',
        activation='relu'))
    classifier.add(keras.layers.MaxPool2D(pool_size=2))

    classifier.add(keras.layers.Conv2D(
        filters=64, kernel_size=3, 
        strides=1, padding='same',
        activation='relu'))
    classifier.add(keras.layers.MaxPool2D(pool_size=2))

    classifier.add(keras.layers.Conv2D(
        filters=32, kernel_size=2, 
        strides=1, padding='same',
        activation='relu'))
    classifier.add(keras.layers.MaxPool2D(pool_size=2))

    classifier.add(keras.layers.Flatten())
    classifier.add(keras.layers.Dense(units=output_shape, activation='softmax'))
    
    classifier.compile(
        optimizer=keras.optimizers.Adam(), 
        loss='sparse_categorical_crossentropy', 
        metrics=['acc'])
    
    return classifier