import pandas as pd
import numpy as np
import json
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import Sequential



def getData():
    """
    return both the dataset and corresponding dataframe
    """
    file = '../data/shipsnet.json'
    with open(file) as dataFile:
        dataset = json.load(dataFile)
        
    ships_df = pd.DataFrame(dataset)
    return dataset, ships_df



def describeData(a,b):
    print('Total number of images: {}'.format(len(a)))
    print('Number of NoShip Images: {}'.format(np.sum(b==0)))
    print('Number of Ship Images: {}'.format(np.sum(b==1)))
    print('Percentage of positive images: {:.2f}%'.format(100*np.mean(b)))
    print('Image shape (Width, Height, Channels): {}'.format(a[0].shape))
    
    
    
def reshape(x, y):
    t = x.reshape([-1, 3, 80, 80]).transpose([0,2,3,1])
    tt = to_categorical(y, num_classes=2)
    return t, tt


def showRandomImages(ships, notships):
    """
    Params: ships-  array of images containing ships
            notships- array of images without ships
    prints the images in two rows with ships on top
    """
    plt.figure(figsize=(16,8))
    rows = 2
    cols = 5
    ships_flag = True
    # get 5 of ships and not ships
    for row in range(rows):
        images = ships if ships_flag else notships
        length = images.shape[0]
        rx = [int(np.random.random() * length) for e in range(5)]
        for col in range(cols):
            mpl_idx = (col+1) + row*5
            plt.subplot(2,5,mpl_idx)
            title = "ship" if ships_flag else "not ship"
            plt.title(title)
            img = images[rx[col]]
            plt.imshow(img)
            plt.axis('off')
        ships_flag = not ships_flag
        

        
def test_augmentation(aug, images, row=2, col=5):
    """
    Params: 
    aug- Sequential tensorflow layers of augmentation rules
    images- array of color images
    
    shows augmented images
    """
 
    idx = int(np.random.random() * images.shape[0])
    print(idx)
    print(images.shape)
    rand_ship_image = images[idx]
    print(rand_ship_image.shape)

    # Add the image to a batch
    image = tf.expand_dims(rand_ship_image, 0)
   
    plt.figure(figsize=(col*3, row*3))
    for i in range(row * col):
        #ai = data_augmentation(image)
        plt.subplot(row,col,i+1);
        plt.imshow(aug(image)[0]);
        plt.axis('off');