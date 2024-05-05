import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import os
import pandas as pd
from csv import writer
from keras import utils
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import h5py

def main():
    with h5py.File('.../Galaxy10.h5', 'r') as f:
        images = np.array(f['images'])
        labels = np.array(f['ans'])
    f.close()
    labels = utils.to_categorical(labels, 10)
    image_num = len(labels)
    label_ext = np.array([])
    image_ext = np.array([])
    for no in range(image_num):
        image_no = images[no]
        image_hflip = tf.image.flip_left_right(image_no)
        if len(image_ext) == 0:
            image_ext = np.vstack((np.array([image_no]), np.array([image_hflip])))
        else:
            image_ext = np.vstack((image_ext, np.array([image_no])))
            image_ext = np.vstack((image_ext, np.array([image_hflip])))
        image_vhflip = tf.image.flip_up_down(image_hflip)
        image_ext = np.vstack((image_ext, np.array([image_vhflip])))
        image_hvhflip = tf.image.flip_left_right(image_vhflip)
        image_ext = np.vstack((image_ext, np.array([image_hvhflip])))
        label_arr = np.array([np.argmax(labels[no]), np.argmax(labels[no]), np.argmax(labels[no]), np.argmax(labels[no])])
        label_ext = np.append(label_ext, label_arr)
    # To convert the labels to categorical 10 classes
    

    # Select 10 of the images to inspect
    img = None
    plt.ion()

    # To convert to desirable type
    labels = label_ext.astype(np.float32)
    images = image_ext.astype(np.float32)

    # Split the dataset into training set and testing set
    train_idx, test_idx = train_test_split(np.arange(labels.shape[0]), test_size=0.1)
    train_images, train_labels, test_images, test_labels = images[train_idx], labels[train_idx], images[test_idx], labels[test_idx]
    train_num = len(train_labels)
    test_num = len(test_labels)
    with open('.../galaxy_train_data.csv', 'a', encoding='utf-8') as f:
        writer_object = writer(f)
        for sno in range(train_num):
            image_no = train_images[sno]
            image_no = image_no.reshape((1, 69*69*3))
            flux_list = list(image_no[0])          
            writer_object.writerow(flux_list)
            # Close the file object
        f.close()
    
    with open('.../galaxy_train_label.csv', 'a', encoding='utf-8') as f:
        writer_object = writer(f)
        for sno in range(train_num):
            label = train_labels[sno]
            writer_object.writerow([int(label)])
            # Close the file object
        f.close()
    
    with open('.../galaxy_test_data.csv', 'a', encoding='utf-8') as f:
        writer_object = writer(f)
        for sno in range(test_num):
            #serial_no = serial_no + 1
            image_no = test_images[sno]
            image_no = image_no.reshape((1, 69*69*3))
            flux_list = list(image_no[0])          
            writer_object.writerow(flux_list)
            # Close the file object
        f.close()
    
    with open('.../galaxy_test_label.csv', 'a', encoding='utf-8') as f:
        writer_object = writer(f)
        for sno in range(test_num):
            label = test_labels[sno]
            writer_object.writerow([int(label)])
            # Close the file object
        f.close()

if __name__ == '__main__':
    main()
