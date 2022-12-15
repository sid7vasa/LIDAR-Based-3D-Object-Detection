import tensorflow as tf
import numpy as np
from imutils import paths
import os
from PIL import Image
import random
from tqdm import tqdm

# Tensorflow datatype conversion for writing tf records.
def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):  # if value ist tensor
        value = value.numpy()  # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_array(array):
    array = tf.io.serialize_tensor(array)
    return array


def preprocess_data(X1, X2, X3):
    """
    Pre processing for the inputs and outputs.
    Parameters
    ----------
    X1 : Input (satellite)
        DESCRIPTION.
    X2 : Output (Maps)
        DESCRIPTION.
    Returns
    -------
    list
        DESCRIPTION. Preprocessed input and output.
    """
    X1 = X1/255.0
    return [X1, X2, X3]


def parse_single_image(image, label, token):
    """
    How to represent data in the tf records.
    Needed information to reparse and load the data as needed.
    Parameters
    ----------
    image : Input images.
        DESCRIPTION.
    label : Output images.
        DESCRIPTION.
    Returns
    -------
    out : TYPE
        DESCRIPTION.
    """
    data = {
        'height': _int64_feature(image.shape[0]),
        'width': _int64_feature(image.shape[1]),
        'depth': _int64_feature(image.shape[2]),
        'raw_image': _bytes_feature(serialize_array(image)),
        'l_height': _int64_feature(label.shape[0]),
        'l_width': _int64_feature(label.shape[1]),
        'l_depth': _int64_feature(label.shape[2]),
        'l_raw_image': _bytes_feature(serialize_array(label)),
        'token': _bytes_feature(serialize_array(token))
    }

    out = tf.train.Example(features=tf.train.Features(feature=data))
    return out


def parse_tfr_element(element):
    """
    Load TFR element/example in the specified format while writing.
    Parameters
    ----------
    element : TYPE
        DESCRIPTION.
    Returns
    -------
    feature : TYPE
        DESCRIPTION.
    label : TYPE
        DESCRIPTION.
    """
    data = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'raw_image': tf.io.FixedLenFeature([], tf.string),
        'depth': tf.io.FixedLenFeature([], tf.int64),
        'l_height': tf.io.FixedLenFeature([], tf.int64),
        'l_width': tf.io.FixedLenFeature([], tf.int64),
        'l_raw_image': tf.io.FixedLenFeature([], tf.string),
        'l_depth': tf.io.FixedLenFeature([], tf.int64),
        'token': tf.io.FixedLenFeature([], tf.string),
    }

    content = tf.io.parse_single_example(element, data)

    height = content['height']
    width = content['width']
    depth = content['depth']
    raw_image = content['raw_image']
    l_height = content['l_height']
    l_width = content['l_width']
    l_depth = content['l_depth']
    l_raw_image = content['l_raw_image']
    token = content['token']

    feature = tf.io.parse_tensor(raw_image, out_type=tf.uint8)
    feature = tf.reshape(feature, shape=[height, width, depth])
    label = tf.io.parse_tensor(l_raw_image, out_type=tf.uint8)
    label = tf.reshape(label, shape=[l_height, l_width, l_depth])
    token = tf.io.parse_tensor(token, out_type=tf.string)
    return (feature, label, token)


def read_image(image_path):
    """
    loading the image from the given path and then splitting the image
    into input and output image. Just another preprocessing step for the dataset.
    Parameters
    ----------
    image_path : TYPE
        DESCRIPTION.
    Returns
    -------
    image : TYPE
        DESCRIPTION.
    label : TYPE
        DESCRIPTION.
    """
    image = Image.open(image_path)
    image = image.resize((336, 336))
    image = np.asarray(image)
    import matplotlib.pyplot as plt
    
    token = image_path.split("/")[-1].replace("_input.png","")
    target_name = image_path.split("/")[-1].split("_")[0] + str("_target.png")
    Y = os.path.join(os.path.dirname(os.path.dirname(image_path)), "Y")
    label_path = os.path.join(Y, os.path.basename(target_name))
    label = Image.open(label_path)
    label = label.resize((336,336))
    label = np.asarray(label)    
    label = label.reshape(336, 336, 1)
    return image, label, token


def write_images_to_tfr_short(store_dir, image_paths, filename: str = "train"):
    """
    To write the image_path images to tf records.
    Parameters
    ----------
    store_dir : TYPE
        DESCRIPTION.
    image_paths : TYPE
        DESCRIPTION.
    filename : str, optional
        DESCRIPTION. The default is "train".
    Returns
    -------
    count : TYPE
        DESCRIPTION.
    """
    filename = filename+".tfrecords"
    writer = tf.io.TFRecordWriter(os.path.join(
        store_dir, filename))
    count = 0

    for image_path in tqdm(image_paths):
        try:
            current_image, current_label, token = read_image(image_path)
            out = parse_single_image(image=current_image, label=current_label, token=token)
            writer.write(out.SerializeToString())
            count += 1
        except:
            print("Faced an exception")
            pass
            

    writer.close()
    print(f"Wrote {count} elements to TFRecord")
    return count


def generate_tfrecords(parameters):
    """
    To generate tf records from the given parameters dictionary.
    Parameters
    ----------
    parameters : TYPE
        DESCRIPTION.
    Returns
    -------
    None.
    """

    train_paths = list(paths.list_images(parameters['dataset']['data_dir']['train']))
    val_paths = list(paths.list_images(parameters['dataset']['data_dir']['val']))

    count_train=write_images_to_tfr_short(
        parameters['dataset']['data_dir']['train'], train_paths, filename="train")
    count_val=write_images_to_tfr_short(
        parameters['dataset']['data_dir']['val'], val_paths, filename='val')
    print("Number of images in Train:", count_train)
    print("Number of images in Val:", count_val)
    return


def load_tfrecords(parameters):
    """
    Load tf records from the given parameters dictionary
    Parameters
    ----------
    parameters : TYPE
        DESCRIPTION.
    Returns
    -------
    train_dataset : TYPE
        DESCRIPTION.
    val_dataset : TYPE
        DESCRIPTION.
    """
    train_dataset=tf.data.TFRecordDataset(os.path.join(
        parameters['dataset']['data_dir']['train'], 'train.tfrecords'))
    val_dataset=tf.data.TFRecordDataset(os.path.join(
        parameters['dataset']['data_dir']['val'], 'val.tfrecords'))

    train_dataset=train_dataset.map(parse_tfr_element)
    train_dataset=train_dataset.map(lambda x, y, z: (
        tf.cast(x, tf.float32), tf.cast(y, tf.int64), z))
    train_dataset=train_dataset.map(preprocess_data)

    val_dataset=val_dataset.map(parse_tfr_element)
    val_dataset=val_dataset.map(lambda x, y, z: (
        tf.cast(x, tf.float32), tf.cast(y, tf.int64), z))

    return (train_dataset, val_dataset)