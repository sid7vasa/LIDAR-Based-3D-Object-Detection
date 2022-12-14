import os
import yaml

import scipy
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from models.unet import Generator
from data.bev_dataset import generate_tfrecords, load_tfrecords
from utils.loss import custom_sparse_categorical_crossentropy
from tqdm import tqdm

print(tf.test.is_gpu_available())


def get_dataset(parameters):
    """
    Loads TF records (If not exists, creates TF Records).
    Preprocesses datset to the needed format.
    returns train and validation dataset tf.data instances.
    Parameters
    ----------
    parameters : parameters loaded from the parameters.yaml file.
        DESCRIPTION.
    Returns
    -------
    train_dataset : tf.data train dataset instance
        DESCRIPTION.
    val_dataset : tf.data validation dataset instance
        DESCRIPTION.
    """
    if not os.path.exists(os.path.join(
            parameters['dataset']['data_dir']['train'],
            'train.tfrecords')) or not os.path.exists(os.path.join(
                parameters['dataset']['data_dir']['val'], 'val.tfrecords')):
        print("Generating TF Records:")
        generate_tfrecords(parameters)
    else:
        print("Using existing TF Records")
    train_dataset, val_dataset = load_tfrecords(parameters)
    train_dataset = train_dataset.batch(
        parameters['dataset']['batch_size']).shuffle(buffer_size=100)
    val_dataset = val_dataset.batch(parameters['dataset']['batch_size'])
    return train_dataset, val_dataset


def visualize_datasets(train_dataset):
    """
    Visualize an example in the dataset by reversing the preprocessing steps.
    using matplotlib.
    Parameters
    ----------
    train_dataset : tf.data training instance
        DESCRIPTION.
    Returns
    -------
    None.
    """
    for data in train_dataset.take(1):
        print(data[0].shape)
        print(data[1].shape)
        picture = data[1].numpy()[0]
        picture = (picture*127.5) + 127.5
        picture = np.array(picture, dtype=np.uint8)
        plt.imshow(picture)
        plt.show()

def plottable_prediction(prediction):
    prediction = prediction[0]
    prediction = scipy.special.softmax(prediction, axis=-1)
    prediction = np.repeat(prediction[:,:,0][..., None], 3, axis=2)
    prediction = 1 - (prediction > 0.5)
    print("Prediction:", prediction.shape)
    return prediction * 255.0
    


def plot_sample_outputs(net, dataset, val=False):
    """
    Takes random examples from the input tf.data instance and then plots the 
    generated output, corresponding inputs and ground truths.
    Parameters
    ----------
    dataset : tf.data validation instance
        DESCRIPTION.
    val : is validation dataset instance
    Returns
    -------
    img : TYPE
        DESCRIPTION.
    """
    
    plt.figure(figsize=(16,8))
    
    if val:
        dataset = dataset.shuffle(buffer_size=100)

    for data in dataset.take(1):
        prediction = 1- plottable_prediction(net(data[0]).numpy())
        plt.title(data[2].numpy()[0])
        imgs = [data[0][0], data[1][0]]
        image_list = [prediction]
        for i, img in enumerate(imgs):
            if i == 1:
                img = np.repeat(img, 3, 2)
                image_list.insert(1, img)
            else:
                image_list.insert(0, img)
        for i, img in enumerate(image_list):
            print(i, " min: ",np.min(img), " max: ", np.max(img), " mean ", np.mean(img))
        h_stack = np.hstack(image_list)
        plt.imshow(h_stack)
        plt.show()
        
def train(train_dataset, val_dataset, epochs, model, optimizer, class_weights_dict):
    
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    for epoch in range(epochs):
        # Train phase
        count = 0
        for inputs, targets, token in tqdm(train_dataset):
            with tf.GradientTape() as tape:
                predictions = model(inputs)
                loss = custom_sparse_categorical_crossentropy(targets, predictions, class_weights_dict)
            
            count += 1
            if count % 500 == 0:    
                print(" loss: ",loss.numpy())
                plot_sample_outputs(model, train_dataset)


            # Compute gradients
            gradients = tape.gradient(loss, model.trainable_variables)

            # Apply gradients
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))


        # Validation phase
        for inputs, targets, token in val_dataset:
            predictions = model(inputs)
            loss = loss_fn(targets, predictions)



if __name__ == "__main__":
    with open('parameters/train.yaml', 'r') as file:
        parameters = yaml.safe_load(file)

    # Creating/Loading TF Records data
    train_dataset, val_dataset = get_dataset(parameters)
    
    # visualize_datasets(train_dataset, val_dataset)

    net = Generator(input_shape=(336, 336, 3)).get_model()

    optimizer = tf.keras.optimizers.Adam(
        lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    # net.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    #             metrics=tf.keras.metrics.mean_absolute_error)

    
    # print(net.summary())
    classes = ["car", "motorcycle", "bus", "bicycle", "truck", "pedestrian", "other_vehicle", "animal", "emergency_vehicle"]
    class_weights = np.array([0.2] + [1.0]*len(classes), dtype=np.float32)
    class_weight_dict = {}
    for i, cls in enumerate(classes):
        class_weight_dict[i] = class_weights[i]
    
    train(train_dataset, val_dataset, 10, net, optimizer, class_weight_dict)
    net.save_weights("./res/custom_train.h5")