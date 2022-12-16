import os
import yaml

import scipy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from models.unet import Generator
from data.bev_dataset import generate_tfrecords, load_tfrecords
from utils.loss import custom_sparse_categorical_crossentropy
from tqdm import tqdm
from datetime import datetime


print(tf.test.is_gpu_available())

def get_dataset(parameters):
    """
    Load or generate a dataset and apply batching and shuffling.
    
    Parameters:
        parameters (dict): A dictionary of parameters, including the paths 
        to the training and validation data, the batch size, and other relevant
        information.
    
    Returns:
        tuple: A tuple containing the training dataset and validation dataset.
    """
    # Check if the TF records for the training and validation datasets exist. 
    # If not, generate them.
    if not os.path.exists(os.path.join(
            parameters['dataset']['data_dir']['train'],
            'train.tfrecords')) or not os.path.exists(os.path.join(
                parameters['dataset']['data_dir']['val'], 'val.tfrecords')):
        print("Generating TF Records:")
        generate_tfrecords(parameters)
    else:
        print("Using existing TF Records")
        
    # Load the training and validation datasets from the TF records
    train_dataset, val_dataset = load_tfrecords(parameters)
    
    # Apply batching and shuffling to the training dataset
    train_dataset = train_dataset.batch(
        parameters['dataset']['batch_size']).shuffle(buffer_size=100)
    
    # Apply batching to the validation dataset
    val_dataset = val_dataset.batch(parameters['dataset']['batch_size'])
    
    # Return the training and validation datasets
    return train_dataset, val_dataset


def plottable_prediction(prediction):
    """
    Process a model prediction to make it more suitable for plotting.
    
    Parameters:
        prediction (numpy array): The prediction from a model, with shape 
        (batch_size, height, width, num_classes).
    
    Returns:
        numpy array: The processed prediction, with shape (height, width, 3).
    """
    
    # Taking one of the sample as prediction
    prediction = prediction[0]
    
    # Applying softmax on classes of the prediction
    prediction = scipy.special.softmax(prediction, axis=-1)
    
    # Takes the foreground class and converts the prediction array to 3 channels. 
    prediction = np.repeat(prediction[:,:,1][..., None], 3, axis=2)
    
    # Threshold the prediction for binary image and rescale.
    prediction = (prediction > 0.5) * 255.0
    
    return prediction 
    

def plot_images(images, title="Train Prediction",grid_shape=(1,3), scale_factor=30):
    """
    Process a model prediction to make it more suitable for plotting.
    
    Parameters:
        prediction (numpy array): The prediction from a model, with shape (batch_size, height, width, num_classes).
    
    Returns:
        numpy array: The processed prediction, with shape (height, width, 3).
    """
    
    axtitle = ["Input", "Label"]
    axtitle.append(title)
    figsize = (grid_shape[0] * scale_factor, grid_shape[1] * scale_factor)
    _, axes = plt.subplots(grid_shape[0], grid_shape[1], figsize=figsize)
    
    for ax, image in zip(axes.flatten(), images):
        if image.shape[-1] == 1:
            ax.imshow(image, cmap="gray")
        else:
            ax.imshow(image, cmap="gray")
        ax.axis("off")
    plt.show()
  

# Write documentation for the following function:
def plot_sample_outputs(net, train_dataset, val_dataset):
    """
    Plots the model's output on a sample of data from the training and validation datasets. 
    The function takes three arguments:

    net: a model object
    train_dataset: a tf.data.Dataset object that contains the training data
    val_dataset: a tf.data.Dataset object that contains the validation data
    It iterates over both the training and validation datasets and selects a sample from each.
    It then processes the sample data and appends the resulting images to the 'image_list' list. 
    Finally, it plots the images using the 'plot_images' function.
    """

    for phase, dataset in enumerate([train_dataset, val_dataset]):
        image_list = []
   
        for data in dataset.take(1):
            imgs = [data[0][0], data[1][0]]

            for i, img in enumerate(imgs):
                if i == 1:
                    img = np.repeat(img, 3, 2)
                    img = img > 0.0
                    img = img.astype(np.float32)*255.0
                    image_list.append(img)
                else:
                    image_list.append(img)
           
            prediction = plottable_prediction(net(data[0]).numpy())
            image_list.append(prediction)
            
        if phase == 1:    
            plot_images(image_list, "Val Prediction")
        else:
            plot_images(image_list)
        
        for i, img in enumerate(image_list):
            print(i, " min: ",np.min(img), " max: ", np.max(img), " mean ", np.mean(img))
            

def train(train_dataset, val_dataset, epochs, model, optimizer, class_weights_dict, checkpoint_dir):
    """
    The train function trains a model on a given dataset for a specified number of epochs. 
    It has several parameters:
    train_dataset: a TensorFlow dataset object representing the training dataset
    val_dataset: a TensorFlow dataset object representing the validation dataset
    epochs: an integer specifying the number of epochs to train the model for
    model: a TensorFlow model object to be trained
    optimizer: a TensorFlow optimizer object to be used for training
    class_weights_dict: a dictionary specifying class weights to be used in the metrics function
    checkpoint_dir: a string specifying the directory to save checkpoints of the model during training
    
    """
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy() 
    class_weights = np.array([0.0] + [1.0]*9, dtype=np.float32)
    for epoch in range(1, epochs+1):
        print("\nStarting Epoch No: ", epoch)
        
        # Train phase
        count = 0
        for inputs, targets, token in tqdm(train_dataset):
            with tf.GradientTape() as tape:
                predictions = model(inputs)
                loss = custom_sparse_categorical_crossentropy(targets, predictions, class_weights_dict)
                accuracy.update_state(targets, predictions, sample_weight = class_weights)
            
            count += 1
            if count % 10 == 0:     
                print(" loss: ",loss.numpy(), " Accuracy: ", accuracy.result().numpy())
                plot_sample_outputs(model, train_dataset, train_dataset)

            # Compute gradients
            gradients = tape.gradient(loss, model.trainable_variables)

            # Apply gradients
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
        print("Train Accuracy: ", accuracy.result().numpy())
        accuracy.reset_state()

        # Validation phase
        for inputs, targets, token in val_dataset:
            predictions = model(inputs)
            loss = custom_sparse_categorical_crossentropy(targets, predictions, class_weights_dict)
            accuracy.update_state(targets, predictions, sample_weight = class_weights)
        
        print("Validation Accuracy: ", accuracy.result().numpy())
        accuracy.reset_state()
        
        if epoch % 10 == 0: 
            print("Saving Model:", epoch)
            model.save(os.path.join("./checkpoints/", checkpoint_dir, "epoch_"+str(epoch)))
            




if __name__ == "__main__":
    with open('parameters/train.yaml', 'r') as file:
        parameters = yaml.safe_load(file)
    # Creating/Loading TF Records data
    train_dataset, val_dataset = get_dataset(parameters)

    net = Generator(input_shape=(336, 336, 3)).get_model()
    net = tf.keras.models.load_model('checkpoints/2022-12-15_00-25-44/epoch_70')

    optimizer = tf.keras.optimizers.Adam(
        lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    
    classes = ["car", "motorcycle", "bus", "bicycle", "truck", "pedestrian", "other_vehicle", "animal", "emergency_vehicle"]
    class_weights = np.array([0.2] + [1.0]*len(classes), dtype=np.float32)
    class_weight_dict = {}
    for i, cls in enumerate(classes):
        class_weight_dict[i] = class_weights[i]
        
    checkpoint_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    train(train_dataset, val_dataset, 300, net, optimizer, class_weight_dict, checkpoint_dir)
    net.save("./res/custom_train.h5")