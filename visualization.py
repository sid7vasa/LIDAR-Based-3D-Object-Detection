#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import yaml
from tqdm import tqdm
import tensorflow as tf
from train import plottable_prediction
import matplotlib.pyplot as plt


def write_video(filename, images):
    """
    Write a video file using a list of images.
    
    Parameters:
        filename (str): The name of the video file to create.
        images (list): A list of numpy arrays representing the images to include in the video.
    """
    
    frame_size = (336, 336)
    
    # Initialize the video writer
    out = cv2.VideoWriter(filename ,cv2.VideoWriter_fourcc(*'DIVX'), 10.0, frame_size)

    for image in tqdm(images, desc="Writing Video:"):
        
        # Convert the image to uint8 data type
        image = image.astype(np.uint8)
        
        # Write the image to the video file
        out.write(image)
    
    # Release the video writer
    out.release()

def inference(model, image_paths, video=False, filename="inference.avi"):
    """
    Run inference on a list of images using the specified model.
    
    Parameters:
        model (tf.keras.Model): The model to use for inference.
        image_paths (list): A list of strings representing the paths to 
        the images.
        video (bool, optional): Whether to create a video from the results.
        Defaults to False.
        filename (str, optional): The name of the video file to create. 
        Defaults to "inference.avi".
        
    """
    out_list = []
    for image_path in image_paths:
        # Read in the image and convert it to RGB format
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Add an extra dimension to the image, cast it to float32, and normalize it
        image = tf.expand_dims(image, axis=0)
        image = tf.cast(image, tf.float32)/255.0
        
        # Run inference on the image
        prediction = model.predict(image)
        

        # Process the prediction to make it more suitable for plotting
        prediction = plottable_prediction(prediction)
        
        # Create an overlay by blending the prediction with the original image
        overlay = prediction * 0.8 + 1.0 * image[0]
        
        # Add the overlay to the list of outputs
        out_list.append(np.clip(overlay.numpy(), 0.0, 255.0))
        
        # Display the overlay using Matplotlib
        plt.imshow(overlay)
        plt.grid(False)
        plt.axis('off')
        plt.show()
        
    # If the video argument is True, write a video using the list of outputs and the specified filename
    if video:
        write_video(filename, out_list)

if __name__ == "__main__":
    
    # Code comments here are uncommented according to the usage.
    
    # with open('parameters/train.yaml', 'r') as file:
    #     parameters = yaml.safe_load(file)
    # # Creating/Loading TF Records data
    # train_dataset, val_dataset = get_dataset(parameters)
    
    # # visualize_datasets(train_dataset, val_dataset)
    net = tf.keras.models.load_model('./checkpoints/2022-12-15_00-25-44/epoch_70')

    lidar_images_dir = "/mnt/e/LFS/3d-object-detection-for-autonomous-vehicles/bev_data/val_bev/X/"
    
    # files = list(filter(os.path.isfile, glob.glob(lidar_images_dir + "*")))
    # files.sort(key=lambda x: os.path.getmtime(x))
    
    # np.save("lidar_inputs.npy",np.array(files), allow_pickle=True)
    
    files = list(np.load("lidar_inputs.npy", allow_pickle=True))
    inference(net, files, video=False)
    
    # image_list = []
    # for i, file_path in enumerate(tqdm(files, desc="Loading Video: ")):
    #     image = cv2.imread(file_path) * 255.0
    #     image_list.append(image)
        
    #     if i == 2000:
    #         break
        
    # write_video("lidar_input.avi", image_list)
    

