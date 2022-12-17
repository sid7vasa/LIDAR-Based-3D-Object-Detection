#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import yaml
from tqdm import tqdm
import tensorflow as tf
from train import plottable_prediction
import matplotlib.pyplot as plt
import scipy


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
    
def get_boxes(prediction_non_class0, prediction_conf):
    detection_boxes = []
    detection_scores = []
    detection_classes = []
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    
    prediction_opened = cv2.morphologyEx(prediction_non_class0, cv2.MORPH_OPEN, kernel).astype(np.uint8)
    class_probability = prediction_conf

    sample_boxes = []
    sample_detection_scores = []
    sample_detection_classes = []
    
    contours, hierarchy = cv2.findContours(prediction_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    
    
    classes = ["car", "motorcycle", "bus", "bicycle", "truck", "pedestrian", "other_vehicle", "animal", "emergency_vehicle"]
    
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        
        # Let's take the center pixel value as the confidence value
        box_center_index = np.int0(np.mean(box, axis=0))
        
        for class_index in range(len(classes)):
            box_center_value = class_probability[box_center_index[1], box_center_index[0], class_index+1]
            
            # Let's remove candidates with very low probability
            if box_center_value < 0.01:
                continue
            
            box_center_class = classes[class_index]

            box_detection_score = box_center_value
            sample_detection_classes.append(box_center_class)
            sample_detection_scores.append(box_detection_score)
            sample_boxes.append(box)
        
    
    detection_boxes.append(np.array(sample_boxes))
    detection_scores.append(sample_detection_scores)
    detection_classes.append(sample_detection_classes)   
    
    return detection_boxes, detection_scores, detection_classes, prediction_opened


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
        prediction, prediction_conf = plottable_prediction(prediction, confidence=True)
        prediction = prediction[:,:,0]
        
        detection_boxes, detection_scores, detection_classes, prediction_opened = get_boxes(prediction, prediction_conf)
        
        t = np.zeros_like(prediction_opened)
        box_pix = np.int0(detection_boxes[0])
        
        plt_image = (image[0,:,:,:] * 255.0).numpy()
        plt_image[:,:,2] = plt_image[:,:,2] * 0.1
        cv2.drawContours(t,box_pix,-1,(255,255,255),1)

        t = plt_image + np.repeat(t[..., None], 3, axis=2) 
        t = np.clip(t, 0, 255)
        plt.imshow(t)
        plt.show()
        
        # plt.hist(detection_scores[0], bins=20)
        # plt.xlabel("Detection Score")
        # plt.ylabel("Count")
        # plt.show()
                
        

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
    
    files = list(np.load("./playground/lidar_inputs.npy", allow_pickle=True))[50:]
    inference(net, files, video=False)
    
    # image_list = []
    # for i, file_path in enumerate(tqdm(files, desc="Loading Video: ")):
    #     image = cv2.imread(file_path) * 255.0
    #     image_list.append(image)
        
    #     if i == 2000:
    #         break
        
    # write_video("lidar_input.avi", image_list)
    

