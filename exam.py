import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode

# Create the configurations.
def setup_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    
    return cfg


class Predict():
    """
    Class that predicts instance segmentation on a single frame or a continous stream of video.
    """
    def __init__(self):
        self.cam = cv2.VideoCapture(0)
        self.cfg = setup_cfg()
        self.predictor = DefaultPredictor(self.cfg)
        self.video_v = VideoVisualizer(MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), ColorMode.IMAGE) # Visualizer for video inference.
        self.video_name = 'exam.mp4'
        self.fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        self.writer = cv2.VideoWriter(self.video_name, self.fourcc, 5, (640,480)) # Create the video writer.

    # Predicts on a single frame.
    def predictFrame(self):
        ret, frame = self.cam.read() # Read frame from the webcam.
        
        if ret:
            outputs = self.predictor(frame[..., ::-1]) # Predict on the image.
            
            image_v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2) # Create the visualization.

            out = image_v.draw_instance_predictions(outputs["instances"].to("cpu")) # Draw the visualization on the frame.
            out = out.get_image()[..., ::-1][..., ::-1]
            
            # Make sure the frame is colored            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            return frame, out
        else:
            return None

    # Predict on multiple frames.
    def predictVideo(self):
        ret, frame = self.cam.read() # Read frame from the webcam.

        if ret:
            outputs = self.predictor(frame[..., ::-1]) # Predict on the image.

            out = self.video_v.draw_instance_predictions(frame, outputs["instances"].to("cpu")) # Draw the visualization on the frame.

            # Make sure the frame is colored
            out = cv2.cvtColor(out.get_image(), cv2.COLOR_BGR2RGB)
            
            # Save frame to video
            self.writer.write(cv2.cvtColor(out, cv2.COLOR_RGB2BGR))

            return out
        else:
            return None

    # Plot a single frame.
    def plotFrame(self, inputFrame, outputFrame):

        fig = plt.figure(figsize=(20, 17))

        fig.add_subplot(1, 2, 1)

        # showing image
        plt.imshow(inputFrame)
        plt.axis('off')
        plt.title("Frame")

        fig.add_subplot(1, 2, 2)

        # showing image
        plt.imshow(outputFrame)
        plt.axis('off')
        plt.title("Frame with Segmentation")

        
def main():
    segmentPredictor = Predict() # Instantiate an object.

    #create two subplots
    ax1 = plt.subplot(1,1,1)

    #create two image plots
    im1 = ax1.imshow(segmentPredictor.predictVideo())

    # define the function to update the matplotlib animation.
    def update(i):
        im1.set_data(segmentPredictor.predictVideo())

    # Create the matplotlib animation.
    ani = FuncAnimation(plt.gcf(), update, interval=1)
    plt.show()

    # Release opencv.
    segmentPredictor.cam.release()
    segmentPredictor.writer.release()
    
    
if __name__ == "__main__":
    main()