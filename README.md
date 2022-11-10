# Applied-AI-Exam

## Description:
Applied AI instance segmentation exam with Detectron2.  <br /><br />

## included Items:
  - [exam.py](exam.py):
    - Includes the code.
  - [exam.mp4](media/exam.mp4):
    - Includes the video. <br /><br />

## Steps to Achieve Results:  
1. The first step was installing Detectron2. <br />
2. The second step was choosing the congiurations. 
  - The "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml" configuration file and weights were used for instance segmentation. The confidence threshold     chosen was 0.5. <br />
3. The third step was getting the model to predict on a single image.
  - A class called Predict was made to encapsulate the functions to preidct and plot the the frame.
  - The frame was then passed into a Dectron2 predictor and visualizer function to produce the visualization. 
  - The frame was then blended with the preidcted mask.
  - Lastly the frame was plotted with matplotlib.  <br />
4. Next, the predictor was implemented using a constant steam from the webcam.
  - The prediction for the live webcam stream is similar to predicting on a single image.
  - The webcam was opened with VideoCapture and the frame was passed to the predictor.
  - The only difference between a signle image and the webcam is matplotlib animation function needed to be used to have a video steam plotted with       
    matplotlib.
  - The exam.mp4 file includes a video of the result.
