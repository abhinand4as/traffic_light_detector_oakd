#!/usr/bin/env python3

from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np
import time

from gpiozero import LED
from time import sleep

'''
Spatial Tiny-yolo example
  Performs inference on RGB camera and retrieves spatial location coordinates: x,y,z relative to the center of depth map.
  Can be used for tiny-yolo-v3 or tiny-yolo-v4 networks
'''

red = LED(4)
green = LED(3)
yellow = LED(14)

#bMotor = LED(4)



# tiny yolo v3/4 label texts
labelMap = ["person",         "bicycle",    "car",           "motorbike",     "aeroplane",   "bus",           "train",
             "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",   "parking meter", "bench",
             "bird",           "cat",        "dog",           "horse",         "sheep",       "cow",           "elephant",
             "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",    "handbag",       "tie",
             "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball", "kite",          "baseball bat",
             "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",      "wine glass",    "cup",
             "fork",           "knife",      "spoon",         "bowl",          "banana",      "apple",         "sandwich",
             "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",       "donut",         "cake",
             "chair",          "sofa",       "pottedplant",   "bed",           "diningtable", "toilet",        "tvmonitor",
             "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",  "microwave",     "oven",
             "toaster",        "sink",       "refrigerator",  "book",          "clock",       "vase",          "scissors",
             "teddy bear",     "hair drier", "toothbrush"]  

syncNN = True

# Get argument first
nnBlobPath = str((Path(__file__).parent / Path('models/mobilenet.blob')).resolve().absolute())
if len(sys.argv) > 1:
    nnBlobPath = sys.argv[1]

# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a source - color camera
colorCam = pipeline.createColorCamera()
spatialDetectionNetwork = pipeline.createYoloSpatialDetectionNetwork()
monoLeft = pipeline.createMonoCamera()
monoRight = pipeline.createMonoCamera()
stereo = pipeline.createStereoDepth()

xoutRgb = pipeline.createXLinkOut()
xoutNN = pipeline.createXLinkOut()
xoutBoundingBoxDepthMapping = pipeline.createXLinkOut()
xoutDepth = pipeline.createXLinkOut()

xoutRgb.setStreamName("rgb")
xoutNN.setStreamName("detections")
xoutBoundingBoxDepthMapping.setStreamName("boundingBoxDepthMapping")
xoutDepth.setStreamName("depth")


colorCam.setPreviewSize(416, 416)
colorCam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
colorCam.setInterleaved(False)
colorCam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# setting node configs
stereo.setOutputDepth(True)
stereo.setConfidenceThreshold(255)

spatialDetectionNetwork.setBlobPath(nnBlobPath)
spatialDetectionNetwork.setConfidenceThreshold(0.5)
spatialDetectionNetwork.input.setBlocking(False)
spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
spatialDetectionNetwork.setDepthLowerThreshold(100)
spatialDetectionNetwork.setDepthUpperThreshold(5000)
# yolo specific parameters
spatialDetectionNetwork.setNumClasses(80)
spatialDetectionNetwork.setCoordinateSize(4)
spatialDetectionNetwork.setAnchors(np.array([10,14, 23,27, 37,58, 81,82, 135,169, 344,319]))
spatialDetectionNetwork.setAnchorMasks({ "side26": np.array([1,2,3]), "side13": np.array([3,4,5]) })
spatialDetectionNetwork.setIouThreshold(0.5)

# Create outputs

monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

colorCam.preview.link(spatialDetectionNetwork.input)
if(syncNN):
    spatialDetectionNetwork.passthrough.link(xoutRgb.input)
else:
    colorCam.preview.link(xoutRgb.input)

spatialDetectionNetwork.out.link(xoutNN.input)
spatialDetectionNetwork.boundingBoxMapping.link(xoutBoundingBoxDepthMapping.input)

stereo.depth.link(spatialDetectionNetwork.inputDepth)
spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)

# Pipeline defined, now the device is connected to
with dai.Device(pipeline) as device:
    # Start pipeline
    device.startPipeline()

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
    xoutBoundingBoxDepthMapping = device.getOutputQueue(name="boundingBoxDepthMapping", maxSize=4, blocking=False)
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    frame = None
    detections = []

    start_time = time.monotonic()
    counter = 0
    fps = 0
    color = (255, 255, 255)

    ###################################################
    red_lower = np.array([0, 150, 150], np.uint8) 
    red_upper = np.array([10, 255, 255], np.uint8) 

    #green_lower = np.array([74, 143, 171], np.uint8) 
    #green_upper = np.array([94, 163, 251], np.uint8)
    
    green_lower = np.array([50, 100, 100], np.uint8) 
    green_upper = np.array([70, 255, 255], np.uint8)

    #yellow_lower = np.array([26, 187, 174], np.uint8) 
    #yellow_upper = np.array([46, 207, 254], np.uint8)
    
    #yellow_lower = np.array([26, 160, 162], np.uint8) 
    #yellow_upper = np.array([46, 180, 242], np.uint8)
    
    yellow_lower = np.array([25, 200, 103], np.uint8) 
    yellow_upper = np.array([45, 220, 183], np.uint8)
    
    ###################################################

    while True:

        red.off()
        green.off()
        yellow.off()
        #bMotor.on()
        inPreview = previewQueue.get()
        inNN = detectionNNQueue.get()
        depth = depthQueue.get()

        counter+=1
        current_time = time.monotonic()
        if (current_time - start_time) > 1 :
            fps = counter / (current_time - start_time)
            counter = 0
            start_time = current_time
        
        frame = inPreview.getCvFrame()
        depthFrame = depth.getFrame()

        depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        depthFrameColor = cv2.equalizeHist(depthFrameColor)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)
        detections = inNN.detections
        if len(detections) != 0:
            boundingBoxMapping = xoutBoundingBoxDepthMapping.get()
            roiDatas = boundingBoxMapping.getConfigData()

            for roiData in roiDatas:
                roi = roiData.roi
                roi = roi.denormalize(depthFrameColor.shape[1], depthFrameColor.shape[0])
                topLeft = roi.topLeft()
                bottomRight = roi.bottomRight()
                xmin = int(topLeft.x)
                ymin = int(topLeft.y)
                xmax = int(bottomRight.x)
                ymax = int(bottomRight.y)

                cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)


        # if the frame is available, draw bounding boxes on it and show the frame
        height = frame.shape[0]
        width  = frame.shape[1]

        ###################################################
        hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        kernal = np.ones((5, 5), "uint8") 
        imCrop = frame
        ###################################################


        for detection in detections:
            # denormalize bounding box
            x1 = int(detection.xmin * width)
            x2 = int(detection.xmax * width)
            y1 = int(detection.ymin * height)
            y2 = int(detection.ymax * height)
            try:
                label = labelMap[detection.label]
            except:
                label = detection.label

            #######################################################################
            if (label == "traffic light"):
                imCrop = hsvFrame[y1:y2, x1:x2]

                red_mask = cv2.inRange(imCrop, red_lower, red_upper)
                green_mask = cv2.inRange(imCrop, green_lower, green_upper) 
                yellow_mask = cv2.inRange(imCrop, yellow_lower, yellow_upper) 

                #  # For red color 
                red_mask = cv2.dilate(red_mask, kernal) 
                res_red = cv2.bitwise_and(imCrop, imCrop,  
                                        mask = red_mask) 
                
                # For green color 
                green_mask = cv2.dilate(green_mask, kernal) 
                res_green = cv2.bitwise_and(imCrop, imCrop, 
                                            mask = green_mask) 
                # For yellow color 
                yellow_mask = cv2.dilate(yellow_mask, kernal) 
                res_yellow = cv2.bitwise_and(imCrop, imCrop, 
                                            mask = yellow_mask) 
                
                # Creating contour to track red color 
                contours, hierarchy = cv2.findContours(red_mask, 
                                                    cv2.RETR_TREE, 
                                                    cv2.CHAIN_APPROX_SIMPLE) 
                
                for pic, contour in enumerate(contours): 
                    area = cv2.contourArea(contour) 
                    if(area > 30): 
                        x_, y_, w, h = cv2.boundingRect(contour) 
                        # img = cv2.rectangle(imCrop, (x_, y_),  
                        #                         (x_ + w, y_ + h),  
                        #                         (0, 0, 255), 2) 
                        
                        cv2.putText(frame, "Red", (x1, y1-20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                                    (0, 0, 255), 2)     
                        print("Traffic Light: RED")
                        
                        #VIBRATION
                        red.on()
                        green.on()
                        yellow.on()
                        sleep(0.1)

                # Creating contour to track green color 
                contours, hierarchy = cv2.findContours(green_mask, 
                                                    cv2.RETR_TREE, 
                                                    cv2.CHAIN_APPROX_SIMPLE) 
                
                for pic, contour in enumerate(contours): 
                    area = cv2.contourArea(contour) 
                    if(area > 3): 
                        x_, y_, w, h = cv2.boundingRect(contour) 
                        # img = cv2.rectangle(imCrop, (x_, y_),  
                        #                         (x_ + w, y_ + h), 
                        #                         (0, 255, 0), 2) 
                        
                        cv2.putText(frame, "Green", (x1, y1-20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                                    (0, 255, 0), 2)  
                        print("Traffic Light: GREEN")

                        #VIBRATION
                        green.off()
                        green.on()
                        sleep(.1)


                # Creating contour to track yellow color 
                contours, hierarchy = cv2.findContours(yellow_mask, 
                                                    cv2.RETR_TREE, 
                                                    cv2.CHAIN_APPROX_SIMPLE) 
                
                for pic, contour in enumerate(contours): 
                    area = cv2.contourArea(contour) 
                    if(area > 3): 
                        x_, y_, w, h = cv2.boundingRect(contour) 
                        # img = cv2.rectangle(imCrop, (x_, y_),  
                        #                         (x_ + w, y_ + h), 
                        #                         (0, 255, 0), 2) 
                        
                        cv2.putText(frame, "Yellow", (x1+40, y1-20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                                    (0, 255, 255), 2)  
                        print("Traffic Light: YELLOW")

                        #VIBRATION
                        yellow.on()
                        green.on()
                        sleep(0.05)
                        
                #######################################################################
            cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.putText(frame, "{:.2f}".format(detection.confidence*100), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.putText(frame, f"X: {int(detection.spatialCoordinates.x)} mm", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.putText(frame, f"Y: {int(detection.spatialCoordinates.y)} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.putText(frame, f"Z: {int(detection.spatialCoordinates.z)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

        cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)
        #cv2.imshow("depth", depthFrameColor)
        cv2.imshow("rgb", frame)

        if cv2.waitKey(1) == ord('q'):
            break

