import cv2 as cv2
import os
from Stabilization import *
from BackgroundSubtraction import *
from Matting import *
from Tracking import *


####################
### Chose Video  ###
####################
VideoName = 'INPUT.avi'
NewBackground = 'background.png'

INPUT_DIR_PATH = './INPUT/'

OutputVideo_DIR_PATH = './OUTPUT/'
OutputPlots_DIR_PATH = './OUTPUT/'

if not os.path.exists(OutputVideo_DIR_PATH):
    os.makedirs(OutputVideo_DIR_PATH)

if not os.path.exists(OutputPlots_DIR_PATH):
    os.makedirs(OutputPlots_DIR_PATH)

####################
### Parameters  ###
####################
frameStart = 180    # Number of frames to crop from beginning
frameCropEnd = 0    # Number of frames to crop from the end of the video

####################
###     MAIN     ###
####################
#### Get Video and cropping of new background ####
inVid = cv2.VideoCapture(INPUT_DIR_PATH + VideoName)
wallpaper = cv2.imread(INPUT_DIR_PATH+NewBackground)
success, frame1 = inVid.read()
if success:
    VidSize = frame1.shape
    Resized_NewBackground = cv2.resize(wallpaper, (VidSize[1],VidSize[0]))
else:
    print("Invalid input video")

#### Stabilization ####
inVid = cv2.VideoCapture(INPUT_DIR_PATH + VideoName)
b = VidStabilization(inVid, SMOOTHING_RADIUS = 50, INLINERS_PERCENT=0.2, MAX_ERROR=1,Output_DIR_PATH= OutputVideo_DIR_PATH)

#### Binary Video and Trimap ####

StabilizedVid = cv2.VideoCapture(OutputVideo_DIR_PATH + 'stabilized.avi')
b = Binary(StabilizedVid, MEDIAN_SIZE=10, OutputVideo_DIR_PATH=OutputVideo_DIR_PATH, frameStart =frameStart, frameCropEnd=frameCropEnd)

StabilizedVid = cv2.VideoCapture(OutputVideo_DIR_PATH + 'stabilized.avi')
BinaryVid = cv2.VideoCapture(OutputVideo_DIR_PATH + 'binary.avi')
extraction(StabilizedVid, BinaryVid,OutputVideo_DIR_PATH=OutputVideo_DIR_PATH, frameStart =frameStart)


## Changing Background ####

# Likelihood functions based on several frames
StabilizedVid = cv2.VideoCapture(OutputVideo_DIR_PATH + 'stabilized.avi')
BinaryVid = cv2.VideoCapture(OutputVideo_DIR_PATH + 'binary.avi')
FGLikelihood, BGLikelihood, x_grid = FG_BG_likelihood_all_vid(StabilizedVid, BinaryVid,frameStart=frameStart, frameInterval=20)

# Matting on new background
StabilizedVid = cv2.VideoCapture(OutputVideo_DIR_PATH + 'stabilized.avi ')
BinaryVid = cv2.VideoCapture(OutputVideo_DIR_PATH + 'binary.avi')
ChangeVideoBackground(StabilizedVid, BinaryVid, Resized_NewBackground, FGLikelihood, BGLikelihood, OutputVideo_DIR_PATH= OutputVideo_DIR_PATH,frameStart =frameStart, frameCropEnd=frameCropEnd)


#### Tracking ####
MattedVideo = cv2.VideoCapture(OutputVideo_DIR_PATH + 'matted.avi')
VideoParameters = getVideoParameters(MattedVideo)
Tracking(MattedVideo, VideoParameters, Output_DIR_PATH= OutputVideo_DIR_PATH, showFrame1=True)
