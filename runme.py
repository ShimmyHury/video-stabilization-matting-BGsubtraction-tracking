import cv2 as cv2
from Stabilization import *
from BackgroundSubtraction import *
from Matting import *
from Tracking import *
import os
###############
#### RUNME ####
###############

####################
### Chose Video  ###
####################
VideoName = 'INPUT.avi'
NewBackground = 'background.png'

INPUT_DIR_PATH = './INPUT/'
Output_DIR_PATH = './OUTPUT/'

if not os.path.exists(Output_DIR_PATH):
    os.makedirs(Output_DIR_PATH)

###################
### Parameters  ###
###################
### These parameters let you define how much you want
### to crop from the begining and the end of the video
### (KNN algorithm takes a few frames before giving good performances)
frameStart = 180    # Number of frames to crop from beginning
frameCropEnd = 0    # Number of frames to crop from the end of the video


########################
### RUNME functions  ###
########################
def runme_Stabilization(VideoName, INPUT_DIR_PATH):
    inVid = cv2.VideoCapture(INPUT_DIR_PATH + VideoName)
    b = VidStabilization(inVid, SMOOTHING_RADIUS=50, INLINERS_PERCENT=0.2, MAX_ERROR=1,
                         Output_DIR_PATH=Output_DIR_PATH)
    return

def runme_BackgroundSubtraction(Output_DIR_PATH):
    StabilizedVid = cv2.VideoCapture(Output_DIR_PATH + 'stabilized.avi')
    b = Binary(StabilizedVid, MEDIAN_SIZE=10, OutputVideo_DIR_PATH=Output_DIR_PATH, frameStart=frameStart,
               frameCropEnd=frameCropEnd)

    StabilizedVid = cv2.VideoCapture(Output_DIR_PATH + 'stabilized.avi')
    BinaryVid = cv2.VideoCapture(Output_DIR_PATH + 'binary.avi')
    extraction(StabilizedVid, BinaryVid, OutputVideo_DIR_PATH=Output_DIR_PATH, frameStart=frameStart)
    return

def runme_Matting(INPUT_DIR_PATH, VideoName, Output_DIR_PATH):
    #### Get Video and cropping of new background ####
    inVid = cv2.VideoCapture(INPUT_DIR_PATH + VideoName)
    wallpaper = cv2.imread(INPUT_DIR_PATH + NewBackground)
    success, frame1 = inVid.read()
    if success:
        VidSize = frame1.shape
        Resized_NewBackground = cv2.resize(wallpaper, (VidSize[1], VidSize[0]))
    else:
        print("Invalid input video")

    # Likelihood functions based on several frames
    StabilizedVid = cv2.VideoCapture(Output_DIR_PATH + 'stabilized.avi')
    BinaryVid = cv2.VideoCapture(Output_DIR_PATH + 'binary.avi')
    FGLikelihood, BGLikelihood, x_grid = FG_BG_likelihood_all_vid(StabilizedVid, BinaryVid, frameStart=frameStart,
                                                                  frameInterval=20)

    # Matting on new background
    StabilizedVid = cv2.VideoCapture(Output_DIR_PATH + 'stabilized.avi ')
    BinaryVid = cv2.VideoCapture(Output_DIR_PATH + 'binary.avi')
    ChangeVideoBackground(StabilizedVid, BinaryVid, Resized_NewBackground, FGLikelihood, BGLikelihood,
                          OutputVideo_DIR_PATH=Output_DIR_PATH, frameStart=frameStart, frameCropEnd=frameCropEnd)
    return

def runme_Tracking(Output_DIR_PATH):
    MattedVideo = cv2.VideoCapture(Output_DIR_PATH + 'matted.avi')
    VideoParameters = getVideoParameters(MattedVideo)
    Tracking(MattedVideo, VideoParameters, Output_DIR_PATH=Output_DIR_PATH, showFrame1=True)
    return

###################
###    RUNME    ###
###################
###
### Chose which action you want to test by uncommenting it / comment those you don't want to run.
###

### Stabilization => Stabilized video will be saved in Output_DIR_PATH

# runme_Stabilization(VideoName, INPUT_DIR_PATH)

### Background subtraction => Binary video and extracted video will be saved in Output_DIR_PATH as binary.avi and extracted.avi

# runme_BackgroundSubtraction(Output_DIR_PATH)

### Matting => Matted video will be saved in Output_DIR_PATH as matted.avi

runme_Matting(INPUT_DIR_PATH, VideoName, Output_DIR_PATH)

### Tracking => Tracked video will be saved in Output_DIR_PATH as OUTPUT.avi

# runme_Tracking(Output_DIR_PATH)

