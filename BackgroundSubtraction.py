import numpy as np
import cv2


##################################  Background Subtraction #########################################

def Binary(inVid, MEDIAN_SIZE=25,OutputVideo_DIR_PATH="",frameStart =50, frameCropEnd=0):
    '''

    :param inVid: input video
    :param MEDIAN_SIZE: size of spacial median filter to remoce salt and pepper noise
    :param OutputVideo_DIR_PATH: path for output
    :param frameStart: starting frame position for output video
    :param frameCropEnd: ending point position for the output video
    :return: binary video
    '''

    fgbg = cv2.createBackgroundSubtractorKNN(history=100,dist2Threshold=500,detectShadows= False)


    # Get video parameters
    n_frames = int(inVid.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(inVid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(inVid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = inVid.get(cv2.CAP_PROP_FPS)

    # Set up output video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    binaryVid = cv2.VideoWriter(OutputVideo_DIR_PATH+'binary.avi', fourcc, fps, (width, height), 0)
    for frameNum in range(n_frames):
        print("Binary: Frame: " + str(frameNum+1) + "/" + str(n_frames) +" processed")
        # Read next frame
        success, frame = inVid.read()
        if not success:
            break
        fgmask = fgbg.apply(frame)
        if (MEDIAN_SIZE % 2 == 0):
            MEDIAN_SIZE = MEDIAN_SIZE+1
        binaryFrame = cv2.medianBlur(fgmask, MEDIAN_SIZE)
        kernel = np.ones((3, 3), np.uint8)
        binaryFrame = cv2.morphologyEx(binaryFrame, cv2.MORPH_CLOSE, kernel, iterations=1)

        sobelx = cv2.Sobel(binaryFrame, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(binaryFrame, cv2.CV_64F, 0, 1, ksize=5)
        gradient = np.sqrt(np.multiply(sobelx, sobelx) + np.multiply(sobely, sobely))
        gradient = cv2.GaussianBlur(gradient, (9, 9), 0)
        binaryFrame = (binaryFrame - binaryFrame * ((gradient / (gradient.max() + 1e-17))>=0.1) )
        sobelx = cv2.Sobel(binaryFrame, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(binaryFrame, cv2.CV_64F, 0, 1, ksize=5)
        gradient = np.sqrt(np.multiply(sobelx, sobelx) + np.multiply(sobely, sobely))
        gradient = cv2.GaussianBlur(gradient, (9, 9), 0)

        binaryFrame = (binaryFrame + 255*((gradient/(gradient.max() + 1e-17))>=0.1 ).astype(np.uint8))
        binaryFrame = cv2.dilate(binaryFrame, kernel, iterations=3)

        if (frameNum >= frameStart and frameNum < n_frames-frameCropEnd):
            binaryVid.write(binaryFrame)
    # Release video
    inVid.release()
    binaryVid.release()
    # Close windows
    cv2.destroyAllWindows()
    return

def extraction(StabilizedVid, Binary, OutputVideo_DIR_PATH="", frameStart =50):
    '''
    :param StabilizedVid: stabilized video
    :param Binary: binary video of FG
    :param OutputVideo_DIR_PATH: path for output
    :param frameStart: starting frame position for output video
    :return: Binary video*stabilized video
    '''
    # Get video parameters
    n_frames = int(Binary.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(StabilizedVid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(StabilizedVid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(StabilizedVid.get(cv2.CAP_PROP_FPS))

    # Set up output video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    outVidB = cv2.VideoWriter(OutputVideo_DIR_PATH+'extracted.avi', fourcc, fps, (width, height))

    StabilizedVid.set(1,frameStart)
    for frameNum in range(n_frames):
        # Read next frame
        success1, Frame = StabilizedVid.read()
        success2, BinaryFrame = Binary.read()

        if not success1 or not success2:
            break
        Masked_Binary = cv2.bitwise_and(Frame, BinaryFrame)

        outVidB.write(Masked_Binary)
        print("Frame: " + str(frameNum) + "/" + str(n_frames) + " - Binary Filtered")

    # Release video
    StabilizedVid.release()
    outVidB.release()

    # Close windows
    cv2.destroyAllWindows()

    return
