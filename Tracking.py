import numpy as np
import numpy.matlib
import cv2
import GUI_tracking as gui


###################################################  Tracking   ########################################################

def Tracking(inVid, VideoParameters, Output_DIR_PATH = "", showFrame1=False):
    #Main track,ng function:
    # Input : - video CV2 imported
    #         - Output folder
    #         - Initial position of the tracked object
    #         - ShowFrame1 boolean True if you want to check that the inital particle is well placed

    s_initial = VideoParameters[0]
    # Get video parameters
    n_frames = int(inVid.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(inVid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(inVid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = inVid.get(cv2.CAP_PROP_FPS)
    # Read first frame
    _, I = inVid.read()

    ##### OUTPUT VIDEO INIT ######
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    outVid = cv2.VideoWriter(Output_DIR_PATH+'OUTPUT.avi', fourcc, fps, (width, height))

    if showFrame1:
        Ibis = drawRectangle(I,(s_initial[1],s_initial[0]),s_initial[2],s_initial[3],'blue')
        cv2.imshow('First frame: tracked object.',Ibis)
        cv2.waitKey(1500)
        cv2.destroyAllWindows()
    ##### INITIALIZE #####
    # Create initial particle matrix 'S' (SIZE 6xN)
    S = predictParticles(np.matlib.repmat(s_initial, 1, n_frames),VideoParameters)

    # COMPUTE NORMALIZED HISTOGRAM
    q = compNormHist(I, s_initial)

    # COMPUTE NORMALIZED WEIGHTS (W) AND PREDICTOR CDFS (C)
    W = []
    for n in range(n_frames - 1):
        p = compNormHist(I, S[:, n])
        W.append(compBatDist(p, q))
    W = W / np.sum(W)
    C = np.cumsum(W)
    # ........

    for i in range(n_frames):
        S_prev = S
        # LOAD NEW IMAGE FRAME
        success, I = inVid.read()
        if not success:
            break
        # SAMPLE THE CURRENT PARTICLE FILTERS
        S_next_tag = sampleParticles(S_prev, C)

        # PREDICT THE NEXT PARTICLE FILTERS (YOU MAY ADD NOISE
        S = predictParticles(S_next_tag, VideoParameters)

        # COMPUTE NORMALIZED WEIGHTS (W) AND PREDICTOR CDFS (C)
        # YOU NEED TO FILL THIS PART WITH CODE:
        # ........
        W = []
        for n in range(n_frames):
            p = compNormHist(I, S[:, n])
            W.append(compBatDist(p, q))
        W = W / np.sum(W)
        C = np.cumsum(W)

        # CREATE DETECTOR PLOTS
        frame_out = showParticles(I, S, W)
        outVid.write(frame_out)
        print("Tracking: Frame: " + str(i) + "/" + str(n_frames))  # Release video
    inVid.release()
    outVid.release()
    # Close windows
    cv2.destroyAllWindows()
    return

def compBatDist(p, q):
    return np.exp(20*np.sum(np.sqrt(np.multiply(p,q))))

def predictParticles(S_next_tag, VideoParameters):
    N = int(S_next_tag.shape[1]*S_next_tag.shape[0] / 6)
    if S_next_tag.shape!= (6,N):
        S_next = np.reshape(S_next_tag,(N,6))
        S_next = S_next.T
    else: S_next = S_next_tag
    S_next[0, :] = S_next[0, :] + S_next[4, :]
    S_next[1, :] = S_next[1, :] + S_next[5, :]
    #Add Noise
    mu = np.zeros(6)  # mean and standard deviation
    cov = VideoParameters[1]
    noise = np.random.multivariate_normal(mu,cov,N).T
    S_next = S_next+noise
    return S_next

def compNormHist(I, S):
    xc = np.round_(S[0]).astype(int)
    yc = np.round_(S[1]).astype(int)
    hlfWidth = np.round_(S[2]).astype(int)
    hlfHeight = np.round_(S[3]).astype(int)

    if yc < hlfHeight:
        yc = hlfHeight
    elif yc + hlfHeight > I.shape[0]:
        yc = I.shape[0] - hlfHeight
    if xc < hlfWidth:
        xc = hlfWidth
    elif xc+hlfWidth > I.shape[1]:
        xc = I.shape[1]-hlfWidth
    I_subportion = I[yc-hlfHeight:yc+hlfHeight,xc-hlfWidth:xc+hlfWidth,:]
    I_subportion_q = np.floor(I_subportion/16).astype(np.uint8)

    hist = cv2.calcHist([I_subportion_q], [0, 1, 2], None, [16, 16, 16], [0, 15, 0, 15, 0, 15])
    normHist = hist/np.sum(hist)

    return normHist

def sampleParticles(S_prev, C):
    S_next_tag=np.zeros_like(S_prev)
    for n in range(S_prev.shape[1]):
        r = np.random.uniform(0, 1, 1)
        j = next(x[0] for x in enumerate(C) if x[1] >= r)       # returns first index of x such that the element greater than r
        S_next_tag[:, n] = S_prev[:, j]
    return S_next_tag

def showParticles(I1, S, W):

    particle_max = S[:,np.argmax(W)]
    I2 = drawRectangle(I1, (particle_max[1], particle_max[0]), particle_max[2], particle_max[3], 'red')

    particle_avg = np.average(S, axis=1).astype(int)
    I3 = drawRectangle(I2, (particle_avg[1], particle_avg[0]), particle_avg[2], particle_avg[3], 'green')

    return I3

def drawRectangle(Img,Center,Semi_Width,Semi_Height,color):
    #Returns the image with a rectangle  of Height, widht, traced around the center
    Center = [int(Center[0]), int(Center[1])]
    Semi_Width = int(Semi_Width)
    Semi_Height = int(Semi_Height)

    #Reminder CV2 works in BGR
    if color =='blue':
        pix_color = (255, 0, 0)
    elif color == 'green':
        pix_color = (0, 255, 0)
    else:
        pix_color = (0, 0, 255)
    Img[Center[0] - Semi_Height : Center[0] + Semi_Height , Center[1] - Semi_Width-2 :Center[1] - Semi_Width+2 ] = pix_color
    Img[Center[0] - Semi_Height : Center[0] + Semi_Height , Center[1] + Semi_Width-2 :Center[1] + Semi_Width+2 ] = pix_color
    Img[Center[0] - Semi_Height-2 : Center[0] - Semi_Height+2 , Center[1] - Semi_Width : Center[1] + Semi_Width] = pix_color
    Img[Center[0] + Semi_Height-2 : Center[0] + Semi_Height+2 , Center[1] - Semi_Width : Center[1] + Semi_Width] = pix_color
    return Img

### Get Tracking parameters

def getVideoParameters(inVid):
    _,First_frame = inVid.read()
    CroppedSelection = gui.GUI_selection_of_tracked_person(First_frame)
    x_center =int( CroppedSelection[0]+CroppedSelection[2]/2)
    y_center =int( CroppedSelection[1]+CroppedSelection[3]/2)
    half_width = int(CroppedSelection[2] /2)
    half_height = int(CroppedSelection[3] /2)

    s_initial = [x_center,  # x center
                 y_center,  # y center
                 half_width,  # half width
                 half_height,  # half height
                -2,  # velocity x
                0]  # velocity y
    cov = [[5, 0, 0, 0, 0, 0],
           [0, 0.1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 3, 0],
           [0, 0, 0, 0, 0, 0]]

    parameters = [s_initial,cov]
    inVid.set(1,0)
    return parameters
