import numpy as np
import numpy.matlib
import cv2
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import wdt as wdt  # For distance computation


### Main functions of background subtraction and matting
def ChangeVideoBackground(StabilizedVid, BinaryVid, wallpaper, FGLikelihood, BGLikelihood, OutputVideo_DIR_PATH="",
                          frameStart=50, frameCropEnd=50):
    # Input: Stabilized video, new background, liklihood maps for BG and FG and binary video
    # Output: Video with replaced background
    # Uses changeFrameBackground for each frame
    print("############# Changing Video Background #######################")
    n_frames = int(BinaryVid.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(StabilizedVid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(StabilizedVid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = StabilizedVid.get(cv2.CAP_PROP_FPS)

    ##### OUTPUT VIDEO INIT ######
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    outVid = cv2.VideoWriter(OutputVideo_DIR_PATH + 'matted.avi', fourcc, fps, (width, height))

    StabilizedVid.set(1, frameStart)
    BinaryVid.set(1, 0)
    for frameNum in range(n_frames):
        success1, frame = StabilizedVid.read()  # Read the frame
        success2, binaryframeRGB = BinaryVid.read()  # Read the frame
        if not success1:
            print("Could not read video frame" + str(frameNum + 1) + "/" + str(n_frames))
            break
        if not success2:
            print("Could not read binary video frame" + str(frameNum + 1) + "/" + str(n_frames))
            break

        binaryframe = binaryframeRGB[:, :, 0]

        FG_L_Map, BG_L_Map = FG_BG_Proba_Map(frame, FGLikelihood, BGLikelihood)
        try:
            Matted = changeFrameBackground(frame, binaryframe, wallpaper, FG_L_Map, BG_L_Map)
            outVid.write(Matted)
            print("Changed Background of frames:" + str(frameNum + 1) + "/" + str(n_frames))
        except:
            print(str(frameNum + 1) + "not matted-- ERROR")
    StabilizedVid.release()
    BinaryVid.release()
    outVid.release()
    return


def changeFrameBackground(OriginalFrame, binaryframe, NewBackground, FG_L_Map, BG_L_Map, OutputPlots_DIR_PATH="",
                          saveFigure=False):
    """ Input: Stabilized frame, new background, liklihood maps for BG and FG and binary frame
        Output: frame with replaced background"""
    ### Gradient map for BG and FG probilities
    Grad_FG_Map, Grad_BG_Map = Gradient_Map(FG_L_Map, BG_L_Map)  # Calculate gradient maps for the liklihood maps

    ### Computation of Distance Maps
    distance_transform_FG, distance_transform_BG = get_Grad_Distance_Map(Grad_FG_Map, Grad_BG_Map, binaryframe,
                                                                         EROSION_SIZE=5)

    ### Trimap computation
    Trimap = get_Trimap_frame(distance_transform_BG, distance_transform_FG, binaryframe)

    ### Trimap refinement
    FG_L_Map_refined, BG_L_Map_refined = Trimap_refinement(Trimap, OriginalFrame)

    ### Opactiy Map computation
    AlphaMap = computeAlphaMap(Trimap, FG_L_Map_refined, BG_L_Map_refined, distance_transform_FG, distance_transform_BG,
                               r=2)

    ### Matting
    Matted = Matting_simple(OriginalFrame, NewBackground, Trimap, AlphaMap, distance_transform_FG,
                            distance_transform_BG)

    if saveFigure:
        # figures
        plt.imshow(Trimap)
        plt.colorbar(extend='both')
        plt.savefig(OutputPlots_DIR_PATH + "Trimap.png")
        plt.close()

        cv2.imwrite(OutputPlots_DIR_PATH + "Matted.png", Matted)

        plt.imshow(distance_transform_FG, cmap='jet')
        plt.colorbar()
        plt.clim(0, 10)
        plt.savefig(OutputPlots_DIR_PATH + "distance_transform_FG.png")
        plt.close()

        plt.imshow(distance_transform_BG, cmap='jet')
        plt.colorbar()
        plt.clim(0, 10);
        plt.savefig(OutputPlots_DIR_PATH + "distance_transform_BG.png")
        plt.close()

    return Matted


### Functions for Likelihood of FG and BG comutation
def FG_BG_likelihood_frame(stabFrame, BinaryFrame, ChannelNum=2, bandwidth_KDE=0.3):
    '''Input: Stabilized frame, binary frame, HSV channel, KDE bandwidth
        Ourput: FG and BG liklihood and grid per frame'''

    stabFrameV = cv2.cvtColor(stabFrame, cv2.COLOR_BGR2HSV)[:, :,
                 ChannelNum]  # convert stabilized frame to Luma channel

    # identify FG and BG scribbles based on the binary frame
    FG_scribbles = stabFrameV[np.where(BinaryFrame > 200)]
    BG_scribbles = stabFrameV[np.where(BinaryFrame == 0)]

    # Takes 1000 random points of BG for KDE computation
    randomIdx = numpy.random.randint(0, BG_scribbles.shape[0], 1000)
    BG_scribbles = BG_scribbles[randomIdx]
    randomIdx = numpy.random.randint(0, FG_scribbles.shape[0], 1000)
    FG_scribbles = FG_scribbles[randomIdx]

    x_grid = np.linspace(0, 255, 256)
    BGLikelihood = np.zeros(256)
    FGLikelihood = np.zeros(256)

    # compute KDE for BG and FG
    try:
        FGLikelihood = kde_scipy(FG_scribbles, x_grid, bandwidth=bandwidth_KDE)
    except:
        print('no foreground detected')
    try:
        BGLikelihood = kde_scipy(BG_scribbles, x_grid, bandwidth=bandwidth_KDE)
    except:
        print('no background detected')
    return FGLikelihood, BGLikelihood, x_grid


def FG_BG_likelihood_all_vid(StabilizedVid, BinaryVid, frameInterval=20, frameStart=50, saveFig=False,
                             Output_DIR_PATH=""):
    '''Input: Stabilized frame, binary frame, HSV channel, KDE bandwidth, Interval
        Ourput: FG and BG liklihood and grid for all video by sampling frames with frameInterval as sampling rate'''
    n_frames = int(StabilizedVid.get(cv2.CAP_PROP_FRAME_COUNT))

    # Sample frames
    FGpix = np.array([])
    BGpix = np.array([])
    for frameNum in range(frameStart, n_frames - 1, frameInterval):
        StabilizedVid.set(1, frameNum)  # Where frame_no is the frame you want
        BinaryVid.set(1, frameNum - frameStart)  # Where frame_no is the frame you want
        success1, frame = StabilizedVid.read()  # Read the frame
        success2, BinaryFrame = BinaryVid.read()  # Read the frame
        if not (success1 and success2):
            break
        BinaryFrame = BinaryFrame[:, :, 0]

        stabFrameV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 2]  # convert stabilized frame to Luma channel

        # identify FG and BG scribbles based on the binary frame
        FG_scribbles = stabFrameV[np.where(BinaryFrame > 200)]
        BG_scribbles = stabFrameV[np.where(BinaryFrame == 0)]

        # Takes 1000 random points of BG for KDE computation
        randomIdx = numpy.random.randint(0, BG_scribbles.shape[0], 1000)
        BG_scribbles = BG_scribbles[randomIdx]
        randomIdx = numpy.random.randint(0, FG_scribbles.shape[0], 1000)
        FG_scribbles = FG_scribbles[randomIdx]

        FGpix = np.append(FGpix, BG_scribbles)
        BGpix = np.append(BGpix, FG_scribbles)
        print("Adding FG and BG for likelihood computation:" + str(frameNum + 1) + "/" + str(n_frames))

    # compute KDE for BG and FG
    x_grid = np.linspace(0, 255, 256)
    try:
        FGLikelihood = kde_scipy(FGpix, x_grid, bandwidth=0.15)
    except:
        FGLikelihood = np.zeros(256)
        print('no foreground detected')
    try:
        BGLikelihood = kde_scipy(BGpix, x_grid, bandwidth=0.15)
    except:
        BGLikelihood = np.zeros(256)
        print('no background detected')

    print('FG and BG likelihoods computed ')

    if saveFig:
        plot_KDE_Likelihood(FGLikelihood, BGLikelihood, x_grid, Output_DIR_PATH)

    return FGLikelihood, BGLikelihood, x_grid


def pixelLikelihood(Pixel, FGLikelihood, BGLikelihood):
    '''Input: Probability maps of BG and FG and pixel value
        Output: Liklihood of a pixel to be drawn form BG and FG distributions'''
    if not (FGLikelihood[Pixel] == 0 and BGLikelihood[Pixel] == 0):
        FG_Proba = FGLikelihood[Pixel] / (FGLikelihood[Pixel] + BGLikelihood[Pixel])
    else:
        FG_Proba = 0.5
    BG_Proba = 1 - FG_Proba
    return FG_Proba, BG_Proba


def kde_scipy(x, x_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scipy"""
    print("Computing FG / BG likelihood KDE:")
    kde = gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1), **kwargs)
    kde_evaluated = kde.evaluate(x_grid)
    print("KDE computed")
    return kde_evaluated


### Function for probability MAP computation
def FG_BG_Proba_Map(stabFrame, FGLikelihood, BGLikelihood, Output_DIR_PATH="", saveFigures=False):
    '''Input:liklihood of all pixel values of BG and FG and pixel value
        Output: probability maps for BG and FG'''
    stabFrameV = cv2.cvtColor(stabFrame, cv2.COLOR_BGR2HSV)[:, :, 2]

    # Initialize
    FG_ProbaMap = np.zeros(shape=(stabFrameV.shape[0], stabFrameV.shape[1]))
    BG_ProbaMap = np.zeros(shape=(stabFrameV.shape[0], stabFrameV.shape[1]))

    # Loop through all pixel values and update the probability maps
    for color in range(0, 256):
        FG_ProbaMap[stabFrameV == color], BG_ProbaMap[stabFrameV == color] = pixelLikelihood(color, FGLikelihood,
                                                                                             BGLikelihood)

    if saveFigures:
        # Figures
        plt.imshow(stabFrameV)
        plt.title('Stab Frame V')
        plt.savefig(Output_DIR_PATH + 'StabFrameV.png')

        plt.imshow(FG_ProbaMap)
        plt.title('FG probability map')
        plt.savefig(Output_DIR_PATH + 'FG_Proba_Map.png')

        plt.imshow(BG_ProbaMap)
        plt.title('BG probability map')
        plt.savefig(Output_DIR_PATH + 'BG_Proba_Map.png')

    return FG_ProbaMap, BG_ProbaMap


### Functions for Trimap estimation
def Gradient_Map(FG_L_Map, BG_L_Map):
    '''Input: L_Map likelihood map
       Output: Gradient of the likelihood map'''

    sobelx = cv2.Sobel(FG_L_Map, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(FG_L_Map, cv2.CV_64F, 0, 1, ksize=3)
    Grad_FG_Map = np.sqrt(np.multiply(sobelx, sobelx) + np.multiply(sobely, sobely))

    sobelx = cv2.Sobel(BG_L_Map, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(BG_L_Map, cv2.CV_64F, 0, 1, ksize=3)
    Grad_BG_Map = np.sqrt(np.multiply(sobelx, sobelx) + np.multiply(sobely, sobely))

    return Grad_FG_Map, Grad_BG_Map


def get_Grad_Distance_Map(Grad_FG_Map, Grad_BG_Map, binaryFG, EROSION_SIZE=5, saveFig=False):
    '''Input: Gradient maps for BG and FG, Binary video and Eroision size
        Output:FG and BG distance maps'''

    # Prepare scribble points for the FG distance map. we use the binary vid as scribbles
    # we erode the binary to ensure no boundaries are chosen as scribbles
    binaryFGEroded = binaryFG
    tmp = binaryFGEroded
    k = 0
    while (np.sum(tmp) > 0 and k < EROSION_SIZE):
        binaryFGEroded = tmp
        tmp = cv2.erode(binaryFGEroded, np.ones((3, 3), np.uint8), iterations=1)
        k = k + 1
    # Prepare scribble points for the BG distance map. we use the binary vid as scribbles
    # we flip the binary picture so now the BG is chosen as scribbles
    # we erode the binary to ensure no boundaries are chosen as scribbles
    binaryBG = np.zeros_like(binaryFG)
    binaryBG[binaryFG > 0] = 0
    binaryBG[binaryFG == 0] = 255

    binaryBGEroded = binaryBG
    tmp = binaryBGEroded
    k = 0
    while (np.sum(tmp) > 0 and k < EROSION_SIZE):
        binaryBGEroded = tmp
        tmp = cv2.erode(binaryBGEroded, np.ones((3, 3), np.uint8), iterations=1)
        k = k + 1

    # calculate distance map
    # we use the BG scribble points as obstcles for the FG distance map and vice versa
    distance_transform_FG = get_Distance_Map(Grad_FG_Map, binaryFGEroded, binaryBGEroded)
    distance_transform_BG = get_Distance_Map(Grad_BG_Map, binaryBGEroded, binaryFGEroded)

    # Figures
    if saveFig:
        plt.imshow(Grad_FG_Map)
        plt.title('Grad_FG_Map')
        plt.savefig('Grad_FG_Map.png')

        plt.imshow(Grad_BG_Map)
        plt.title('Grad_BG_Map')
        plt.savefig('Grad_BG_Map.png')
    return distance_transform_FG, distance_transform_BG


def get_Distance_Map(Grad_Map, scribbles, Obstcles):
    # Input: Grad_Map: Gradient of the likelihood map (FG or BG)
    #       scribbles: Serve as the exits for the distance map (distance=0)
    #       Obstcles: pixels with distance set to inf
    # Output: Distance_Map
    print("Computing distance map ...")

    # check if there are scribbles, if not, distance map is set to inf
    # check if the scribbles are set to whole image. if so, set whole map to zero
    if (np.any(scribbles) and np.any(scribbles - 255)):
        cost_field = wdt.map_image_to_costs(Grad_Map, scribbles, Obstcles)
        cost_field = cv2.equalizeHist(np.uint8(cost_field))
        Dist_map = wdt.get_weighted_distance_transform(cost_field)
    elif np.any(scribbles):
        Dist_map = np.zeros_like(scribbles) + 1e-17
    else:
        Dist_map = np.full((scribbles.shape), np.inf)
    print("Distance map computed")

    Dist_map[Obstcles == 255] = 1e17
    return Dist_map


### Function for trimap computation
def get_Trimap_frame(distance_transform_BG, distance_transform_FG, binaryframe, DILATION_SIZE=5):
    """
    OutPut: Trimap with
                        Background 0
                        Forground 1
                        Unknown 3
    """

    binaryFGEroded = cv2.erode(binaryframe, np.ones((3, 3), np.uint8), iterations=DILATION_SIZE)

    binaryBG = np.zeros_like(binaryframe)
    binaryBG[binaryframe > 0] = 0
    binaryBG[binaryframe == 0] = 255
    binaryBGEroded = cv2.erode(binaryBG, np.ones((3, 3), np.uint8), iterations=DILATION_SIZE)

    Trimap = cv2.bitwise_not(cv2.bitwise_or(binaryBGEroded, binaryFGEroded))
    Trimap[Trimap > 0] = 3
    Trimap[binaryFGEroded > 0] = 1
    Trimap[binaryBGEroded > 0] = 0
    return Trimap


### Functions for trimap Refinement
def Trimap_refinement(Trimap, OriginalFrame):
    '''Input: Trimap and frame
       Output: Refined probability maps'''

    # find the outer band of FG pixels
    FGband = np.zeros_like(Trimap)
    FGband[Trimap == 1] = 255
    tmp = cv2.erode(FGband, np.ones((3, 3), np.uint8), iterations=1)
    FGband = FGband - tmp

    # find the inner band of BG pixels
    BGband = np.zeros_like(Trimap)
    BGband[Trimap == 0] = 255
    tmp = cv2.erode(BGband, np.ones((3, 3), np.uint8), iterations=1)
    BGband = BGband - tmp

    # convert image to luma
    Luma = cv2.cvtColor(OriginalFrame, cv2.COLOR_BGR2HSV)[:, :, 2]
    FGpix = Luma[np.where(FGband == 255)]
    BGpix = Luma[np.where(BGband == 255)]
    x_grid = np.linspace(0, 255, 256)

    # compute KDE for the pixles in the chosen bands
    FGLikelihood = kde_scipy(FGpix, x_grid, bandwidth=5)
    BGLikelihood = kde_scipy(BGpix, x_grid, bandwidth=5)

    FG_ProbaMap = np.zeros(shape=(Luma.shape[0], Luma.shape[1]))
    BG_ProbaMap = np.zeros(shape=(Luma.shape[0], Luma.shape[1]))
    # caclulate the new probability maps
    for color in range(0, 256):
        FG_ProbaMap[Luma == color], BG_ProbaMap[Luma == color] = pixelLikelihood(color, FGLikelihood, BGLikelihood)

    return FG_ProbaMap, BG_ProbaMap


### Function for Alpha Map computation
def computeAlphaMap(Trimap, FG_L_Map, BG_L_Map, DistMapFG, DistMapBG, r=2):
    """
    Input: FG / BG Likelihood map
           Distance Maps Foreground/ Background
           r=2 - Parameter must be [0,2]
    """
    # Output: AlphaMap - Map of parameter alpha for alpha matting
    wF = np.zeros_like(Trimap)
    wF[Trimap == 1] = 1
    wF[Trimap == 0] = 0

    wB = np.zeros_like(Trimap)
    wB[Trimap == 1] = 0
    wB[Trimap == 0] = 1

    wF[Trimap == 3] = np.multiply(np.power(DistMapFG[Trimap == 3] + 1e-17, -r), FG_L_Map[Trimap == 3])
    wB[Trimap == 3] = np.multiply(np.power(DistMapBG[Trimap == 3] + 1e-17, -r), BG_L_Map[Trimap == 3])

    DEN = np.power(wF + wB + 1e-17, (-1))
    AlphaMap = np.multiply(wF, DEN)

    AlphaMap[Trimap == 1] = 1
    AlphaMap[Trimap == 0] = 0

    return AlphaMap


### Functions for Matting
def Matting_simple(Stabilized, wallpaper, Trimap, AlphaMap, DistMapFG, DistMapBG, savefig=False):
    '''Compute Matted image
        Input: Stabilized frams, new background(wallpaper) opacity map (alpha) distnace mape
        Output: Matted image I = alpha*Stabilized + (1-alpha)*wallpaper'''
    Matted = Stabilized
    # All sure BG pixels copied from wallpaper
    Matted[DistMapBG < DistMapFG] = wallpaper[DistMapBG < DistMapFG]

    one_minus_alpha = (np.ones(AlphaMap.shape) - AlphaMap)
    tmpMatted = np.zeros_like(Stabilized)
    for i in range(3):
        tmpMatted[:, :, i] = np.multiply(AlphaMap, Stabilized[:, :, i]) + np.multiply(one_minus_alpha,
                                                                                      wallpaper[:, :, i])
    Matted[Trimap == 3] = tmpMatted[Trimap == 3]
    if savefig:
        plt.imshow(Matted)
        plt.title('Matted')
        plt.savefig('Matted_Frame.png')
    return Matted


def Matting_Enchanced(Stabilized, wallpaper, AlphaMap, DistMapFG, DistMapBG, Trimap, SrchRadius=5):
    '''Compute Matted image
          Input: Stabilized frams, new background(wallpaper) opacity map (alpha) distnace mape
          Output: Matted image I = alpha*X_fg + (1-alpha)*wallpaper
          where X_fg is the closest value close the the pixel which explains the pixel'''
    stabFrameV = cv2.cvtColor(Stabilized, cv2.COLOR_BGR2HSV)[:, :, 2]

    Unknown_idx = np.argwhere(Trimap == 3)
    FG_idx = np.argwhere(Trimap == 1)
    BG_idx = np.argwhere(Trimap == 0)

    price = 1e17

    for idxNum in range(Unknown_idx.shape[0]):
        Puk = Unknown_idx[idxNum]
        PcolorV = stabFrameV[Puk[0], Puk[1]]

        alpha = AlphaMap[Puk[0], Puk[1]]

        Frame_For_matting = Stabilized
        genFG = (Pfg for Pfg in FG_idx if Puk[0] - Pfg[0] < SrchRadius and Puk[1] - Pfg[1] < SrchRadius)
        genBG = (Pbg for Pbg in BG_idx if Puk[0] - Pbg[0] < SrchRadius and Puk[1] - Pbg[1] < SrchRadius)

        # Getting the coordinate of the best FG and BG pixel explaining the unknown pixel's color.
        for Pfg in genFG:
            for Pbg in genBG:
                price_tmp = abs(alpha * stabFrameV[Pfg[0], Pfg[1]] + (1 - alpha) * stabFrameV[Pbg[0], Pbg[1]] - PcolorV)
                if price_tmp < price:
                    price = price_tmp
                    Frame_For_matting[idxNum] = Stabilized[Pfg[0], Pfg[1]]

    Matted = np.zeros_like(Stabilized)

    # All sure BG pixels copied from wallpaper
    Matted[Trimap == 0] = wallpaper[Trimap == 3]

    one_minus_alpha = (np.ones(AlphaMap.shape) - AlphaMap)
    tmpMatted = np.zeros_like(Stabilized)
    for i in range(3):
        tmpMatted[:, :, i] = np.multiply(AlphaMap, Frame_For_matting[:, :, i]) + np.multiply(one_minus_alpha,
                                                                                             wallpaper[:, :, i])
    Matted[Trimap == 255] = tmpMatted[Trimap == 255]
    return Matted


#### Plot functions ####
def plot_KDE_Likelihood(FGLikelihood, BGLikelihood, x_grid, Output_DIR_PATH=""):
    # Plot of KDE
    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(13, 3))
    fig.subplots_adjust(wspace=0)

    ax[0].plot(x_grid, FGLikelihood, color='red', alpha=0.5, lw=3)
    ax[0].set_title('Foreground likelihood ')
    ax[0].set_xlim(0, 255)

    ax[1].plot(x_grid, BGLikelihood, color='blue', alpha=0.5, lw=3)
    ax[1].set_title('Background likelihood ')
    ax[1].set_xlim(0, 255)
    plt.title('Background likelihood')
    plt.savefig(Output_DIR_PATH + 'KDE likelihood functions.png')
    plt.close()
    return
