import numpy as np
import cv2


##############################################  Video Stabilization ####################################################
def VidStabilization(inVid, SMOOTHING_RADIUS = 50, INLINERS_PERCENT=0.2, MAX_ERROR=1,Output_DIR_PATH="" ):
    # Get video parameters
    n_frames = int(inVid.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(inVid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(inVid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = inVid.get(cv2.CAP_PROP_FPS)

    # Set up output video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    outVid = cv2.VideoWriter(Output_DIR_PATH + 'stabilized.avi', fourcc, fps, (width, height))

    # Read first frame
    _, prevFrame = inVid.read()

    # Convert frame to grayscale
    prev_gray = cv2.cvtColor(prevFrame, cv2.COLOR_BGR2GRAY)

    # Pre-define transformation-store array
    transforms = np.zeros((n_frames - 1, 3), np.float32)

    for i in range(n_frames - 2):
        # Detect feature points in previous frame
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)

        # Read next frame
        success, curr = inVid.read()
        if not success:
            break
        # Convert to grayscale
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow (i.e. track feature points)
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

        # Sanity check
        assert prev_pts.shape == curr_pts.shape

        # Filter only valid points
        idx = np.where(status == 1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]

        M = transformMatrixRANSAC(prev_pts, curr_pts)

        # Extract traslation
        dx = M[0, 2]
        dy = M[1, 2]

        # Extract rotation angle
        da = np.arctan2(M[1, 0], M[0, 0])

        # Store transformation
        transforms[i] = [dx, dy, da]

        # Move to next frame
        prev_gray = curr_gray

        print("Frame: " + str(i) + "/" + str(n_frames) + " - Stabilized")

    # Compute trajectory using cumulative sum of transformations
    trajectory = np.cumsum(transforms, axis=0)

    # Create variable to store smoothed trajectory
    smoothed_trajectory = smooth(trajectory, SMOOTHING_RADIUS)

    # Calculate difference in smoothed_trajectory and trajectory
    difference = smoothed_trajectory - trajectory

    # Calculate newer transformation array
    transforms_smooth = transforms + difference

    # Reset stream to first frame
    inVid.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Write n_frames-1 transformed frames
    for i in range(n_frames - 2):
        # Read next frame
        success, frame = inVid.read()
        if not success:
            break

        # Extract transformations from the new transformation array
        dx = transforms_smooth[i, 0]
        dy = transforms_smooth[i, 1]
        da = transforms_smooth[i, 2]

        # Reconstruct transformation matrix accordingly to new values
        m = np.zeros((2, 3), np.float32)
        m[0, 0] = np.cos(da)
        m[0, 1] = -np.sin(da)
        m[1, 0] = np.sin(da)
        m[1, 1] = np.cos(da)
        m[0, 2] = dx
        m[1, 2] = dy

        # Apply affine wrapping to the given frame
        frame_stabilized = cv2.warpAffine(frame, m, (width, height))

        # Fix border artifacts
        frame_stabilized = fixBorder(frame_stabilized)

        # Write the frame to the file
        frame_out = frame_stabilized

        outVid.write(frame_out)

    # Release video
    inVid.release()
    outVid.release()
    return

def smooth(sig,SMOOTHING_RADIUS = 50):
    smoothed_sig = np.copy(sig)
    winSize = 2 * SMOOTHING_RADIUS + 1
    filt = np.ones(winSize) / winSize  # Define the filter
    for i in range(3):
        sig_pad = np.lib.pad(sig[:, i], (SMOOTHING_RADIUS, SMOOTHING_RADIUS), 'edge')  # padding the signal
        sig_smoothed = np.convolve(sig_pad, filt, mode='same')  # apply filter
        smoothed_sig[:, i] = sig_smoothed[SMOOTHING_RADIUS:-SMOOTHING_RADIUS]  # Remove padding
    return smoothed_sig

def fixBorder(frame):
    s = frame.shape
    # Scale the image 4% without moving the center
    T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.04)
    frame = cv2.warpAffine(frame, T, (s[1], s[0]))
    return frame

def transformMatrixRANSAC(prev_pts, curr_pts, INLINERS_PERCENT=0.2, MAX_ERROR=1):
    #prev = np.zeros((prev_pts.shape[0], 2))
    prev = prev_pts[:, 0, :]
    #curr = np.zeros((prev_pts.shape[0], 2))
    curr = curr_pts[:, 0, :]
    p = 0.99
    w = INLINERS_PERCENT
    n = 4
    k = int(np.log10(1 - p) / np.log10(1 - w ** n))
    max_fit_percent = 0
    for i in range(k):
        try:
            rnd_ind = np.random.choice(prev_pts.shape[0], 4, replace=False)
            prev_pts_rnd = prev_pts[rnd_ind, :, :]
            curr_pts_rnd = curr_pts[rnd_ind, :, :]
            # m = cv2.estimateRigidTransform(prev_pts_rnd, curr_pts_rnd, fullAffine=False)
            m = cv2.estimateAffinePartial2D(prev_pts_rnd, curr_pts_rnd)[0]
            curr_new = np.matmul(m[:, 0:2], prev.T)
            curr_new = curr_new.T
            curr_new[:, 0] = curr_new[:, 0] + m[0, 2]
            curr_new[:, 1] = curr_new[:, 1] + m[1, 2]
            delta = np.linalg.norm(curr - curr_new, axis=1)
            curr_mse = np.mean(delta)
            fit_percent = np.sum(delta <= MAX_ERROR) / len(delta)
            if fit_percent == max_fit_percent and curr_mse < min_mse:
                tMat = m
                max_fit_percent = fit_percent
                min_mse = curr_mse
            if fit_percent > max_fit_percent:
                tMat = m
                max_fit_percent = fit_percent
                min_mse = curr_mse
        except:
            pass
    return tMat

