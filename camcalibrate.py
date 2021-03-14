import sys
import cv2
import numpy as np

# OpenCV distortion parameters created using OpenCV 4.5

# Amcrest IP8M-2496EB-40MM black bullet PoE camera
# RMS: 1.6206204022029178
K_Amcrest_IP8M_2496EB_40MM = np.array(
    [[2.23369116e+03, 0.00000000e+00, 1.94516801e+03],
     [0.00000000e+00, 2.25400639e+03, 1.06112635e+03],
     [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
D_Amcrest_IP8M_2496EB_40MM = np.array(
    [-0.34239597, 0.15648334, 0.00159584, -0.00458602, -0.04445101])
M1Type_CV_32FC1 = 5


def undistort(img, K, D):
    if img is None:
        return None
    h, w = img.shape[:2]
    # Generate new camera matrix from parameters
    newcameramatrix, _roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 0)
    # Generate look-up tables for remapping the camera image
    mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, newcameramatrix, (w, h), M1Type_CV_32FC1)
    # Remap the original image to a new image
    return cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)


if __name__ == "__main__":
    # expecting an Amcrest IP8M-2496EB-40MM image filename
    if len(sys.argv) < 2:
        print("Expecting an image filename")
        sys.exit(2)
    img_file = sys.argv[1]
    img = cv2.imread(img_file)
    if img is None:
        print("Invalid image")
        sys.exit(2)
    img = undistort(img, K_Amcrest_IP8M_2496EB_40MM, D_Amcrest_IP8M_2496EB_40MM)
    img_outfile = 'undistort.jpeg'
    print(f"Undistorted file: {img_outfile}")
    cv2.imwrite(img_outfile, img)

