import sys
import cv2
import numpy as np

# OpenCV distortion parameters created using OpenCV 4.5
M1Type_CV_32FC1 = 5

# Amcrest IP8M-2496EB-40MM black bullet PoE camera
# Using 25.3 mm length square
# RMS: 1.620620402202919
K_Amcrest_IP8M_2496EB_40MM_mm_sq_len = np.array(
    [[2.23369116e+03, 0.00000000e+00, 1.94516801e+03],
     [0.00000000e+00, 2.25400639e+03, 1.06112635e+03],
     [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
D_Amcrest_IP8M_2496EB_40MM_mm_sq_len = np.array(
    [-0.34239597, 0.15648334, 0.00159584, -0.00458602, -0.04445101])

# Using 1 inch length square
# RMS: 1.6206178443558983
K_Amcrest_IP8M_2496EB_40MM = np.array(
    [[2.23368664e+03, 0.00000000e+00, 1.94517003e+03],
    [0.00000000e+00, 2.25400130e+03, 1.06112610e+03],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
D_Amcrest_IP8M_2496EB_40MM = np.array(
    [-0.34239419, 0.15648216, 0.00159585, -0.00458635, -0.04445046])


# KK-USBFHD01M-L36 Webcamera_USB branded 3.6mm wide angle 1/2.7" CMOS OV2710
# RMS: 0.9355137503915556
K_CMOS_OV2710_3_6MM = np.array(
    [[2.23832455e+03, 0.00000000e+00, 9.65907071e+02],
    [0.00000000e+00, 2.21520248e+03, 5.32389956e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
D_CMOS_OV2710_3_6MM = np.array(
    [-0.91590794, 1.3943585, 0.03393086, -0.00706482, -1.80818351])


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

