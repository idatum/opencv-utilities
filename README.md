# OpenCV Utilities

## OpenCV camera calibration
OpenCV can calculate calibration parameters to undistort wide angle (or any pinhole) camera. I peformed OpenCV's chessboard calibratoin on an Amcrest IP8M-2496EB-40MM PoE IP camera (3840x2160 image, 4.0mm narrower angle lens). The normally barrel shaped distortion is removed with a small loss in FoV, a good compromise. The result is a nice high-resolution image that is distortion free, which I use for weather time-lapse videos.

See the documentation for details:
[Camera calibration with square chessboard](https://docs.opencv.org/4.5.1/d4/d94/tutorial_camera_calibration.html)

<code>camcalibration.py</code>has calibration parameters for the Amcrest IP8M-2496EB-40MM and a simple example of how to undistort.