# Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

## Goal
To detect and display the road lane boundaries on a video sample using advanced lane finding techniques.

## Steps

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

NOTE: The IPython notebook `Advanced-Lane-Lines.ipynb` contains my code laid out with the same headings as in this readme. Please use the headings as a guide to finding the correct code snippets. The main cell, *Advanced Lane Finding Method*, is near the end of the notebook. This cell calls the various functions for each video image.


[//]: # (Image References)

[image1]: ./output_images/calibration.png "Calibration"
[image2]: ./output_images/undistorted.png "Undistorted"
[image3]: ./output_images/grad_binaries.png "Gradient Binaries"
[image4]: ./output_images/color_binaries.png "Color Binaries"
[image5]: ./output_images/combination.png "Combination of Gradient and Color"
[image6]: ./output_images/warped_cal.png "Warped Calibration"
[image7]: ./output_images/warped.png "Warped Images"
[image8]: ./output_images/histogram.png "Histogram"
[image9]: ./output_images/windows.png "Windows"
[image10]: ./output_images/subsequent.png "Subsequent Images"
[image11]: ./output_images/curvature.png "Lane Curvature"
[image12]: ./output_images/lanes.png "Lanes"
[image13]: ./output_images/chal_grad.png "Challenge Gradient Binaries"
[image14]: ./output_images/chal_color.png "Challenge Color Binaries"
[image15]: ./output_images/chal_combo.png "Challenge Combination"
[image16]: ./output_images/chal_warped.png "Challenge Warped"
[image17]: ./output_images/chal_hist.png "Challenge Histogram"
[image18]: ./output_images/chal_windows.png "Challenge Windows"
[image19]: ./output_images/chal_lanes.png "Challenge Lanes"
[image20]: ./output_images/hchal_grad.png "Harder Challenge Gradient Binaries"
[image21]: ./output_images/hchal_color.png "Harder Challenge Color Binaries"
[image22]: ./output_images/hchal_combo.png "Harder Challenge Combination"
[image23]: ./output_images/hchal_warped.png "Harder Challenge Warped"
[image24]: ./output_images/hchal_hist.png "Harder Challenge Histogram"
[image25]: ./output_images/hchal_windows.png "Harder Challenge Windows"
[image26]: ./output_images/hchal_lanes.png "Harder Challenge Lanes"

[video1]: ./project_output.mp4 "Output Video"

## Camera Calibration

In order to correctly remove distortion from an image, a camera calibration is performed. The camera is calibrated on 20 chessboard images, from which the "object points" and "image points" are obtained. The inner chessboard corners could not be found on 3 images, probably due to distortion of the corners near the borders. This is acceptable as the camera is still able to compute the camera matrix (mtx) and distortion coefficients (dist) to be used to remove image distortion.

![alt text][image1]

## Image Distortion

The camera matrix and distortion coefficients are used to remove image distortion. The function take in a RGB image and returns the undistorted version.

![alt text][image2]

## Gradient Thresholded Binaries

The gradient of an image is essential to generalizing lane detection as it is not affected by image brightness. Below you can see the results of various gradient transforms. These are displayed to assist in choosing an effective combination. The final combined binary image is comprised of the following parts:

```
combined[((sxbinary == 1) & (sybinary == 1)) | ((mag_binary == 1) & (dir_binary == 1))
         | ((sxbinary == 1) & (dir_binary == 1))] = 1
```

A kernel size of 9 is chosen to smooth over noise. The kernel size is the area in the image over which the gradient is calculated.

![alt text][image3]

## Color Thresholded Binaries

The color of an image is also essential to generalizing lane detection as it is able to differentiate and effectively isolate desired colors. The final combination of colors is chosen as follows, which is effectively the S channel with limitations from H, R and L:

`combo[((S_binary == 1) & ((H_binary == 1) | (R_binary == 1) | (L_binary == 1)))] = 1`

The red and luminosity channels don't provide any value in the concrete section, but together with the H channel, they remove the shadow from the saturation image. They play an important role in more difficult scenarios (** see harder challenge section*********)

![alt text][image4]

## Gradient + Color Combination

The gradient and color binary thresholded combinations are then OR'd together as follows:

`combined_binary[(combined == 1) | (combo == 1)] = 1`

The center stacked combination image shows the contributions from gradient and color in green and blue respectively. The resultant binary image clearly locates the lanes without much noise nearby.

![alt text][image5]

## Perspective Transform

In a normal image the lanes are seen to almost converge as you look into the distance, whereas they are actually parallel. To correctly view and detect this, a perspective transform is required. This also allows for the lane curvature to be calculated.

The source points for the perspective transform are chosen in a trapezoidal shape around the lane. They are plotted to the destination points.

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 589, 453      | 300, 0        | 
| 691, 453      | 980, 0        |
| 1088, 705     | 980, 720      |
| 217, 705      | 300, 720      |

An image with straight road lanes was used to verify that the lanes are correctly parallel. This is then applied with confidence to subsequent images.

![alt text][image6]

![alt text][image7]

## Class Line()

A Class Line was defined to remember variables from previous images.

## Histogram

A histogram of the binary warped image is used to determine the start of the left and right lane lines.

![alt text][image8]

## Lane Detection

The basic method used to detect a lane is to first determine the base position of each lane using the histogram function and then implement sliding windows from the bottom of the image (predicted base) to the top. The polynomial coefficients of *f(y) = Ay^2 + By + C* are calculated from the lane pixels and the resultant curve plotted in yellow. Subsequent images use the previous images' curve as a starting point to detect the next lane position.

![alt text][image9]

![alt text][image10]

The following code works well to extend lanes when very few pixels are detected (< minpix). It simply takes the average of the previous 5 lane increments and uses that to move the window.

```
#If found > minpix pixels, re-center the next window to the mean position
if len(good_left_inds) > minpix:
    leftx_current = np.int(np.mean(nonzerox[good_left_inds]))

#If found < minpix pixels, examine dwin_list for previous movements and move accordingly
else:
    dlwin_avg = sum(dlwin_list[-5:]) / 5
    leftx_current = int(leftx_current + dlwin_avg)
```

If the detected lane is 'incorrect' or 'lost', the histogram and sliding windows function must be used to attempt to detect the lane again. The problem is how to define when a lane is incorrect. In this case, when the lane size = 0 or if the number of detected pixels is less than the sum of the minimum per window, then it is assumed that the lane was incorrectly detected.

```
#Right lane
if (rightx.size == 0) | (righty.size == 0):
    rlane.detected = False
else:
    rlane.detected = True
    rlane.allx = rightx
    rlane.ally = righty

#Define limit - if the number of detected pixels is below a threshold,
#run sliding windows on the next image.
thresh = 15 * 50    # no. windows * min pixels per window
if len(llane.allx < thresh):
    llane.detected = False
if len(rlane.allx < thresh):
    rlane.detected = False
```

## Radius of Curvature

The pixel coordinates are converted to meters using the assumption that the lanes are 3.7m apart and are 30m long. The radius of curvature is then calculated using the polynomial coefficients in the following equation:

Rcurve = ((1 + (2A + B)^2)^1.5) / np.absolute(2A)

The distance of the car left of the center of both lanes is also calculated.

![alt text][image11]

## Warp to Original Image

The left and right lane detected pixels and the road between the calculated curves are drawn onto the warped image in red, blue and green respectively. The modified warped binary image is stacked and 'unwarped', using the inverse transform matrix (Minv), back to its original form. The lane offset and left and right lane radii are displayed on the image to. The result can be seen below.

![alt text][image12]

## Pipeline - Advanced Lane Finding Method

This is the main function that takes as input the video images and returns the image with detected left and right lane pixels and region between the lanes colored in red, blue and green respectively. 

Every step described in this report is called from this function. It also decides whether to run the *sliding windows* or *subsequent* function, based on the lane status.

```
if llane.detected == False:
    #This is the first image
    #Get the base of the left and right lanes from the histogram of the binary warped image
    leftx_base, rightx_base = get_hist(binary_warped)
    #Implement sliding windows and fit a polynomial
    out_img, ploty = get_poly_first_image(binary_warped, leftx_base, rightx_base)
else:
    #These are subsequent images
    #Calculate the polynomial coefficients
    ploty = get_poly_subs_image(binary_warped)
```

Below is the result of the pipeline on project_video.mp4

![alt text][video1]

## Challenge Video

The pipeline above is created for the project video. The challenge video is not required for project submission, but nevertheless provides an interesting insight into the limitations and strengths of the pipeline. The images below are from the challenge video.

The gradient binary images detect too much information here. The barrier on the far left and changes in road surface to the right of both lanes are detected. 

![alt text][image13]

The red color binary provides a perfect answer in this image. The H and S channels also detect the yellow line. The combination image is derived from the S channel and so has limited, but correct information.

![alt text][image14]

The combination image receives correct information from the color image and correct and incorrect information from the gradient image. The resultant combination includes all the unwanted lines, revealing some of the shortfalls with the pipeline.

![alt text][image15]

The perspective transform is as expected. It contains all the unwanted lines and will most likely lead to incorrect curve predictions.

![alt text][image16]

The histogram correctly predicts the both lane starting points.

![alt text][image17]

The left lane windows follow the yellow lane as it has the most detected pixels in each window. It moves in the general direction of the yellow lane but is thrown off when it detects the barrier. This could possibly be avoided by reducing the depth of the source points in the perspective transform.

The right lane windows favor the black mark as it has a higher pixel count than the dashed road lane. A disadvantage to ORing the gradient binary image with the color binary image.

![alt text][image18]

The final image clearly shows that the detected depth is too long.

![alt text][image19]

## Harder Challenge Video

The pipeline above is created for the project video. The harder challenge video is not required for project submission, but nevertheless provides an interesting insight into the limitations and strengths of the pipeline. The images below are from the harder challenge video.

The gradient binary images are not affected by the shadows and excellently detect both lanes. The trees, however, are clearly a problem.

![alt text][image13]

The H channel detects the left lane and S channel the left lane. THe L channel thresholds can also be changed to detect the right lane, but this adds no value.

![alt text][image14]

The combination image has correctly detected the lanes, but the noise is concerning, especially just outside the right lane.

![alt text][image15]

Once again the ROI is problematic. It is too deep and when the road bends later in the video, the perspective image results are terrible. The perspective transform is extremely noisy outside the right lane.

![alt text][image16]

The histogram is not exactly correct, but with the sliding windows method should pick up the lane base.

![alt text][image17]

The windows quickly detect the lane lines, but are soon led astray by the noise in the warped image.

![alt text][image18]

The final image clearly shows that the detected depth is too long.

![alt text][image19]