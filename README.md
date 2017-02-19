# Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

## Goal
To detect and display the road lane boundaries on a video sample using advanced lane finding techniques.

## Steps

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

NOTE: The IPython notebook `Advanced-Lane-Lines.ipynb` contains the code laid out with the same headings as in this readme. Please use the headings as a guide to finding the correct code snippets in the notebook. The main pipeline, *Advanced Lane Finding Method*, is near the end of the notebook. This cell calls the various functions for each video image.


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
[video2]: ./challenge_output.mp4 "Challenge Video"
[video3]: ./harder_challenge_output.mp4 "Harder Challenge Video"

## Camera Calibration

In order to correctly remove distortion from an image, a camera calibration is performed. The camera is calibrated on 20 chessboard images, from which the "object points" and "image points" are obtained. The inner chessboard corners could not be found on 3 images, probably due to distortion of the corners near the borders. This is acceptable as the camera is still able to compute the camera matrix (mtx) and distortion coefficients (dist) to be used to remove image distortion.

![alt text][image1]

## Image Distortion

The camera matrix and distortion coefficients are used to remove image distortion. The function takes in a RGB image and returns the undistorted copy.

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

The red and luminosity channels don't provide any value in the concrete section, but together with the H channel, they remove the shadow from the saturation image to provide a very clean result.

![alt text][image4]

## Gradient + Color Combination

The gradient and color binary thresholded combinations are then OR'd together as follows:

`combined_binary[(combined == 1) | (combo == 1)] = 1`

The center stacked combination image shows the contributions from gradient and color in green and blue respectively. The resultant binary image clearly locates the lanes without much noise nearby.

![alt text][image5]

## Perspective Transform

In a normal image the lanes are seen to almost converge as you look into the distance, whereas they are actually somewhat parallel. To correctly view and detect this, a perspective transform is required. This also allows for the lane curvature to be calculated.

The source points for the perspective transform are chosen in a trapezoidal shape around the lane. They are plotted to the destination points. The depth of the selected points is quite short to improve the pipeline in the challenge and harder challenge scenarios.

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 550, 482      | 200, 0        | 
| 742, 482      | 1080, 0       |
| 1088, 705     | 1080, 720     |
| 217, 705      | 200, 720      |

An image with straight road lanes was used to calibrate the source and destination points and ensure the lanes are parallel in the warped output. This is then applied with confidence to subsequent images.

A Morphological Transformation is applied to the binary warped image. It is very effective at removing noise. This is especially evident in the harder challenge video. 

```
#Clearing noise in binary_warped
small_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
binary_warped = cv2.morphologyEx(noisy_warped, cv2.MORPH_OPEN, small_kernel)
```

![alt text][image6]

![alt text][image7]

## Class Line()

A Class, *Line*, is defined to remember variables from previous images.

## Histogram

A histogram of the binary warped image is used to determine the start of the left and right lane lines. The calculated left and right bases are detected correctly in the image below. This method is quite robust at correctly detecting the lane base if a good perspective transform is achieved.

![alt text][image8]

## Lane Detection

The first cell under lane detection implements the sliding windows method. The second cell detects lanes using the previous images' curve.

The basic method used to detect a lane is to first determine the base position of each lane using the histogram function and then implement sliding windows from the bottom of the image (predicted base) to the top. The polynomial coefficients of *f(y) = Ay^2 + By + C* are calculated from the lane pixels and the resultant curve plotted in yellow. Subsequent images use the previous images' curve as a starting point to detect the next lane position.

![alt text][image9]

![alt text][image10]

The following code works well to extend lanes when very few pixels are detected (ie. < minpix). It simply takes the average of the previous 5 lane increments and uses that to move the window.

```
#If found > minpix pixels, re-center the next window to the mean position
if len(good_left_inds) > minpix:
    leftx_current = np.int(np.mean(nonzerox[good_left_inds]))

#If found < minpix pixels, examine dwin_list for previous movements and move accordingly
else:
    dlwin_avg = sum(dlwin_list[-5:]) / 5
    leftx_current = int(leftx_current + dlwin_avg)
if len(good_right_inds) > minpix:        
    rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
else:
    drwin_avg = sum(drwin_list[-5:]) / 5
    rightx_current = int(rightx_current + drwin_avg)
```

If the detected lane is 'incorrect' or 'lost', the histogram and sliding windows function must be used to attempt to detect the lane again. The problem is how to define when a lane is incorrect. In this case, when the lane size = 0 or if the number of detected pixels is less than a threshold value, then it is assumed that the lane was incorrectly detected. This is only applicable to lanes detected using pervious curves.

```
#If points are detected in the left or right lanes, update self.allx and self.ally
#If points aren't detected, don't update
#Left lane
if (leftx.size == 0) | (lefty.size == 0):
    llane.detected = False
else:
    llane.detected = True
    llane.allx = leftx
    llane.ally = lefty

#Right lane
if (rightx.size == 0) | (righty.size == 0):
    rlane.detected = False
else:
    rlane.detected = True
    rlane.allx = rightx
    rlane.ally = righty

#Define limit - if the number of detected pixels is below a threshold,
#run sliding windows on the next image.
thresh = 5000
if len(llane.allx) < thresh:
    llane.detected = False
if len(rlane.allx) < thresh:
    rlane.detected = False
```

The new calculated curve coefficients are smoothed by weighting the new coefficients by p and the previous 3 coefficients by 1-p and the adding them. This is only applicable to lanes detected using previous curves. The windows method is used to correct an incorrectly detected lane, so it's not desirable to have influence from previous curves in this case. 

```
#Update the current coefficients
p = 0.7
if len(llane.best_fit) > 1:
    llane.current_fit = (p*left_fit) + ((1-p)*(sum(llane.best_fit)/len(llane.best_fit)))
    rlane.current_fit = (p*right_fit) + ((1-p)*(sum(rlane.best_fit)/len(rlane.best_fit)))
else:
    llane.current_fit = left_fit
    rlane.current_fit = right_fit
```

## Radius of Curvature

The pixel coordinates are converted to meters using the assumption that the lanes are 3.7m apart and are 30m long. The radius of curvature is then calculated using the polynomial coefficients in the following equation:

Rcurve = ((1 + (2A + B)^2)^1.5) / np.absolute(2A)

```
#Calculate the radii of curvature in meters
l_rad = (((1 + (2*real_left_coeffs[0]*y_eval*ym_per_pix + real_left_coeffs[1])**2)**1.5)
         / np.absolute(2*real_left_coeffs[0]))
r_rad = (((1 + (2*real_right_coeffs[0]*y_eval*ym_per_pix + real_right_coeffs[1])**2)**1.5)
         / np.absolute(2*real_right_coeffs[0]))
```

The distance of the car left of the center of both lanes is also calculated.

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

![Please watch project_output.mp4][video1]

## Challenge Video

The pipeline above is created for the project video. The challenge video is not required for project submission, but nevertheless provides an interesting insight into the limitations and strengths of the pipeline. The images below are from the challenge video.

The challenge video works quite well with the pipeline, but has a few areas for improvement, such as the first black mark on the far right and the shadow under the bridge.

![Please watch challenge_output.mp4][video2]

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

The left lane windows follow the yellow lane as it has the most detected pixels in each window. Due to the reduced depth of source points, it does not detect the barrier on the left.

The right lane windows favor the black mark as it has a higher pixel count than the dashed road lane. A disadvantage to ORing the gradient binary image with the color binary image.

![alt text][image18]

The final image clearly shows a correctly implemented left lane but incorrect right lane.

![alt text][image19]

## Harder Challenge Video

The pipeline above is created for the project video. The harder challenge video is not required for project submission, but nevertheless provides an interesting insight into the limitations and strengths of the pipeline. The images below are from the harder challenge video.

The pipeline struggles to provide a good result with the harder challenge video. The tight road bends suggested that a very wide or dynamic ROI is required for the perspective transform.

![Please watch harder_challenge_output.mp4][video3]

The gradient binary images are not affected by the shadows and excellently detect both lanes. The trees, however, are clearly a problem.

![alt text][image20]

The H channel detects the left lane and S channel the right lane. THe L channel thresholds can also be changed to detect the right lane, but this adds no value. The combination, dominated by the S channel, fully detects the right lane and partially the left lane (not the shadowed portion).

![alt text][image21]

The combination image has correctly detected the lanes, but the noise is concerning, especially just outside the right lane.

![alt text][image22]

Here it is possible to see that the ROI is slightly problematic. In later images when the road turns sharply, the perspective image only picks up a small portion of lane line with much added noise. If the width of the ROI is increased, the tree and ground coverage cause havoc in the warped binary image. This is somewhat reduced by the noise reduction method discussed earlier.

![alt text][image23]

The histogram is very close to correct, and the margin around the previous curve should pick up the lane base.

![alt text][image24]

The windows quickly detect the lane lines. The left lane correclty identifies the entire lane line through the shadows. The right lane line gets drawn to the right by the noise in the top right corner.

![alt text][image25]

The final image shows a good detection of the lanes.

![alt text][image26]

## Future Improvements

It is evident that creating a pipeline using these computer vision methods to satisfy all road conditions is close to impossible. The value of machine learning to learn road conditions and generalize a model is probably the best way to make an solution that can be used in self driving cars. Deep Learning is the answer.