## Advanced Lane Finding

---
**See the final product on youtube**

[Youtube Link To Advanced Lane Line Video](https://youtu.be/G2Ni5QpkCDs "Pipeline Output")


[![Advanced Lane Lines](http://img.youtube.com/vi/G2Ni5QpkCDs/0.jpg)](https://youtu.be/G2Ni5QpkCDs "Advanced Lane Lines")

---

[//]: # (Image References)

[image1]: ./output_images/chesboard_1.png "Chesboard 1"
[image2]: ./output_images/chesboard_2.png "Chesboard 2"
[image3]: ./output_images/binary1.png "Binary example 1"
[image4]: ./output_images/binary2.png "Binary example 2"
[image5]: ./output_images/highlighted_lines.jpng "Line Fit Original"
[image6]: ./output_images/loop_fit.png "Polynomial fit"
[image7]: ./output_images/reg.png "Original Image"
[image8]: ./output_images/undist.png "Undistorted Image"
[image9]: ./output_images/draw_test.png "Draw lane line test"
[image10]: ./output_images/final_test.png "test of the pipeline"
[image11]: ./output_images/warp1.png "Warped example 1"
[image12]: ./output_images/warp2.png "warped example 2"


**The goals / steps of this project are the following:**

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

**For a technical step-by-step explination, launch the Ipython notebook for this project at:**

[Ipy notebook](https://github.com/andrewsommerlot/Carnd-Advanced-Lane-Lines/blob/master/advanced_lane_lines.ipynb "Ipy notebook") 

## Compute the camera calibration matrix and distortion coefficients

This is not too difficult as most of the code was given in the lesson. I just printed the images to the console instead of the window and waiting thing. Additionally, CV2 provides a calibratecamera function, which I used along with the calculated objpoint, imgpoint, and the shape of the images. The main output of this function is a matrix I will use to undistort images next. Also it will be used to warp images which will greatly help when fitting lane lines. 

![Chesboard Corners 1][image1]

**Example of Chesboard Calibration image with chesboard corners highlighted.**



![Chesboard Corners 2][image2]

**Another Example of Chesboard Calibration image with chesboard corners highlighted.**


## Apply a distortion correction to raw images

The first step after calibration, is to use the calibration to undistort the images. This is important to get accurate measruemetns later on, even though in this particular application, it can be difficult to see the difference. The original image is on the left, and the undistorted on the right, you can see the slight difference when focusing on the road sign. Effectively this "flattens out" or "smooths out" the image so distances are more constant along the x and y axis. 

![Raw image][image7]

**Original image before any distortion corrections.**



![Undistorted Image][image8]

**The Same image after the distortion correction. The small differences can be seen in the objects in the foreground.**


## Use color transforms, gradients, etc., to create a thresholded binary image

Here I use HSL color space transofmations and the sobel operator to create binary (pixels on or off) images from the undistorted images. This is a very important set which greatly determines the effectiveness of the pipeline. It is analogous to a convolutional neural net learning features to identify the lane. This is of course, "by hand" selecting the features, which requires some though and trial and error. Additionally, there are tunable parameters here that can be changed to create different outcomes. I spend a lot of time here goofing around to find good thredholds. 

![Binary Example 1][image3]

**Example of resulting binary image from threshold process.**



![Binary Example 2][image4]

**Example of resulting binary image from threshold process with cars in the foreground.**




## Apply a perspective transform to rectify binary image ("birds-eye view")

Next I used the matricies from the calibation to create warped images that make a birds eye view of the road. This setp also has parameters which needed to be tuned, again, simply by trial and error. Additionaly, this set required designating the area which was warped, I did this by trial and error and also using gimp to get pixel locations. In my oppinion, this step can cause a lot of error if care is not take, I did this rather quickly. If this pipeine was to be implemented, I would defninately review the warp procudure. 


![Warped Example 1][image11]

**Example warped image, showing the lanelines from a 90 degree above view**



![Warped Example 1][image12]

**Another Example of a warped image**


## Detect lane pixels and polynomail fit to find the lane boundary

This step is almost entirely from the class notes. Also, this is not a function. Its worth noting that because this particular piece of code will not be in the final pipeline. I use this code to get a first fit from an example image. I then use that first fit to initilize the last_fits slot in the lane class I define later. Major things happening here are a histogram search of the bottom half of the image, and then an iterative window search that identifies pixels that will be considered part of the lane. The next piece of code was also provided by udacity and shows how this process performed on one of the example images. 


![Detecting lane pixels][image5]

**Lane pixels detected**



Next, I used more code provided in the class to create a polynomial fit function that will go into the pipeline. This function uses the previous polynomial fits to make the next polynomial fits. It is initialized once with the code from the section above and then loop through using the last fit for the next fit. That way, the window search only needs to be used once. There is one wrinkle to this, in that I ended up using the previous 200 fits with a weighted average, but, the poly function deals only with 2 fits at a time. The next piece of code shows how the function performs on an example image. 


![Polynomial fitting process which will be used in final pipeline][image6]

**Lane line polynomial function to be used in final pipeline**


## Determine the curvature of the lane and vehicle position with respect to center

The class notes provided some code and suggestions for calculating radius and center distance. These will later be plotted on to the final images. The function given in the class notes was used to calcualte radius of curvature for each lane line. I then rounded each to the nearest whole number and took the minimum of them to print out. the reason I did this was incase I ended up needing a "sanity check" on the radius then I wanted to use the potentially worst one for the given fit. I calculated center distance by taking the average pixel x position in the bottom 25% of the picture and finding the pixel distance to each side from the center of image, and then converting pixel distances to meters with a constant. There are a lot of assumptions wrapped up in there, but the reasult did not seem too bad. I calculated the following four numbers
* Left Raduis of Curvature 
* Right Radius of Curvature 
* Center Distace from Left Lane Line 
* Center Distance from Right Lane Line

After this, I wrote out the results in an annotation on the output image with cv2.putText. 


### Draw lane lines

Finally I'm drawing lane lines! This is the first time the fits will be projected back onto the road visually. I used the suggested code from the lesson to build the fucntion. Additionally, this fucntion takes the output of the  polynomial fits to write text in the top right corner displaying the calculations. I ran the funtion on a test image below. 


![predicted lane lines][image9]

**Projected lane lines onto undistorted image**



## Output visual display of the lane boundaries, curvature and lane postition

### Lane Class Definition

I did not use the structure suggested in the class notes, as my code as written to this point could not take advantage of all the slots. So, I paired it down to what I hoped would be enough to smoothing predict lane lines. Here, I store a nd array abject of n number of past polynomial fits. I used this later to calcuate a weighted average, with recent fits being more important than previous fits through a learning function calculating the weights. The n number flexibiity is important as I needed to tune this number to get good results.



### Saving polynomial fits and applying a weighted average

When I ran this pipeline "raw" or just displaying each fit as the came, the resutling video was not encuraging. In an effort to solve this problem I built a class object to save n number of previous fits, and caculate the weighted average of the fits (latest predictions beining most important). This greatly improved the result, but added more parameters which need to be tuned and likely have different optima depending on the state of the video (dark, light, raining, shadows, etc). The average fit function calculates the averge of the polynomial wieghts, and the lane step function keeps track of the last n polynomial weights. 


### Putting together the whole pipeline

Yes! Finally the whole pipeline can be put together. The pipeline utilizes the undistort, apply_threholds, warp, average_fit, poly_fit, calc_radius, draw_lane, and lane_step functions. Addidtionally, it adds a parameter dictating how many polynomial weights to save and average over. This is "hard-coded" as a few other things are, but it works fine for a research document like this. The whole pipline would have to be more robust to be acutally implemented. 


### Initilize and Test Pipeline

The pipeline requires an initilized object for right_lane and left_lane last_fits slots. I made this object with repeated fits for the first test image for the same n as is set in the pipeline. After making the object I ran the pipeline on a test image.


![predicted lane lines][image10]

**Output of final pipeline on test image**


    
## Video output 

The final pipeline was then applied to the project video. I uploaded the video to youtube, you can see it by clicking the link below or clicking the image. 


[Youtube Link To Advanced Lane Line Video](https://youtu.be/G2Ni5QpkCDs "Pipeline Output")


[![Advanced Lane Lines](http://img.youtube.com/vi/G2Ni5QpkCDs/0.jpg)](https://youtu.be/G2Ni5QpkCDs "Advanced Lane Lines")



## Discussion and Conclusion

Wow, that was quite a project, one which could really be upgraded a lot and have much better performance. It was interesting to get a taste of what "doing it by hand" takes rather than using deep learning. Although this project sucessfully tracked the lane lines with reasonable accuracy, there is definately room for improvement. The biggest improvement from single fitting I made was saving a specified number of polynomial fits from each side and applying a wiehgted average to each of the three polynomial weights. The wieghted average took the most recent fits as more important than previous ones. Although this improved the pipeline, it is still not perfect and there are times when the polyfit can be seen wandering a bit. 

One interesting challange was everytime I implemented more code to solve a problem, the process introduced more tunable parameters. Although these processes and parameters made the output much better, they took deep knowledge of the process or just a bunch of trial and error to tune. Also, some of these parameters had interdependence, where the optimal values for some change when values are changed for others. Since this is a pretty hetogenous process, even a score or so parameters can start to be a hassle when coding by hand. 

I'm pretty convienced that my process is not operating at its optimal and more parameter tuning could really help. One way to do this in the future would be to use a metaheuristic optimization such as particle swarm or a genetic algorithm. These types of optimizations are valuable when the gradient of the problem is difficult or impossible to esitmate and the parameters are very heterogenous. It would be cool to see how much better the pipeline could get by tuning before making it more complex and thus slower. 

 
