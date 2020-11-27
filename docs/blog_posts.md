---
layout: "default"
title: blog_posts 
---
## Project Update (xx/xx/xxxx)
## Project Update (11/23/2020)
Phase 1 of our project consisted of two major parts, object localization from video and creating a custom gazebo world.  We identified that we hadn't interfaced these two topics before and that they were both critical parts of the project. In moving our sign recognition from static images to real-time, we needed to begin developing object localization from a video and understanding the challenge there. Creating a custom gazebo world is needed so that our simulated gazebo has road signs from our chosen data set that it can respond to. To tackle these challenges, we took an asynchronous approach and you can read more about the progress in phase 1 below.
### Object Localization - Abby
To get experience quickly with different object localization strategies, I started by using my computer's webcam and experimenting with different methods that came up during my research. Many implementations and papers on the topic of real-time video processing are focused on object-detection. However, for our project, we want to use the simple CNN we developed during the computer vision project so that we only want to localize the object and then feed that localized image into a classifier. To determine if an object in the simulated world is a road sign or part of the gazebo simulated world we may need to use a variety of different features.
 * Color: This is one of the simpler methods I found. It involves applying a mask to the live video that narrows the range of colors in the processed video. For applications where the object being detected are all the same color and are differently colored than the background, this seems like a good option. Though our gazebo simulation will likely be mostly gray and our signs brightly colored, we don't want to rely on the color mask as this solution doesn't have very much viability in real-world applications.
 * Size: In our simulated world, we perceive that it may be necessary to distinguish between a road sign immediately coming up and one down the road. Distinguishing the area of the sign-like object maybe a good way to distinguish between two signs at different distances on the same stretch of road. At this point, it is unclear how populated our gazebo world will be and there may be many objects the same size at the road signs that the neato may try to localize to. This method is a viable candidate but has some limitation to how complex of a world it can navigate.
 * Approximated Shape: The road signs in our chosen dataset are all triangles, rectangles, and circles which makes our dataset a good candidate for using shape detection. However, similar to the size-dependent implementation, there complexity of shapes in the world around the neato may cause it to localize to objects other than the road sign.
 
![gif](https://github.com/amfry/sign_recognition/blob/main/docs/images/openCV_practice.gif)  
*Fig. 1: Here is an example of shape detection were only simplified rectangles with areas greater than 50 pixels are being identified. The value next to each bounding box is the approximated area of the rectangle. The post-it is frequently bnot eing detected as an independt rectangle but rather with other objects in the frame  

In addition to continuing to implement object localization algorithms, I would also like to implement some speed improvements to the localization. A strategy like skipping to every nth frame seems like a simple and effective optimization but I plan on looking into other strategies as well.




### Custom Gazebo World - Vienna
I have been working on creating custom road sign models for our Gazebo world. The ultimate goal is to make a Gazebo world with at least two types of signs based on the Chinese Road Sign dataset that the Neato can recognize and respond to using the CNN image classifier from our Computer Vision project.

I made a model in Solidworks based on a triangular yellow sign and exported it using the URDF exporter Solidworks add-on. I found that this only exports the mesh as .STL files without the image overlay, so I am currently trying to figure out how to use diffuse and spec images to add the image to the URDF model.

Here is the model in Solidworks:

Some ideas/concerns I've been considering:
* How does image brightness impact the accuracy of the CNN? How does this impact the lighting I program into each model?
* Can the Neato zone in on one sign in its field of vision to respond to? This is what Abbys working on with OpenCV, and for now we are optimistic that the Neato will be able to at least pick the closest (and therefore largest) sign in sight.

![img](https://github.com/amfry/sign_recognition/blob/main/docs/images/yield_sign_sldprt.PNG)  
