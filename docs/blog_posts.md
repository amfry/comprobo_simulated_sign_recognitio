---
layout: "default"
title: blog_posts 
---
## Project Update (xx/xx/xxxx)
## Project Update (11/23/2020)
Phase 1 of our project consisted of two major parts, object localization from video and creating a custom gazebo world.  We identified that we hadn't interfaced these two topics before and that they were both critical parts of the project. In moving our sign recognition from static images to real-time, we needed to begin developing object localization from a video and understanding the challenge there. Creating a custom gazebo world is needed so that our simulated gazebo has road signs from our chosen data set that it can respond to. To tackle these challenges, we took an asynchronous approach and you can read more about the progress in phase 1 below.
### Abby
To get experince quickly with different object localization strategies, I started by using my computers webcam and experimenting with different methods that came up during my research. Many implementation and papers on the topic of real-time video processing are focused on object-detection. However, for our project we want to use the simple CNN we developed during the computer vision project so that we only want to localize the object and then feed that localized image into the a classifier. To determine if an object in the simulated world is a road sign orthe rest of the gazebo simulated world we could use a variety of different features.
 * Color:This is one of the simpler methods I found. It involes apply a mask to the live video that narrows the range of colors in the processed video. For applications were the object being detected are all the same color and are differently colored than the background this seems like a good option. Though our gazebo simulation will likely be mostly gray and our signs brighly colored, we don't want to rely on the color mask as this solution doesn't have very much viability in real world applications.
 * Size: 
 * Approximated Shape: The road signs in our choosen dataset are all traingles, rectangles, and circles. 
![gif](https://github.com/amfry/sign_recognition/blob/main/docs/images/openCV_practice.gif)
 




### Vienna


