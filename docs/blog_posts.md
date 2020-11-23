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
 * Size: In our simulated world, we percieve that it may be necessary to distinguish between a road sign imeediatley coming up and one down the road. Distinguishing the area the sign-like object needs to be may be a good way to distinguish between two signs at different distnaces on the same strech of road. At this point it is unclear how populated our gazebo world will be and it is possible there will be many object the same size at the road signs that the neato may try to localize to. This method is a viable canidate but has some limitation to how complex of a world it can naviage.
 * Approximated Shape: The road signs in our choosen dataset are all traingles, rectangles, and circles which makes our dataset a good canidate for using shape detection. However, similiar to the size dependent implemenation, there complexity of shapes in the world around the neato may cause it to localize to objects other than the road sign.  
 
![gif](https://github.com/amfry/sign_recognition/blob/main/docs/images/openCV_practice.gif)  
In addition to continuing to implemnet object localization algorithims, I would also like to implement some speed improvemnts to the localization. A strategy like skipping to every nth frame seems like a simple and effective optimization but I plan on looking into other strategies as well.
 




### Vienna


