# MediaPipe-pose-estimation-using-intel-realsense-debth-camera
Human activity monitoring via pose estimation has a lot of real-world applications. Real-time pose estimation can be used for solving many problems in fields like fitness training, sports coaching, gaming, motion capture, assisted living, and Robotics. Our focus was on utilizing human pose estimation and classification for Robotics perception. I use MediaPipe which is Google's open-source framework that estimates the human pose and Intel RealSense D435 depth camera for getting depth frame. The focus of this software is to identify some poses to our quadruped robot. To achieve this, we reviewed different pose estimation models, aligning depth frame with RGB frame, using extracted human body joints and their depth value for posture classification. the time and depth differences help us to the determination of velocity and acceleration.

![6](https://user-images.githubusercontent.com/83344958/151194012-ac4dfc67-36ae-4eba-9531-0aec9c0cf114.png)

#3D plot of Pose detection using depth frame

![9](https://user-images.githubusercontent.com/83344958/151190981-664ae63f-301c-480b-a999-3c1c639c702f.PNG)

With a combination of two or three poses, we can create movements. Therefore some concepts like velocity stand out helping classify our orders based on movements velocity. Currently, we have a movement including three poses (Hey, Ready, Go). So thanks to the velocity of the wrist from the Hey pose and Go pose or time of the Ready pose, we divide the movement into the GO FAST and GO SLOW .

![16](https://user-images.githubusercontent.com/83344958/151191200-40db769d-5046-4783-81cc-497a7c343f11.PNG)

![17](https://user-images.githubusercontent.com/83344958/151191222-41b2b49a-5376-4f29-a971-5c43d521da12.PNG)

######you can find more information in main.pdf
