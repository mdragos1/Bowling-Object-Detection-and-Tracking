# Computer Vision Project - Video analysis of bowling

## Objective

The goal of this project is to develop an automatic system for video analysis for the game of
bowling. The system should be able to: (i) track the bowling ball that rolls down a synthetic
lane with the goal of knocking down all the pins; (ii) detect the remaining standing pins.

## Description of the bowling game1

Ten-pin bowling is a type of bowling in which a bowler rolls a bowling ball down a wood
or synthetic lane toward ten pins positioned evenly in four rows in an equilateral triangle
at the far end of the lane (Figure 1). The objective is to knock down all ten pins on the first
roll of the ball (a strike), or failing that, on the second roll (a spare).

## Data and Task descriptions

The data directory contains three directories: train, test and evaluation. The directories train
and test have similar structure, although the test data will be made available after the deadline.
The train directory contains data organized in three subdirectories corresponding to the three tasks
that you need to solve. The subdirectories are:\
   - Task1 - this directory contains 25 training images showing the remaining standing
pins on the synthetic lane after the bowling has rolled down on it in a constrained
scenario. In this scenario, each image shows the remaining standing pins from above
the lane, with a static camera capturing the images. The data is collected from several
lanes using the corresponding static cameras. We include in the Task1 directory the
full configurations of the ten pins for four of the cameras corresponding to the lanes
(see the full-configurations-templates). These images correspond to the arrangement of
the ten pins before the player tries to knock down all the pins.
The Task 1 consists in correctly classifying the positions of some pins provided into
the query file (see Figure 2). The format that you need to follow is the one used in
the annotation files with the first line containing the total number of positions that
need to be classified and starting with the second line the positions that need to be
classified (0 -empty, 1-occupied).\
   - Task2 - this directory contains 15 training videos in the scenario showing the bowling
ball rolling down the lane and hitting the pins. The task is to track the bowling ball
released by the player. You should track the bowling ball from the initial frame to the
final frame of the video (see Figure 3). The initial bounding box of the bowling ball
to be tracked is provided for the first frame (the annotation follows the format [xmin
ymin xmax ymax], where (xmin,ymin) is the top left corner and (xmax,ymax) is the
bottom right corner of the initial bounding-box).\
In each video we will consider that your algorithm correctly tracks the bowling ball if in
more (greater or equal) than 80% of the video frames your algorithm correctly localizes
the bowling ball to be tracked. We consider that your algorithm correctly localizes the 
bowling ball to be tracked in a specific frame if the value of the IOU (intersection over union)
between the window provided by your algorithm and the ground-truth window is
more than 30%.
