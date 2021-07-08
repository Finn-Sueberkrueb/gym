# RoboSkate environments with different observation spaces.

Since there are only small differences, a common class would be better from which the specific classes are derived.
However, as many changes are still being made, they are still separate classes.

## RoboSkate Numerical
RoboSkate without image data. An optimal trajectory is given and one input is the steering angle that leads to the optimal trajectory in the current position and orientation.

## RoboSkate Segmentation
RoboSkate with image data. The image data is pre-processed by a trained CNN and only the 8 Latentspace features are combined with the numerical observations.\
The CNN will not be trained here!

## RoboSkate Multi Input Policy
Here, a custom multi input policy is used which directly uses the image data and the numerical observations as observations. The CNN part will also be trained.\
For the Multi Input Policy, the RoboSkateCombinedFeaturesExtractor.py is defined in the Stable Baseline repository under /common. 
