# EECE-5554-FinalProject

Contributors:

Veera Ragav for feature detection adn matching with outlier rejection,  
Askash for Essential matrix and camera pose  
Arvind for triangulation  
Haojie Huang for solvePnP and bundle adjustment  

Usages:  

data structure:(the code and the image datasets should be under the same fold)  
               \sfm_code.py  
               \templeSparseRing(it contains 16 images of a temple)  
  
Prerequisite:(python3.5 and pip)  
pip install opencv-python==3.4.2.16  
pip install opencv-contrib-python==3.4.2.16  
pip install numpy  
pip install open3d (OR pip3 install open3d-python)  
pip install scipy  

During use:  
It will plot the 3d structure from the first two images at the beginning.  
After that you could enter the image number for sfm, press the enter to use the default number==4 (our code works well when the images   
number is samller than 6, otherwise it will give some noisy output due to that we used a very dense feature detection)  
