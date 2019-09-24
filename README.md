# computer-vision-sift-knn-RANSAC

Introduction: Play the jigsaw puzzle through the algorithm.

Read x images target sample.  
↓  
Sift  
↓  
KNN  
↓  
RANSAC  
↓  
Receive a 8\*9 matrix A  
A<sup>t</sup>\*A  
Get eigenvalues and eigenvectors (9-dimension) and resize to 3\*3 matrix.  
↓  
warping A.inv()\*[x,y,1]<sup>t</sup>  

Mainly implement KNN, RANSAC and warping.

![image](https://raw.githubusercontent.com/frank83413/computer-vision-sift-knn-RANSAC/master/img/output.jpg)  
![image](https://raw.githubusercontent.com/frank83413/computer-vision-sift-knn-RANSAC/master/img/output2.jpg)  

opencv document

https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html?highlight=findhomography
