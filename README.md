# computer-vision-sift-knn-RANSAC

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

![image](https://raw.githubusercontent.com/frank83413/computer-vision-sift-knn-RANSAC/master/img/output.jpg)  
![image](https://raw.githubusercontent.com/frank83413/computer-vision-sift-knn-RANSAC/master/img/output2.jpg)  
