# computer-vision-sift-knn-RANSAC

read x images target sample  
↓  
Sift  
↓  
KNN  
↓  
RANSAC  
↓  
Receive a 8*9 matrix A  
At*A  
get eigenvalue and eigenvector(9 dimension) resize to 3*3 matrix  
↓  
warping A.inv()*[x y 1]  

![image](https://raw.githubusercontent.com/frank83413/pattern-recognition/master/img/output.jpg)  
![image](https://raw.githubusercontent.com/frank83413/pattern-recognition/master/img/output2.jpg)  
