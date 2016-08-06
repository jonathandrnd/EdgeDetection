// OpenCV.cpp: define el punto de entrada de la aplicación de consola.
//

#include "stdafx.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

Mat src, grey;
int thresh = 80;

const char* windowName = "Contours";

void detectContours(int,void*);

int main(){
	//carga la imagen en escala de grises
	src = cvLoadImage("C:\\Users\\Lenovo\\Desktop\\MAESTRIA\\lenna.jpg", CV_LOAD_IMAGE_GRAYSCALE);

    //Display the original image
    namedWindow("Original", WINDOW_AUTOSIZE);
    imshow("Original",src);
	//Create the trackbar
    cv::createTrackbar("Thresholding",windowName,&thresh,255,detectContours);
    detectContours(0,0);
    waitKey(0);
	return 0;

}

void detectContours(int,void*){
    Mat canny_output;

    //Detect edges using canny
    cv::Canny(src,canny_output,thresh,2*thresh);
	namedWindow("Canny",WINDOW_AUTOSIZE);
    imshow("Canny",canny_output);
	
}

