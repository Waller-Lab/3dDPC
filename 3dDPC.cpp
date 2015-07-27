/*
Refocusing.cpp
5/5/2015
This is a C++ program for digitally refocusing a set of images taken
with different single LEDs using the CellScope with a dome of LEDs.
Compile with ./compile.sh
*/
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include <stdio.h>
#include <dirent.h>

#include "omp.h"
/* Tried to integrate OpenMP, doesn't really work since OpenCV has a lot
of critical sections. */

#define FILENAME_LENGTH 32
#define FILE_HOLENUM_DIGITS 3
using namespace std;
using namespace cv;

#include "domeHoleCoordinates.h"

/* Produces a set of images from zMin to zMax, incrementing by zStep.
All measured in microns. */
int zMin, zMax, zStep;
string datasetRoot, outputDir;//folder of input images

class R_image{
  
  public:
        cv::Mat Image;
        int led_num;
        float tan_x;
        float tan_y;
};

void circularShift(Mat img, Mat result, int x, int y){
    int w = img.cols;
    int h  = img.rows;

    int shiftR = x % w;
    int shiftD = y % h;
    
    if (shiftR < 0)//if want to shift in -x direction
        shiftR += w;
    
    if (shiftD < 0)//if want to shift in -y direction
        shiftD += h;

    cv::Rect gate1(0, 0, w-shiftR, h-shiftD);//rect(x, y, width, height)
    cv::Rect out1(shiftR, shiftD, w-shiftR, h-shiftD);
    
	  cv::Rect gate2(w-shiftR, 0, shiftR, h-shiftD); 
	  cv::Rect out2(0, shiftD, shiftR, h-shiftD);
    
	  cv::Rect gate3(0, h-shiftD, w-shiftR, shiftD);
	  cv::Rect out3(shiftR, 0, w-shiftR, shiftD);
    
	  cv::Rect gate4(w-shiftR, h-shiftD, shiftR, shiftD);
	  cv::Rect out4(0, 0, shiftR, shiftD);
   
    cv::Mat shift1 = img ( gate1 );
    cv::Mat shift2 = img ( gate2 );
    cv::Mat shift3 = img ( gate3 );
    cv::Mat shift4 = img ( gate4 );

	  shift1.copyTo(cv::Mat(result, out1));//copyTo will fail if any rect dimension is 0
	  if(shiftR != 0)
        shift2.copyTo(cv::Mat(result, out2));
	  if(shiftD != 0)
    	  shift3.copyTo(cv::Mat(result, out3));
	  if(shiftD != 0 && shiftR != 0)
    	  shift4.copyTo(cv::Mat(result, out4));

}

int loadImages(string datasetRoot, vector<R_image> *images) {
	DIR *dir;
	struct dirent *ent;
	if ((dir = opendir (datasetRoot.c_str())) != NULL) {
      int num_images = 0;
      cout << "Loading Images..." << endl;
	    while ((ent = readdir (dir)) != NULL) {
		  //add ent to list
    		  string fileName = ent->d_name;
    		  string filePrefix = "_scanning_";
          /* Get data from file name, if name is right format.
          such as 20141015_184917999_scanning_159.jpeg */
    		  if (fileName.compare(".") != 0 && fileName.compare("..") != 0) {
    		      string holeNum = fileName.substr(fileName.find(filePrefix)+filePrefix.length(),FILE_HOLENUM_DIGITS);
              R_image currentImage;
    		      currentImage.led_num = atoi(holeNum.c_str());
    		      currentImage.Image = imread(datasetRoot + "/" + fileName, -1);
          		currentImage.tan_x = domeCoordinates[currentImage.led_num][0] / domeCoordinates[currentImage.led_num][2];
          		currentImage.tan_y = domeCoordinates[currentImage.led_num][1] / domeCoordinates[currentImage.led_num][2];
          		(*images).push_back(currentImage);
          		num_images ++;
    		  }
	    }
	  closedir (dir);
	  return num_images;

	} else {
	  /* could not open directory */
	  perror ("");
	  return EXIT_FAILURE;
	}
}

void computeFocusDPC(vector<R_image> iStack, int fileCount, float z, int width, int height, int xcrop, int ycrop, Mat* results)
{
	int newWidth = width;// - 2*xcrop;
	int newHeight = height;// - 2*ycrop;

	cv::Mat bf_result = cv::Mat(newHeight, newWidth, CV_16UC3, double(0));
	cv::Mat dpc_result_tb = cv::Mat(newHeight, newWidth, CV_16SC1,double(0));
	cv::Mat dpc_result_lr = cv::Mat(newHeight, newWidth, CV_16SC1,double(0));

	cv::Mat bf_result8 = cv::Mat(newHeight, newWidth, CV_8UC3);
	cv::Mat dpc_result_tb8 = cv::Mat(newHeight, newWidth, CV_8UC1);
	cv::Mat dpc_result_lr8 = cv::Mat(newHeight, newWidth, CV_8UC1);

	cv::Mat img;
	cv::Mat img16;
	cv::Mat shifted = cv::Mat(iStack[0].Image.rows, iStack[0].Image.cols, CV_16UC3,double(0));
	vector<Mat> channels(3);
	for (int idx = 0; idx < fileCount; idx++)
	{
		// Load image, convert to 16 bit grayscale image
		img = iStack[idx].Image;

		// Get home number
		int holeNum = iStack[idx].led_num;

		// Calculate shift based on array coordinates and desired z-distance
		int xShift = (int) round(z*iStack[idx].tan_x);
		int yShift = (int) round(z*iStack[idx].tan_y);

		// Shift the Image in x and y
		circularShift(img, shifted, yShift, xShift);

		// Add Brightfield image
		cv::add(bf_result, shifted, bf_result);

		// Convert shifted to b/w for DPC
		split(shifted, channels);
		channels[1].convertTo(channels[1],dpc_result_lr.type());

		if (std::find(std::begin(leftList),std::end(leftList),holeNum) != std::end(leftList)){
			cv::add(dpc_result_lr, channels[1], dpc_result_lr);
		}
		else
			cv::subtract(dpc_result_lr, channels[1], dpc_result_lr);

		if (std::find(std::begin(topList),std::end(topList),holeNum) != std::end(topList)) 
			cv::add(dpc_result_tb, channels[1], dpc_result_tb);
		else
			cv::subtract(dpc_result_tb, channels[1], dpc_result_tb);
		
		//float progress = 100*((idx+1) / (float)fileCount);
		//cout << progress << endl;
	}
		  
	// Scale the values to 8-bit images
	double min_1, max_1, min_2, max_2, min_3, max_3;

	cv::minMaxLoc(bf_result, &min_1, &max_1);
	bf_result.convertTo(bf_result8, CV_8UC4, 255/(max_1 - min_1), - min_1 * 255.0/(max_1 - min_1));

	cv::minMaxLoc(dpc_result_lr.reshape(1), &min_2, &max_2);
	dpc_result_lr.convertTo(dpc_result_lr8, CV_8UC4, 255/(max_2 - min_2), -min_2 * 255.0/(max_2 - min_2));

	cv::minMaxLoc(dpc_result_tb.reshape(1), &min_3, &max_3);
	dpc_result_tb.convertTo(dpc_result_tb8, CV_8UC4, 255/(max_3 - min_3), -min_3 * 255.0/(max_3 - min_3));
		  
	results[0] = bf_result8;
	results[1] = dpc_result_lr8;
	results[2] = dpc_result_tb8;
}
/* Compile with ./compile.sh */
int main(int argc, char** argv )
{

	if (argc < 6) {
      cout << "Error: Not enough inputs.\nUSAGE: ./3dDPC zMin zStep zMax DatasetRoot outputDir (units um)" << endl;
      return 0;
	} else {
		zMin = atoi(argv[1]);
		zStep = atoi(argv[2]);
		zMax = atoi(argv[3]);
		datasetRoot = argv[4];
		outputDir = argv[5];
	}

	std::cout << "zMin: " << zMin << std::endl;
	std::cout << "zStep: " << zStep << std::endl;
	std::cout << "zMax: " << zMax << std::endl;
	std::cout << "DatasetRoot: " << datasetRoot << std::endl;
	std::cout << "OutputDir: " << outputDir << std::endl;

	vector<R_image> * imageStack;
	imageStack = new vector<R_image>;
	int16_t imgCount = loadImages(datasetRoot,imageStack);
	cout << "Processing Refocusing..."<<endl;

	Mat results[3];

	for (int zDist = zMin; zDist <= zMax; zDist += zStep) {
		cout << "Processing: " << zDist << "um..." <<endl;
		computeFocusDPC(*imageStack, imgCount, zDist, imageStack->at(0).Image.cols, imageStack->at(0).Image.rows, 0, 0, results);

		char bfFilename[FILENAME_LENGTH];
		char dpcLRFilename[FILENAME_LENGTH];
		char dpcTBFilename[FILENAME_LENGTH];
		snprintf(bfFilename,sizeof(bfFilename), "%s/BF_%.03d.png",outputDir.c_str(),zDist);
		snprintf(dpcLRFilename,sizeof(dpcLRFilename), "%s/DPCLR_%.02d.png",outputDir.c_str(),zDist);
		snprintf(dpcTBFilename,sizeof(dpcTBFilename), "%s/DPCTB_%.02d.png",outputDir.c_str(),zDist);

		imwrite(bfFilename, results[0]);
		imwrite(dpcLRFilename, results[1]);
		imwrite(dpcTBFilename, results[2]);
		cout << "Finished!"<<endl;
	}	
	namedWindow("Result bf", WINDOW_NORMAL);
	imshow("Result bf", results[0]);

	namedWindow("Result lr", WINDOW_NORMAL);
	imshow("Result lr", results[1]);

	namedWindow("Result tb", WINDOW_NORMAL);
	imshow("Result tb", results[2]);

	waitKey(0);
} 

