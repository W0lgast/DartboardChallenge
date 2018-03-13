// opencv.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <math.h>
#include <utility> 
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define M_PI 3.14159

pair<Mat, vector<pair<Point, Point>>> detectAndDisplay(Mat frame);

/** Global variables */
String cascade_name = "cascade.xml";
CascadeClassifier cascade;

int main(int argc, const char** argv)
{
	// 1. Read Input Image
	Mat img = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	string fileName = argv[1];

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if (!cascade.load(cascade_name)) { printf("--(!)Error loading\n"); return -1; };

	// 3. Detect Frames Poitentially Containing Dartboards
	pair<Mat, vector<pair<Point, Point>>> cascadeInfo = detectAndDisplay(img);
	vector<pair<Point, Point>> detectedDartboards = cascadeInfo.second;

	// 4. Here we output the number of detected dartboards in the frame, an create and save the image with detected dartboards
	cout << fileName + ": I've found " << detectedDartboards.size() << " dartboards in the image " << endl;
	cout << "Saving image with detections as Task2DetectedDartboards" + fileName << endl;
	if (detectedDartboards.size() > 0) {
		for (int i = 0; i < detectedDartboards.size(); i++) {
			rectangle(img, detectedDartboards[i].first, detectedDartboards[i].second, Scalar(255, 0, 255), 3, 8, 0);
		}
	}
	imwrite("Task2DetectedDartboards" + fileName, img);	
	
	detectedDartboards.clear();

	return 0;
	}


pair<Mat, vector<pair<Point, Point>>> detectAndDisplay(Mat frame)
{
	std::vector<Rect> dartboards;
	Mat frame_gray;

	Mat detectedFaces = frame.clone();
	vector<pair<Point,Point>> dartboardFrames;
	pair<Point, Point> dartboardFrame;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	// 2. Perform Viola-Jones Object Detection 
	cascade.detectMultiScale(frame_gray, dartboards, 1.1, 1, 0 | CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500, 500));

	// 3. Draw box around dartboards found
	for (int i = 0; i < dartboards.size(); i++)
	{
		int frameLength = max(dartboards[i].width, dartboards[i].height); //this line makes sure all frames are square
		dartboardFrame = make_pair(Point(dartboards[i].x, dartboards[i].y), Point(dartboards[i].x + dartboards[i].width, dartboards[i].y + dartboards[i].height));
		dartboardFrames.push_back(dartboardFrame);
		cv::rectangle(detectedFaces, dartboardFrame.first, dartboardFrame.second, Scalar(0, 255, 0), 2);
	}

	return(make_pair(detectedFaces, dartboardFrames));

}
