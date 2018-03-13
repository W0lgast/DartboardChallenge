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
vector<pair<String, vector<pair<Point, Point>>>> GetFileInfo();
tuple<Mat, int, int, int> CalculateAccuracy(vector<pair<Point, Point>> trueFaceRects, vector<pair<Point, Point>> detectedFaceFrames, Mat img, String fileName, float hitThresh);

/** Global variables */
String cascade_name = "frontalface.xml";
CascadeClassifier cascade;

int main()
{
	float TPR, F1;
	int TP = 0; int FP = 0; int FN = 0;

	vector<pair<String, vector<pair<Point, Point>>>> fileInfo = GetFileInfo();

	for (int pic = 0; pic < fileInfo.size(); pic++) {

		// 1. Read Input Image
		string fileName = fileInfo[pic].first;
		Mat img = imread(fileName);

		// 2. Load the Strong Classifier in a structure called `Cascade'
		if (!cascade.load(cascade_name)) { printf("--(!)Error loading\n"); return -1; };

		// 3. Detect Frames Poitentially Containing Dartboards using VJ
		pair<Mat, vector<pair<Point, Point>>> cascadeInfo = detectAndDisplay(img);
		vector<pair<Point, Point>> detectedFaces = cascadeInfo.second;


		/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
		// THE FOLLOWING CODE COMPARES ACCURACY OF DETECTION FRAMES TO GROUND TRUTH FRAMES

		// 1. Load ground truth frames from user generated data
		vector<pair<Point, Point>> trueFaceRects = fileInfo[pic].second;

		// 2. CalculateAccuracy calclates number of True Positives, False Positives, and False Negatives in our detections, it also generates an accuracyDepiction figure
		tuple<Mat, int, int, int> accuracyInfo = CalculateAccuracy(trueFaceRects, detectedFaces, img, fileName, 0.125);
		Mat accuracyDepiction = get<0>(accuracyInfo);

		// 3. Save our accuracyDepiction image, and add number of TPs, FPs and FNs to totals.
		imwrite("Task1DetectedFaces" + fileName, accuracyDepiction);
		TP += get<1>(accuracyInfo);	
		FP += get<2>(accuracyInfo);
		FN += get<3>(accuracyInfo);

		/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

		detectedFaces.clear();
	}

	// Here we output total number of True Positives, False Positives, and False Negatives in our detections, as well as the calculated TPR and F1-Score for out detector
	cout << "" << endl;
	cout << "There are " << TP << " True Positives in total" << endl;
	cout << "There are " << FP << " False Positives in total" << endl;
	cout << "There are " << FN << " False Negatives in total" << endl;

	TPR = float(TP) / float(TP + FN);
	F1 = (2.*float(TP)) / (2.*float(TP) + float(FP + FN));

	cout << "" << endl;
	cout << "The TPR = " << TPR << endl;
	cout << "The F1 Score = " << F1 << endl;
	cout << "" << endl;

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

vector<pair<String, vector<pair<Point, Point>>>> GetFileInfo() {

	vector<pair<String, vector<pair<Point, Point>>>> fileInfo;

	//dart0
	String fileName = "dart0.jpg";
	vector<pair<Point, Point>> faceRects;
	fileInfo.push_back(make_pair(fileName, faceRects));
	faceRects.clear();

	//dart1
	fileName = "dart1.jpg";
	fileInfo.push_back(make_pair(fileName, faceRects));
	faceRects.clear();

	//dart2
	fileName = "dart2.jpg";
	fileInfo.push_back(make_pair(fileName, faceRects));
	faceRects.clear();

	//dart3
	fileName = "dart3.jpg";
	fileInfo.push_back(make_pair(fileName, faceRects));
	faceRects.clear();

	//dart4
	fileName = "dart4.jpg";
	faceRects.push_back(make_pair(Point(331, 115), Point(478, 265)));
	fileInfo.push_back(make_pair(fileName, faceRects));
	faceRects.clear();

	//dart5
	fileName = "dart5.jpg";
	faceRects.push_back(make_pair(Point(59, 141), Point(122, 196)));
	faceRects.push_back(make_pair(Point(246, 170), Point(306, 224)));
	faceRects.push_back(make_pair(Point(188, 221), Point(252, 280)));
	faceRects.push_back(make_pair(Point(50, 255), Point(119, 318)));
	faceRects.push_back(make_pair(Point(291, 250), Point(347, 308)));
	faceRects.push_back(make_pair(Point(373, 195), Point(445, 251)));
	faceRects.push_back(make_pair(Point(424, 241), Point(484, 299)));
	faceRects.push_back(make_pair(Point(512, 186), Point(576, 237)));
	faceRects.push_back(make_pair(Point(560, 257), Point(619, 312)));
	faceRects.push_back(make_pair(Point(644, 192), Point(707, 245)));
	faceRects.push_back(make_pair(Point(680, 256), Point(732, 312)));
	fileInfo.push_back(make_pair(fileName, faceRects));
	faceRects.clear();

	//dart6
	fileName = "dart6.jpg";
	faceRects.push_back(make_pair(Point(289, 121), Point(324, 157)));
	fileInfo.push_back(make_pair(fileName, faceRects));
	faceRects.clear();

	//dart7
	fileName = "dart7.jpg";
	faceRects.push_back(make_pair(Point(346, 200), Point(423, 281)));
	fileInfo.push_back(make_pair(fileName, faceRects));
	faceRects.clear();

	//dart8
	fileName = "dart8.jpg";
	fileInfo.push_back(make_pair(fileName, faceRects));
	faceRects.clear();

	//dart9
	fileName = "dart9.jpg";
	faceRects.push_back(make_pair(Point(89, 222), Point(198, 332)));
	fileInfo.push_back(make_pair(fileName, faceRects));
	faceRects.clear();

	//dart10
	fileName = "dart10.jpg";
	fileInfo.push_back(make_pair(fileName, faceRects));
	faceRects.clear();

	//dart11
	fileName = "dart11.jpg";
	faceRects.push_back(make_pair(Point(326, 82), Point(382, 138)));
	fileInfo.push_back(make_pair(fileName, faceRects));
	faceRects.clear();

	//dart12
	fileName = "dart12.jpg";
	fileInfo.push_back(make_pair(fileName, faceRects));
	faceRects.clear();

	//dart13
	fileName = "dart13.jpg";
	faceRects.push_back(make_pair(Point(419, 136), Point(528, 243)));
	fileInfo.push_back(make_pair(fileName, faceRects));
	faceRects.clear();

	//dart14
	fileName = "dart14.jpg";
	faceRects.push_back(make_pair(Point(469, 227), Point(552, 315)));
	faceRects.push_back(make_pair(Point(725, 202), Point(825, 290)));
	fileInfo.push_back(make_pair(fileName, faceRects));
	faceRects.clear();

	//dart15
	fileName = "dart15.jpg";
	fileInfo.push_back(make_pair(fileName, faceRects));
	faceRects.clear();

	//return info
	return fileInfo;
}

tuple<Mat, int, int, int> CalculateAccuracy(vector<pair<Point, Point>> trueFaceRects, vector<pair<Point, Point>> detectedFaceFrames, Mat img, String fileName, float hitThresh) {

	/*
	This function uses the rules defined in our write up to calculate the number of TPs, FPs and FNs in ones detections frames
	*/

	int ax, ay, bx, by, cx, cy, dx, dy;
	float overlap, frameArea, truthArea;
	vector<bool> hasBeenHit;
	Point truthCenter, detectedCenter;
	bool isTP;
	int TPCount = 0; int FPCount = 0; int FNCount = 0;
	float distBetweenCenters;
	float normConst = sqrt(img.rows * img.cols);

	for (int i = 0; i < trueFaceRects.size(); i++) {
		hasBeenHit.push_back(false);
	}

	if (trueFaceRects.size() > 0) {
		for (int trueFrame = 0; trueFrame < trueFaceRects.size(); trueFrame++) {
			rectangle(img, trueFaceRects[trueFrame].first, trueFaceRects[trueFrame].second, Scalar(0, 255, 0), 3, 8, 0);
		}
	}


	if (detectedFaceFrames.size() > 0) {
		for (int frame = 0; frame < detectedFaceFrames.size(); frame++)
		{

			isTP = false;

			if (trueFaceRects.size() > 0) {
				for (int truth = 0; truth < trueFaceRects.size(); truth++) {

					ax = detectedFaceFrames[frame].first.x;
					ay = detectedFaceFrames[frame].first.y;
					bx = detectedFaceFrames[frame].second.x;
					by = detectedFaceFrames[frame].second.y;
					cx = trueFaceRects[truth].second.x;
					cy = trueFaceRects[truth].second.y;
					dx = trueFaceRects[truth].first.x;
					dy = trueFaceRects[truth].first.y;

					frameArea = (bx - ax)*(by - ay);
					truthArea = (dx - cx)*(dy - cy);

					truthCenter = Point(float(cx + dx) / 2., float(cy + dy) / 2.);
					detectedCenter = Point(float(ax + bx) / 2., float(ay + by) / 2.);
					distBetweenCenters = (pow(pow(truthCenter.x - detectedCenter.x, 2) + pow(truthCenter.y - detectedCenter.y, 2), 0.5))/normConst;
				
					if ((distBetweenCenters)*(max(frameArea / truthArea, truthArea / frameArea)) < hitThresh)
					{
						hasBeenHit[truth] = true;
						isTP = true;
					}

				}
			}

			if (isTP == true) {
				TPCount += 1;
				rectangle(img, detectedFaceFrames[frame].first, detectedFaceFrames[frame].second, Scalar(255, 0, 255), 3, 8, 0);
			}
			if (isTP != true) {
				FPCount += 1;
				rectangle(img, detectedFaceFrames[frame].first, detectedFaceFrames[frame].second, Scalar(255, 255, 153), 3, 8, 0);
			}

		}
	}

	if (hasBeenHit.size() > 0) {
		for (int i = 0; i < hasBeenHit.size(); i++) {
			if (hasBeenHit[i] == false) {
				FNCount += 1;
			}
		}
	}

	cout << fileName + " has " << TPCount << " True Positives, " << FPCount << " False Positives and " << FNCount << " False Negatives" << endl;

	return(make_tuple(img, TPCount, FPCount, FNCount));
}
