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

Mat Convolve(float kernel[3][3], Mat grayImg);
Mat Magnitude(Mat ddx, Mat ddy);
Mat Direction(Mat ddx, Mat ddy);
pair<Mat, Mat> GetDirAndMag(Mat img);
Mat Threshold(Mat magImg, int Thresh);
pair<Mat, Mat> HoughCircles(Mat threshMagImg, Mat dirImg, int minRad, int maxRad);
Mat HoughLines(Mat threshMagImg, Mat dirImg, float error);
pair<Mat, Mat> PlotLines(Mat linesHoughSpace, Mat img, int lineThresh, int zeroDist, int thickness);
pair<Mat, vector<pair<Point, Point>>> detectAndDisplay(Mat frame);
vector<pair<Point, int>> FindPossibleCenters(vector<pair<Point, Point>> dartboardFrames, Mat img, int edgeThresh, int lineThresh, int minOverlap);
vector<Point> FindBestCenters(vector<pair<Point, int>> possibleDartboardCenters, int minDist);
vector<tuple<Point, int, int>> VerifyDartboard(Mat img, vector<Point> bestCenters, int dartboardThresh, int minRad, int maxFrameSize, int resizedFrameSize, int resizes, float edgeThresh);
vector<pair<String, vector<pair<Point, Point>>>> GetFileInfo();
tuple<Mat, int, int, int> CalculateAccuracy(vector<pair<Point, Point>> trueFaceRects, vector<pair<Point, Point>> detectedFaceFrames, Mat img, String fileName, float hitThresh);
float distance(Point a, Point b);
Point Average(vector<Point> points);

/** Global variables */
String cascade_name = "cascade.xml";
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
		vector<pair<Point, Point>> detectedDartboards = cascadeInfo.second;


		/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
		// THE FOLLOWING CODE COMPARES ACCURACY OF DETECTION FRAMES TO GROUND TRUTH FRAMES

		// 1. Load ground truth frames from user generated data
		vector<pair<Point, Point>> trueFaceRects = fileInfo[pic].second;

		// 2. CalculateAccuracy calclates number of True Positives, False Positives, and False Negatives in our detections, it also generates an accuracyDepiction figure
		tuple<Mat, int, int, int> accuracyInfo = CalculateAccuracy(trueFaceRects, detectedDartboards, img, fileName, 0.125);
		Mat accuracyDepiction = get<0>(accuracyInfo);

		// 3. Save our accuracyDepiction image, and add number of TPs, FPs and FNs to totals.
		imwrite("Task2DetectedDartboards" + fileName, accuracyDepiction);
		TP += get<1>(accuracyInfo);	
		FP += get<2>(accuracyInfo);
		FN += get<3>(accuracyInfo);

		/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

		detectedDartboards.clear();
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

float distance(Point a, Point b) { return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y)); }

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

	// This function contains ground truth frames for dart0.jpg - dart15.jpg

	vector<pair<String, vector<pair<Point, Point>>>> fileInfo;

	//dart0
	String fileName = "dart0.jpg";
	vector<pair<Point, Point>> dartRects;
	dartRects.push_back(make_pair(Point(425, 0), Point(620, 215)));
	fileInfo.push_back(make_pair(fileName, dartRects));
	dartRects.clear();

	//dart1
	fileName = "dart1.jpg";
	dartRects.push_back(make_pair(Point(165, 105), Point(415, 353)));
	fileInfo.push_back(make_pair(fileName, dartRects));
	dartRects.clear();

	//dart2
	fileName = "dart2.jpg";
	dartRects.push_back(make_pair(Point(90, 85), Point(202, 196)));
	fileInfo.push_back(make_pair(fileName, dartRects));
	dartRects.clear();

	//dart3
	fileName = "dart3.jpg";
	dartRects.push_back(make_pair(Point(317, 137), Point(398, 230)));
	fileInfo.push_back(make_pair(fileName, dartRects));
	dartRects.clear();

	//dart4
	fileName = "dart4.jpg";
	dartRects.push_back(make_pair(Point(155, 65), Point(380, 320)));
	fileInfo.push_back(make_pair(fileName, dartRects));
	dartRects.clear();

	//dart5
	fileName = "dart5.jpg";
	dartRects.push_back(make_pair(Point(417, 127), Point(546, 266)));
	fileInfo.push_back(make_pair(fileName, dartRects));
	dartRects.clear();

	//dart6
	fileName = "dart6.jpg";
	dartRects.push_back(make_pair(Point(203, 107), Point(281, 189)));
	fileInfo.push_back(make_pair(fileName, dartRects));
	dartRects.clear();

	//dart7
	fileName = "dart7.jpg";
	dartRects.push_back(make_pair(Point(234, 148), Point(404, 334)));
	fileInfo.push_back(make_pair(fileName, dartRects));
	dartRects.clear();

	//dart8
	fileName = "dart8.jpg";
	dartRects.push_back(make_pair(Point(62, 242), Point(135, 351)));
	dartRects.push_back(make_pair(Point(830, 202), Point(971, 352)));
	fileInfo.push_back(make_pair(fileName, dartRects));
	dartRects.clear();

	//dart9
	fileName = "dart9.jpg";
	dartRects.push_back(make_pair(Point(169, 15), Point(466, 316)));
	fileInfo.push_back(make_pair(fileName, dartRects));
	dartRects.clear();

	//dart10
	fileName = "dart10.jpg";
	dartRects.push_back(make_pair(Point(77, 90), Point(198, 229)));
	dartRects.push_back(make_pair(Point(577, 117), Point(645, 223)));
	dartRects.push_back(make_pair(Point(912, 141), Point(954, 222)));
	fileInfo.push_back(make_pair(fileName, dartRects));
	dartRects.clear();

	//dart11
	fileName = "dart11.jpg";
	dartRects.push_back(make_pair(Point(165, 93), Point(240, 180)));
	fileInfo.push_back(make_pair(fileName, dartRects));
	dartRects.clear();

	//dart12
	fileName = "dart12.jpg";
	dartRects.push_back(make_pair(Point(151, 57), Point(223, 230)));
	fileInfo.push_back(make_pair(fileName, dartRects));
	dartRects.clear();

	//dart13
	fileName = "dart13.jpg";
	dartRects.push_back(make_pair(Point(256, 101), Point(420, 270)));
	fileInfo.push_back(make_pair(fileName, dartRects));
	dartRects.clear();

	//dart14
	fileName = "dart14.jpg";
	dartRects.push_back(make_pair(Point(103, 85), Point(262, 244)));
	dartRects.push_back(make_pair(Point(972, 79), Point(1126, 236)));
	fileInfo.push_back(make_pair(fileName, dartRects));
	dartRects.clear();

	//dart15
	fileName = "dart15.jpg";
	dartRects.push_back(make_pair(Point(131, 36), Point(302, 211)));
	fileInfo.push_back(make_pair(fileName, dartRects));
	dartRects.clear();

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
