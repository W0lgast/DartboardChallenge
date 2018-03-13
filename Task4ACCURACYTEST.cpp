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
pair<Mat, vector<pair<Point, Point>>> GetNewFrames(Mat img, int windowSize, int threshold);

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
		vector<pair<Point, Point>> detectedDartboards;
		Mat img = imread(fileName);

		// 2. Load the Strong Classifier in a structure called `Cascade'
		if (!cascade.load(cascade_name)) { printf("--(!)Error loading\n"); return -1; };

		// 3. Detect Frames Poitentially Containing Dartboards
		pair<Mat, vector<pair<Point, Point>>> cascadeInfo = detectAndDisplay(img);
		vector<pair<Point, Point>> dartboardFrames = cascadeInfo.second;

		// 4. Take each of these frames and run through our Hough Line detector to erase False Positives and detect positions of potential dartboard centers
		vector<pair<Point, int>> possibleDartboardCenters = FindPossibleCenters(dartboardFrames, img, 50, 15, 5);
		vector<Point> bestCenters = FindBestCenters(possibleDartboardCenters, 5);

		// 5. Take each of these centers and run through our Hough Ellipse detector to check whether these center points are the centers of some ellipse
		vector<tuple<Point, int, int>> verifiedDartboardInfo = VerifyDartboard(img, bestCenters, 50, 5, 250, 100, 5, 20.);

		// 6. Store detection frames in vector detectedDartboards
		Point topRight, bottomLeft;
		for (int i = 0; i < verifiedDartboardInfo.size(); i++) {
			bottomLeft = Point(get<0>(verifiedDartboardInfo[i]).x - get<2>(verifiedDartboardInfo[i]), get<0>(verifiedDartboardInfo[i]).y + get<2>(verifiedDartboardInfo[i]));
			topRight = Point(get<0>(verifiedDartboardInfo[i]).x + get<2>(verifiedDartboardInfo[i]), get<0>(verifiedDartboardInfo[i]).y - get<2>(verifiedDartboardInfo[i]));
			detectedDartboards.push_back(make_pair(Point(bottomLeft.x, topRight.y), Point(topRight.x, bottomLeft.y) ));
		}

		// 7. Generate new frames using McAdam-Freud McCarthy Algorithm
		Mat imgClone = img.clone();
		pair<Mat, vector<pair<Point, Point>>> newFrameInfo = GetNewFrames(imgClone, 2, 200);
		vector<pair<Point, Point>> newFrames = newFrameInfo.second;

		// 8. Take each of these new frames and run through our Hough Line detector to erase False Positives and detect positions of potential dartboard centers
		vector<pair<Point, Point>> detectedDartboardsNewFrames;
		vector<pair<Point, int>> possibleDartboardCentersNewFrames = FindPossibleCenters(newFrames, img, 145, 8, 5);
		vector<Point> bestCentersNewFrames = FindBestCenters(possibleDartboardCentersNewFrames, 5);

		// 9. Take each of these centers and run through our Hough Ellipse detector to check whether these center points are the centers of some ellipse
		vector<tuple<Point, int, int>> verifiedDartboardInfoNewFrames = VerifyDartboard(img, bestCentersNewFrames, 10, 50, 250, 100, 5, 20.);
		for (int i = 0; i < verifiedDartboardInfoNewFrames.size(); i++) {
			bottomLeft = Point(get<0>(verifiedDartboardInfoNewFrames[i]).x - get<2>(verifiedDartboardInfoNewFrames[i]), get<0>(verifiedDartboardInfoNewFrames[i]).y + get<2>(verifiedDartboardInfoNewFrames[i]));
			topRight = Point(get<0>(verifiedDartboardInfoNewFrames[i]).x + get<2>(verifiedDartboardInfoNewFrames[i]), get<0>(verifiedDartboardInfoNewFrames[i]).y - get<2>(verifiedDartboardInfoNewFrames[i]));
			detectedDartboardsNewFrames.push_back(make_pair(Point(bottomLeft.x, topRight.y), Point(topRight.x, bottomLeft.y)));
		}

		// 10. Delete any new dartboard detections which are too close to original detections 
		Point detectedDartboardsCenter, newDetectedCenter;
		if (detectedDartboards.size() > 0) {
			for (int i = 0; i < detectedDartboards.size(); i++) {
				detectedDartboardsCenter = Point(detectedDartboards[i].first.x / 2. + detectedDartboards[i].first.x / 2.,
					detectedDartboards[i].first.y / 2. + detectedDartboards[i].first.y / 2.);
				for (int j = detectedDartboardsNewFrames.size() -1 ; j >= 0; j--) {
					newDetectedCenter = Point(detectedDartboardsNewFrames[j].first.x / 2. + detectedDartboardsNewFrames[j].first.x / 2.,
						detectedDartboardsNewFrames[j].first.y / 2. + detectedDartboardsNewFrames[j].first.y / 2.);
					if (distance(detectedDartboardsCenter, newDetectedCenter) < 100) {
						detectedDartboardsNewFrames.erase(detectedDartboardsNewFrames.begin() + j);
					}
				}
			}
		}

		// 11. Add undeleted new detections to original detections for finished set of detected dartboard frames
		detectedDartboards.insert(detectedDartboards.end(), detectedDartboardsNewFrames.begin(), detectedDartboardsNewFrames.end());

		/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
		// THE FOLLOWING CODE COMPARES ACCURACY OF DETECTION FRAMES TO GROUND TRUTH FRAMES

		// 1. Load ground truth frames from user generated data
		vector<pair<Point, Point>> trueFaceRects = fileInfo[pic].second;

		// 2. CalculateAccuracy calclates number of True Positives, False Positives, and False Negatives in our detections, it also generates an accuracyDepiction figure
		tuple<Mat, int, int, int> accuracyInfo = CalculateAccuracy(trueFaceRects, detectedDartboards, img, fileName, 0.125);
		Mat accuracyDepiction = get<0>(accuracyInfo);

		// 3. Save our accuracyDepiction image, and add number of TPs, FPs and FNs to totals.
		imwrite("Task4DetectedDartboards" + fileName, accuracyDepiction);
		TP += get<1>(accuracyInfo);	
		FP += get<2>(accuracyInfo);
		FN += get<3>(accuracyInfo);

		/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

		detectedDartboards.clear();
		detectedDartboardsNewFrames.clear();
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

Mat Convolve(float kernel[3][3], Mat grayImg) {
	
	Mat grayFilteredImg = Mat(grayImg.rows, grayImg.cols, CV_32F, Scalar(0));

	for (int y = 1; y<grayImg.rows - 1; y++) {
		for (int x = 1; x<grayImg.cols - 1; x++) {

			grayFilteredImg.at<float>(y, x) = 0.0;

			for (int i = -1; i <= 1; i++) {
				for (int j = -1; j <= 1; j++) {

					grayFilteredImg.at<float>(y, x) = (grayFilteredImg.at<float>(y, x) + float(grayImg.at<uchar>(y + i, x + j) * kernel[i + 1][j + 1]));
				
				}
			}
		}
	}

	return grayFilteredImg;
}

Mat Magnitude(Mat ddx, Mat ddy) {

	Mat magImg = Mat(ddx.rows, ddx.cols, CV_32F, Scalar(0));

	for (int y = 0; y < magImg.rows; y++) {
		for (int x = 0; x < magImg.cols; x++) {
			magImg.at<float>(y, x) = pow(pow(float(ddx.at<float>(y, x)), 2.0) + pow(float(ddy.at<float>(y, x)), 2.0), 0.5);
		}
	}

	normalize(magImg, magImg, 0, 255, NORM_MINMAX, CV_8UC1);

	return magImg;
}

Mat Direction(Mat ddx, Mat ddy) {

	Mat dirImg = Mat(ddx.rows, ddx.cols, CV_32F, Scalar(0));

	for (int y = 0; y < dirImg.rows; y++) {
		for (int x = 0; x < dirImg.cols; x++) {

			dirImg.at<float>(y, x) = float((atan2(double(ddy.at<float>(y, x)), double(ddx.at<float>(y, x)))));

		}
	}

	return dirImg;
}

Mat Threshold(Mat magImg, int Thresh) {

	Mat threshMag = magImg.clone();

	for (int y = 0; y < threshMag.rows; y++) {
		for (int x = 0; x < threshMag.cols; x++) {
			if (threshMag.at<uchar>(y, x) >= Thresh) {
				threshMag.at<uchar>(y, x) = 255;
			}
			else {
				threshMag.at<uchar>(y, x) = 0;
			}
		}
	}

	return threshMag;
}

pair<Mat, Mat> GetDirAndMag(Mat img)
{
	// Takes an image and returns directional image and gradient image

	// 1. Calculate gradient in x direction
	float ddxkernel[3][3] =
	{
		{ -1,0 ,1 },
		{ -2,0 ,2 },
		{ -1,0 ,1 }
	};
	Mat ddx = Convolve(ddxkernel, img);

	// 2. Calculate gradient in y direction
	float ddykernel[3][3] =
	{
		{ -1,-2,-1 },
		{ 0 ,0 ,0 },
		{ 1 ,2 ,1 }
	};
	Mat ddy = Convolve(ddykernel, img);

	// 3. Calculate gradient image and directional image using Magnitude and Direction functions
	Mat magImg = Magnitude(ddx, ddy);
	Mat dirImg = Direction(ddx, ddy);

	return make_pair(dirImg, magImg);
}

Point Average(vector<Point> points) {
	Point p_avg = Point(0, 0);
	if (points.size() > 0) {
		for (int i = 0; i < points.size(); i++) {
			p_avg.x += points[i].x;
			p_avg.y += points[i].y;
		}
	}
	p_avg.x = p_avg.x / points.size();
	p_avg.y = p_avg.y / points.size();
	return p_avg;
}

pair<Mat, Mat> HoughCircles(Mat threshMagImg, Mat dirImg, int minRad, int maxRad) {

	// This function calculates the circle Hough Space and circle Hough Plot for circles with radius between minRad and maxRad

	int sizesSpace[] = { threshMagImg.rows, threshMagImg.cols, maxRad - minRad + 1 };
	int sizesPlot[] = { threshMagImg.rows, threshMagImg.cols };

	Mat houghSpace = Mat(3, sizesSpace, CV_32F, Scalar(0));
	Mat houghPlot = Mat(2, sizesPlot, CV_32F, Scalar(0));

	for (int y = 0; y < threshMagImg.rows; y++) {
		for (int x = 0; x < threshMagImg.cols; x++) {

			if (threshMagImg.at<uchar>(y, x) == 255) {
				
				// Calculates potential circle centers for (y,x), and encodes findings in HoughSpace and HoughPlot 
				for (int r = minRad; r <= maxRad; r++) {
					for (int i = 1; i <= 4; i++) {

						int x0 = 0;
						int y0 = 0;

						switch (i) {
						case 1:
							x0 = int(float(x) + (float(r) * float(cos(dirImg.at<float>(y, x)))));
							y0 = int(float(y) + (float(r) * float(sin(dirImg.at<float>(y, x)))));
							break;
						case 2:
							x0 = int(float(x) + (float(r) * float(cos(dirImg.at<float>(y, x))))); 
							y0 = int(float(y) - (float(r) * float(sin(dirImg.at<float>(y, x)))));
							break;
						case 3:
							x0 = int(float(x) - (float(r) * float(cos(dirImg.at<float>(y, x))))); 
							y0 = int(float(y) + (float(r) * float(sin(dirImg.at<float>(y, x)))));
							break;
						case 4:
							x0 = int(float(x) - (float(r) * float(cos(dirImg.at<float>(y, x))))); 
							y0 = int(float(y) - (float(r) * float(sin(dirImg.at<float>(y, x)))));
							break;
						}

						if ((0 < x0 && x0 < int(threshMagImg.cols - 1)) && (0 < y0 && y0 < int(threshMagImg.rows - 1))) {
							houghSpace.at<float>(y0, x0, r - minRad)++;
							houghPlot.at<float>(y0, x0)++;
						}
					}
				}
			}
		}
	}

	return make_pair(houghSpace, houghPlot);
}

Mat HoughLines(Mat threshMagImg, Mat dirImg, float error) {

	// This function calculates the circle Hough Space and circle Hough Plot for circles with radius between minRad and maxRad
	int maxRho = sqrt(threshMagImg.rows*threshMagImg.rows + threshMagImg.cols*threshMagImg.cols);
	int sizesSpace[] = { maxRho + 1, 360 };
	Mat houghSpace = Mat(2, sizesSpace, CV_32F, Scalar(0));

	for (int y = 0; y < threshMagImg.rows; y++) {
		for (int x = 0; x < threshMagImg.cols; x++) {

			if (threshMagImg.at<uchar>(y, x) == 255) {

				// Calculates potential rho and phi values for the suspected line passing through (y,x), findings are encoded in houghSpace
				float theta = dirImg.at<float>(y, x);
				int thetaRound = abs((int(theta*180.0 / M_PI)) % 360);
				int deltaTheta = 5;

				for (int phi = max(thetaRound - deltaTheta, 0); phi < min(thetaRound + deltaTheta, 360); phi++)
				{
					int rho = int(float(x)*cos(phi / 180.0*M_PI) + float(y)*sin(phi / 180.0*M_PI));
					if (rho >= 0 && rho <= maxRho) {
						houghSpace.at<float>(rho, phi)++;
					}
				}

			}

		}
	}

	return houghSpace;

}

pair<Mat, Mat> PlotLines(Mat linesHoughSpace, Mat img, int lineThresh, int zeroDist, int thickness) {

	/*
	This function examines the lines Hough Space. It considers any point on the Hough space with a value above lineThresh to be a true line of width "thickness" on the image
	It returns an image with these lines drawn onto the image, and a new Mat file lineCount encoding the amount of overlapping lines at any point of the image
	*/

	int linePlotDims[] = { img.size[0], img.size[1] };
	Mat linePlot = Mat(2, linePlotDims, CV_8U, Scalar(0));
	float maxPixelVal = -1.;
	int rhoMaxPixel, thetaMaxPixel;
	bool isEnd = false;
	Mat imgWithLines = img.clone();
	Mat justLines = Mat(2, linePlotDims, CV_8U, Scalar(0));
	Mat lineCount = justLines.clone();

	while (isEnd == false) {

		// 1. find the pixel with the maximum value on the hough space
		for (int y = 0; y < linesHoughSpace.rows; y++) {
			for (int x = 0; x < linesHoughSpace.cols; x++) {
				if (linesHoughSpace.at<float>(y, x) > maxPixelVal)
				{
					maxPixelVal = linesHoughSpace.at<float>(y, x);
					rhoMaxPixel = y;
					thetaMaxPixel = x;
				}
			}
		}

		// 2. if this value is above lineThresh, draw the line defined by rhoMaxPixel and thetaMaxPixel onto imgWithLines, justLines, and lineCount
		if (maxPixelVal >= lineThresh) {
			for (int x = 0; x < img.cols; x++) {

				int y = int(-(cos(thetaMaxPixel*M_PI / 180.) / sin(thetaMaxPixel*M_PI / 180.))*float(x) + float(rhoMaxPixel) / sin(thetaMaxPixel*M_PI / 180.));

				if (0 <= y && y < img.rows)
				{
					for (int yy = max(y - thickness, 0); yy <= min(y + thickness, img.rows - 1); yy++) {
						imgWithLines.at<Vec3b>(yy, x)[0] = 0;
						imgWithLines.at<Vec3b>(yy, x)[1] = 255;
						imgWithLines.at<Vec3b>(yy, x)[2] = 0;

						justLines.at<uchar>(yy, x) = 255;
					}
				}

			}

			for (int y = 0; y < img.rows; y++) {


				int x = int(float(rhoMaxPixel) / cos(thetaMaxPixel*M_PI / 180.) - (sin(thetaMaxPixel*M_PI / 180.) / cos(thetaMaxPixel*M_PI / 180.))*float(y));

				if (0 <= x && x < img.cols)
				{
					for (int xx = max(x - thickness, 0); xx <= min(x + thickness, img.cols - 1); xx++) {
						imgWithLines.at<Vec3b>(y, xx)[0] = 0;
						imgWithLines.at<Vec3b>(y, xx)[1] = 255;
						imgWithLines.at<Vec3b>(y, xx)[2] = 0;

						justLines.at<uchar>(y, xx) = 255;
					}
				}

			}

			for (int y = 0; y < lineCount.rows; y++) {
				for (int x = 0; x < lineCount.cols; x++) {

					if (justLines.at<uchar>(y, x) == 255)
					{
						lineCount.at<uchar>(y, x)++;
						justLines.at<uchar>(y, x) = 0;
					}

				}
			}

			//step 3 set maxPixel to zero and zero a circle all around rhoMaxPixel and thetaMaxPixel on the linesHoughSpace
			maxPixelVal = -1.;

			for (int y = max((rhoMaxPixel - zeroDist), 0); y < min((rhoMaxPixel + zeroDist), linesHoughSpace.rows); y++) {
				for (int x = max((thetaMaxPixel - zeroDist), 0); x < min((thetaMaxPixel + zeroDist), linesHoughSpace.cols); x++) {

					if (pow((y - rhoMaxPixel), 2) + pow((x - thetaMaxPixel), 2) <= pow(zeroDist, 2)) {
						linesHoughSpace.at<float>(y, x) = 0.0;	//set hough plots to zero
					}
				}
			}
		}
		else
		{
			isEnd = true;
		}
	}

	return make_pair(imgWithLines, lineCount);

}

vector<pair<Point, int>> FindPossibleCenters(vector<pair<Point, Point>> dartboardFrames, Mat img, int edgeThresh, int lineThresh, int minOverlap) {

	/* 
	This function takes frames potentially containing dartboards, and uses HoughLines to predict if these frames contain the centers of dartboards
	by looking for points with minOverlap overlapping lines
	*/

	pair<Point, Point> currentFrameRectangle;
	pair<Mat, Mat> resizedCurrentFrameInfo, currentFrameLinesInfo;
	Mat resizedCurrentFrameDirImg, resizedCurrentFrameMagImg, resizedCurrentFrameThreshMagImg, currentFrameLinesHoughSpace, currentFrameWithLines, currentFrameLineCount;
	Mat resizedCurrentFrame = Mat(150, 150, CV_8UC3, Scalar(0));
	Mat resizedCurrentFrameGrey = Mat(150, 150, CV_8UC3, Scalar(0));
	int maxOverlappingLines, dartboardCenterX, dartboardCenterY, dartboardCenterTrueX, dartboardCenterTrueY;
	vector<pair<Point, int>> possibleDartboardCenters;
	int Y, X;

	// 1. Loop through every frame in dartboardFrames
	if (dartboardFrames.size() > 0) {
		for (int frameNumber = 0; frameNumber < dartboardFrames.size(); frameNumber++)
		{

			Mat currentFrame;
			currentFrameRectangle = dartboardFrames[frameNumber];

			// 2. Create Mat containing the frame currently being examined
			currentFrame = Mat((currentFrameRectangle.second).x - (currentFrameRectangle.first).x + 1,
				(currentFrameRectangle.second).y - (currentFrameRectangle.first).y + 1, CV_8UC3, Scalar(0));
			for (int y = 0; y < currentFrame.rows; y++) {
				for (int x = 0; x < currentFrame.cols; x++) {

					Y = max(min(y + (currentFrameRectangle.first).y, img.rows-1), 0);
					X = max(min(x + (currentFrameRectangle.first).x, img.cols-1), 0);
					currentFrame.at<Vec3b>(y, x)[0] = img.at<Vec3b>(Y , X )[0];
					currentFrame.at<Vec3b>(y, x)[1] = img.at<Vec3b>(Y , X )[1];
					currentFrame.at<Vec3b>(y, x)[2] = img.at<Vec3b>(Y , X )[2];

				}
			}

			// 3. Convert frame to grayscale, and use Hough Lines technique to generate information about the lines in the frame, this information is incoded in currentFrameLinesInfo

			resize(currentFrame, resizedCurrentFrame, resizedCurrentFrame.size(), 0, 0);
			cvtColor(resizedCurrentFrame, resizedCurrentFrameGrey, CV_BGR2GRAY);
			resizedCurrentFrameInfo = GetDirAndMag(resizedCurrentFrameGrey);
			resizedCurrentFrameDirImg = resizedCurrentFrameInfo.first;
			resizedCurrentFrameMagImg = resizedCurrentFrameInfo.second;
			resizedCurrentFrameThreshMagImg = Threshold(resizedCurrentFrameMagImg, edgeThresh);

			currentFrameLinesHoughSpace = HoughLines(resizedCurrentFrameThreshMagImg, resizedCurrentFrameDirImg, 0.01);

			currentFrameLinesInfo = PlotLines(currentFrameLinesHoughSpace, resizedCurrentFrame, lineThresh, 20, 1);
			currentFrameWithLines = currentFrameLinesInfo.first;
			currentFrameLineCount = currentFrameLinesInfo.second;

			// 4. Search for the point on the frame with the highest number of overlapping lines
			maxOverlappingLines = -1;
			for (int y = 0; y < currentFrameLineCount.rows; y++) {
				for (int x = 0; x < currentFrameLineCount.cols; x++) {
					if (currentFrameLineCount.at<uchar>(y, x) > maxOverlappingLines)
					{
						maxOverlappingLines = currentFrameLineCount.at<uchar>(y, x);
						dartboardCenterX = x;
						dartboardCenterY = y;
					}
				}
			}

			// 5. If this point has minOverlap or more overlapping lines, it is considered a potential dartboard center, and its position on the original image is stored in possibleDartboardCenters
			if (maxOverlappingLines >= minOverlap)
			{
				dartboardCenterTrueY = (currentFrameRectangle.first).y + (float(dartboardCenterY) *float((currentFrameRectangle.second).y - (currentFrameRectangle.first).y) / 150.);
				dartboardCenterTrueX = (currentFrameRectangle.first).x + (float(dartboardCenterX) *float((currentFrameRectangle.second).x - (currentFrameRectangle.first).x) / 150.);
				possibleDartboardCenters.push_back(make_pair(Point(dartboardCenterTrueX, dartboardCenterTrueY), maxOverlappingLines));

			}

			currentFrame.release();
		}
	}

	return possibleDartboardCenters;

}

vector<Point> FindBestCenters(vector<pair<Point, int>> possibleDartboardCenters, int minDist) {

	/*
	This function checks that the distance between potential dartboard centers is greater than minDist
	If two centers are close together, this function deletes the center with the fewest overlapping lines 
	*/

	int maxMaxOverlappingLines, bestCenterLocation;
	Point bestCenter;
	vector<Point> bestCenters;
	bool isEnd = false;
	bool tooClose = false;

	// 1. Loop through every possible dartboard center, either storing the dartboard center in bestCenters or deleting them, until vector is empty
	while (possibleDartboardCenters.size() > 0) {

		maxMaxOverlappingLines = 0;
		tooClose = false;

		//find the maximum number of overlapping lines on any possible center in possibleDartboardCenters
		for (int possibleCenter = 0; possibleCenter < possibleDartboardCenters.size(); possibleCenter++) {
			if (possibleDartboardCenters[possibleCenter].second > maxMaxOverlappingLines)
			{
				maxMaxOverlappingLines = possibleDartboardCenters[possibleCenter].second;
				bestCenter = possibleDartboardCenters[possibleCenter].first;
				bestCenterLocation = possibleCenter;
			}
		}

		//check this point isnt close to a previously found center, forgetting it if it is, and storing it in bestCenters if it isn't
		if (bestCenters.size() > 0) {
			for (int i = 0; i < bestCenters.size(); i++) {
				if ((bestCenter.x - bestCenters[i].x)*(bestCenter.x - bestCenters[i].x) + (bestCenter.y - bestCenters[i].y)*(bestCenter.y - bestCenters[i].y) <= minDist*minDist)
				{
					tooClose = true;
				}
			}
			if (tooClose == false)
			{
				bestCenters.push_back(bestCenter);
			}
		}
		else {
			bestCenters.push_back(bestCenter);
		}

		//the potential center we have just considered is now erased, though its value may have been stored in bestCenters
		possibleDartboardCenters.erase(possibleDartboardCenters.begin() + bestCenterLocation);
	}

	return bestCenters;

}

vector<tuple<Point, int, int>> VerifyDartboard(Mat img, vector<Point> bestCenters, int dartboardThresh, int minRad, int maxFrameSize, int resizedFrameSize, int resizes, float edgeThresh) {
	
	/*
	This function checks potential dartboard centers in bestCenters to check wether these points are also the centers of some ellipse
	This is done by:
	1. Getting a new square frame around the potential center, the size of this frame is maxFrameSize unless the suspected center is close to the edges of the picture, in which case
	   it is as big as it can be while remaining a square frame on the original image.
	2. Resizes this frame to resizedFrameSize x resizedFrameSize for computational efficiency
	2. Stretching the image in the x direction resizes times, where each stretch is 1 + (.2 * resize)
	3. Using HoughCircles on these stretched images. 
	4. If any point within a 5x5 grid of the potential center has a hough plot value of over dartboardThresh, it is considered the center of an ellipse
	*/

	Mat currentFrame, dirImg, magImg;
	Mat resizedCurrentFrame = Mat(resizedFrameSize, resizedFrameSize, CV_8UC3, Scalar(0));
	Mat grayImg = Mat(resizedFrameSize, resizedFrameSize, CV_8UC1, Scalar(0));
	pair<Mat, Mat> dirAndMag, hough;
	float maxPixelVal, maxHough, rad;
	Point newBestCenter;
	int resizedNewBestCenterX, resizedNewBestCenterY, Rad, frameSize, resizeMaxPixel;
	vector<tuple<Point, int, int>> verifiedDartboardInfo;
	int maxRad = resizedFrameSize/2;

	// 1. Loop through all potential centers in bestCenters
	if (bestCenters.size() > 0) {
		for (int frame = 0; frame < bestCenters.size(); frame++) {

			// 2. Create new frame around potential center
			frameSize = min(min(min(bestCenters[frame].y, bestCenters[frame].x), min(img.rows - bestCenters[frame].y, img.cols - bestCenters[frame].x)), maxFrameSize / 2);
			currentFrame = Mat(2 * frameSize, 2 * frameSize, CV_8UC3, Scalar(0));
			for (int y = 0; y < currentFrame.rows; y++) {
				for (int x = 0; x < currentFrame.cols; x++) {

					currentFrame.at<Vec3b>(y, x)[0] = img.at<Vec3b>(y + bestCenters[frame].y - frameSize, x + bestCenters[frame].x - frameSize)[0];
					currentFrame.at<Vec3b>(y, x)[1] = img.at<Vec3b>(y + bestCenters[frame].y - frameSize, x + bestCenters[frame].x - frameSize)[1];
					currentFrame.at<Vec3b>(y, x)[2] = img.at<Vec3b>(y + bestCenters[frame].y - frameSize, x + bestCenters[frame].x - frameSize)[2];

				}
			}

			// 3. Resize the frame for computational efficiency
			resize(currentFrame, resizedCurrentFrame, resizedCurrentFrame.size(), 0, 0);
			newBestCenter.y = int((float(frameSize) / (2.0*float(frameSize)))*float(resizedFrameSize));
			newBestCenter.x = int((float(frameSize) / (2.0*float(frameSize)))*float(resizedFrameSize));

			vector<Mat> AllHoughSpaces;
			vector<Mat> AllHoughPlots;
			Mat threshMagImg;
			pair<Mat, Mat> hough;
			maxPixelVal = -1.;
			
			// 4. Stretch the resized frame resizes time and use HoughCircles to look for circles in each resized frame. The houghSpaces and houghPlots for each resize are stored in AllHoughSpaces and AllHoughPlots

			Mat stretchedResizedCurrentFrame;
			for (int i = resizes - 1; i >= 0; i--) {

				resize(resizedCurrentFrame, stretchedResizedCurrentFrame, Size(), 1.0 + float(i) / 5.0, 1);
				cvtColor(stretchedResizedCurrentFrame, grayImg, COLOR_RGB2GRAY);
				dirAndMag = GetDirAndMag(grayImg);
				dirImg = dirAndMag.first;
				magImg = dirAndMag.second;
				threshMagImg = Threshold(magImg, edgeThresh);

				hough = HoughCircles(threshMagImg, dirImg, minRad, maxRad);

				AllHoughSpaces.push_back(hough.first);
				AllHoughPlots.push_back(hough.second);
			}

			// 5. Find the brightest point on the houghPlot within a 5x5 grid of the potential center

			for (int i = 0; i < resizes; i++) {

				resizedNewBestCenterY = newBestCenter.y;
				resizedNewBestCenterX = newBestCenter.x * (1.0 + float(resizes - i - 1.) / 5.0);
				
				for (int y = max(resizedNewBestCenterY - 5, 0); y < min(resizedNewBestCenterY + 6, AllHoughPlots[i].rows); y++) {
					for (int x = max(resizedNewBestCenterX - 5, 0); x < min(resizedNewBestCenterX + 6, AllHoughPlots[i].cols); x++) {
						if (AllHoughPlots[i].at<float>(y, x) > maxPixelVal) {
							maxPixelVal = AllHoughPlots[i].at<float>(y, x);
							resizeMaxPixel = i;
						}
					}
				}
			}

			// 6. Look on the hough space corresponding to this brightest points resized frame and find the radius with the most votes

			maxHough = -1.0;
			for (int r = 0; r < AllHoughSpaces[resizeMaxPixel].size[2]; r++)
			{
				for (int y = max(resizedNewBestCenterY - 5, 0); y < min(resizedNewBestCenterY + 6, AllHoughPlots[resizeMaxPixel].rows); y++) {
					for (int x = max(resizedNewBestCenterX - 5, 0); x < min(resizedNewBestCenterX + 6, AllHoughPlots[resizeMaxPixel].cols); x++) {

						if (AllHoughSpaces[resizeMaxPixel].at<float>(y, x, r) > maxHough)
						{
							maxHough = AllHoughSpaces[resizeMaxPixel].at<float>(y, x, r);
							Rad = (float(r) + float(minRad));
						}
					}
				}
			}

			// 7. Calculate the newBestCenter location on the original image

			newBestCenter.x = (float(newBestCenter.x) / float(resizedFrameSize)) * (2.0*float(frameSize)) + (bestCenters[frame].x - frameSize);
			newBestCenter.y = (float(newBestCenter.y) / float(resizedFrameSize)) * (2.0*float(frameSize)) + (bestCenters[frame].y - frameSize);
			Rad = (float(Rad) / float(resizedFrameSize))*(2.0*float(frameSize));


			// 8. If the brightest point on the hough space is above dartboardThresh, the center has been verified as a dartboard center, and its position, resize value, and radius are stored
			if (maxPixelVal >= dartboardThresh) {

				verifiedDartboardInfo.push_back(make_tuple(newBestCenter, resizeMaxPixel, Rad));

			}

			AllHoughSpaces.clear();
			AllHoughPlots.clear();
		}
	}

	return(verifiedDartboardInfo);
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
	int left, right, top, bottom;
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

pair<Mat, vector<pair<Point, Point>>> GetNewFrames(Mat img, int windowSize, int threshold) {

	/*
	This function uses the McAdam Freud McCarthy algoithm to generate new potential dartboard frames
	The function creates white pixels on a black image at points where a window of size 2*windowsize + 1, centered around that point, countains a green, red and cream pixel
	It then clusters these points and returns the average position of each cluster, where each cluster is at least threshold pixels away
	Frames of various sizes are creaated around these average cluster positions
	*/

	Mat imgOrig = img.clone();
	Mat imgClone = img.clone();

	cvtColor(imgClone, imgClone, cv::COLOR_BGR2HSV);
	vector<Point> windowCenters, averageClusterPositions;
	Mat centerData;
	Mat averageClusterPositionPlot = Mat(imgClone.rows, imgClone.cols, CV_8UC1, Scalar(0));
	vector<pair<Point, Point>> newFrames;
	Point bottomLeft, topRight;
	int frameSize;
	int prevFrameSize;

	// 1. Here we define the color ranges of Red, Green and Cream
	int minHRed = 170;
	int maxHRed = 10;
	int minSRed = 100;
	int maxSRed = 255;
	int minVRed = 0;
	int maxVRed = 256;

	int minHGreen = 50;
	int maxHGreen = 80;
	int minSGreen = 75;
	int maxSGreen = 255;
	int minVGreen = 0;
	int maxVGreen = 255;

	int minHCream = 10;
	int maxHCream = 25;
	int minSCream = 100;
	int maxSCream = 255;
	int minVCream = 0;
	int maxVCream = 255;

	bool red, green, cream;

	// 2. Here we loop through every pixel and check its corresponding window to test whether the window contains a red, green and cream pixel
	//    If it does, the position of the center of the window is stored in windowCenters
	for (int y = windowSize; y < imgClone.rows - windowSize; y++) {
		for (int x = windowSize; x < imgClone.cols - windowSize; x++) {

			red = false;
			green = false;
			cream = false;

			for (int i = -windowSize; i <= windowSize; i++) {
				for (int j = -windowSize; j <= windowSize; j++) {

					//red
					if (minHRed < imgClone.at<Vec3b>(y + i, x + j)[0] || imgClone.at<Vec3b>(y + i, x + j)[0] < maxHRed &&
						minSRed < imgClone.at<Vec3b>(y + i, x + j)[1] && imgClone.at<Vec3b>(y + i, x + j)[1] < maxSRed &&
						minVRed < imgClone.at<Vec3b>(y + i, x + j)[2] && imgClone.at<Vec3b>(y + i, x + j)[2] < maxVRed) {
						red = true;
					}

					//green
					if (minHGreen < imgClone.at<Vec3b>(y + i, x + j)[0] && imgClone.at<Vec3b>(y + i, x + j)[0] < maxHGreen &&
						minSGreen < imgClone.at<Vec3b>(y + i, x + j)[1] && imgClone.at<Vec3b>(y + i, x + j)[1] < maxSGreen &&
						minVGreen < imgClone.at<Vec3b>(y + i, x + j)[2] && imgClone.at<Vec3b>(y + i, x + j)[2] < maxVGreen) {
						green = true;
					}

					//cream
					if (minHCream < imgClone.at<Vec3b>(y + i, x + j)[0] && imgClone.at<Vec3b>(y + i, x + j)[0] < maxHCream &&
						minSCream < imgClone.at<Vec3b>(y + i, x + j)[1] && imgClone.at<Vec3b>(y + i, x + j)[1] < maxSCream &&
						minVCream < imgClone.at<Vec3b>(y + i, x + j)[2] && imgClone.at<Vec3b>(y + i, x + j)[2] < maxVCream) {
						cream = true;
					}
				}
			}

			if (red && green && cream) {
				windowCenters.push_back(Point(x, y));
			}

		}
	}

	// 3. Here we generate centerData, these points wwill be used in our clustering algorithm
	if (windowCenters.size() > 0) {
		for (int i = 0; i < windowCenters.size(); i++) {

			centerData = Mat(windowCenters.size(), 2, CV_32F);
			centerData.at<float>(i, 0) = windowCenters[i].x;
			centerData.at<float>(i, 1) = windowCenters[i].y;

		}
	}

	// 4. Here we cluster our white pixels 

	vector<vector<Point>> closePointVertices;							// closePointVertices will contain vectors of points considered to be clusters 
	closePointVertices.push_back(vector<Point>{Point(100, 200)});		// we have to start with an arbitrary Point
	
	for (auto const &a : windowCenters) {

		size_t size = closePointVertices.size();
		bool is_added = false; 
		
		for (size_t i = 0; i < size; ++i) {
				
			if (distance(closePointVertices[i][0], a) < threshold) {	// if windowCenters[a] is close to another point in closePointVertices, it is stored in that vector of closePointVertices
				closePointVertices[i].push_back(a);						// we add the nearby point location in the same dimension closePointVertices[i]
				is_added = true;
			}
		}

		if (!is_added) { 
			closePointVertices.push_back(vector<Point>{a});
		}																// if no nearby point found in closePointVertices then we add an other vector<Point> to closePointVertices
	}

	// 5. Here we find the average position of the points in each cluster

	for (auto const &v : closePointVertices) {
		averageClusterPositions.push_back(Average(v));
	}

	// 6. Here we create frames of various sizes around these averages, storing them in newFrames

	if (averageClusterPositions.size() > 0) {
		for (int i = 0; i < averageClusterPositions.size(); i++) {

			averageClusterPositionPlot.at<uchar>(averageClusterPositions[i]) = 255;
			prevFrameSize = 0;

			for (int siz = 25; siz <= 75; siz += 25) {

				frameSize = min((min(averageClusterPositions[i].y, averageClusterPositions[i].x)), (min(img.cols - 1 - averageClusterPositions[i].x, img.rows - 1 - averageClusterPositions[i].y)));
				frameSize = min(frameSize, siz);

				if (abs(frameSize - prevFrameSize) > 15) {

					bottomLeft = Point(averageClusterPositions[i].x - frameSize, averageClusterPositions[i].y - frameSize);
					topRight = Point(averageClusterPositions[i].x + frameSize, averageClusterPositions[i].y + frameSize);

					newFrames.push_back(make_pair(bottomLeft, topRight));

					rectangle(imgOrig, bottomLeft, topRight, Scalar(180, 105, 255), 3);
				}

				prevFrameSize = frameSize;
			}

			circle(imgOrig, averageClusterPositions[i], 3, Scalar(180, 105, 255), -1);
		}
	}

	return (make_pair(imgOrig, newFrames));

}