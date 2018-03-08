#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include "point_calculator.h"

using namespace std;
using namespace cv;

/* ------------------------------- */

const int BACKGROUND_LEARNING_TIMES = 500;
const int CONTOUR_MIN_AREA_SIZE = 5000;

Mat currentFrame;
Mat backgroundImage;
Mat foregroundImage;
Ptr<BackgroundSubtractorMOG2> backgroundSubtractor;

void backgroundSubtractorSetup() {
    backgroundSubtractor = createBackgroundSubtractorMOG2();
    backgroundSubtractor->setNMixtures(3);
    backgroundSubtractor->setDetectShadows(false);
}

void createWindowsToDisplayFramesAndBackground() {
    namedWindow("Frame");
    namedWindow("Background");
}

void updateBackgroundLearning(int &learningTimes) {
    if(learningTimes > 0) {
        backgroundSubtractor->apply(currentFrame, foregroundImage);
        learningTimes--;
    } else
        backgroundSubtractor->apply(currentFrame, foregroundImage, 0);
}

void refreshBackgroundImage() {
    backgroundSubtractor->getBackgroundImage(backgroundImage);
}

void enhanceForegroundImage() {
    erode(foregroundImage, foregroundImage, Mat());
    dilate(foregroundImage, foregroundImage, Mat());
}

vector<vector<Point>> findContoursFromForeground() {
    vector<vector<Point>> contours;
    findContours(foregroundImage, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    return contours;
}

vector<vector<Point>> drawHandOutline(vector<Point> contourToStore) {
    vector<vector<Point>> contourToDraw;
    contourToDraw.push_back(contourToStore);
    drawContours(currentFrame, contourToDraw, -1, cv::Scalar(0,0, 255), 2);
    return contourToDraw;
}

vector<vector<int>> drawHulls(vector<Point> contour) {
    vector<vector<Point>> hulls(1);
    vector<vector<int>> hullsI(1);
    convexHull(Mat(contour), hulls[0], false);
    convexHull(Mat(contour), hullsI[0], false);
    drawContours(currentFrame, hulls, -1, cv::Scalar(0, 255, 0), 2);
    return hullsI;
}

void drawRectEnclosingHand(Mat contour) {
    RotatedRect rect = minAreaRect(contour);
    Point2f rectPoints[4];
    rect.points(rectPoints);
    for(int i = 0; i < 4; i++)
        line(currentFrame, rectPoints[i], rectPoints[(i+1)%4], Scalar(255, 0, 0), 1, 8);
}

void calculatePalmPointsAndCenter(vector<vector<Point>> contour, vector<Vec4i> defects, vector<Point> &palmPoints, Point &palmCenter) {
    for (int i = 0; i < defects.size(); i++) {
        int startidx = defects[i][0];
        Point ptStart(contour[0][startidx]);
        
        int endidx = defects[i][1];
        Point ptEnd(contour[0][endidx]);
        
        int faridx = defects[i][2];
        Point ptFar(contour[0][faridx]);
        
        palmCenter += ptFar + ptStart + ptEnd;
        palmPoints.push_back(ptFar);
        palmPoints.push_back(ptStart);
        palmPoints.push_back(ptEnd);
    }
    
    palmCenter.x /= defects.size()*3;
    palmCenter.y /= defects.size()*3;
}

    vector<pair<double,int>> getDistanceVector(vector<Point> palmPoints, Point palmCenter) {
        vector<pair<double,int>> distvec;
        for(int i = 0; i < palmPoints.size(); i++)
            distvec.push_back(make_pair(euclideanDistance(palmCenter, palmPoints[i]), i));
        sort(distvec.begin(), distvec.end());
        return distvec;
    }

/* ------------------------------- */

int main(int argc, char *argv[]) {
    int backgroundLearningTimes = BACKGROUND_LEARNING_TIMES;
    
    VideoCapture cap(0);
    backgroundSubtractorSetup();
    createWindowsToDisplayFramesAndBackground();
    
    vector<pair<Point, double>> palmCenters;
    
    for(;;) {
        cap >> currentFrame;
        
        updateBackgroundLearning(backgroundLearningTimes);
        refreshBackgroundImage();
        enhanceForegroundImage();
        
        vector<vector<Point>> allContours = findContoursFromForeground();
        
        for (int i = 0; i < allContours.size(); i++) {
            if(contourArea(allContours[i]) >= CONTOUR_MIN_AREA_SIZE) {
                vector<vector<Point>> currentCountour = drawHandOutline(allContours[i]);
                vector<vector<int>> hullsI = drawHulls(currentCountour[0]);
                
                if (hullsI[0].size() > 0) {
                    drawRectEnclosingHand(Mat(currentCountour[0]));
                    
                    vector<Vec4i> defects;
                    convexityDefects(currentCountour[0], hullsI[0], defects);
                    
                    if(defects.size() >= 3) {
                        vector<Point> palmPoints;
                        Point roughPalmCenter;
                        calculatePalmPointsAndCenter(currentCountour, defects, palmPoints, roughPalmCenter);
                        
                        vector<pair<double,int>> distvec = getDistanceVector(palmPoints, roughPalmCenter);
                        
                        // Keep choosing 3 points till you find a circle with a valid radius
                        // As there is a high chance that the closes points might be in a linear line or too close that it forms a very large circle
                        pair<Point,double> solnCircle;
                        for(int i = 0; i + 2 < distvec.size(); i++) {
                            Point p1 = palmPoints[distvec[i+0].second];
                            Point p2 = palmPoints[distvec[i+1].second];
                            Point p3 = palmPoints[distvec[i+2].second];
                            solnCircle = circleFromPoints(p1, p2, p3); //Final palm center,radius
                            if(solnCircle.second!=0) break;
                        }
                        
                        // Find avg palm centers for the last few frames to stabilize its centers, also find the avg radius
                        palmCenters.push_back(solnCircle);
                        if(palmCenters.size() > 10)
                            palmCenters.erase(palmCenters.begin());
                        
                        Point palm_center;
                        double radius = 0;
                        for(int i = 0; i < palmCenters.size(); i++) {
                            palm_center+=palmCenters[i].first;
                            radius+=palmCenters[i].second;
                        }
                        palm_center.x /= palmCenters.size();
                        palm_center.y /= palmCenters.size();
                        radius /= palmCenters.size();
                        
                        // Draw the palm center and the palm circle
                        // The size of the palm gives the depth of the hand
                        circle(currentFrame, palm_center, 5, Scalar(144,144,255), 3);
                        circle(currentFrame, palm_center, radius, Scalar(144,144,255), 2);
                        
                        // Detect fingers by finding points that form an almost isosceles triangle with certain thesholds
                        int num_of_fingers = 0;
                        for(int j = 0; j < defects.size(); j++) {
                            int startidx = defects[j][0];
                            Point ptStart(currentCountour[0][startidx]);
                            
                            int endidx = defects[j][1];
                            Point ptEnd(currentCountour[0][endidx]);
                            
                            int faridx = defects[j][2];
                            Point ptFar(currentCountour[0][faridx]);
                            
                            // X o--------------------------o Y
                            
                            double Xdist = sqrt(euclideanDistance(palm_center,ptFar));
                            double Ydist = sqrt(euclideanDistance(palm_center,ptStart));
                            double length = sqrt(euclideanDistance(ptFar,ptStart));
                            double retLength = sqrt(euclideanDistance(ptEnd,ptFar));
                            
                            // Play with these thresholds to improve performance
                            if (length <= 3*radius && Ydist >= 0.4*radius && length >= 10 && retLength >= 10 && max(length, retLength)/min(length, retLength) >= 0.8) {
                                if (min(Xdist,Ydist)/max(Xdist,Ydist) <= 0.8) {
                                    if ((Xdist >= 0.1*radius && Xdist <= 1.3*radius && Xdist < Ydist)||(Ydist >= 0.1*radius && Ydist <= 1.3*radius && Xdist > Ydist)) {
                                        line(currentFrame, ptEnd, ptFar, Scalar(0,255,0), 1);
                                        num_of_fingers++;
                                    }
                                }
                            }
                        }
                        
                        num_of_fingers = min(5, num_of_fingers);
                        cout << "NO OF FINGERS: " << num_of_fingers << endl;
                    }
                }
            }
        }
        
        if(backgroundLearningTimes > 0)
            putText(currentFrame, "Recording Background", cvPoint(30,30), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200,200,250), 1, CV_AA);
        imshow("Frame",currentFrame);
        imshow("Background", backgroundImage);
        
        if (waitKey(10) >= 0) break;
    }
    
    return 0;
}
