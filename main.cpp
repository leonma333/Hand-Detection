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
vector<pair<Point, double>> palmCenters;
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

/* ------------------------------- */

int main(int argc, char *argv[]) {
    int backgroundLearningTimes = BACKGROUND_LEARNING_TIMES;
    
    VideoCapture cap(0);
    backgroundSubtractorSetup();
    createWindowsToDisplayFramesAndBackground();
    
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
                
                // Find minimum area rectangle to enclose hand
                RotatedRect rect = minAreaRect(Mat(currentCountour[0]));
                
                // Find Convex Defects
                vector<Vec4i> defects;
                if (hullsI[0].size() > 0) {
                    // Draw rectangle enclosing hand
                    Point2f rect_points[4];
                    rect.points(rect_points);
                    for(int j = 0; j < 4; j++)
                        line(currentFrame, rect_points[j], rect_points[(j+1)%4], Scalar(255,0,0), 1, 8);
                    
                    Point rough_palm_center;
                    convexityDefects(currentCountour[0], hullsI[0], defects);
                    if(defects.size() >= 3) {
                        vector<Point> palm_points;
                        for(int j = 0;j < defects.size(); j++) {
                            int startidx=defects[j][0];
                            Point ptStart(currentCountour[0][startidx]);
                            
                            int endidx=defects[j][1];
                            Point ptEnd(currentCountour[0][endidx]);
                            
                            int faridx=defects[j][2];
                            Point ptFar(currentCountour[0][faridx]);
                            
                            // Sum up all the hull and defect points to compute average
                            rough_palm_center += ptFar + ptStart + ptEnd;
                            palm_points.push_back(ptFar);
                            palm_points.push_back(ptStart);
                            palm_points.push_back(ptEnd);
                        }
                        
                        // Get palm center by 1st getting the average of all defect points, this is the rough palm center,
                        // Then U chose the closest 3 points ang get the circle radius and center formed from them which is the palm center.
                        rough_palm_center.x /= defects.size()*3;
                        rough_palm_center.y /= defects.size()*3;
                        Point closest_pt=palm_points[0];
                        vector<pair<double,int>> distvec;
                        for(int i = 0; i < palm_points.size(); i++)
                            distvec.push_back(make_pair(euclideanDistance(rough_palm_center, palm_points[i]), i));
                        sort(distvec.begin(), distvec.end());
                        
                        // Keep choosing 3 points till you find a circle with a valid radius
                        // As there is a high chance that the closes points might be in a linear line or too close that it forms a very large circle
                        pair<Point,double> soln_circle;
                        for(int i = 0; i + 2 < distvec.size(); i++) {
                            Point p1=palm_points[distvec[i+0].second];
                            Point p2=palm_points[distvec[i+1].second];
                            Point p3=palm_points[distvec[i+2].second];
                            soln_circle = circleFromPoints(p1, p2, p3); //Final palm center,radius
                            if(soln_circle.second!=0) break;
                        }
                        
                        // Find avg palm centers for the last few frames to stabilize its centers, also find the avg radius
                        palmCenters.push_back(soln_circle);
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
