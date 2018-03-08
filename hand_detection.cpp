#include <opencv2/opencv.hpp>
#include <vector>
//#include "point_calculator.h"

using namespace cv;
using namespace std;

class HandDetection {
    
public:
    
    const int BACKGROUND_LEARNING_TIMES = 500;
    const int CONTOUR_MIN_AREA_SIZE = 5000;
    
    void start() {
        int backgroundLearningTimes = BACKGROUND_LEARNING_TIMES;
        
        VideoCapture cap(0);
        backgroundSubtractorSetup();
        createWindowsToDisplayFramesAndBackground();
        
        vector<pair<Point, double>> palmCenters;
        
        for (;;) {
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
                            
                            //vector<pair<double,int>> distvec = getDistanceVector(palmPoints, roughPalmCenter);
                        }
                    }
                }
            }
        }
    }

private:
    
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
    
//    vector<pair<double,int>> getDistanceVector(vector<Point> palmPoints, Point palmCenter) {
//        vector<pair<double,int>> distvec;
//        for(int i = 0; i < palmPoints.size(); i++)
//            distvec.push_back(make_pair(euclideanDistance(palmCenter, palmPoints[i]), i));
//        sort(distvec.begin(), distvec.end());
//        return distvec;
//    }
    
};
