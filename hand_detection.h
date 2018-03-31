#include <opencv2/opencv.hpp>
#include <mutex>
#include <vector>
#include <string>
#include <unistd.h>
#include "point_calculator.h"

using namespace cv;
using namespace std;

class HandDetection {
    
public:
    
    const string RIGHT = "RIGHT";
    const string LEFT = "LEFT";
    const string UP = "UP";
    const string DOWN = "DOWN";
    const string NOT_MOVING = "IDLE";
    
    HandDetection() {
        running = false;
        
        Mat black(320, 240, CV_8UC3, Scalar(0,0,0));
        currentFrame = black;
        background = black;
        foreground = black;
    }
    
    void start() {
        running = true;
        int backgroundLearningTimes = BACKGROUND_LEARNING_TIMES;
        
        VideoCapture cap(0);
        backgroundSubtractorSetup();
        
        vector< pair<Point, double> > palmCenters;
        
        for(;;) {
            Mat frame;
            cap >> frame;
            
            updateBackgroundLearning(frame, backgroundLearningTimes);
            refreshBackgroundImage();
            enhanceForegroundImage();
            
            vector< vector<Point> > allContours = findContoursFromForeground();
            
            for (int i = 0; i < allContours.size(); i++) {
                handDrawing(frame, allContours[i], palmCenters);
            }
            
            #if defined TEST
            drawFrame(frame, backgroundLearningTimes);
            #endif
            
            mtx.lock();
            currentFrame = frame;
            background = backgroundImage;
            foreground = foregroundImage;
            mtx.unlock();
            
            if (!running) {
                cap.release();
                break;
            }
            
            usleep(10 * 1000);
        }
    }
    
    void stop() {
        running = false;
    }
    
    Mat getFrame() {
        return currentFrame;
    }
    
    Mat getBackground() {
        return background;
    }
    
    Mat getForeground() {
        return foreground;
    }
    
    int getFingers() {
        return fingers;
    }
    
    string getDirection() {
        return direction;
    }
    
private:
    
    const static int BACKGROUND_LEARNING_TIMES = 500;
    const static int CONTOUR_MIN_AREA_SIZE = 5000;
    const static int MIN_HORIZONTAL_DIST = 500;
    const static int MIN_VERTICAL_DIST = 400;
    
    int fingers;
    bool running;
    string direction;
    
    mutex mtx;
    Mat currentFrame;
    Mat background;
    Mat foreground;
    
    Mat backgroundImage;
    Mat foregroundImage;
    Ptr<BackgroundSubtractorMOG2> backgroundSubtractor;
    
    void backgroundSubtractorSetup() {
        backgroundSubtractor = createBackgroundSubtractorMOG2();
        backgroundSubtractor->setNMixtures(3);
        backgroundSubtractor->setDetectShadows(false);
    }
    
    void updateBackgroundLearning(Mat frame, int &learningTimes) {
        if (learningTimes > 0) {
            backgroundSubtractor->apply(frame, foregroundImage);
            learningTimes--;
        } else
            backgroundSubtractor->apply(frame, foregroundImage, 0);
    }
    
    void refreshBackgroundImage() {
        backgroundSubtractor->getBackgroundImage(backgroundImage);
    }
    
    void enhanceForegroundImage() {
        erode(foregroundImage, foregroundImage, Mat());
        dilate(foregroundImage, foregroundImage, Mat());
    }
    
    vector< vector<Point> > findContoursFromForeground() {
        vector< vector<Point> > contours;
        findContours(foregroundImage, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
        return contours;
    }
    
    vector< vector<Point> > drawHandOutline(Mat frame, vector<Point> contourToStore) {
        vector< vector<Point> > contourToDraw;
        contourToDraw.push_back(contourToStore);
        drawContours(frame, contourToDraw, -1, cv::Scalar(0,0, 255), 2);
        return contourToDraw;
    }
    
    vector< vector<int> > drawHulls(Mat frame, vector<Point> contour) {
        vector< vector<Point> > hulls(1);
        vector< vector<int> > hullsI(1);
        convexHull(Mat(contour), hulls[0], false);
        convexHull(Mat(contour), hullsI[0], false);
        drawContours(frame, hulls, -1, cv::Scalar(0, 255, 0), 2);
        return hullsI;
    }
    
    void drawRectEnclosingHand(Mat frame, Mat contour) {
        RotatedRect rect = minAreaRect(contour);
        Point2f rectPoints[4];
        rect.points(rectPoints);
        for (int i = 0; i < 4; i++)
            line(frame, rectPoints[i], rectPoints[(i+1)%4], Scalar(255, 0, 0), 1, 8);
    }
    
    void calculatePalmPointsAndCenter(vector< vector<Point> > contour, vector<Vec4i> defects, vector<Point> &palmPoints, Point &palmCenter) {
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
    
    vector< pair<double,int> > getDistanceVector(vector<Point> palmPoints, Point palmCenter) {
        vector< pair<double,int> > distvec;
        for(int i = 0; i < palmPoints.size(); i++)
            distvec.push_back(make_pair(euclideanDistance(palmCenter, palmPoints[i]), i));
        sort(distvec.begin(), distvec.end());
        return distvec;
    }
    
    pair<Point, double> getCircle(vector< pair<double,int> > distvec, vector<Point> points) {
        pair<Point,double> circle;
        for(int i = 0; i + 2 < distvec.size(); i++) {
            Point p1 = points[distvec[i+0].second];
            Point p2 = points[distvec[i+1].second];
            Point p3 = points[distvec[i+2].second];
            circle = circleFromPoints(p1, p2, p3);
            if(circle.second != 0) break;
        }
        return circle;
    }
    
    void updatePalmCenters(vector< pair<Point,double> > &centers, pair<Point,double> circle) {
        centers.push_back(circle);
        if(centers.size() > 10)
            centers.erase(centers.begin());
    }
    
    string calculateMovingDirection(Point start, Point end) {
        string movement = NOT_MOVING;
        int distanceX = end.x - start.x;
        int distanceY = end.y - start.y;
        
        if (abs(distanceX)  > MIN_HORIZONTAL_DIST)
            movement = distanceX > 0 ? RIGHT : LEFT;
        else if (abs(distanceY) > MIN_VERTICAL_DIST)
            movement = distanceY > 0 ? DOWN : UP;
        
        mtx.lock();
        direction = movement;
        mtx.unlock();
        
        return movement;
    }
    
    void drawPalmCircle(Mat frame, vector< pair<Point, double> > centers, Point &center, double &radius) {
        for (int i = 0; i < centers.size(); i++) {
            center += centers[i].first;
            radius += centers[i].second;
        }
        center.x /= centers.size();
        center.y /= centers.size();
        radius /= centers.size();
        
        circle(frame, center, 5, Scalar(144,144,255), 3);
        circle(frame, center, radius, Scalar(144,144,255), 2);
    }
    
    bool isFinger(Mat frame, double length, double distY, double distX, double retLength, double radius, Point ptEnd, Point ptFar) {
        if (length <= 3*radius && distY >= 0.4*radius && length >= 10 && retLength >= 10 && max(length, retLength)/min(length, retLength) >= 0.8) {
            if (min(distX,distY)/max(distX,distY) <= 0.8) {
                if ((distX >= 0.1*radius && distX <= 1.3*radius && distX < distY) || (distY >= 0.1*radius && distY <= 1.3*radius && distX > distY)) {
                    line(frame, ptEnd, ptFar, Scalar(0,255,0), 1);
                    return true;
                }
            }
        }
        return false;
    }
    
    int calculateNumberOfFingers(Mat frame, vector<Vec4i> defects, vector<Point> pointvec, Point palmCenter, double radius) {
        int numOfFingers = 0;
        for (int i = 0; i < defects.size(); i++) {
            int startidx = defects[i][0];
            Point ptStart(pointvec[startidx]);
            
            int endidx = defects[i][1];
            Point ptEnd(pointvec[endidx]);
            
            int faridx = defects[i][2];
            Point ptFar(pointvec[faridx]);
            
            double distX = sqrt(euclideanDistance(palmCenter, ptFar));
            double distY = sqrt(euclideanDistance(palmCenter, ptStart));
            double length = sqrt(euclideanDistance(ptFar, ptStart));
            double retLength = sqrt(euclideanDistance(ptEnd, ptFar));
            
            if (isFinger(frame, length, distY, distX, retLength, radius, ptEnd, ptFar)) {
                line(frame, ptEnd, ptFar, Scalar(0,255,0), 1);
                numOfFingers++;
            }
        }
        numOfFingers = min(5, numOfFingers);
        fingers = numOfFingers;
        return numOfFingers;
    }
    
    void handDrawing(Mat frame, vector<Point> pointvec, vector< pair<Point, double> > centers) {
        if (contourArea(pointvec) < CONTOUR_MIN_AREA_SIZE) return;
        
        vector< vector<Point> > currentCountour = drawHandOutline(frame, pointvec);
        vector< vector<int> > hullsI = drawHulls(frame, currentCountour[0]);
        
        if (hullsI[0].size() > 0) {
            drawRectEnclosingHand(frame, Mat(currentCountour[0]));
            
            vector<Vec4i> defects;
            convexityDefects(currentCountour[0], hullsI[0], defects);
            
            if (defects.size() >= 3) {
                vector<Point> palmPoints;
                Point roughPalmCenter;
                calculatePalmPointsAndCenter(currentCountour, defects, palmPoints, roughPalmCenter);
                
                vector< pair<double,int> > distvec = getDistanceVector(palmPoints, roughPalmCenter);
                pair<Point,double> solnCircle = getCircle(distvec, palmPoints);
                
                updatePalmCenters(centers, solnCircle);
                
                Point palmCenter;
                double radius = 0;
                drawPalmCircle(frame, centers, palmCenter, radius);
                
                calculateMovingDirection(centers[0].first, centers[9].first);
                calculateNumberOfFingers(frame, defects, currentCountour[0], palmCenter, radius);
            }
        }
    }
    
    void drawFrame(Mat frame, int learningTimes) {
        if (learningTimes > 0)
            putText(frame, "Recording Background", cvPoint(30,30), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200,200,250), 1, CV_AA);
        putText(frame, "Number of Fingers: " + to_string(fingers), cvPoint(30,50), FONT_HERSHEY_COMPLEX_SMALL, 0.5, cvScalar(200,200,250), 1, CV_AA);
        putText(frame, "Moving Direction: " + direction, cvPoint(30,70), FONT_HERSHEY_COMPLEX_SMALL, 0.5, cvScalar(200,200,250), 1, CV_AA);
    }
    
};
