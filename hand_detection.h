#include <opencv2/opencv.hpp>
#include <vector>
#include <mutex>
#include "point_calculator.h"

using namespace cv;
using namespace std;

class HandDetection {
    
public:
    
    const static int BACKGROUND_LEARNING_TIMES = 500;
    const static int CONTOUR_MIN_AREA_SIZE = 5000;
    
    HandDetection() {
        init();
    }
    
    void reset() {
        if (running) return;
        init();
    }
    
    void start() {
//        backgroundSubtractorSetup();
//        createWindowsToDisplayFramesAndBackground();
//
//        for(;;) {
//            updateBackgroundLearning();
//            refreshBackgroundImage();
//            enhanceForegroundImage();
//
//            vector< vector<Point> > allContours = findContoursFromForeground();
//
//            for (int i = 0; i < allContours.size(); i++) {
//                handDrawing(allContours[i], palmCenters);
//            }
//
//            if (!running) break;
//        }
        
        thread producer_t(&HandDetection::producer, this);
        thread ui_t(&HandDetection::ui, this);
        thread processor_t(&HandDetection::processor, this);
        producer_t.join();
    }
    
    void stop() {
        mtx.lock();
        running = false;
        mtx.unlock();
    }
    
    int getNumberOfFingersUp() {
        return fingersNumber;
    }
    
private:
    
    const static int BUFFER_LENGTH = 100;
    
    mutex mtx;
    bool running;
    int fingersNumber;
    int backgroundLearningTimes;
    vector< pair<Point, double> > palmCenters;
    
    int position;
    int index;
    Mat frameBuffer [BUFFER_LENGTH];
    
    Mat currentFrame;
    Mat backgroundImage;
    Mat foregroundImage;
    Ptr<BackgroundSubtractorMOG2> backgroundSubtractor;
    
    void init() {
        running = false;
        fingersNumber = 0;
        backgroundLearningTimes = BACKGROUND_LEARNING_TIMES;
        palmCenters.clear();
        
        position = 0;
        index = 0;
        
        currentFrame = NULL;
        backgroundImage = NULL;
        foregroundImage = NULL;
        backgroundSubtractor.release();
    }
    
    /* =============================
       Producer thread
       ============================= */
    void producer() {
        VideoCapture cap(0);
        Mat f;
        
        backgroundSubtractorSetup();
        createWindowsToDisplayFramesAndBackground();
        
        while(1) {
            cap >> f;
            putFrameInBuffer(f);
            if (!running) break;
        }
    }
    
    void putFrameInBuffer(Mat &f){
        position = index % BUFFER_LENGTH;
        frameBuffer[position] = f.clone();
        index++;
    }
    
    /* =============================
     UI thread
     ============================= */
    
    void ui(){
        while(1) {
            if (currentFrame.empty()) continue;
            showFrame();
            waitKey(20);
        }
    }
    
    void showFrame(){
        if(backgroundLearningTimes > 0)
            putText(currentFrame, "Recording Background", cvPoint(30,30), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200,200,250), 1, CV_AA);
        imshow("Frame", currentFrame);
        imshow("Background", backgroundImage);
    }
    
    /* =============================
     Processing thread
     ============================= */
    
    void processor() {
        while(1) {
            getCurrentFrame();
            if (currentFrame.empty()) continue;
            processCurrentFrame();
        }
    }
    
    void processCurrentFrame() {
        updateBackgroundLearning();
        refreshBackgroundImage();
        enhanceForegroundImage();
        
        vector< vector<Point> > allContours = findContoursFromForeground();
        
        for (int i = 0; i < allContours.size(); i++) {
            handDrawing(allContours[i], palmCenters);
        }
    }
    
    void getCurrentFrame(){
        int end = position;
        
        mtx.lock();
        currentFrame = frameBuffer[end];
        mtx.unlock();
    }
    
    /* =============================
     Helpers
     ============================= */
    
    void backgroundSubtractorSetup() {
        backgroundSubtractor = createBackgroundSubtractorMOG2();
        backgroundSubtractor->setNMixtures(3);
        backgroundSubtractor->setDetectShadows(false);
    }
    
    void createWindowsToDisplayFramesAndBackground() {
        namedWindow("Frame");
        namedWindow("Background");
    }
    
    void updateBackgroundLearning() {
        if (backgroundLearningTimes > 0) {
            backgroundSubtractor->apply(currentFrame, foregroundImage);
            backgroundLearningTimes--;
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
    
    vector< vector<Point> > findContoursFromForeground() {
        vector< vector<Point> > contours;
        findContours(foregroundImage, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
        return contours;
    }
    
    vector< vector<Point> > drawHandOutline(vector<Point> contourToStore) {
        vector< vector<Point> > contourToDraw;
        contourToDraw.push_back(contourToStore);
        drawContours(currentFrame, contourToDraw, -1, cv::Scalar(0,0, 255), 2);
        return contourToDraw;
    }
    
    vector< vector<int> > drawHulls(vector<Point> contour) {
        vector< vector<Point> > hulls(1);
        vector< vector<int> > hullsI(1);
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
    
    void drawPalmCircle(vector<pair< Point, double> > centers, Point &center, double &radius) {
        for (int i = 0; i < centers.size(); i++) {
            center += centers[i].first;
            radius += centers[i].second;
        }
        center.x /= centers.size();
        center.y /= centers.size();
        radius /= centers.size();
        
        circle(currentFrame, center, 5, Scalar(144,144,255), 3);
        circle(currentFrame, center, radius, Scalar(144,144,255), 2);
    }
    
    bool isFinger(double length, double distY, double distX, double retLength, double radius, Point ptEnd, Point ptFar) {
        if (length <= 3*radius && distY >= 0.4*radius && length >= 10 && retLength >= 10 && max(length, retLength)/min(length, retLength) >= 0.8) {
            if (min(distX,distY)/max(distX,distY) <= 0.8) {
                if ((distX >= 0.1*radius && distX <= 1.3*radius && distX < distY) || (distY >= 0.1*radius && distY <= 1.3*radius && distX > distY)) {
                    line(currentFrame, ptEnd, ptFar, Scalar(0,255,0), 1);
                    return true;
                }
            }
        }
        return false;
    }
    
    int numberOfFingers(vector<Vec4i> defects, vector<Point> pointvec, Point palmCenter, double radius) {
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
            
            if (isFinger(length, distY, distX, retLength, radius, ptEnd, ptFar)) {
                line(currentFrame, ptEnd, ptFar, Scalar(0,255,0), 1);
                numOfFingers++;
            }
        }
        mtx.lock();
        fingersNumber = min(5, numOfFingers);
        mtx.unlock();
        return fingersNumber;
    }
    
    void handDrawing(vector<Point> pointvec, vector< pair<Point, double> > centers) {
        if (contourArea(pointvec) < CONTOUR_MIN_AREA_SIZE) return;
        
        vector< vector<Point> > currentCountour = drawHandOutline(pointvec);
        vector< vector<int> > hullsI = drawHulls(currentCountour[0]);
        
        if (hullsI[0].size() > 0) {
            drawRectEnclosingHand(Mat(currentCountour[0]));
            
            vector<Vec4i> defects;
            convexityDefects(currentCountour[0], hullsI[0], defects);
            
            if(defects.size() >= 3) {
                vector<Point> palmPoints;
                Point roughPalmCenter;
                calculatePalmPointsAndCenter(currentCountour, defects, palmPoints, roughPalmCenter);
                
                vector< pair<double,int> > distvec = getDistanceVector(palmPoints, roughPalmCenter);
                pair<Point,double> solnCircle = getCircle(distvec, palmPoints);
                
                updatePalmCenters(centers, solnCircle);
                
                Point palmCenter;
                double radius = 0;
                drawPalmCircle(centers, palmCenter, radius);
                
                cout << "NO OF FINGERS: " << numberOfFingers(defects, currentCountour[0], palmCenter, radius) << endl;
            }
        }
    }
    
};
