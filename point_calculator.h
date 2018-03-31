#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

double euclideanDistance(Point x, Point y) {
    return (x.x - y.x) * (x.x - y.x) + (x.y - y.y) * (x.y - y.y);
}

pair<Point, double> circleFromPoints(Point p1, Point p2, Point p3) {
    double offset = pow(p2.x, 2) + pow(p2.y, 2);
    double bc = (pow(p1.x, 2) + pow(p1.y, 2) - offset)/2.0;
    double cd = (offset - pow(p3.x, 2) - pow(p3.y, 2))/2.0;
    double det = (p1.x - p2.x) * (p2.y - p3.y) - (p2.x - p3.x)* (p1.y - p2.y);
    double TOL = 0.0000001;
    
    if (abs(det) < TOL) {
        return make_pair(Point(0, 0), 0);
    }
    
    double idet = 1 / det;
    double centerx = (bc * (p2.y - p3.y) - cd * (p1.y - p2.y)) * idet;
    double centery = (cd * (p1.x - p2.x) - bc * (p2.x - p3.x)) * idet;
    double radius = sqrt(pow(p2.x - centerx, 2) + pow(p2.y - centery, 2));
    
    return make_pair(Point(centerx, centery), radius);
}
