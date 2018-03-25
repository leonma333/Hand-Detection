#include <thread>
#include "hand_detection.h"

using namespace std;

int main(int argc, char *argv[]) {
    HandDetection handDetector;
    
    thread t(&HandDetection::start, &handDetector);
    
    namedWindow("Frame");
    namedWindow("Background");
    
    while(1) {
        imshow("Frame", handDetector.getFrame());
        imshow("Background", handDetector.getBackground());
        if (waitKey(10) >= 0) {
            handDetector.stop();
            break;
        }
    }
    
    t.join();
    
    return 0;
}
