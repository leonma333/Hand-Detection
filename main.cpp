#include <thread>
#include "hand_detection.h"

using namespace std;

int main(int argc, char *argv[]) {
    HandDetection handDetector;
    
    thread t(&HandDetection::start, &handDetector);
    
    namedWindow("Frame");
    namedWindow("Foreground");
    
    while(1) {
        imshow("Frame", handDetector.getFrame());
        imshow("Foreground", handDetector.getForeground());
        if (waitKey(10) >= 0) {
            handDetector.stop();
            break;
        }
    }
    
    t.join();
    
    return 0;
}
