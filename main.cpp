#include <thread>
#include "hand_detection.h"

using namespace std;

int main(int argc, char *argv[]) {
    HandDetection handDetector;
    handDetector.start();
    
    return 0;
}
