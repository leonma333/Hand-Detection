# Hand-Detection

<img alt="opencv" src="https://qph.fs.quoracdn.net/main-qimg-748316a749bdb46f5cdbe02e976e5500" height="40" hspace="10"> <img alt="c++" src="https://raw.githubusercontent.com/isocpp/logos/master/cpp_logo.png" height="40" hspace="10"> <img alt="mit license" src="https://pre00.deviantart.net/4938/th/pre/f/2016/070/3/b/mit_license_logo_by_excaliburzero-d9ur2lg.png" height="40" hspace="10">

**HandDetection** is a light weighted C++ library to track hands using [OpenCV](https://opencv.org/).

## Requriments

`C++` (11 or above) and `OpenCV` (2 or above) installed.

## How does it work?

The library using [background subtraction](https://docs.opencv.org/3.2.0/d1/dc5/tutorial_background_subtraction.html) method to detect hand(s) to do the tracking. The pictures below show an instance of the frame and foreground from the library.

Current Frame                                             |Foreground Frame
:--------------------------------------------------------:|:-------------------------------------------------------------:
![hand detection frame](https://i.imgur.com/TsJ9fsR.jpg)  |  ![hand detection foreground](https://i.imgur.com/BwoV6Oe.jpg)

With clear and clean foreground images, accurate analysis can be done. For example, finger tip locations, palm center, movement, etc, endless things can be acheived.

Compare to pure skin color detection, it is more adaptable, as it does not need to be calibrated to adapt to different skin colors and surrounding. Therefore, as long as the background is stationary, the background subtraction will always work like a charm.

## Features

**HandDetection** class so far provides the following functions:

- `start()` - start the detector
- `stop()` - stop the detector
- `getFrame()` - get the current frame from webcam
- `getBackground()` - get the current background frame by using background subtraction
- `getForeground()` - get the current foreground frame by using background subtraction
- `getFingers()` - get the number of fingers are currently up
- `getDirection()` - get the current hand moving direction (beta)

`main.cpp` already shown an example usage of the **HandDetection** class, but the following code snippet is provided to give a more comprehensive example:

``` C++
#include <thread>
#include "hand_detection.h"

using namespace std;

int main(int argc, char *argv[]) {
    HandDetection handDetector;
    
    // always put start() in another thread
    thread t(&HandDetection::start, &handDetector);
    
    namedWindow("Frame");
    namedWindow("Foreground");
    namedWindow("Background");
    
    while(1) {
        imshow("Frame", handDetector.getFrame());
        imshow("Foreground", handDetector.getForeground());
        imshow("Background", handDetector.getBackground());
        
        cout << "Finger Number: " << handDetector.getFingers() << endl;
        cout << "Moving Ditection: " << handDetector.getDirection() << endl;
        
        if (waitKey(10) >= 0) {
            handDetector.stop();
            break;
        }
    }
    
    t.join();
    
    return 0;
}
```

Enjoy ヽ(^o^)ノ
