#include <iostream>     // std::cout
#include <sstream>
#include <fstream>      // std::ifstream
#include <stdio.h>
#include <string>
#include <streambuf>
#include <regex>
#include <math.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>

using namespace dlib;
using namespace std;

//svm을 기반으로 landmark training xml의 face rect 사이즈를 다시 잡아줌
//todo 하드코딩된 값을 수정필요
int main(int argc, char** argv) {
    
    if (argc != 2)
    {
        cout << "Call this program like this:" << endl;
        cout << "./modifyLandmarkTrainingXml resource/landmark.xml resource/modify.xml" << endl;
//        return 0;
    }

    std::string path = "/Users/hongkeunyoo/Source/dlib-skt/resource/ibug_300W_large_face_landmark_dataset/landmark_11_0_15.xml";
//    std::string path = "/Users/hongkeunyoo/Source/dlib-skt/resource/ibug_300W_large_face_landmark_dataset/test.xml";
    std::string outputPath = "/Users/hongkeunyoo/Source/dlib-skt/resource/ibug_300W_large_face_landmark_dataset/landmark_11_0_15_modify8.xml";
    
    if (argc == 2) {
        path = argv[0];
        outputPath = argv[1];
    }
    cout << "argv[0] :" << argv[0] << endl;
    
    std::ifstream file(path);
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    
    std::stringstream ss(content);
    std::string to;
    
    std::stringstream result;
    
    string frontSvm = "/Users/hongkeunyoo/Source/dlib-skt/examples/build/test_data/svm/object_detector_20180418_0_15_front.svm";
    string leftSvm = "/Users/hongkeunyoo/Source/dlib-skt/examples/build/test_data/svm/object_detector_20180418_0_15_left.svm";
    string rightSvm = "/Users/hongkeunyoo/Source/dlib-skt/examples/build/test_data/svm/object_detector_20180418_0_15_right.svm";
    string leftRotateSvm = "/Users/hongkeunyoo/Source/dlib-skt/examples/build/test_data/svm/object_detector_20180418_0_15_leftRotate.svm";
    string rightRotateSvm = "/Users/hongkeunyoo/Source/dlib-skt/examples/build/test_data/svm/object_detector_20180418_0_15_rightRotate.svm";
    string faceUpSvm = "/Users/hongkeunyoo/Source/dlib-skt/examples/build/test_data/svm/detector_fu4_20180316.svm";
    string faceDownSvm = "/Users/hongkeunyoo/Source/dlib-skt/examples/build/test_data/svm/detector_fd3_20180315.svm";
    string faceAbsoulteLeftSvm = "/Users/hongkeunyoo/Source/dlib-skt/examples/build/test_data/svm/detector_left3.svm";
    string faceAbsoulteRightSvm = "/Users/hongkeunyoo/Source/dlib-skt/examples/build/test_data/svm/detector_right3.svm";

    dlib::frontal_face_detector front; dlib::deserialize(frontSvm) >> front;
    dlib::frontal_face_detector left; dlib::deserialize(leftSvm) >> left;
    dlib::frontal_face_detector right; dlib::deserialize(rightSvm) >> right;
    dlib::frontal_face_detector left_rotate_front; dlib::deserialize(leftRotateSvm) >> left_rotate_front;
    dlib::frontal_face_detector right_rotate_front; dlib::deserialize(rightRotateSvm) >> right_rotate_front;
    dlib::frontal_face_detector frontUp; dlib::deserialize(faceUpSvm) >> frontUp;
    dlib::frontal_face_detector frontDown; dlib::deserialize(faceDownSvm) >> frontDown;
    dlib::frontal_face_detector faceAbsoulteLeft; dlib::deserialize(faceAbsoulteLeftSvm) >> faceAbsoulteLeft;
    dlib::frontal_face_detector faceAbsoulteRight; dlib::deserialize(faceAbsoulteRightSvm) >> faceAbsoulteRight;
    
    std::vector<dlib::frontal_face_detector> detectors;
    // 1. front
    // 2. front up
    // 3. front down
    // 4. left
    // 5. absolutely left
    // 6. right
    // 7. absolutely right
    // 8. left rotated front
    // 9. right rotate front
    detectors.push_back(front);
    detectors.push_back(frontUp);
    detectors.push_back(frontDown);
    detectors.push_back(left);
    detectors.push_back(faceAbsoulteLeft);
    detectors.push_back(right);
    detectors.push_back(faceAbsoulteRight);
    detectors.push_back(left_rotate_front);
    detectors.push_back(right_rotate_front);
    
    dlib::frontal_face_detector detector = dlib::frontal_face_detector(detectors);
//    dlib::frontal_face_detector detector = get_frontal_face_detector();
    string imagePrefix = "/Users/hongkeunyoo/Source/dlib-skt/resource/ibug_300W_large_face_landmark_dataset/";
    int count = 0;
    int modifyCount = 0;
    int skipCount = 0;
    int lineCount = 0;
    int imageLineCount = 0;
    array2d<rgb_pixel> img;
    
    while(std::getline(ss,to,'\n'))
    {
        std::string imageString("image file=");
        std::regex imageValueRegex(".*image file=\'(.*?)\'.*");
        std::smatch match;
        
        bool isImageFound = false;
        lineCount++;
        
        if (to.find(imageString) != std::string::npos) {
            if (std::regex_match(to, match, imageValueRegex))
            {
                if (match.size() > 1) {
                    std::ssub_match sub_match = match[1];
                    std::string imagePath = sub_match.str();
                    count++;
                    std::cout << "count : " << count << ", imagePath : " << imagePrefix + imagePath <<std::endl;
                    load_image(img, imagePrefix + imagePath);
                    
//                    pyramid_up(img);
                    
                    isImageFound = true;
                    imageLineCount = lineCount;
                    
                } else {
                    isImageFound = false;
                }
            } else {
                isImageFound = false;
            }
        }
        
        // rect 크기는 image xml 바로 아래에 위치함
        if (lineCount == (imageLineCount + 1)) {
            std::vector<rectangle> dets = detector(img);
            if (dets.size() == 1) {
                cout << "before : " << to << endl;
//                cout << "top :" << dets[0].top() <<  " , left :"<< dets[0].left() <<
//                ", width :" << dets[0].width() <<  " , height :"<< dets[0].height() <<
//                endl;
                std::string widthString("width");
                std::regex widthValueRegex(".*width=\'([0-9]+).*");
                std::string widthPrefix("width='");
                
                std::string heightString("height");
                std::regex heightValueRegex(".*height=\'([0-9]+).*");
                std::string heightPrefix("height='");
                
                std::string topString("top");
                std::regex topValueRegex(".*top=\'([0-9]+).*");
                std::string topPrefix("top='");
                
                std::string leftString("left");
                std::regex leftValueRegex(".*left=\'([0-9]+).*");
                std::string leftPrefix("left='");
                
                std::smatch match;
                int modifyValue;
                
                if (to.find(widthString) != std::string::npos) {
                    
                    std::string topString = "";
                    std::string leftString;
                    
                    //top 값 조절
                    if (std::regex_match(to, match, topValueRegex))
                    {
                        if (match.size() > 1) {
                            std::ssub_match sub_match = match[1];
                            topString = sub_match.str();
                            int topInt = atoi(topString.c_str());
                            cout << "topString00 : " << topString << endl;
                            if (abs(topInt-dets[0].top()) > 60) {
                                cout << "skip1 top :" << dets[0].top() <<  " , left :" << dets[0].left() << ", width :" << dets[0].width() <<  " , height :"<< dets[0].height() << endl;
                                skipCount++;
                                topString = "";
                                result << to << std::endl;
                                continue;
                            }
                        }
                    }
                    
                    //left 값 조절
                    if (std::regex_match(to, match, leftValueRegex))
                    {
                        if (match.size() > 1) {
                            std::ssub_match sub_match = match[1];
                            leftString = sub_match.str();
                            int leftInt = atoi(leftString.c_str());
                            if (abs(leftInt-dets[0].left()) > 60) {
                                cout << "skip2 top :" << dets[0].top() <<  " , left :" << dets[0].left() << ", width :" << dets[0].width() <<  " , height :"<< dets[0].height() << endl;
                                skipCount++;
                                result << to << endl;
                                continue;
                            }
                        }
                        
                        if (topString.length() != 0) {
                            to.replace(to.find(topPrefix + topString), string(topString).size() + 5,  topPrefix + to_string(dets[0].top()));
                            cout << "topString11 : " << topString << endl;
                            to.replace(to.find(leftPrefix + leftString), string(leftString).size() + 6,  leftPrefix + to_string(dets[0].left()));
                        } else {
                            cout << "skip3 top :" << dets[0].top() <<  " , left :" << dets[0].left() << ", width :" << dets[0].width() <<  " , height :"<< dets[0].height() << endl;
                            skipCount++;
                            result << to << endl;
                            continue;
                        }
                    }
                    
                    //width 값 조절
                    if (std::regex_match(to, match, widthValueRegex))
                    {
                        if (match.size() > 1) {
                            std::ssub_match sub_match = match[1];
                            std::string widthString = sub_match.str();
                            
                            to.replace(to.find(widthPrefix + widthString), string(widthString).size() + 7,  widthPrefix + to_string(dets[0].width()));
                        }
                    }
                    //height 값 조절
                    if (std::regex_match(to, match, heightValueRegex))
                    {
                        if (match.size() > 1) {
                            std::ssub_match sub_match = match[1];
                            std::string heightString = sub_match.str();
                            
                            to.replace(to.find(heightPrefix + heightString), string(heightString).size() + 8,  heightPrefix + to_string(dets[0].height()));
                        }
                    }
                    modifyCount++;
                    std::cout << "after : " << to << endl;
                }
            }
        }
        
        result << to << std::endl;
    }
    cout << "modifyCount : " << modifyCount << ", skipCount : " << skipCount << endl;
    
    std::ofstream out(outputPath);
    
    out << result.str();
    out.close();
	return 0;
}
