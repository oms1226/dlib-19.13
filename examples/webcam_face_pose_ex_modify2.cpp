// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
 
 This example program shows how to find frontal human faces in an image and
 estimate their pose.  The pose takes the form of 68 landmarks.  These are
 points on the face such as the corners of the mouth, along the eyebrows, on
 the eyes, and so forth.
 
 
 This example is essentially just a version of the face_landmark_detection_ex.cpp
 example modified to use OpenCV's VideoCapture object to read from a camera instead
 of files.
 
 
 Finally, note that the face detector is fastest when compiled with at least
 SSE2 instructions enabled.  So if you are using a PC with an Intel or AMD
 chip then you should enable at least SSE2 instructions.  If you are using
 cmake to compile this program you can enable them by using one of the
 following commands when you create the build project:
 cmake path_to_dlib_root/examples -DUSE_SSE2_INSTRUCTIONS=ON
 cmake path_to_dlib_root/examples -DUSE_SSE4_INSTRUCTIONS=ON
 cmake path_to_dlib_root/examples -DUSE_AVX_INSTRUCTIONS=ON
 This will set the appropriate compiler options for GCC, clang, Visual
 Studio, or the Intel compiler.  If you are using another compiler then you
 need to consult your compiler's manual to determine how to enable these
 instructions.  Note that AVX is the fastest but requires a CPU from at least
 2011.  SSE4 is the next fastest and is supported by most current machines.
 */

#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>

#define USE_WEBCAM 0

#define NEXT_FRAME_WITH_KEY 0
#define NEXT_FRAME_WITH_KEY_ONLY_IF_NOT_DETECTED 0

#define SAVE_IMG_IF_NEEDED 1
#define SAVE_IMG_IF_NOT_DETECTED 1

#define USE_CUSTOM_FD 1
#define DETECT_LANDMARK 1
#define OVERLAY_RENDER_FACE_RECT 1

#define PRINT_RESULT_STATS 1

#define USE_MD 1

using namespace dlib;
using namespace std;

#if !USE_WEBCAM
static int fileIndex = 0;

// 216x384
// 360x640
// 720x1280/Users/hongkeunyoo/Documents/dlib_test/IosTest/15.mp4

static const string filepath = "/Users/hongkeunyoo/Documents/dlib_test/IosTest/";
static const string files[] = {
//    (filepath + "1.m4v"),
//    (filepath + "2.m4v")
//    (filepath + "3.mp4"),
//    (filepath + "4.mp4")
//    (filepath + "5.mp4"),
//    (filepath + "6.mp4"),
//    (filepath + "8.mp4"),
//    (filepath + "9.mp4"),
//    (filepath + "10.mp4"),
//    (filepath + "11.m4v"),
//    (filepath + "12.mp4")
//    (filepath + "test1.mp4"),
    (filepath + "58.mp4"),
    (filepath + "59.mp4"),
    (filepath + "60.mp4"),
    (filepath + "61.mp4"),
    (filepath + "62.mp4"),
    (filepath + "63.mp4"),
    (filepath + "64.mp4"),
    (filepath + "65.mp4"),
    (filepath + "66.mp4")

//
//    (filepath + "13.mp4"),
//    (filepath + "14.mp4"),
//    (filepath + "15.mp4"),
//    (filepath + "16.mp4"),
//    (filepath + "17.mp4"),
//    (filepath + "18.mp4"),
//    (filepath + "19.mp4"),
//    (filepath + "20.mp4"),
//    (filepath + "21.mp4"),
//    (filepath + "22.mp4"),
//    (filepath + "23.mp4"),
//    (filepath + "24.mp4"),
//    (filepath + "25.mp4"),
    
//    ("../../../test_data/test_mv/_1.mp4"),
//    "../../../test_data/test_mv/216x384/w/25/223_wrong.png",
//    "../../../test_data/test_mv/216x384/w/25/224_wrong.png",
};

//static const string files[] = {
//    (filepath + "nd/test1/test%1d.png"),
//    (filepath + "nd/test2/test%1d.png"),
//    (filepath + "nd/test3/test%1d.png"),
//    (filepath + "nd/test4/test%1d.png"),
//    (filepath + "nd/test5/test%1d.png"),
//    (filepath + "nd/test6/test%1d.png"),
//    (filepath + "nd/test8/test%1d.png"),
//    (filepath + "nd/test9/test%1d.png"),
//    (filepath + "nd/test10/test%1d.png"),
//    (filepath + "nd/test11/test%1d.png"),
//    (filepath + "nd/test12/test%1d.png"),
//};

#ifdef SAVE_IMG_IF_NEEDED
//static const string directoryName = "nd/";
static const string directoryName = "dontdetect/";
static const string savefiles[] = {
//    (filepath + directoryName + "1_2/"),
//    (filepath + directoryName + "2_2/")
//    (filepath + directoryName + "3_1/"),
//    (filepath + directoryName + "4_1/")
//    (filepath + directoryName + "5/"),
//    (filepath + directoryName + "6/"),
//    (filepath + directoryName + "8/"),
//    (filepath + directoryName + "9/"),
//    (filepath + directoryName + "10/"),
//    (filepath + directoryName + "11/"),
//    (filepath + directoryName + "12/")
//    (filepath + directoryName + "test1_1/"),
//
    (filepath + directoryName + "58/"),
    (filepath + directoryName + "59/"),
    (filepath + directoryName + "60/"),
    (filepath + directoryName + "61/"),
    (filepath + directoryName + "62/"),
    (filepath + directoryName + "63/"),
    (filepath + directoryName + "64/"),
    (filepath + directoryName + "65/"),
    (filepath + directoryName + "66/")
//    (filepath + directoryName + "21/"),
//    (filepath + directoryName + "22/"),
//    (filepath + directoryName + "23/"),
//    (filepath + directoryName + "24/"),
//    (filepath + directoryName + "25/"),
};
#endif

//  "../../../test_data/gif/file_%1d.png"
//  "../../../test_data/gif/%1d.png"

#endif

bool moreData() {
#if USE_WEBCAM
    return false;
#else
    const int count = sizeof(files)/sizeof(files[0]);
    return fileIndex < count;
#endif
}


#if USE_MD
static int MD_FU_INDEX = -100;
static int MD_FD_INDEX = -200;
static int MD_L_INDEX = -300;
static int MD_R_INDEX = -400;

bool isMd(int d) {
    return (d == MD_FU_INDEX || d == MD_FD_INDEX || d == MD_L_INDEX || d == MD_R_INDEX);
}
#endif


int main()
{
#if SAVE_IMG_IF_NEEDED
    {
        const int count = sizeof(files)/sizeof(files[0]);
        const int svcount = sizeof(savefiles)/sizeof(savefiles[0]);
        if(count != svcount) {
            cerr << "error. savefiles count != files count" << endl;
            return -1;
        }
    }
#endif
    
    try
    {
        while(1) {
            string fileName;
#if USE_WEBCAM
            cv::VideoCapture cap(0);
#else
            const int count = sizeof(files)/sizeof(files[0]);
            cv::VideoCapture cap;
            if(count > 0) {
                fileName = files[fileIndex++];
                cout << "\nopen file : " << fileName << endl;
                cap.open(fileName);
            }
#endif
            if (!cap.isOpened())
            {
                cerr << "Unable to connect to camera" << endl;
                return 1;
            }
            
            
#if USE_WEBCAM
            cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
            cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
#endif
//            cout << "VideoCapture : " << cap.get(CV_CAP_PROP_FRAME_WIDTH) << "x" << cap.get(CV_CAP_PROP_FRAME_HEIGHT) << "@" << cap.get(CV_CAP_PROP_FPS) << endl;
            
            image_window win;
            
            // Load face detection and pose estimation models.
#if !USE_CUSTOM_FD
            frontal_face_detector detector = get_frontal_face_detector();
#else
            
            typedef object_detector<scan_fhog_pyramid<pyramid_down<6>>> my_obj_etector;
            
#if 0
            //
            // madeOriginal
            //
//            my_obj_etector detector1; deserialize("../../../test_data/svm/made_original/custom_front.svm") >> detector1;
            
            // lr madeorigin
//            my_obj_etector detector2; deserialize("../../../test_data/svm/made_original/custom_left.svm") >> detector2;
//            my_obj_etector detector3; deserialize("../../../test_data/svm/made_original/custom_right.svm") >> detector3;
            
            // lr added seil
//            my_obj_etector detector2; deserialize("../../../test_data/svm/added_seil/left.svm") >> detector2;
//            my_obj_etector detector3; deserialize("../../../test_data/svm/added_seil/right.svm") >> detector3;
            
            // lr added1
//            my_obj_etector detector2; deserialize("../../../test_data/svm/added_1/left.svm") >> detector2;
//            my_obj_etector detector3; deserialize("../../../test_data/svm/added_1/right.svm") >> detector3;
            
            // lr added2
//            my_obj_etector detector2; deserialize("../../../test_data/svm/added_2/left.svm") >> detector2;
//            my_obj_etector detector3; deserialize("../../../test_data/svm/added_2/right.svm") >> detector3;
            
            // lr added3
//            my_obj_etector detector2; deserialize("../../../test_data/svm/added_3/left.svm") >> detector2;
//            my_obj_etector detector3; deserialize("../../../test_data/svm/added_3/right.svm") >> detector3;
            
            // lr added4
            my_obj_etector detector2; deserialize("../../../test_data/svm/added_4/left.svm") >> detector2;
            my_obj_etector detector3; deserialize("../../../test_data/svm/added_4/right.svm") >> detector3;
            
            my_obj_etector detector4; deserialize("../../../test_data/svm/made_original/custom_front_rotate_left.svm") >> detector4;
            my_obj_etector detector5; deserialize("../../../test_data/svm/made_original/custom_front_rotate_right.svm") >> detector5;
            
            std::vector<my_obj_etector> detectors;
            detectors.push_back(detector1);
            detectors.push_back(detector2);
            detectors.push_back(detector3);
            detectors.push_back(detector4);
            detectors.push_back(detector5);
            
            my_obj_etector detector(detectors);
            
            
            // render HOG image
//            image_window hogwin_cu1(draw_fhog(detector1), "1 detector");
//            image_window hogwin_cu2(draw_fhog(detector2), "2 detector");
//            image_window hogwin_cu3(draw_fhog(detector3), "3 detector");
//            image_window hogwin_cu4(draw_fhog(detector4), "4 detector");
//            image_window hogwin_cu5(draw_fhog(detector5), "5 detector");
#endif
            
            
            
            
#if USE_MD
//            my_obj_etector detector1; deserialize("../../../test_data/svm/made_original/custom_front.svm") >> detector1;
//            my_obj_etector detector2; deserialize("../../../test_data/svm/made_original/custom_left.svm") >> detector2;
//            my_obj_etector detector3; deserialize("../../../test_data/svm/made_original/custom_right.svm") >> detector3;
//            my_obj_etector detector4; deserialize("../../../test_data/svm/made_original/custom_front_rotate_left.svm") >> detector4;
//            my_obj_etector detector5; deserialize("../../../test_data/svm/made_original/custom_front_rotate_right.svm") >> detector5;
            
            
            
            // Original + More Detector
            // original detector
            my_obj_etector origin = get_frontal_face_detector();
            
            test_box_overlap my_overlap_tester(0.20, 0.25);
            my_obj_etector front(origin.get_scanner(), my_overlap_tester, origin.get_w(0));
            my_obj_etector left(origin.get_scanner(), my_overlap_tester, origin.get_w(1));
            my_obj_etector right(origin.get_scanner(), my_overlap_tester, origin.get_w(2));
            my_obj_etector left_rotate_front(origin.get_scanner(), my_overlap_tester, origin.get_w(3));
            my_obj_etector right_rotate_front(origin.get_scanner(), my_overlap_tester, origin.get_w(4));
            
            
            // more detector
            // up front
//            std::string upfile    = "../../../test_data/svm/md_2/fu3.svm";
            std::string upfile    = "/Users/hongkeunyoo/Documents/Dlib FaceDetection/training/detector_fu3.svm";
//            std::string upfile    = "../../../test_data/svm/md_2/fu_c700.svm";
//            std::string upfile    = "../../../test_data/svm/md_2/fu_c350.svm";
//            std::string upfile    = "../../../test_data/svm/md_2/fu_c900.svm";
    
            
            // down front
//            std::string downfile  = "../../../test_data/svm/md_2/fd2.svm";
            std::string downfile    = "/Users/hongkeunyoo/Documents/Dlib FaceDetection/training_add/fd_new.svm";
//            std::string downfile  = "../../../test_data/svm/md_2/fd1_700.svm";
            
            // left
//            std::string leftfile  = "../../../test_data/svm/md_1/lr1/left.svm";
//            std::string leftfile  = "../../../test_data/svm/md_2/left_c100.svm"; // 오검출
//            std::string leftfile  = "../../../test_data/svm/md_2/left3.svm";
            std::string leftfile    = "/Users/hongkeunyoo/Documents/Dlib FaceDetection/training/detector_left3.svm";
//            std::string leftfile  = "../../../test_data/svm/md_2/left4.svm";
            
            // right
//            std::string rightfile  = "../../../test_data/svm/md_1/lr1/right.svm";
//            std::string rightfile  = "../../../test_data/svm/md_2/right_c100.svm"; // 오검출
//            std::string rightfile  = "../../../test_data/svm/md_2/right3.svm";
            std::string rightfile  = "/Users/hongkeunyoo/Documents/Dlib FaceDetection/training/detector_right3.svm";
//            std::string rightfile  = "../../../test_data/svm/md_2/right4.svm";
           
//            std::string newfile = "/Users/hongkeunyoo/Documents/Dlib FaceDetection/detector_f2.svm";
            
            cout << "keun22" << endl;
            
            my_obj_etector md_up; deserialize(upfile) >> md_up;
            my_obj_etector md_down; deserialize(downfile) >> md_down;
            my_obj_etector md_left; deserialize(leftfile) >> md_left;
            my_obj_etector md_right; deserialize(rightfile) >> md_right;
 //           my_obj_etector md_new; deserialize(newfile) >> md_new;
            
            
            // 5 original + 4 md
            //
            // 1. front
            // 2.  ** up front
            // 3.  ** down front
            // 4. left
            // 5.  ** ab left
            // 6. right
            // 7.  ** ab right
            // 8. left rotated front
            // 9. right rotate front
            //
            std::vector<my_obj_etector> detectors;
            
            detectors.push_back(front);
   //         detectors.push_back(md_new);
            detectors.push_back(md_up);
            detectors.push_back(md_down);
            detectors.push_back(left);
            detectors.push_back(md_left);
            detectors.push_back(right);
            detectors.push_back(md_right);
            detectors.push_back(left_rotate_front);
            detectors.push_back(right_rotate_front);
            
            my_obj_etector detector(detectors);
            
            // 디텍터 순서가 변경되면 같이 바꿔라
            MD_FU_INDEX = 1;
            MD_FD_INDEX = 2;
            MD_L_INDEX = 4;
            MD_R_INDEX = 6;
            
            // render HOG image
        
//            image_window hogwin_ori1(draw_fhog(front), "ori1 detector");
//            image_window hogwin_ori2(draw_fhog(left), "ori2 detector");
//            image_window hogwin_ori3(draw_fhog(right), "ori3 detector");
//            image_window hogwin_ori4(draw_fhog(left_rotate_front), "ori4 detector");
//            image_window hogwin_ori5(draw_fhog(right_rotate_front), "ori5 detector");
//            image_window hogwin_up(draw_fhog(md_up), "md1");
//            image_window hogwin_d(draw_fhog(md_down), "md2");
//            image_window hogwin_l(draw_fhog(md_left), "md1");
//            image_window hogwin_r(draw_fhog(md_right), "md2");
#endif
#endif
            
#if DETECT_LANDMARK
            shape_predictor pose_model;
            deserialize("/Users/hongkeunyoo/Documents/Dlib FaceDetection/training/sp_default_15points_20180118.dat") >> pose_model;
#endif
            
            // for stats
            int videoframeCount = 0;
            int detectedCount = 0;
            double accumulatedTime = 0;
#if USE_MD
            int mdCount = 0;
#endif
            
            // Grab and process frames until the main window is closed by the user.
            while(!win.is_closed())
            {
                // Grab a frame
                cv::Mat temp;
                if (!cap.read(temp))
                {
                    break;
                }
                
                // increase video frame count
                ++videoframeCount;
                
                // Turn OpenCV's Mat into something dlib can deal with.  Note that this just
                // wraps the Mat object, it doesn't copy anything.  So cimg is only valid as
                // long as temp is valid.  Also don't do anything to temp that would cause it
                // to reallocate the memory which stores the image as that will make cimg
                // contain dangling pointers.  This basically means you shouldn't modify temp
                // while using cimg.
                
#if USE_WEBCAM
                cv::Mat flipped;
                flip(temp, flipped, 1);
                cv_image<bgr_pixel> cimg(flipped);
#else
                cv_image<bgr_pixel> cimg(temp);
#endif
                
                int64 frameStartTick = cv::getTickCount();
                
                // Detect faces
                int usedDetector = -1000;
//                std::vector<rectangle> faces = detector(cimg, 0, usedDetector);
                std::vector<rectangle> faces = detector(cimg);
                const bool faceDetected = (faces.size() > 0);
                
                // calculate stats
                int64 frameEndTick = cv::getTickCount();
                double frameTime = ((double)frameEndTick - (double)frameStartTick)/ cv::getTickFrequency();
                accumulatedTime += frameTime;
                if(faceDetected) {
                    ++detectedCount;
#if USE_MD
                    if( isMd(usedDetector) ) {
                        mdCount++;
                    }
#endif
                }
                
                // Print log every frames
                // cout << "[" << (videoframeCount) << "] face found (" << faces.size() << "). elapsed time = " << frameTime << endl;

#if OVERLAY_RENDER_FACE_RECT
                std::vector<image_window::overlay_line> rectlines;
#endif
                
                // Find the pose of each face.
                std::vector<full_object_detection> shapes;
                for (unsigned long i = 0; i < faces.size(); ++i) {
#if DETECT_LANDMARK
                    shapes.push_back(pose_model(cimg, faces[i]));
#endif
#if OVERLAY_RENDER_FACE_RECT
                    rectangle rect = faces[i];
                    rectlines.push_back(image_window::overlay_line(rect.tl_corner(), rect.tr_corner(), dlib::rgb_pixel(0, 255, 0)));
                    rectlines.push_back(image_window::overlay_line(rect.br_corner(), rect.bl_corner(), dlib::rgb_pixel(0, 255, 0)));
                    rectlines.push_back(image_window::overlay_line(rect.tr_corner(), rect.br_corner(), dlib::rgb_pixel(0, 255, 0)));
                    rectlines.push_back(image_window::overlay_line(rect.tl_corner(), rect.bl_corner(), dlib::rgb_pixel(0, 255, 0)));
#endif
                }
                
                // Display it all on the screen
                win.clear_overlay();
                win.set_image(cimg);
                win.add_overlay(render_face_detections(shapes));
                
#if OVERLAY_RENDER_FACE_RECT
                win.add_overlay(rectlines);
#endif
                
                
                // finished one frame processing
#if !USE_WEBCAM
#if NEXT_FRAME_WITH_KEY
                cout << "Enter Key press to next frame : " << videoframeCount << endl;
                cin.get();
#elif NEXT_FRAME_WITH_KEY_ONLY_IF_NOT_DETECTED
                if (!faceDetected) {
                    cout << "checking : Not Detected Frame" << endl;
                    cin.get();
                }
#endif

                // wrong detected save img
                if(faces.size() > 1) {
#if SAVE_IMG_IF_NEEDED
                    string imgName = std::to_string(videoframeCount) + "_wrong.png";
                    string imgpath = savefiles[(fileIndex-1)] + imgName;
                    bool wSave = cv::imwrite(imgpath, temp);
                    cout << ">> Wrong Detected Frame? save Image " << imgpath << " : " << wSave << endl;
#else
                    cout << ">> Wrong Detected Frame? need to check " << videoframeCount << endl;
//                    cin.get();
#endif
                }
                
#if SAVE_IMG_IF_NEEDED
#if SAVE_IMG_IF_NOT_DETECTED
                if (!faceDetected) {
                    string imgName = std::to_string(videoframeCount) + ".png";
                    string imgpath = savefiles[(fileIndex-1)] + imgName;
                    bool wSave = cv::imwrite(imgpath, temp);
                    cout << "Not Detected Frame. save Image " << imgpath << " : " << wSave << endl;
                }
#endif // end SAVE_IMG_IF_NOT_DETECTED
#endif // end SAVE_IMG_IF_NEEDED
#endif // end !USE_WEBCAM
            }

#if PRINT_RESULT_STATS
            cout << "  Result : " << detectedCount << " / " << videoframeCount << " = " << ((float)detectedCount/(float)videoframeCount)
                 << " (" << ((double)accumulatedTime/(double)videoframeCount)*1000 << " ms) " << endl;
#if USE_MD
//            cout << "   MD : " << mdCount << " / " << detectedCount << " (" << ((float)mdCount/(float)deWrongtectedCount) << ")" << endl;
#endif
#endif
            if(!moreData()) {
                break;
            }
        }
    }
    catch(serialization_error& e)
    {
        cout << endl << e.what() << endl;
    }
    catch(exception& e)
    {
        cout << e.what() << endl;
    }
}

