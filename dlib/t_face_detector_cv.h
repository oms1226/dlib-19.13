// Copyright (C) 2027  Jonghwa.Jo (jonghwa.jo@sk.com)
// License:


#ifndef DLIB_T_FACE_DETECTION_
#define DLIB_T_FACE_DETECTION_


//#include <stdio.h>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <ctime>

//#include <glog/logging.h>

#define MULTI_DETECTING_MODE               true
#define TRACKING_ALLOW_TIME                100
#define EARLY_TERM_ALLOW_TIME              1000
#define ADJUST_THRESHOLD                   0

//#define WHEN_NO_FACE_TRY_UNTIL             1000
//#define WHEN_NO_FACE_TRY_EVERY             1000

#if defined(__APPLE__)
#define MAX_DETECTING_SCORE                    5
#define CV_TEMPLATE_MATCHING_METHOD            CV_TM_CCOEFF_NORMED
#define CV_TEMPLATE_MATCHING_LIMIT_MAX_VALUE   0.98
#define CV_TEMPLATE_MATCHING_LIMIT_LOW_VALUE   0.1
#define CV_TEMPLATE_SIZE_FACTOR                0.8f
#else
#define MAX_DETECTING_SCORE                    10
#define CV_TEMPLATE_MATCHING_METHOD            CV_TM_CCOEFF_NORMED
#define CV_TEMPLATE_MATCHING_LIMIT_MAX_VALUE   0.5
#define CV_TEMPLATE_MATCHING_LIMIT_LOW_VALUE   0.1
#define CV_TEMPLATE_SIZE_FACTOR                0.5f
#endif //#if defined(WEBRTC_IOS)


namespace dlib
{
  using namespace std;
  using namespace std::chrono;
  
  struct t_face_result {
  public:
    int id;
    rectangle rect;
  };
  
  template <unsigned int PYRAMID_DOWN_SCALE_FACTOR>
  class t_face_detector {
  public:
    t_face_detector()
    : tracker_id(0)
    , prefer_detector(0)
    , prefer_level(0)
    {
    }

    template <typename object_detector>
    void setDetector(object_detector det) {
      detector = det;
    }
    
    template <typename image_type>
    std::vector<t_face_result> detect(const cv::Mat &frame, const image_type& img) {
      std::vector<t_face_result> result;
      
//      if(lastFaceDetectedTime.time_since_epoch().count() && duration_cast<milliseconds>(steady_clock::now() - lastFaceDetectedTime).count() > WHEN_NO_FACE_TRY_UNTIL) {
//        if(nextFullDetectTime.time_since_epoch().count() && duration_cast<milliseconds>(steady_clock::now() - nextFullDetectTime).count() > WHEN_NO_FACE_TRY_EVERY) {
//          nextFullDetectTime = {};   //WHEN_NO_FACE_TRY_EVERY 마다 FD를 진행하기 위한 초기화
//
//        } else {
//          if(nextFullDetectTime.time_since_epoch().count() == 0)
//            nextFullDetectTime = steady_clock::now();
//
//          return result;
//        }
//      }
      
      FullDetect(frame, img);
      
      auto it = trackers.begin();
      while (it != trackers.end()) {
        if((*it)->detecting_score <= 0) {
          it = trackers.erase(it);
        } else {
          t_face_result tResult;
          tResult.id = (*it)->tracker_id;
          tResult.rect = (*it)->faceRoi;
          result.push_back(tResult);
          ++it;
        }
      }
      
//      if(result.size() > 0) {
//        lastFaceDetectedTime = steady_clock::now();
//      }
      
      return result;
    }
    
  private:
    struct Tracker {
    public:
      int detecting_score;
      int tracker_id;
      bool detected;
      
      cv::Mat faceTemplate;
      rectangle faceRoi;
      
      Tracker(int id)
      : detecting_score(2)
      , tracker_id (id)
      , detected(true) {
      }
      
      void setFaceTemplate(const cv::Mat &frame, rectangle rect) {
        faceRoi = rect;
        
        //Resize Face Template
//        cv::Rect faceRect = convertRect(rect.left() + rect.width() * 0.25f,
//                                        rect.top() + rect.height() * 0.25f,
//                                        rect.width() * 0.5f,
//                                        rect.height() * 0.5f,
//                                        frame.cols,
//                                        frame.rows);
        
        cv::Rect faceRect = convertRect(rect.left() + rect.width() * (1 - CV_TEMPLATE_SIZE_FACTOR) * 0.5f,
                                        rect.top() + rect.height() * (1 - CV_TEMPLATE_SIZE_FACTOR) * 0.5f,
                                        rect.width() * CV_TEMPLATE_SIZE_FACTOR,
                                        rect.height() * CV_TEMPLATE_SIZE_FACTOR,
                                        frame.cols,
                                        frame.rows);

        
        faceTemplate = frame(faceRect).clone();
      }
      
      bool matchTemplate(const cv::Mat &frame) {
        if(faceRoi.width() <= 2 ||
           faceRoi.height() <= 2 ||
           faceRoi.left() <= 0 ||
           faceRoi.top() <= 0 ||
           faceRoi.right() >= frame.cols ||
           faceRoi.bottom() >= frame.rows)
          return false;
        
        cv::Rect cvRect = convertRect(faceRoi, frame.cols, frame.rows);
        if(cvRect.width <= (faceTemplate.cols + 2) || cvRect.height <= (faceTemplate.rows + 2) )
          return false;
        
        static cv::Mat matchingResult;
        cv::matchTemplate(frame(cvRect), faceTemplate, matchingResult, CV_TEMPLATE_MATCHING_METHOD);
        
        double min, max;
        cv::Point minLoc, maxLoc, matchLoc;
        cv::minMaxLoc(matchingResult, &min, &max, &minLoc, &maxLoc);
        
        /// For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better
        if( CV_TEMPLATE_MATCHING_METHOD == CV_TM_SQDIFF || CV_TEMPLATE_MATCHING_METHOD == CV_TM_SQDIFF_NORMED ) {
          if( min > CV_TEMPLATE_MATCHING_LIMIT_LOW_VALUE ) {
            return false;
          }
          matchLoc = minLoc;
        } else {
          if( max < CV_TEMPLATE_MATCHING_LIMIT_MAX_VALUE ) {
            return false;
          }
          matchLoc = maxLoc;
        }
        
        // Add roi offset to face position
        matchLoc.x += cvRect.x;
        matchLoc.y += cvRect.y;
        
//        faceRoi.set_left(matchLoc.x - faceTemplate.cols * 0.5f);
//        faceRoi.set_top(matchLoc.y - faceTemplate.rows * 0.5f);
//        faceRoi.set_right(matchLoc.x + faceTemplate.cols * 1.5f);
//        faceRoi.set_bottom(matchLoc.y + faceTemplate.rows * 1.5f);
        
        float marginX = (faceRoi.width() - faceTemplate.cols) / 2;
        float marginY = (faceRoi.height() - faceTemplate.rows) / 2;
        
        faceRoi = rectangle(matchLoc.x - marginX,
                            matchLoc.y - marginY,
                            matchLoc.x + faceTemplate.cols + marginX,
                            matchLoc.y + faceTemplate.rows + marginY);
        
        return true;
      }
    };
    
    static cv::Rect convertRect(int x, int y, int width, int height, int screenWidth, int screenHeight) {
      //LOG(INFO) << "check be: " << x << ", " << y << ", " << width << ", " << height << ", " << screenWidth << ", " << screenHeight;
      
      int xx = x;
      if(x > screenWidth) {
        xx = screenWidth;
      } else if(x < 0) {
        xx = 0;
        width += x;
      }
      
      int yy = y;
      if(y > screenHeight) {
        yy = screenHeight;
      } else if(y < 0) {
        yy = 0;
        height += y;
      }
      
      int ww = std::max(0, (xx + width > screenWidth) ? (screenWidth - xx) : width);
      int hh = std::max(0, (yy + height > screenHeight) ? (screenHeight - yy) : height);
      
      //LOG(INFO) << "check af: " << xx << ", " << yy << ", " << ww << ", " << hh;
      
      return cv::Rect(xx, yy, ww, hh);
    }
    
    static cv::Rect convertRect(rectangle& rect, int screenWidth, int screenHeight) {
      int x = rect.left();
      int y = rect.top();
      int width = rect.right() - rect.left();
      int height = rect.bottom() - rect.top();
      
      return convertRect(x, y, width, height, screenWidth, screenHeight);
    }
    
    static double distance(double x, double y, double xx, double yy) {
      double distX = std::abs(x - xx);
      double distY = std::abs(y - yy);
      return std::sqrt(distX*distX + distY*distY);
    }
    
    template <typename image_type>
    void FullDetect(const cv::Mat &frame, const image_type& img) {
      std::vector<rectangle> faceRects;
      
      if( trackers.size() == 0 || (MULTI_DETECTING_MODE && duration_cast<milliseconds>(steady_clock::now() - earlyTermAllowTime).count() > 0) ) {
        earlyTermAllowTime = steady_clock::now() + milliseconds(EARLY_TERM_ALLOW_TIME);
        if(MULTI_DETECTING_MODE)
          faceRects = detector(img, ADJUST_THRESHOLD);
        else
          faceRects = detector(img, 1, 0, 0, &prefer_detector, &prefer_level, ADJUST_THRESHOLD);
      } else {
        int pDet = prefer_detector;
        int pLevel = prefer_level;
        faceRects = detector(img, trackers.size(), pDet, pLevel, &prefer_detector, &prefer_level, ADJUST_THRESHOLD);
      }
      
      if(faceRects.size() == 0) {
        FullTracking(frame);
        
      } else {
        for(int i=0; i<trackers.size(); i++)
          trackers[i]->detected = false;
          
        for(int i=0; i<faceRects.size(); i++) {
          Tracker* tracker = FindTrackerByDistance(faceRects[i]);
          if(tracker) {
            tracker->setFaceTemplate(frame, faceRects[i]);
            tracker->detecting_score = std::min(tracker->detecting_score + 1, MAX_DETECTING_SCORE);
            tracker->detected = true;
            
          } else {
            //std::cout << "new face detected" << endl;
            std::unique_ptr<Tracker> tracker(new Tracker(tracker_id++));
            if(tracker_id > 1000)
              tracker_id = 0;
            tracker->setFaceTemplate(frame, faceRects[i]);
            trackers.push_back( std::move(tracker) );
          }
        }
      }
      
      for(int i=0; i<trackers.size(); i++) {
        if(!trackers[i]->detected)
          Tracking(frame, trackers[i].get());
      }
    }
    
    Tracker* FindTrackerByDistance(rectangle rect) {
      for(int i=0; i<trackers.size(); i++) {
        if(trackers[i]->faceRoi.width() == 0)
          continue;
        
        if(distance(trackers[i]->faceRoi.left(), trackers[i]->faceRoi.top(), rect.left(), rect.top()) < (trackers[i]->faceRoi.width() * 0.8f) ) {
          return trackers[i].get();
        }
      }
      return nullptr;
    }
    
    void FullTracking(const cv::Mat &frame) {
      for(int i=0; i<trackers.size(); i++)
        Tracking(frame, trackers[i].get());
    }
    
    void Tracking(const cv::Mat &frame, Tracker* tracker) {
      if(--tracker->detecting_score > 0 && !tracker->matchTemplate(frame))
        tracker->detecting_score = 0;
    }
    
  private:
    int tracker_id;
    int prefer_detector;
    int prefer_level;
    dlib::object_detector< dlib::scan_fhog_pyramid<dlib::pyramid_down<PYRAMID_DOWN_SCALE_FACTOR>> > detector;
    std::vector< std::unique_ptr<Tracker> > trackers;
//    steady_clock::time_point lastFaceDetectedTime;
//    steady_clock::time_point nextFullDetectTime;
    steady_clock::time_point earlyTermAllowTime;
  };
}

#endif // DLIB_T_FACE_DETECTION_
