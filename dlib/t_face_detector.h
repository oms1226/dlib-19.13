// Copyright (C) 2017  Jonghwa.Jo (jonghwa.jo@sk.com)
// License:


#ifndef DLIB_T_FACE_DETECTION_
#define DLIB_T_FACE_DETECTION_


//#include <stdio.h>
//#include <opencv2/imgproc.hpp>
//#include "image_processing/frontal_face_detector.h"

#include <algorithm>
#include <chrono>
#include <ctime>

//#include <glog/logging.h>

#define FULL_SEARCH_INTERVAL               300
#define TRACKING_ALLOW_TIME                100
#define MIN_PSR                            8
#define DETECT_SCORE_EACH                  4

namespace dlib
{
  using namespace std;
  using namespace std::chrono;
  
  struct t_face_result {
  public:
    int id;
    rectangle rect;
  };
  
  class t_face_detector {
  public:
    t_face_detector()
    : tracker_id(0) {
      //detector = get_frontal_face_detector();
    }

    template <typename object_detector>
    void setDetector(object_detector det) {
      detector = det;
    }
    
    template <typename image_type>
    std::vector<t_face_result> detect(const image_type& img) {
      if(duration_cast<milliseconds>(steady_clock::now() - nextFullDetTime).count() > 0) {
        nextFullDetTime = steady_clock::now() + milliseconds(FULL_SEARCH_INTERVAL);
        FullDetect(img);
      } else {
        FullTracking(img);
      }
      
//        auto begin = std::chrono::steady_clock::now();
//        std::cout << "update elapsed: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin).count()

      std::vector<t_face_result> result;
      for(int i=0; i<trackers.size(); i++) {
        t_face_result tResult;
        tResult.id = trackers[i]->tracker_id;
        tResult.rect = trackers[i]->c_tracker->get_position();
        result.push_back(tResult);
      }
      
      return result;
    }
    
  private:
    struct Tracker : public t_face_result {
    public:
      int detecting_score;
      int tracker_id;
      std::unique_ptr<correlation_tracker> c_tracker;
      //std::chrono::steady_clock::time_point trackingAllowTime;
      
      Tracker(int id)
      : detecting_score(DETECT_SCORE_EACH)
      , tracker_id (id) {
        c_tracker = std::unique_ptr<correlation_tracker>(new correlation_tracker());
      }
      
//      int GetId() {
//        return tracker_id;
//      }
//
//      rectangle GetRect() {
//        return c_tracker->get_position();
//      }
      
    //private:
      
    };
    
    template <typename image_type>
    void FullDetect(const image_type& img) {
      std::vector<rectangle> faceRects = detector(img);
      
      if(faceRects.size() == 0) {
        //std::cout << "detector returns 0" << endl;
        FullTracking(img);
        
      } else {
        for(int i=0; i<faceRects.size(); i++) {
          Tracker* tracker = GetTracker(faceRects[i]);
          if(tracker) {
            //std::cout << "found tracker by contains" << endl;
            tracker->c_tracker->start_track(img, faceRects[i]);
            tracker->detecting_score = std::min(tracker->detecting_score + DETECT_SCORE_EACH, 100);
            //tracker->trackingAllowTime = steady_clock::now() + milliseconds(TRACKING_ALLOW_TIME * ++tracker->detecting_score);
            
          } else {
            //std::cout << "new face detected" << endl;
            std::unique_ptr<Tracker> tracker(new Tracker(tracker_id++));
            if(tracker_id > 1000)
              tracker_id = 0;
            tracker->c_tracker->start_track(img, faceRects[i]);
            //tracker->trackingAllowTime = steady_clock::now() + milliseconds(TRACKING_ALLOW_TIME * ++tracker->detecting_score);
            trackers.push_back( std::move(tracker) );
          }
        }
      }
      
      for(int i=0; i<trackers.size(); i++)
        trackers[i]->detecting_score--;
    }
    
    Tracker* GetTracker(rectangle rect) {
      for(int i=0; i<trackers.size(); i++) {
        double distX = std::abs(trackers[i]->c_tracker->get_position().left() - rect.left());
        double distY = std::abs(trackers[i]->c_tracker->get_position().top() - rect.top());
        double dist = std::sqrt(distX*distX + distY*distY);
        //cout << "dist: " << dist << endl;
        
        if(dist < trackers[i]->c_tracker->get_position().width()) {
          return trackers[i].get();
        }
        
//        if( trackers[i]->c_tracker->get_position().contains(rect.tl_corner()) ||
//            trackers[i]->c_tracker->get_position().contains(rect.bl_corner()) ||
//            trackers[i]->c_tracker->get_position().contains(rect.tr_corner()) ||
//            trackers[i]->c_tracker->get_position().contains(rect.br_corner()) ) {
//          return trackers[i].get();
//        }
      }
      return nullptr;
    }
    
    template <typename image_type>
    void FullTracking(const image_type& img) {
      auto it = trackers.begin();
      while (it != trackers.end()) {
        //cout << "tracker_id: " << (*it)->tracker_id << ", score: " << (*it)->detecting_score << endl;
        
        double psr = (*it)->c_tracker->update(img);
        //LOG(INFO) << "psr: " << psr;
        
        if(MIN_PSR > psr || (*it)->detecting_score <= 0)
           //duration_cast<milliseconds>(steady_clock::now() - (*it)->trackingAllowTime).count() > 0 )
          it = trackers.erase(it);
        else
          ++it;
      }
    }
    
  private:
    int tracker_id;
    object_detector<scan_fhog_pyramid<pyramid_down<DLIB_FD_PYRAMID_DOWN_SCALE_FACTOR> > > detector;
    std::vector< std::unique_ptr<Tracker> > trackers;
    std::chrono::steady_clock::time_point nextFullDetTime;
  };
}

#endif // DLIB_T_FACE_DETECTION_
