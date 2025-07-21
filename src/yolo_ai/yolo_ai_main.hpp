#ifndef YOLO_AI_MAIN_H
#define YOLO_AI_MAIN_H


#include <thread>         // std::thread

#include "../de_common/de_common_callback.hpp"
#include "yolo_ai_facade.hpp"
#include "yolo_ai.hpp"


#include "../helpers/json_nlohmann.hpp"
using Json_de = nlohmann::json;


namespace de
{
namespace yolo_ai
{
 class CYOLOAI_Main: public de::comm::CCommon_Callback, de::yolo_ai::CCallBack_YOLOAI
    {

        public:
            
            static CYOLOAI_Main& getInstance()
            {
                static CYOLOAI_Main instance;

                return instance;
            }

            CYOLOAI_Main(CYOLOAI_Main const&)           = delete;
            void operator=(CYOLOAI_Main const&)        = delete;

        
        private:

            CYOLOAI_Main()
            {
            }

        public:
            
            ~CYOLOAI_Main()
            {
                if (m_exit_thread == false)
                {
                    uninit();
                }
            };
                

        public:
            
            bool init() ;
            bool uninit() ;
        

        public:
            
            void startYolo();
            void stopYolo();

        

        public:
            void startTrackingObjects(const Json_de& allowed_class_indices);
            void stopTracking();
            void pauseTracking();

        public:
            //CCallBack_Tracker
            void onTrack (const float& x, const float& y, const float& width, const float& height, const uint16_t camera_orientation, const bool camera_forward) override ;
            void onTrackStatusChanged (const int& track) override ;

        
        public:

            inline const std::vector<std::string> getTrackingObjectsList()
            {
                return m_class_names;
            }
        
        private:
            
            
            bool m_exit_thread;

            std::thread m_threadSenderID;

            std::string m_hef_model_path;
            std::string m_source_video_device;
            std::string m_output_virtual_video_path;
            std::vector<std::string> m_class_names;

            de::yolo_ai::CYOLOAI& m_yolo_ai = de::yolo_ai::CYOLOAI::getInstance();
            de::yolo_ai::CYOLOAI_Facade& m_trackerFacade = de::yolo_ai::CYOLOAI_Facade::getInstance();
    };

}
}

#endif