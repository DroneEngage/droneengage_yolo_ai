/***************************************************************************************
 * 
 * This module is heavily inspired by [hailo-rpi5-yolov8] example.
 * Original Author: https://github.com/bmharper
 * Origin of Code: https://github.com/bmharper/hailo-rpi5-yolov8
 * 
 ***************************************************************************************/


#ifndef YOLO_AI_H
#define YOLO_AI_H

#include <set> // Using std::set for efficient lookup of allowed class indices

#include "../helpers/json_nlohmann.hpp"

using Json_de = nlohmann::json;


namespace de
{
namespace yolo_ai
{

class CCallBack_YOLOAI
{
    public:
        virtual void onTrack (const Json_de targets) = 0;
        virtual void onBestObject (const Json_de targets) = 0;
        virtual void onTrackStatusChanged (const int& status) = 0;
};


 class CYOLOAI
    {

        public:
            
            static CYOLOAI& getInstance()
            {
                static CYOLOAI instance;

                return instance;
            }

            CYOLOAI(CYOLOAI const&)           = delete;
            void operator=(CYOLOAI const&)        = delete;

        
        private:

            CYOLOAI()
            {
            }

        public:
            
            ~CYOLOAI()
            {
                if (m_exit_thread == false)
                {
                    uninit();
                }
            };
                

        public:
            
            bool init(const std::string& cam_path, const std::string& hef_path, const std::string& virtual_device_path, std::vector<std::string> & classNames, CCallBack_YOLOAI *callback_yolo_ai) ;
            bool uninit() ;
        
            int  run();
            
            void detect();
            void pause();
            void stop();


            void loadAllowedClassIndices(const Json_de& json_array);


        private:
            
            bool m_exit_thread;
            bool m_is_AI_yolo_active_initial = false;
            bool m_object_found = false;

            std::string m_source_video_device;
            std::string m_output_video_device;
            std::string m_hef_model_path;
            std::vector<std::string> m_class_names;

            std::set<size_t> m_allowed_class_indices; // To store the simplified list

            CCallBack_YOLOAI* m_callback_yolo_ai = nullptr;

    };

}
}



#endif

