/***************************************************************************************
 * 
 * This module is heavily inspired by [hailo-rpi5-yolov8] example.
 * Original Author: https://github.com/bmharper
 * Origin of Code: https://github.com/bmharper/hailo-rpi5-yolov8
 * 
 ***************************************************************************************/

#ifndef YOLO_AI_H
#define YOLO_AI_H


namespace de
{
namespace yolo_ai
{
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
            
            bool init(const std::string& cam_path, const std::string& hef_path, const std::string& virtual_device_path, std::vector<std::string> & classNames) ;
            bool uninit() ;
        
            int  run();
        
        
        private:
            
            bool m_exit_thread;

            std::string m_source_video_device;
            std::string m_output_video_device;
            std::string m_hef_model_path;
            std::vector<std::string> m_class_names;

    };

}
}



#endif

