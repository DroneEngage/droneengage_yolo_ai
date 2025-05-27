#ifndef YOLO_AI_MAIN_H
#define YOLO_AI_MAIN_H


#include "../de_common/de_common_callback.hpp"
#include "yolo_ai.hpp"


#include "../helpers/json_nlohmann.hpp"
using Json_de = nlohmann::json;


namespace de
{
namespace yolo_ai
{
 class CYOLOAI_Main: public de::comm::CCommon_Callback
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


        private:
            
            
            bool m_exit_thread;


            std::string m_hef_model_path;
            std::string m_source_video_device;
            std::string m_output_virtual_video_path;


            de::yolo_ai::CYOLOAI& m_yolo_ai = de::yolo_ai::CYOLOAI::getInstance();
    };

}
}

#endif