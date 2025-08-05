#include <stdio.h>
#include <iostream>

#include "../helpers/colors.hpp"
#include "../helpers/helpers.hpp"

#include "../de_common/configFile.hpp"
#include "../de_common/messages.hpp"

#include "yolo_ai_main.hpp"




using namespace de::yolo_ai;


bool CYOLOAI_Main::init()
{
    de::CConfigFile& cConfigFile = de::CConfigFile::getInstance();
    const Json_de& jsonConfig = cConfigFile.GetConfigJSON();
    if (!validateField(jsonConfig, "source_video_device", Json_de::value_t::string))
    {
        std::cout << _ERROR_CONSOLE_BOLD_TEXT_ << "Fatal Error:" << _NORMAL_CONSOLE_TEXT_ << " Missing field or bad string format " << _INFO_CONSOLE_BOLD_TEXT << " source_video_device " << _NORMAL_CONSOLE_TEXT_<< std::endl;
        exit(1);
    }

    std::string source_video_device = jsonConfig["source_video_device"].get<std::string>();

    std::cout << _SUCCESS_CONSOLE_BOLD_TEXT_ << "Video Capture:" << _INFO_CONSOLE_TEXT << source_video_device << _NORMAL_CONSOLE_TEXT_ << std::endl;
    
    if (!validateField(jsonConfig, "output_video_device", Json_de::value_t::string))
    {
        std::cout << _ERROR_CONSOLE_BOLD_TEXT_ << "Fatal Error:" << _NORMAL_CONSOLE_TEXT_ << " Missing field or bad string format " << _INFO_CONSOLE_BOLD_TEXT << " output_video_device " << _NORMAL_CONSOLE_TEXT_<< std::endl;
        exit(1);
    }
    
    std::string output_video_device = jsonConfig["output_video_device"].get<std::string>();

    std::cout << _SUCCESS_CONSOLE_BOLD_TEXT_ << "Video Output: " << _INFO_CONSOLE_TEXT << output_video_device << _NORMAL_CONSOLE_TEXT_ << std::endl;
    
    if (!validateField(jsonConfig, "model_path", nlohmann::json::value_t::string))
    {
        std::cout << _ERROR_CONSOLE_BOLD_TEXT_ << "Fatal Error: " << _NORMAL_CONSOLE_TEXT_ << " Missing field or bad string format " << _INFO_CONSOLE_BOLD_TEXT << " model_path " << _NORMAL_CONSOLE_TEXT_<< std::endl;
        exit(1);
    }
    
    std::string model_path = jsonConfig["model_path"].get<std::string>();

    std::cout << _SUCCESS_CONSOLE_BOLD_TEXT_ << "model_path: " << _INFO_CONSOLE_TEXT << model_path << _NORMAL_CONSOLE_TEXT_ << std::endl;
    
    if (!validateField(jsonConfig, "class_names", nlohmann::json::value_t::array))
    {
        std::cout << _ERROR_CONSOLE_BOLD_TEXT_ << "Fatal Error: " << _NORMAL_CONSOLE_TEXT_ << "Missing field or bad array format " << _INFO_CONSOLE_BOLD_TEXT << " classNames " << _NORMAL_CONSOLE_TEXT_<< std::endl;
        exit(1);
    }
    
    m_class_names.clear();

    for (const auto& element : jsonConfig["class_names"]) {
            if (element.is_string()) {
                m_class_names.push_back(element.get<std::string>());
            } else {
                std::cout << "Warning: Non-string element found in classNames array. Skipping." << std::endl;
            }
        }
    
    std::cout << _SUCCESS_CONSOLE_BOLD_TEXT_ << "class_names: " << _INFO_CONSOLE_TEXT << "filled." << _NORMAL_CONSOLE_TEXT_ << std::endl;
    
    m_yolo_ai.init(source_video_device, model_path, output_video_device, m_class_names, this);


    m_threadSenderID = std::thread {[&](){ m_yolo_ai.run();}};

    return true;
}


bool CYOLOAI_Main::uninit()
{
    m_yolo_ai.stop();
    if(m_threadSenderID.joinable())
    {
        m_threadSenderID.join();
    }
    return true;
}



void CYOLOAI_Main::startYolo()
{

}


/**
 * Called when there is a a tracked object.
 * output from -0.5 to 0.5
 * (0,0) top left
 * center = [(x + w )/2 , (y + h)/2]
 */
void CYOLOAI_Main::onTrack (const Json_de targets) 
{

    m_trackerFacade.sendTrackingTargetsLocation (
        std::string(""),
        targets
    );
}

/**
 * Called once trackig status changed.
 */
void CYOLOAI_Main::onTrackStatusChanged (const int& status)  
{
    m_trackerFacade.sendTrackingTargetStatus (
        std::string(""),
        status
    );
    

    #ifdef DDEBUG
    std::cout << _INFO_CONSOLE_BOLD_TEXT << "onTrackStatusChanged:" << _LOG_CONSOLE_BOLD_TEXT << std::to_string(status) << _NORMAL_CONSOLE_TEXT_ << std::endl;
    #endif
}
    
 void CYOLOAI_Main::startTrackingObjects(const Json_de& allowed_class_indices)
 {
    m_yolo_ai.loadAllowedClassIndices(allowed_class_indices);
    m_yolo_ai.detect();
 }

 void CYOLOAI_Main::stopTracking()
 {
    m_yolo_ai.stop();
 }

void CYOLOAI_Main::pauseTracking()
{
    m_yolo_ai.pause();

}
