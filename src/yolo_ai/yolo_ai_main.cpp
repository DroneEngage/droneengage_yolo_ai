#include <stdio.h>
#include <iostream>

#include "../helpers/colors.hpp"
#include "../helpers/helpers.hpp"

#include "../de_common/configFile.hpp"
#include "../de_common/messages.hpp"

#include "video.hpp"

#include "yolo_ai_main.hpp"




using namespace de::yolo_ai;


bool CYOLOAI_Main::init()
{
    de::CConfigFile& cConfigFile = de::CConfigFile::getInstance();
    const Json_de& jsonConfig = cConfigFile.GetConfigJSON();

#ifndef UDP_AI_DETECTION

    std::string source_video_device = "";
    std::string output_video_device = "";

    if (jsonConfig.contains("source_video_device_name"))
    {
        const int video_index = CVideo::findVideoDeviceIndex(jsonConfig["source_video_device_name"]);
        if (video_index != -1) 
        {
            source_video_device = "/dev/video" + std::to_string(video_index);

            std::cout << _SUCCESS_CONSOLE_BOLD_TEXT_ << "Using source_video_device_name:" << _INFO_CONSOLE_BOLD_TEXT << source_video_device 
                    << _NORMAL_CONSOLE_TEXT_
                    << std::endl;
        }
    }

    if (source_video_device.empty())
    {
        if (!jsonConfig.contains("source_video_device"))
        {
            std::cout << _ERROR_CONSOLE_BOLD_TEXT_ << "FATAL ERROR: " << _INFO_CONSOLE_TEXT << CConfigFile::getInstance().getFileName() 
                    << " does not have field " << _ERROR_CONSOLE_TEXT_ "[source_video_device]" <<  _NORMAL_CONSOLE_TEXT_ 
                    << std::endl;
        
            exit(1);
        }
        else
        {
            source_video_device = jsonConfig["source_video_device"];

            std::cout << _SUCCESS_CONSOLE_BOLD_TEXT_ << "Using source_video_device:" << _INFO_CONSOLE_BOLD_TEXT << source_video_device 
                    << _NORMAL_CONSOLE_TEXT_
                    << std::endl;
        }
    }

    
    if (jsonConfig.contains("output_video_device_name"))
    {
        const int video_index = CVideo::findVideoDeviceIndex(jsonConfig["output_video_device_name"]);
        if (video_index != -1) 
        {
            output_video_device = "/dev/video" + std::to_string(video_index);

            std::cout << _SUCCESS_CONSOLE_BOLD_TEXT_ << "Using output_video_device_name:" << _INFO_CONSOLE_BOLD_TEXT << output_video_device 
                    << _NORMAL_CONSOLE_TEXT_
                    << std::endl;

        }
    }

    
    if (output_video_device.empty())
    {
        if (!jsonConfig.contains("output_video_device"))
        {
            std::cout << _ERROR_CONSOLE_BOLD_TEXT_ << "FATAL ERROR:" << _INFO_CONSOLE_TEXT << " No output_video_device specified in config.json" <<  _NORMAL_CONSOLE_TEXT_ << std::endl;
            exit(1);
        }
        else
        {
            output_video_device = jsonConfig["output_video_device"];

            std::cout << _SUCCESS_CONSOLE_BOLD_TEXT_ << "Using output_video_device:" << _INFO_CONSOLE_BOLD_TEXT << output_video_device 
                    <<   _NORMAL_CONSOLE_TEXT_
                    << std::endl;
        }
    }
    
    if (!validateField(jsonConfig, "model_path", nlohmann::json::value_t::string))
    {
        std::cout << _ERROR_CONSOLE_BOLD_TEXT_ << "Fatal Error: " << _NORMAL_CONSOLE_TEXT_ << " Missing field or bad string format " << _INFO_CONSOLE_BOLD_TEXT << " model_path " << _NORMAL_CONSOLE_TEXT_<< std::endl;
        exit(1);
    }
    
    std::string model_path = jsonConfig["model_path"].get<std::string>();

    std::cout << _SUCCESS_CONSOLE_BOLD_TEXT_ << "model_path: " << _INFO_CONSOLE_TEXT << model_path << _NORMAL_CONSOLE_TEXT_ << std::endl;
    
#endif

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
    
#ifdef UDP_AI_DETECTION
    de::yolo_ai::CUDP_AI_Receiver& m_udp_ai_receiver = de::yolo_ai::CUDP_AI_Receiver::getInstance();
    int port = jsonConfig.contains("external_ai_feed_port")
        && jsonConfig["external_ai_feed_port"].is_number_unsigned()?jsonConfig["external_ai_feed_port"].get<int>():12347;
        
    m_udp_ai_receiver.init(port, [this](ParsedDetection detection) {
            this->onReceive(detection);
        });
    

#else
    de::yolo_ai::CYOLOAI& m_yolo_ai = de::yolo_ai::CYOLOAI::getInstance();
            
    m_yolo_ai.init(source_video_device, model_path, output_video_device, m_class_names, this);
    m_threadSenderID = std::thread {[&](){ m_yolo_ai.run();}};
#endif

    return true;
}


bool CYOLOAI_Main::uninit()
{
#ifdef UDP_AI_DETECTION

#else
    de::yolo_ai::CYOLOAI& m_yolo_ai = de::yolo_ai::CYOLOAI::getInstance();
            
    m_yolo_ai.stop();
#endif
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

    #ifdef DEBUG
        std::cout << _INFO_CONSOLE_BOLD_TEXT << "onTrack >> " 
        << _LOG_CONSOLE_BOLD_TEXT << targets.dump() << _NORMAL_CONSOLE_TEXT_ << std::endl;
    

        // Too much traffic ... dont send this.
        m_trackerFacade.sendTrackingTargetsLocation (
            std::string(""),
            targets
        );
    
    #endif

    
}

void CYOLOAI_Main::onBestObject (const Json_de targets) 
{

    m_trackerFacade.sendTrackingBestTargetsLocation (
        std::string(""),
        targets
    );
}

/**
 * Called once trackig status changed.
 */
void CYOLOAI_Main::onTrackStatusChanged (const int& status)  
{
    m_ai_tracker_status = status;

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
    if (m_ai_tracker_status == TrackingTarget_STATUS_AI_Recognition_DISABLED) 
        return ;  //TODO: Can report Message Here
#ifdef UDP_AI_DETECTION

#else
    de::yolo_ai::CYOLOAI& m_yolo_ai = de::yolo_ai::CYOLOAI::getInstance();
            
    m_yolo_ai.loadAllowedClassIndices(allowed_class_indices);
    m_yolo_ai.detect();
#endif
 }

 void CYOLOAI_Main::disableTracking()
 {
    pauseTracking(); 
 }

void CYOLOAI_Main::pauseTracking()
{
#ifdef UDP_AI_DETECTION

#else
    de::yolo_ai::CYOLOAI& m_yolo_ai = de::yolo_ai::CYOLOAI::getInstance();
            
    m_yolo_ai.pause();
#endif
}

void CYOLOAI_Main::onReceive (ParsedDetection detection)
{
    Json_de best_object_json = Json_de::object();
        best_object_json["x"] = roundToPrecision(detection.x, 3);
        best_object_json["y"] = roundToPrecision(detection.y, 3);
        best_object_json["w"] = roundToPrecision(detection.width, 3);
        best_object_json["h"] = roundToPrecision(detection.height, 3);
    onBestObject(best_object_json);
#ifdef DEBUG
    std::cout << "detection:" << detection.name << ":" << detection.category << std::endl;
#endif
}

void CYOLOAI_Main::enableTracking()
{
   // this state means I will accept start AI command.
   // this is not a real start for the AI core.
    m_ai_tracker_status = TrackingTarget_STATUS_AI_Recognition_ENABLED;

     // ACK
    m_trackerFacade.sendTrackingTargetStatus (
        std::string(""),
        m_ai_tracker_status
    );
}
