#include <stdio.h>
#include <iostream>

#include "../de_common/helpers/colors.hpp"
#include "../de_common/helpers/helpers.hpp"

#include "../de_common/de_databus/configFile.hpp"
#include "../de_common/de_databus/messages.hpp"

#include "video.hpp"
#include "yolo_ai_best_box.hpp"
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

    m_de_camera_draw = false;
    m_frames_to_skip_between_messages = 3;

    if (jsonConfig.contains("advanced_tracking"))
    {
        const Json_de advanced_tracking = jsonConfig["advanced_tracking"];

        if (advanced_tracking.contains("de_camera_draw"))
        {
            m_de_camera_draw = advanced_tracking["de_camera_draw"].get<bool>();
        }

        if (advanced_tracking.contains("frames_to_skip_between_messages"))
        {
            m_frames_to_skip_between_messages = advanced_tracking["frames_to_skip_between_messages"].get<int>();
        }
        m_frames_to_skip_between_messages = m_frames_to_skip_between_messages <= 0 ? 1 : m_frames_to_skip_between_messages;
    }

    std::cout << _LOG_CONSOLE_BOLD_TEXT
              << "Using de_camera_draw:" << _INFO_CONSOLE_BOLD_TEXT
              << m_de_camera_draw << _NORMAL_CONSOLE_TEXT_
              << std::endl;

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
    m_yolo_ai.setAdvancedTrackingParameters(m_frames_to_skip_between_messages, m_de_camera_draw);
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


    #ifdef DEBUG
    std::cout << _INFO_CONSOLE_BOLD_TEXT << "onTrackStatusChanged:" << _LOG_CONSOLE_BOLD_TEXT << std::to_string(status) << _NORMAL_CONSOLE_TEXT_ << std::endl;
    #endif
}

/**
 * Called by CYOLOAI when de_camera_draw is enabled and the detection set is
 * updated. overlay_detections is an array of {x,y,w,h normalized [0..1],
 * label, best} forwarded to de_camera's overlay.
 */
void CYOLOAI_Main::onCameraOverlayDetections (const Json_de overlay_detections)
{
    m_trackerFacade.sendCameraOverlayDetections (
        std::string(""),
        overlay_detections
    );
}

/**
 * Called by CYOLOAI when the detection overlay shapes should be removed
 * from de_camera (detections lost, paused, stopped, or draw mode disabled).
 */
void CYOLOAI_Main::onCameraOverlayRemove ()
{
    m_trackerFacade.sendCameraOverlayRemove (
        std::string("")
    );
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
    // B4-01: keep the detector's class + confidence on the wire
    const Json_de best_object_json = yoloBestBoxJson(
        detection.x, detection.y, detection.width, detection.height,
        detection.confidence, (int) detection.category, detection.name);
    onBestObject(best_object_json);
#ifdef DEBUG
    std::cout << "detection:" << detection.name << ":" << detection.category << std::endl;
#endif

    // UDP-fed detections carry no local video frames, so de_camera_draw sends
    // the box straight to de_camera's overlay (single best detection, on the
    // messaging cadence).
    if (m_de_camera_draw &&
        (m_overlay_send_counter++ % m_frames_to_skip_between_messages) == 0)
    {
        if (detection.width > 0 && detection.height > 0)
        {
            Json_de overlay_detections = Json_de::array();
            overlay_detections.push_back({
                {"x", detection.x},
                {"y", detection.y},
                {"w", detection.width},
                {"h", detection.height},
                {"label", detection.name},
                {"best", true}});
            m_trackerFacade.sendCameraOverlayDetections(std::string(""), overlay_detections);
        }
        else
        {
            m_trackerFacade.sendCameraOverlayRemove(std::string(""));
        }
    }
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
