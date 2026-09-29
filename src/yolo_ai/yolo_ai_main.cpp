#include <stdio.h>
#include <iostream>

#include "../de_common/helpers/colors.hpp"
#include "../de_common/helpers/helpers.hpp"

#include "../de_common/de_databus/configFile.hpp"
#include "../de_common/de_databus/messages.hpp"

#include "video.hpp"

#include "../version.hpp"
#include "yolo_ai_main.hpp"




using namespace de::yolo_ai;

// AI_Recognition_STATUS (1077) code -> capability state string
static const char* aiStateName (const int status)
{
    switch (status)
    {
        case TrackingTarget_STATUS_AI_Recognition_LOST:      return "lost";
        case TrackingTarget_STATUS_AI_Recognition_DETECTED:  return "detected";
        case TrackingTarget_STATUS_AI_Recognition_ENABLED:   return "enabled";
        case TrackingTarget_STATUS_AI_Recognition_DISABLED:  return "disabled";
        default:                                             return "disabled";
    }
}


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

    // P3-07: mirror into the visual_tracker capability state
    if (m_caps_advertised)
    {
        de::comm::CModule::getInstance().publishState("visual_tracker",
            {{"ai", aiStateName(status)}});
    }
    

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
        // P3-07: see the same fields added in yolo_ai.cpp
        best_object_json["conf"] = roundToPrecision(detection.confidence, 3);
        best_object_json["tm"] = get_time_usec_monotonic();
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

    if (m_caps_advertised)
    {
        de::comm::CModule::getInstance().publishState("visual_tracker",
            {{"ai", aiStateName(m_ai_tracker_status)}});
    }
}


// ---------------------------------------------------------------------------
// Phase-3 capability advert "visual_tracker" (TASK-P3-07).
// start/stop map to the same handlers the legacy AI_Recognition_ACTION (1076)
// parser uses; de_tracker advertises "track" under the same namespace.
// ---------------------------------------------------------------------------

void CYOLOAI_Main::setupCapabilities ()
{
    de::comm::CModule& cModule = de::comm::CModule::getInstance();

    // start {class}: an enum of the configured class list when it is known,
    // a plain string otherwise (class_names come from the config file)
    Json_de class_param = {{"type", "string"}, {"desc", "class to search for"}};
    if (!m_class_names.empty())
    {
        class_param = {{"type", "enum"}, {"values", m_class_names}};
    }

    const Json_de advert =
    {
        {"schema",  "de.cap/1"},
        {"ns",      "visual_tracker"},
        {"module",  "droneengage_yolo_ai"},
        {"ver",     std::string(version_string)},
        {"actions", {
            {"start", {
                {"desc", "Enable AI recognition and search for a class"},
                {"params", {
                    {"class", class_param}
                }}
            }},
            {"stop", {
                {"desc", "Disable AI recognition"}
            }}
        }},
        {"state", {
            {"ai", {{"type", "enum"},
                    {"values", {"lost", "detected", "enabled", "disabled"}},
                    {"desc", "AI_Recognition_STATUS mirror"}}}
        }}
    };

    if (!cModule.setCapabilities({advert.dump()}))
    {
        std::cout << _ERROR_CONSOLE_BOLD_TEXT_ << "capability advert rejected for ns visual_tracker" << _NORMAL_CONSOLE_TEXT_ << std::endl;
        return;
    }

    cModule.onInvoke([this](const std::string& id, const std::string& ns,
                            const std::string& act, const Json_de& params,
                            std::string& err) -> Json_de
    {
        return onCapabilityInvoke(id, ns, act, params, err);
    });

    m_caps_advertised = true;
}


Json_de CYOLOAI_Main::onCapabilityInvoke (const std::string& id,
                                        const std::string& ns,
                                        const std::string& act,
                                        const Json_de& params,
                                        std::string& err)
{
    (void) id;

    if (ns != "visual_tracker")
    {
        err = "unknown namespace " + ns;
        return Json_de::object();
    }

    if (act == "start")
    {
        const std::string cls = params.value("class", "");
        if (cls.empty())
        {
            err = "missing class";
            return Json_de::object();
        }

        // resolve the class name to its index in the configured list
        int index = -1;
        for (size_t i = 0; i < m_class_names.size(); ++i)
        {
            if (m_class_names[i] == cls) { index = (int) i; break; }
        }
        if (index < 0)
        {
            err = "unknown class '" + cls + "'";
            return Json_de::object();
        }

        // same path as AI_Recognition_ACTION ENABLE + SEARCH (1076)
        enableTracking();
        startTrackingObjects(Json_de::array({index}));
        // let de_mavlink's detect projector map the selected index to a name
        m_trackerFacade.sendTrackingClassesList(std::string(""));
        return {{"started", true}, {"class", cls}, {"index", index}};
    }

    if (act == "stop")
    {
        // same path as AI_Recognition_ACTION DISABLE (1076)
        disableTracking();
        return {{"stopped", true}};
    }

    err = "unknown action " + ns + "." + act;
    return Json_de::object();
}
