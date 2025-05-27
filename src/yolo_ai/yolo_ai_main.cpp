#include <stdio.h>
#include <iostream>

#include "../helpers/colors.hpp"
#include "../helpers/helpers.hpp"

#include "../de_common/configFile.hpp"
#include "../de_common/configFile.hpp"

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

    std::cout << _SUCCESS_CONSOLE_BOLD_TEXT_ << "Video Output:" << _INFO_CONSOLE_TEXT << output_video_device << _NORMAL_CONSOLE_TEXT_ << std::endl;
    
    if (!validateField(jsonConfig, "model_path", nlohmann::json::value_t::string))
    {
        std::cout << _ERROR_CONSOLE_BOLD_TEXT_ << "Fatal Error:" << _NORMAL_CONSOLE_TEXT_ << " Missing field or bad string format " << _INFO_CONSOLE_BOLD_TEXT << " model_path " << _NORMAL_CONSOLE_TEXT_<< std::endl;
        exit(1);
    }
    
    std::string model_path = jsonConfig["model_path"].get<std::string>();

    std::cout << _SUCCESS_CONSOLE_BOLD_TEXT_ << "model_path:" << _INFO_CONSOLE_TEXT << model_path << _NORMAL_CONSOLE_TEXT_ << std::endl;
    
    if (!validateField(jsonConfig, "class_names", nlohmann::json::value_t::array))
    {
        std::cout << _ERROR_CONSOLE_BOLD_TEXT_ << "Fatal Error:" << _NORMAL_CONSOLE_TEXT_ << "Missing field or bad array format " << _INFO_CONSOLE_BOLD_TEXT << " classNames " << _NORMAL_CONSOLE_TEXT_<< std::endl;
        exit(1);
    }
    
    std::vector<std::string> class_names;

    for (const auto& element : jsonConfig["class_names"]) {
            if (element.is_string()) {
                class_names.push_back(element.get<std::string>());
            } else {
                std::cout << "Warning: Non-string element found in classNames array. Skipping." << std::endl;
            }
        }
    
    std::cout << _SUCCESS_CONSOLE_BOLD_TEXT_ << "class_names:" << _INFO_CONSOLE_TEXT << "filled." << _NORMAL_CONSOLE_TEXT_ << std::endl;
    
    m_yolo_ai.init(source_video_device, model_path, output_video_device, class_names);

    int status = m_yolo_ai.run();

    if (status == 0) {
        fprintf(stderr, "Application completed successfully.\n");
    } else {
        fprintf(stderr, "Application failed with error code: %d\n", status);
    }
    return true;
}


bool CYOLOAI_Main::uninit()
{
    
    return true;
}



void CYOLOAI_Main::startYolo()
{

}
   
    
