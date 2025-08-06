#include "../helpers/colors.hpp"
#include "yolo_ai_facade.hpp"

#include "yolo_ai_main.hpp"

using namespace de::yolo_ai;

void CYOLOAI_Facade::sendTrackingTargetsLocation(const std::string& target_party_id, const Json_de targets_location) const
{
    if (targets_location.empty())
    {
        return;
    }

    Json_de message =
    {
        {"t", targets_location}
    };

    #ifdef DEBUG
        std::cout << _INFO_CONSOLE_BOLD_TEXT << "onBestObject >> " 
        << _LOG_CONSOLE_BOLD_TEXT << targets_location.dump() << _NORMAL_CONSOLE_TEXT_ << std::endl;
    #endif

    // internal message
    m_module.sendJMSG(target_party_id, message, TYPE_AndruavMessage_AI_Recognition_TargetLocation, true);
}

void CYOLOAI_Facade::sendTrackingBestTargetsLocation(const std::string& target_party_id, const Json_de targets_location) const
{
    if (targets_location.empty())
    {
        return;
    }

    Json_de message =
    {
        {"b", targets_location}
    };

    #ifdef DEBUG
        std::cout << _INFO_CONSOLE_BOLD_TEXT << "onBestObject >> " 
        << _LOG_CONSOLE_BOLD_TEXT << targets_location.dump() << _NORMAL_CONSOLE_TEXT_ << std::endl;
    #endif

    // internal message
    m_module.sendJMSG(target_party_id, message, TYPE_AndruavMessage_AI_Recognition_TargetLocation, true);
}
            


void CYOLOAI_Facade::sendTrackingClassesList(const std::string& target_party_id)
{
    const std::vector<std::string> class_names = de::yolo_ai::CYOLOAI_Main::getInstance().getTrackingObjectsList();
    
    Json_de json_class_names = Json_de::array();
    
    for (const auto& element : class_names) {
        json_class_names.push_back(
            element
        );
    }

    Json_de message =
    {
        {"a", TrackingTarget_STATUS_AI_Recognition_CLASS_LIST},
        {"c", json_class_names}
    };

    m_module.sendJMSG(target_party_id, message, TYPE_AndruavMessage_AI_Recognition_STATUS, false);
}

void CYOLOAI_Facade::sendTrackingTargetStatus(const std::string& target_party_id, const int status) const
{
    Json_de message =
    {
        {"a", status}
    };

    m_module.sendJMSG(target_party_id, message, TYPE_AndruavMessage_AI_Recognition_STATUS, false);

    #ifdef DEBUG
        std::cout << "TrackingStatus:" << status << std::endl;
    #endif
}