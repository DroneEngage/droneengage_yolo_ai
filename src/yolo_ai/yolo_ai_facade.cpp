#include "../helpers/colors.hpp"
#include "yolo_ai_facade.hpp"



using namespace de::yolo_ai;

void CYOLOAI_Facade::sendTrackingTargetsLocation( const std::string& target_party_id, const Json_de targets_location) const
{
    if (targets_location.empty())
    {
        return;
    }

    Json_de message =
    {
        {"t", targets_location}
    };

    #ifdef DDEBUG
        std::cout << "tracking:" << targets_location.dump() << std::endl;
    #endif
    #ifdef DEBUG
        std::cout << _INFO_CONSOLE_BOLD_TEXT << "onTrack >> " 
        << _LOG_CONSOLE_BOLD_TEXT << targets_location.dump() << _NORMAL_CONSOLE_TEXT_ << std::endl;
    #endif
    m_module.sendJMSG(target_party_id, message, TYPE_AndruavMessage_TrackingTargetLocation, true);
}


void CYOLOAI_Facade::sendTrackingTargetStatus(const std::string& target_party_id, const int status) const
{
    Json_de message =
    {
        {"a", status}
    };

    m_module.sendJMSG(target_party_id, message, TYPE_AndruavMessage_AI_Recognition_STATUS, true);

    #ifdef DEBUG
        std::cout << "TrackingStatus:" << status << std::endl;
    #endif
}