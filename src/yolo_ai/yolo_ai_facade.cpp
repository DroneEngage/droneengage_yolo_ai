#include "yolo_ai_facade.hpp"



using namespace de::yolo_ai;

// void CYOLOAI_Facade::sendTrackingTargetsLocation( const std::string& target_party_id, const Json_de targets_location) const
// {
//     if (targets_location.empty())
//     {
//         return;
//     }

//     Json_de message =
//     {
//         {"t", targets_location}
//     };


//     m_module.sendJMSG(target_party_id, target_party_id, TYPE_AndruavMessage_TrackingTargetLocation, true);
// }