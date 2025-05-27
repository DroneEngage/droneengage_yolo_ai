#include <iostream>
#include "../defines.hpp"
#include "../de_common/messages.hpp"
#include "yolo_ai_parser.hpp"


using namespace de::yolo_ai;

/// @brief Parses & executes messages received from uavos_comm"
/// @param parsed JSON message received from uavos_comm 
/// @param full_message 
/// @param full_message_length 
void CYOLOAI_Parser::parseMessage (Json_de &andruav_message, const char * full_message, const int & full_message_length)
{
    const int messageType = andruav_message[ANDRUAV_PROTOCOL_MESSAGE_TYPE].get<int>();
    bool is_binary = !(full_message[full_message_length-1]==125 || (full_message[full_message_length-2]==125));   // "}".charCodeAt(0)  IS TEXT / BINARY Msg  
    UNUSED(is_binary);

    if (messageType == TYPE_AndruavMessage_RemoteExecute)
    {
        parseRemoteExecute(andruav_message);

        return ;
    }

    else
    {
        Json_de message = andruav_message[ANDRUAV_PROTOCOL_MESSAGE_CMD];
        
        switch (messageType)
        {

            // case TYPE_AndruavMessage_TrackingTarget:
            // {
            //     // a: center X
            //     // b: center Y
            //     // r: radius

            //     if ((message.contains("s")) && (message["s"].get<bool>() == true))
            //     {
            //         // stop tracking
            //         m_trackerMain.stopTracking();
            //         return ;
            //     }
            //     if (!validateField(message, "a", Json_de::value_t::number_float)) return ;
            //     if (!validateField(message, "b", Json_de::value_t::number_float)) return ;
            //     if (!validateField(message, "r", Json_de::value_t::number_unsigned)) return ;
                
            //     m_trackerMain.startTracking(message["a"].get<float>(),
            //                                 message["b"].get<float>(), 
            //                                 message["r"].get<float>());
            // }
            break;

        }
    }
}


/**
 * @brief part of parseMessage that is responsible only for
 * parsing remote execute command.
 * 
 * @param andruav_message 
 */
void CYOLOAI_Parser::parseRemoteExecute (Json_de &andruav_message)
{
    const Json_de cmd = andruav_message[ANDRUAV_PROTOCOL_MESSAGE_CMD];
    
    if (!validateField(cmd, "C", Json_de::value_t::number_unsigned)) return ;
                
    const int remoteCommand = cmd["C"].get<int>();
    std::cout << "cmd: " << remoteCommand << std::endl;
    switch (remoteCommand)
    {

    }
}