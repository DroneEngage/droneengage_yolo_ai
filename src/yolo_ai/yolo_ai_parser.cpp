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
        Json_de cmd = andruav_message[ANDRUAV_PROTOCOL_MESSAGE_CMD];
        std::cout << cmd << std::endl;
        switch (messageType)
        {

            case TYPE_AndruavMessage_AI_Recognition_ACTION:
            {
                
                if (!cmd.contains("a") || !cmd["a"].is_number_integer()) return ;

                switch (cmd["a"].get<int>())
                {

                    case TrackingTarget_ACTION_AI_Recognition_SEARCH:
                    {
                        if (!cmd.contains("i") || !cmd["i"].is_array()) return ; // bad command parameters
                        const Json_de& class_indicies = cmd["i"];
                        
                        #ifdef DEBUG
                        std::cout << "TrackingTarget_ACTION_AI_Recognition_SEARCH" << std::endl;
                        #endif
                        m_tracker_main.startTrackingObjects(class_indicies);
                    }
                    break;
                    
                    case TrackingTarget_ACTION_AI_Recognition_DISABLE:
                    {
                        #ifdef DEBUG
                        std::cout << "TrackingTarget_ACTION_AI_Recognition_DISABLE" << std::endl;
                        #endif
                        m_tracker_main.disableTracking();
                    }
                    break;

                    
                    case TrackingTarget_ACTION_AI_Recognition_ENABLE:
                    {
                        #ifdef DEBUG
                        std::cout << "TrackingTarget_ACTION_AI_Recognition_ENABLE" << std::endl;
                        #endif
                        m_tracker_main.enableTracking();
                    }
                    break;

                    case TrackingTarget_ACTION_AI_Recognition_CLASS_LIST:
                    {
                        std::cout << "TrackingTarget_ACTION_AI_Recognition_CLASS_LIST" << std::endl;
                        m_tracker_facade.sendTrackingClassesList(std::string(""));
                    }
                    break;
                }
            }
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