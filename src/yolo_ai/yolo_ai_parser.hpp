#ifndef YOLO_AI_PARSER_H
#define YOLO_AI_PARSER_H


#include "./yolo_ai_main.hpp"
#include "./yolo_ai_facade.hpp"

#include "../helpers/json_nlohmann.hpp"

using Json_de = nlohmann::json;

namespace de
{
namespace yolo_ai
{

    class CYOLOAI_Parser
    {
        public:

        CYOLOAI_Parser()
        {

        };


        public:

            void parseMessage (Json_de &andruav_message, const char * message, const int & message_length);
            
        protected:
            
            void parseRemoteExecute (Json_de &andruav_message);

            inline bool validateField (const Json_de& message, const char *field_name, const Json_de::value_t field_type)
            {
                if (
                    (message.contains(field_name) == false) 
                    || (message[field_name].type() != field_type)
                    ) 
                    return false;

                return true;
            }

        private:
            de::yolo_ai::CYOLOAI_Main&  m_tracker_main = de::yolo_ai::CYOLOAI_Main::getInstance();
            de::yolo_ai::CYOLOAI_Facade& m_tracker_facade = de::yolo_ai::CYOLOAI_Facade::getInstance();
    };
}
}


#endif