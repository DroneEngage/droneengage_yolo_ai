#ifndef YOLO_AI_PARSER_H
#define YOLO_AI_PARSER_H

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
            //de::yolo_ai::CTrackerMain&  m_trackerMain = de::yolo_ai::CTrackerMain::getInstance();

    };
}
}


#endif