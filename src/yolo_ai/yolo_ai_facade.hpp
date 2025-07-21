#ifndef YOLO_AI_FACADE_H
#define YOLO_AI_FACADE_H

#include "../helpers/json_nlohmann.hpp"

using Json_de = nlohmann::json;


#include "../de_common/de_facade_base.hpp"
#include "../de_common/messages.hpp"


namespace de
{
namespace yolo_ai
{
    class CYOLOAI_Facade : public de::comm::CFacade_Base
    {
        public:
            //https://stackoverflow.com/questions/1008019/c-singleton-design-pattern
            static CYOLOAI_Facade& getInstance()
            {
                static CYOLOAI_Facade instance;

                return instance;
            }

            CYOLOAI_Facade(CYOLOAI_Facade const&)         = delete;
            void operator=(CYOLOAI_Facade const&)          = delete;

        
            // Note: Scott Meyers mentions in his Effective Modern
            //       C++ book, that deleted functions should generally
            //       be public as it results in better error messages
            //       due to the compilers behavior to check accessibility
            //       before deleted status

        private:

            CYOLOAI_Facade()
            {
            };

        public:
            
            ~CYOLOAI_Facade ()
            {
                
            };
                

        public:
           
            void sendTrackingTargetsLocation(const std::string& target_party_id, const Json_de targets_location) const;
            void sendTrackingClassesList(const std::string& target_party_id);
            void sendTrackingTargetStatus(const std::string& target_party_id, const int status) const;
        
        
        private:

            
    };
}
}


#endif