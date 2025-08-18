#ifndef YOLO_AI_CALLBACK_H
#define YOLO_AI_CALLBACK_H

#include "../helpers/json_nlohmann.hpp"

using Json_de = nlohmann::json;

namespace de
{
namespace yolo_ai
{

class CCallBack_YOLOAI
{
    public:
        virtual void onTrack (const Json_de targets) = 0;
        virtual void onBestObject (const Json_de targets) = 0;
        virtual void onTrackStatusChanged (const int& status) = 0;
};
}
}

#endif