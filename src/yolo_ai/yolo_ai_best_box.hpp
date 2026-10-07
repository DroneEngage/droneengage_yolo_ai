#ifndef YOLO_AI_BEST_BOX_H_
#define YOLO_AI_BEST_BOX_H_

/**
 * @brief TASK-B4-01: builds the best-object box "b" of
 * TYPE_AndruavMessage_AI_Recognition_TargetLocation (1078).
 *
 * Both producers (the Hailo NMS loop and the UDP/IMX500 source) emit the
 * same shape so the geo projectors can report the DETECTED class and
 * gate on the real confidence:
 *
 *     {"x":..,"y":..,"w":..,"h":..,"conf":..,
 *      "class_id":0,"class_name":"person"}
 *
 * x,y is the normalized top-left corner; w,h the normalized size - all
 * rounded to 3 decimals like before. class_name resolves through the
 * class list when the id is in range, "unknown" otherwise (the
 * yolo_ai_generic convention); a producer with an already-resolved name
 * (the UDP/IMX500 path) passes it directly.
 *
 * Pure header so the box shape is unit-testable without Hailo
 * (src/test/test_yolo_best_box.cpp).
 */

#include <string>
#include <vector>

#include "../de_common/helpers/json_nlohmann.hpp"
#include "../de_common/helpers/helpers.hpp"

using Json_de = nlohmann::json;

namespace de
{
namespace yolo_ai
{

    inline Json_de yoloBestBoxJson (const double x, const double y,
                                    const double w, const double h,
                                    const double confidence,
                                    const int class_id,
                                    const std::string& class_name)
    {
        Json_de box = Json_de::object();
        box["x"] = roundToPrecision(x, 3);
        box["y"] = roundToPrecision(y, 3);
        box["w"] = roundToPrecision(w, 3);
        box["h"] = roundToPrecision(h, 3);
        box["conf"] = confidence;
        box["class_id"] = class_id;
        box["class_name"] = class_name.empty() ? "unknown" : class_name;
        return box;
    }


    inline Json_de yoloBestBoxJson (const double x, const double y,
                                    const double w, const double h,
                                    const double confidence,
                                    const int class_id,
                                    const std::vector<std::string>& class_names)
    {
        const std::string class_name =
            (class_id >= 0 && class_id < (int) class_names.size())
                ? class_names[class_id] : std::string();
        return yoloBestBoxJson(x, y, w, h, confidence, class_id, class_name);
    }

}
}

#endif
