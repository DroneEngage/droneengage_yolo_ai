#include <algorithm>

#include "../de_common/helpers/colors.hpp"
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

// Shape ids used for the AI detection overlay on de_camera:
// 9300+i bounding rects, 9350+i labels (i = detection slot).
// High values to avoid colliding with shapes set by other senders
// (tracker 900x, precland 910x, ir_camera 920x, yolo_ai_generic 940x).
static constexpr uint32_t YOLOAI_OVERLAY_RECT_ID_BASE = 9300;
static constexpr uint32_t YOLOAI_OVERLAY_TEXT_ID_BASE = 9350;
static constexpr size_t   YOLOAI_OVERLAY_MAX_DETECTIONS = 50;

/**
 * Sends the detection bounding boxes to de_camera to be drawn by its
 * internal overlay library. overlay_detections is an array of
 * {x,y,w,h normalized [0..1], label string, best bool}. Every update is
 * a batched SET_SHAPE so de_camera recovers the shapes after a restart
 * or a CLEAR; ids beyond the current detection count are removed so
 * stale boxes do not linger.
 */
void CYOLOAI_Facade::sendCameraOverlayDetections(const std::string& target_party_id, const Json_de overlay_detections) const
{
    if (!overlay_detections.is_array())
    {
        return;
    }

    Json_de shapes = Json_de::array();
    size_t slot = 0;

    for (const Json_de& det : overlay_detections)
    {
        if (slot >= YOLOAI_OVERLAY_MAX_DETECTIONS) break;
        if (!det.contains("x") || !det.contains("y") || !det.contains("w") || !det.contains("h")) continue;
        if (!det["x"].is_number() || !det["y"].is_number() || !det["w"].is_number() || !det["h"].is_number()) continue;

        const float x = det["x"].get<float>();
        const float y = det["y"].get<float>();
        const float w = det["w"].get<float>();
        const float h = det["h"].get<float>();
        if (w <= 0.0f || h <= 0.0f) continue;

        const bool is_best = det.contains("best") && det["best"].is_boolean() && det["best"].get<bool>();
        const std::string label = (det.contains("label") && det["label"].is_string()) ? det["label"].get<std::string>() : "";

        // mirrors the local drawing: green boxes, the best one in black
        Json_de rect = {
            {"id", YOLOAI_OVERLAY_RECT_ID_BASE + (uint32_t)slot},
            {"type", "rect"},
            {"x1", x},
            {"y1", y},
            {"x2", x + w},
            {"y2", y + h},
            {"thickness", 2},
            {"filled", false},
            {"color", is_best ? Json_de{0, 0, 0, 255} : Json_de{0, 200, 0, 255}}};
        shapes.push_back(rect);

        if (!label.empty())
        {
            Json_de text = {
                {"id", YOLOAI_OVERLAY_TEXT_ID_BASE + (uint32_t)slot},
                {"type", "text"},
                {"x1", x},
                {"y1", std::max(0.0f, y - 0.035f)},
                {"text_height", 0.03f},
                {"text", label},
                {"color", {0, 0, 0, 255}}};
            shapes.push_back(text);
        }

        ++slot;
    }

    if (slot > 0)
    {
        Json_de message = {{"a", CAMERA_OVERLAY_ACTION_SET_SHAPE}, {"s", shapes}};
        m_module.sendJMSG(target_party_id, message, TYPE_AndruavMessage_CAMERA_OVERLAY_ACTION, true);
    }

    // drop shapes of detections that disappeared since the last update
    if (slot < (size_t)m_overlay_drawn_count)
    {
        Json_de ids = Json_de::array();
        for (size_t i = slot; i < (size_t)m_overlay_drawn_count; ++i)
        {
            ids.push_back(YOLOAI_OVERLAY_RECT_ID_BASE + (uint32_t)i);
            ids.push_back(YOLOAI_OVERLAY_TEXT_ID_BASE + (uint32_t)i);
        }
        Json_de message = {{"a", CAMERA_OVERLAY_ACTION_REMOVE_SHAPE}, {"i", ids}};
        m_module.sendJMSG(target_party_id, message, TYPE_AndruavMessage_CAMERA_OVERLAY_ACTION, true);
    }

    m_overlay_drawn_count = (int)slot;

#ifdef DEBUG_OVERLAY_MSG
    static int overlay_send_counter = 0;
    if ((overlay_send_counter++ % 30) == 0) {
        std::cout << "overlay send detections:" << slot << std::endl;
    }
#endif
}

void CYOLOAI_Facade::sendCameraOverlayRemove(const std::string& target_party_id) const
{
    if (m_overlay_drawn_count <= 0)
    {
        return;
    }

    Json_de ids = Json_de::array();
    for (int i = 0; i < m_overlay_drawn_count; ++i)
    {
        ids.push_back(YOLOAI_OVERLAY_RECT_ID_BASE + (uint32_t)i);
        ids.push_back(YOLOAI_OVERLAY_TEXT_ID_BASE + (uint32_t)i);
    }
    m_overlay_drawn_count = 0;

    Json_de message = {{"a", CAMERA_OVERLAY_ACTION_REMOVE_SHAPE}, {"i", ids}};
    m_module.sendJMSG(target_party_id, message, TYPE_AndruavMessage_CAMERA_OVERLAY_ACTION, true);

#ifdef DEBUG_OVERLAY_MSG
    std::cout << "overlay send REMOVE all detections" << std::endl;
#endif
}