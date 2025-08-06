#include <chrono> // For high-resolution timing
#include <iostream>
#include <signal.h>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>  // For std::sort
#include <memory>     // For std::unique_ptr, std::shared_ptr
#include <opencv2/opencv.hpp>
#include "../helpers/colors.hpp"
#include "../helpers/helpers.hpp"

#include "../defines.hpp"

#ifndef TEST_MODE_NO_HAILO_LINK
#include <hailo/hailort.hpp>
#include <hailo/hailort_common.hpp>
#include <hailo/vdevice.hpp>
#include <hailo/infer_model.hpp>
#endif

#include "../de_common/messages.hpp"

// Headers for V4L2
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>
#include "output_tensor.hpp"
#include "video.hpp"
#include "yolo_ai.hpp"

using namespace de::yolo_ai;

// Default paths and parameters
float confidenceThreshold = 0.5f;
float nmsIoUThreshold = 0.45f;

// Helper function to print V4L2 errors
void errno_exit(const char *s) {
    fprintf(stderr, "%s error %d, %s\n", s, errno, strerror(errno));
    exit(EXIT_FAILURE);
}

bool CYOLOAI::init(const std::string& source_video_path, const std::string& hef_model_path, const std::string& output_video_device, std::vector<std::string>& class_names, CCallBack_YOLOAI *callback_yolo_ai)
{
    #ifdef DDEBUG
    std::cout << __FILE__ << "." << __FUNCTION__ << " line:" << __LINE__ << " " << _NORMAL_CONSOLE_TEXT_ << std::endl;
    #endif

    m_callback_yolo_ai = callback_yolo_ai;
    m_class_names = class_names;
    m_source_video_device = source_video_path;
    m_output_video_device = output_video_device;
    m_hef_model_path = hef_model_path;
    
    #ifdef DDEBUG
    std::cout << _SUCCESS_CONSOLE_BOLD_TEXT_ << __FILE__ << "." << __FUNCTION__ << " line:" << __LINE__ << " " << _NORMAL_CONSOLE_TEXT_ << std::endl;
    #endif
    return true;
}


bool CYOLOAI::uninit() { 
    
    stop();
    return true;
}



void CYOLOAI::detect()
{
    m_is_AI_yolo_active_initial = true;
    
    if (m_callback_yolo_ai != nullptr)
    {
        m_callback_yolo_ai->onTrackStatusChanged(TrackingTarget_STATUS_AI_Recognition_ENABLED);
    }
}


void CYOLOAI::pause()
{
    std::cout << "pause" <<std::endl;
    m_is_AI_yolo_active_initial = false;

    if (m_callback_yolo_ai != nullptr)
    {
        m_callback_yolo_ai->onTrackStatusChanged(TrackingTarget_STATUS_AI_Recognition_DISABLED);
    }
}


void CYOLOAI::stop()
{
    if (m_exit_thread) return ; 
    m_exit_thread = true;
    
    if (m_callback_yolo_ai != nullptr)
    {
        m_callback_yolo_ai->onTrackStatusChanged(TrackingTarget_STATUS_AI_Recognition_DISABLED);
    }
}


void CYOLOAI::loadAllowedClassIndices(const Json_de& json_array) {
    m_allowed_class_indices.clear(); // Clear any previous entries
    if (json_array.is_array()) {
        for (const auto& item : json_array) {
            if (item.is_number_integer()) {
                m_allowed_class_indices.insert(item.get<size_t>());
            } else {
                std::cerr << "Warning: Non-integer value found in class index list. Skipping." << std::endl;
            }
        }
    } else {
        std::cerr << "Error: Provided JSON for class indices is not an array." << std::endl;
    }
}

int CYOLOAI::run() {
    using namespace std::literals::chrono_literals;

#ifndef TEST_MODE_NO_HAILO_LINK
    using namespace hailort;

    // --- HailoRT Initialization ---
    // Use `auto` for cleaner type deduction, and check directly if `vdevice_exp` is valid.
    auto vdevice_exp = VDevice::create();
    if (!vdevice_exp) {
        std::cerr << _ERROR_CONSOLE_BOLD_TEXT_ << "Error: Failed to create vdevice: " << vdevice_exp.status() << _NORMAL_CONSOLE_TEXT_ << std::endl;
        return vdevice_exp.status();
    }
    // No need for .release() here as Expected can be directly converted/moved
    // to a unique_ptr. If VDevice::create returns Expected<unique_ptr<VDevice>>,
    // then vdevice_exp itself holds the unique_ptr.
    std::unique_ptr<VDevice> vdevice = vdevice_exp.release();


    auto infer_model_exp = vdevice->create_infer_model(m_hef_model_path);
    if (!infer_model_exp) {
        std::cerr << _ERROR_CONSOLE_BOLD_TEXT_ << "Error: Failed to create infer model from HEF " << m_hef_model_path << ": " << infer_model_exp.status() << _NORMAL_CONSOLE_TEXT_ << std::endl;
        return infer_model_exp.status();
    }
    std::shared_ptr<InferModel> infer_model = infer_model_exp.release();

    infer_model->set_hw_latency_measurement_flags(HAILO_LATENCY_MEASURE);
    infer_model->output()->set_nms_score_threshold(confidenceThreshold);
    infer_model->output()->set_nms_iou_threshold(nmsIoUThreshold);

    // Use `const auto&` for efficiency when accessing members of a collection.
    // Ensure `inputs()` returns a non-empty vector before accessing `[0]`.
    if (infer_model->inputs().empty()) {
        std::cerr << _ERROR_CONSOLE_BOLD_TEXT_ << "Error: Model has no input layers." << _NORMAL_CONSOLE_TEXT_ << std::endl;
        return HAILO_UNINITIALIZED; // Or appropriate error
    }
    const auto& input_layer_shape = infer_model->inputs()[0].shape();
    int nnWidth = input_layer_shape.width;
    int nnHeight = input_layer_shape.height;
 
    std::cout << _LOG_CONSOLE_BOLD_TEXT << "Network Input Resolution: " << _INFO_CONSOLE_TEXT << nnWidth << "x" << nnHeight << _NORMAL_CONSOLE_TEXT_ << std::endl;

    auto configured_infer_model_exp = infer_model->configure();
    if (!configured_infer_model_exp) {
        std::cerr << _ERROR_CONSOLE_BOLD_TEXT_ << "Error: Failed to configure infer model: " << configured_infer_model_exp.status() << _NORMAL_CONSOLE_TEXT_ << std::endl;
        return configured_infer_model_exp.status();
    }
    // Using `std::move` from `configured_infer_model_exp` directly into `make_shared` to avoid an unnecessary copy.
    std::shared_ptr<ConfiguredInferModel> configured_infer_model =
        std::make_shared<ConfiguredInferModel>(configured_infer_model_exp.release());

    auto bindings_exp = configured_infer_model->create_bindings();
    if (!bindings_exp) {
        std::cerr << _ERROR_CONSOLE_BOLD_TEXT_ << "Error: Failed to create bindings: " << bindings_exp.status() << _NORMAL_CONSOLE_TEXT_ << std::endl;
        return bindings_exp.status();
    }
    
    // Efficiently move bindings
    ConfiguredInferModel::Bindings bindings = bindings_exp.release();

    const std::string& input_name = infer_model->get_input_names()[0];
    size_t input_frame_size = infer_model->input(input_name)->get_frame_size();
    fprintf(stderr, "Input tensor name: %s, frame size: %zu bytes (for %dx%d input)\n", input_name.c_str(), input_frame_size, nnWidth, nnHeight);
#else // TEST_MODE_NO_HAILO_LINK
    unsigned int nnWidth = 640; // Default or example AI input width
    unsigned int nnHeight = 480; // Default or example AI input height
    std::cout << _LOG_CONSOLE_BOLD_TEXT << "Target AI Input Resolution (for resizing): " << _INFO_CONSOLE_TEXT << nnWidth << "x" << nnHeight << _NORMAL_CONSOLE_TEXT_ << std::endl;
#endif

    // --- OpenCV Camera Initialization ---
    cv::VideoCapture video_capture;
    /**
     * without an explicit backend. 
     *  When you don't specify cv::CAP_V4L2 (or cv::CAP_DSHOW on Windows, etc.),
     *  OpenCV attempts to auto-detect the camera and the best API to use. 
     * This auto-detection process can be time-consuming and often defaults to a less efficient or more generic backend
     *  (like GStreamer if it's available and configured,
     *  or a generic CAP_ANY interface) that adds more layers of abstraction and overhead.
     */
    if (!video_capture.open(m_source_video_device.c_str(), cv::CAP_V4L2)) {
        std::cerr << _ERROR_CONSOLE_BOLD_TEXT_ << "Error: Failed to open camera " << m_source_video_device << " with V4L2 backend" << _NORMAL_CONSOLE_TEXT_ << std::endl;
        return 1;
    }

    // Set camera properties once after opening.
    // Error checking for `set` calls is good practice, though not strictly required if non-critical.
    video_capture.set(cv::CAP_PROP_FPS, 30);
    video_capture.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));

    // Get the actual full resolution from the camera
    int original_frame_width = static_cast<int>(video_capture.get(cv::CAP_PROP_FRAME_WIDTH));
    int original_frame_height = static_cast<int>(video_capture.get(cv::CAP_PROP_FRAME_HEIGHT));
    double actual_fps = video_capture.get(cv::CAP_PROP_FPS);
    UNUSED(actual_fps);
    if (original_frame_width == 0 || original_frame_height == 0) {
        std::cerr << _ERROR_CONSOLE_BOLD_TEXT_ << "Error: Failed to get camera resolution from " << m_source_video_device << _NORMAL_CONSOLE_TEXT_ << std::endl;
        video_capture.release();
        return 1;
    }

    if (original_frame_width < nnWidth || original_frame_height < nnHeight) {
        // fprintf(stderr, "Warning: Original camera resolution (%dx%d) is smaller than AI network input (%dx%d). AI input will be upscaled.\n",
        //         original_frame_width, original_frame_height, nnWidth, nnHeight);
        std::cerr << _ERROR_CONSOLE_BOLD_TEXT_ << "Info: Original camera resolution (" << _INFO_CONSOLE_BOLD_TEXT << original_frame_width << _LOG_CONSOLE_BOLD_TEXT << "x" << _INFO_CONSOLE_BOLD_TEXT << original_frame_height << _ERROR_CONSOLE_BOLD_TEXT_ 
            << ") is smaller than AI network input " << _INFO_CONSOLE_BOLD_TEXT << "(" << _INFO_CONSOLE_BOLD_TEXT << nnWidth << _LOG_CONSOLE_BOLD_TEXT << "x" << _INFO_CONSOLE_BOLD_TEXT << nnHeight << "). AI input will be upscaled. " <<  _NORMAL_CONSOLE_TEXT_ << std::endl;
    } else if (original_frame_width != nnWidth || original_frame_height != nnHeight) {
        //  fprintf(stderr, "Info: Original camera resolution (%dx%d) differs from AI network input (%dx%d). Frame will be resized for AI.\n",
        //         original_frame_width, original_frame_height, nnWidth, nnHeight);
        std::cerr << _ERROR_CONSOLE_BOLD_TEXT_ << "Info: Original camera resolution (" << _INFO_CONSOLE_BOLD_TEXT << original_frame_width << _LOG_CONSOLE_BOLD_TEXT << "x" << _INFO_CONSOLE_BOLD_TEXT << original_frame_height << _ERROR_CONSOLE_BOLD_TEXT_ 
            << ") differs from AI network input (" << _INFO_CONSOLE_BOLD_TEXT << nnWidth << _LOG_CONSOLE_BOLD_TEXT << "x" << _INFO_CONSOLE_BOLD_TEXT << nnHeight << "). Frame will be resized for AI. " <<  _NORMAL_CONSOLE_TEXT_ << std::endl;
            
    }

    // --- V4L2 Output Device Initialization ---
    int video_fd = -1;
    video_fd = open(m_output_video_device.c_str(), O_WRONLY | O_NONBLOCK);
    if (video_fd < 0) {
        std::cout << _ERROR_CONSOLE_BOLD_TEXT_ << "Failed to open virtual video device " << m_output_video_device << ": " << strerror(errno) << _NORMAL_CONSOLE_TEXT_ << std::endl;
        video_capture.release();
        return 1;
    }
    std::cout << _SUCCESS_CONSOLE_BOLD_TEXT_<< "Successfully opened virtual video device: " << _LOG_CONSOLE_BOLD_TEXT << m_output_video_device << _NORMAL_CONSOLE_TEXT_ << std::endl;

    struct v4l2_format fmt = {0};
    fmt.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
    fmt.fmt.pix.width = static_cast<__u32>(original_frame_width);
    fmt.fmt.pix.height = static_cast<__u32>(original_frame_height);
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUV420;  // YUV420 planar (I420)
    fmt.fmt.pix.field = V4L2_FIELD_NONE;            // Progressive scan
    fmt.fmt.pix.bytesperline = static_cast<__u32>(original_frame_width); // Stride of the Y plane
    fmt.fmt.pix.sizeimage = (static_cast<__u32>(original_frame_width) * static_cast<__u32>(original_frame_height) * 3) / 2;

    if (de::yolo_ai::CVideo::xioctl(video_fd, VIDIOC_S_FMT, &fmt) < 0) {
        std::cout << _ERROR_CONSOLE_BOLD_TEXT_ << "Failed to set video format on " << m_output_video_device << ": " << strerror(errno) << _NORMAL_CONSOLE_TEXT_ << std::endl;
        close(video_fd);
        video_capture.release();
        return 1;
    }
    
    std::cout << _SUCCESS_CONSOLE_BOLD_TEXT_<< "Successfully set format for " << m_output_video_device << ":" << _INFO_CONSOLE_BOLD_TEXT << fmt.fmt.pix.width << _LOG_CONSOLE_BOLD_TEXT << "x" << _INFO_CONSOLE_BOLD_TEXT << fmt.fmt.pix.height << _LOG_CONSOLE_BOLD_TEXT << " pixformat YUV420" << _NORMAL_CONSOLE_TEXT_ <<  std::endl;

    // Pre-allocate Mats outside the loop to avoid reallocations.
    // if the size is already correct. This avoids repeated memory allocation/deallocation.
    cv::Mat original_bgr_frame;
    cv::Mat nn_input_rgb_frame(nnHeight, nnWidth, CV_8UC3); // Directly allocate for RGB to avoid an intermediate BGR.
    cv::Mat yuv_output_frame(original_frame_height * 3 / 2, original_frame_width, CV_8UC1); // YUV420 size

    const size_t output_yuv_frame_size = (static_cast<size_t>(original_frame_width) * original_frame_height * 3) / 2;
    
    // Pre-calculate the scale factors for converting normalized coordinates
    const float scale_x = static_cast<float>(original_frame_width);
    const float scale_y = static_cast<float>(original_frame_height);

    // Outside the loop, after infer_model is configured:
    std::vector<std::vector<uint8_t>> pre_allocated_output_buffers;
    pre_allocated_output_buffers.reserve(infer_model->get_output_names().size());
    // Populate pre_allocated_output_buffers with correctly sized buffers
    for (const auto& output_name_str : infer_model->get_output_names()) {
        size_t output_size = infer_model->output(output_name_str)->get_frame_size();
        pre_allocated_output_buffers.emplace_back(output_size); // Allocates a vector of `output_size` bytes
    }

    m_exit_thread = false;

    uint64_t frame_counter = 0;


    std::vector<OutTensor> output_tensors;
output_tensors.reserve(infer_model->get_output_names().size());

for (size_t i = 0; i < infer_model->get_output_names().size(); ++i) {
    const auto& output_name_str = infer_model->get_output_names()[i];
    uint8_t* output_buffer_ptr = pre_allocated_output_buffers[i].data();
    size_t output_size = pre_allocated_output_buffers[i].size();
    auto status = bindings.output(output_name_str)->set_buffer(MemoryView(output_buffer_ptr, output_size));
    if (status != HAILO_SUCCESS) {
        std::cerr << _ERROR_CONSOLE_BOLD_TEXT_ << "Failed to set output buffer for " << output_name_str << ": " << status << _NORMAL_CONSOLE_TEXT_ << std::endl;
        return status; // Critical setup error, exit function
    }
    const auto quant = infer_model->output(output_name_str)->get_quant_infos();
    const auto shape = infer_model->output(output_name_str)->shape();
    const auto format = infer_model->output(output_name_str)->format();
    output_tensors.emplace_back(output_buffer_ptr, output_name_str, quant[0], shape, format);
}

// Constants defined outside the loop
const float max_distance_sq = 0.5f; // Max squared distance in normalized coordinates
const cv::Point2f frame_center(0.5f, 0.5f); // Constant within scope
            
// Main processing loop
while (!m_exit_thread) {
    auto start_time = std::chrono::high_resolution_clock::now();

    // Capture frame
    if (!video_capture.read(original_bgr_frame) || original_bgr_frame.empty()) {
        std::cerr << _ERROR_CONSOLE_BOLD_TEXT_ << "Failed to capture frame or end of stream." << _NORMAL_CONSOLE_TEXT_ << std::endl;
        continue;
    }

    // AI processing (when active)
    if (m_is_AI_yolo_active_initial) {
        // Resize for neural network input
        cv::resize(original_bgr_frame, nn_input_rgb_frame, cv::Size(nnWidth, nnHeight));

#ifndef TEST_MODE_NO_HAILO_LINK
        // Set input buffer (must be inside loop as input changes per frame)
        auto status = bindings.input(input_name)->set_buffer(MemoryView(nn_input_rgb_frame.data, input_frame_size));
        if (status != HAILO_SUCCESS) {
            std::cerr << _ERROR_CONSOLE_BOLD_TEXT_ << "Failed to set input memory buffer: " << status << _NORMAL_CONSOLE_TEXT_ << std::endl;
            break;
        }

        // Run inference
        status = configured_infer_model->wait_for_async_ready(1s);
        if (status != HAILO_SUCCESS) {
            std::cerr << _ERROR_CONSOLE_BOLD_TEXT_ << "Error: Failed to wait for async ready - " << status << _NORMAL_CONSOLE_TEXT_ << std::endl;
            break;
        }

        auto job_exp = configured_infer_model->run_async(bindings, [](const AsyncInferCompletionInfo&){});
        if (!job_exp) {
            std::cerr << _ERROR_CONSOLE_BOLD_TEXT_ << "Error: Failed to start async job: " << job_exp.status() << _NORMAL_CONSOLE_TEXT_ << std::endl;
            break;
        }
        AsyncInferJob job = job_exp.release();

        status = job.wait(1s);
        if (status != HAILO_SUCCESS) {
            std::cerr << _ERROR_CONSOLE_BOLD_TEXT_ << "Error: Failed to wait for job completion: " << status << _NORMAL_CONSOLE_TEXT_ << std::endl;
            break;
        }
#endif // TEST_MODE_NO_HAILO_LINK

        // Post-processing and drawing
#ifndef TEST_MODE_NO_HAILO_LINK
        bool nmsOnHailo = !infer_model->outputs().empty() && infer_model->outputs()[0].is_nms();
        if (nmsOnHailo) {
            if (output_tensors.empty()) {
                std::cerr << "Error: No output tensors received from Hailo." << std::endl;
                break;
            }

            const OutTensor& out = output_tensors[0];
            const float* raw = reinterpret_cast<const float*>(out.data);
            size_t nms_output_byte_size = infer_model->output(out.name)->get_frame_size();

            size_t numClasses = static_cast<size_t>(out.shape.height);
            size_t current_idx = 0;

            float best_score = -1.0f;
            Json_de best_object_json;
            cv::Rect best_bbox;

            bool object_found = false;
            Json_de object_found_list = Json_de::array();

            for (size_t classIdx = 0; classIdx < numClasses; ++classIdx) {
                if (!m_allowed_class_indices.empty() && m_allowed_class_indices.find(classIdx) == m_allowed_class_indices.end()) {
                    if (current_idx * sizeof(float) >= nms_output_byte_size) {
                        std::cerr << "Error: NMS output parsing index out of bounds (class count - " << classIdx << ")." << std::endl;
                        break;
                    }
                    const size_t numBoxesToSkip = static_cast<size_t>(raw[current_idx++]);
                    current_idx += numBoxesToSkip * 5;
                    continue;
                }

                if (current_idx * sizeof(float) >= nms_output_byte_size) {
                    std::cerr << "Error: NMS output parsing index out of bounds (class count - " << classIdx << ")." << std::endl;
                    break;
                }
                const size_t numBoxes = static_cast<size_t>(raw[current_idx++]);

                for (size_t i = 0; i < numBoxes; ++i) {
                    if ((current_idx + 4) * sizeof(float) >= nms_output_byte_size) {
                        std::cerr << "Error: NMS output parsing index out of bounds (box data - " << i << " of class " << classIdx << ")." << std::endl;
                        break;
                    }
                    const float ymin_norm = raw[current_idx];
                    const float xmin_norm = raw[current_idx + 1];
                    const float ymax_norm = raw[current_idx + 2];
                    const float xmax_norm = raw[current_idx + 3];
                    const float confidence = raw[current_idx + 4];

                    if (confidence >= confidenceThreshold) {
                        object_found = true;

                        cv::Point2f box_center((xmin_norm + xmax_norm) / 2.0f, (ymin_norm + ymax_norm) / 2.0f);
                        const float distance_to_center_sq = pow(box_center.x - frame_center.x, 2) + pow(box_center.y - frame_center.y, 2);
                        const float normalized_distance_to_center = distance_to_center_sq / max_distance_sq;

                        float normalized_distance_to_prev = 1.0f;
                        if (!m_prev_best_object_json.empty()) {
                            float prev_x = m_prev_best_object_json["x"].get<float>();
                            float prev_y = m_prev_best_object_json["y"].get<float>();
                            float prev_center_x = prev_x + m_prev_best_object_json["w"].get<float>() / 2.0f;
                            float prev_center_y = prev_y + m_prev_best_object_json["h"].get<float>() / 2.0f;
                            const float distance_to_prev_sq = pow(box_center.x - prev_center_x, 2) + pow(box_center.y - prev_center_y, 2);
                            normalized_distance_to_prev = distance_to_prev_sq / max_distance_sq;
                        }

                        const float area = (xmax_norm - xmin_norm) * (ymax_norm - ymin_norm);
                        const float normalized_area = area / 1.0f; // Max area = 1.0

                        const float score = 0.5f * confidence + 0.3f * (1.0f - normalized_distance_to_center) +
                                            0.15f * (1.0f - normalized_distance_to_prev) + 0.05f * normalized_area;

                        const int x1 = static_cast<int>(xmin_norm * scale_x);
                        const int y1 = static_cast<int>(ymin_norm * scale_y);
                        const int x2 = static_cast<int>(xmax_norm * scale_x);
                        const int y2 = static_cast<int>(ymax_norm * scale_y);

                        std::string label_text;
                        if (best_score < 0 || score > best_score) {
                            best_score = score;
                            best_object_json = Json_de::object();
                            best_object_json["x"] = roundToPrecision(xmin_norm, 3);
                            best_object_json["y"] = roundToPrecision(ymin_norm, 3);
                            best_object_json["w"] = roundToPrecision(xmax_norm - xmin_norm, 3);
                            best_object_json["h"] = roundToPrecision(ymax_norm - ymin_norm, 3);
                            best_bbox = cv::Rect(x1, y1, x2 - x1, y2 - y1);
                        } else {
                            const cv::Scalar color = cv::Scalar(0, 200, 0) :
                            cv::rectangle(original_bgr_frame, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);
                        }

                        label_text = (classIdx < m_class_names.size()) ?
                                     cv::format("%s: %.2f", m_class_names[classIdx].c_str(), confidence) :
                                     cv::format("Class %zu: %.2f", classIdx, confidence);

                        int baseline = 0;
                        cv::getTextSize(label_text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
                        cv::putText(original_bgr_frame, label_text, cv::Point(x1, y1 - 5),
                                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);

                        object_found_list.push_back({
                            {"x", roundToPrecision(xmin_norm, 3)},
                            {"y", roundToPrecision(ymin_norm, 3)},
                            {"w", roundToPrecision(xmax_norm - xmin_norm, 3)},
                            {"h", roundToPrecision(ymax_norm - ymin_norm, 3)}
                        });
                    }
                    current_idx += 5; // Move to the next box
                }
            }

            // Draw the best box in black if it exists
            if (!best_object_json.empty()) {
                cv::rectangle(original_bgr_frame, best_bbox, cv::Scalar(0, 0, 0), 2);
                m_callback_yolo_ai->onBestObject(best_object_json);
                m_prev_best_object_json = best_object_json; // Update previous best object for next frame
            } else {
                m_prev_best_object_json.clear(); // Clear previous object if no best object is found
            }

            if (m_object_found != object_found) {
                if (object_found) {
                    m_callback_yolo_ai->onTrackStatusChanged(TrackingTarget_STATUS_AI_Recognition_DETECTED);
#ifdef DDEBUG
                    std::cout << _SUCCESS_CONSOLE_BOLD_TEXT_ << "Object Detected." << _NORMAL_CONSOLE_TEXT_ << std::endl;
#endif
                } else {
                    m_callback_yolo_ai->onTrackStatusChanged(TrackingTarget_STATUS_AI_Recognition_LOST);
#ifdef DDEBUG
                    std::cout << _INFO_CONSOLE_BOLD_TEXT << "No Object Found." << _NORMAL_CONSOLE_TEXT_ << std::endl;
#endif
                }
                m_object_found = object_found;
            }
            if (object_found) {
                m_callback_yolo_ai->onTrack(object_found_list);
                object_found_list.clear();
            }
        } else {
            std::cout << "Info: NMS on CPU is not implemented or NMS output structure is different." << std::endl;
        }
#endif // TEST_MODE_NO_HAILO_LINK
    } // End of AI processing block

    // Convert to YUV420p and write to virtual device
    cv::cvtColor(original_bgr_frame, yuv_output_frame, cv::COLOR_BGR2YUV_I420);
    if (yuv_output_frame.isContinuous() && yuv_output_frame.total() * yuv_output_frame.elemSize() == output_yuv_frame_size) {
        const ssize_t bytes_written = write(video_fd, yuv_output_frame.data, output_yuv_frame_size);
        if (bytes_written < 0) {
            std::cerr << _ERROR_CONSOLE_BOLD_TEXT_ << "Error: Failed to write frame to V4L2 device " << m_output_video_device << ": " << strerror(errno) << _NORMAL_CONSOLE_TEXT_ << std::endl;
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                std::cerr << _INFO_CONSOLE_TEXT << "Warning: Virtual device " << m_output_video_device << " buffer full? Try reading from it." << _NORMAL_CONSOLE_TEXT_ << std::endl;
            } else {
                break;
            }
        }
    } else {
        std::cerr << _ERROR_CONSOLE_BOLD_TEXT_ << "Error: YUV output frame is not continuous or has unexpected size." << _NORMAL_CONSOLE_TEXT_ << std::endl;
        
        #ifdef DDEBUG
                fprintf(stderr, "Debug: yuv_output_frame.total() * elemSize() = %zu, expected output_yuv_frame_size = %zu\n",
                    yuv_output_frame.total() * yuv_output_frame.elemSize(), output_yuv_frame_size);
                fprintf(stderr, "Debug: yuv_output_frame continuous: %d, dims: %dx%d, channels: %d, depth: %d, elemSize: %zu\n",
                    yuv_output_frame.isContinuous(), yuv_output_frame.cols, yuv_output_frame.rows, yuv_output_frame.channels(), yuv_output_frame.depth(), yuv_output_frame.elemSize());
            #endif
            break; // Fatal error, exit loop
        }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

#ifdef DDEBUG
    if (frame_counter % 10 == 0) {
        std::cout << _INFO_CONSOLE_BOLD_TEXT << "Elapsed: " << _LOG_CONSOLE_BOLD_TEXT << elapsed_time.count() << "ms" << std::endl;
    }
#endif
    ++frame_counter;
} // End of while loop

    // --- Cleanup ---
    // 1. Close V4L2 output device (C-style file descriptor)
    #ifdef DDEBUG
    fprintf(stderr, "Exiting application...\n");
    #endif

    if (video_fd >= 0) {
        close(video_fd);
        video_fd = -1; // Mark as closed
        #ifdef DDEBUG
        fprintf(stderr, "Closed virtual video device %s\n", m_output_video_device.c_str());
        #endif
    }
    
    // 2. Release OpenCV camera capture
    if (video_capture.isOpened()) { // Check if it was successfully opened
        video_capture.release();
        #ifdef DDEBUG
        fprintf(stderr, "Released OpenCV camera capture %s\n", m_source_video_device.c_str());
        #endif
    }

    #ifdef DDEBUG
    // 3. HailoRT resources (managed by smart pointers, mostly self-cleaning)
    // The unique_ptr (vdevice) and shared_ptr (infer_model, configured_infer_model)
    // will automatically call their destructors when they go out of scope at the end of this function.
    // This is the beauty of RAII!
    fprintf(stderr, "HailoRT resources (vdevice, infer_model, configured_infer_model, bindings) are being deallocated by smart pointers.\n");

    fprintf(stderr, "DEBUG: Exiting cleanup phase of CYOLOAI::run().\n");

    std::cout << _SUCCESS_CONSOLE_BOLD_TEXT_ << "Successfully exit CYOLOAI::run()." << _NORMAL_CONSOLE_TEXT_ << std::endl;
    #endif
    return 0;
}