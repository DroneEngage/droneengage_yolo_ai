/***************************************************************************************
 * 
 * Author: https://github.com/bmharper
 * Origin of Code: https://github.com/bmharper/hailo-rpi5-yolov8
 * 
 ***************************************************************************************/

 #pragma once

#include <memory>
#include <string>

#ifndef TEST_MODE_NO_HAILO_LINK

class OutTensor {
public:
	uint8_t*               data;
	std::string            name;
	hailo_quant_info_t     quant_info;
	hailo_3d_image_shape_t shape;
	hailo_format_t         format;

	OutTensor(uint8_t* data, const std::string& name, const hailo_quant_info_t& quant_info,
	          const hailo_3d_image_shape_t& shape, hailo_format_t format)
	    : data(data), name(name), quant_info(quant_info), shape(shape), format(format) {
	}

	static bool SortFunction(const OutTensor& l, const OutTensor& r) {
		return l.shape.width < r.shape.width;
	}
};


#endif