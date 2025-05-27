#ifndef DEFINES_H_
#define DEFINES_H_

#include <limits> 
#include <string>
#include <stdint.h>

#if !defined(UNUSED)
#define UNUSED(x) (void)(x) // Variables and parameters that are not used
#endif




/**
 * @brief configuration file path & name
 * 
 */
static std::string configName = "de_yolo_ai.config.module.json";
static std::string localConfigName = "de_yolo_ai.config.local";

#endif