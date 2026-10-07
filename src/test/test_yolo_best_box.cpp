/**
 * @file test_yolo_best_box.cpp
 * @brief standalone tests for the 1078 best-object box builder
 *        (yolo_ai/yolo_ai_best_box.hpp, TASK-B4-01).
 *
 * Covers: in-range and out-of-range class ids, the "unknown" class-name
 * fallback, an explicitly-provided name (the UDP/IMX500 path), geometry
 * rounding, and confidence preservation - all without Hailo.
 *
 * Build & run:  make -C src/test test   &&   ./src/test/test_yolo_best_box
 */

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "../de_common/helpers/json_nlohmann.hpp"
#include "../yolo_ai/yolo_ai_best_box.hpp"

using Json = nlohmann::json;

#define ASSERT_TRUE(condition) \
    do { \
        if (!(condition)) { \
            std::cerr << "ASSERTION FAILED: " << #condition << " at line " << __LINE__ << std::endl; \
            return false; \
        } \
    } while(0)

#define ASSERT_EQ(val1, val2) ASSERT_TRUE((val1) == (val2))
#define ASSERT_NEAR(val1, val2, eps) ASSERT_TRUE(std::fabs((val1) - (val2)) <= (eps))


static const std::vector<std::string> CLASS_NAMES = {
    "person", "bicycle", "car"
};


bool test_in_range_class ()
{
    std::cout << "Testing: in-range class id resolves its name..." << std::endl;

    const Json box = de::yolo_ai::yoloBestBoxJson(
        0.1, 0.2, 0.3, 0.4, 0.85, 2, CLASS_NAMES);

    ASSERT_EQ(box["class_id"], 2);
    ASSERT_EQ(box["class_name"], std::string("car"));
    ASSERT_NEAR(box["conf"].get<double>(), 0.85, 1e-9);

    std::cout << "passed" << std::endl;
    return true;
}


bool test_out_of_range_class ()
{
    std::cout << "Testing: out-of-range class id falls back to 'unknown'..." << std::endl;

    Json box = de::yolo_ai::yoloBestBoxJson(
        0.1, 0.2, 0.3, 0.4, 0.7, 99, CLASS_NAMES);
    ASSERT_EQ(box["class_id"], 99);
    ASSERT_EQ(box["class_name"], std::string("unknown"));

    // negative id as well
    box = de::yolo_ai::yoloBestBoxJson(
        0.1, 0.2, 0.3, 0.4, 0.7, -1, CLASS_NAMES);
    ASSERT_EQ(box["class_name"], std::string("unknown"));

    // empty class list
    box = de::yolo_ai::yoloBestBoxJson(
        0.1, 0.2, 0.3, 0.4, 0.7, 0, std::vector<std::string>());
    ASSERT_EQ(box["class_name"], std::string("unknown"));

    std::cout << "passed" << std::endl;
    return true;
}


bool test_explicit_class_name ()
{
    std::cout << "Testing: explicit name (UDP/IMX500 path)..." << std::endl;

    const Json box = de::yolo_ai::yoloBestBoxJson(
        0.1, 0.2, 0.3, 0.4, 0.6, 7, std::string("bottle"));

    ASSERT_EQ(box["class_id"], 7);
    ASSERT_EQ(box["class_name"], std::string("bottle"));

    std::cout << "passed" << std::endl;
    return true;
}


bool test_rounding_and_confidence ()
{
    std::cout << "Testing: geometry rounds to 3 decimals, conf preserved..." << std::endl;

    const Json box = de::yolo_ai::yoloBestBoxJson(
        0.1234567, 0.9876543, 0.20001, 0.39999, 0.8123456, 0, CLASS_NAMES);

    ASSERT_NEAR(box["x"].get<double>(), 0.123, 1e-9);
    ASSERT_NEAR(box["y"].get<double>(), 0.988, 1e-9);
    ASSERT_NEAR(box["w"].get<double>(), 0.200, 1e-9);
    ASSERT_NEAR(box["h"].get<double>(), 0.400, 1e-9);
    ASSERT_NEAR(box["conf"].get<double>(), 0.8123456, 1e-9);

    std::cout << "passed" << std::endl;
    return true;
}


int main ()
{
    bool ok = true;

    ok &= test_in_range_class();
    ok &= test_out_of_range_class();
    ok &= test_explicit_class_name();
    ok &= test_rounding_and_confidence();

    if (ok)
    {
        std::cout << std::endl << "ALL TESTS PASSED" << std::endl;
        return 0;
    }

    std::cout << std::endl << "TESTS FAILED" << std::endl;
    return 1;
}
