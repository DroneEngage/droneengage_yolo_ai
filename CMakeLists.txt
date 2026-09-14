cmake_minimum_required(VERSION 3.10.0)

if(NOT WIN32)
  string(ASCII 27 Esc)
  SET(ColourReset "${Esc}[m")
  SET(ColourBold  "${Esc}[1m")
  SET(Red         "${Esc}[31m")
  SET(Green       "${Esc}[32m")
  SET(Yellow      "${Esc}[33m")
  SET(Blue        "${Esc}[34m")
  SET(Magenta     "${Esc}[35m")
  SET(Cyan        "${Esc}[36m")
  SET(White       "${Esc}[37m")
  SET(BoldRed     "${Esc}[1;31m")
  SET(BoldGreen   "${Esc}[1;32m")
  SET(BoldYellow  "${Esc}[1;33m")
  SET(BoldBlue    "${Esc}[1;34m")
  SET(BoldMagenta "${Esc}[1;35m")
  SET(BoldCyan    "${Esc}[1;36m")
  SET(BoldWhite   "${Esc}[1;37m")
endif()


if ( ${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR} )
message ("${Red} In-source builds not allowed! Create a build directory and run CMake from there.  ${ColourReset}")
message( FATAL_ERROR “In-source builds not allowed! Create a build directory and run CMake from there. ” )
exit(1)
endif()

# ──────────────────────────────────────────────────────────────
# Auto-increment version on each build
# ──────────────────────────────────────────────────────────────
set(VERSION_FILE "${CMAKE_SOURCE_DIR}/.version")

# Manually set major, minor, and bugfix versions (you can change these)
set(MAJOR_VERSION 1)
set(MINOR_VERSION 4)
set(BUGFIX_VERSION 0)

if(EXISTS ${VERSION_FILE})
    file(READ ${VERSION_FILE} CURRENT_VERSION)
    string(STRIP "${CURRENT_VERSION}" CURRENT_VERSION)
    # Extract current version components
    string(REPLACE "." ";" VERSION_LIST ${CURRENT_VERSION})
    list(LENGTH VERSION_LIST LIST_LENGTH)
    if(LIST_LENGTH EQUAL 4)
        list(GET VERSION_LIST 0 STORED_MAJOR)
        list(GET VERSION_LIST 1 STORED_MINOR)
        list(GET VERSION_LIST 2 STORED_BUGFIX)
        list(GET VERSION_LIST 3 STORED_BUILD)
        
        # Check if major, minor, or bugfix version changed
        if(NOT (${STORED_MAJOR} EQUAL ${MAJOR_VERSION} AND ${STORED_MINOR} EQUAL ${MINOR_VERSION} AND ${STORED_BUGFIX} EQUAL ${BUGFIX_VERSION}))
            set(BUILD_VERSION 0)  # Reset build when major/minor/bugfix changes
            message(STATUS "Version changed from ${STORED_MAJOR}.${STORED_MINOR}.${STORED_BUGFIX}.${STORED_BUILD} to ${MAJOR_VERSION}.${MINOR_VERSION}.${BUGFIX_VERSION}, resetting build to 0")
        else()
            set(BUILD_VERSION ${STORED_BUILD})  # Continue with existing build
        endif()
    else()
        set(BUILD_VERSION 0)  # Default build version
    endif()
else()
    set(BUILD_VERSION 0)  # Default build version
endif()

# Increment build version only for RELEASE builds
IF(CMAKE_BUILD_TYPE MATCHES RELEASE)
    math(EXPR BUILD_VERSION "${BUILD_VERSION} + 1")
    message(STATUS "Incrementing build version for RELEASE build")
ELSE()
    message(STATUS "Non-release build - keeping current build version")
ENDIF()

# Create new version string
set(__APP__VERSION__ "${MAJOR_VERSION}.${MINOR_VERSION}.${BUGFIX_VERSION}.${BUILD_VERSION}")

message(STATUS "Building version: ${__APP__VERSION__}")
message(STATUS "To change major/minor/bugfix versions, edit MAJOR_VERSION, MINOR_VERSION, and BUGFIX_VERSION above")

SET (__APP__VERSION__ "${__APP__VERSION__}")
add_definitions( -D__APP__VERSION__="${__APP__VERSION__}")
project(DE_YOLO_AI VERSION "${__APP__VERSION__}")



SET (CMAKE_C_COMPILER /usr/bin/gcc)
SET (CMAKE_CXX_COMPILER /usr/bin/g++)

#C++ Version
SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(Threads REQUIRED)

# Find OpenCV package
find_package(OpenCV 4.5 REQUIRED)
message(STATUS "OpenCV_INCLUDE_DIRS found: ${OpenCV_INCLUDE_DIRS}")


#define default build
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
SET(CMAKE_BUILD_TYPE DEBUG)
ENDIF()

message ("${Yellow}=========================================================================${ColourReset}")
message ("CMAKE_C_COMPILER: "  ${CMAKE_C_COMPILER})
message ("CMAKE_CXX_COMPILER: " ${CMAKE_CXX_COMPILER})

#deceide debug or release
IF(CMAKE_BUILD_TYPE MATCHES DEBUG)
message("CMAKE_BUILD_TYPE: ${BoldYellow} DEBUG ${ColourReset}")
ELSEIF (CMAKE_BUILD_TYPE MATCHES RELEASE)
message("CMAKE_BUILD_TYPE: ${BoldGreen} RELEASE ${ColourReset}")
ENDIF()

# adding CMAKE_CXX_FLAGS properly
add_compile_options(
  -Wno-return-type 
  -Wno-address-of-packed-member
  -Werror=unused-variable
  -Werror=unused-result
  $<$<CONFIG:DEBUG>:-g3>
  $<$<CONFIG:DEBUG>:-Og>
  $<$<CONFIG:RELEASE>:-O2>
  $<$<CONFIG:RELEASE>:-Werror=parentheses>
)

# adding DEFINEs
add_compile_definitions(
    $<$<CONFIG:DEBUG>:DEBUG>
    $<$<CONFIG:RELEASE>:RELEASE>
)

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--no-export-dynamic")


SET(OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
file(MAKE_DIRECTORY ${OUTPUT_DIRECTORY})
SET(EXECUTABLE_OUTPUT_PATH ${OUTPUT_DIRECTORY})


#add_subdirectory(src)
file(GLOB folder_src "./src/*.cpp")
file(GLOB folder_common "./src/de_common/de_databus/*.cpp")
file(GLOB folder_common1 "./src/de_common/helpers/*.cpp")
file(GLOB folder_ai "./src/yolo_ai/*.cpp")

# Define an option for user control over DETAILED_DEBUG
option(DDEBUG "Detailed Debug" OFF) # Default is OFF
option(UDP_AI_DETECTION "SOURCE:  IMX500 or any UDP-AI" OFF) # Default is OFF
option(HAILO_AI_DETECTION "SOURCE: HAILO Board" ON) # Default is OFF
option(TEST_MODE_NO_HAILO_LINK "SOURCE: FAKE SOURCE - Do not link to Hailo LIB" OFF) # Default is OFF


set(files ${folder_src} ${folder_common} ${folder_common1} ${folder_ai})

include_directories(${OpenCV_INCLUDE_DIRS}) # Add OpenCV include directories

add_executable( de_yolo_ai ${files})


# Conditionally define DDEBUG based on the DDEBUG option
target_compile_definitions(de_yolo_ai PRIVATE
    $<$<BOOL:${DDEBUG}>:DDEBUG>
)
# Link the required libraries
if((NOT TEST_MODE_NO_HAILO_LINK) AND (NOT UDP_AI_DETECTION))
  target_link_libraries(de_yolo_ai 
    hailort
  )
  target_compile_definitions (de_yolo_ai PRIVATE HAILO_AI_DETECTION)
elseif (UDP_AI_DETECTION)
  target_compile_definitions (de_yolo_ai PRIVATE UDP_AI_DETECTION)
else()
  # Define the preprocessor macro for TEST_MODE_NO_HAILO_LINK
  target_compile_definitions(de_yolo_ai PRIVATE TEST_MODE_NO_HAILO_LINK)
endif()


target_link_libraries(de_yolo_ai ${OpenCV_LIBS})
target_link_libraries(de_yolo_ai  Threads::Threads)

set_target_properties( de_yolo_ai 
                PROPERTIES 
                    OUTPUT_NAME "de_ai_tracker"
                    SUFFIX ".so"
                )




target_compile_options(de_yolo_ai
    PRIVATE
        -Wall
        -Wno-return-type-c-linkage
)

# Highlight if DDEBUG or TEST_MODE_NO_HAILO_LINK are enabled
if (DDEBUG)
    message(STATUS "${Red}Option DDEBUG is ENABLED.${ColourReset}")
endif()

if (TEST_MODE_NO_HAILO_LINK)
    message(STATUS "${Green}Option TEST_MODE_NO_HAILO_LINK is ENABLED.${ColourReset}")
    message(STATUS "${Red}(No Hailo Link) HAILO will not be detected.${ColourReset}")
endif()

if (HAILO_AI_DETECTION)
    message(STATUS "${Green}Option HAILO-AI is ENABLED.${ColourReset}")
endif()

if (UDP_AI_DETECTION)
    message(STATUS "${Green}Option UDP_AI_DETECTION (UDP-Remote-Detection) is ENABLED.${ColourReset}")
    message(STATUS "${Red}(No Hailo Link) HAILO will not be detected.${ColourReset}")
endif()

configure_file(de_ai_tracker.config.module.json ${OUTPUT_DIRECTORY}/de_ai_tracker.config.module.json COPYONLY)

message ("${Yellow}=========================================================================${ColourReset}")
message (STATUS "Version              : ${__APP__VERSION__}")

# ──────────────────────────────────────────────────────────────
# Build success script and version management
# ──────────────────────────────────────────────────────────────
set(BUILD_SUCCESS_SCRIPT "${CMAKE_BINARY_DIR}/build_success.cmake")
file(WRITE "${BUILD_SUCCESS_SCRIPT}" "")
file(APPEND "${BUILD_SUCCESS_SCRIPT}" "execute_process(COMMAND \"${CMAKE_COMMAND}\" -E echo \"\")\n")
file(APPEND "${BUILD_SUCCESS_SCRIPT}" "execute_process(COMMAND \"${CMAKE_COMMAND}\" -E cmake_echo_color --green --bold \"=========================================================================\")\n")
file(APPEND "${BUILD_SUCCESS_SCRIPT}" "execute_process(COMMAND \"${CMAKE_COMMAND}\" -E cmake_echo_color --green --bold \"BUILD COMPLETED SUCCESSFULLY\")\n")
file(APPEND "${BUILD_SUCCESS_SCRIPT}" "execute_process(COMMAND \"${CMAKE_COMMAND}\" -E cmake_echo_color --cyan --bold \"Version: ${__APP__VERSION__}\")\n")
file(APPEND "${BUILD_SUCCESS_SCRIPT}" "execute_process(COMMAND \"${CMAKE_COMMAND}\" -E cmake_echo_color --cyan --bold \"Build Number: ${BUILD_VERSION}\")\n")
file(APPEND "${BUILD_SUCCESS_SCRIPT}" "execute_process(COMMAND \"${CMAKE_COMMAND}\" -E cmake_echo_color --green --bold \"=========================================================================\")\n")
file(APPEND "${BUILD_SUCCESS_SCRIPT}" "execute_process(COMMAND \"${CMAKE_COMMAND}\" -E echo \"\")\n")

# Write version file directly for RELEASE builds
IF(CMAKE_BUILD_TYPE MATCHES RELEASE)
    file(WRITE "${VERSION_FILE}" "${__APP__VERSION__}")
    message(STATUS "Version written to ${VERSION_FILE}: ${__APP__VERSION__}")
ENDIF()

