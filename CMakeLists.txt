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

SET (__APP__VERSION__ "1.3.3")
add_definitions( -D__APP__VERSION__="${__APP__VERSION__}")
project(DE_YOLO_AI VERSION ${__APP__VERSION__})



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
file(GLOB folder_helpers "./src/helpers/*.cpp")
file(GLOB folder_common "./src/de_common/*.cpp")
file(GLOB folder_ai "./src/yolo_ai/*.cpp")

# Define an option for user control over DETAILED_DEBUG
option(DDEBUG "Detailed Debug" OFF) # Default is OFF
option(TEST_MODE_NO_HAILO_LINK "Do not link to Hailo LIB" OFF) # Default is OFF


set(files ${folder_src} ${folder_common} ${folder_ai} ${folder_helpers})

include_directories(${OpenCV_INCLUDE_DIRS}) # Add OpenCV include directories

add_executable( de_yolo_ai ${files})


# Conditionally define DDEBUG based on the DDEBUG option
target_compile_definitions(de_yolo_ai PRIVATE
    $<$<BOOL:${DDEBUG}>:DDEBUG>
)
# Link the required libraries
if(NOT TEST_MODE_NO_HAILO_LINK)
  target_link_libraries(de_yolo_ai 
    hailort
  )
else()
  # Define the preprocessor macro for TEST_MODE_NO_HAILO_LINK
  target_compile_definitions(de_yolo_ai PRIVATE TEST_MODE_NO_HAILO_LINK)
endif()


target_link_libraries(de_yolo_ai ${OpenCV_LIBS})
target_link_libraries(de_yolo_ai  Threads::Threads)

set_target_properties( de_yolo_ai 
                PROPERTIES 
                    OUTPUT_NAME "de_yolo_ai"
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
    message(STATUS "${Red}Option TEST_MODE_NO_HAILO_LINK (No Hailo Link) is ENABLED.${ColourReset}")
endif()

configure_file(de_yolo_ai.config.module.json ${OUTPUT_DIRECTORY}/de_yolo_ai.config.module.json COPYONLY)


message ("${Yellow}=========================================================================${ColourReset}")

