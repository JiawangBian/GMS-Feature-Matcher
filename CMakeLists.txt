cmake_minimum_required(VERSION 2.8)
project( gms_match_demo)

# OpenCV 
find_package( OpenCV REQUIRED )

# 
include_directories( 
 include
${OpenCV_INCLUDE_DIRS} 
)


add_executable( gms_match_demo ./src/demo.cpp )
target_link_libraries( gms_match_demo ${OpenCV_LIBS} )
