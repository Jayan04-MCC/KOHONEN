cmake_minimum_required(VERSION 3.30)
project(KOHONEN)

set(CMAKE_CXX_STANDARD 20)

set(OpenCV_DIR "C:/opencv/build/opencv/build/x64/vc16/lib")  # Ruta al build de OpenCV
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(KOHONEN main.cpp)
target_link_libraries(KOHONEN ${OpenCV_LIBS})

