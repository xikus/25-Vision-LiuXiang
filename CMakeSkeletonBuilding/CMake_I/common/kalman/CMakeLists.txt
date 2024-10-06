set(kalman_INCLUDE_DIR)
list(APPEND kalman_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/include")
list(APPEND kalman_INCLUDE_DIR "${OpenCV_INCLUDE_DIRS}")
message(STATUS "kalman include dir: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "kalman include dir: ${kalman_INCLUDE_DIR}")
set(kalman ${kalman_INCLUDE_DIR}
                        CACHE PATH "kalman filter include directories")