set(singleton_INCLUDE_DIR)
list(APPEND singleton_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/include)
list(APPEND singleton_INCLUDE_DIR "${OpenCV_INCLUDE_DIRS}")

set(singleton ${singleton_INCLUDE_DIR} 
                        CACHE PATH "singleton include directories")
                        