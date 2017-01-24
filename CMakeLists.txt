cmake_minimum_required(VERSION 2.6.0)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive -g -Wall -DNDEBUG -std=c++11 -fPIC")

# Check the architecture and set CMAKE_LIBRARY_ARCHITECTURE accordingly
if(UNIX)
  if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(CMAKE_LIBRARY_ARCHITECTURE "x86_64-linux-gnu")
  endif()
endif()

include(cmake/FindNumPy.cmake)
include(cmake/FindCholmod.cmake)
include(cmake/boost-python.cmake)

###################################################################################
# Find GTSAM components
find_package(GTSAM REQUIRED) # Uses installed package

# Include python
set(EIGEN3_INCLUDE_DIR ${CMAKE_INSTALL_PREFIX}/include/gtsam/3rdparty/Eigen)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_INSTALL_PREFIX}/include)
include_directories(${CMAKE_INSTALL_PREFIX}/include/gtsam/3rdparty/Eigen)
include_directories(${NUMPY_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${CHOLMOD_INCLUDES})

# =========== bot_eigen_types Python wrapper ==============
boost_python_module(pybot_eigen_types
    src/pybot_eigen_types.cpp
    src/eigen_numpy_conversion.cpp)
target_link_libraries(pybot_eigen_types boost_system boost_python glog ${CHOLMOD_LIBRARIES})
install(TARGETS pybot_eigen_types
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

# =========== pygtsam Python wrapper ==============
boost_python_module(pygtsam src/pygtsam.cpp)
target_link_libraries(pygtsam boost_python boost_system pybot_eigen_types gtsam)
install(TARGETS pygtsam
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
