cmake_minimum_required(VERSION 3.28.1)

# Metadata
project("nastro")

# Compilation
set (CMAKE_CXX_STANDARD 20)
set (CMAKE_CXX_COMPILER "/usr/bin/g++")
set (CMAKE_CXX_FLAGS_DEBUG
	"-g -O0 -Wall -fcolor-diagnostics -fansi-escape-codes")

if (NOT CMAKE_BUILD_TYPE)
	set (CMAKE_BUILD_TYPE Debug)
endif()
message(STATUS "Build configuration: ${CMAKE_BUILD_TYPE}")

# Build configuration
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)

# Install configuration
set (INSTALL_LIB_DIR ${CMAKE_INSTALL_PREFIX}/lib)
set (INSTALL_BIN_DIR ${CMAKE_INSTALL_PREFIX}/bin)
set (INSTALL_INCLUDE_DIR ${CMAKE_INSTALL_PREFIX}/include)

# Headers and sources
set (PROJECT_SOURCE_DIR ${CMAKE_SOURCE_DIR}/src)
set (PROJECT_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
include_directories(${PROJECT_INCLUDE_DIR})

# Add libraries
# add_library(<name> SHARED/STATIC <sources>)

# Add executables
# add_executable(<name> <sources>)

# Install libraries
# install (DIRECTORY lib/ DESTINATION ${INSTALL_LIB_DIR})

# Install headers
# install (DIRECTORY include/ DESTINATION ${INSTALL_INCLUDE_DIR})

# Install executables
# install (DIRECTORY bin/ DESTINATION ${INSTALL_BIN_DIR})
