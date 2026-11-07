cmake_minimum_required(VERSION 3.24)
project(llama_inference_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- library: every .cpp under src/ ---
file(GLOB_RECURSE ENGINE_SOURCES CONFIGURE_DEPENDS src/*.cpp)
add_library(llm_engine ${ENGINE_SOURCES})
target_include_directories(llm_engine PUBLIC include)

# --- GoogleTest ---
include(FetchContent)
FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz)
FetchContent_MakeAvailable(googletest)

enable_testing()
file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS tests/*.cpp)
add_executable(engine_tests ${TEST_SOURCES})
target_link_libraries(engine_tests PRIVATE llm_engine GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(engine_tests)
