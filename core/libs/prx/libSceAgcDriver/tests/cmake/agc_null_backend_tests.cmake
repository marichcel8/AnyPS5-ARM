if(BUILD_TESTING AND ANYPS5_GPU_BACKEND STREQUAL "Metal")
    add_test_executable(agc_null_backend_tests tests/NullBackend.cpp)
    target_include_directories(agc_null_backend_tests PRIVATE
            ${LIBS_INCLUDE_DIR}
            ${CMAKE_SOURCE_DIR}/core/shader/recompiler
            ${ANYPS5_VULKAN_HEADERS_INCLUDE_DIR}
    )
    target_link_libraries(agc_null_backend_tests PRIVATE libSceAgcDriver libkernel libc)
    set_target_properties(agc_null_backend_tests PROPERTIES CXX_EXTENSIONS OFF)
    add_test(NAME agc_null_backend COMMAND agc_null_backend_tests)
    set_tests_properties(agc_null_backend PROPERTIES TIMEOUT 20)
endif()

