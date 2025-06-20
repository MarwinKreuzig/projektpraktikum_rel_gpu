function(enable_coverage project_name)
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES
            ".*Clang")
        target_compile_options(${project_name} INTERFACE --coverage -O0 -g)
        target_link_libraries(${project_name} INTERFACE --coverage)
    endif ()
endfunction()

function(enable_sanitizers project_name)

    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES
            ".*Clang")
        option(ENABLE_COVERAGE "Enable coverage reporting for gcc/clang" OFF)

        if (ENABLE_COVERAGE)
            enable_coverage(${project_name})
        endif ()

        set(SANITIZERS "")

        option(ENABLE_SANITIZER_ADDRESS "Enable address sanitizer" OFF)
        if (ENABLE_SANITIZER_ADDRESS)
            list(APPEND SANITIZERS "address")
        endif ()

        option(ENABLE_SANITIZER_LEAK "Enable leak sanitizer" OFF)
        if (ENABLE_SANITIZER_LEAK)
            list(APPEND SANITIZERS "leak")
        endif ()

        option(ENABLE_SANITIZER_UNDEFINED_BEHAVIOR
                "Enable undefined behavior sanitizer" OFF)
        if (ENABLE_SANITIZER_UNDEFINED_BEHAVIOR)
            list(APPEND SANITIZERS "undefined")
        endif ()

        option(ENABLE_SANITIZER_THREAD "Enable thread sanitizer" OFF)
        if (ENABLE_SANITIZER_THREAD)
            if ("address" IN_LIST SANITIZERS OR "leak" IN_LIST SANITIZERS)
                message(
                        WARNING
                        "Thread sanitizer does not work with Address and Leak sanitizer enabled"
                )
            else ()
                list(APPEND SANITIZERS "thread")
            endif ()
        endif ()

        option(ENABLE_SANITIZER_MEMORY "Enable memory sanitizer" OFF)
        if (ENABLE_SANITIZER_MEMORY AND CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
            message(
                    WARNING
                    "Memory sanitizer requires all the code (including libc++) to be MSan-instrumented otherwise it reports false positives"
            )
            if ("address" IN_LIST SANITIZERS
                    OR "thread" IN_LIST SANITIZERS
                    OR "leak" IN_LIST SANITIZERS)
                message(
                        WARNING
                        "Memory sanitizer does not work with Address, Thread and Leak sanitizer enabled"
                )
            else ()
                list(APPEND SANITIZERS "memory")
            endif ()
        endif ()

        list(
                JOIN
                SANITIZERS
                ","
                LIST_OF_SANITIZERS)

    endif ()

    if (LIST_OF_SANITIZERS)
        if (NOT
                "${LIST_OF_SANITIZERS}"
                STREQUAL
                "")
            target_compile_options(${project_name}
                    INTERFACE -fsanitize=${LIST_OF_SANITIZERS})
            target_link_options(${project_name} INTERFACE
                    -fsanitize=${LIST_OF_SANITIZERS})
        endif ()
    endif ()

endfunction()
