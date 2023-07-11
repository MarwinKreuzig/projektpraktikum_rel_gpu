	
	add_library(relearn_gpu STATIC)

	set(CMAKE_CUDA_STANDARD 17)
	set(CMAKE_CUDA_STANDARD_REQUIRED ON)

	target_compile_features(relearn_gpu PRIVATE cxx_std_17 cuda_std_17)


	enable_language(CUDA)

	target_sources(
        relearn_gpu
        PRIVATE
          gpu/models/PoissonModel.cu)

		  set_target_properties(relearn_gpu PROPERTIES CUDA_ARCHITECTURES 61)

		  set_target_properties(relearn_gpu PROPERTIES CMAKE_CUDA_STANDARD 17 CMAKE_CXX_STANDARD 17 CMAKE_CUDA_STANDARD_REQUIRED ON CMAKE_CXX_STANDARD_REQUIRED ON)

		  target_include_directories(relearn_gpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

		  target_link_libraries(relearn_gpu PUBLIC project_libraries_gpu)

		  #set_target_properties(relearn_gpu PROPERTIES LINKER_LANGUAGE CUDA)

	
