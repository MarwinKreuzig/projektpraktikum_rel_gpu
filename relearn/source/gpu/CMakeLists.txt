if (CUDA_FOUND)
    add_library(relearn_gpu STATIC
            neurons/input/BackgroundActivity.cuh
            neurons/input/BackgroundActivity.cuh
            neurons/models/AEIFModel.cuh
            neurons/models/FitzHughNagumoModel.cuh
            neurons/models/IzhikevichModel.cuh
            neurons/models/NeuronModel.cuh
            neurons/models/PoissonModel.cuh
            structure/CudaArray.cuh
            structure/CudaVector.cuh
            neurons/NeuronsExtraInfos.cuh
            utils/Random.cuh
            RelearnGPUException.h
            Commons.cuh
            structure/Octree.cuh
            structure/OctreeCPUCopy.h)

    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    target_include_directories(relearn_gpu PRIVATE ../shared)

    target_compile_features(relearn_gpu PRIVATE cxx_std_17 cuda_std_17)

    target_compile_options(relearn_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
            --expt-relaxed-constexpr --expt-extended-lambda -rdc=true
            >)

    target_sources(
            relearn_gpu
            PRIVATE
            neurons/models/NeuronModels.cu
            structure/Octree.cu
            # for visual studio
            ${relearn_gpu_additional_files})

    #set_target_properties(relearn_gpu PROPERTIES CUDA_ARCHITECTURES 61)

    set_target_properties(relearn_gpu PROPERTIES CMAKE_CUDA_STANDARD 17 CMAKE_CXX_STANDARD 17 CMAKE_CUDA_STANDARD_REQUIRED ON CMAKE_CXX_STANDARD_REQUIRED ON)

    target_include_directories(relearn_gpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

    target_link_libraries(relearn_gpu PUBLIC project_libraries_gpu)


endif ()
