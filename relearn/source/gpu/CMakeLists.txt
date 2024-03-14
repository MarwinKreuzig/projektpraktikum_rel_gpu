if (CUDA_FOUND)
    add_library(relearn_gpu STATIC
            neurons/input/BackgroundActivity.cuh
            neurons/models/NeuronModel.cuh
            neurons/models/PoissonModel.cuh
            neurons/models/SynapticElements.cuh
            structure/CudaArray.cuh
            structure/CudaVector.cuh
            neurons/NeuronsExtraInfos.cuh
            utils/Random.cuh
            utils/Random.cuh
            RelearnGPUException.h
            Commons.cuh
            structure/Octree.cuh

            structure/OctreeCPUCopy.h
            
            algorithm/BarnesHutKernel.h
            algorithm/BarnesHutData.cuh
            algorithm/kernel/KernelGPU.cuh
            algorithm/kernel/GammaGPU.cuh
            algorithm/kernel/GaussianGPU.cuh
            algorithm/kernel/LinearGPU.cuh
            algorithm/kernel/WeibullGPU.cuh)

    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    target_include_directories(relearn_gpu PRIVATE ../shared)

    target_compile_features(relearn_gpu PRIVATE cxx_std_17 cuda_std_17)
    
    # Watch the -G for benchmarks and stuff cause it makes things slower
    target_compile_options(relearn_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
            --expt-relaxed-constexpr --expt-extended-lambda -rdc=true
            >)

    target_sources(
            relearn_gpu
            PRIVATE
            neurons/models/NeuronModel.cu
            neurons/NeuronsExtraInfos.cu
            structure/Octree.cu
            algorithm/BarnesHutKernel.cu
            neurons/models/SynapticElements.cu
            utils/RandomNew.cu
            algorithm/kernel/KernelGPU.cu
            algorithm/kernel/GammaGPU.cu
            algorithm/kernel/GaussianGPU.cu
            algorithm/kernel/LinearGPU.cu
            algorithm/kernel/WeibullGPU.cu
            algorithm/BarnesHutData.cu
            neurons/input/BackgroundActivity.cu
            neurons/models/AEIFModel.cu
            neurons/models/FitzHughNagumoModel.cu
            neurons/models/IzhikevichModel.cu
            neurons/models/PoissonModel.cu
            utils/Random.cu
            RelearnGPUException.cu
            
            # for visual studio
            ${relearn_gpu_additional_files})

    #set_target_properties(relearn_gpu PROPERTIES CUDA_ARCHITECTURES 61)

    set_target_properties(relearn_gpu PROPERTIES CMAKE_CUDA_STANDARD 17 CMAKE_CXX_STANDARD 17 CMAKE_CUDA_STANDARD_REQUIRED ON CMAKE_CXX_STANDARD_REQUIRED ON)

    target_include_directories(relearn_gpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

    target_link_libraries(relearn_gpu PUBLIC project_libraries_gpu)
endif ()
