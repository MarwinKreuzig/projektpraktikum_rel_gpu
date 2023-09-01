#pragma once

#include "Commons.cuh"
#include "Random.cuh"


namespace gpu::background {

class BackgroundActivity {
    public:

    //virtual void init(const size_t number_neurons) {}

    //virtual void create_neurons_gpu(const number_neurons_type creation_count) {}

    //virtual void update_input_gpu([[maybe_unused]] const size_t step) {}

     __device__ virtual double get(size_t step, size_t neuron_id) const =0;

};

class Constant : public BackgroundActivity {

    public:
    __device__ Constant(double c) : constant(c) {

    }

    __device__  double get(size_t step, size_t neuron_id) const override {
        return constant;
    }

    private:
    double constant;
};

class Normal : public BackgroundActivity {

    public:
    __device__ Normal(double _mean, double _stddev) : mean(_mean), stddev(_stddev) {

    }

    __device__ inline double get(size_t step, size_t neuron_id) const override {
        //return 7.0;
        auto curand_state = gpu::RandomHolder::init(step, gpu::RandomHolder::BACKGROUND, neuron_id);
        const auto random_value = gpu::RandomHolder::get_normal(&curand_state, mean, stddev);
        return random_value;
    }


    private:
    double mean;
    double stddev;
};

}