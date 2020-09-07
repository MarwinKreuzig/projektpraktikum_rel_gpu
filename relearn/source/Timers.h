/*
 * File:   Timers.h
 * Author: rinke
 *
 * Created on Oct 21, 2015
 */

#ifndef TIMERS_H
#define TIMERS_H

#include <chrono>
#include <cassert>
#include <string>

using namespace std::chrono;

enum TimerRegion : int {
	INITIALIZATION,
	SIMULATION_LOOP,
	UPDATE_ELECTRICAL_ACTIVITY,
	BARRIER_1,
	PREPARE_SENDING_SPIKES,
	PREPARE_NUM_NEURON_IDS,
	BARRIER_2,
	ALL_TO_ALL,
	ALLOC_MEM_FOR_NEURON_IDS,
	BARRIER_3,
	EXCHANGE_NEURON_IDS,
	CALC_SYNAPTIC_INPUT,
	CALC_ACTIVITY,
	UPDATE_SYNAPTIC_ELEMENTS_DELTA,
	UPDATE_CONNECTIVITY,
	UPDATE_NUM_SYNAPTIC_ELEMENTS_AND_DELETE_SYNAPSES,
	UPDATE_LOCAL_TREES,
	EXCHANGE_BRANCH_NODES,
	INSERT_BRANCH_NODES_INTO_GLOBAL_TREE,
	UPDATE_GLOBAL_TREE,
	FIND_TARGET_NEURONS,
	EMPTY_REMOTE_NODES_CACHE,
	CREATE_SYNAPSES,
	NUM_TIMER_REGIONS
};

class Timers;

namespace GlobalTimers {
	extern Timers timers;
}

class Timers {

public:

	Timers(size_t num_timers) :
		num_timers(num_timers) {
		time_start = new high_resolution_clock::time_point[num_timers];
		time_stop = new high_resolution_clock::time_point[num_timers];
		time_elapsed = new duration<double>[num_timers];

		// Reset elapsed to zero
		for (size_t i = 0; i < num_timers; i++) {
			reset_elapsed(i);
		}
	}

	~Timers() {
		delete[] time_start;
		delete[] time_stop;
		delete[] time_elapsed;
	}

	size_t get_num_timers() { return num_timers; }

	inline void start(size_t timer_id) {
		assert(timer_id < num_timers);
		time_start[timer_id] = high_resolution_clock::now();
	}

	inline void stop(size_t timer_id) {
		assert(timer_id < num_timers);
		time_stop[timer_id] = high_resolution_clock::now();
	}

	inline void stop_and_add(size_t timer_id) {
		stop(timer_id);
		add_start_stop_diff_to_elapsed(timer_id);
	}

	inline void add_start_stop_diff_to_elapsed(size_t timer_id) {
		assert(timer_id < num_timers);
		time_elapsed[timer_id] += duration_cast<duration<double>>(time_stop[timer_id] - time_start[timer_id]);
	}

	void reset_elapsed(size_t timer_id) {
		assert(timer_id < num_timers);
		time_elapsed[timer_id] = duration<double>::zero();
	}

	// Return elapsed time in seconds
	double get_elapsed(size_t timer_id) {
		assert(timer_id < num_timers);
		return time_elapsed[timer_id].count();
	}

	/**
	 * Static function to get current time in string
	 */
	static std::string wall_clock_time() {
#ifdef __linux__ 
		time_t rawtime;
		struct tm* timeinfo;
		char* string;

		time(&rawtime);
		timeinfo = localtime(&rawtime);
		string = asctime(timeinfo);

		// Remove linebreak in string
		string[24] = '\0';

		return std::string(string);
#else
		time_t rawtime;
		struct tm timeinfo;
		char char_buff[30];

		time(&rawtime);
		localtime_s(&timeinfo, &rawtime);
		asctime_s(char_buff, &timeinfo);

		// Remove linebreak in string
		char_buff[24] = '\0';

		return std::string(char_buff);
#endif
	}

private:
	size_t num_timers;     // Number of timers
	high_resolution_clock::time_point* time_start;
	high_resolution_clock::time_point* time_stop;
	duration<double>* time_elapsed;
};

#endif /* TIMERS_H */
