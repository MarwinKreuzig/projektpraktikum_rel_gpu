/*
 * File:   LogMessages.h
 * Author: rinke
 *
 * Created on Apr 13, 2016
 */

#ifndef LOGMESSAGES_H
#define LOGMESSAGES_H

class LogMessages {
public:

	/**
	 * Static functions for printing a tagged log message to std::cout
	 */
	static void print_message(char const* string);

	// Print tagged message only at MPI rank "rank"
	static void print_message_rank(char const* string, int rank);

	static void print_error(char const* string);

	static void print_debug(char const* string);
};

#endif /* LOGMESSAGES_H */
