#ifndef TIME_H
#define TIME_H

#include <ctime>

char *wall_clock_time()
{
    time_t rawtime;
    struct tm* timeinfo;
    char *string;

    time (&rawtime);
    timeinfo = localtime (&rawtime);
    string = asctime(timeinfo);

    // Remove linebreak in string
    string[24] = '\0';

    return string;
}

#endif /* TIME_H */
