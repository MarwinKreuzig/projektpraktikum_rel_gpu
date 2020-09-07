#ifndef UTILITY_H
#define UTILITY_H

namespace Util {
	inline int num_digits(int val) {
		int num_digits = 0;

		do {
			++num_digits;
			val /= 10;
		} while (val);

		return num_digits;
	}
}

#endif /* UTILITY_H */
