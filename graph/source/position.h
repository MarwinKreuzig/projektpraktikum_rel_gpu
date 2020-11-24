#ifndef POSITION_H_
#define POSITION_H_

// Position of vertex
struct Position {
	double x, y, z;

	Position() { }
	Position(double x, double y, double z)
	: x(x), y(y), z(z) { }

	struct less {
		bool operator() (const Position& lhs, const Position& rhs) const {
			return  lhs.x < rhs.x ||
					(lhs.x == rhs.x && lhs.y < rhs.y) ||
					(lhs.x == rhs.x && lhs.y == rhs.y && lhs.z < rhs.z);
		}
	};

	void MinForEachCoordinate(const Position& pos) {
		x = std::min(x, pos.x);
		y = std::min(y, pos.y);
		z = std::min(z, pos.z);
	}

	void Add(const Position& pos) {
		x += pos.x;
		y += pos.y;
		z += pos.z;
	}
};

#endif /* POSITION_H_ */
