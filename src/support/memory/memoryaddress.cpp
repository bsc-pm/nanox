
#include "memory/memoryaddress.hpp"

std::ostream& operator<<(std::ostream& out, Address const &entry)
{
	return out << std::hex << entry.value;
}

