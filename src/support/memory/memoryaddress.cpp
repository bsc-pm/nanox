
#include "memory/memoryaddress.hpp"

namespace nanos {
namespace utils {

std::ostream& operator<<(std::ostream& out, nanos::utils::Address const &entry)
{
	return out << std::hex << entry.value;
}

}
}

