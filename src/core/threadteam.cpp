#include "threadteam.hpp"
#include "atomic.hpp"
#include "debug.hpp"

using namespace nanos;


bool ThreadTeam::singleGuard(int local)
{
	return compare_and_swap(&single, local-1, local);
}
