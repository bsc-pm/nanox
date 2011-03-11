#include "memtracker.hpp"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <map>
#include <limits>
#include <exception>
#include "debug.hpp"
#include "atomic.hpp"

using namespace nanos;

MemTracker *nanos::mem = NULL;

NanosMemTrackerHelper dummy;

