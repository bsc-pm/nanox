#include "workdescriptor.hpp"
#include "processingelement.hpp"
#include <stdarg.h>
#include <stdio.h>
#include <stdexcept>
#include <string.h>

using namespace nanos;

bool SimpleWD::canRunIn(PE &pe)
{
	return pe.getArchitecture() == architecture;
}

