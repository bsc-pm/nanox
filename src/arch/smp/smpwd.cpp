#include "smpprocessor.hpp"
#include "schedule.hpp"
#include "debug.hpp"
#include <iostream>

using namespace nanos;

Architecture nanos::SMP("SMP");

void SMPWD::allocateStack ()
{
    stack = new intptr_t[stackSize];
}

void SMPWD::initStack ()
{
    if (!hasStack()) {
	allocateStack();
    }

    initStackDep((void *)getWorkFct(),(void *)Scheduler::exit);
}
