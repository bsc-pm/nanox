#include "smpprocessor.hpp"
#include "schedule.hpp"
#include "debug.hpp"
#include <iostream>

using namespace nanos;

Device nanos::SMP("SMP");

void SMPDD::allocateStack ()
{
    stack = new intptr_t[stackSize];
}

void SMPDD::initStack (void *data)
{
    if (!hasStack()) {
	allocateStack();
    }

    initStackDep((void *)getWorkFct(),data,(void *)Scheduler::exit);
}
