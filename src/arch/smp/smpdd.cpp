#include "smpprocessor.hpp"
#include "schedule.hpp"
#include "debug.hpp"
#include "system.hpp"
#include <iostream>

using namespace nanos;

Device nanos::SMP("SMP");

int SMPDD::stackSize = 1024;

void SMPDD::prepareConfig(Config &config)
{
    // Get the stack size from system config
    stackSize = sys.getDeviceStackSize();

    // Get the stack size for this device
    config.registerArgOption(new Config::PositiveVar("nth-smp-stack-size",stackSize));
    config.registerEnvOption(new Config::PositiveVar("nth-SMP_STACK_SIZE",stackSize));
}

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
