#include "smpprocessor.hpp"
#include "schedule.hpp"
#include "debug.hpp"
#include "system.hpp"
#include <iostream>

using namespace nanos;

Device nanos::SMP("SMP");

int SMPDD::stackSize = 1024;

/*! \fn prepareConfig(Config &config)
  \brief Registers the Device's configuration options
  \param reference to a configuration object.
  \sa Config System
*/
void SMPDD::prepareConfig(Config &config)
{
    /*!
       Get the stack size from system configuration
     */
    stackSize = sys.getDeviceStackSize();

    /*!
       Get the stack size for this device
    */
    config.registerArgOption(new Config::PositiveVar("nth-smp-stack-size",stackSize));
    config.registerEnvOption(new Config::PositiveVar("NTH_SMP_STACK_SIZE",stackSize));
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
