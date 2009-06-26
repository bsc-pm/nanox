#include "smpprocessor.hpp"
#include "schedule.hpp"
#include "debug.hpp"
#include <iostream>

namespace nanos {

bool SMPProcessor::useUserThreads = true;

void SMPProcessor::prepareConfig (Config &config)
{
    // CHECK
	config.registerArgOption(new Config::FlagOption("nth-no-ut",useUserThreads,false));
}

WorkDescriptor & SMPProcessor::getWorkerWD () const
{
	SMPWD * wd = new SMPWD((SMPWD::work_fct)Scheduler::idle,0);
	return *wd;
}

WorkDescriptor & SMPProcessor::getMasterWD () const
{
    SMPWD * wd = new SMPWD();
    return *wd;
}

BaseThread &SMPProcessor::createThread (WorkDescriptor &helper)
{
	SMPWD &wd = static_cast<SMPWD &>(helper);
	SMPThread &th = *new SMPThread(wd,this);

	return th;
}


};
