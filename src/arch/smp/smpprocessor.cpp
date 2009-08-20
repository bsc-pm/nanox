#include "smpprocessor.hpp"
#include "schedule.hpp"
#include "debug.hpp"
#include <iostream>

namespace nanos {

bool SMPProcessor::useUserThreads = true;

void SMPProcessor::prepareConfig (Config &config)
{
    // TODO: CHECK, move to plugin and to smpthread
	config.registerArgOption(new Config::FlagOption("nth-no-ut",useUserThreads,false));
}

WorkDescriptor & SMPProcessor::getWorkerWD () const
{
	SMPDD * dd = new SMPDD((SMPDD::work_fct)Scheduler::idle);
    WD *wd = new WD(dd);
	return *wd;
}

WorkDescriptor & SMPProcessor::getMasterWD () const
{
    WD * wd = new WD(new SMPDD());
    return *wd;
}

BaseThread &SMPProcessor::createThread (WorkDescriptor &helper)
{
    ensure(helper.canRunIn(SMP),"Incompatible worker thread");
	SMPThread &th = *new SMPThread(helper,this);

	return th;
}


};
