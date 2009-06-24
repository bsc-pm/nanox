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

BaseThread &SMPProcessor::startThread (WorkDescriptor &helper, SchedulingGroup *sg)
{
	SMPWD &wd = static_cast<SMPWD &>(helper);
	SMPThread &th = *new SMPThread(wd,this);
	// TODO: generalize
	workerThread = &th;

    if (sg) sg->addMember(th);

	th.start();

	return th;
}

// TODO: move part to base PE
BaseThread &SMPProcessor::associateThisThread (SchedulingGroup *sg)
{
      SMPWD *wd = new SMPWD();
      //CHECK wd->tieTo(*this);
      SMPThread &th = *new SMPThread(*wd,this);
      associate();
      //setCurrentWD(wd);
      myThread = &th;
      th.setCurrentWD(wd);
      if (sg) sg->addMember(th);
      
      return th;
}


};
