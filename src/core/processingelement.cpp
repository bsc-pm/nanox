#include "processingelement.hpp"
#include "debug.hpp"
#include "schedule.hpp"

using namespace nanos;

__thread ProcessingElement *nanos::myPE=0;

void BaseThread::run ()
{
    if (pe) pe->associate();
    
    run_dependent();
}

ProcessingElement::ProcessingElement (int newId,const Architecture *arch,SchedulingGroup *sg) 
 : id(newId),architecture(arch),currentWD(0)
{
      if (sg) sg->addMember(*this);
}

void ProcessingElement::startWorker ()
{
	WD & master = getWorkerWD();
	master.tieTo(*this);
	startThread(master);
}

void ProcessingElement::associate ()
{
        myPE = this;
}

