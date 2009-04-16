#include "processingelement.hpp"
#include "debug.hpp"

using namespace nanos;

__thread ProcessingElement *nanos::myPE=0;

void BaseThread::run ()
{
    if (pe) pe->associate();
    
    run_dependent();
}


void ProcessingElement::startWorker ()
{
	WD & master = getWorkerWD();
	startThread(master);
}

void ProcessingElement::submit (WD &work)
{
        readyQueue.push(&work);
	SimpleMessage("work added");
}

void ProcessingElement::associate ()
{
        myPE = this;
}

