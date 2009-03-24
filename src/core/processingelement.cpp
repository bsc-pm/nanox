#include "processingelement.hpp"
#include "debug.hpp"

using namespace nanos;

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

