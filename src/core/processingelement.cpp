#include "processingelement.hpp"
#include "debug.hpp"
#include "schedule.hpp"

using namespace nanos;

void ProcessingElement::startWorker (SchedulingGroup *sg)
{
	WD & master = getWorkerWD();
	//CHECK: master.tieTo(*this);
	startThread(master,sg);
}

void ProcessingElement::associate ()
{
 //CHECK       myPE = this;
}

void ProcessingElement::stopAll ()
{
       workerThread->stop();
       workerThread->join();
}
