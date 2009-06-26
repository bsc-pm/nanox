#include "processingelement.hpp"
#include "debug.hpp"
#include "schedule.hpp"

using namespace nanos;

void ProcessingElement::startWorker (SchedulingGroup *sg)
{
	WD & master = getWorkerWD();
	startThread(master,sg);
}

BaseThread & ProcessingElement::startThread (WD &work, SchedulingGroup *sg)
{
       BaseThread &thread = createThread(work);
       if (sg) sg->addMember(thread);
       thread.start();

       workerThread = &thread; // TODO: should be a vector
       return thread; 
}

BaseThread & ProcessingElement::associateThisThread (SchedulingGroup *sg)
{
	WD & master = getMasterWD();
    BaseThread &thread = createThread(master);
    if (sg) sg->addMember(thread);
    
    thread.associate();
	
	return thread;
}

void ProcessingElement::stopAll ()
{
       workerThread->stop();
       workerThread->join();
}
