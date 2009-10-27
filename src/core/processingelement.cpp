#include "processingelement.hpp"
#include "debug.hpp"
#include "schedule.hpp"

using namespace nanos;

BaseThread& ProcessingElement::startWorker (SchedulingGroup *sg)
{
	WD & master = getWorkerWD();
	return startThread(master,sg);
}

BaseThread & ProcessingElement::startThread (WD &work, SchedulingGroup *sg)
{
       BaseThread &thread = createThread(work);
       if (sg) sg->addMember(thread);
       thread.start();

      threads.push_back(&thread);

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
        std::vector<BaseThread *>::iterator it;
       for(it = threads.begin(); it != threads.end(); it++) {
            (*it)->stop();
            (*it)->join();
        }
}
