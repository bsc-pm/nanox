#include "schedule.hpp"
#include "processingelement.hpp"

using namespace nanos;

void Scheduler::submit (WD &wd)
{
  PE *pe = myPE;
  // TODO: increase ready count

  debug("submitting task " << wd.getId());
  WD *next = pe->getSchedulingGroup()->atCreation(pe,wd);
  if (next) {
      pe->switchTo(next);
  }
}

void Scheduler::exit (void)
{
  PE *pe = myPE;

  // TODO:
  // Cases:
  // The WD was running on its own stack, switch to a new one
  // The WD was running on a thread stack, exit to the loop

  WD *next = pe->getSchedulingGroup()->atExit(pe);
  if (!next) next = pe->getSchedulingGroup()->getIdle(pe);
  if (next) {
      pe->exitTo(next);
  }

  fatal("No more tasks to execute!");
}

void Scheduler::blockOnCondition (volatile int *var, int condition)
{
	PE *pe = myPE;
	
	if ( *var != condition ) {
	    while ( *var != condition ) {
	          // set every iteration to avoid some race-conditions
		  pe->getCurrentWD()->setIdle();
		  
		  WD *next = pe->getSchedulingGroup()->atBlock(pe);
		  if (!next) next = pe->getSchedulingGroup()->getIdle(pe);
		  if (next) pe->switchTo(next);
		  // TODO: implement sleeping

// 		  verbose("waiting for " << (void *)var << " to reach " << condition << " current=" << *var);
	    }
	}
	pe->getCurrentWD()->setIdle(false);
}

void Scheduler::idle ()
{
      PE *pe = myPE;

     verbose("PE " << myPE->getId() << " entering idle loop");
     pe->getCurrentWD()->setIdle();
     while ( pe->isRunning() ) {            
	    if ( pe->getSchedulingGroup() ) {
	      WD *next = pe->getSchedulingGroup()->atIdle(pe);
	      if (!next) next = pe->getSchedulingGroup()->getIdle(pe);
	      if (next) pe->switchTo(next);
	    }
      }
      pe->getCurrentWD()->setIdle(false);

      verbose("Working thread finishing");
}

void Scheduler::queue (WD &wd)
{
    PE *pe=myPE;

    if (wd.isIdle())
      pe->getSchedulingGroup()->queueIdle(pe,wd);
    else
      pe->getSchedulingGroup()->queue(pe,wd);
}

void SchedulingGroup::init (int groupSize)
{
    size = 0;
    group.reserve(groupSize);
}

void SchedulingGroup::addMember (PE &pe)
{
    SchedulingData *data = createMemberData(pe);

    data->setSchId(size);    
    pe.setSchedulingGroup(this,data);

    group[size++] = data;
}

void SchedulingGroup::removeMember (PE &pe)
{
//TODO
}

void SchedulingGroup::queueIdle (PE *pe,WD &wd)
{
    idleQueue.push(&wd);
}

WD * SchedulingGroup::getIdle (PE *pe)
{
    WD *result = 0;

    if ( idleQueue.try_pop(result) ) return result;
    return 0;
}