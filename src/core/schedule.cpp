#include "schedule.hpp"
#include "processingelement.hpp"

using namespace nanos;

void Scheduler::submit (WD &wd)
{
  PE *pe = myPE;
  // TODO: increase ready count

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
  if (next) {
      pe->exitTo(next);
  }

  //TODO: fatal error
}

void Scheduler::blockOnCondition (volatile int *var, int condition)
{
	PE *pe = myPE;
	
	if ( *var != condition ) {
	    while ( *var != condition ) {
		  WD *next = pe->getSchedulingGroup()->atBlock(pe);
		  if (next) pe->switchTo(next);
		  // TODO: implement sleeping

		  verbose("waiting for " << (void *)var << " to reach " << condition << " current=" << *var);
	    }
	}
}

void Scheduler::idle ()
{
      PE *pe = myPE;

     while ( pe->isRunning() ) {
	    if ( pe->getSchedulingGroup() ) {
	      WD *next = pe->getSchedulingGroup()->atIdle(pe);
	      if (next) pe->switchTo(next);
	    }
      }

      verbose("Working thread finishing");
}

void Scheduler::queue (WD &wd)
{
    PE *pe=myPE;
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

