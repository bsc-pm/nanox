#include "schedule.hpp"
#include "processingelement.hpp"

using namespace nanos;

void Scheduler::submit (WD &wd)
{
  // TODO: increase ready count

  debug("submitting task " << wd.getId());
  WD *next = myPE->getSchedulingGroup()->atCreation(myPE,wd);
  if (next) {
      myPE->switchTo(next);
  }
}

void Scheduler::exit (void)
{
  // TODO:
  // Cases:
  // The WD was running on its own stack, switch to a new one
  // The WD was running on a thread stack, exit to the loop

  WD *next = myPE->getSchedulingGroup()->atExit(myPE);
  if (!next) next = myPE->getSchedulingGroup()->getIdle(myPE);
  if (next) {
      myPE->exitTo(next);
  }

  fatal("No more tasks to execute!");
}

/*
 * G++ optimizes TLS accesses by obtaining only once the address of the TLS variable
 * In this function this optimization does not work because the task can move from one thread to another
 * in different iterations and it will still be seeing the old TLS variable (creating havoc
 * and destruction and colorful runtime errors).
 * getMyPESafe ensures that the TLS variable is reloaded at least once per iteration while we still do some
 * reuse of the address (inside the iteration) so we're not forcing to go through TLS for each myPE access
 * It's important that the compiler doesn't inline it or the optimazer will cause the same wrong behavior anyway.
 */
__attribute__((noinline)) PE * getMyPESafe() { return myPE; }
void Scheduler::blockOnCondition (volatile int *var, int condition)
{
	while ( *var != condition ) {
	     // get current TLS value
	     PE *pe = getMyPESafe();
	     // set every iteration to avoid some race-conditions
             pe->getCurrentWD()->setIdle();
		  
	     WD *next = pe->getSchedulingGroup()->atBlock(pe);
             if (!next) next = pe->getSchedulingGroup()->getIdle(pe);
	     if (next) {
	       pe->switchTo(next);
	     }
             // TODO: implement sleeping
	}
	myPE->getCurrentWD()->setIdle(false);
}

void Scheduler::idle ()
{
      // This function is run always by the same PE so we don't need to use getMyPESafe
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
    if (wd.isIdle())
      myPE->getSchedulingGroup()->queueIdle(myPE,wd);
    else
      myPE->getSchedulingGroup()->queue(myPE,wd);
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
    idleQueue.push_back(&wd);
}

WD * SchedulingGroup::getIdle (PE *pe)
{
    return idleQueue.pop_front(pe);
}