#include "smpprocessor.hpp"
#include "schedule.hpp"
#include "debug.hpp"
#include "system.hpp"
#include <iostream>

extern "C" {
// low-level routine to switch stacks
void switchStacks(void *,void *,void *,void *);
}

using namespace nanos;

void * smp_bootthread (void *arg)
{
    SMPThread *self = static_cast<SMPThread *>(arg);
    
#if 0

TODO:
		cpu_set_t cpu_set;
		pid_t me = my_gettid();

		sched_getaffinity(
			(pid_t) me,
			sizeof(cpu_set),
			&cpu_set
		);
		int i;
		for (i = 0; i < CPU_SETSIZE; i++)
		{
			CPU_SET(i, &cpu_set);
		}
		sched_setaffinity(
			(pid_t) me,
			sizeof(cpu_set),
			&cpu_set);
	}

	NTH_MYSELF = (nth_desc_t*) me;
	NTH_KTH = NTH_MYSELF->vp;

	NTH_MYSELF->executed = 1;
	nth_instrument_thread_start();

	nth_data.kth_myself[NTH_KTH] = me;

	/* I'm running.. */
	nth_atm_add(&NTH_MYSELF->ndep, 1);

	/* but not for long :) */
	nth_block();


	nth_instrument_thread_end();

	pthread_exit(0);

#endif

   self->run();

    pthread_exit (0);
}

void SMPThread::start ()
{
// TODO:
//        /* initialize thread_attr: init. attr */
//        pthread_attr_init(&nth_data.thread_attr);
//        /* initialize thread_attr: stack attr */
//        rv_pthread = pthread_attr_setstack(
//                         (pthread_attr_t *) &nth_data.thread_attr,
//                         (void *) aux_desc->stack_addr,
//                         (size_t) aux_desc->stack_size
//                 );


      if (pthread_create( &pth, NULL, smp_bootthread, this ) )
        fatal("couldn't create thread");
}

void SMPThread::run_dependent ()
{
    WD &work = getThreadWD();
    setCurrentWD(work);

    SMPDD &dd = (SMPDD &) work.activateDevice(SMP);
    dd.getWorkFct()(work.getData());
}

void SMPThread::join ()
{
    pthread_join(pth,NULL);
}

// This is executed in between switching stacks
static void switchHelper ( WD *oldWD, WD *newWD, intptr_t *oldState  )
{
    SMPDD & dd = (SMPDD &)oldWD->getActiveDevice();
    dd.setState(oldState);
    Scheduler::queue(*oldWD);
    myThread->setCurrentWD(*newWD);
}

void SMPThread::switchTo ( WD *wd )
{
    // wd MUST have an active Device when it gets here
    ensure(wd->hasActiveDevice(),"WD has no active SMP device");
    SMPDD &dd = (SMPDD &)wd->getActiveDevice();

   if ( useUserThreads ) {
       debug("switching from task " << getCurrentWD() << ":" << getCurrentWD()->getId() << " to " << wd << ":" << wd->getId());

      if (!dd.hasStack()) {
	    dd.initStack(wd->getData());
      }

      ::switchStacks(
 		   (void *) getCurrentWD(),
 		   (void *) wd,
           (void *) dd.getState(),
           (void *) switchHelper);
   } else {
      WD *oldwd = getCurrentWD();
      setCurrentWD(*wd);
      (dd.getWorkFct())(wd->getData());
      // TODO: not delete work descriptor if is a parent with pending children
      setCurrentWD(*oldwd);
      delete wd;
   }
}

static void exitHelper (  WD *oldWD, WD *newWD, intptr_t *oldState )
{
    delete oldWD;
    myThread->setCurrentWD(*newWD);
}

void SMPThread::exitTo (WD *wd)
{
    // wd MUST have an active Device when it gets here
    ensure(wd->hasActiveDevice(),"WD has no active SMP device");
    SMPDD &dd = (SMPDD &)wd->getActiveDevice();


    debug("exiting task " << getCurrentWD() << ":" << getCurrentWD()->getId() << " to " << wd << ":" << wd->getId());
    // TODO: reuse stack

    if (!dd.hasStack()) {
	  dd.initStack(wd->getData());
    }

    //TODO: optimize... we don't really need to save a context in this case
    ::switchStacks(
 		   (void *) getCurrentWD(),
 		   (void *) wd,
           (void *) dd.getState(),
           (void *) exitHelper);
}

