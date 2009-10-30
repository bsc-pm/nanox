#include "schedule.hpp"
#include "processingelement.hpp"
#include "basethread.hpp"
#include "system.hpp"

using namespace nanos;



void Scheduler::submit ( WD &wd )
{
   // TODO: increase ready count

   sys.taskNum++;

   debug ( "submitting task " << wd.getId() );
   WD *next = myThread->getSchedulingGroup()->atCreation ( myThread, wd );

   if ( next ) {
      myThread->switchTo ( next );
   }
}

void Scheduler::exit ( void )
{
   // TODO: Support WD running on lended stack
   // Cases:
   // The WD was running on its own stack, switch to a new one
   // The WD was running on a thread stack, exit to the loop

   sys.taskNum--;

   WD *next = myThread->getSchedulingGroup()->atExit ( myThread );

   if ( next ) {
      sys.numReady--;
   }

   if ( !next ) {
      next = myThread->getSchedulingGroup()->getIdle ( myThread );
   }

   if ( next ) {
      myThread->exitTo ( next );
   }

   fatal ( "No more tasks to execute!" );
}

/*
 * G++ optimizes TLS accesses by obtaining only once the address of the TLS variable
 * In this function this optimization does not work because the task can move from one thread to another
 * in different iterations and it will still be seeing the old TLS variable (creating havoc
 * and destruction and colorful runtime errors).
 * getMyThreadSafe ensures that the TLS variable is reloaded at least once per iteration while we still do some
 * reuse of the address (inside the iteration) so we're not forcing to go through TLS for each myThread access
 * It's important that the compiler doesn't inline it or the optimazer will cause the same wrong behavior anyway.
 */
__attribute__ ( ( noinline ) ) BaseThread * getMyThreadSafe()
{
   return myThread;
}

template<typename T>
void Scheduler::blockOnCondition ( volatile T *var, T condition )
{
   while ( *var != condition ) {
      //get current TLS value
      BaseThread *thread = getMyThreadSafe();
      // set every iteration to avoid some race-conditions
      thread->getCurrentWD()->setIdle();

      WD *next = thread->getSchedulingGroup()->atBlock ( thread );

      if ( next ) {
         sys.numReady--;
      }

      if ( !next )
         next = thread->getSchedulingGroup()->getIdle ( thread );

      if ( next ) {
         thread->switchTo ( next );
      }
   }

   myThread->getCurrentWD()->setIdle ( false );
}

template<typename T>
void Scheduler::blockOnConditionLess ( volatile T *var, T condition )
{
   while ( *var < condition ) {
      //get current TLS value
      BaseThread *thread = getMyThreadSafe();
      // set every iteration to avoid some race-conditions
      thread->getCurrentWD()->setIdle();

      WD *next = thread->getSchedulingGroup()->atBlock ( thread );

      if ( next ) {
         sys.numReady--;
      }

      if ( !next )
         next = thread->getSchedulingGroup()->getIdle ( thread );

      if ( next ) {
         thread->switchTo ( next );
      }

      // TODO: implement sleeping
   }

   myThread->getCurrentWD()->setIdle ( false );
}

template
void Scheduler::blockOnCondition<int>( volatile int *var, int condition );
template
void Scheduler::blockOnCondition<bool>( volatile bool *var, bool condition );
template
void Scheduler::blockOnConditionLess<int>( volatile int *var, int condition );
template
void Scheduler::blockOnConditionLess<bool>( volatile bool *var, bool condition );


void Scheduler::idle ()
{

   // This function is run always by the same BaseThread so we don't need to use getMyThreadSafe
   BaseThread *thread = myThread;

   thread->getCurrentWD()->setIdle();

   sys.idleThreads++;

   while ( thread->isRunning() ) {
      if ( thread->getSchedulingGroup() ) {
         WD *next = thread->getSchedulingGroup()->atIdle ( thread );

         if ( next ) {
            sys.numReady--;
         }

         if ( !next )
            next = thread->getSchedulingGroup()->getIdle ( thread );

         if ( next ) {
            sys.idleThreads--;
            sys.numTasksRunning++;
            thread->switchTo ( next );
            sys.numTasksRunning--;
            sys.idleThreads++;
         }
      }
   }

   thread->getCurrentWD()->setIdle ( false );

   sys.idleThreads--;

   verbose ( "Working thread finishing" );
}

void Scheduler::queue ( WD &wd )
{
   if ( wd.isIdle() )
      myThread->getSchedulingGroup()->queueIdle ( myThread, wd );
   else {
      myThread->getSchedulingGroup()->queue ( myThread, wd );
      sys.numReady++;
   }
}

void SchedulingGroup::init ( int groupSize )
{
   group.reserve ( groupSize );
}

void SchedulingGroup::addMember ( BaseThread &thread )
{
   SchedulingData *data = createMemberData ( thread );

   data->setSchId ( getSize() );
   thread.setScheduling ( this, data );
   group.push_back( data );
}

void SchedulingGroup::removeMember ( BaseThread &thread )
{
//TODO
}

void SchedulingGroup::queueIdle ( BaseThread *thread, WD &wd )
{
   idleQueue.push_back ( &wd );
}

WD * SchedulingGroup::getIdle ( BaseThread *thread )
{
   return idleQueue.pop_front ( thread );
}

