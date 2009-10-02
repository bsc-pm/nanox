#include "schedule.hpp"
#include "wddeque.hpp"
#include "plugin.hpp"
#include "system.hpp"

using namespace nanos;


class TaskStealData : public SchedulingData
{

      friend class TaskStealPolicy; //in this way, the policy can access the readyQueue

   protected:
      //  int schId;
      WDDeque readyQueue;

   public:
      // constructor
      TaskStealData ( int id = 0 ) : SchedulingData ( id ) {}

      //TODO: copy & assigment costructor

      // destructor
      ~TaskStealData() {}
};

class TaskStealPolicy : public SchedulingGroup
{

   public:
      // constructor
      TaskStealPolicy() : SchedulingGroup ( "task-steal-sch" ) {}

      TaskStealPolicy ( int groupsize ) : SchedulingGroup ( "task-steal-sch", groupsize ) {}

      // TODO: copy and assigment operations
      // destructor
      virtual ~TaskStealPolicy() {}

      virtual WD *atCreation ( BaseThread *thread, WD &newWD );
      virtual WD *atIdle ( BaseThread *thread );
      virtual void queue ( BaseThread *thread, WD &wd );
      virtual SchedulingData * createMemberData ( BaseThread &thread );
};

void TaskStealPolicy::queue ( BaseThread *thread, WD &wd )
{
   TaskStealData *data = ( TaskStealData * ) thread->getSchedulingData();
   data->readyQueue.push_front ( &wd );
}

WD * TaskStealPolicy::atCreation ( BaseThread *thread, WD &newWD )
{
   //NEW: now it does not enqueue the created task, but it moves down to the generated son: DEPTH-FIRST
   return &newWD;
}

WD * TaskStealPolicy::atIdle ( BaseThread *thread )
{
   WorkDescriptor * wd;

   TaskStealData *data = ( TaskStealData * ) thread->getSchedulingData();

   if ( ( wd = data->readyQueue.pop_front ( thread ) ) != NULL ) {
      return wd;
   } else {
      //first try to steal parent task
      if ( ( wd = ( thread->getCurrentWD() )->getParent() ) != NULL ) {
         //removing it from the queue. Try to remove from one queue: if someone move it, I stop looking for it to avoid ping-pongs.
         if ( ( wd->isEnqueued() ) == true && ( ! ( wd )->isTied() || ( wd )->isTiedTo() == thread ) ) { //not in queue = in execution, in queue = not in execution
            if ( wd->getMyQueue()->removeWD ( wd ) == true ) { //found it!
               return wd;
            }
         }
      }

      //if parent task is NULL or someone has moved it while I was trying to steal it --> steal other tasks
      int newposition = ( ( data->getSchId() ) + 1 ) % getSize();

 

      //should be random: for now it checks neighbour queues in round robin
      while ( ( newposition != data->getSchId() ) && ( ( wd = ( ( ( TaskStealData * ) ( getMemberData ( newposition ) ) )->readyQueue.pop_back ( thread ) ) ) == NULL ) ) {
         newposition = ( newposition + 1 ) % getSize();
      }

      return wd;
   }
}

SchedulingData * TaskStealPolicy::createMemberData ( BaseThread &thread )
{
   return new TaskStealData();
}

// Factories
SchedulingGroup * createTaskStealPolicy ()
{
   return new TaskStealPolicy();
}

SchedulingGroup * createTaskStealPolicy ( int groupsize )
{
   return new TaskStealPolicy ( groupsize );
}

class CilkSchedPlugin : public Plugin
{
   public:
      CilkSchedPlugin() : Plugin("Cilk scheduling Plugin",1) {}
      virtual void init() {
           sys.setDefaultSGFactory(createTaskStealPolicy);
      }
};

CilkSchedPlugin NanosXPlugin;
