/*
   This schedules is just provided for debugging purposes:
   it is exactly equal to the bf scheduler, except that it performs 
   a pratically void function on the wddeque of all other
   threads whenever the atIdle method is invoked.
*/

#include "schedule.hpp"
#include "wddeque.hpp"
#include "plugin.hpp"
#include "system.hpp"
#include "config.hpp"

using namespace nanos;



class DebugData : public SchedulingData
{
   friend class DebugPolicy; //in this way, the policy can access the readyQueue

protected:
   WDDeque readyQueue;

public:
   // constructor
   DebugData(int id=0) : SchedulingData(id) {}
   //TODO: copy & assigment costructor

   // destructor
   ~DebugData() {}
};


class DebugPolicy : public SchedulingGroup {
private:
     WDDeque   readyQueue;

public:
     // constructor
     DebugPolicy(bool stack) : SchedulingGroup("debug-sch") {}
     // TODO: copy and assigment operations
     // destructor
     virtual ~DebugPolicy() {}

     virtual WD *atCreation (BaseThread *thread, WD &newWD);
     virtual WD *atIdle (BaseThread *thread);
     virtual void queue (BaseThread *thread, WD &wd);
};

void DebugPolicy::queue (BaseThread *thread, WD &wd)
{
    readyQueue.push_back(&wd);
}

WD * DebugPolicy::atCreation (BaseThread *thread, WD &newWD)
{
    queue(thread,newWD);
    return 0;
}

WD * DebugPolicy::atIdle (BaseThread *thread)
{
   //first invokes the doNothing methon on all other thread queues

   int newposition = ((data->getSchId()) +1) % getSize();

   //should be random: for now it checks neighbour queues in round robin
   while( (newposition != data->getSchId()) && (
            ((stealPolicy == LIFO) && (((wd = (((WFData *) (getMemberData(newposition)))->readyQueue.pop_back(thread))) == NULL))) ||
            ((stealPolicy == FIFO) && (((wd = (((WFData *) (getMemberData(newposition)))->readyQueue.pop_front(thread))) == NULL)))
            ) ) {
      newposition = (newposition +1) % getSize();
   }

   return readyQueue.pop_back(thread);
}

static bool useStack = false;

// Factory
SchedulingGroup * createDebugPolicy ()
{
    return new DebugPolicy(useStack);
}

class DebugSchedPlugin : public Plugin
{
   public:
      DebugSchedPlugin() : Plugin("BF scheduling Plugin",1) {}
      virtual void init() {
         Config config;

         sys.setDefaultSGFactory(createDebugPolicy);
      }
};

DebugSchedPlugin NanosXPlugin;

