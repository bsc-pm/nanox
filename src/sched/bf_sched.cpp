#include "schedule.hpp"
#include "wddeque.hpp"

using namespace nanos;

class BreadthFirstPolicy : public SchedulingGroup {
private:
     WDDeque   readyQueue;
public:
     // constructor
     BreadthFirstPolicy() : SchedulingGroup("breadth-first-sch") {}
     // TODO: copy and assigment operations
     // destructor
     virtual ~BreadthFirstPolicy() {}

     virtual WD *atCreation (BaseThread *thread, WD &newWD);
     virtual WD *atIdle (BaseThread *thread);
     virtual void queue (BaseThread *thread, WD &wd);
};

void BreadthFirstPolicy::queue (BaseThread *thread, WD &wd)
{
    readyQueue.push_back(&wd);
}

WD * BreadthFirstPolicy::atCreation (BaseThread *thread, WD &newWD)
{
    queue(thread,newWD);
    return 0;
}

WD * BreadthFirstPolicy::atIdle (BaseThread *thread)
{
    return readyQueue.pop_front(thread);
}

// Factory
SchedulingGroup * createBreadthFirstPolicy ()
{
    return new BreadthFirstPolicy();
}
