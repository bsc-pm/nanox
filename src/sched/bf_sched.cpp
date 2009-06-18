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

     virtual WD *atCreation (PE *pe, WD &newWD);
     virtual WD *atIdle (PE *pe);
     virtual void queue (PE *pe, WD &wd);
};

void BreadthFirstPolicy::queue (PE *pe, WD &wd)
{
    readyQueue.push_back(&wd);
}

WD * BreadthFirstPolicy::atCreation (PE *pe, WD &newWD)
{
    queue(pe,newWD);
    return 0;
}

WD * BreadthFirstPolicy::atIdle (PE *pe)
{
    return readyQueue.pop_front(pe);
}

// Factory
SchedulingGroup * createBreadthFirstPolicy ()
{
    return new BreadthFirstPolicy();
}
