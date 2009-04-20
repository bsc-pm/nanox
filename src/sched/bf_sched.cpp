#include "schedule.hpp"
#include "queue.hpp"

using namespace nanos;

class BreadthFirstPolicy : public SchedulingGroup {
private:
     Queue<WD *>   readyQueue;
public:
     // constructor
     BreadthFirstPolicy() : SchedulingGroup("breadth-first-sch") {}
     // destructor
     ~BreadthFirstPolicy() {}

     virtual WD *atCreation (PE *pe, WD &newWD);
     virtual WD *atIdle (PE *pe);
     virtual void queue (PE *pe, WD &wd);
};

void BreadthFirstPolicy::queue (PE *pe, WD &wd)
{
    readyQueue.push(&wd);
}

WD * BreadthFirstPolicy::atCreation (PE *pe, WD &newWD)
{
    queue(pe,newWD);
    return 0;
}

WD * BreadthFirstPolicy::atIdle (PE *pe)
{
    WD *result;
    // TODO: handle teams, heterogenity, tiedness, ... probably at the generic class
    if (readyQueue.try_pop(result)) return result;
    else return 0;
}

// Factory
SchedulingGroup * createBreadthFirstPolicy ()
{
    return new BreadthFirstPolicy();
}
