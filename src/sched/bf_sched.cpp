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
     virtual WD *atBlock (PE *pe, WD *hint=0);
};

WD * BreadthFirstPolicy::atCreation (PE *pe, WD &newWD)
{
    readyQueue.push(&newWD);
    return 0;
}

WD * BreadthFirstPolicy::atIdle (PE *pe)
{
    // TODO: handle teams, heterogenity, tiedness, ... probably at the generic class
    return readyQueue.pop();
}

WD * BreadthFirstPolicy::atBlock (PE *pe, WD *hint) {
    return atIdle(pe);
}

// Factory
SchedulingGroup * createBreadthFirstPolicy ()
{
    return new BreadthFirstPolicy();
}
