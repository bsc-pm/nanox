#include "schedule.hpp"
#include "queue.hpp"

using namespace nanos;

class BreadthFirstData : public SchedulingData
{
friend class BreadthFirstPolicy;
private:
     Queue<WD *>   readyQueue;
// TODO: constructor & destructor
};

class BreadthFirstPolicy : public SchedulingGroup {
private:
     Queue<WD *>   readyQueue;
public:
     // constructor
     BreadthFirstPolicy() : SchedulingGroup("breadth-first-sch") {}
     // TODO: copy and assigment operations
     // destructor
     virtual ~BreadthFirstPolicy() {}

     virtual BreadthFirstData * createMemberData (PE &pe);
     virtual WD *atCreation (PE *pe, WD &newWD);
     virtual WD *atIdle (PE *pe);
     virtual void queue (PE *pe, WD &wd);
};

BreadthFirstData *BreadthFirstPolicy::createMemberData (PE &pe)
{
    return new BreadthFirstData();
}

void BreadthFirstPolicy::queue (PE *pe, WD &wd)
{
    if ( wd.isTied() ) {
	BreadthFirstData *data = (BreadthFirstData *) pe->getSchedulingData();
	data->readyQueue.push(&wd);
    } else readyQueue.push(&wd);
}

WD * BreadthFirstPolicy::atCreation (PE *pe, WD &newWD)
{
    queue(pe,newWD);
    return 0;
}

WD * BreadthFirstPolicy::atIdle (PE *pe)
{
    WD *result;
    BreadthFirstData *data = (BreadthFirstData *) pe->getSchedulingData();
    // TODO: handle teams, heterogenity, tiedness, ... probably at the generic class
    
    if (readyQueue.try_pop(result)) return result;
    else if ( data->readyQueue.try_pop(result) ) return result;
    else return 0;
}

// Factory
SchedulingGroup * createBreadthFirstPolicy ()
{
    return new BreadthFirstPolicy();
}
