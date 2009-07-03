#include "schedule.hpp"
#include "wddeque.hpp"

using namespace nanos;

class TaskStealPolicy : public SchedulingGroup {
private:
     WDDeque   readyQueue;
public:
     // constructor
     TaskStealPolicy() : SchedulingGroup("task-steal-sch") {}
     // TODO: copy and assigment operations
     // destructor
     virtual ~TaskStealPolicy() {}

     virtual WD *atCreation (PE *pe, WD &newWD);
     virtual WD *atIdle (PE *pe);
     virtual void queue (PE *pe, WD &wd);
};

void TaskStealPolicy::queue (PE *pe, WD &wd)
{
    readyQueue.push_back(&wd);
}

WD * TaskStealPolicy::atCreation (PE *pe, WD &newWD)
{
    queue(pe,newWD);
    return 0;
}

WD * TaskStealPolicy::atIdle (PE *pe)
{
  WorkDescriptor * wd;

  if ((wd = readyQueue.pop_front(pe)) != NULL) {
    std::cout << "I have a task in my queue: do not need to steal one" << std::endl;
    return wd;
  } else { //steal tasks from other pes?
    //TODO
    std::cout << "I should steal a task, but I don't yet know how to do this" << std::endl;
    return wd;
  }
}

// Factory
SchedulingGroup * createTaskStealPolicy ()
{
    return new TaskStealPolicy();
}
