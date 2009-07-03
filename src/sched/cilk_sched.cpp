#include "schedule.hpp"
#include "wddeque.hpp"

using namespace nanos;


class TaskStealData : public SchedulingData
{
  friend class TaskStealPolicy; //in this way, the policy can access the readyQueue

private:
  int schId;
  WDDeque readyQueue;

public:
  // constructor
  TaskStealData(int id=0) : schId(id) {}
  //TODO: copy & assigment costructor
  
  // destructor
  ~TaskStealData() {}
  
  void setSchId(int id)  { schId = id; }
  int getSchId() const { return schId; }
};

class TaskStealPolicy : public SchedulingGroup {
public:
     // constructor
     TaskStealPolicy() : SchedulingGroup("task-steal-sch") {}
     // TODO: copy and assigment operations
     // destructor
     virtual ~TaskStealPolicy() {}

     virtual WD *atCreation (BaseThread *thread, WD &newWD);
     virtual WD *atIdle (BaseThread *thread);
     virtual void queue (BaseThread *thread, WD &wd);
};

void TaskStealPolicy::queue (BaseThread *thread, WD &wd)
{
  TaskStealData *data = (TaskStealData *) thread->getSchedulingData();
  data->readyQueue.push_back(&wd);
}

WD * TaskStealPolicy::atCreation (BaseThread *thread, WD &newWD)
{
    queue(thread,newWD);
    return 0;
}

WD * TaskStealPolicy::atIdle (BaseThread *thread)
{
  WorkDescriptor * wd;

  TaskStealData *data = (TaskStealData *) thread->getSchedulingData();
  if ( (wd = data->readyQueue.pop_front(thread)) != NULL ) {
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
