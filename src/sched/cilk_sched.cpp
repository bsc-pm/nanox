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
  } else { //steal tasks from other pes
    //select a new task queue: should be random, but for now round-robin works..
    //data->schId corresponds to queue position in group!
    int newposition = ((data->schId) +1) % size;
    while((wd = (((TaskStealData *) group[newposition])->readyQueue.pop_front(thread))) == NULL)
      newposition = newposition + 1 % size; //cyclic on the number of elements in group    
    
    std::cout << "Task stolen, but with a very fine grain...I should steal a packet of tasks" << std::endl;
    return wd;
  }
}

// Factory
SchedulingGroup * createTaskStealPolicy ()
{
    return new TaskStealPolicy();
}
