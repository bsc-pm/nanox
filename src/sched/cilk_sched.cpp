#include "schedule.hpp"
#include "wddeque.hpp"

using namespace nanos;


class TaskStealData : public SchedulingData
{
	friend class TaskStealPolicy; //in this way, the policy can access the readyQueue

protected:
  	//  int schId;
  	WDDeque readyQueue;

public:
	// constructor
	TaskStealData(int id=0) : SchedulingData(id) {}
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
	TaskStealPolicy(int groupsize) : SchedulingGroup("task-steal-sch", groupsize) {}
	// TODO: copy and assigment operations
	// destructor
	virtual ~TaskStealPolicy() {}

	virtual WD *atCreation (BaseThread *thread, WD &newWD);
	virtual WD *atIdle (BaseThread *thread);
	virtual void queue (BaseThread *thread, WD &wd);
	virtual SchedulingData * createMemberData (BaseThread &thread);
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

    		return wd;
	} else {
		int newposition = ((data->schId) +1) % size;

		//should be random: for now it checks neighbour queues in round robin
		while((newposition != data->schId) && (wd = (((TaskStealData *) group[newposition])->readyQueue.pop_front(thread))) == NULL) {
			newposition = (newposition +1) % size;
        	}
		return wd;
	}
}

SchedulingData * TaskStealPolicy::createMemberData (BaseThread &thread)
{
	return new TaskStealData();
}


// Factories
SchedulingGroup * createTaskStealPolicy ()
{
	return new TaskStealPolicy();
}

SchedulingGroup * createTaskStealPolicy (int groupsize)
{
	return new TaskStealPolicy(groupsize);
}
