#ifndef _BASE_THREAD_ELEMENT
#define _BASE_THREAD_ELEMENT

#include "workdescriptor.hpp"
#include "atomic.hpp"

namespace nanos {

// forward declarations
class ProcessingElement;
class SchedulingGroup;
class SchedulingData;

// Threads are binded to a PE for its life-time
class BaseThread {
private:
  static Atomic<int> idSeed;
  // Thread info
  int id;

  ProcessingElement *pe;
  WD & threadWD;

  // Thread status
  bool started;
  volatile bool mustStop;
  WD *    currentWD;

  // scheduling info
  SchedulingGroup *schedGroup;
  SchedulingData  *schedData;

   //disable copy and assigment
  BaseThread(const BaseThread &);
  const BaseThread operator= (const BaseThread &);

  virtual void run_dependent () = 0;
public:

  // constructor
  BaseThread (WD &wd, ProcessingElement *creator=0) : 
     id(idSeed++),pe(creator), threadWD(wd), started(false), mustStop(false) {}
  // destructor
  virtual ~BaseThread() {}

  virtual void start () = 0;
  void run();
  void stop() { mustStop = true; }
  virtual void join() = 0;

  // WD micro-scheduling
  virtual void inlineWork (WD *work) = 0;
  virtual void switchTo(WD *work) = 0;
  virtual void exitTo(WD *work) = 0;

  // set/get methods
  void setCurrentWD (WD &current) { currentWD = &current; }
  WD * getCurrentWD () const { return currentWD; }
  WD & getThreadWD () const { return threadWD; }

  SchedulingGroup * getSchedulingGroup () const { return schedGroup; }
  SchedulingData * getSchedulingData () const { return schedData; }
  void setScheduling (SchedulingGroup *sg, SchedulingData *sd)  { schedGroup = sg; schedData = sd; }

  bool isStarted () const { return started; }
  bool isRunning () const { return started && !mustStop; }
  ProcessingElement * runningOn() const { return pe; }
  void associate();

  int getId() { return id; }
};

extern __thread BaseThread *myThread;

}

#endif
