#ifndef _BASE_THREAD_ELEMENT
#define _BASE_THREAD_ELEMENT

#include "workdescriptor.hpp"
#include "atomic.hpp"
namespace nanos {

// forward declarations
class ProcessingElement;
class SchedulingGroup;
class SchedulingData;
class ThreadTeam;

// Threads are binded to a PE for its life-time
class BaseThread {
private:
  static Atomic<int> idSeed;
  Lock   mlock;
  
  // Thread info
  int id;

  ProcessingElement *pe;
  WD & threadWD;

  // Thread status
  bool started;
  volatile bool mustStop;
  WD *    currentWD;

  // Team info
  bool  has_team;
  ThreadTeam *team;

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
     id(idSeed++),pe(creator), threadWD(wd), started(false), mustStop(false), has_team(false), team(NULL) {}
  // destructor
  virtual ~BaseThread() {}

  // atomic access
  void lock () { mlock++; }
  void unlock () { mlock--; }

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

  void reserve() { has_team = 1; }
  void enterTeam(ThreadTeam *newTeam) { has_team=1; team = newTeam; }
  bool hasTeam() const { return has_team; }
  void leaveTeam() { has_team = 0; team = 0; }

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
