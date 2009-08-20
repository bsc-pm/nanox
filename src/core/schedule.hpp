#ifndef _NANOS_SCHEDULE
#define _NANOS_SCHEDULE

#include "workdescriptor.hpp"
#include "wddeque.hpp"
#include "basethread.hpp"
#include "atomic.hpp"
#include <string>

namespace nanos {

class SchedulingData {
private:
  int schId;
public:

      // constructor
      SchedulingData(int id=0) : schId(id) {}
      //TODO: copy & assigment costructor

      // destructor
      ~SchedulingData() {}

      void setSchId(int id)  { schId = id; }
      int getSchId() const { return schId; }
};
  
// Groups a number of BaseThreads and a number of WD with a policy
// Each BaseThread and WD can pertain only to a SG
class SchedulingGroup {
private:
  typedef std::vector<SchedulingData *> group_t;

  std::string    name;
  WDDeque        idleQueue;

  int            size;
  group_t        group;



    // disable copy and assignment
    SchedulingGroup(const SchedulingGroup &);
    SchedulingGroup & operator= (const SchedulingGroup &);

    void init(int groupSize);
public:
    // constructors
    SchedulingGroup(std::string &policy_name, int groupSize=1) : name(policy_name) { init(groupSize); }
    SchedulingGroup(const char  *policy_name, int groupSize=1) : name(policy_name) { init(groupSize); }
    // destructor
    virtual ~SchedulingGroup() {}


    //modifiers
    SchedulingData * getMemberData(int id) { return group[id]; }
    int getSize() { return group.size(); }
 

    // membership related methods. This members are not thread-safe
    virtual void addMember (BaseThread &thread);
    virtual void removeMember (BaseThread &thread);
    virtual SchedulingData * createMemberData (BaseThread &thread) { return new SchedulingData(); };

// TODO: void (*parse_option) (char *opt, char *value);

    // policy related methods
    virtual WD *atCreation (BaseThread *thread, WD &newWD) { return 0; }
    virtual WD *atIdle     (BaseThread *thread) = 0;
    virtual WD *atExit     (BaseThread *thread) { return atIdle(thread); }
    virtual WD *atBlock    (BaseThread *thread, WD *hint=0) { return atIdle(thread); }
    virtual WD *atWakeUp   (BaseThread *thread, WD &wd) { return 0; }

    virtual void queue (BaseThread *thread,WD &wd)  = 0;

    // idle management
    virtual void queueIdle (BaseThread *thread,WD &wd);
    virtual WD *getIdle(BaseThread *thread);
};

// singleton class to encapsulate scheduling data and methods
class Scheduler {
public:
    static void submit (WD &wd);
    static void exit (void);
    static void blockOnCondition (volatile int *var, int condition = 0);
    static void idle (void);
    static void queue (WD &wd);
};



typedef SchedulingGroup SG;
typedef SG * (*sgFactory) ();

};

#endif

