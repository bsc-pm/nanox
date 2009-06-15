#ifndef _NANOS_SCHEDULE
#define _NANOS_SCHEDULE

#include "workdescriptor.hpp"
#include "processingelement.hpp"
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

// Groups a number of PEs and a number of WD with a policy
// Each PE and WD can pertain only to a SG
class SchedulingGroup {
private:
    typedef std::vector<SchedulingData *> group_t;

    std::string    name;
    int		   size;
    group_t        group;
    Queue<WD *>    idleQueue;
    
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

    // membership related methods. This members are not thread-safe
    virtual void addMember (PE &pe);
    virtual void removeMember (PE &pe);
    virtual SchedulingData * createMemberData (PE &pe) { return new SchedulingData(); };

// TODO: void (*parse_option) (char *opt, char *value);

    // policy related methods
    virtual WD *atCreation (PE *pe, WD &newWD) { return 0; }
    virtual WD *atIdle     (PE *pe) = 0;
    virtual WD *atExit     (PE *pe) { return atIdle(pe); }
    virtual WD *atBlock    (PE *pe, WD *hint=0) { return atIdle(pe); }
    virtual WD *atWakeUp   (PE *pe, WD &wd) { return 0; }

    virtual void queue (PE *pe,WD &wd)  = 0;

    // idle management
    virtual void queueIdle (PE *pe,WD &wd);
    virtual WD *getIdle(PE *pe);
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

};

#endif

