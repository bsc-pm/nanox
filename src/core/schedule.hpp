#ifndef _NANOS_SCHEDULE
#define _NANOS_SCHEDULE

#include "workdescriptor.hpp"
#include "processingelement.hpp"
#include <string>

namespace nanos {

// Groups a number of PEs and a number of WD with a policy
// Each PE and WD can pertain only to a SG
class SchedulingGroup {
private:
    std::string    name;
    // disable copy and assignment
    SchedulingGroup(const SchedulingGroup &);
    SchedulingGroup & operator= (const SchedulingGroup &);
public:
    // constructors
    SchedulingGroup(std::string &policy_name) : name(policy_name) {}
    SchedulingGroup(const char  *policy_name) : name(policy_name) {}
    // destructor
    virtual ~SchedulingGroup() {}

    virtual WD *atCreation (PE *pe, WD &newWD) { return 0; }
    virtual WD *atIdle     (PE *pe) = 0;
    virtual WD *atExit     (PE *pe) { return atIdle(pe); }
    virtual WD *atBlock    (PE *pe, WD *hint=0) { return atIdle(pe); }
    virtual WD *atWakeUp   (PE *pe, WD &wd) { return 0; }

    virtual void queue (PE *pe,WD &wd)  = 0;
// 	void (*enqueue_desc) (nth_desc_t *desc);
// 	void (*parse_option) (char *opt, char *value);
// 	int  (*num_ready) (void);

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

