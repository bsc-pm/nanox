#ifndef _NANOS_SCHEDULE
#define _NANOS_SCHEDULE

#include "workdescriptor.hpp"
#include "processingelement.hpp"
#include <string>

namespace nanos {

class SchedulerGroup {
};

class SchedulerPolicy {
private:
    std::string    name;
    // disable copy and assignment
    SchedulerPolicy(const SchedulerPolicy &);
    SchedulerPolicy & operator= (const SchedulerPolicy &);
public:
    // constructors
    SchedulerPolicy(std::string &policy_name) : name(policy_name) {}
    SchedulerPolicy(const char  *policy_name) : name(policy_name) {}
    // destructor
    ~SchedulerPolicy() {}

    virtual WD *atCreation (PE *pe, WD &newWD) { return 0; }
    virtual WD *atExit     (PE *pe) { return 0; }
    virtual WD *atIdle     (PE *pe) { return 0; }
    virtual WD *atBlock    (PE *pe, WD *hint=0) { return 0; }
    virtual WD *atWakeUp   (PE *pe, WD &wd) { return 0; }

// 	void (*enqueue_desc) (nth_desc_t *desc);
// 	void (*parse_option) (char *opt, char *value);
// 	int  (*num_ready) (void);

};

// singleton class to encapsulate scheduling data and methods
class Scheduler {
private:
    static SchedulerPolicy *policy;
public:
    static void submit (WD &wd);
    static void exit (void);
    static void blockOnCondition (volatile int *var, int condition = 0);
    static void idle (void);
};

};

#endif