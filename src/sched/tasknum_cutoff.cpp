#include "schedule.hpp"
#include "cutoff.hpp"


#define DEFAULT_CUTOFF_NUM 100

using namespace nanos;


class tasknum_cutoff: public cutoff {
private:
    int max_cutoff;

public:
    tasknum_cutoff() : max_cutoff(DEFAULT_CUTOFF_NUM) {}

    void init() {}
    void setMaxCutoff(int mc) { max_cutoff = mc; } 
    bool cutoff_pred();

    ~tasknum_cutoff() {}
};


bool tasknum_cutoff::cutoff_pred() {
  if( Scheduler::getTaskNum() > max_cutoff ) return false;
  return true;
}

//factory
cutoff * createTasknumCutoff() {
    return new tasknum_cutoff();
}
