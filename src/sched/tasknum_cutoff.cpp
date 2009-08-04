
#include "cutoff.hpp"

using namespace nanos;

//simple cutoff info, with the number of tasks
class tasknum_cutoff_info: cutoff_info {
    friend class tasknum_cutoff;

    int num_tasks;

public:
    tasknum_cutoff_info(int nt): num_tasks(nt) {}
};


class tasknum_cutoff: cutoff {
private:
    int max_cutoff;

public:
    void init() {}
    void setMaxCutoff(int mc) { max_cutoff = mc; } 
    bool cutoff_pred(tasknum_cutoff_info *);
};


bool tasknum_cutoff::cutoff_pred(tasknum_cutoff_info * info) {
    if( info->num_tasks > max_cutoff ) return false;
    return true;
}

//factory
tasknum_cutoff * createTasknumCutoff() {
    return new tasknum_cutoff();
}
