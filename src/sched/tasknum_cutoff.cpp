
#include "cutoff.hpp"

using namespace nanos;

//simple cutoff info, with the number of tasks
class tasknum_cutoff_info: public cutoff_info {
    friend class tasknum_cutoff;

    int num_tasks;

public:
    tasknum_cutoff_info(int nt): num_tasks(nt) {}
};


class tasknum_cutoff: public cutoff {
private:
    int max_cutoff;

public:
    void init() {}
    void setMaxCutoff(int mc) { max_cutoff = mc; } 
    bool cutoff_pred(cutoff_info *);

    ~tasknum_cutoff() {}
};


bool tasknum_cutoff::cutoff_pred(cutoff_info * info) {
  tasknum_cutoff_info * info_tn = (tasknum_cutoff_info *) info;
  if( info_tn->num_tasks > max_cutoff ) return false;
  return true;
}

//factory
cutoff * createTasknumCutoff() {
    return new tasknum_cutoff();
}
