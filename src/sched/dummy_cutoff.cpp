#include "cutoff.hpp"

using namespace nanos;

//no information in dummy cutoff info: always returns the same value
class dummy_cutoff_info: public cutoff_info {
    friend class dummy_cutoff;
};

class dummy_cutoff: public cutoff {
private:
    //we decide one time for all if new tasks are to be created during the execution
    bool createTask;
    //if createTask == true, then we have the maximum number of tasks else we have only one task (sequential comp.)
public:
    dummy_cutoff() : cutoff(), createTask(true) {}
    void setCreateTask(bool ct) { createTask = ct; }
    void init() {}
    //the predicate does not use the void dummy info
    bool cutoff_pred(cutoff_info *);

    ~dummy_cutoff() {};
};


bool dummy_cutoff::cutoff_pred(cutoff_info * c) {
    return createTask;
}

//factory
cutoff * createDummyCutoff() {
    return new dummy_cutoff();
}
