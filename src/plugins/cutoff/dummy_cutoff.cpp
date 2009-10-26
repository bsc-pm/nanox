#include "cutoff.hpp"

using namespace nanos;


class dummy_cutoff: public cutoff {
private:
    //we decide one time for all if new tasks are to be created during the execution
    bool createTask;
    //if createTask == true, then we have the maximum number of tasks else we have only one task (sequential comp.)
public:
    dummy_cutoff() : createTask(true) {}
    void setCreateTask(bool ct) { createTask = ct; }
    void init() {}

    bool cutoff_pred();

    ~dummy_cutoff() {};
};


bool dummy_cutoff::cutoff_pred() {
    return createTask;
}

//factory
cutoff * createDummyCutoff() {
    return new dummy_cutoff();
}
