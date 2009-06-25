#include "basethread.hpp"
#include "processingelement.hpp"

using namespace nanos;

__thread BaseThread * nanos::myThread=0;

Atomic<int> BaseThread::idSeed = 0;

void BaseThread::run ()
{
    started = true;
    myThread = this;
    threadWD->tieTo(*this);

    run_dependent();
}
