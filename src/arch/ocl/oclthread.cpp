
#include "oclprocessor.hpp"
#include "oclthread.hpp"

using namespace nanos;
using namespace nanos::ext;

namespace {

    class CommandProfiler {
    public:

        CommandProfiler(OCLDD &dd, OCLDD::ProfData *prof) :
        _proc(*static_cast<OCLProcessor *> (myThread->runningOn())),
        _prof(dd.isProfiled() ? prof : NULL) {
            if (_prof)
                *(_prof->_startTick) = _proc.readTicks();
        }

        ~CommandProfiler() {
            if (_prof)
                *(_prof->_endTick) = _proc.readTicks();
        }

    private:
        OCLProcessor &_proc;
        OCLDD::ProfData *_prof;
    };

} // End anonymous namespace.

//
// OCLLocalThread implementation.
//

void OCLLocalThread::initializeDependent() {
    // Since we create an OCLLocalThread for each OCLProcessor, and an
    // OCLProcessor for each OpenCL device, force device initialization here, in
    // order to be executed in parallel.
    OCLProcessor *myProc = static_cast<OCLProcessor *> (myThread->runningOn());
    myProc->initialize();
}

void OCLLocalThread::runDependent() {
    WD &wd = getThreadWD();
    SMPDD &dd = static_cast<SMPDD &> (wd.activateDevice(SMP));

    SMPDD::work_fct workFct = dd.getWorkFct();
    workFct(wd.getData());
}

void OCLLocalThread::inlineWorkDependent(WD &wd) {
    wd.start(WD::IsNotAUserLevelThread);

    OCLDD &dd = static_cast<OCLDD &> (wd.getActiveDevice());

    if (OCLNDRangeKernelStarSSDD * ndRangeKernss =
            dynamic_cast<OCLNDRangeKernelStarSSDD *> (&dd)) {
        void *data = ndRangeKernss->getOpenCLData();      
        
        for( OCLNDRangeKernelStarSSDD::arg_iterator j = OCLNDRangeKernelStarSSDD::ArgsIterator::begin(data); j != OCLNDRangeKernelStarSSDD::ArgsIterator::end(data); ++j )
        {
                OCLNDRangeKernelStarSSDD::Arg &arg = *j;
                CopyData *copy_array = wd.getCopies();
                unsigned int i = 0;
                for (i = 0; i < wd.getNumCopies(); ++i) {
                    CopyData cpd = copy_array[i];
                    //In pointers we have pointer size, not buffer size
                    //This way we put buffer size
                    void * ptr = (void*) cpd.getAddress();
                    if (ptr==arg._ptr){
                        size_t sizeBuff=cpd.getSize();
                        sizeBuff |= 1;
                        j->_size=sizeBuff;
                    }
                }
        }
        //CommandProfiler prof(dd, OCLDD::getProfilePtr(data, offset));
        OCLProcessor &proc = *static_cast<OCLProcessor *> (myThread->runningOn());
        proc.exec(ndRangeKernss,  *static_cast<OCLNDRangeKernelStarSSDD::Data *> (data), OCLNDRangeKernelStarSSDD::ArgsIterator::begin(data), OCLNDRangeKernelStarSSDD::ArgsIterator::end(data));
        
        //Delete previously allocated data
        ndRangeKernss->deleteData(data);
    }
    else
        fatal0("Unknown DD");
}

void OCLLocalThread::yield() {
    OCLProcessor &proc = *static_cast<OCLProcessor *> (myThread->runningOn());

    proc.execTransfers();

#ifdef CLUSTER_DEV
    proc.processPendingMessages();
#endif
}

void OCLLocalThread::idle() {
    OCLProcessor &proc = *static_cast<OCLProcessor *> (myThread->runningOn());

    proc.execTransfers();

#ifdef CLUSTER_DEV
    proc.processPendingMessages();
#endif
}

bool OCLLocalThread::checkForAbort(OCLDD::event_iterator i,
        OCLDD::event_iterator e) {
    bool abortNeeded = false;

    for (; i != e; ++i)
        abortNeeded = abortNeeded || **i < 0;

    return abortNeeded;
}
