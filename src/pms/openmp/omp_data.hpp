#ifndef _NANOX_OMP_DATA
#define _NANOX_OMP_DATA

#include "omp.h"

namespace nanos {
   namespace OpenMP {
      
      struct LoopSchedule {
         omp_sched_t      _kind;
         unsigned int     _modifier;

         LoopSchedule ( omp_sched_t kind, unsigned int modifier ) : _kind(kind), _modifier(modifier) {}
      };

      /**! per tasks-icvs */
      class TaskICVs {
         private:
         
         bool             _dynVar;
         bool             _nestVar;
         unsigned int     _nthreadsVar;
         LoopSchedule     _runSchedVar;

         public:
            bool getDynamic() const { return _dynVar; }
            void setDynamic(bool value) { _dynVar = value; }

            bool getNested() const { return _nestVar; }
            void setNested(bool value) { _nestVar = value; }

            unsigned int getNumThreads() const { return _nthreadsVar; }
            void setNumThreads(unsigned int value) { _nthreadsVar = value; }

            const LoopSchedule & getSchedule() const { return _runSchedVar; }
            void  setSchedule ( const LoopSchedule &schedule ) { _runSchedVar = schedule; }
      };

      class OmpData {
         private:
            TaskICVs         _icvs;

         public:

            TaskICVs & icvs() { return _icvs; }
      };

      class OmpState {
         private:
            /* global ICVs */
            /* def-sched-var */
            /* stacksize-var */
            /* wait-policy-var */
            unsigned int _threadLimit;
            unsigned int _maxActiveLevels;

         public:

            unsigned int getThreadLimit () const { return _threadLimit; }

            unsigned int getMaxActiveLevels() const { return _maxActiveLevels; }
            void setMaxActiveLevels( unsigned int levels ) { _maxActiveLevels = levels; }


      };

      extern OmpState *globalState;

   }
}

#endif

