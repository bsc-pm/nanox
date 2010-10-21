#ifndef _NANOX_OMP_DATA
#define _NANOX_OMP_DATA

#include "omp.h"

namespace nanos {
   namespace OpenMP {
      
      struct LoopSchedule {
         omp_sched_t      _kind;
         unsigned int     _modifier;

         LoopSchedule ( omp_sched_t kind, unsigned int modifier=0 ) : _kind(kind), _modifier(modifier) {}
      };

      /**! per tasks-icvs */
      class TaskICVs {
         private:
         
         bool             _dynVar;
         bool             _nestVar;
         unsigned int     _nthreadsVar;
         LoopSchedule     _runSchedVar;

            TaskICVs ( const TaskICVs &);

         public:
            TaskICVs () : _dynVar(false), _nestVar(false), _nthreadsVar(1),
                          _runSchedVar(omp_sched_static) {}

            bool getDynamic() const { return _dynVar; }
            void setDynamic(bool value) { _dynVar = value; }

            bool getNested() const { return _nestVar; }
            void setNested(bool value) { _nestVar = value; }

            unsigned int getNumThreads() const { return _nthreadsVar; }
            void setNumThreads(unsigned int value) { _nthreadsVar = value; }

            const LoopSchedule & getSchedule() const { return _runSchedVar; }
            void  setSchedule ( const LoopSchedule &schedule ) { _runSchedVar = schedule; }

            TaskICVs & operator= ( const TaskICVs & parent )
            {
              _dynVar = parent._dynVar;
              _nestVar = parent._nestVar;
              _runSchedVar = parent._runSchedVar;
              _nthreadsVar = parent._nthreadsVar;
              return *this;
            }
      };

      class OmpData {
         private:
            TaskICVs         _icvs;

            explicit OmpData ( const OmpData & );
         public:

            TaskICVs & icvs() { return _icvs; }

            OmpData & operator= ( const OmpData & parent )
            {
              if ( &parent != this ) {
                 _icvs = parent._icvs;
              }
              return *this;
            }

      };

      class OmpState {
         private:
            /* global ICVs */
    	      TaskICVs     _globalICVs;
            /* def-sched-var */
            /* stacksize-var */
            /* wait-policy-var */
            unsigned int _threadLimit;
            unsigned int _maxActiveLevels;

         public:

            unsigned int getThreadLimit () const { return _threadLimit; }

            unsigned int getMaxActiveLevels() const { return _maxActiveLevels; }
            void setMaxActiveLevels( unsigned int levels ) { _maxActiveLevels = levels; }

            TaskICVs & getICVs () { return _globalICVs; }
      };

      extern OmpState *globalState;
   }
}

#endif

