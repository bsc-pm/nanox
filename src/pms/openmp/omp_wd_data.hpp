#ifndef _NANOX_OMP_DATA
#define _NANOX_OMP_DATA

#include "omp.h"
#include <iostream>

namespace nanos {
   namespace OpenMP {
      
      struct LoopSchedule {
         omp_sched_t      _kind;
         unsigned int     _modifier;

         LoopSchedule ( omp_sched_t kind, unsigned int modifier=0 ) : _kind(kind), _modifier(modifier) {}
      };

      /**! per tasks-icvs */
      class TaskICVs
      {
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

      /* OpenMP and OmpSs common data */
      class OmpData
      {
         protected:
            bool _final; /**< This is a final WD */

         public:
            /*! \brief OmpData default constructor
             */
            OmpData() {}
            /*! \brief OmpData destructor
             */
            virtual ~OmpData() {}
            /*! \brief Set the Data to be final
             */
            void setFinal ( bool final ) { _final = final; }
            /*! \brief Whether the WD is final or not
             */
            bool isFinal ( void ) const { return _final; }

            virtual TaskICVs * icvs() = 0;
            virtual void setICVs ( TaskICVs *icvs_in ) = 0;

            OmpData & operator= ( const OmpData & parent )
            {
               if ( &parent != this ) {
                  _final = parent._final;
               }
               return *this;
            }
      };

      class OpenMPData : public OmpData
      {
         private:
            TaskICVs _icvs;

            explicit OpenMPData ( const OpenMPData & );

         public:
            OpenMPData() : OmpData() , _icvs() { }
            ~OpenMPData() {}

            TaskICVs * icvs() { return &_icvs; }
            void setICVs ( TaskICVs *icvs_in ) { _icvs = *icvs_in; }

            OpenMPData & operator= ( const OpenMPData & parent )
            {
               OmpData::operator= ( parent );
               if ( &parent != this ) {
                  _icvs = parent._icvs;
               }
               return *this;
            }
      };

      class OmpSsData : public OmpData
      {
         private:
            TaskICVs *_icvs;

            explicit OmpSsData ( const OmpSsData & );

         public:
            OmpSsData() : OmpData(), _icvs(NULL) { }
            ~OmpSsData() {}

            TaskICVs * icvs() { return _icvs; }
            void setICVs ( TaskICVs *icvs_in ) { _icvs = icvs_in; }

            OmpSsData & operator= ( const OmpSsData & parent )
            {
               OmpData::operator= ( parent );
               if ( &parent != this ) {
                  _icvs = parent._icvs;
               }
               return *this;
            }
      };

      class OmpState
      {
         private:
            /* global ICVs */
            TaskICVs     _globalICVs;
            unsigned int _threadLimitVar;
            unsigned int _maxActiveLevelsVar;

            /* bindVar becomes local per task in OpenMP 4.0 */
            // bool         _bindVar;

            /* Not implemented global ICV's */
            // unsigned int _stacksizeVar;
            // LoopSchedule _defSchedVar;
            // WaitPolicy   _waitPolicyVar;

         public:

            unsigned int getThreadLimit () const { return _threadLimitVar; }

            unsigned int getMaxActiveLevels() const { return _maxActiveLevelsVar; }
            void setMaxActiveLevels( unsigned int levels ) { _maxActiveLevelsVar = levels; }

            TaskICVs & getICVs () { return _globalICVs; }
      };

      extern OmpState *globalState;
   }
}

#endif

