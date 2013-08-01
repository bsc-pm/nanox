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

      /*!
       * \brief Internal Control Variables per task
       */
      class TaskICVs
      {
         private:
            bool             _dynVar;        /*!< \brief Dynamic adjustment of the number of threads for encountered parallel regions */
            bool             _nestVar;       /*!< \brief Nested parallelism is enabled for encountered parallel regions */
            unsigned int     _nthreadsVar;   /*!< \brief Number of threads requested for encountered parallel regions */
            LoopSchedule     _runSchedVar;   /*!< \brief Schedule that the runtime schedule clause uses for loop regions */

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

      /*!
       * \brief OpenMP and OmpSs common data for each WD
       */
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

      /*!
       * \brief OpenMP data for each WD
       */
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

      /*!
       * \brief OmpSs data for each WD
       */
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

      /*!
       * \brief Global State for either OpenMP or OmpSs programming model
       */
      class OmpState
      {
         private:
            TaskICVs    _globalICVs;            /*!< \brief Model's global Internal Control Variables */
            int         _threadLimitVar;        /*!< \brief Maximum number of threads participating in the program */
            int         _maxActiveLevelsVar;    /*!< \brief Maximum number of nested active parallel regions */

            /* Not implemented global ICV's */
            // unsigned int _stacksizeVar;
            // LoopSchedule _defSchedVar;
            // WaitPolicy   _waitPolicyVar;

         public:

            OmpState() : _globalICVs(), _threadLimitVar(INT_MAX), _maxActiveLevelsVar(INT_MAX) {}
            ~OmpState() {}

            int getThreadLimit () const { return _threadLimitVar; }

            int getMaxActiveLevels() const { return _maxActiveLevelsVar; }
            void setMaxActiveLevels( unsigned int levels ) { _maxActiveLevelsVar = levels; }

            TaskICVs & getICVs () { return _globalICVs; }
      };

      extern OmpState *globalState;
   }
}

#endif

