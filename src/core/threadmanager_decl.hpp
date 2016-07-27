/*************************************************************************************/
/*      Copyright 2015 Barcelona Supercomputing Center                               */
/*                                                                                   */
/*      This file is part of the NANOS++ library.                                    */
/*                                                                                   */
/*      NANOS++ is free software: you can redistribute it and/or modify              */
/*      it under the terms of the GNU Lesser General Public License as published by  */
/*      the Free Software Foundation, either version 3 of the License, or            */
/*      (at your option) any later version.                                          */
/*                                                                                   */
/*      NANOS++ is distributed in the hope that it will be useful,                   */
/*      but WITHOUT ANY WARRANTY; without even the implied warranty of               */
/*      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                */
/*      GNU Lesser General Public License for more details.                          */
/*                                                                                   */
/*      You should have received a copy of the GNU Lesser General Public License     */
/*      along with NANOS++.  If not, see <http://www.gnu.org/licenses/>.             */
/*************************************************************************************/

#ifndef THREADMANAGER_DECL_HPP
#define THREADMANAGER_DECL_HPP

#include "config.hpp"
#include "atomic_decl.hpp"
#include "cpuset.hpp"
#include "basethread_decl.hpp"

namespace nanos {

   //! ThreadManager base class
   /*!
    * This class is used when yield, block and sleep are disabled.
    * All the virtual methods are defined with an empty body.
    * Function lastActiveThread is common to all Thread Managers, and can be defined
    * here as it should protect us against an all-threads-blocked situation
    *
    * Used when neither --thread-manager / yield / block /sleep are set
    */
   class ThreadManager
   {
      protected:
         Lock              _lock;
         bool              _initialized;
         unsigned int      _maxThreads;
         const CpuSet     *_cpuProcessMask;     /* Read-only masks from SMPPlugin */
         const CpuSet     *_cpuActiveMask;

      private:
         bool              _warmupThreads;

      public:
         ThreadManager( bool warmup=false );
         virtual ~ThreadManager() {}

         virtual void init();
         bool lastActiveThread();
         unsigned int getMaxThreads() { return _maxThreads; }

         virtual bool isGreedy() { return false; }
         virtual void idle( int& yields
#ifdef NANOS_INSTRUMENTATION_ENABLED
                     , unsigned long long& total_yields, unsigned long long& total_blocks
                     , unsigned long long& time_yields, unsigned long long& time_blocks
#endif
                     ) {}
         virtual void acquireOne() {}
         virtual void acquireResourcesIfNeeded() {}
         virtual void returnClaimedCpus() {}
         virtual void returnMyCpuIfClaimed() {}
         virtual void waitForCpuAvailability() {}
         virtual void blockThread(BaseThread*) {}
         virtual void unblockThread(BaseThread*) {}
         virtual void processMaskChanged() {}
   };

   //! BlockingThreadManager class
   /*!
    * This derived class is used when yield or block are enabled.
    * Thread management relies entirely on the runtime, no DLB micro-load balancing
    * is allowed, although DLB can be called eventually through DLB_Update for
    * resource manager or statistics reasons.
    *
    * Used when --thread-manager=nanos, or either yield or block are set
    */
   class BlockingThreadManager : public ThreadManager
   {
      private:
         bool              _isMalleable;
         unsigned int      _numYields;
         bool              _useBlock;
         bool              _useDLB;

      public:
         BlockingThreadManager( unsigned int num_yields, bool use_block, bool use_dlb, bool warmup );
         virtual ~BlockingThreadManager();
         virtual void init();
         virtual bool isGreedy();
         virtual void idle( int& yields
#ifdef NANOS_INSTRUMENTATION_ENABLED
                     , unsigned long long& total_yields, unsigned long long& total_blocks
                     , unsigned long long& time_yields, unsigned long long& time_blocks
#endif
                     );
         virtual void acquireOne();
         virtual void acquireResourcesIfNeeded();
         virtual void returnClaimedCpus();
         virtual void returnMyCpuIfClaimed();
         virtual void waitForCpuAvailability();
         virtual void blockThread(BaseThread*);
         virtual void unblockThread(BaseThread*);
         virtual void processMaskChanged();
   };

   //! BusyWaitThreadManager class
   /*!
    * This derived class is used when either sleep is enabled or when DLB is running
    * with basic policies like LeWI or LeWI_mask
    *
    * Used when --thread-manager=nanos and --enable-sleep,
    *   or when --enable-dlb and block is not explicitly selected
    */
   class BusyWaitThreadManager : public ThreadManager
   {
      private:
         bool              _isMalleable;
         unsigned int      _numYields;
         unsigned int      _sleepTime;
         bool              _useSleep;
         bool              _useDLB;

      public:
         BusyWaitThreadManager( unsigned int num_yields, unsigned int sleep_time,
                                 bool use_sleep, bool use_dlb, bool warmup );
         virtual ~BusyWaitThreadManager();
         virtual void init();
         virtual bool isGreedy();
         virtual void idle( int& yields
#ifdef NANOS_INSTRUMENTATION_ENABLED
                     , unsigned long long& total_yields, unsigned long long& total_blocks
                     , unsigned long long& time_yields, unsigned long long& time_blocks
#endif
                     );
         virtual void acquireOne();
         virtual void acquireResourcesIfNeeded();
         virtual void returnClaimedCpus();
         virtual void returnMyCpuIfClaimed();
         virtual void waitForCpuAvailability();
         virtual void blockThread(BaseThread*);
         virtual void unblockThread(BaseThread*);
         virtual void processMaskChanged();
   };

   //! DlbThreadManager class
   /*!
    * This derived class is used when DLB is running with advanced policies like auto_LeWI_mask
    * Block is enabled regardless of the user option
    *
    * Used when --thread-manager=dlb
    */
   class DlbThreadManager : public ThreadManager
   {
      private:
         bool              _isMalleable;
         unsigned int      _numYields;

      public:
         DlbThreadManager( unsigned int num_yields, bool warmup );
         virtual ~DlbThreadManager();
         virtual void init();
         virtual bool isGreedy();
         virtual void idle( int& yields
#ifdef NANOS_INSTRUMENTATION_ENABLED
                     , unsigned long long& total_yields, unsigned long long& total_blocks
                     , unsigned long long& time_yields, unsigned long long& time_blocks
#endif
                     );
         virtual void acquireOne();
         virtual void acquireResourcesIfNeeded();
         virtual void returnClaimedCpus();
         virtual void returnMyCpuIfClaimed();
         virtual void waitForCpuAvailability();
         virtual void blockThread(BaseThread*);
         virtual void unblockThread(BaseThread*);
         virtual void processMaskChanged();
   };

   //! ThreadManagerConf class
   /*!
    * This class is used to construct the right Thread Manager object.
    *
    * It firsts implements a Config method to configure the user options.
    * The create method work as a factory method according to the user options.
    * The other methods are needed by other classes when maybe the thread manager
    * is not yet created.
    */
   class ThreadManagerConf
   {
      private:
         typedef enum { TM_UNDEFINED = 0, TM_NONE, TM_NANOS, TM_DLB } ThreadManagerOption;

         ThreadManagerOption  _tm;              //!< Thread Manager name option
         unsigned int         _numYields;       //!< Number of yields before block
         unsigned int         _sleepTime;       //!< Number of nanoseconds to sleep
         bool                 _useYield;        //!< Yield is enabled
         bool                 _useBlock;        //!< Block is enabled
         bool                 _useSleep;        //!< Sleep is enabled
         bool                 _useDLB;          //!< DLB library will be used
         bool                 _forceTieMaster;  //!< Force Master WD (user code) to run on Master Thread
         bool                 _warmupThreads;   //!< Force the initialization of as many threads as number of CPUs, then block them if needed

      public:
         static const unsigned int DEFAULT_SLEEP_NS;
         static const unsigned int DEFAULT_YIELDS;

         ThreadManagerConf();

         void setUseYield ( const bool value ) { _useYield = value; }
         void setUseBlock ( const bool value ) { _useBlock = value; }
         unsigned int getNumYields ( void ) const { return _useYield ? _numYields : 0; }
         bool getUseYield ( void ) const { return _useYield; }
         bool getUseBlock ( void ) const { return _useBlock; }

         void config( Config &cfg );
         ThreadManager* create();
         bool canUntieMaster() const;
         void setForceTieMaster ( const bool value ) { _forceTieMaster = value; }
   };

} // namespace nanos

#endif /* THREADMANAGER_DECL_HPP */
