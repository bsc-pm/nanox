/*************************************************************************************/
/*      Copyright 2013 Barcelona Supercomputing Center                               */
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

namespace nanos
{
   //! ThreadManager base class
   /*!
    * This class is used when yield and block are disabled.
    * All the virtual methods are defined with an empty body.
    * Function lastActiveThread is common to all Thread Managers, and can be defined
    * here as it should protect us against an all-threads-blocked situation
    *
    * Used when neither --thread-manager / yield / block are set
    */
   class ThreadManager
   {
      protected:
         Lock              _lock;

      public:
         ThreadManager() : _lock() {}
         virtual ~ThreadManager() {}
         virtual void idle( int& yields
#ifdef NANOS_INSTRUMENTATION_ENABLED
                     , unsigned long long& total_yields, unsigned long long& total_blocks
                     , unsigned long long& time_yields, unsigned long long& time_blocks
#endif
                     ) {}
         virtual void acquireResourcesIfNeeded() {}
         virtual void releaseCpu() {}
         virtual void returnClaimedCpus() {}
         virtual void returnMyCpuIfClaimed() {}
         virtual void waitForCpuAvailability() {}

         bool lastActiveThread();
   };

   //! BasicThreadManager class
   /*!
    * This derived class is used when yield or block are enabled.
    * Thread management relies entirely on the runtime, no DLB micro-load balancing
    * is allowed, although DLB can be called eventually through DLB_Update for
    * resource manager or statistics reasons.
    *
    * Used when --thread-manager=basic, or either yield or block are set
    */
   class BasicThreadManager : public ThreadManager
   {
      private:
         int               _maxCPUs;
         bool              _isMalleable;
         unsigned int      _numYields;
         bool              _useYield;
         bool              _useBlock;
         bool              _useDLB;

      public:
         BasicThreadManager( unsigned int num_yields, bool use_yield,
                              bool use_block, bool use_dlb );
         virtual ~BasicThreadManager();
         virtual void idle( int& yields
#ifdef NANOS_INSTRUMENTATION_ENABLED
                     , unsigned long long& total_yields, unsigned long long& total_blocks
                     , unsigned long long& time_yields, unsigned long long& time_blocks
#endif
                     );
         virtual void acquireResourcesIfNeeded();
         virtual void releaseCpu();
         virtual void returnClaimedCpus();
         virtual void returnMyCpuIfClaimed();
         virtual void waitForCpuAvailability();
   };

   //! BasicDlbThreadManager class
   /*!
    * This derived class is used when DLB is running with basic policies like LeWI or LeWI_mask
    * Yield and Block are allowed, although the block feature is replaced by a short sleep
    *
    * Used when --thread-manager=basic-dlb
    */
   class BasicDlbThreadManager : public ThreadManager
   {
      private:
         cpu_set_t         _waitingCPUs;
         int               _maxCPUs;
         bool              _isMalleable;
         unsigned int      _numYields;
         bool              _useYield;
         bool              _useBlock;
         bool              _useDLB;

      public:
         BasicDlbThreadManager( unsigned int num_yields, bool use_yield,
                                 bool use_block, bool use_dlb );
         virtual ~BasicDlbThreadManager();
         virtual void idle( int& yields
#ifdef NANOS_INSTRUMENTATION_ENABLED
                     , unsigned long long& total_yields, unsigned long long& total_blocks
                     , unsigned long long& time_yields, unsigned long long& time_blocks
#endif
                     );
         virtual void acquireResourcesIfNeeded();
         virtual void releaseCpu();
         virtual void returnClaimedCpus();
         virtual void returnMyCpuIfClaimed();
         virtual void waitForCpuAvailability();
   };

   //! AutoDlbThreadManager class
   /*!
    * This derived class is used when DLB is running with advanced policies like auto_LeWI_mask
    * Block is enabled regardless of the user option
    *
    * Used when --thread-manager=auto-dlb
    */
   class AutoDlbThreadManager : public ThreadManager
   {
      private:
         cpu_set_t         _waitingCPUs;
         int               _maxCPUs;
         bool              _isMalleable;
         unsigned int      _numYields;
         bool              _useYield;
         bool              _useBlock;
         bool              _useDLB;

      public:
         AutoDlbThreadManager( unsigned int num_yields, bool use_yield,
                                 bool use_block, bool use_dlb );
         virtual ~AutoDlbThreadManager();
         virtual void idle( int& yields
#ifdef NANOS_INSTRUMENTATION_ENABLED
                     , unsigned long long& total_yields, unsigned long long& total_blocks
                     , unsigned long long& time_yields, unsigned long long& time_blocks
#endif
                     );
         virtual void acquireResourcesIfNeeded();
         virtual void releaseCpu();
         virtual void returnClaimedCpus();
         virtual void returnMyCpuIfClaimed();
         virtual void waitForCpuAvailability();
   };

   class ThreadManagerConf
   {
      private:
         typedef enum { UNDEFINED, NONE, BASIC, GENERIC_DLB, BASIC_DLB, AUTO_DLB } ThreadManagerOption;

         ThreadManagerOption  _tm;              //!< Thread Manager name option
         unsigned int         _numYields;       //!< Number of yields before block
         bool                 _useYield;        //!< Yield is allowed
         bool                 _useBlock;        //!< Block is allowed
         bool                 _useDLB;          //!< DLB library will be used
         bool                 _forceTieMaster;  //!< Force Master WD (user code) to run on Master Thread
         bool                 _warmupThreads;   //!< Force the initialization of as many threads as number of CPUs, then block them if needed

      public:
         ThreadManagerConf();

         void setUseYield ( const bool value ) { _useYield = value; }
         void setUseBlock ( const bool value ) { _useBlock = value; }
         unsigned int getNumYields ( void ) const { return _numYields; }
         bool getUseYield ( void ) const { return _useYield; }
         bool getUseBlock ( void ) const { return _useBlock; }

         void config( Config &cfg );
         ThreadManager* create();
         bool canUntieMaster() const;
         bool threadWarmupEnabled() const;
   };
}
#endif /* THREADMANAGER_DECL_HPP */
