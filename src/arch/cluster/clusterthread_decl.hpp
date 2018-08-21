/*************************************************************************************/
/*      Copyright 2009-2018 Barcelona Supercomputing Center                          */
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
/*      along with NANOS++.  If not, see <https://www.gnu.org/licenses/>.            */
/*************************************************************************************/

#ifndef _CLUSTERTHREAD_DECL
#define _CLUSTERTHREAD_DECL

//#include "smpthread.hpp"
#include "basethread_decl.hpp"
//#include "wddeque.hpp"
#include <list>

#define MAX_PRESEND 1024

namespace nanos {
namespace ext {

   class ClusterThread : public BaseThread
   {
      private:

      class RunningWDQueue {
         Atomic<unsigned int> _numRunning;
         Atomic<unsigned int> _completedHead;
         Atomic<unsigned int> _completedHead2;
         unsigned int _completedTail;
         WD* _completedWDs[MAX_PRESEND];
      std::list< WD * > _waitingDataWDs;
      WD *_pendingInitWD;
         
         public:
         RunningWDQueue();
         ~RunningWDQueue();
         void addRunningWD( WorkDescriptor *wd );
         unsigned int numRunningWDs() const;
         void clearCompletedWDs( ClusterThread *self );
         void completeWD( void *remoteWdAddr );

         bool hasAPendingWDToInit() const;
         WD *getPendingInitWD();
         void setPendingInitWD( WD *wd );

         bool hasWaitingDataWDs() const;
         WD *getWaitingDataWD();
         void addWaitingDataWD( WD *wd );
      };

      unsigned int                     _clusterNode; // Assigned Cluster device Id
      Lock _lock;
      RunningWDQueue _runningWDs[4]; //0: SMP, 1: GPU, 3: OCL, 4: FPGA

      // disable copy constructor and assignment operator
      ClusterThread( const ClusterThread &th );
      const ClusterThread & operator= ( const ClusterThread &th );

      public:
      // constructor
      ClusterThread( WD &w, PE *pe, SMPMultiThread *parent, int device );

      // destructor
      virtual ~ClusterThread();

      virtual void lock();
      virtual void unlock();
      virtual bool tryLock();

      virtual void runDependent ( void );
      virtual bool inlineWorkDependent ( WD &wd );
      virtual void preOutlineWorkDependent ( WD &wd );
      virtual void outlineWorkDependent ( WD &wd );

      void addRunningWD( unsigned int archId, WorkDescriptor *wd );
      unsigned int numRunningWDs( unsigned int archId ) const;
      void clearCompletedWDs( unsigned int archId );
      bool acceptsWDs( unsigned int archId ) const;

      virtual void join();
      virtual void start();
      virtual BaseThread * getNextThread ();

      virtual void idle( bool debug=false );

      virtual void notifyOutlinedCompletionDependent( WD *completedWD ); 
      virtual bool isCluster();


         virtual void switchTo( WD *work, SchedulerHelper *helper );
         virtual void exitTo( WD *work, SchedulerHelper *helper );
         virtual void switchHelperDependent( WD* oldWD, WD* newWD, void *arg );
         virtual void exitHelperDependent( WD* oldWD, WD* newWD, void *arg );
         virtual void initializeDependent( void );

         virtual void switchToNextThread();

      virtual void setupSignalHandlers();

      bool hasAPendingWDToInit( unsigned int arch_id ) const;
      WD *getPendingInitWD( unsigned int arch_id );
      void setPendingInitWD( unsigned int arch_id, WD *wd );

      bool hasWaitingDataWDs( unsigned int archId ) const;
      WD *getWaitingDataWD( unsigned int archId );
      void addWaitingDataWD( unsigned int archId, WD *wd );


      static void workerClusterLoop ( void );
      static WD * getClusterWD( BaseThread *thread );
   };


} // namespace ext
} // namespace nanos


#endif /* _CLUSTERTHREAD_DECL */
