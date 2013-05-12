/*************************************************************************************/
/*      Copyright 2009 Barcelona Supercomputing Center                               */
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

#ifndef _CLUSTERTHREAD_DECL
#define _CLUSTERTHREAD_DECL

//#include "smpthread.hpp"
#include "basethread_decl.hpp"
//#include "wddeque.hpp"

#define MAX_PRESEND 1024

namespace nanos {
namespace ext
{

   class ClusterThread : public BaseThread
   {
      private:

      class RunningWDQueue {
         Atomic<unsigned int> _numRunning;
         Atomic<unsigned int> _completedHead;
         Atomic<unsigned int> _completedHead2;
         unsigned int _completedTail;
         WD* _completedWDs[MAX_PRESEND];
         
         public:
         RunningWDQueue();
         ~RunningWDQueue();
         void addRunningWD( WorkDescriptor *wd );
         unsigned int numRunningWDs() const;
         void clearCompletedWDs( ClusterThread *self );
         void completeWD( void *remoteWdAddr );
      };

      unsigned int                     _clusterNode; // Assigned Cluster device Id
      RunningWDQueue _runningWDs[2]; //0: SMP, 1: GPU
      Lock _lock;

      // disable copy constructor and assignment operator
      ClusterThread( const ClusterThread &th );
      const ClusterThread & operator= ( const ClusterThread &th );

      public:
      // constructor
      ClusterThread( WD &w, PE *pe, SMPMultiThread *parent, int device );

      // destructor
      virtual ~ClusterThread();

      void unlock();
      bool tryLock();

      virtual void runDependent ( void );
      virtual bool inlineWorkDependent ( WD &wd );
      virtual void outlineWorkDependent ( WD &wd );

      void addRunningWDSMP( WorkDescriptor *wd );
      unsigned int numRunningWDsSMP() const;
      void clearCompletedWDsSMP2( );

      void addRunningWDGPU( WorkDescriptor *wd );
      unsigned int numRunningWDsGPU() const;
      void clearCompletedWDsGPU2( );

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

      bool acceptsWDsSMP() const;
   };


}
}

#endif /* _CLUSTERTHREAD_DECL */
