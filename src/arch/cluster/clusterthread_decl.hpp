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

#ifndef _NANOS_GPU_THREAD
#define _NANOS_GPU_THREAD

#include "smpthread.hpp"
#include "wddeque.hpp"
//#include "clusternode.hpp"
#include <queue>

#define MAX_PRESEND 64

namespace nanos {
namespace ext
{

   class ClusterThread : public SMPThread
   {

      friend class ClusterProcessor;

      private:
      unsigned int                     _clusterNode; // Assigned Cluster device Id

      std::queue<WD*> _blockedWDsSMP;
      std::queue<WD*> _blockedWDsGPU;

      Atomic<unsigned int> _numRunningSMP;
      Atomic<unsigned int> _completedSMPHead;
      Atomic<unsigned int> _completedSMPHead2;
      unsigned int _completedSMPTail;
      WD* _completedWDsGPU[MAX_PRESEND];
      Atomic<unsigned int> _numRunningGPU;
      Atomic<unsigned int> _completedGPUHead;
      Atomic<unsigned int> _completedGPUHead2;
      unsigned int _completedGPUTail;
      WD* _completedWDsSMP[MAX_PRESEND];

      // disable copy constructor and assignment operator
      ClusterThread( const ClusterThread &th );
      const ClusterThread & operator= ( const ClusterThread &th );

      public:
      // constructor
      ClusterThread( WD &w, PE *pe, SMPMultiThread *parent, int device ) : SMPThread( w, pe, parent ),
      _clusterNode( device ),
      _numRunningSMP (0) , 
      _completedSMPHead(0),
      _completedSMPHead2(0), 
      _completedSMPTail(0), 
      _numRunningGPU (0) ,
      _completedGPUHead(0), 
      _completedGPUHead2(0), 
      _completedGPUTail(0)
      {
         setCurrentWD( w );
         (void) w.getDirectory(true); 
         for ( int i = 0; i < MAX_PRESEND; i++ )
         {
            _completedWDsGPU[i]=NULL; 
            _completedWDsSMP[i]=NULL;
         }
      }

      // destructor
      virtual ~ClusterThread() {}

      virtual void runDependent ( void );
      virtual void inlineWorkDependent ( WD &wd ) { fatal( "inline execution is not supported in this architecture (cluster)."); }
      virtual void outlineWorkDependent ( WD &wd );

      void addRunningWDSMP( WorkDescriptor *wd );
      unsigned int numRunningWDsSMP();
      void clearCompletedWDsSMP2( );
      void completeWDSMP_2( void *remoteWdAddr );

      void addRunningWDGPU( WorkDescriptor *wd );
      unsigned int numRunningWDsGPU();
      void clearCompletedWDsGPU2( );
      void completeWDGPU_2( void *remoteWdAddr );

      virtual void join();
      virtual void start() {}
      virtual BaseThread * getNextThread ();

      void idle() { sys.getNetwork()->poll(getId()); };

      void addBlockingWDSMP( WD * wd );
      WD *fetchBlockingWDSMP();
      void addBlockingWDGPU( WD * wd );
      WD *fetchBlockingWDGPU();

      virtual void notifyOutlinedCompletionDependent( WD &completedWD ); 
   };


}
}

#endif
