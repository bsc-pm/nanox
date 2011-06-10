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
#include <queue>


namespace nanos {
namespace ext
{

   class ClusterThread : public SMPThread
   {

      friend class ClusterProcessor;

      private:
      unsigned int                     _clusterNode; // Assigned Cluster device Id
      //WD*                          _myWDs[2];
      std::queue<WD*> _blockedWDsSMP;
      //std::map<int, WD*> _runningWDsSMP;
      //Lock _qgpuLock;
      std::queue<WD*> _blockedWDsGPU;
      //std::map<int, WD*> _runningWDsGPU;
      Atomic<unsigned int> _numRunningSMP;
      Atomic<unsigned int> _completedSMPHead;
      Atomic<unsigned int> _completedSMPHead2;
      unsigned int _completedSMPTail;
      WD* _completedWDsGPU[16];
      Atomic<unsigned int> _numRunningGPU;
      Atomic<unsigned int> _completedGPUHead;
      Atomic<unsigned int> _completedGPUHead2;
      unsigned int _completedGPUTail;
      WD* _completedWDsSMP[16];
      //WD* preoutlinedWD;
      //WD* _blockingWDSMP;
      //WD* _blockingWDGPU;

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
      //preoutlinedWD(NULL),
      //_blockingWDSMP(NULL),
      //_blockingWDGPU(NULL)
      {
         //preoutlinedWD = NULL;
         setCurrentWD( w );
         (void) w.getDirectory(true); 
         //_myWDs[0] = NULL; 
         //_myWDs[1] = NULL;
         for (int i=0;i<16;i++){
            _completedWDsGPU[i]=NULL; 
            _completedWDsSMP[i]=NULL;
         }
      }

      // destructor
      virtual ~ClusterThread() {}

      virtual void runDependent ( void );
      virtual void inlineWorkDependent ( WD &wd ) { fatal( "inline execution is not supported in this architecture (cluster)."); }
      virtual void outlineWorkDependent ( WD &wd );

      //void addRunningWD( WorkDescriptor *wd , int numPe) { _myWDs[numPe] = wd; } ;
      //WD * getRunningWD( int numPe) { return _myWDs[numPe]; } ;
      //void clearRunningWD(int numPe) { _myWDs[numPe] = NULL; } ;

      //void addRunningWDSMP( WorkDescriptor *wd ) { _runningWDsSMP[ wd->getId() ] = wd; }
      //unsigned int numRunningWDsSMP() { return _runningWDsSMP.size(); }
      //void completeWDSMP( int id ) { _completedWDsSMP.push( _runningWDsSMP[ id ] ); _runningWDsSMP.erase( id ); } 
      //bool areThereCompletedWDsSMP( ) {  return !_completedWDsSMP.empty(); } 
      //WD * fetchCompletedWDSMP() { WD* retWD =  _completedWDsSMP.front(); _completedWDsSMP.pop(); return retWD; }

      void addRunningWDSMP( WorkDescriptor *wd ) { 
         _numRunningSMP++;
      }
      unsigned int numRunningWDsSMP() {
         return _numRunningSMP.value();
         /* unsigned int res ;
            _qgpuLock.acquire();
            res = _runningWDsSMP.size();
            _qgpuLock.release();
            return res; */
      }
      void clearCompletedWDsSMP2( ) {
         unsigned int lowval = _completedSMPTail % 16;
         unsigned int highval = ( _completedSMPHead2.value() ) % 16;
         unsigned int pos = lowval;
         if ( lowval > highval ) highval +=16;
         //if (lowval < highval) std::cerr << "thd: "<< getId() << "clear wd from " << lowval << " to " << highval << std::endl;
         while ( lowval < highval )
         {
            WD *completedWD = _completedWDsSMP[pos];
            Scheduler::postOutlineWork( completedWD, false, this );
            delete completedWD;
            _completedWDsSMP[pos] =(WD *) 0xdeadbeef;
            pos = (pos+1) % 16;
            lowval += 1;
            _completedSMPTail += 1;
         }
      }
      void completeWDSMP_2( void *remoteWdAddr ) {
         unsigned int realpos = _completedSMPHead++;
         _numRunningSMP--;
         //if (pos == 16) pos = 0; 
         unsigned int pos = realpos %16;
         _completedWDsSMP[pos] = (WD *) remoteWdAddr;
         //WD * completedWD = (WD *) remoteWdAddr;
         while( !_completedSMPHead2.cswap( realpos, realpos+1) ) {}
      }

      void addRunningWDGPU( WorkDescriptor *wd ) { 
         _numRunningGPU++;
         /* _qgpuLock.acquire();
            _runningWDsGPU[ wd->getId() ] = wd;
            _qgpuLock.release();*/
      }
      unsigned int numRunningWDsGPU() {
         return _numRunningGPU.value();
         /* unsigned int res ;
            _qgpuLock.acquire();
            res = _runningWDsGPU.size();
            _qgpuLock.release();
            return res; */
      }

      //void completeWDGPU( void * id ) { 
      //        intptr_t tmp = (intptr_t) id;
      //	/*_numRunningGPU--; */
      //	_qgpuLock.acquire(); 
      //	_completedWDsGPU.push( _runningWDsGPU[ (unsigned int) tmp ] ); 
      //	_runningWDsGPU.erase( (unsigned int ) tmp ); 
      //	_qgpuLock.release(); 
      //} 
      // WD * areThereCompletedWDsGPU( ) { WD *ret = NULL; if ( !_completedWDsGPU.empty() ) { _qgpuLock.acquire(); if ( !_completedWDsGPU.empty() ) { ret = _completedWDsGPU.front(); _completedWDsGPU.pop(); } _qgpuLock.release(); } return ret; } 
      void clearCompletedWDsGPU2( ) {
         unsigned int lowval = _completedGPUTail % 16;
         unsigned int highval = ( _completedGPUHead2.value() ) % 16;
         unsigned int pos = lowval;
         if ( lowval > highval ) highval +=16;
         //if (lowval < highval) std::cerr << "thd: "<< getId() << "clear wd from " << lowval << " to " << highval << std::endl;
         while ( lowval < highval )
         {
            WD *completedWD = _completedWDsGPU[pos];
            Scheduler::postOutlineWork( completedWD, false, this );
            delete completedWD;
            _completedWDsGPU[pos] =(WD *) 0xdeadbeef;
            pos = (pos+1) % 16;
            lowval += 1;
            _completedGPUTail += 1;
         }
      }
      //WD * fetchCompletedWDGPU() { _qgpuLock.acquire(); WD* retWD = _completedWDsGPU.front(); _completedWDsGPU.pop(); _qgpuLock.release(); return retWD; }

      void completeWDGPU_2( void *remoteWdAddr ) {
         unsigned int realpos = _completedGPUHead++;
         _numRunningGPU--;
         //if (pos == 16) pos = 0; 
         unsigned int pos = realpos %16;
         _completedWDsGPU[pos] = (WD *) remoteWdAddr;
         //WD * completedWD = (WD *) remoteWdAddr;
         while( !_completedGPUHead2.cswap( realpos, realpos+1) ) {}
      }

      //WorkDescriptor *getWD();
      virtual void join();
      virtual void start() {}
      virtual BaseThread * getNextThread ()
      {
         BaseThread *next;
         if ( getParent() != NULL )
         {
            next = getParent()->getNextThread();
         }
         else
         {
            next = this;
         }
         return next;
      }
      //WD* getPreOutlinedWD() {return preoutlinedWD; }
      //void setPreOutlinedWD( WD * wd) { preoutlinedWD = wd; }
      //void unsetPreOutlinedWD() {preoutlinedWD = NULL; }

      //virtual int checkStateDependent( int numPe );
      void idle() { sys.getNetwork()->poll(getId()); };

      //WD* getBlockingWDSMP() {return _blockingWDSMP; }
      //void setBlockingWDSMP( WD * wd) { _blockingWDSMP = wd; }
      //WD* getBlockingWDGPU() {return _blockingWDGPU; }
      //void setBlockingWDGPU( WD * wd) { _blockingWDGPU = wd; }

      void addBlockingWDSMP( WD * wd ) { _blockedWDsSMP.push(wd);  }
      WD *fetchBlockingWDSMP() { WD *wd = NULL; if ( !_blockedWDsSMP.empty() ) { wd = _blockedWDsSMP.front(); _blockedWDsSMP.pop(); } return wd; }
      void addBlockingWDGPU( WD * wd ) { _blockedWDsGPU.push(wd);  }
      WD *fetchBlockingWDGPU() { WD *wd = NULL; if ( !_blockedWDsGPU.empty() ) { wd = _blockedWDsGPU.front(); _blockedWDsGPU.pop(); } return wd; }

   };


}
}

#endif
