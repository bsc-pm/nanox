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

#include <iostream>
#include "instrumentation.hpp"
#include "clusterthread_decl.hpp"
#include "clusternode_decl.hpp"
#include "system_decl.hpp"
#include "workdescriptor_decl.hpp"
#include "basethread.hpp"
#include "smpthread.hpp"
#include "netwd_decl.hpp"
#ifdef OpenCL_DEV
#include "opencldd.hpp"
#endif
#ifdef FPGA_DEV
#include "fpgadd.hpp"
#endif

using namespace nanos;
using namespace ext;

ClusterThread::RunningWDQueue::RunningWDQueue() : _numRunning(0), _completedHead(0), _completedHead2(0), _completedTail(0), _waitingDataWDs(), _pendingInitWD( NULL ) {
   for ( unsigned int i = 0; i < MAX_PRESEND; i++ )
   {
      _completedWDs[i] = NULL;
   }
}

ClusterThread::RunningWDQueue::~RunningWDQueue() {
}

void ClusterThread::RunningWDQueue::addRunningWD( WorkDescriptor *wd ) {
   _numRunning++;
}

unsigned int ClusterThread::RunningWDQueue::numRunningWDs() const {
   return _numRunning.value();
}

void ClusterThread::RunningWDQueue::clearCompletedWDs( ClusterThread *self ) {
   unsigned int lowval = _completedTail % MAX_PRESEND;
   unsigned int highval = ( _completedHead2.value() ) % MAX_PRESEND;
   unsigned int pos = lowval;
   if ( lowval > highval ) highval +=MAX_PRESEND;
   while ( lowval < highval )
   {
      WD *completedWD = _completedWDs[pos];
      Scheduler::postOutlineWork( completedWD, false, self );
      delete[] (char *) completedWD;
      _completedWDs[pos] =(WD *) 0xdeadbeef;
      pos = (pos+1) % MAX_PRESEND;
      lowval += 1;
      _completedTail += 1;
   }
}

void ClusterThread::RunningWDQueue::completeWD( void *remoteWdAddr ) {
   unsigned int realpos = _completedHead++;
   unsigned int pos = realpos %MAX_PRESEND;
   _completedWDs[pos] = (WD *) remoteWdAddr;
   while( !_completedHead2.cswap( realpos, realpos+1) ) {}
   ensure( _numRunning.value() > 0, "invalid value");
   _numRunning--;
}

ClusterThread::ClusterThread( WD &w, PE *pe, SMPMultiThread *parent, int device )
   : BaseThread( (unsigned int) -1, w, pe, parent ), _clusterNode( device ), _lock() {
   setCurrentWD( w );
}

ClusterThread::~ClusterThread() {
}

void ClusterThread::runDependent () {
   WD &work = getThreadWD();
   setCurrentWD( work );

   SMPDD &dd = ( SMPDD & ) work.activateDevice( getSMPDevice() );

   dd.getWorkFct()( work.getData() );
}


bool ClusterThread::inlineWorkDependent ( WD &wd ) {
   fatal( "inline execution is not supported in this architecture (cluster).");
   return true;
}

void ClusterThread::preOutlineWorkDependent ( WD &wd ) {
   wd.preStart(WorkDescriptor::IsNotAUserLevelThread);
}

void ClusterThread::outlineWorkDependent ( WD &wd )
{
   SMPDD &dd = ( SMPDD & )wd.getActiveDevice();
   ProcessingElement *pe = this->runningOn();
   if (dd.getWorkFct() == NULL ) return;

   //wd.getGE()->setNode( ( ( ClusterNode * ) pe )->getClusterNodeNum() );

#if 0
   unsigned int totalDimensions = 0;
   for (i = 0; i < wd.getNumCopies(); i += 1) {
      totalDimensions += wd.getCopies()[i].getNumDimensions();
   }

   size_t totalBufferSize = wd.getDataSize() + 
      sizeof(int) + wd.getNumCopies() * sizeof( CopyData ) + 
      sizeof(int) + totalDimensions * sizeof( nanos_region_dimension_t ) + 
      sizeof(int) + wd.getNumCopies() * sizeof( uint64_t );

   char *buff = new char[ totalBufferSize ];


   // Copy WD data to tmp buffer
   if ( wd.getDataSize() > 0 )
   {
      memcpy( &buff[ 0 ], wd.getData(), wd.getDataSize() );
   }

   *((int *) &buff[ wd.getDataSize() ] ) = wd.getNumCopies();
   *((int *) &buff[ wd.getDataSize() + sizeof( int ) + wd.getNumCopies() * sizeof( CopyData ) ] ) = totalDimensions;

   CopyData *newCopies = ( CopyData * ) ( buff + wd.getDataSize() + sizeof( int ) );
   nanos_region_dimension_internal_t *dimensions = ( nanos_region_dimension_internal_t * ) ( buff + wd.getDataSize() + sizeof( int ) + wd.getNumCopies() * sizeof( CopyData ) + sizeof( int ) );
   
   uintptr_t dimensionIndex = 0;
   for (i = 0; i < wd.getNumCopies(); i += 1) {
      new ( &newCopies[i] ) CopyData( wd.getCopies()[i] );
      memcpy( &dimensions[ dimensionIndex ], wd.getCopies()[i].getDimensions(), sizeof( nanos_region_dimension_internal_t ) * wd.getCopies()[i].getNumDimensions());
      newCopies[i].setDimensions( ( nanos_region_dimension_internal_t *  ) dimensionIndex ); // This is the index because it makes no sense to send an address over the network
      newCopies[i].setHostBaseAddress( (uint64_t) wd.getCopies()[i].getBaseAddress() );
      newCopies[i].setRemoteHost( true );
      //newCopies[i].setBaseAddress( (void *) ( wd._ccontrol.getAddress( i ) - wd.getCopies()[i].getOffset() ) );
      newCopies[i].setBaseAddress( (void *) wd._mcontrol.getAddress( i ) );
      newCopies[i].setHostRegionId( wd._mcontrol._memCacheCopies[i]._reg.id );
      dimensionIndex += wd.getCopies()[i].getNumDimensions();
   }
#endif

#if 0

   int arch = -1;
   if ( wd.canRunIn( getSMPDevice() ) ) {
      arch = 0;
   }
#ifdef GPU_DEV
   else if (wd.canRunIn( GPU) )
   {
      arch = 1;
   }
#endif
#ifdef OpenCL_DEV
   else if (wd.canRunIn( OpenCLDev ) )
   {
      arch = 2;
   }
#endif
#ifdef FPGA_DEV
   else if (wd.canRunIn( FPGA ) )
   {
      arch = 3;
   }
#endif
   else {
      fatal("unsupported architecture");
   }
   #endif

   //std::cerr << "run remote task, target pe: " << pe << " node num " << (unsigned int) ((ClusterNode *) pe)->getClusterNodeNum() << " arch: "<< arch << " " << (void *) &wd << ":" << (unsigned int) wd.getId() << " data size is " << wd.getDataSize() << " copies " << wd.getNumCopies() << " dimensions " << dimensionIndex << std::endl;

   ( ( ClusterNode * ) pe )->incExecutedWDs();
   sys.getNetwork()->sendWorkMsg( ( ( ClusterNode * ) pe )->getClusterNodeNum(), wd );
   //sys.getNetwork()->sendWorkMsg( ( ( ClusterNode * ) pe )->getClusterNodeNum(), dd.getWorkFct(), wd.getDataSize(), wd.getId(), /* this should be the PE id */ arch, nwd.getBufferSize(), nwd.getBuffer(), wd.getTranslateArgs(), arch, (void *) &wd );

}

void ClusterThread::join() {
   message( "Node " << ( ( ClusterNode * ) this->runningOn() )->getClusterNodeNum() << " executed " <<( ( ClusterNode * ) this->runningOn() )->getExecutedWDs() << " WDs" );
   sys.getNetwork()->sendExitMsg( _clusterNode );
}

void ClusterThread::start() {
}

BaseThread * ClusterThread::getNextThread ()
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

void ClusterThread::notifyOutlinedCompletionDependent( WD *completedWD ) {
   int arch = -1;
   if ( completedWD->canRunIn( getSMPDevice() ) )
   {
      arch = 0;
   }
#ifdef GPU_DEV
   else if ( completedWD->canRunIn( GPU ) )
   {
      arch = 1;
   }
#endif
#ifdef OpenCL_DEV
   else if ( completedWD->canRunIn( OpenCLDev ) )
   {
      arch = 2;
   }
#endif
#ifdef FPGA_DEV
   else if ( completedWD->canRunIn( FPGA ) )
   {
      arch = 3;
   }
#endif
   else {
      fatal("Unsupported architecture");
   }
   _runningWDs[ arch ].completeWD( completedWD );
}
void ClusterThread::addRunningWD( unsigned int archId, WorkDescriptor *wd ) {
   _runningWDs[archId].addRunningWD( wd );
}
unsigned int ClusterThread::numRunningWDs( unsigned int archId ) const {
   return _runningWDs[archId].numRunningWDs();
}
void ClusterThread::clearCompletedWDs( unsigned int archId ) {
   _runningWDs[archId].clearCompletedWDs( this );
}
bool ClusterThread::acceptsWDs( unsigned int archId ) const {
   unsigned int presend_setting = 0;
   switch (archId) {
      case 0: //SMP
         presend_setting = sys.getNetwork()->getSmpPresend();
         break;
      case 1: //GPU
         presend_setting = sys.getNetwork()->getGpuPresend();
         break;
      case 2: //OCL
         presend_setting = sys.getNetwork()->getGpuPresend(); //FIXME
         break;
      case 3: //FPGA
         presend_setting = sys.getNetwork()->getGpuPresend(); //FIXME
         break;
      default:
         fatal("Impossible path");
         break;
   }
   return ( numRunningWDs(archId) < presend_setting );
}

void ClusterThread::idle( bool debug )
{
   // poll the network as the parent thread
   BaseThread *orig_myThread = myThread;
   BaseThread *parent = myThread->getParent();
   myThread = parent;
   sys.getNetwork()->poll(0);
   myThread = orig_myThread;

   if ( !_pendingRequests.empty() ) {
      std::set<void *>::iterator it = _pendingRequests.begin();
      while ( it != _pendingRequests.end() ) {
         GetRequest *req = (GetRequest *) (*it);
         if ( req->isCompleted() ) {
           std::set<void *>::iterator toBeDeletedIt = it;
           it++;
           _pendingRequests.erase(toBeDeletedIt);
           req->clear();
           delete req;
         } else {
            it++;
         }
      }
   }
}

bool ClusterThread::isCluster() {
   return true;
}

void ClusterThread::switchTo( WD *work, SchedulerHelper *helper ) {}
void ClusterThread::exitTo( WD *work, SchedulerHelper *helper ) {}
void ClusterThread::switchHelperDependent( WD* oldWD, WD* newWD, void *arg ) {}
void ClusterThread::exitHelperDependent( WD* oldWD, WD* newWD, void *arg ) {}
void ClusterThread::initializeDependent( void ) {}
void ClusterThread::switchToNextThread() {}

void ClusterThread::lock() {
   _lock.acquire();
}

void ClusterThread::unlock() {
   _lock.release();
}

bool ClusterThread::tryLock() {
   return _lock.tryAcquire();
}

bool ClusterThread::hasAPendingWDToInit( unsigned int arch_id ) const {
   return _runningWDs[arch_id].hasAPendingWDToInit();
}

bool ClusterThread::RunningWDQueue::hasAPendingWDToInit() const {
   return _pendingInitWD != NULL;
}

WD *ClusterThread::getPendingInitWD( unsigned int arch_id ) {
   return _runningWDs[arch_id].getPendingInitWD();
}

WD *ClusterThread::RunningWDQueue::getPendingInitWD() {
   WD *wd = _pendingInitWD;
   _pendingInitWD = NULL;
   return wd;
}

void ClusterThread::setPendingInitWD( unsigned int arch_id, WD *wd ) {
   _runningWDs[arch_id].setPendingInitWD( wd );
}

void ClusterThread::RunningWDQueue::setPendingInitWD( WD *wd ) {
   _pendingInitWD = wd;
}

bool ClusterThread::RunningWDQueue::hasWaitingDataWDs() const {
   return !_waitingDataWDs.empty();
}

bool ClusterThread::hasWaitingDataWDs( unsigned int archId ) const {
   return _runningWDs[archId].hasWaitingDataWDs();
}
WD* ClusterThread::getWaitingDataWD( unsigned int archId ) {
   return _runningWDs[archId].getWaitingDataWD();
}

WD *ClusterThread::RunningWDQueue::getWaitingDataWD() {
   WD *wd = _waitingDataWDs.front();
   _waitingDataWDs.pop_front();
//std::cerr << "popped a wd ( " << wd << " )" << wd->getId() << ", count is " << _waitingDataWDs.size() << std::endl;
   return wd;
}

void ClusterThread::addWaitingDataWD( unsigned int archId, WD *wd ) {
   _runningWDs[archId].addWaitingDataWD( wd );
}

void ClusterThread::RunningWDQueue::addWaitingDataWD( WD *wd ) {
   _waitingDataWDs.push_back( wd );
//std::cerr << "Added a wd ( " << wd << " )" << wd->getId() << ", count is " << _waitingDataWDs.size() << std::endl;
}

void ClusterThread::setupSignalHandlers() {
   std::cerr << __FUNCTION__ << ": unimplemented in ClusterThread." << std::endl;
}


WD * ClusterThread::getClusterWD( BaseThread *thread )
{
   WD * wd = NULL;
   if ( thread->getTeam() != NULL ) {
      wd = thread->getNextWD();
      if ( wd ) {
         if ( !thread->runningOn()->canRun( *wd ) )
         { // found a non compatible wd in "nextWD", ignore it
            wd = thread->getTeam()->getSchedulePolicy().atIdle ( thread, 0 );
            //if(wd!=NULL)std::cerr << "GN got a wd with depth " <<wd->getDepth() << std::endl;
         } else {
            //thread->resetNextWD();
           // std::cerr << "FIXME" << std::endl;
         }
      } else {
         wd = thread->getTeam()->getSchedulePolicy().atIdle ( thread, 0 );
         //if(wd!=NULL)std::cerr << "got a wd with depth " <<wd->getDepth() << std::endl;
      }
   }
   return wd;
}

void ClusterThread::workerClusterLoop ()
{
   BaseThread *parent = myThread;
   BaseThread *current_thread = ( myThread = myThread->getNextThread() );

   for ( ; ; ) {
      if ( !parent->isRunning() ) break;

      if ( parent != current_thread ) // if parent == myThread, then there are no "soft" threads and just do nothing but polling.
      {
         ClusterThread *myClusterThread = ( ClusterThread * ) current_thread;
         if ( myClusterThread->tryLock() ) {
            ClusterNode *thisNode = ( ClusterNode * ) current_thread->runningOn();

            ClusterNode::ClusterSupportedArchMap const &archs = thisNode->getSupportedArchs();
            for ( ClusterNode::ClusterSupportedArchMap::const_iterator it = archs.begin();
                  it != archs.end(); it++ ) {
               unsigned int arch_id = it->first;
               thisNode->setActiveDevice( it->second );
               myClusterThread->clearCompletedWDs( arch_id );
               if ( myClusterThread->hasWaitingDataWDs( arch_id ) ) {
                  WD * wd_waiting = myClusterThread->getWaitingDataWD( arch_id );
                  if ( wd_waiting->isInputDataReady() ) {
                     myClusterThread->addRunningWD( arch_id, wd_waiting );
                     Scheduler::outlineWork( current_thread, wd_waiting );
                  } else {
                     myClusterThread->addWaitingDataWD( arch_id, wd_waiting );


                     // Try to get a WD normally, this is needed because otherwise we will keep only checking the WaitingData WDs
                     if ( myClusterThread->hasAPendingWDToInit( arch_id ) ) {
                        WD * wd = myClusterThread->getPendingInitWD( arch_id );
                        if ( Scheduler::tryPreOutlineWork(wd) ) {
                           current_thread->preOutlineWorkDependent( *wd );
                           //std::cerr << "GOT A PENDIGN WD for thd " << current_thread->getId() <<" wd is " << wd->getId() << std::endl;
                           if ( wd->isInputDataReady() ) {
                              myClusterThread->addRunningWD( arch_id, wd );
                              //NANOS_INSTRUMENT( InstrumentState inst2(NANOS_OUTLINE_WORK, true); );
                              Scheduler::outlineWork( current_thread, wd );
                              //NANOS_INSTRUMENT( inst2.close(); );
                           } else {
                              myClusterThread->addWaitingDataWD( arch_id, wd );
                           }
                        } else {
                           //std::cerr << "REPEND WD for thd " << current_thread->getId() <<" wd is " << wd->getId() << std::endl;
                           myClusterThread->setPendingInitWD( arch_id, wd );
                        }
                     } else {
                        if ( myClusterThread->acceptsWDs( arch_id ) )
                        {
                           WD * wd = getClusterWD( current_thread );
                           if ( wd )
                           {
                              Scheduler::prePreOutlineWork(wd);
                              if ( Scheduler::tryPreOutlineWork(wd) ) {
                                 current_thread->preOutlineWorkDependent( *wd );
                                 if ( wd->isInputDataReady() ) {
                                    //std::cerr << "SUCCED WD for thd " << current_thread->getId() <<" wd is " << wd->getId() << std::endl;
                                    myClusterThread->addRunningWD( arch_id, wd );
                                    //NANOS_INSTRUMENT( InstrumentState inst2(NANOS_OUTLINE_WORK, true); );
                                    Scheduler::outlineWork( current_thread, wd );
                                    //NANOS_INSTRUMENT( inst2.close(); );
                                 } else {
                                    myClusterThread->addWaitingDataWD( arch_id, wd );
                                 }
                              } else {
                                 //std::cerr << "ADDED A PENDIGN WD for thd " << current_thread->getId() <<" wd is " << wd->getId() << std::endl;
                                 myClusterThread->setPendingInitWD( arch_id, wd );
                              }
                           }
                        }// else { std::cerr << "Max presend reached "<<myClusterThread->getId()  << std::endl; }
                     }
                  }
               } else {
                  if ( myClusterThread->hasAPendingWDToInit( arch_id ) ) {
                     WD * wd = myClusterThread->getPendingInitWD( arch_id );
                     if ( Scheduler::tryPreOutlineWork(wd) ) {
                        current_thread->preOutlineWorkDependent( *wd );
                        //std::cerr << "GOT A PENDIGN WD for thd " << current_thread->getId() <<" wd is " << wd->getId() << std::endl;
                        if ( wd->isInputDataReady() ) {
                           myClusterThread->addRunningWD( arch_id, wd );
                           //NANOS_INSTRUMENT( InstrumentState inst2(NANOS_OUTLINE_WORK, true); );
                           Scheduler::outlineWork( current_thread, wd );
                           //NANOS_INSTRUMENT( inst2.close(); );
                        } else {
                           myClusterThread->addWaitingDataWD( arch_id, wd );
                        }
                     } else {
                        //std::cerr << "REPEND WD for thd " << current_thread->getId() <<" wd is " << wd->getId() << std::endl;
                        myClusterThread->setPendingInitWD( arch_id, wd );
                     }
                  } else {
                     if ( myClusterThread->acceptsWDs( arch_id ) )
                     {
                        WD * wd = getClusterWD( current_thread );
                        if ( wd )
                        {
                           Scheduler::prePreOutlineWork(wd);
                           if ( Scheduler::tryPreOutlineWork(wd) ) {
                              current_thread->preOutlineWorkDependent( *wd );
                              if ( wd->isInputDataReady() ) {
                                 //std::cerr << "SUCCED WD for thd " << current_thread->getId() <<" wd is " << wd->getId() << std::endl;
                                 myClusterThread->addRunningWD( arch_id, wd );
                                 //NANOS_INSTRUMENT( InstrumentState inst2(NANOS_OUTLINE_WORK, true); );
                                 Scheduler::outlineWork( current_thread, wd );
                                 //NANOS_INSTRUMENT( inst2.close(); );
                              } else {
                                 myClusterThread->addWaitingDataWD( arch_id, wd );
                              }
                           } else {
                              //std::cerr << "ADDED A PENDIGN WD for thd " << current_thread->getId() <<" wd is " << wd->getId() << std::endl;
                              myClusterThread->setPendingInitWD( arch_id, wd );
                           }
                        }
                     }// else { std::cerr << "Max presend reached "<<myClusterThread->getId()  << std::endl; }
                  }
               }
            }
            myClusterThread->unlock();
         }
      }
      //sys.getNetwork()->poll(parent->getId());
      myThread->processTransfers();
      current_thread = ( myThread = myThread->getNextThread() );
   }

   SMPMultiThread *parentM = ( SMPMultiThread * ) parent;
   for ( unsigned int i = 0; i < parentM->getNumThreads(); i += 1 ) {
      myThread = parentM->getThreadVector()[ i ];
      myThread->leaveTeam();
      myThread->joined();
   }
   myThread = parent;
}
