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

#include <iostream>
#include "instrumentation.hpp"
#include "clusterthread.hpp"
#include "clusternode.hpp"
#include "clusterdevice.hpp"
#include "system.hpp"

using namespace nanos;
using namespace ext;


void ClusterThread::runDependent ()
{
   WD &work = getThreadWD();
   setCurrentWD( work );

   SMPDD &dd = ( SMPDD & ) work.activateDevice( SMP );

   dd.getWorkFct()( work.getData() );
}

void ClusterThread::inlineWorkDependent ( WD &wd )
{
   unsigned int i;
   SMPDD &dd = ( SMPDD & )wd.getActiveDevice();
   //ProcessingElement *pe = myThread->runningOn();
   if (dd.getWorkFct() == NULL ) return;

   wd.start(false);

   if ( wd.isClusterMigrable() )
   {
   
   ProcessingElement *pe = wd.getPe();
   //std::cerr << "run remote task, target pe: " << pe << " node num " << (unsigned int) ((ClusterNode *) pe)->getClusterNodeNum() << " numPe " << wd.getPeId() << " " << (void *) &wd << ":" << (unsigned int) wd.getId() << " WDprev " << wd.getPrevious() << ":" << wd.getPrevious()->getId() << std::endl;
   
   CopyData newCopies[ wd.getNumCopies() ]; 

   for (i = 0; i < wd.getNumCopies(); i += 1) {
       new ( &newCopies[i] ) CopyData( wd.getCopies()[i] );
   }

   //NANOS_INSTRUMENT ( static nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("user-code") );
   //NANOS_INSTRUMENT ( nanos_event_value_t val = wd.getId() );
   //NANOS_INSTRUMENT ( sys.getInstrumentation()->raiseOpenStateAndBurst ( NANOS_RUNNING, key, val ) );

   for (i = 0; i < wd.getNumCopies(); i += 1) {
      newCopies[i].setAddress( ( uint64_t ) pe->getAddress( wd, newCopies[i].getAddress(), newCopies[i].getSharing() ) );
    //  std::cerr << "copy " << i << " addr " << (void *) newCopies[i].getAddress() << std::endl;
   }

   char buff[ wd.getDataSize() + wd.getNumCopies() * sizeof( CopyData ) ];
   if ( wd.getDataSize() > 0 )
   {
      memcpy( &buff[ 0 ], wd.getData(), wd.getDataSize() );
   }
   for (i = 0; i < wd.getNumCopies(); i += 1) {
      memcpy( &buff[ wd.getDataSize() + sizeof( CopyData ) * i ], &newCopies[i], sizeof( CopyData ) );
   }

   ( ( ClusterNode * ) pe )->incExecutedWDs();
   sys.getNetwork()->sendWorkMsg( ( ( ClusterNode * ) pe )->getClusterNodeNum(), dd.getWorkFct(), wd.getDataSize(), wd.getId(), wd.getPeId(), wd.getDataSize() + ( wd.getNumCopies() * sizeof( CopyData ) ), buff );

   //WD *prefetchedWD = Scheduler::prefetch( this, wd );

   //if ( prefetchedWD != NULL ) 
   //{
   //   prefetchedWD->setPe( pe );
   //   prefetchedWD->setPeId( wd.getPeId() );
   //   if ( !prefetchedWD->started() )
   //      prefetchedWD->init();
   //}
   //wd.setPrefetchedWD( prefetchedWD );

   sys.getNetwork()->waitWorkCompletion( ( ( ClusterNode * ) pe )->getClusterNodeNum(), wd.getPeId() );
   //std::cerr << "finished remote call at node " << ((ClusterNode *)pe)->getClusterNodeNum() << std::endl;
   //std::cerr << "finished remote task, target pe: " << pe << " node num " << (unsigned int) ((ClusterNode *) pe)->getClusterNodeNum() << " numPe " << wd.getPeId() << " " << (void *) &wd << ":" << (unsigned int) wd.getId() << " WDprev " << wd.getPrevious() << ":" << wd.getPrevious()->getId() << std::endl;

   //NANOS_INSTRUMENT ( sys.getInstrumentation()->raiseCloseStateAndBurst ( key ) );
   }
   else
   {
      std::cerr << "i should execute the function!" << std::endl;
      ( dd.getWorkFct() )( wd.getData() );
   }
}

void ClusterThread::addWD( WorkDescriptor *wd )
{
   _myWDs.push_back( wd );
}
WorkDescriptor *ClusterThread::getWD( )
{
   WorkDescriptor * wd = _myWDs.pop_front( this );
   return wd;
}

void ClusterThread::join()
{
   unsigned int i;
   for ( i = 1; i < sys.getNetwork()->getNumNodes(); i++ )
      sys.getNetwork()->sendExitMsg( i );

   SMPThread::join();

}
