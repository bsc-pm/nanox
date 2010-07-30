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

#include "clusterthread.hpp"
#include "system.hpp"

using namespace nanos;
using namespace nanos::ext;


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
   ProcessingElement *pe = myThread->runningOn();
   CopyData *newCopies[wd.getNumCopies()]; 

   if (dd.getWorkFct() == NULL)
      fprintf(stderr, "ERROR, wd with NULL fct, DD addr is %p, wd %p\n", &dd, &wd);

   for (i = 0; i < wd.getNumCopies(); i += 1) {
      newCopies[i] = new CopyData( wd.getCopies()[i] );
   }
   //NANOS_INSTRUMENT ( static nanos_event_key_t key = sys.getInstrumentor()->getInstrumentorDictionary()->getEventKey("user-code") );
   //NANOS_INSTRUMENT ( nanos_event_value_t val = wd.getId() );
   //NANOS_INSTRUMENT ( sys.getInstrumentor()->raiseOpenStateAndBurst ( RUNNING, key, val ) );

   //fprintf(stderr, "wd = %p, fct %p\n", dd, dd.getWorkFct());

   //fprintf(stderr, "host: num copies is %d\n", wd.getNumCopies());
   //fprintf(stderr, "host: addr is %llx\n", wd.getCopies()[0].getAddress());
   //fprintf(stderr, "host: is input? %s\n", wd.getCopies()[0].isInput() ? "yes" : "no");
   //fprintf(stderr, "host: is output? %s\n", wd.getCopies()[0].isOutput() ? "yes" : "no");

   for (i = 0; i < wd.getNumCopies(); i += 1) {
      CopyData *cd = newCopies[i];
      //fprintf(stderr, "addr is %llx\n", cd->getAddress());
      cd->setAddress( ( uint64_t ) pe->getAddress( wd, cd->getAddress(), cd->getSharing() ) );
      //fprintf(stderr, "new addr is %llx\n", cd->getAddress());
   }

   char *buff = new char[ wd.getDataSize() + wd.getNumCopies() * sizeof( CopyData ) ];
   if ( wd.getDataSize() > 0 )
      memcpy( &buff[ 0 ], wd.getData(), wd.getDataSize() );
   for (i = 0; i < wd.getNumCopies(); i += 1) {
      memcpy( &buff[ wd.getDataSize() + sizeof( CopyData ) * i ], newCopies[i], sizeof( CopyData ) );
   }

   //fprintf(stderr, "I have to SEND WORK to node %d\n", _clusterNode);
   // fprintf(stderr, "NUM COPIES %d addr %llx, in? %s, out? %s\n",
   //       wd.getNumCopies(),
   //       ((CopyData *) &buff[ wd.getDataSize() ])->getAddress(),
   //       ((CopyData *) &buff[ wd.getDataSize() ])->isInput() ? "yes" : "no",
   //       ((CopyData *) &buff[ wd.getDataSize() ])->isOutput() ? "yes" : "no" );
   sys.getNetwork()->sendWorkMsg( _clusterNode, dd.getWorkFct(), wd.getDataSize(), wd.getDataSize() + ( wd.getNumCopies() * sizeof( CopyData ) ), buff );
   //( dd.getWorkFct() )( wd.getData() );

   for (i = 0; i < wd.getNumCopies(); i += 1) {
      delete newCopies[i];
   }
   delete buff;

   //NANOS_INSTRUMENT ( sys.getInstrumentor()->raiseCloseStateAndBurst ( key ) );
}
