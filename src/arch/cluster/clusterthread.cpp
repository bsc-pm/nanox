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

#include "instrumentation.hpp"
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

   NANOS_INSTRUMENT ( static nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("user-code") );
   NANOS_INSTRUMENT ( nanos_event_value_t val = wd.getId() );
   NANOS_INSTRUMENT ( sys.getInstrumentation()->raiseOpenStateAndBurst ( NANOS_RUNNING, key, val ) );

   for (i = 0; i < wd.getNumCopies(); i += 1) {
      CopyData *cd = newCopies[i];
      cd->setAddress( ( uint64_t ) pe->getAddress( wd, cd->getAddress(), cd->getSharing() ) );
   }

   char *buff = new char[ wd.getDataSize() + wd.getNumCopies() * sizeof( CopyData ) ];
   if ( wd.getDataSize() > 0 )
   {
      memcpy( &buff[ 0 ], wd.getData(), wd.getDataSize() );
   }
   for (i = 0; i < wd.getNumCopies(); i += 1) {
      memcpy( &buff[ wd.getDataSize() + sizeof( CopyData ) * i ], newCopies[i], sizeof( CopyData ) );
   }

   sys.getNetwork()->sendWorkMsg( _clusterNode, dd.getWorkFct(), wd.getDataSize(), wd.getDataSize() + ( wd.getNumCopies() * sizeof( CopyData ) ), buff );

   for (i = 0; i < wd.getNumCopies(); i += 1) {
      delete newCopies[i];
   }
   delete buff;

   NANOS_INSTRUMENT ( sys.getInstrumentation()->raiseCloseStateAndBurst ( key ) );
}
