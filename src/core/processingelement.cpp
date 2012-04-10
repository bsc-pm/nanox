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

#include <string.h>
#include "processingelement.hpp"
#include "basethread.hpp"
#include "debug.hpp"
#include "schedule.hpp"
#include "copydata.hpp"
#include "system.hpp"
#include "instrumentation.hpp"
#include "directory.hpp"
#include "regiondirectory.hpp"

using namespace nanos;

bool ProcessingElement::dataCanBlockUs( WorkDescriptor& wd )
{
   return false;
}

void ProcessingElement::copyDataIn( WorkDescriptor &work )
{
   //Directory *dir = work.getParent()->getDirectory(true);
   //if ( dir != NULL ) {
   //   CopyData *copies = work.getCopies();
   //   for ( unsigned int i = 0; i < work.getNumCopies(); i++ ) {
   //      CopyData & cd = copies[i];
   //      if ( !cd.isPrivate() ) {
   //           dir->registerAccess( cd.getAddress(), cd.getSize(), cd.isInput(), cd.isOutput() );
   //      }
   //   }
   //}
   NewDirectory *dir = work.getNewDirectory();
   CopyData *copies = work.getCopies();
   for ( unsigned int index = 0; index < work.getNumCopies(); index++ ) {
      NewDirectory::LocationInfoList locations;
      Region reg = NewDirectory::build_region( copies[ index ] );
      dir->lock();
      dir->registerAccess( reg, copies[ index ].isInput(), copies[ index ].isOutput(), 0, ((uint64_t)copies[ index ].getBaseAddress()) + copies[ index ].getOffset(), locations );
      dir->unlock();
      if ( !copies[ index ].isInput() ) continue;

      {
         std::map<unsigned int, std::list<Region> > locationMap;

         for ( NewDirectory::LocationInfoList::iterator it = locations.begin(); it != locations.end(); it++ ) {
            if (!it->second.isLocatedIn( 0 ) ) { 
               int loc = it->second.getFirstLocation();
               locationMap[ loc ].push_back( it->first );
               //std::cerr << "Houston, we have a problem, data is not in Host and we need it back. HostAddr: " << (void *) (((it->first)).getFirstValue()) << it->second << std::endl;
            }
            //else { if ( sys.getNetwork()->getNodeNum() == 0) std::cerr << "["<<sys.getNetwork()->getNodeNum()<<"] wd " << work.getId() << "All ok, location is " << *(it->second) << std::endl; }
         }

         std::map<unsigned int, std::list<Region> >::iterator locIt;
         for( locIt = locationMap.begin(); locIt != locationMap.end(); locIt++ ) {
            sys.getCaches()[ locIt->first ]->syncRegion( locIt->second );
         }
      }
   }
}

void ProcessingElement::copyDataOut( WorkDescriptor &work )
{
#if 0
   Directory *dir = work.getParent()->getDirectory(false);
   if ( dir != NULL ) {
      CopyData *copies = work.getCopies();
      for ( unsigned int i = 0; i < work.getNumCopies(); i++ ) {
         CopyData & cd = copies[i];
         if ( !cd.isPrivate() ) {
            dir->unRegisterAccess( cd.getAddress(), cd.isOutput(), work.getDirectory(false) );
            if ( cd.isOutput() ) {
               Directory *sons = work.getDirectory(false);
               if ( sons!=NULL ) {
                  dir->updateCurrentDirectory( cd.getAddress(), *sons );
               }
            }
         }
      }
      //if ( sys.getNetwork()->getNodeNum() > 0 ) { 
      //unsigned int wo_copies = 0;
      //for ( unsigned int i = 0; i < work.getNumCopies(); i++ ) {
      // wo_copies += ( copies[i].isOutput() && !copies[i].isInput() );
      //}
      //if ( wo_copies == work.getNumCopies() )
      //{

      //for ( unsigned int i = 0; i < work.getNumCopies(); i++ ) {
      //   CopyData & cd = copies[i];
      //        dir->fwAccess( cd.getAddress(), cd.getSize(), cd.isInput(), cd.isOutput() );
      //}
      //  
      //}
      //}
   }
#endif
}

void ProcessingElement::waitInputs( WorkDescriptor &work )
{
   //Directory *dir = work.getParent()->getDirectory(false);
   //if ( dir != NULL ) {
   //   CopyData *copies = work.getCopies();
   //   for ( unsigned int i = 0; i < work.getNumCopies(); i++ ) {
   //      CopyData & cd = copies[i];
   //      if ( !cd.isPrivate() && cd.isInput() ) {
   //           dir->waitInput( cd.getAddress(), cd.isOutput() );
   //      }
   //   }
   //}
}

BaseThread& ProcessingElement::startWorker ( ext::SMPMultiThread *parent )
{
   WD & worker = getWorkerWD();

   NANOS_INSTRUMENT (sys.getInstrumentation()->raiseOpenPtPEventNkvs ( NANOS_WD_DOMAIN, (nanos_event_id_t) worker.getId(), 0, NULL, NULL ); )
   NANOS_INSTRUMENT (InstrumentationContextData *icd = worker.getInstrumentationContextData() );
   NANOS_INSTRUMENT (icd->setStartingWD(true) );

   return startThread( worker, parent );
}

BaseThread& ProcessingElement::startMultiWorker ( unsigned int numPEs, ProcessingElement **repPEs )
{
   WD & worker = getMultiWorkerWD();

   NANOS_INSTRUMENT (sys.getInstrumentation()->raiseOpenPtPEventNkvs ( NANOS_WD_DOMAIN, (nanos_event_id_t) worker.getId(), 0, NULL, NULL ); )
   NANOS_INSTRUMENT (InstrumentationContextData *icd = worker.getInstrumentationContextData() );
   NANOS_INSTRUMENT (icd->setStartingWD(true) );

   return startMultiThread( worker, numPEs, repPEs );
}

BaseThread & ProcessingElement::startThread ( WD &work, ext::SMPMultiThread *parent )
{
   BaseThread &thread = createThread( work, parent );

   thread.start();

   _threads.push_back( &thread );

   return thread;
}

BaseThread & ProcessingElement::startMultiThread ( WD &work, unsigned int numPEs, PE **repPEs )
{
   BaseThread &thread = createMultiThread( work, numPEs, repPEs );

   thread.start();

   _threads.push_back( &thread );

   return thread;
}

BaseThread & ProcessingElement::associateThisThread ( bool untieMain )
{
   WD & worker = getMasterWD();
   NANOS_INSTRUMENT (sys.getInstrumentation()->raiseOpenPtPEventNkvs ( NANOS_WD_DOMAIN, (nanos_event_id_t) worker.getId(), 0, NULL, NULL ); )
   NANOS_INSTRUMENT (InstrumentationContextData *icd = worker.getInstrumentationContextData() );
   NANOS_INSTRUMENT (icd->setStartingWD(true) );
   
   BaseThread &thread = createThread( worker );

   thread.associate();

   if ( !untieMain ) {
      worker.tieTo(thread);
   }

   return thread;
}

void ProcessingElement::stopAll ()
{
   ThreadList::iterator it;
   BaseThread *thread;

   for ( it = _threads.begin(); it != _threads.end(); it++ ) {
      thread = *it;
      thread->stop();
      thread->join();
      if ( thread->hasTeam() )
         thread->leaveTeam();
   }
}

void* ProcessingElement::getAddress( WorkDescriptor &wd, uint64_t tag, nanos_sharing_t sharing )
{
   void *actualTag = (void *) ( sharing == NANOS_PRIVATE ? (char *)wd.getData() + (unsigned long)tag : (void *)tag );
   return actualTag;
}

void* ProcessingElement::newGetAddress( CopyData const &cd )
{
   message("Returning base address of cd (PE::newGetAddress)");
   return cd.getBaseAddress();
}

void ProcessingElement::copyTo( WorkDescriptor& wd, void *dst, uint64_t tag, nanos_sharing_t sharing, size_t size )
{
   void *actualTag = (void *) ( sharing == NANOS_PRIVATE ? (char *)wd.getData() + (unsigned long)tag : (void *)tag );
   // FIXME: should this be done by using the local copeir of the device?
   memcpy( dst, actualTag, size );
}

Device const *ProcessingElement::getCacheDeviceType() const {
   return NULL;
}
