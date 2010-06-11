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

#include "accelerator.hpp"
#include "debug.hpp"
#include "schedule.hpp"
#include "copydata.hpp"
#include "instrumentor_decl.hpp"
#include "system.hpp"

using namespace nanos;

void Accelerator::copyDataIn( WorkDescriptor &work )
{
   CopyData *copies = work.getCopies();
   for ( unsigned int i = 0; i < work.getNumCopies(); i++ ) {
      CopyData & cd = copies[i];
      uint64_t tag = (uint64_t) cd.isPrivate() ? ((uint64_t) work.getData() + (unsigned long)cd.getAddress()) : cd.getAddress();
      if ( cd.isInput() ) {
         NANOS_INSTRUMENTOR( static nanos_event_key_t key = sys.getInstrumentor()->getInstrumentorDictionary()->getEventKey("copy-in") );
         NANOS_INSTRUMENTOR( sys.getInstrumentor()->throwPointEvent( key, (nanos_event_value_t) cd.getSize() ) );
      }
      if ( cd.isPrivate() ) {
         this->registerPrivateAccessDependent( tag, cd.getSize(), cd.isInput(), cd.isOutput() );
      } else {
         this->registerCacheAccessDependent( tag, cd.getSize(), cd.isInput(), cd.isOutput() );
      }
   }
}

void Accelerator::copyDataOut( WorkDescriptor& work )
{
   CopyData *copies = work.getCopies();
   for ( unsigned int i = 0; i < work.getNumCopies(); i++ ) {
      CopyData & cd = copies[i];
      uint64_t tag = (uint64_t) cd.isPrivate() ? ((uint64_t) work.getData() + (unsigned long) cd.getAddress()) : cd.getAddress();
      if ( cd.isOutput() ) {
         NANOS_INSTRUMENTOR( static nanos_event_key_t key = sys.getInstrumentor()->getInstrumentorDictionary()->getEventKey("copy-out") );
		NANOS_INSTRUMENTOR( sys.getInstrumentor()->throwPointEvent( key, (nanos_event_value_t) cd.getSize() ) );
      }
      if ( cd.isPrivate() ) {
         this->unregisterPrivateAccessDependent( tag, cd.getSize() );
      } else {
         this->unregisterCacheAccessDependent( tag, cd.getSize() );
      }
   }
}

void* Accelerator::getAddress( WorkDescriptor &wd, uint64_t tag, nanos_sharing_t sharing )
{
   uint64_t actualTag = (uint64_t) ( sharing == NANOS_PRIVATE ? (uint64_t) wd.getData() + (unsigned long) tag : tag );
   return getAddressDependent( actualTag );
}

void Accelerator::copyTo( WorkDescriptor &wd, void *dst, uint64_t tag, nanos_sharing_t sharing, size_t size )
{
   uint64_t actualTag = (uint64_t) ( sharing == NANOS_PRIVATE ? (uint64_t) wd.getData() + (unsigned long) tag : tag );
   copyToDependent( dst, actualTag, size );
}
