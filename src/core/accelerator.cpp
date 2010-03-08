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

using namespace nanos;

void Accelerator::copyDataIn( WorkDescriptor &work )
{
   CopyData *copies = work.getCopies();
   for ( unsigned int i = 0; i < work.getNumCopies(); i++ ) {
      CopyData & cd = copies[i];
      uint64_t tag = (uint64_t) cd.isPrivate() ? ((uint64_t) work.getData() + (unsigned long)cd.getAddress()) : cd.getAddress();
      this->registerDataAccessDependent( tag, cd.getSize() );
      if ( cd.isInput() )
         this->copyDataDependent( tag, cd.getSize() );
   }      
}

void Accelerator::copyDataOut( WorkDescriptor& work )
{
   CopyData *copies = work.getCopies();
   for ( unsigned int i = 0; i < work.getNumCopies(); i++ ) {
      CopyData & cd = copies[i];
      uint64_t tag = (uint64_t) cd.isPrivate() ? ((uint64_t) work.getData() + (unsigned long) cd.getAddress()) : cd.getAddress();
      this->unregisterDataAccessDependent( tag );
      if ( cd.isOutput() )
          this->copyBackDependent( tag, cd.getSize() );
   }
}

void* Accelerator::getAddress( WorkDescriptor &wd, uint64_t tag, nanos_sharing_t sharing )
{
   uint64_t actualTag = (uint64_t) ( sharing == NX_PRIVATE ? (uint64_t) wd.getData() + (unsigned long) tag : tag );
   return getAddressDependent( actualTag );
}

void Accelerator::copyTo( WorkDescriptor &wd, void *dst, uint64_t tag, nanos_sharing_t sharing, size_t size )
{
   uint64_t actualTag = (uint64_t) ( sharing == NX_PRIVATE ? (uint64_t) wd.getData() + (unsigned long) tag : tag );
   copyToDependent( dst, actualTag, size );
}
