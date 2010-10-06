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

#ifndef _NANOS_ACCELERATOR
#define _NANOS_ACCELERATOR

#include <stdint.h>
#include "workdescriptor.hpp"
#include "processingelement.hpp"
#include <algorithm>
#include "functors.hpp"
#include "atomic.hpp"

#define LOCK_TRANSFER 0

namespace nanos
{

   class Accelerator : public ProcessingElement
   {

      private:
         Accelerator ( const Accelerator &pe );
         const Accelerator & operator= ( const Accelerator &pe );
#if LOCK_TRANSFER
         static Lock _transferLock;
#endif
         
      protected:
         virtual WorkDescriptor & getMasterWD () const = 0;
         virtual WorkDescriptor & getWorkerWD () const = 0;

      public:
         // constructors
         Accelerator ( int newId, const Device *arch ) : ProcessingElement( newId, arch) {}

         // destructor
         virtual ~Accelerator() {}

         virtual bool hasSeparatedMemorySpace() const { return true; };

         virtual void copyDataIn( WorkDescriptor& wd );
         virtual void copyDataOut( WorkDescriptor& wd );

         virtual void waitInputs( WorkDescriptor& wd );

         virtual void waitInputDependent( uint64_t tag ) = 0;

         virtual void registerCacheAccessDependent( uint64_t tag, size_t size, bool input, bool output ) = 0;
         virtual void unregisterCacheAccessDependent( uint64_t tag, size_t size, bool output ) = 0;
         virtual void registerPrivateAccessDependent( uint64_t tag, size_t size, bool input, bool output ) = 0;
         virtual void unregisterPrivateAccessDependent( uint64_t tag, size_t size ) = 0;

         virtual void* getAddress( WorkDescriptor& wd, uint64_t tag, nanos_sharing_t sharing );
         virtual void copyTo( WorkDescriptor& wd, void *dst, uint64_t tag, nanos_sharing_t sharing, size_t size );

         virtual void* getAddressDependent( uint64_t tag ) = 0;
         virtual void copyToDependent( void *dst, uint64_t tag, size_t size ) = 0;
   };

};

#endif
