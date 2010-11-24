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

#ifndef _NANOS_CACHED_ACCELERATOR
#define _NANOS_CACHED_ACCELERATOR

#include "accelerator.hpp"
#include "cache.hpp"

namespace nanos
{

   template <class Device, class Policy = WriteThroughPolicy>
   class CachedAccelerator : public Accelerator
   {
      private:
        DeviceCache<Device,Policy>        _cache;

        /*! \brief CachedAccelerator default constructor (private)
         */
         CachedAccelerator ();
        /*! \brief CachedAccelerator copy constructor (private)
         */
         CachedAccelerator ( const CachedAccelerator &a );
        /*! \brief CachedAccelerator copy assignment operator (private)
         */
         const CachedAccelerator& operator= ( const CachedAccelerator &a );
      public:
        /*! \brief CachedAccelerator constructor - from 'newId' and 'arch'
         */
         CachedAccelerator ( int newId, const Device *arch, int cacheSize = 0 ) : Accelerator( newId, arch), _cache( cacheSize, this ) {}
        /*! \brief CachedAccelerator destructor
         */
         virtual ~CachedAccelerator() {}

         void setCacheSize( size_t size )
         {
            _cache.setSize( size );
         }

         void registerCacheAccessDependent( Directory& dir, uint64_t tag, size_t size, bool input, bool output )
         {
            _cache.registerCacheAccess( dir, tag, size, input, output );
         }

         void unregisterCacheAccessDependent( Directory& dir, uint64_t tag, size_t size, bool output )
         {
            _cache.unregisterCacheAccess( dir, tag, size, output );
         }
         
         void registerPrivateAccessDependent( Directory& dir, uint64_t tag, size_t size, bool input, bool output )
         {
            _cache.registerPrivateAccess( dir, tag, size, input, output );
         }
         
         void unregisterPrivateAccessDependent( Directory& dir, uint64_t tag, size_t size )
         {
            _cache.unregisterPrivateAccess( dir, tag, size );
         }
         
         void synchronize( CopyDescriptor &cd )
         {
            _cache.synchronize( cd );
         }
         
         void synchronize( std::list<CopyDescriptor> &cds )
         {
            _cache.synchronize( cds );
         }
         
         void waitInputDependent( uint64_t tag )
         {
            _cache.waitInput( tag );
         }
         
         void* GPUProcessor::getAddressDependent( uint64_t tag )
         {
            return _cache.getAddress( tag );
         }
         
         void GPUProcessor::copyToDependent( void *dst, uint64_t tag, size_t size )
         {
            _cache.copyTo( dst, tag, size );
         }
   };

};

#endif
