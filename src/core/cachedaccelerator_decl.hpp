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

#ifndef _NANOS_CACHED_ACCELERATOR_DECL
#define _NANOS_CACHED_ACCELERATOR_DECL

#include "accelerator_decl.hpp"
#include "cache_decl.hpp"
#include "system_decl.hpp"

namespace nanos
{

   template <class Device>
   class CachedAccelerator : public Accelerator
   {
      private:
        DeviceCache<Device> * _cache;
        CachePolicy         * _cachePolicy;

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
         CachedAccelerator ( int newId, const Device *arch, int uniqueId, System::CachePolicyType policy, int cacheSize = 0 ) :
            Accelerator( newId, arch, uniqueId ), _cache( NEW DeviceCache<Device>( cacheSize, NULL, this ) )
         {
            configureCache( cacheSize, policy );
         }

         /*! \brief CachedAccelerator constructor - from 'newId' and 'arch'
          *
          *  Function 'configureCache()' needs to be called at some point in order to initialize it.
          */
          CachedAccelerator ( int newId, const Device *arch, int uniqueId ) :
             Accelerator( newId, arch, uniqueId ), _cache( NULL ) {}

        /*! \brief CachedAccelerator destructor
         */
         virtual ~CachedAccelerator() { delete _cache; }

         unsigned int getMemorySpaceId() const { return _cache->getId(); }

         void configureCache( size_t cacheSize, System::CachePolicyType cachePolicy );

         void registerCacheAccessDependent( Directory& dir, uint64_t tag, size_t size, bool input, bool output );

         void unregisterCacheAccessDependent( Directory& dir, uint64_t tag, size_t size, bool output );
         
         void registerPrivateAccessDependent( Directory& dir, uint64_t tag, size_t size, bool input, bool output );
         
         void unregisterPrivateAccessDependent( Directory& dir, uint64_t tag, size_t size );
         
         void synchronize( CopyDescriptor &cd );
         
         void synchronize( std::list<CopyDescriptor> &cds );
         
         void waitInputDependent( uint64_t tag );
         
         void* getAddressDependent( uint64_t tag );
         
         void copyToDependent( void *dst, uint64_t tag, size_t size );
   };

};

#endif
