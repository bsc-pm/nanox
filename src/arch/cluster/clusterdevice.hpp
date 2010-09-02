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

#ifndef _CLUSTER_DEVICE
#define _CLUSTER_DEVICE

#include "workdescriptor_decl.hpp"

namespace nanos
{

/* \brief Device specialization for cluster architecture
 * provides functions to allocate and copy data in the device
 */

   class ClusterDevice : public Device
   {
      private:
         //static void **memPublicSegmentStart;
         //static size_t *memPublicSegmentSize;
         //static memChunkDescriptor *memPublicSegment;

         typedef std::map< uintptr_t, size_t > SegmentMap;

         static std::vector< SegmentMap > allocatedChunks;
         static std::vector< SegmentMap > freeChunks;
         
      public:
         /*! \brief GPUDevice constructor
          */
         ClusterDevice ( const char *n ) : Device ( n ) {}

         /*! \brief GPUDevice copy constructor
          */
         ClusterDevice ( const ClusterDevice &arch ) : Device ( arch ) {}

         /*! \brief GPUDevice destructor
          */
         ~ClusterDevice() {};

         static void * allocate( size_t size );
         static void free( void *address );

         static void copyIn( void *localDst, uint64_t remoteSrc, size_t size );
         static void copyOut( uint64_t remoteDst, void *localSrc, size_t size );

         static void copyLocal( void *dst, void *src, size_t size )
         {
            // Do not allow local copies in cluster memory
         }

         static void addPublicSegment(void *segmentAddr, size_t segmentSize)
         {
            SegmentMap *mallocated = new SegmentMap;
            SegmentMap *mfree = new SegmentMap;

            ( *mfree )[ ( uintptr_t ) segmentAddr ] = segmentSize;

            allocatedChunks.push_back( *mallocated );
            freeChunks.push_back( *mfree );
         }
   };
}

#endif
