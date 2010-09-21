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
         static unsigned int _numSegments;
         static void ** _segmentAddrList;
         static size_t * _segmentLenList;

         //typedef std::map< uintptr_t, size_t > SegmentMap;
         //
         //static std::vector< SegmentMap > _allocatedChunks;
         //static std::vector< SegmentMap > _freeChunks;

         static unsigned int _extraPEsCount;
         
      public:
         /*! \brief ClusterDevice constructor
          */
         ClusterDevice ( const char *n ) : Device ( n ) {}

         /*! \brief ClusterDevice copy constructor
          */
         ClusterDevice ( const ClusterDevice &arch ) : Device ( arch ) {}

         /*! \brief ClusterDevice destructor
          */
         ~ClusterDevice()
         {
            if ( _segmentAddrList != NULL )
               delete _segmentAddrList;

            if ( _segmentLenList != NULL )
               delete _segmentLenList;
         };

         static unsigned int getExtraPEsCount()
         {
            return _extraPEsCount;
         }

         static void setExtraPEsCount( unsigned int num)
         {
            _extraPEsCount = num;
         }

         static void * allocate( size_t size );
         static void free( void *address );

         static void copyIn( void *localDst, uint64_t remoteSrc, size_t size );
         static void copyOut( uint64_t remoteDst, void *localSrc, size_t size );

         static void copyLocal( void *dst, void *src, size_t size )
         {
            // Do not allow local copies in cluster memory
         }

         static void addSegments( unsigned int numSegments, void **segmentAddr, size_t *segmentSize )
         {
            unsigned int idx;
            _numSegments = numSegments;
            _segmentAddrList = new void *[ numSegments ];
            _segmentLenList = new size_t[ numSegments ];

            for ( idx = 0; idx < numSegments; idx += 1)
            {
               _segmentAddrList[ idx ] = segmentAddr[ idx ];
               _segmentLenList[ idx ] = segmentSize[ idx ];
            }
         }

         static void * getSegmentAddr( unsigned int idx )
         {
            return _segmentAddrList[ idx ];
         }

         static size_t getSegmentLen( unsigned int idx )
         {
            return _segmentLenList[ idx ];
         }
   };
}

#endif
