#ifndef _CLUSTER_INFO
#define _CLUSTER_INFO

#include <iostream>

namespace nanos {
   namespace ext {

   class ClusterInfo
   {
      private:
         static unsigned int _numSegments;
         static void ** _segmentAddrList;
         static size_t * _segmentLenList;
         static unsigned int _extraPEsCount;

      public:
         ClusterInfo() {}
         
         ~ClusterInfo()
         {
            if ( _segmentAddrList != NULL )
               delete _segmentAddrList;

            if ( _segmentLenList != NULL )
               delete _segmentLenList;
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

         static unsigned int getExtraPEsCount()
         {
            return _extraPEsCount;
         }

         static void setExtraPEsCount( unsigned int num)
         {
            _extraPEsCount = num;
         }
   };
}
}

#endif
