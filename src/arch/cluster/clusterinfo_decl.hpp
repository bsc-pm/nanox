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

            static void addSegments( unsigned int numSegments, void **segmentAddr, size_t *segmentSize );
            static void * getSegmentAddr( unsigned int idx );
            static size_t getSegmentLen( unsigned int idx );
            static unsigned int getExtraPEsCount();
            static void setExtraPEsCount( unsigned int num);
      };
   }
}

#endif
