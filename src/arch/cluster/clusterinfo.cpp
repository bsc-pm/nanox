#include "clusterinfo_decl.hpp"

using namespace nanos;
using namespace ext;

unsigned int ClusterInfo::_numSegments = 0;
void ** ClusterInfo::_segmentAddrList = NULL;
size_t * ClusterInfo::_segmentLenList = NULL;
unsigned int ClusterInfo::_extraPEsCount = 0;


void ClusterInfo::addSegments( unsigned int numSegments, void **segmentAddr, size_t *segmentSize )
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

void * ClusterInfo::getSegmentAddr( unsigned int idx )
{
   return _segmentAddrList[ idx ];
}

size_t ClusterInfo::getSegmentLen( unsigned int idx )
{
   return _segmentLenList[ idx ];
}

unsigned int ClusterInfo::getExtraPEsCount()
{
   return _extraPEsCount;
}

void ClusterInfo::setExtraPEsCount( unsigned int num)
{
   _extraPEsCount = num;
}
