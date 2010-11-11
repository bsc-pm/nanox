#include "clusterinfo.hpp"

using namespace nanos;
using namespace ext;

unsigned int ClusterInfo::_numSegments = 0;
void ** ClusterInfo::_segmentAddrList = NULL;
size_t * ClusterInfo::_segmentLenList = NULL;
unsigned int ClusterInfo::_extraPEsCount = 0;
