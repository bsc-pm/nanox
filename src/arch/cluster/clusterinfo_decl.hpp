#ifndef _CLUSTERINFO_DECL
#define _CLUSTERINFO_DECL

#include <iostream>
#include "config_decl.hpp"
#include "system_decl.hpp"
#include "remoteworkgroup_fwd.hpp"

namespace nanos {
   namespace ext {

      class ClusterInfo
      {
         private:
            static unsigned int _numSegments;
            static void ** _segmentAddrList;
            static std::size_t * _segmentLenList;
            static unsigned int _numPinnedSegments;
            static void ** _pinnedSegmentAddrList;
            static std::size_t * _pinnedSegmentLenList;
            static unsigned int _extraPEsCount;
            static std::string _conduit;
            static std::size_t _nodeMem;
            static bool _allocWide;
            static int _gpuPresend;
            static int _smpPresend;
            static System::CachePolicyType _cachePolicy;

         public:
            ClusterInfo();
            ~ClusterInfo();

            static void prepare( Config& cfg );
            static void addSegments( unsigned int numSegments, void **segmentAddr, size_t *segmentSize );
            static void * getSegmentAddr( unsigned int idx );
            static std::size_t getSegmentLen( unsigned int idx );
            static void addPinnedSegments( unsigned int numSegments, void **segmentAddr, size_t *segmentSize );
            static void * getPinnedSegmentAddr( unsigned int idx );
            static std::size_t getPinnedSegmentLen( unsigned int idx );
            static unsigned int getExtraPEsCount();
            static void setExtraPEsCount( unsigned int num );
            static std::size_t getNodeMem();
            static int getGpuPresend();
            static int getSmpPresend();
            //static const std::string & getNetworkConduit();
            static void setUpCache();
            static System::CachePolicyType getCachePolicy ( void );
            static RemoteWorkGroup * getRemoteWorkGroup( int archId );
            static bool getAllocWide();
      };
   }
}

#endif /* _CLUSTERINFO_DECL */
