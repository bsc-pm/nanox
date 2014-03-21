#ifndef CLUSTERPLUGIN_DECL_H
#define CLUSTERPLUGIN_DECL_H

#include "plugin.hpp"
#include "gasnetapi_decl.hpp"

namespace nanos {
namespace ext {

class ClusterPlugin : public Plugin
{
      GASNetAPI _gasnetApi;

      unsigned int _numPinnedSegments;
      void ** _pinnedSegmentAddrList;
      std::size_t * _pinnedSegmentLenList;
      unsigned int _extraPEsCount;
      std::string _conduit;
      std::size_t _nodeMem;
      bool _allocWide;
      int _gpuPresend;
      int _smpPresend;
      System::CachePolicyType _cachePolicy;

   public:
      ClusterPlugin();
      virtual void config( Config& cfg );
      virtual void init();

      void prepare( Config& cfg );
      void addSegments( unsigned int numSegments, void **segmentAddr, size_t *segmentSize );
      void * getSegmentAddr( unsigned int idx );
      std::size_t getSegmentLen( unsigned int idx );
      void addPinnedSegments( unsigned int numSegments, void **segmentAddr, size_t *segmentSize );
      void * getPinnedSegmentAddr( unsigned int idx );
      std::size_t getPinnedSegmentLen( unsigned int idx );
      unsigned int getExtraPEsCount();
      void setExtraPEsCount( unsigned int num );
      std::size_t getNodeMem();
      int getGpuPresend();
      int getSmpPresend();
      void setUpCache();
      System::CachePolicyType getCachePolicy ( void );
      RemoteWorkGroup * getRemoteWorkGroup( int archId );
      bool getAllocWide();
};

}
}

#endif /* CLUSTERPLUGIN_DECL_H */
