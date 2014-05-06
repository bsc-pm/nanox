#ifndef CLUSTERPLUGIN_DECL_H
#define CLUSTERPLUGIN_DECL_H

#include "plugin.hpp"
#include "system_decl.hpp"
#include "gasnetapi_decl.hpp"

namespace nanos {
namespace ext {

class ClusterPlugin : public ArchPlugin
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
      std::vector<ext::ClusterNode *> *_nodes;
      ext::SMPMultiThread *_clusterThread;

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
      RemoteWorkDescriptor * getRemoteWorkDescriptor( int archId );
      bool getAllocWide();


      virtual void startSupportThreads();
      virtual void startWorkerThreads(std::vector<BaseThread *> &workers);
      virtual void finalize();

      virtual ProcessingElement * createPE( unsigned id, unsigned uid );
      virtual unsigned getNumThreads() const; 
      virtual unsigned getNumPEs() const;
      void addPEs( std::vector<ProcessingElement *> &pes ) const;
};

}
}

#endif /* CLUSTERPLUGIN_DECL_H */
