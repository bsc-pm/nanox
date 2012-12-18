
#ifndef _NANOS_OCL_REMOTE_PROCESSOR_DECL
#ifdef CLUSTER_DEV

#include "clusterprocessor.hpp"
#include "ocldd.hpp"

namespace nanos {
namespace ext {

class OCLProcessor;

class OCLRemotePEFactory : public ClusterRemotePEFactory
{
public:
   OCLRemotePEFactory();

public:
   virtual ProcessingElement *operator()( int localId,
                                          int remoteId,
                                          unsigned remoteNode );
};

class OCLRemoteIncomingHandler : public ClusterIncomingHandler
{
public:
   enum DeviceDataId
   {
      OCLDeviceInfoDDId,
      OCLReadTicksDDId,
      OCLNDRangeKernelDDId,
      OCLUEWaitDDId,
      OCLUESignalDDId,
      OCLReadBufferDDId,
      OCLWriteBufferDDId,
      OCLMapBufferDDId,
      OCLUnmapBufferDDId,
   };

public:
   OCLRemoteIncomingHandler( OCLProcessor &pe );

protected:
   virtual void buildDevice( nanos_device_t &dev,
                             ext::LendTaskMessage::DeviceDataId id,
                             void *dd );
   virtual void buildDeps( std::vector<Dependency> &deps,
                           ext::LendTaskMessage::DeviceDataId id,
                           void *dd,
                           void *data );

private:
   void buildDeps( std::vector<Dependency> &deps,
                   OCLDeviceInfoDD &dd,
                   OCLDeviceInfoDD::Data &data,
                   OCLDD::event_iterator i,
                   OCLDD::event_iterator e );

   void buildDeps( std::vector<Dependency> &deps,
                   OCLReadTicksDD &dd,
                   OCLReadTicksDD::Data &data,
                   OCLDD::event_iterator i,
                   OCLDD::event_iterator e );

   void buildDeps( std::vector<Dependency> &deps,
                   OCLNDRangeKernelDD &dd,
                   OCLNDRangeKernelDD::Data &data,
                   OCLNDRangeKernelDD::arg_iterator j,
                   OCLNDRangeKernelDD::arg_iterator f,
                   OCLDD::event_iterator i,
                   OCLDD::event_iterator e );

   void buildDeps( std::vector<Dependency> &deps,
                   OCLReadBufferDD &dd,
                   OCLReadBufferDD::Data &data,
                   OCLDD::event_iterator i,
                   OCLDD::event_iterator e );

   void buildDeps( std::vector<Dependency> &deps,
                   OCLWriteBufferDD &dd,
                   OCLWriteBufferDD::Data &data,
                   OCLDD::event_iterator i,
                   OCLDD::event_iterator e );

   void buildDeps( std::vector<Dependency> &deps,
                   OCLMapBufferDD &dd,
                   OCLMapBufferDD::Data &data,
                   OCLDD::event_iterator i,
                   OCLDD::event_iterator e );

   void buildDeps( std::vector<Dependency> &deps,
                   OCLUnmapBufferDD &dd,
                   OCLUnmapBufferDD::Data &data,
                   OCLDD::event_iterator i,
                   OCLDD::event_iterator e );

   void buildDeps( std::vector<Dependency> &deps,
                   OCLDD::event_iterator i,
                   OCLDD::event_iterator e );

   void buildDeps( std::vector<Dependency> &deps, OCLDD::ProfData *profData );

   bool isBufferArg( OCLNDRangeKernelDD::Arg &arg );
};

class OCLRemoteProcessor : public ClusterProcessor<OCLDevice>
{
public:
   OCLRemoteProcessor( int localId,
                       int remoteId,
                       unsigned remoteNode,
                       ClusterNetwork &net ) :
      ClusterProcessor<OCLDevice>( localId,
                                   remoteId,
                                   remoteNode,
                                   &OCLDev,
                                   net ) { }

   // Do not implement.
   OCLRemoteProcessor( const OCLRemoteProcessor &pe );

   // Do not implement.
   const OCLRemoteProcessor &operator=( const OCLRemoteProcessor &pe );

public:
   virtual BaseThread &createThread( WorkDescriptor &wd );

protected:
   // TODO: remove virtual.
   virtual void serializeDeviceData(
      ClusterIncomingHandler::SerializedDeviceData &serializedDD,
      const DeviceData &dd ) const;

   // TODO: remove virtual.
   virtual void findPointersInData( std::vector<void **> &pointers,
                                    void *data,
                                    const DeviceData &dd ) const;

private:
   void findPointersInData( std::vector<void **> &pointers,
                            void *data,
                            const OCLDeviceInfoDD &dd ) const;

   void findPointersInData( std::vector<void **> &pointers,
                            void *data,
                            const OCLReadTicksDD &dd ) const;

   void findPointersInData( std::vector<void **> &pointers,
                            void *data,
                            const OCLNDRangeKernelDD &dd ) const;

   void findPointersInData( std::vector<void **> &pointers,
                            void *data,
                            const OCLUEWaitDD &dd ) const;

   void findPointersInData( std::vector<void **> &pointers,
                            void *data,
                            const OCLUESignalDD &dd ) const;

   void findPointersInData( std::vector<void **> &pointers,
                            void *data,
                            const OCLReadBufferDD &dd ) const;

   void findPointersInData( std::vector<void **> &pointers,
                            void *data,
                            const OCLWriteBufferDD &dd ) const;

   void findPointersInData( std::vector<void **> &pointers,
                            void *data,
                            const OCLMapBufferDD &dd ) const;

   void findPointersInData( std::vector<void **> &pointers,
                            void *data,
                            const OCLUnmapBufferDD &dd ) const;

   void findEventPointersInData( std::vector<void **> &pointers,
                                 void *data,
                                 size_t eventsOffset = 0 ) const;

   void findProfilePointersInData(std::vector<void **> &pointers,
                                  void *data,
                                  size_t eventsOffset ) const;
};

} // End namespace ext.
} // End namespace nanos.

#endif // CLUSTER_DEV
#endif // _NANOS_OCL_REMOTE_PROCESSOR_DECL
