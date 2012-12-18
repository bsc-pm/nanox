
#include "oclprocessor.hpp"
#include "oclremoteprocessor.hpp"
#include "oclthread.hpp"

using namespace nanos;
using namespace nanos::ext;

#ifdef CLUSTER_DEV

namespace {

class OCLRemoteThread : public OCLThread, public ClusterRemoteThread
{
public:
   OCLRemoteThread( WD &wd, PE *pe ) : ClusterRemoteThread( wd, pe )
   {
      setCurrentWD( wd );
      
   }

   // Do not implement.
   OCLRemoteThread( const OCLRemoteThread &thr );

   // Do not implement.
   const OCLRemoteThread &operator=( const OCLRemoteThread &thr );
};

template <typename Ty>
class OCLDep : public Dependency
{
public:
   OCLDep( Ty **addr, bool input, bool output ) :
      Dependency( reinterpret_cast<void **>( addr ), // Address.
                  0,                                 // Offset.
                  input,                             // Input.
                  output,                            // Output.
                  false,                             // Rename.
                  false                              // Commutative.
                ) { }
};

template <typename Ty>
class OCLDep<const Ty> : public Dependency
{
public:
   OCLDep( const Ty **addr, bool input, bool output ) :
      Dependency( reinterpret_cast<void **>( const_cast<Ty **>( addr ) ),
                  0,
                  input,
                  output,
                  false,
                  false
                ) { }
};

template <typename Ty>
class OCLInputDep : public OCLDep<Ty>
{
public:
   OCLInputDep( Ty **addr ) : OCLDep<Ty>( addr, true, false ) { }
};

template <typename Ty>
class OCLOutputDep : public OCLDep<Ty>
{
public:
   OCLOutputDep( Ty **addr ) : OCLDep<Ty>( addr, false, true ) { }
};

template <typename Ty>
class OCLInputOutputDep : public OCLDep<Ty>
{
public:
   OCLInputOutputDep( Ty **addr ) : OCLDep<Ty>( addr, true, true ) { }
};

} // End anonymous namespace.

//
// OCLRemotePEFactory implementation.
//

OCLRemotePEFactory::OCLRemotePEFactory() : ClusterRemotePEFactory( OCLDev ) { }

ProcessingElement *OCLRemotePEFactory::operator()( int localId,
                                                   int remoteId,
                                                   unsigned remoteNode )
{
   return new OCLRemoteProcessor( localId,
                                  remoteId,
                                  remoteNode,
                                  clusterDev.getNetwork() );
}

//
// OCLRemoteIncomingHandler implementation.
//

OCLRemoteIncomingHandler::OCLRemoteIncomingHandler( OCLProcessor &pe ) :
   ClusterIncomingHandler( pe ) { }

#define OCL_NANOS_DEVICE( DD )                   \
   case DD ## Id:                                \
      dev.factory = ClusterCloningDDFactory<DD>; \
      dev.dd_size = sizeof( DD );                \
      dev.arg = dd;                              \
      break;

void OCLRemoteIncomingHandler::buildDevice(
   nanos_device_t &dev,
   LendTaskMessage::DeviceDataId id,
   void *dd )
{
   switch( static_cast<DeviceDataId>( id ) )
   {
   OCL_NANOS_DEVICE( OCLDeviceInfoDD )
   OCL_NANOS_DEVICE( OCLReadTicksDD )
   OCL_NANOS_DEVICE( OCLNDRangeKernelStarSSDD )
   OCL_NANOS_DEVICE( OCLUEWaitDD )
   OCL_NANOS_DEVICE( OCLUESignalDD )
   OCL_NANOS_DEVICE( OCLReadBufferDD )
   OCL_NANOS_DEVICE( OCLWriteBufferDD )
   OCL_NANOS_DEVICE( OCLMapBufferDD )
   OCL_NANOS_DEVICE( OCLUnmapBufferDD )

   default:
      fatal0( "Unknown device" );
   }
}

#undef OCL_NANOS_DEVICE

void OCLRemoteIncomingHandler::buildDeps( std::vector<Dependency> &deps,
                                          ext::LendTaskMessage::DeviceDataId id,
                                          void *dd,
                                          void *data )
{
   switch( static_cast<DeviceDataId>( id ) )
   {
   case OCLDeviceInfoDDId:
   {
      size_t offset = OCLDeviceInfoDD::eventsOffset( data );

      buildDeps( deps,
                 *static_cast<OCLDeviceInfoDD *>( dd ),
                 *static_cast<OCLDeviceInfoDD::Data *>( data ),
                 OCLDD::EventIterator::begin( data, offset ),
                 OCLDD::EventIterator::end( data, offset) );
      break;
   }

   case OCLReadTicksDDId:
   {
      size_t offset = OCLReadTicksDD::eventsOffset( data );

      buildDeps( deps,
                 *static_cast<OCLReadTicksDD *>( dd ),
                 *static_cast<OCLReadTicksDD::Data *>( data ),
                 OCLDD::EventIterator::begin( data, offset ),
                 OCLDD::EventIterator::end( data, offset ) );
      break;
   }

   case OCLNDRangeKernelDDId:
   {
      size_t offset = OCLNDRangeKernelStarSSDD::eventsOffset( data );

      buildDeps( deps,
                 *static_cast<OCLNDRangeKernelStarSSDD *>( dd ),
                 *static_cast<OCLNDRangeKernelStarSSDD::Data *>( data ),
                 OCLNDRangeKernelStarSSDD::ArgsIterator::begin( data ),
                 OCLNDRangeKernelStarSSDD::ArgsIterator::end( data ),
                 OCLDD::EventIterator::begin( data, offset ),
                 OCLDD::EventIterator::end( data, offset ) );
      break;
   }

   case OCLUEWaitDDId:
   {
      buildDeps( deps,
                 OCLDD::EventIterator::begin( data ),
                 OCLDD::EventIterator::end( data ) );
      break;
   }

   case OCLUESignalDDId:
   {
      // No deps here -- correct.
      break;
   }

   case OCLReadBufferDDId:
   {
      size_t offset = OCLReadBufferDD::eventsOffset( data );

      buildDeps( deps,
                 *static_cast<OCLReadBufferDD *>( dd ),
                 *static_cast<OCLReadBufferDD::Data *>( data ),
                 OCLDD::EventIterator::begin( data, offset ),
                 OCLDD::EventIterator::end( data, offset ) );
      break;
   }

   case OCLWriteBufferDDId:
   {
      size_t offset = OCLWriteBufferDD::eventsOffset( data );

      buildDeps( deps,
                 *static_cast<OCLWriteBufferDD *>( dd ),
                 *static_cast<OCLWriteBufferDD::Data *>( data ),
                 OCLDD::EventIterator::begin( data, offset ),
                 OCLDD::EventIterator::end( data, offset ) );
      break;
   }

   case OCLMapBufferDDId:
   {
      size_t offset = OCLMapBufferDD::eventsOffset( data );

      buildDeps( deps,
                 *static_cast<OCLMapBufferDD *>( dd ),
                 *static_cast<OCLMapBufferDD::Data *>( data ),
                 OCLDD::EventIterator::begin( data, offset ),
                 OCLDD::EventIterator::end( data, offset ) );
      break;
   }

   case OCLUnmapBufferDDId:
   {
      size_t offset = OCLUnmapBufferDD::eventsOffset( data );

      buildDeps( deps,
                 *static_cast<OCLUnmapBufferDD *>( dd ),
                 *static_cast<OCLUnmapBufferDD::Data *>( data ),
                 OCLDD::EventIterator::begin( data, offset ),
                 OCLDD::EventIterator::end( data, offset ) );
      break;
   }

   default:
      fatal0( "Unknown DD" );
   }
}

void OCLRemoteIncomingHandler::buildDeps( std::vector<Dependency> &deps,
                                          OCLDeviceInfoDD &dd,
                                          OCLDeviceInfoDD::Data &data,
                                          OCLDD::event_iterator i,
                                          OCLDD::event_iterator e )
{
   deps.push_back( OCLOutputDep<OCLDeviceInfoDD::Info>( &data._info ) );

   buildDeps( deps, i, e );
}

void OCLRemoteIncomingHandler::buildDeps( std::vector<Dependency> &deps,
                                          OCLReadTicksDD &dd,
                                          OCLReadTicksDD::Data &data,
                                          OCLDD::event_iterator i,
                                          OCLDD::event_iterator e )
{
   deps.push_back( OCLOutputDep<unsigned long long>( &data._ticks ) );

   buildDeps( deps, i, e );
}

void OCLRemoteIncomingHandler::buildDeps( std::vector<Dependency> &deps,
                                          OCLNDRangeKernelStarSSDD &dd,
                                          OCLNDRangeKernelStarSSDD::Data &data,
                                          OCLNDRangeKernelStarSSDD::arg_iterator j,
                                          OCLNDRangeKernelStarSSDD::arg_iterator f,
                                          OCLDD::event_iterator i,
                                          OCLDD::event_iterator e )
{
   deps.push_back( OCLInputDep<const char>( &data._programSrcs ) );
   deps.push_back( OCLInputDep<const char>( &data._kernName ) );
   deps.push_back( OCLInputDep<const char>( &data._compilerOptions ) );

   for( ; j != f; ++j )
     if( isBufferArg( *j ) )
       deps.push_back( OCLInputDep<void>( j.getRaw() ) );

   buildDeps( deps, i, e );

   if( dd.isProfiled() )
   {
      size_t offset = OCLNDRangeKernelStarSSDD::eventsOffset( &data );
      buildDeps( deps, OCLDD::getProfilePtr( &data, offset ) );
   }
}

void OCLRemoteIncomingHandler::buildDeps( std::vector<Dependency> &deps,
                                          OCLReadBufferDD &dd,
                                          OCLReadBufferDD::Data &data,
                                          OCLDD::event_iterator i,
                                          OCLDD::event_iterator e )
{
   deps.push_back( OCLOutputDep<void>( &data._dst ) );
   deps.push_back( OCLInputDep<void>( &data._buf ) );

   buildDeps( deps, i, e );

   if( dd.isProfiled() )
   {
      size_t offset = OCLReadBufferDD::eventsOffset( &data );
      buildDeps( deps, OCLDD::getProfilePtr( &data, offset ) );
   }
}

void OCLRemoteIncomingHandler::buildDeps( std::vector<Dependency> &deps,
                                          OCLWriteBufferDD &dd,
                                          OCLWriteBufferDD::Data &data,
                                          OCLDD::event_iterator i,
                                          OCLDD::event_iterator e )
{
   deps.push_back( OCLInputDep<void>( &data._buf ) );
   deps.push_back( OCLInputDep<void>( &data._src ) );

   buildDeps( deps, i, e );

   if( dd.isProfiled() )
   {
      size_t offset = OCLWriteBufferDD::eventsOffset( &data );
      buildDeps( deps, OCLDD::getProfilePtr( &data, offset ) );
   }
}

void OCLRemoteIncomingHandler::buildDeps( std::vector<Dependency> &deps,
                                          OCLMapBufferDD &dd,
                                          OCLMapBufferDD::Data &data,
                                          OCLDD::event_iterator i,
                                          OCLDD::event_iterator e )
{
   if( data._dst )
      deps.push_back( OCLOutputDep<void>( &data._dst ) );

   if( data._buf )
      deps.push_back( OCLInputDep<void>( &data._buf ) );

   buildDeps( deps, i, e );

   if( dd.isProfiled() )
   {
      size_t offset = OCLMapBufferDD::eventsOffset( &data );
      buildDeps( deps, OCLDD::getProfilePtr( &data, offset ) );
   }
}

void OCLRemoteIncomingHandler::buildDeps( std::vector<Dependency> &deps,
                                          OCLUnmapBufferDD &dd,
                                          OCLUnmapBufferDD::Data &data,
                                          OCLDD::event_iterator i,
                                          OCLDD::event_iterator e )
{
   if( data._buf ) {
      deps.push_back( OCLInputDep<void>( &data._buf ) );
      deps.push_back( OCLInputDep<void>( &data._src ) );
   }

   buildDeps( deps, i, e );

   if( dd.isProfiled() )
   {
      size_t offset = OCLUnmapBufferDD::eventsOffset( &data );
      buildDeps( deps, OCLDD::getProfilePtr( &data, offset ) );
   }
}

void OCLRemoteIncomingHandler::buildDeps( std::vector<Dependency> &deps,
                                          OCLDD::event_iterator i,
                                          OCLDD::event_iterator e )
{
  OCLDD::event_iterator j = i,
                        f = e - 1;

   // The first n - 1 events are input-deps.
   for( ; j != f; ++j )
      deps.push_back( OCLInputDep<void>( j.getRaw() ) );

   // The last is an output-dep.
   deps.push_back( OCLOutputDep<void>( j.getRaw() ) );
}

void OCLRemoteIncomingHandler::buildDeps( std::vector<Dependency> &deps,
                                          OCLDD::ProfData *profData )
{
   deps.push_back( OCLOutputDep<unsigned long long>( &profData->_startTick ) );
   deps.push_back( OCLOutputDep<unsigned long long>( &profData->_endTick ) );
}

bool OCLRemoteIncomingHandler::isBufferArg( OCLNDRangeKernelStarSSDD::Arg &arg )
{
   return Size( arg._size ).isDeviceOperation() && arg._ptr;
}

//
// OCLRemoteProcessor implementation.
//

BaseThread &OCLRemoteProcessor::createThread( WorkDescriptor &wd )
{
   ClusterRemoteThread *thr;

   ensure( wd.canRunIn( SMP ), "Incompatible worker thread" );

   thr = NEW OCLRemoteThread( wd, this );

   return *thr;
}

#define OCL_NANOS_DD( DD )                                                    \
   if( dynamic_cast<const DD *>( &dd ) )                                      \
   {                                                                          \
      OCLRemoteIncomingHandler::DeviceDataId id;                              \
      id = OCLRemoteIncomingHandler:: DD ## Id;                               \
                                                                              \
      serializedDD.setId( static_cast<LendTaskMessage::DeviceDataId>( id ) ); \
      return;                                                                 \
   }

// TODO: use template specialization instead of VT.
void OCLRemoteProcessor::serializeDeviceData(
   ClusterIncomingHandler::SerializedDeviceData &serializedDD,
   const DeviceData &dd ) const
{
   ClusterProcessor<OCLDevice>::serializeDeviceData( serializedDD, dd );

   OCL_NANOS_DD( OCLDeviceInfoDD )
   OCL_NANOS_DD( OCLReadTicksDD )
   OCL_NANOS_DD( OCLNDRangeKernelStarSSDD )
   OCL_NANOS_DD( OCLUEWaitDD )
   OCL_NANOS_DD( OCLUESignalDD )
   OCL_NANOS_DD( OCLReadBufferDD )
   OCL_NANOS_DD( OCLWriteBufferDD )
   OCL_NANOS_DD( OCLMapBufferDD )
   OCL_NANOS_DD( OCLUnmapBufferDD )

   fatal0( "Unknown DD" );
}

#undef OCL_NANOS_DD

#define OCL_NANOS_DD( DD )                                    \
   if( const DD *castedDD = dynamic_cast<const DD *>( &dd ) ) \
   {                                                          \
      findPointersInData( pointers, data, *castedDD );        \
      return;                                                 \
   }

// TODO: use template specialization instead of VT.
void OCLRemoteProcessor::findPointersInData( std::vector<void **> &pointers,
                                             void *data,
                                             const DeviceData &dd ) const
{
   OCL_NANOS_DD( OCLDeviceInfoDD )
   OCL_NANOS_DD( OCLReadTicksDD )
   OCL_NANOS_DD( OCLNDRangeKernelStarSSDD )
   OCL_NANOS_DD( OCLUEWaitDD )
   OCL_NANOS_DD( OCLUESignalDD )
   OCL_NANOS_DD( OCLReadBufferDD )
   OCL_NANOS_DD( OCLWriteBufferDD )
   OCL_NANOS_DD( OCLMapBufferDD )
   OCL_NANOS_DD( OCLUnmapBufferDD )

   fatal0( "Unknown DD" );
}

#undef OCL_NANOS_DD

void OCLRemoteProcessor::findPointersInData( std::vector<void **> &pointers,
                                             void *data,
                                             const OCLDeviceInfoDD &dd ) const
{
   OCLDeviceInfoDD::Data *ddData;
   ddData = reinterpret_cast<OCLDeviceInfoDD::Data *>( data );

   pointers.push_back( reinterpret_cast<void **>( &ddData->_info ) );
   findEventPointersInData( pointers,
                            data,
                            OCLDeviceInfoDD::eventsOffset( data ) );
}

void OCLRemoteProcessor::findPointersInData( std::vector<void **> &pointers,
                                             void *data,
                                             const OCLReadTicksDD &dd ) const
{
   OCLReadTicksDD::Data *ddData;
   ddData = reinterpret_cast<OCLReadTicksDD::Data *>( data );

   pointers.push_back( reinterpret_cast<void **>( &ddData->_ticks ) );
   findEventPointersInData( pointers,
                            data,
                            OCLReadTicksDD::eventsOffset( data ) );
}

void OCLRemoteProcessor::findPointersInData(
   std::vector<void **> &pointers,
   void *data,
   const OCLNDRangeKernelStarSSDD &dd ) const
{
   OCLNDRangeKernelStarSSDD::Data *ddData;
   ddData = reinterpret_cast<OCLNDRangeKernelStarSSDD::Data *>( data );

   OCLNDRangeKernelStarSSDD::ArgsIterator i, e;

   for( i = OCLNDRangeKernelStarSSDD::ArgsIterator::begin( data ),
        e = OCLNDRangeKernelStarSSDD::ArgsIterator::end( data );
        i != e;
        ++i )
      pointers.push_back( &i->_ptr );

   char **srcs = const_cast<char **>( &ddData->_programSrcs ),
        **kern = const_cast<char **>( &ddData->_kernName ),
        **opts = const_cast<char **>( &ddData->_compilerOptions );

   size_t offset = OCLNDRangeKernelStarSSDD::eventsOffset( data );

   pointers.push_back( reinterpret_cast<void **>( srcs ) );
   pointers.push_back( reinterpret_cast<void **>( kern ) );
   pointers.push_back( reinterpret_cast<void **>( opts ) );

   findEventPointersInData( pointers, data, offset );

   if( dd.isProfiled() )
      findProfilePointersInData( pointers, data, offset );
}

void OCLRemoteProcessor::findPointersInData( std::vector<void **> &pointers,
                                             void *data,
                                             const OCLUEWaitDD &dd ) const
{
   findEventPointersInData( pointers, data );
}

void OCLRemoteProcessor::findPointersInData( std::vector<void **> &pointers,
                                             void *data,
                                             const OCLUESignalDD &dd ) const
{
   // No pointer here -- this is correct!
}

void OCLRemoteProcessor::findPointersInData( std::vector<void **> &pointers,
                                             void *data,
                                             const OCLReadBufferDD &dd ) const
{
   OCLReadBufferDD::Data *ddData;
   ddData = reinterpret_cast<OCLReadBufferDD::Data *>( data  );

   size_t offset = OCLReadBufferDD::eventsOffset( data );

   pointers.push_back( reinterpret_cast<void **>( &ddData->_dst ) );
   pointers.push_back( reinterpret_cast<void **>( &ddData->_buf ) );

   findEventPointersInData( pointers, data, offset );

   if( dd.isProfiled() )
      findProfilePointersInData( pointers, data, offset );
}

void OCLRemoteProcessor::findPointersInData( std::vector<void **> &pointers,
                                             void *data,
                                             const OCLWriteBufferDD &dd ) const
{
   OCLWriteBufferDD::Data *ddData;
   ddData = reinterpret_cast<OCLWriteBufferDD::Data *>( data );

   size_t offset = OCLWriteBufferDD::eventsOffset( data );

   pointers.push_back( reinterpret_cast<void **>( &ddData->_buf ) );
   pointers.push_back( reinterpret_cast<void **>( &ddData->_src ) );

   findEventPointersInData( pointers, data, offset );

   if( dd.isProfiled() )
      findProfilePointersInData( pointers, data, offset );
}

void OCLRemoteProcessor::findPointersInData( std::vector<void **> &pointers,
                                             void *data,
                                             const OCLMapBufferDD &dd ) const
{
   OCLMapBufferDD::Data *ddData;
   ddData = reinterpret_cast<OCLMapBufferDD::Data *>( data );

   size_t offset = OCLMapBufferDD::eventsOffset( data );

   if( dd.isReadMapping() )
   {
      pointers.push_back( reinterpret_cast<void **>( &ddData->_dst ) );
      pointers.push_back( reinterpret_cast<void **>( &ddData->_buf ) );
   }

   findEventPointersInData( pointers, data, offset );

   if( dd.isProfiled() )
      findProfilePointersInData( pointers, data, offset );
}

void OCLRemoteProcessor::findPointersInData( std::vector<void **> &pointers,
                                             void *data,
                                             const OCLUnmapBufferDD &dd ) const
{
   OCLUnmapBufferDD::Data *ddData;
   ddData = reinterpret_cast<OCLUnmapBufferDD::Data *>( data );

   size_t offset = OCLUnmapBufferDD::eventsOffset( data );

   if( dd.isWriteMapping() )
   {
      pointers.push_back( reinterpret_cast<void **>( &ddData->_buf ) );
      pointers.push_back( reinterpret_cast<void **>( &ddData->_src ) );
   }

   findEventPointersInData( pointers, data, offset );

   if( dd.isProfiled() )
      findProfilePointersInData( pointers, data, offset );
}

void OCLRemoteProcessor::findEventPointersInData(
   std::vector<void **> &pointers,
   void *data,
   size_t eventsOffset ) const
{
   OCLDD::EventIterator i = OCLDD::EventIterator::begin( data, eventsOffset ),
                        e = OCLDD::EventIterator::end( data, eventsOffset );

   for( ; i != e; ++i )
      pointers.push_back( reinterpret_cast<void **>( &*i ) );
}

void OCLRemoteProcessor::findProfilePointersInData(
   std::vector<void **> &pointers,
   void *data,
   size_t eventsOffset ) const
{
   OCLDD::ProfData *prof = OCLDD::getProfilePtr( data, eventsOffset );

   pointers.push_back( reinterpret_cast<void **>( &prof->_startTick ) );
   pointers.push_back( reinterpret_cast<void **>( &prof->_endTick ) );
}

#endif // CLUSTER_DEV
