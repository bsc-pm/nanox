
#include "memoryaddress.hpp"
#include "mpidevice_decl.hpp"

#if 0
enum {
	TAG_M2S_ORDER = 1200,
	TAG_CACHE_DATA_IN,
	TAG_CACHE_DATA_OUT, 
   TAG_CACHE_ANSWER,
	TAG_INI_TASK,
	TAG_END_TASK,
	TAG_ENV_STRUCT,
	TAG_CACHE_ANSWER_REALLOC,
   TAG_CACHE_ANSWER_ALLOC,
	TAG_CACHE_ANSWER_CIN,
	TAG_CACHE_ANSWER_COUT,
	TAG_CACHE_ANSWER_FREE,
	TAG_CACHE_ANSWER_DEV2DEV,
	TAG_CACHE_ANSWER_CL,
	TAG_FP_NAME_SYNC,
	TAG_FP_SIZE_SYNC,
	TAG_CACHE_DEV2DEV,
	TAG_EXEC_CONTROL,
	TAG_NUM_PENDING_COMMS,
	TAG_UNIFIED_MEM
};

//Because of DEV2DEV OPIDs <=0 are RESERVED, and OPIDs > OPID_DEVTODEV too
enum {
	OPID_FINISH=1,
	OPID_COPYIN = 2,
	OPID_COPYOUT=3,
	OPID_FREE = 4,
	OPID_ALLOCATE =5,
	OPID_COPYLOCAL = 6,
	OPID_REALLOC = 7,
	OPID_CONTROL = 8, 
	OPID_CREATEAUXTHREAD=9,
	OPID_UNIFIED_MEM_REQ=10,
	OPID_TASK_INIT=11,
	OPID_DEVTODEV=999 /*Keep DEV2DEV value as highest in the OPIDs*/
};
#endif

namespace nanos {
namespace mpi {
namespace command {

using namespace ext;

template < int id >
class Command {
	private:
		MPIProcessor const& _destination; // Only works from the point of view of the sender

	public:
		Command( MPIProcessor const& destination ) :
			_destination( destination )
		{
		}

		virtual ~Command()
		{
		}

		MPIProcessor const& getDestination() const
		{
			return _destination;
		}

		int getCommandId() const
		{
			return id;
		}

		virtual void execute() = 0;
};

template < int id >
class CacheOrder {
	private:
      MPI_Comm _communicator;
		int      _source;
      int      _destination;
		Address  _hostAddress;
		Address  _deviceAddress;
		size_t   _size;

		static MPI_Datatype _type = 0;

	public:
		CacheOrder( MPIProcessor const& destination ) :
			_communicator( destination.getCommunicator() ), _source( MPI_ANY_SOURCE ), _destination( destination.getRank() ),
			_hostAddress( Address::uninitialized() ), _deviceAddress( Address::uninitialized() ), _size( 0 )
		{
		}

		CacheOrder( MPIProcessor const& destination, size_t size ) :
			_communicator( destination.getCommunicator() ), _source( MPI_ANY_SOURCE ), _destination( destination.getRank() ),
			_hostAddress( Address::uninitialized() ), _deviceAddress( Address::uninitialized() ), _size( size )
		{
		}

		CacheOrder( MPIProcessor const& destination, Address deviceAddress ) :
			_communicator( destination.getCommunicator() ), _source( MPI_ANY_SOURCE ), _destination( destination.getRank() ),
			_hostAddress( Address::uninitialized() ), _deviceAddress( deviceAddress ), _size( size )
		{
		}

		CacheOrder( MPIProcessor const& destination, Address hostAddress, Address deviceAddress, size_t size ) :
			_communicator( destination.getCommunicator() ), _source( MPI_ANY_SOURCE ), _destination( destination.getRank() ),
			_hostAddress( hostAddress ), _deviceAddress( deviceAddress ), _size( size )
		{
		}

		CacheOrder( MPIProcessor const& source, MPIProcessor const& destination, Address hostAddress, Address deviceAddress, size_t size ) :
			_communicator( destination.getCommunicator() ), _source( source.getRank() ), _destination( destination.getRank() ),
			_hostAddress( hostAddress ), _deviceAddress( deviceAddress ), _size( size )
		{
			int comm_comparison;
			MPI_Comm_compare( source.getCommunicator(), destination.getCommunicator(), &comparison );
			assert( comm_comparison == MPI_IDENT );
		}

		MPI_Comm getCommunicator() const
		{
			return _communicator;
		}

		int getSource() const
		{
			return _source;
		}

		int getDestination() const
		{
			return _destination;
		}

		Address getHostAddress () const
		{
			return _hostAddress;
		}

		Address getDeviceAddress() const
		{
			return _deviceAddress;
		}

		size_t size() const
		{
			return _size;
		}

		static MPI_Datatype getDataType()
		{
			return _type;
		}

		static void initDataType()
		{
			MPI_Datatype typelist[3] = {
					MPI_INT,
					MPI_BYTE,
					MPI_BYTE };

			int blocklen[3] = {
					2,
					2*sizeof(Address),
					sizeof(size_t) };

			MPI_Aint disp[3] = { 
					offsetof(CacheOrder,_source),
					offsetof(CacheOrder,_hostAddress),
					offsetof(CacheOrder,_size) };

			MPI_Type_create_struct(3, blocklen, disp, typelist, &_type );
			MPI_Type_commit( &_type );		
		}

		static void freeDataType()
		{
			MPI_Type_free( &_type );
		}
};

class Allocate : public CacheOrder<OPID_ALLOCATE> {
	public:
		Allocate( MPIProcessor const& destination, size_t size ) :
			CacheOrder( destination.getCommunicator(), destination.getRank(), size )
		{
		}

		virtual ~Allocate()
		{
		}

		virtual void execute ()
		{
			// FIXME: using MPIDevice::cacheStruct is incorrect since we are not putting the id in the struct.
			//        in addition, the original struct does not have a MPIProcessor const reference.
    		MPIRemoteNode::nanosMPISend( this, 1, MPIDevice::cacheStruct, getDestination().getRank(), TAG_M2S_ORDER, getDestination().getCommunicator());
    		MPIRemoteNode::nanosMPIRecv( this, 1, MPIDevice::cacheStruct, getDestination().getRank(), TAG_CACHE_ANSWER_ALLOC, getDestination().getCommunicator(), MPI_STATUS_IGNORE );
		}
};

class Free : public CacheOrder<OPID_FREE> {
	public:
		Free( MPIProcessor const& destination, Address deviceAddress ) :
			CacheOrder( destination.getCommunicator(), destination.getRank(), deviceAddress, 0 )
		{
		}

		virtual ~Free()
		{
		}

		virtual void execute ()
		{
    		MPIRemoteNode::nanosMPISend( this, 1, MPIDevice::cacheStruct, getDestination().getRank(), TAG_M2S_ORDER, getDestination().getCommunicator());
		}
};

class Realloc : public CacheOrder<OPID_REALLOC> {
	public:
		Realloc( MPIProcessor const& destination, Address deviceAddress, size_t size ) :
			CacheOrder( destination, Address::uninitialized(), deviceAddress, size )
		{
		}

		virtual ~Realloc()
		{
		}

		virtual void execute ()
		{
    		MPIRemoteNode::nanosMPISend( this, 1, MPIDevice::cacheStruct, getDestination().getRank(), TAG_M2S_ORDER, getDestination().getCommunicator());
    		MPIRemoteNode::nanosMPIRecv( this, 1, MPIDevice::cacheStruct, getDestination().getRank(), TAG_CACHE_ANSWER_REALLOC, getDestination().getCommunicator(), MPI_STATUS_IGNORE );
		}
};

class CopyIn : public CacheOrder<OPID_COPYIN> {
	public:
		CopyIn( MPIProcessor const& destination, Address hostAddress, Address deviceAddress, size_t size ) :
			CacheOrder( destination, hostAddress, deviceAddress, size )
		{
		}

		virtual ~CopyIn()
		{
		}

		virtual void execute ()
		{
			MPI_Request req;
			MPIRemoteNode::nanosMPISend( this, 1, cacheStruct, getDestination().getRank(), TAG_M2S_ORDER, getDestination().getCommunicator() );
			MPIRemoteNode::nanosMPIIsend( getHostAddress(), size(), MPI_BYTE, getDestination().getRank(), TAG_CACHE_DATA_IN, getDestination().getCommunicator(), &req );
			getDestination().appendToPendingRequests(req);
		}
};

class CopyDeviceToDevice : public CacheOrder<OPID_DEVTODEV> {
	private:
		int _source;
		
	public:
		CopyDeviceToDevice( MPIProcessor const& source, MPIProcessor const& destination, Address sourceAddress, Address destinationAddress, size_t size ) :
			CacheOrder( destination, sourceAddress, destinationAddress, size ),
			_source( source.getRank() )
		{
		}

		virtual ~CopyDeviceToDevice()
		{
		}

		virtual void execute ()
		{
			// Send a message to both the source and destination of the transfer
			MPI_Request req[2];
			MPIRemoteNode::nanosMPIIsend( this, 1, cacheStruct, _source, TAG_M2S_ORDER, getDestination.getCommunicator() );
			MPIRemoteNode::nanosMPIIsend( this, 1, cacheStruct, getDestination().getRank(), TAG_M2S_ORDER, getDestination().getCommunicator() );
			MPI_Waitall( 2, req, MPI_STATUSES_IGNORE );
		}
};

class CopyOut : public CacheOrder<OPID_COPYOUT> {
		CopyOut( MPIProcessor const& destination, Address hostAddress, Address deviceAddress, size_t size ) :
			CacheOrder( destination, hostAddress, deviceAddress, size )
		{
		}

		virtual ~CopyOut()
		{
		}

		virtual void execute ()
		{
    		MPIRemoteNode::nanosMPISend( this, 1, MPIDevice::cacheStruct, getDestination().getRank(), TAG_M2S_ORDER, getDestination().getCommunicator());
    		MPIRemoteNode::nanosMPIRecv( getHostAddress(), size(), MPI_BYTE, getDestination().getRank(), TAG_CACHE_DATA_OUT, getDestination().getCommunicator(), MPI_STATUS_IGNORE );
		}
};

// NOTE: this is not a cache command!!
class Finish : public Command<OPID_FINISH> {
};

// NOTE: this is not a cache command!!!
class CreateAuxiliaryThread : public Command<OPID_CREATEAUXTHREAD> {
	public:
		CreateAuxiliaryThread( MPIProcessor const& destination ) :
			Command( destination )
		{
		}

		virtual ~CreateAuxiliaryThread()
		{
		}

		virtual void execute()
		{
      	MPIRemoteNode::nanosMPISend( this, 1, MPIDevice::cacheStruct, getDestination().getRank(), TAG_M2S_ORDER, getDestination().getCommunicator() );
		}
};

} // namespace command
} // namespace mpi
} // namespace nanos

