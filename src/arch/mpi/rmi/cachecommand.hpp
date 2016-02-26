
#ifndef CACHE_COMMAND_HPP
#define CACHE_COMMAND_HPP

#include "commandrequestor.hpp"
#include "commandservant.hpp"
#include "commandchannel.hpp"
#include "cachepayload.hpp"

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
	OPID_INVALID=0,
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

/**
 * Specializes a CommandRequestor for CachePayload.
 * This won't be necessary if we were using C++11 standard, since
 * we can just forward all the arguments for CachePayload construction
 * directly as a template construction taking rvalues and/or lvalues as
 * arguments.
 */
template < int id, typename Channel >
class CommandRequestor< id, CachePayload, Channel > : public Channel {
	private:
		CachePayload _data;

	public:
		CommandRequestor( MPIProcessor const& destination ) :
			Channel( destination ),
			_data( id )
		{
			Channel::sendCommand( _data );
		}

		CommandRequestor( MPIProcessor const& destination, size_t size ) :
			Channel( destination ),
			_data( id, size )
		{
			Channel::sendCommand( _data );
		}

		CommandRequestor( MPIProcessor const& destination, CachePayload const& data ) :
			Channel( destination ),
			_data( data )
		{
			Channel::sendCommand( _data );
		}

		virtual ~CommandRequestor()
		{
		}

		CachePayload &getData()
		{
			return _data;
		}

		CachePayload const& getData() const
		{
			return _data;
		}

		// To be defined by each operation
		void dispatch();
};

/**
 * Pairs Requestor and Servant types for each operation id
 */
template < int op_id, int tag = TAG_M2S_ORDER >
struct CacheCommand {
	typedef CachePayload                             payload_type;
	typedef CommandChannel<op_id, CachePayload, tag> main_channel;

	typedef CommandRequestor< op_id, payload_type, main_channel > Requestor;
	typedef CommandServant  < op_id, payload_type, main_channel > Servant;
};

} // namespace command
} // namespace mpi
} // namespace nanos

#endif // CACHE_COMMAND_HPP

