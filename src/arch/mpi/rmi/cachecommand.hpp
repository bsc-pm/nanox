
#ifndef CACHE_COMMAND_HPP
#define CACHE_COMMAND_HPP

#include "commandid.hpp"
#include "commandrequestor.hpp"
#include "commandservant.hpp"
#include "commandchannel.hpp"
#include "cachepayload.hpp"

#include "memoryaddress.hpp"

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
class CommandRequestor< id, CachePayload, Channel > {
	private:
		CachePayload _data;
		Channel      _channel;

	public:
		CommandRequestor( MPIProcessor const& destination ) :
			_data(), _channel( destination )
		{
			_data.initialize( id );
			_channel.send( _data );
		}

		CommandRequestor( MPIProcessor const& destination, size_t size ) :
			_data(), _channel( destination )
		{
			_data.initialize( id, size );
			_channel.send( _data );
		}

		CommandRequestor( MPIProcessor const& destination,
		                  utils::Address hostAddr, utils::Address deviceAddr,
		                  size_t size ) :
			_data(), _channel( destination )
		{
			_data.initialize( id, hostAddr, deviceAddr, size );
			_channel.send( _data );
		}

		CommandRequestor( MPIProcessor const& destination, CachePayload const& data ) :
			_data(data), _channel( destination )
		{
			_channel.send( _data );
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
template < int op_id, int tag = TAG_M2S_CACHE_COMMAND >
struct CacheCommand {
	static const int id;
	typedef CachePayload                             payload_type;
	typedef CommandChannel<op_id, CachePayload, tag> main_channel_type;

	typedef CommandRequestor< op_id, payload_type, main_channel_type > Requestor;
	typedef CommandServant  < op_id, payload_type, main_channel_type > Servant;
};

template< int op_id, int tag >
const int CacheCommand<op_id,tag>::id = op_id;

} // namespace command
} // namespace mpi
} // namespace nanos

#endif // CACHE_COMMAND_HPP

