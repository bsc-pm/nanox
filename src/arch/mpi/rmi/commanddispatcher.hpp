
#ifndef COMMAND_DISPATCHER_HPP
#define COMMAND_DISPATCHER_HPP

#include "commandservant.hpp"
#include "cachepayload.hpp"
#include "commandpayload.hpp"

#include <mpi.h>
#include <list>
#include <iostream>

namespace nanos {
namespace mpi {
namespace command {
namespace detail {

/* Assumes Iterator's container can access elements randomly */
template < typename Iterator >
class iterator_range {
	private:		
		typedef typename Iterator::value_type  value_type;
		typedef typename Iterator::value_type& reference_type;

		const std::pair<Iterator, Iterator> _iterators;

	public:
		iterator_range( Iterator const& first, Iterator const& last ) :
			_iterators( first, last )
		{
		}

		iterator_range( Iterator const& first, size_t distance ) :
			_iterators( first, first+distance )
		{
		}

		~iterator_range()
		{
		}

		Iterator begin() const
		{
			return _iterators.first;
		}

		Iterator end() const
		{
			return _iterators.second;
		}

		size_t distance() const
		{
			return std::distance( begin(), end() );
		}

		reference_type at( size_t position ) const
		{
			//assert( position < distance() );
			return *( begin() + position );
		}
};

template < typename Iterator >
iterator_range<Iterator> make_range( Iterator const& begin, Iterator const& end )
{
	return iterator_range<Iterator>(begin,end);
}

template < typename Iterator >
iterator_range<Iterator> make_range( Iterator const& begin, size_t size )
{
	return iterator_range<Iterator>(begin,size);
}

template < class CommandPayload >
class SingleDispatcher {
	private:
		typedef std::vector<persistent_request>      request_storage;
		typedef std::vector<MPI_Status>              status_storage;
		typedef typename std::vector<CommandPayload> buffer_storage;

		MPI_Comm                    _communicator;
		std::vector<CommandPayload> _bufferedCommands;

		const iterator_range<request_storage::iterator>   _requests;
		const iterator_range<status_storage::iterator>    _statuses;

		std::list<BaseServant*>  _pendingCommands;

	public:
		SingleDispatcher( MPI_Comm communicator, size_t size, 
		            iterator_range<request_storage::iterator> request_range,
		            iterator_range<status_storage::iterator>  status_range ) :
			_communicator( communicator ), _bufferedCommands( size ), 
			_requests( request_range ), _statuses( status_range ),
			_pendingCommands()
		{
			// assert( size == requests.distance() );
			typename buffer_storage::iterator buffer_it = _bufferedCommands.begin();
			request_storage::iterator request_it;
			for( request_it = _requests.begin(); request_it != _requests.end(); request_it++ ) {
				std::cout << "MPI_Recv_init request: " << std::hex << &(*request_it) << " buffer: 0x" << std::hex << &(*buffer_it) << " oldvalue: " << *request_it;
				MPI_Recv_init( &(*buffer_it), 1, CommandPayload::getDataType(),
				                   MPI_ANY_SOURCE, TAG_M2S_ORDER, _communicator, *request_it );
				std::cout << " newvalue: " << *request_it << std::endl;
				buffer_it++;
			}
		}

		virtual ~SingleDispatcher()
		{
			// Serve all commands that were not previously
			// executed.
			if( !_pendingCommands.empty() )
				servePendingCommands();

			// Clear the request list
			// Does not wait for pending requests since
			// it might issue more receives than actual messages
			// are received.
			_bufferedCommands.clear();
		}

		template< typename Iterator >
		void queuePendingCommands( iterator_range<Iterator> indices )
		{
			Iterator it;
			for( it = indices.begin(); it != indices.end(); it++ ) {
				CommandPayload& order = _bufferedCommands.at( *it );
				MPI_Status& status = _statuses.at(*it);

				_pendingCommands.push_back(
				             BaseServant::createSpecific( status.MPI_SOURCE, _communicator, order ) );

				_requests.at(*it).start();
			}
		}

		void servePendingCommands()
		{
			std::list<BaseServant*>::iterator order_it = _pendingCommands.begin();
			while( order_it != _pendingCommands.end() ) {
				(*order_it)->serve();
				_pendingCommands.erase( order_it++ );
			}
		}
};

} // namespace detail

class Dispatcher {
	private:
		typedef std::vector<persistent_request>               request_storage;
		typedef std::vector<MPI_Status>                       status_storage;
		typedef std::vector<int>                              index_storage;

		typedef detail::SingleDispatcher<CachePayload>        command_dispatcher;
		typedef detail::SingleDispatcher<CommandPayload>      cache_dispatcher;

		MPI_Comm                         _communicator;
		int                              _size;
		int                              _readyRequestNumber;
		index_storage                    _readyRequestIndices;

		request_storage                  _requests;
		status_storage                   _statuses;

		// This may better be a vector of abstract pointers
		// and then we just iterate over the array as needed
		command_dispatcher               _commands;
		cache_dispatcher                 _cacheCommands;

		// Just a temporary replacement for template int and variadic commandtypes
		// since they are not available in c++0x
		static const int num_command_types = 2;

	public:
		Dispatcher( MPI_Comm communicator, size_t size ) :
			_communicator( communicator ), _size( size ),
			_readyRequestNumber(0),
			_readyRequestIndices( num_command_types * size, -1 ),
			_requests( num_command_types * size ),
			_statuses( num_command_types * size ),
			_commands( communicator, size,
			           detail::make_range( _requests.begin(), size ),
			           detail::make_range( _statuses.begin(), size )),
			_cacheCommands( communicator, size,
			           detail::make_range( _requests.begin()+size, size ),
			           detail::make_range( _statuses.begin()+size, size ))
		{
		}

		virtual ~Dispatcher()
		{
			std::cout << "Clearing pending requests" << std::endl;
			request_storage::iterator request_it;
			for( request_it = _requests.begin(); request_it != _requests.end(); request_it++ )
			{
				bool completed = request_it->test();
				if( !completed )
					request_it->cancel();
				request_it->free();
			}
		}

		void waitForCommands()
		{
			std::cout << "Wait some" << std::endl;
			MPI_Waitsome( _size*num_command_types, _requests[0], &_readyRequestNumber, &_readyRequestIndices[0], &_statuses[0] );
		}

		void queueAvailableCommands()
		{
			index_storage::iterator ready_it = _readyRequestIndices.begin();

			int split_point = 0;
			// Search for the split point
			// _readyRequestNumber value will always be lower readyRequestIndices size
			while( (split_point < _readyRequestNumber) && (*ready_it < _size) ) {
				ready_it++;
				split_point++;
			}

			_commands.queuePendingCommands(
			                     detail::make_range( _readyRequestIndices.begin(),
			                                         _readyRequestIndices.begin()+split_point ) );
			_cacheCommands.queuePendingCommands(
			                     detail::make_range( _readyRequestIndices.begin()+split_point,
			                                         _readyRequestIndices.begin()+_readyRequestNumber ));
		}

		void executeCommands()
		{
			_commands.servePendingCommands();
			_cacheCommands.servePendingCommands();
		}
};

} // namespace command
} // namespace mpi
} // namespace nanos

#endif // COMMAND_DISPATCHER_HPP
