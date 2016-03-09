
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

template < class CommandPayload, int tag >
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
				std::cout << "MPI_Recv_init" << std::endl;
				MPI_Recv_init( &(*buffer_it), 1, CommandPayload::getDataType(),
				                   MPI_ANY_SOURCE, tag, _communicator, *request_it );
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
				// FIXME: maybe the computed index (using modulo operator) is not correct
				CommandPayload& order = _bufferedCommands.at( (*it)%_bufferedCommands.size() );
				// FIXME This is not correct. Iterator does not use at()
				MPI_Status& status = _statuses.at(*it);

				_pendingCommands.push_back(
				             BaseServant::createSpecific( status.MPI_SOURCE, _communicator, order ) );

				// FIXME This is not correct. Iterator does not use at()
				_requests.at(*it).start();
			}
		}

		void servePendingCommands()
		{
			std::list<BaseServant*>::iterator order_it = _pendingCommands.begin();
			while( order_it != _pendingCommands.end() ) {
				// FIXME: better get the pointer and delete, then call serve()
				// In case we need locks this will reduce the time we keep the lock
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

		typedef detail::SingleDispatcher<CachePayload,TAG_M2S_CACHE_ORDER> command_dispatcher;
		typedef detail::SingleDispatcher<CommandPayload,TAG_M2S_COMMAND>   cache_dispatcher;

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
			persistent_request::start_all( _requests.begin(), _requests.end() );
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
			assert( _readyRequestNumber != MPI_UNDEFINED );
		}

		void queueAvailableCommands()
		{
			// This could be done clearer if we had a tuple of Single dispatchers
			index_storage::iterator start_ready_it = _readyRequestIndices.begin();
			index_storage::iterator end_ready_it = _readyRequestIndices.begin();

			int split_point = 0;
			while( end_ready_it != _readyRequestIndices.end()
			       && split_point < _readyRequestNumber
			       && split_point < _size ) {
				split_point++;
				end_ready_it++;
			}

			_commands.queuePendingCommands(
			                     detail::make_range( start_ready_it, end_ready_it ) );

			start_ready_it = end_ready_it;
			while( end_ready_it != _readyRequestIndices.end()
			       && split_point < _readyRequestNumber
			       /* && split_point < 2*_size */ ) {
				split_point++;
				end_ready_it++;
			}

			_cacheCommands.queuePendingCommands(
			                     detail::make_range( start_ready_it, end_ready_it ));
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
