
#ifndef CACHE_DISPATCHER_HPP
#define CACHE_DISPATCHER_HPP

#include "cachecommand.hpp"
#include "smartpointer.hpp"

#include <mpi.h>

namespace nanos {
namespace mpi {

namespace detail {

/* Assumes Container can access elements randomly */
template < typename Container >
class range {
	private:		
		typename Container::value_type  value_type;
		typename Container:.value_type &reference_type;
		typename Container              storage_type;
		typename Container::iterator    iterator_type;

		const std::pair<iterator_type, iterator_type> _iterators;

	public:
		iterator_pair( iterator const& first, iterator const& last ) :
			_iterators( first, last )
		{
		}

		iterator_pair( iterator const& first, size_t distance ) :
			_iterators( first, first+distance )
		{
		}

		~iterator_pair()
		{
		}

		iterator_type begin()
		{
			return _iterators.first;
		}

		iterator_type end()
		{
			return _iterators.second;
		}

		size_t distance()
		{
			return std::distance( begin(), end() );
		}

		reference_type at( size_t position )
		{
			//assert( position < distance() );
			return *( begin() + position );
		}
};

template < class CommandType >
class SingleDispatcher {
	private:
		typedef std::vector<persistent_request> request_storage;
		typedef std::vector<MPI_Status>         status_storage;

		MPI_Comm                 _communicator;
		std::vector<CommandType> _bufferedCommands;

		const iterator_pair<request_storage>   _requests;
		const iterator_pair<status_storage>    _statuses;

		std::forward_list<CommandType*>  _pendingCommands;

	public:
		SingleDispatcher( MPI_Comm communicator, size_t size, 
		            iterator_pair<request_storage> requests,
		            iterator_pair<status_storage>  statuses ) :
			_communicator( communicator ), _bufferedCommands( size ), 
			_requests( requests_begin, size ), _statuses( statuses_begin, size ),
			_pendingCommands()
		{
			// assert( size == requests.distance() );
			for( request_storage::iterator request_it = _requests.begin(); request_it != _requests.end() ) {
				MPI_Recv_init( &_bufferedCommands[i], 1, CommandType::getDataType(),
				                   MPI_ANY_SOURCE, TAG_M2S_ORDER, _communicator, &(*_request_it) );
			}
		}

		virtual ~SingleDispatcher()
		{
			if( !_pendingCommands.empty() )
				servePendingCommands();

			_bufferedCommands.clear();
		}

		template< typename Container >
		void queuePendingCommands( iterator_pair<Container> indices )
		{
			for( Container::iterator it = indices.begin(); it != indices.end(); it++ ) {
				CommandType &order = _bufferedCommands.at( *it );

				order.setCommunicator( _communicator );
				_pendingCommands.push_back( CommandType::createSpecific( order ) );

				_requests.at(*it).start();
			}
		}

		void servePendingCommands()
		{
			std::forward_list<CommandType*>::iterator order_it = orders.begin();
			while( order_it != orders.end() ) {
				(*order_it)->serve();
				orders.erase( order_it++ );
			}
		}
};

} // namespace detail

class Dispatcher {
	private:
		typedef std::vector<persistent_request>               request_storage;
		typedef std::vector<MPI_Status>                       status_storage;
		typedef std::vector<int>                              index_storage;

		typedef detail::SingleDispatcher<GenericCommand>      command_dispatcher;
		typedef detail::SingleDispatcher<GenericCacheCommand> cache_dispatcher;

		MPI_Comm                         _communicator;
		size_t                           _size;
		size_t                           _readyNumber;
		index_storage                    _readyIndices;

		request_storage                  _requests;
		status_storage                   _statuses;

		// This may better be a vector of abstract pointers
		// and then we just iterate over the array as needed
		command_dispatcher _commands;
		cache_dispatcher   _cacheCommands;

		// Just a nasty replacement for template int and variadic commandtypes
		static const num_command_types = 2;

	public:
		Dispatcher( MPI_Comm communicator, size_t size ) :
			_communicator( communicator ), _size( size ),
			_requests( num_command_types * size ),
			_statuses( num_command_types * size ),
			_ready( num_command_types * size, -1 ),
			_commands( communicator, size, request_pair( _requests.begin(), size ),
			                               status_pair( _statuses.begin(), size ) ),
			_cacheCommands( communicator, size, request_pair( _requests.begin()+size, size ),
			                               status_pair( _statuses.begin()+size, size ) )
		{
		}

		virtual ~Dispatcher()
		{
		}

		void waitForCommands()
		{
			MPI_Waitsome( _size, _requests[0], &_readyNumber, &_readyIndices[0], &_statuses[0] );
		}

		void queueAvailableCommands()
		{
			index_storage::iterator ready_it = _readyIndices.begin();

			int split_point = 0;
			// Search for the split point
			// _readyNumber value will always be lower readyIndices size
			while( (split_point < _readyNumber) && (*ready_it < _size) ) {
				ready_it++;
				split_point++;
			}

			_commands.queuePendingCommands( iterator_pair( _readyIndices.begin(), _readyIndices.begin()+split_point ) );
			_cacheCommands.queuePendingCommands( iterator_pair( _readyIndices.begin()+split_point, _readyIndices.begin()+_readyNumber ) );
		}

		void executeCommands()
		{
			_commands.servePendingCommands();
			_cacheCommands.servePendingCommands();
		}
};

} // namespace mpi
} // namespace nanos

#endif
