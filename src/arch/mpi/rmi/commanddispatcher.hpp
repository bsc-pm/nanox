/*************************************************************************************/
/*      Copyright 2009-2018 Barcelona Supercomputing Center                          */
/*                                                                                   */
/*      This file is part of the NANOS++ library.                                    */
/*                                                                                   */
/*      NANOS++ is free software: you can redistribute it and/or modify              */
/*      it under the terms of the GNU Lesser General Public License as published by  */
/*      the Free Software Foundation, either version 3 of the License, or            */
/*      (at your option) any later version.                                          */
/*                                                                                   */
/*      NANOS++ is distributed in the hope that it will be useful,                   */
/*      but WITHOUT ANY WARRANTY; without even the implied warranty of               */
/*      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                */
/*      GNU Lesser General Public License for more details.                          */
/*                                                                                   */
/*      You should have received a copy of the GNU Lesser General Public License     */
/*      along with NANOS++.  If not, see <https://www.gnu.org/licenses/>.            */
/*************************************************************************************/

#ifndef COMMAND_DISPATCHER_HPP
#define COMMAND_DISPATCHER_HPP

#include "commanddispatcher_decl.hpp"

#include "commandservant.hpp"
#include "cachepayload.hpp"
#include "commandpayload.hpp"

#include "concurrent_queue.hpp"
#include "lock.hpp"

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

		iterator_range( Iterator const& first, size_t num_elements ) :
			_iterators( first, first+num_elements )
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
inline iterator_range<Iterator> make_range( Iterator const& begin, Iterator const& end )
{
	return iterator_range<Iterator>(begin,end);
}

template < typename Iterator >
inline iterator_range<Iterator> make_range( Iterator const& begin, size_t size )
{
	return iterator_range<Iterator>(begin,size);
}

template < class CommandPayload, int tag >
class SingleDispatcher {
	private:
		typedef std::vector<persistent_request>      request_storage;
		typedef std::vector<MPI_Status>              status_storage;
		typedef typename std::vector<CommandPayload> buffer_storage;

		int                                               _rank;
		MPI_Comm                                          _communicator;
		std::vector<CommandPayload>                       _bufferedCommands;

		ConcurrentQueue<BaseServant*>                     _pendingCommands;

		Lock                                              _headLock;
		Lock                                              _tailLock;

	public:
		SingleDispatcher( MPI_Comm communicator, size_t size, 
		            iterator_range<request_storage::iterator> request_range ) :
			_rank(MPI_PROC_NULL), _communicator( communicator ), _bufferedCommands( size ),
			_pendingCommands()
		{
			MPI_Comm_rank(_communicator, &_rank);
			typename buffer_storage::iterator buffer_it = _bufferedCommands.begin();
			request_storage::iterator request_it;
			for( request_it = request_range.begin(); request_it != request_range.end(); request_it++ ) {
				MPI_Recv_init( &(*buffer_it), 1, CommandPayload::getDataType(),
				                   MPI_ANY_SOURCE, tag, _communicator, *request_it );
				buffer_it++;
			}
		}

		virtual ~SingleDispatcher()
		{
			// Serve all commands that were not previously
			// executed.
			servePendingCommands();

			// Clear the request list
			// Does not wait for pending requests since
			// it might issue more receives than actual messages
			// are received.
			_bufferedCommands.clear();
		}

		void queueCommand( size_t commandIndex, MPI_Status const& status )
		{
			CommandPayload& order = _bufferedCommands.at( commandIndex );
			_pendingCommands.push( BaseServant::createSpecific( status.MPI_SOURCE, _rank, _communicator, order ) );
		}

		void servePendingCommands()
		{
			BaseServant* command;
			bool commandsRemain = _pendingCommands.pop( command );
			while( commandsRemain ) {
				command->serve();
				commandsRemain = _pendingCommands.pop( command );
			}
		}
};

} // namespace detail

class Dispatcher {
	private:
		typedef std::vector<persistent_request>               request_storage;
		typedef std::vector<MPI_Status>                       status_storage;
		typedef std::vector<int>                              index_storage;

		typedef detail::SingleDispatcher<CommandPayload,TAG_M2S_COMMAND>     command_dispatcher;
		typedef detail::SingleDispatcher<CachePayload,TAG_M2S_CACHE_COMMAND> cache_dispatcher;

		MPI_Comm                         _communicator;
		int                              _size;
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
			_readyRequestIndices( num_command_types * size, -1 ),
			_requests( num_command_types * size ),
			_statuses( num_command_types * size ),
			_commands( communicator, size,
			           detail::make_range( _requests.begin(), size )),
			_cacheCommands( communicator, size,
			           detail::make_range( _requests.begin()+size, size ))
		{
			persistent_request::start_all( _requests.begin(), _requests.end() );
		}

		virtual ~Dispatcher()
		{
			_readyRequestIndices.clear();
			_statuses.resize( _requests.size() );

			size_t s = 0;
			for( size_t r = 0; r < _requests.size(); ++r ) {
				bool finished = _requests[r].test( _statuses[s] );
				if( !finished ) {
					_requests[r].cancel();
				} else {
					_readyRequestIndices.push_back(r);
					++s;
				}
				_requests[r].free();
			}
			_statuses.resize( _readyRequestIndices.size() );

			queueAvailableCommands( false );
			executeCommands();
		}

		void testForCommands()
		{
			_readyRequestIndices = mpi::request::test_some( _requests, _statuses );
		}

		void waitForCommands()
		{
			_readyRequestIndices = mpi::request::wait_some( _requests, _statuses );
		}

		void queueAvailableCommands( bool restartRequest = true )
		{
			for( size_t r = 0; r < _readyRequestIndices.size(); r++ ) {
				int index = _readyRequestIndices.at(r);
				// Create the command specialization
				// TODO: Maybe there is a better way to do this
				//       Tried using different loops but I was
				//       not very convinced.
				// Note that index is always between 0 and
				// (_size * num_command_types - 1 )
				// Warning: statuses in array_of_statuses are contiguous
				// they must be indexed by 'r' instead of 'index'
				if( index < _size ) {
					_commands.queueCommand( index, _statuses.at(r) );
				} else {
					_cacheCommands.queueCommand( (index-_size), _statuses.at(r) );
				}
				// and restart the request
				if( restartRequest )
					_requests.at(index).start();
			}
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
