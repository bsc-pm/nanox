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

#ifndef COPYIN_HPP
#define COPYIN_HPP

#include "cachecommand.hpp"
#include "mpiremotenode.hpp"

#include <mpi.h>

namespace nanos {
namespace mpi {
namespace command {

struct CopyIn : public CacheCommand<OPID_COPYIN> {
	typedef CommandChannel<OPID_COPYIN, RawPayload, TAG_CACHE_DATA_IN> transfer_channel_type;
};

/**
 * CopyIn::Requestor
 * Specialization of the CommandRequestor for CopyIn operations, where
 * the MPIProcessor has to be saved for its dispatch, as we need to queue
 * the transfer into MPIProcessor's MPI_Request queue.
 */
template <>
class CommandRequestor<CopyIn::id,CopyIn::payload_type,CopyIn::main_channel_type> {
	private:
		CopyIn::payload_type      _data;
		CopyIn::main_channel_type _channel;
		MPIProcessor             &_remoteProcess;

	public:
		CommandRequestor( MPIProcessor &destination, utils::Address hostAddress, utils::Address deviceAddress, size_t size ) :
			_data(),
			_channel( destination ),
			_remoteProcess( destination )
		{
			_data.initialize( CopyIn::id, MPI_ANY_SOURCE, destination.getRank(), hostAddress, deviceAddress, size );
			_channel.send( _data );
		}

		virtual ~CommandRequestor()
		{
		}

		CopyIn::payload_type &getData()
		{
			return _data;
		}

		CopyIn::payload_type const& getData() const
		{
			return _data;
		}

		void dispatch();
};

/**
 * Send the data to the remote process
 *
 * Appends the pending mpi::request to MPIProcessor request 
 * queue since we don't really need to use the send buffer inmediately.
 */
inline void CopyIn::Requestor::dispatch()
{
	// We transfer the data through a different channel to avoid
	// message collisions (uses a different tag)
	// Source, destination and communicator remain the same
	CopyIn::transfer_channel_type transfer_channel( _channel );

	RawPayload sourceData( _data.getHostAddress() );
	request req = transfer_channel.isend( sourceData, _data.size() );

	_remoteProcess.appendToPendingRequests(req);
}

/**
 * Receive the data from the master process
 */
template<>
// Note: as oposed to CopyIn::Requestor::dispatch(), template<> is needed here
// because we are specializing the function and not defining a function of a
// specialized class.
inline void CopyIn::Servant::serve()
{
	NANOS_MPI_CREATE_IN_MPI_RUNTIME_EVENT(ext::NANOS_MPI_RNODE_COPYIN_EVENT);

	// We transfer the data through a different channel to avoid
	// message collisions (uses a different tag)
	// Source, destination and communicator remain the same
	CopyIn::transfer_channel_type transfer_channel( _channel );

	RawPayload destinationData( _data.getDeviceAddress() );
	transfer_channel.receive( destinationData, _data.size() );

//	TODO: this might need an update to nanos cache v0.9
//	DirectoryEntry *ent = _masterDir->findEntry( (uint64_t) order.devAddr );
//	if (ent != NULL)
//	{
//	   if (ent->getOwner() != NULL) {
//	      ent->getOwner()->invalidate( *_masterDir, (uint64_t) order.devAddr, ent);
//	   } else {
//	      ent->increaseVersion();
//	   }
//	}

	NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT;
}

} // namespace command
} // namespace mpi
} // namespace nanos

#endif // COPY_IN_HPP

