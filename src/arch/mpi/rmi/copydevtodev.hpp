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

#ifndef COPY_DEV_TO_DEV_HPP
#define COPY_DEV_TO_DEV_HPP

#include "cachecommand.hpp"
#include "mpiremotenode.hpp"

#include <mpi.h>

namespace nanos {
namespace mpi {
namespace command {

struct CopyDeviceToDevice : public CacheCommand<OPID_DEVTODEV> {
	typedef CommandChannel<OPID_DEVTODEV, RawPayload, TAG_CACHE_DEV2DEV> transfer_channel_type;
};

/**
 * CopyDeviceToDevice::Requestor
 * Specialization of the CommandRequestor for CopyIn operations, where
 * the MPIProcessor has to be saved for its dispatch, as we need to queue
 * the transfer into MPIProcessor's MPI_Request queue.
 */
template <>
class CommandRequestor<CopyDeviceToDevice::id,CopyDeviceToDevice::payload_type,CopyDeviceToDevice::main_channel_type> {
	private:
		CopyDeviceToDevice::payload_type      _data;
		CopyDeviceToDevice::main_channel_type _channelSource;
		CopyDeviceToDevice::main_channel_type _channelDestination;

	public:
		CommandRequestor( MPIProcessor &source, MPIProcessor &destination,
		                  utils::Address sourceAddr, utils::Address destinationAddr,
		                  size_t size ) :
			_data(),
			_channelSource( source ),
			_channelDestination( destination )
		{
			_data.initialize( CopyDeviceToDevice::id, source.getRank(), destination.getRank(),
			       sourceAddr, destinationAddr, size );

			std::vector<mpi::request> reqs(2);
			reqs[0] = _channelSource.isend( _data );
			reqs[1] = _channelDestination.isend( _data );
			mpi::request::wait_all( reqs.begin(), reqs.end() );
		}

		virtual ~CommandRequestor()
		{
		}

		CopyDeviceToDevice::payload_type &getData()
		{
			return _data;
		}

		CopyDeviceToDevice::payload_type const& getData() const
		{
			return _data;
		}

		void dispatch();
};

/**
 * No additional actions required. Just send the message.
 */
inline void CopyDeviceToDevice::Requestor::dispatch()
{
}

/**
 * The process that matches with sourceRank has to send the data
 * to the one which matches with destinationRank
 * Note: device to device copies are only allowed when both source
 *       and destination processes are created with the same
 *       MPI_Comm_spawn (i.e. they share the same intercommunicator).
 *       Since the intercommunicator is used to communicate with the
 *       process in the other end (masters), this function must make use
 *       of MPI_COMM_WORLD instead (communicate between process in the same
 *       end: slaves).
 */
template<>
inline void CopyDeviceToDevice::Servant::serve()
{
	int myRank;
	MPI_Comm communicator = MPI_COMM_WORLD;
	MPI_Comm_rank( communicator, &myRank );

	CopyDeviceToDevice::transfer_channel_type transfer_channel( _data.getSource(),
	                                                            _data.getDestination(),
	                                                            communicator );

	if ( myRank == _data.getSource() ) {
		// This process has to send the data
		NANOS_MPI_CREATE_IN_MPI_RUNTIME_EVENT(ext::NANOS_MPI_RNODE_DEV2DEV_OUT_EVENT);

		RawPayload destinationData( _data.getHostAddress() );
		transfer_channel.send( destinationData, _data.size() );

		NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT;

	} else if ( myRank == _data.getDestination() ) {
		// This process has to receive the data
		NANOS_MPI_CREATE_IN_MPI_RUNTIME_EVENT(ext::NANOS_MPI_RNODE_DEV2DEV_IN_EVENT);

//		TODO: this might need an update to nanos cache v0.9
//
//		DirectoryEntry *ent = _masterDir->findEntry( (uint64_t) order.devAddr );
//		if (ent != NULL)
//		{
//		   if (ent->getOwner() != NULL) {
//		      ent->getOwner()->invalidate( *_masterDir, (uint64_t) order.devAddr, ent);
//		   } else {
//		      ent->increaseVersion();
//		   }
//		}

		RawPayload destinationData( _data.getDeviceAddress() );
		transfer_channel.receive( destinationData, _data.size() );

		NANOS_MPI_CLOSE_IN_MPI_RUNTIME_EVENT;
	}
}

} // namespace command
} // namespace mpi
} // namespace nanos

#endif // COPY_DEV_TO_DEV

