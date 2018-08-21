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

#ifndef COMMAND_HPP
#define COMMAND_HPP

#include "commandid.hpp"
#include "commandpayload.hpp"

#include "commandrequestor.hpp"
#include "commandservant.hpp"

#include "memoryaddress.hpp"

#include "mpidevice.hpp"
#include "mpiprocessor_decl.hpp"

namespace nanos {
namespace mpi {
namespace command {

using namespace ext;

template < int id, typename Channel >
class CommandRequestor< id, CommandPayload, Channel > {
	private:
		CommandPayload _data;
		Channel        _channel;

	public:
		CommandRequestor( MPIProcessor const& destination ) :
			_data(), _channel( destination )
		{
			_data.initialize(id);
			_channel.send( _data );
		}

		CommandRequestor( MPIProcessor const& destination, int code ) :
			_data(), _channel( destination )
		{
			_data.initialize(id, code);
			_channel.send( _data );
		}

		CommandRequestor( int destination, MPI_Comm communicator, int code ) :
			_data(), _channel( destination, communicator )
		{
			_data.initialize(id, code);
			_channel.send( _data );
		}

		virtual ~CommandRequestor()
		{
		}

		CommandPayload &getData()
		{
			return _data;
		}

		CommandPayload const& getData() const
		{
			return _data;
		}

		// To be defined by each operation
		void dispatch();
};

/**
 * Pairs Requestor and Servant types for each operation id
 */
template < int op_id, int tag = TAG_M2S_COMMAND >
struct Command {
	static const int id;

	typedef CommandPayload	                           payload_type;
	typedef CommandChannel<op_id, CommandPayload, tag> main_channel_type;

	typedef CommandRequestor< op_id, payload_type, main_channel_type > Requestor;
	typedef CommandServant  < op_id, payload_type, main_channel_type > Servant;
};

template< int op_id, int tag >
const int Command<op_id,tag>::id = op_id;

} // namespace command
} // namespace mpi
} // namespace nanos

#endif // COMMAND_HPP

