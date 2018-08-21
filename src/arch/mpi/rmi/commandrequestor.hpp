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

#ifndef COMMAND_REQUESTOR_HPP
#define COMMAND_REQUESTOR_HPP

#include "mpidevice.hpp"
#include "commandchannel.hpp"

namespace nanos {
namespace mpi {
namespace command {

template< int command_id, typename Payload, typename Channel >
class CommandRequestor {
	private:
		Payload _data;
		Channel _channel;

	public:
		CommandRequestor( Channel const& channel ) :
			_data(),
			_channel( channel )
		{
			_data.initialize(command_id);
			_channel.send( _data );
		}

		virtual ~CommandRequestor()
		{
		}

		Payload &getData()
		{
			return _data;
		}

		Payload const& getData() const
		{
			return _data;
		}

		Channel const& getChannel() const
		{
			return _channel;
		}

		/**
		 * Action to be executed after command is sent
		 * To be defined by each specific operation
		 */
		void dispatch();
};

} // namespace command
} // namespace mpi
} // namespace nanos

#endif // COMMAND_REQUESTOR_HPP

