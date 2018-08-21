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

#ifndef COMMAND_SERVANT_HPP
#define COMMAND_SERVANT_HPP

namespace nanos {
namespace mpi {
namespace command {

struct BaseServant {
	/**
	 * Perform action on the remote process
	 * To be defined by each specific operation
	 */
	virtual void serve() = 0;

	template < typename Payload >
	static BaseServant* createSpecific( int source, int destination, MPI_Comm communicator, Payload const& data );
};

template< int command_id, typename Payload, typename Channel >
class CommandServant : public BaseServant {
	private:
		Payload _data;
		Channel _channel;
	public:
		CommandServant( Channel const& channel ) :
			_data( command_id ),
			_channel( channel )
		{
		}

		CommandServant( Channel const& channel, Payload const& data ) :
			_data( data ), _channel( channel )
		{
		}

		virtual ~CommandServant()
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

		virtual void serve();
};

} // namespace command
} // namespace mpi
} // namespace nanos

#endif // COMMAND_SERVANT_HPP

