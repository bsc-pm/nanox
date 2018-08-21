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

#ifndef FINISH_HPP
#define FINISH_HPP

#include "atomic_flag.hpp"
#include "command.hpp"

#include "mpiremotenode_decl.hpp"

namespace nanos {
namespace mpi {
namespace command {

typedef Command<OPID_FINISH> Finish;

template<>
class CommandServant<
    Finish::id,
    Finish::payload_type,
    Finish::main_channel_type
                      > : public BaseServant {
	private:
		typedef Finish::payload_type payload_type;
		typedef Finish::main_channel_type main_channel_type;

		payload_type        _data;
		main_channel_type   _channel;

		static atomic_flag  _finished;

	public:
		CommandServant( const main_channel_type& channel ) :
			_data(),
			_channel( channel )
		{
			_data.initialize( Finish::id );
		}

		CommandServant( const main_channel_type& channel, const payload_type& data ) :
			_data( data ), _channel( channel )
		{
		}

		virtual ~CommandServant()
		{
		}

		payload_type &getData()
		{
			return _data;
		}

		payload_type const& getData() const
		{
			return _data;
		}

		virtual void serve();

		static bool isFinished()
		{
			return _finished.load();
		}
};

atomic_flag Finish::Servant::_finished;

template<>
inline void Finish::Requestor::dispatch()
{
}

inline void Finish::Servant::serve()
{
	bool alreadyFinalized = _finished.test_and_set();
	if( !alreadyFinalized )
		ext::MPIRemoteNode::addTaskToQueue( TASK_END_PROCESS, _channel.getSource() );
}

} // namespace command
} // namespace mpi
} // namespace nanos

#endif // FINISH_HPP

