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

#ifndef HOSTINFO_HPP
#define HOSTINFO_HPP

#include "info.hpp"

#include <mpi.h>
#include <string>

namespace nanos {
namespace mpi {

// MPI_Info wrapper class
// Avoids manual allocation/deallocation
// Enable transparent copies easily
// Can return a MPI_Info handle or be casted into it.
class HostInfo : public Info {
private:
	void defaultSettings();

public:
	HostInfo( MPI_Info info ) : Info(info)
	{
	}

	HostInfo( HostInfo const& o ) : Info(o)
	{
	}

	HostInfo() : Info()
	{
		defaultSettings();
	}

	virtual ~HostInfo()
	{
	}
};

} // namespace mpi
} // namespace nanos

#endif //HOSTINFO_HPP
