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

#ifndef CACHE_PAYLOAD_HPP
#define CACHE_PAYLOAD_HPP

#include "memoryaddress.hpp"
#include "commandid.hpp"

#include <cstddef>
#include <mpi.h>

namespace nanos {
namespace mpi {
namespace command {

class CachePayload {
	private:
		int       _id;
		int       _source;
		int       _destination;
		uintptr_t _hostAddress;
		uintptr_t _deviceAddress;
		size_t    _size;

		static MPI_Datatype _type;

	public:
		/**
		 * \brief default initializes the instance
		 *
		 * This function acts as a constructor.
		 * Since CachePayload has to fulfill
		 * POD (plain old data) restrictions, it
		 * can not contain user defined constructors
		 * or destructors
		 */
		void initialize()
		{
			_id = OPID_INVALID;
			_source = MPI_ANY_SOURCE;
			_destination = MPI_PROC_NULL;
			_hostAddress = 0;
			_deviceAddress = 0;
			_size = 0;
		}

		void initialize( int id )
		{
			initialize();
			_id = id;
		}

		void initialize( int id, size_t buffer_size )
		{
			initialize(id);
			_size = buffer_size;
		}

		void initialize( int id, utils::Address hostAddr, utils::Address deviceAddr, size_t buffer_size )
		{
			initialize(id,buffer_size);
			_hostAddress = hostAddr;
			_deviceAddress = deviceAddr;
		}

		void initialize( int id, int source, int destination, utils::Address hostAddress, utils::Address deviceAddress, size_t buffer_size )
		{
			initialize(id, hostAddress, deviceAddress, buffer_size );
			_source = source;
			_destination = destination;
		}

		int getSource() const
		{
			return _source;
		}

		void setSource( int source )
		{
			_source = source;
		}

		int getDestination() const
		{
			return _destination;
		}

		void setDestination( int destination )
		{
			_destination = destination;
		}

		utils::Address getHostAddress () const
		{
			return _hostAddress;
		}

		void setHostAddress ( utils::Address address )
		{
			_hostAddress = address;
		}

		utils::Address getDeviceAddress() const
		{
			return _deviceAddress;
		}

		void setDeviceAddress ( utils::Address address )
		{
			_deviceAddress = address;
		}

		size_t size() const
		{
			return _size;
		}

		void setSize( size_t buffer_size )
		{
			_size = buffer_size;
		}

		int getId() const
		{
			return _id;
		}

		static MPI_Datatype getDataType()
		{
			return _type;
		}

		static void initDataType()
		{
			MPI_Datatype typelist[3] = {
					MPI_INT,
					MPI_BYTE,
					MPI_BYTE };

			int blocklen[3] = {
					3,
					2*sizeof(utils::Address),
					sizeof(size_t) };

			MPI_Aint disp[3] = { 
					offsetof(CachePayload,_id),
					offsetof(CachePayload,_hostAddress),
					offsetof(CachePayload,_size) };

			MPI_Type_create_struct(3, blocklen, disp, typelist, &_type );
			MPI_Type_commit( &_type );		
		}

		static void freeDataType()
		{
			MPI_Type_free( &_type );
		}
};

} // namespace command
} // namespace mpi
} // namespace nanos

#endif

