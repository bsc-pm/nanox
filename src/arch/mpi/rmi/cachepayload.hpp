
#ifndef CACHE_PAYLOAD_HPP
#define CACHE_PAYLOAD_HPP

#include "memoryaddress.hpp"
#include "commandid.hpp"

#include <mpi.h>

namespace nanos {
namespace mpi {
namespace command {

class CachePayload {
	private:
		int             _id;
		int             _source;
      int             _destination;
		utils::Address  _hostAddress;
		utils::Address  _deviceAddress;
		size_t          _size;

		static MPI_Datatype _type;

	public:
		CachePayload() :
			_id( OPID_INVALID ), _source( MPI_ANY_SOURCE ), _destination( MPI_PROC_NULL ),
			_hostAddress( utils::Address::uninitialized() ), _deviceAddress( utils::Address::uninitialized() ),
			_size( 0 )
		{
		}

		CachePayload( int id ) :
			_id( id ), _source( MPI_ANY_SOURCE ), _destination( MPI_PROC_NULL ),
			_hostAddress( utils::Address::uninitialized() ), _deviceAddress( utils::Address::uninitialized() ),
			_size( 0 )
		{
		}

		CachePayload( int id, size_t size ) :
			_id( id ), _source( MPI_ANY_SOURCE ), _destination( MPI_PROC_NULL ),
			_hostAddress( utils::Address::uninitialized() ), _deviceAddress( utils::Address::uninitialized() ),
			_size( size )
		{
		}

		CachePayload( int id, utils::Address hostAddr, utils::Address deviceAddr, size_t size ) :
			_id( id ), _source( MPI_ANY_SOURCE ), _destination( MPI_PROC_NULL ),
			_hostAddress( hostAddr ), _deviceAddress( deviceAddr ),
			_size( size )
		{
		}

		CachePayload( int id, int source, int destination, utils::Address hostAddress, utils::Address deviceAddress, size_t size ) :
			_id( id ), _source( source ), _destination( destination ),
			_hostAddress( hostAddress ), _deviceAddress( deviceAddress ), _size( size )
		{
		}

		~CachePayload()
		{
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

		void setSize( size_t size )
		{
			_size = size;
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

