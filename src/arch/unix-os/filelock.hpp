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

#ifndef FILE_MUTEX_HPP
#define FILE_MUTEX_HPP

#include "debug.hpp"

#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/file.h>
#include <unistd.h>

namespace nanos {

class FileMutex {
	private:
		int          _file_descriptor;
		struct flock _lock;

		FileMutex( const FileMutex & ); // Non copyable

		FileMutex& operator=( const FileMutex & ); // Non assignable
	public:
		FileMutex( const char* name ) :
			_file_descriptor( open( name, O_RDWR|O_CREAT,0666) ),
			_lock()
		{
			_lock.l_type   = F_WRLCK;
			_lock.l_start  = 0;
			_lock.l_whence = SEEK_SET;
			_lock.l_len    = 0;

			fatal_cond0( _file_descriptor == -1, "Could not open lock file: " << strerror(errno) );
		}

		~FileMutex()
		{
			close( _file_descriptor );
		}

		void lock()
		{
			int err = fcntl( _file_descriptor, F_SETLKW, &_lock);
			fatal_cond0( err != 0, "Failed to lock file: " << strerror(errno) );
		}

		void unlock()
		{
			int err = fcntl( _file_descriptor, F_UNLCK, &_lock);
			fatal_cond0( err != 0, "Failed to unlock file:" << strerror(errno) );
		}

		/* Unavailable
		bool try_lock()
		{
		}*/

		int native_handle()
		{
			return _file_descriptor;
		}
};

}

#endif // FILE_MUTEX_HPP
