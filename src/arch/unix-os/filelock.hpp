
#ifndef FILE_MUTEX_HPP
#define FILE_MUTEX_HPP

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/file.h>
#include <fcntl.h>
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

			// assert( _file_descriptor > 0 );
		}

		~FileMutex()
		{
			close(fd);
		}

		void lock()
		{
			fcntl(fd, F_SETLKW, &_lock);
		}

		void unlock()
		{
			fcntl(fd, F_UNLCK, &_lock);
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
