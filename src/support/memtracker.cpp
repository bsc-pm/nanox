#include "memtracker.hpp"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <map>
#include <limits>
#include <exception>
#include "debug.hpp"
#include "atomic.hpp"

using namespace nanos;
#if 0

class MemTracker
{

      /* Minimal allocator for internal use of MemTracker. It is *NOT* intended for other
	use than supporting an internal std::map that doesn't call recursively the regular allocator*/
      template<typename T>
      class IntAllocator
      {
	public :
	    //    typedefs
	    typedef T value_type;
	    typedef value_type* pointer;
	    typedef const value_type* const_pointer;
	    typedef value_type& reference;
	    typedef const value_type& const_reference;
	    typedef std::size_t size_type;

	public :
	    //    convert an allocator<T> to allocator<U>
	    template<typename U>
	    struct rebind {
	      typedef IntAllocator<U> other;
	    };

	public :
	    inline explicit IntAllocator() {}
	    inline ~IntAllocator() {}
	    /* Dan Tsafrir [11/2/2011]: 'explicit' here, for gcc-4.1, results in compile error */
	    template<typename U>
	    inline /*explicit*/ IntAllocator( const IntAllocator<U> & ) {}

	    //    memory allocation
	    inline pointer allocate( size_type cnt,
				    typename std::allocator<void>::const_pointer = 0 ) {
	      return reinterpret_cast<pointer>( malloc( cnt * sizeof ( T ) ) );
	    }
	    inline void deallocate( pointer p, size_type ) { free( p ); }

	    //    construction/destruction
	    inline void construct( pointer p, const T& t ) { new( p ) T( t ); }
	    inline void destroy( pointer p ) { p->~T(); }
      };

      template<class K, class T>
      struct IntMap {
	typedef std::map<K,T,std::less<K>,IntAllocator<std::pair<const K, T> > > type;
      };

      struct BlockInfo {
	size_t _size;
	const char * _file;
	int    _line;

	BlockInfo ( ) { }
	BlockInfo ( const size_t size, const char *file, const int line ) : _size(size), _file(file), _line(line) {}
      };

      struct DistrInfo {
	size_t _current;
	size_t _max;
	size_t _total;
      };

      typedef IntMap<void *,BlockInfo>::type AddrMap;
      typedef IntMap<size_t,DistrInfo>::type SizeMap;

      AddrMap     _blocks;
      SizeMap     _stats;
      
      size_t      _totalMem;
      size_t      _numBlocks;
      size_t      _maxMem;

      Lock        _lock;

  public:

      MemTracker() : _blocks(),_stats(), _totalMem( 0 ), _numBlocks( 0 ),_maxMem( 0 ), _lock() {}
      ~MemTracker() { showStats(); }

      void *allocate ( size_t size, const char *file = 0, int line = 0 )
      {
         LockBlock guard(_lock);

         void *p = malloc( size );

	if ( p ) {
	    _blocks[p] = BlockInfo(size,file,line);
	    _numBlocks++;
	    _totalMem += size;
	    _stats[size]._current++;
	    _stats[size]._total++;
	    _stats[size]._max = std::max( _stats[size]._max, _stats[size]._current );
	    _maxMem = std::max( _maxMem, _totalMem );
	} else {
	    throw std::bad_alloc();
	}
	
	return p;
      }

      void deallocate ( void * p, const char *file = 0, int line = 0 )
      {
	LockBlock guard(_lock);
	
	AddrMap::iterator it = _blocks.find( p );

	if ( it != _blocks.end() ) {
	    _numBlocks--;
	    _totalMem -= it->second._size;
	    free( p );
	    _blocks.erase( it );
	    _stats[it->second._size]._current--;
	} else {
	    guard.release();
	    
	    if ( file != NULL ) {
	      message0("Trying to free invalid pointer " << p << " at " << file << ":" << line);
	    } else {
	      message0("Trying to free invalid pointer " << p);
	    }    
	}
      }

      void showStats ()
      {
	message0("======================= General Memory stats ============");
	std::cout
	    << "# blocks              " << _numBlocks << std::endl
	    << "total unfreed memory  " << _totalMem << " bytes" << std::endl
	    << "max allocated memory  " << _maxMem << " bytes" << std::endl
	    ;
        message0("=========================================================");
	message0("======================= Unfreed blocks ==================");
	for ( AddrMap::iterator it = _blocks.begin(); it != _blocks.end(); it++ )
	{
	    BlockInfo &info = it->second;
	    if ( info._file != NULL ) {
	      message0(info._size << " bytes allocated in " << info._file << ":" << info._line);
	    } else {
	      message0(info._size << " bytes allocated in an unknown location");
	    }
	}
        message0("=========================================================");
#if 0        
	message0("======================= Block Sizes Stats ===============");
	message0("Size   Unfreed   Max   Total");
	for ( SizeMap::iterator it = _stats.begin(); it != _stats.end(); it ++ ) {
	    DistrInfo &info = it->second;
	    message0(it->first << " " << info._current << " " << info._max << " " << info._total );    
	}
	message0("=========================================================");
#endif        
      }
};

MemTracker *mem = 0;

inline MemTracker & getMemTracker ()
{
   if (!mem) {
      mem = (MemTracker *) malloc(sizeof(MemTracker));
      new (mem) MemTracker();
   }
   return *mem;
}

void* operator new ( size_t size )
{
   return getMemTracker().allocate( size );
}

void* operator new ( size_t size, const char *file, int line )
{
   return getMemTracker().allocate( size, file, line );
}

void* operator new[] ( size_t size, const char *file, int line )
{
   return getMemTracker().allocate( size, file, line );
}

void operator delete ( void *p )
{
   getMemTracker().deallocate( p );
}

void operator delete[] ( void *p )
{
   getMemTracker().deallocate( p );
}

class A {
  public:
    A() { getMemTracker(); }
    ~A() { getMemTracker().showStats(); }
};

A dummy;
#endif
