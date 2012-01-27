#ifndef MEMORYMAP_DECL_H
#define MEMORYMAP_DECL_H

#include <map>
#include <list>
#include <stdint.h>

namespace nanos {

class MemoryChunk {
   private:
      uint64_t _addr;
      std::size_t _len;
   public:
      typedef enum {
         NO_OVERLAP,
         BEGIN_OVERLAP,
         END_OVERLAP,
         TOTAL_OVERLAP,
         SUBCHUNK_OVERLAP,
         TOTAL_BEGIN_OVERLAP,
         SUBCHUNK_BEGIN_OVERLAP,
         TOTAL_END_OVERLAP,
         SUBCHUNK_END_OVERLAP
       } OverlapType;
      
      static char const * strOverlap[];

      MemoryChunk( uint64_t addr, std::size_t len ) : _addr( addr ), _len( len ) { }
      MemoryChunk( MemoryChunk const &mc ) : _addr( mc._addr ), _len( mc._len ) { }
      MemoryChunk( ) : _addr( 0 ), _len( 0 ) { }

      MemoryChunk& operator=( MemoryChunk const &mc );
      bool operator<( MemoryChunk const &chunk ) const;

      uint64_t getAddress() const;
      std::size_t getLength() const;
      OverlapType checkOverlap( MemoryChunk const &target ) const;
      bool equal( MemoryChunk const &target ) const;
      bool contains( MemoryChunk const &target ) const;
      void expandIncluding( MemoryChunk const &mcB );
      void expandExcluding( MemoryChunk const &mcB );
      void cutAfter( MemoryChunk const &mc );

      static void intersect( MemoryChunk &mcA, MemoryChunk &mcB, MemoryChunk &mcC );
      static void partition( MemoryChunk &mcA, MemoryChunk const &mcB, MemoryChunk &mcC );
      static void partitionBeginAgtB( MemoryChunk &mcA, MemoryChunk &mcB );
      static void partitionBeginAltB( MemoryChunk const &mcA, MemoryChunk &mcB );
      static void partitionEnd( MemoryChunk &mcA, MemoryChunk const &mcB );
};

template <typename _Type>
class MemoryMap : public std::map< MemoryChunk, _Type * > { 
   private:
      typedef std::map< MemoryChunk, _Type * > BaseMap;

      MemoryMap( const MemoryMap &mm ) { }
      const MemoryMap & operator=( const MemoryMap &mm ) { }
   
   public:
      typedef enum { MEM_CHUNK_FOUND, MEM_CHUNK_NOT_FOUND, MEM_CHUNK_NOT_FOUND_BUT_ALLOCATED } QueryResult;
      typedef std::pair< const MemoryChunk *, _Type ** > MemChunkPair;
      typedef std::list< MemChunkPair > MemChunkList;
      typedef std::pair< const MemoryChunk *, _Type * const * > ConstMemChunkPair;
      typedef std::list< ConstMemChunkPair > ConstMemChunkList;
      typedef typename BaseMap::iterator iterator;
      typedef typename BaseMap::const_iterator const_iterator;

      MemoryMap() { }
      ~MemoryMap() {
         for ( iterator it = this->begin(); it != this->end(); it++ ) {
            delete it->second;
         }
      }

   private:
      void insertWithOverlap( const MemoryChunk &key, iterator &hint, MemChunkList &ptrList );
      void getWithOverlap( const MemoryChunk &key, const_iterator &hint, ConstMemChunkList &ptrList ) const;

   public:
      void getOrAddChunk( uint64_t addr, std::size_t len, MemChunkList &resultEntries );
      void getChunk2( uint64_t addr, std::size_t len, ConstMemChunkList &resultEntries ) const;
      void merge( const MemoryMap< _Type > &mm );
      void merge2( const MemoryMap< _Type > &mm );
      void print() const;
};

}
#endif /* MEMORYMAP_DECL_H */
