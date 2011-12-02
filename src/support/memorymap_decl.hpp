#ifndef MEMORYMAP_DECL_H
#define MEMORYMAP_DECL_H

#include <map>
#include <stdint.h>
//#include "atomic_decl.hpp"

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

      MemoryChunk( uint64_t addr, std::size_t len ) : _addr( addr ), _len( len ) { }
      MemoryChunk( const MemoryChunk &mc ) : _addr( mc._addr ), _len( mc._len ) { }
      MemoryChunk( ) : _addr( 0 ), _len( 0 ) { }

      MemoryChunk& operator=( const MemoryChunk &mc );
      bool operator<( const MemoryChunk &chunk ) const;

      uint64_t getAddress() const;
      std::size_t getLength() const;
      OverlapType checkOverlap( const MemoryChunk &target ) const;
      bool equal( const MemoryChunk &target ) const;
      bool contains( const MemoryChunk &target ) const;
      void expandIncluding( const MemoryChunk &mcB );
      void expandExcluding( const MemoryChunk &mcB );
      void cutAfter( const MemoryChunk &mc );

      static void intersect( MemoryChunk &mcA, MemoryChunk &mcB, MemoryChunk &mcC );
      static void partition( MemoryChunk &mcA, MemoryChunk &mcB, MemoryChunk &mcC );
      static void partitionBeginAgtB( MemoryChunk &mcA, MemoryChunk &mcB );
      static void partitionBeginAltB( MemoryChunk &mcA, MemoryChunk &mcB );
      static void partitionEnd( MemoryChunk &mcA, MemoryChunk &mcB );
};

template <typename _Type>
class MemoryMap : public std::map< MemoryChunk, _Type * > { 
   private:
      typedef std::map< MemoryChunk, _Type * > BaseMap;

      //Lock _memMapLock;

      MemoryMap( const MemoryMap &mm ) { }
      const MemoryMap & operator=( const MemoryMap &mm ) { }
   
   public:
      typedef enum { MEM_CHUNK_FOUND, MEM_CHUNK_NOT_FOUND, MEM_CHUNK_NOT_FOUND_BUT_ALLOCATED } QueryResult;
      typedef std::pair< const MemoryChunk *, _Type ** > MemChunkPair;
      typedef std::list< MemChunkPair > MemChunkList;
      typedef typename BaseMap::iterator iterator;

      MemoryMap() { }
      ~MemoryMap() {
         for ( iterator it = this->begin(); it != this->end(); it++ ) {
            delete it->second;
         }
      }

   private:
      void insertWithOverlap( const MemoryChunk &key, typename BaseMap::iterator &hint, MemChunkList &ptrList );
      void processNoOverlap( MemoryChunk &key, typename BaseMap::iterator &hint, MemChunkList &ptrList );
      void processBeginOverlap( MemoryChunk &key, typename BaseMap::iterator &hint, MemChunkList &ptrList );
      void processEndOverlap( MemoryChunk &key, typename BaseMap::iterator &hint, MemChunkList &ptrList );
      void processTotalOverlap( MemoryChunk &key, typename BaseMap::iterator &hint, MemChunkList &ptrList );
      void processSubchunkOverlap( MemoryChunk &key, typename BaseMap::iterator &hint, MemChunkList &ptrList );
      void processTotalBeginOverlap( MemoryChunk &key, typename BaseMap::iterator &hint, MemChunkList &ptrList );
      void processSubchunkBeginOverlap( MemoryChunk &key, typename BaseMap::iterator &hint, MemChunkList &ptrList );
      void processTotalEndOverlap( MemoryChunk &key, typename BaseMap::iterator &hint, MemChunkList &ptrList );
      void processSubchunkEndOverlap( MemoryChunk &key, typename BaseMap::iterator &hint, MemChunkList &ptrList );

      void getWithOverlap( const MemoryChunk &key, typename BaseMap::iterator &hint, MemChunkList &ptrList );
      void processNoOverlapNI( MemoryChunk &key, typename BaseMap::iterator &hint, MemChunkList &ptrList );
      void processBeginOverlapNI( MemoryChunk &key, typename BaseMap::iterator &hint, MemChunkList &ptrList );
      void processEndOverlapNI( MemoryChunk &key, typename BaseMap::iterator &hint, MemChunkList &ptrList );
      void processTotalOverlapNI( MemoryChunk &key, typename BaseMap::iterator &hint, MemChunkList &ptrList );
      void processSubchunkOverlapNI( MemoryChunk &key, typename BaseMap::iterator &hint, MemChunkList &ptrList );
      void processTotalBeginOverlapNI( MemoryChunk &key, typename BaseMap::iterator &hint, MemChunkList &ptrList );
      void processSubchunkBeginOverlapNI( MemoryChunk &key, typename BaseMap::iterator &hint, MemChunkList &ptrList );
      void processTotalEndOverlapNI( MemoryChunk &key, typename BaseMap::iterator &hint, MemChunkList &ptrList );
      void processSubchunkEndOverlapNI( MemoryChunk &key, typename BaseMap::iterator &hint, MemChunkList &ptrList );

   public:
      void getOrAddChunk( uint64_t addr, std::size_t len, MemChunkList &resultEntries );
      void getChunk2( uint64_t addr, std::size_t len, MemChunkList &resultEntries );
      void merge( const MemoryMap< _Type > &mm );
      void merge2( const MemoryMap< _Type > &mm );
      void expand( MemoryChunk &inputKey, _Type *inputData, typename BaseMap::iterator &thisIt );
      void print() const;
};

}
#endif /* MEMORYMAP_DECL_H */
