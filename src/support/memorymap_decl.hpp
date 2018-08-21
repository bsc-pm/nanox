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

#ifndef _NANOS_MEMORYMAP_DECL_H
#define _NANOS_MEMORYMAP_DECL_H

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
            /* [_____ this _____]
             *                     [_ target _]
             */
         BEGIN_OVERLAP,
            /*     [_____ this _____]
             * [_ target _]
             */
         END_OVERLAP,
            /* [_____ this _____]
             *             [_ target _]
             */
         TOTAL_OVERLAP,
            /*     [__ this __]
             *  [____ target ____]
             */
         SUBCHUNK_OVERLAP,
            /* [_____ this _____]
             *    [_ target _]
             */
         TOTAL_BEGIN_OVERLAP,
            /*  [__ this __]
             *  [____ target ____]
             */
         SUBCHUNK_BEGIN_OVERLAP,
            /* [_____ this _____]
             * [_ target _]
             */
         TOTAL_END_OVERLAP,
            /*        [__ this __]
             *  [____ target ____]
             */
         SUBCHUNK_END_OVERLAP
            /* [_____ this _____]
             *       [_ target _]
             */
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
   using std::map< MemoryChunk, _Type *>::operator=;
   //private:
   //   const MemoryMap & operator=( const MemoryMap &mm ) { this->std::map< MemoryChunk}
   public:
      //typedef enum { MEM_CHUNK_FOUND, MEM_CHUNK_NOT_FOUND, MEM_CHUNK_NOT_FOUND_BUT_ALLOCATED } QueryResult;
      typedef std::map< MemoryChunk, _Type * > BaseMap;
      typedef std::pair< const MemoryChunk *, _Type ** > MemChunkPair;
      typedef std::list< MemChunkPair > MemChunkList;
      typedef std::pair< MemoryChunk, _Type * > ConstMemChunkPair;
      typedef std::list< ConstMemChunkPair > ConstMemChunkList;
      typedef typename BaseMap::iterator iterator;
      typedef typename BaseMap::const_iterator const_iterator;

      MemoryMap() : std::map< MemoryChunk, _Type * >() { }
      MemoryMap( const MemoryMap &mm ) : std::map< MemoryChunk, _Type * >( mm ) { }
      ~MemoryMap() {
         for ( iterator it = this->begin(); it != this->end(); it++ ) {
            delete it->second;
         }
      }

   private:
      void insertWithOverlap( const MemoryChunk &key, iterator &hint, MemChunkList &ptrList );
      void insertWithOverlapButNotGenerateIntersects( const MemoryChunk &key, iterator &hint, MemChunkList &ptrList );
      void getWithOverlapNoExactKey( const MemoryChunk &key, const_iterator &hint, ConstMemChunkList &ptrList ) const;
   public:
      void getOrAddChunk( uint64_t addr, std::size_t len, MemChunkList &resultEntries );
      void getOrAddChunkDoNotFragment( uint64_t addr, std::size_t len, MemChunkList &resultEntries );
      void getChunk( uint64_t addr, std::size_t len, ConstMemChunkList &resultEntries ) const;
      void print(std::ostream &o) const;
      bool canPack() const;
      void removeChunks( uint64_t addr, std::size_t len );
      _Type **getExactInsertIfNotFound( uint64_t addr, std::size_t len );
      _Type *getExactByAddress( uint64_t addr ) const;
      void eraseByAddress( uint64_t addr );
      _Type **getExactOrFullyOverlappingInsertIfNotFound( uint64_t addr, std::size_t len, bool &exact );
};

#if 1
template <> 
class MemoryMap<uint64_t> : public std::map< MemoryChunk, uint64_t > {
   public:
      MemoryMap( const MemoryMap &mm ) : std::map< MemoryChunk, uint64_t> () { }
      const MemoryMap & operator=( const MemoryMap &mm );// { return *this; }
      typedef std::map< MemoryChunk, uint64_t > BaseMap;
      typedef BaseMap::iterator iterator;
      typedef BaseMap::const_iterator const_iterator;

      MemoryMap() { }
      ~MemoryMap() { }

   private:
      void insertWithOverlapButNotGenerateIntersects( const MemoryChunk &key, iterator &hint, uint64_t data );
   public:
      void addChunk( uint64_t addr, std::size_t len, uint64_t value );
      //void print() const;
      uint64_t getExactOrFullyOverlappingInsertIfNotFound( uint64_t addr, std::size_t len, bool &exact, uint64_t valIfNotFound, uint64_t valIfNotValid, uint64_t &conflictAddr, std::size_t &conflictSize );
      uint64_t getExactInsertIfNotFound( uint64_t addr, std::size_t len, uint64_t valIfNotFound, uint64_t valIfNotValid );
      uint64_t getExactByAddress( uint64_t addr, uint64_t valIfNotFound ) const;
      void eraseByAddress( uint64_t addr );
};
#endif

} // namespace nanos

#endif /* _NANOS_MEMORYMAP_DECL_H */
