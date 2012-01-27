#ifndef MEMORYMAP_H
#define MEMORYMAP_H

#include "memorymap_decl.hpp"
#include <iostream>
#include <stdio.h>

namespace nanos {

inline MemoryChunk& MemoryChunk::operator=( MemoryChunk const &mc ) {
   _addr = mc._addr;
   _len = mc._len;
   return *this;
}

inline bool MemoryChunk::operator<( MemoryChunk const &chunk ) const {
   return _addr < chunk._addr;
} 

inline uint64_t MemoryChunk::getAddress() const {
   return _addr;
}
inline std::size_t MemoryChunk::getLength() const {
   return _len;
}

inline bool MemoryChunk::equal( MemoryChunk const &target ) const {
   return ( _addr == target._addr && _len == target._len );
}

inline bool MemoryChunk::contains( MemoryChunk const &target ) const {
   return ( ( _addr <= target._addr) && ( ( _addr + _len ) >= ( target._addr + target._len ) ) );
}

inline MemoryChunk::OverlapType MemoryChunk::checkOverlap( MemoryChunk const &target ) const {
   OverlapType ret;
   if ( _addr < target._addr ) {
      if ( _addr + _len > target._addr ) {
         if ( _addr + _len > target._addr + target._len ) {
            ret = SUBCHUNK_OVERLAP; 
         } else if ( _addr + _len < target._addr + target._len ) {
            ret = END_OVERLAP;
         } else {
            ret = SUBCHUNK_END_OVERLAP; 
         }
      } else {
         ret = NO_OVERLAP;
      }
   } else if ( _addr > target._addr ) {
      if ( target._addr + target._len > _addr ) {
         if ( target._addr + target._len > _addr + _len ) {
            ret = TOTAL_OVERLAP; 
         } else if ( target._addr + target._len < _addr + _len ) {
            ret = BEGIN_OVERLAP;
         } else {
            ret = TOTAL_END_OVERLAP;
         }
      } else {
         ret = NO_OVERLAP;
      }
   } else { //addr == target._addr
      if ( target._len > _len ) {
         ret = TOTAL_BEGIN_OVERLAP;
      } else {
         ret = SUBCHUNK_BEGIN_OVERLAP;
      }
   }
      //fprintf( stderr, "detected %s\n", strOverlap[ret] );
   return ret;
}

inline void MemoryChunk::intersect( MemoryChunk &mcA, MemoryChunk &mcB, MemoryChunk &mcC ) {
   //assume A < B and NO TOTAL overlap
   std::size_t intersectionLen = mcA._addr + mcA._len - mcB._addr;
   mcC._addr = mcA._addr + mcA._len;
   mcC._len = mcB._len - intersectionLen;
   mcB._len = intersectionLen;
   mcA._len -= intersectionLen;
}

inline void MemoryChunk::partition( MemoryChunk &mcA, MemoryChunk const &mcB, MemoryChunk &mcC ) {
   //assume A < B and A totally overlaps B
   std::size_t remainingLen = ( mcA._addr + mcA._len ) - ( mcB._addr + mcB._len );
   mcC._addr = mcB._addr + mcB._len;
   mcC._len = remainingLen;
   mcA._len -= ( remainingLen + mcB._len );
}

inline void MemoryChunk::partitionBeginAgtB( MemoryChunk &mcA, MemoryChunk &mcB ) {
   //assume A.addr = B.addr, A.len > B.len,  A.addr is NOT modified 
   std::size_t bLen = mcA._len - mcB._len;
   mcB._addr = mcA._addr + mcB._len;
   mcB._len = bLen;
   mcA._len -= mcB._len;
}

inline void MemoryChunk::partitionBeginAltB( MemoryChunk const &mcA, MemoryChunk &mcB ) {
   //assume A.addr = B.addr, A.len < B.len,  A.addr is NOT modified 
   std::size_t bLen = mcB._len - mcA._len;
   mcB._addr = mcA._addr + mcA._len;
   mcB._len = bLen;
}

inline void MemoryChunk::partitionEnd( MemoryChunk &mcA, MemoryChunk const &mcB ) {
   //assume A.addr+A.len = B.addr+B.len, A.len > B.len,  B.addr is NOT modified 
   mcA._len -= mcB._len;
}

inline void MemoryChunk::expandIncluding( MemoryChunk const &mcB ) {
   _len = ( mcB._addr + mcB._len ) - _addr;
}

inline void MemoryChunk::expandExcluding( MemoryChunk const &mcB ) {
   _len = ( mcB._addr - _addr );
}

inline void MemoryChunk::cutAfter( MemoryChunk const &mc ) {
   _len = ( _addr + _len ) - ( mc._addr + mc._len );
   _addr = mc._addr + mc._len;
}


template < typename _Type >
void MemoryMap< _Type >::insertWithOverlap( const MemoryChunk &key, typename BaseMap::iterator &hint, MemChunkList &ptrList )
{
   class LocalFunctions {
      MemoryMap< _Type > &_thisMap;
      MemoryChunk        &_key;
      iterator           &_hint;
      MemChunkList       &_ptrList;

      public:
      LocalFunctions( MemoryMap< _Type > &thisMap, MemoryChunk &key, iterator &hint, MemChunkList &ptrList ) :
         _thisMap( thisMap ), _key( key ), _hint( hint ), _ptrList( ptrList ) { }

      void insertNoOverlap()
      {
         _Type *ptr = (_Type *) NULL;
         _hint = _thisMap.insert( _hint, typename BaseMap::value_type( _key, ptr ) );
         _ptrList.push_back( MemChunkPair ( &( _hint->first ), &( _hint->second ) ) );
      }

      void insertBeginOverlap()
      {
         /*
          *                +=====================+
          *                |     rightChunk      |
          *                | (already inserted)  |
          *                +=====================+
          *   +-----------------------+
          *   |      leftChunk        |
          *   |    (to be inserted)   |
          *   +-----------------------+
          *           == intersect ==
          *   +------------+
          *   | leftChunk  |
          *   +------------+
          *        ^       +==========+............+
          *        |       |rightChunk|righLeftOver|
          *        |       +==========+............+
          *   added to result----^
          */
         _Type *ptr = (_Type *) NULL;
         MemoryChunk &leftChunk = _key;
         MemoryChunk &rightChunk = const_cast< MemoryChunk & >( _hint->first );
         MemoryChunk rightLeftOver;

         MemoryChunk::intersect( leftChunk, rightChunk, rightLeftOver );

         _hint = _thisMap.insert( _hint, typename BaseMap::value_type( leftChunk, ptr ) ); //add the left chunk, 
         _ptrList.push_back( MemChunkPair ( &( _hint->first ), &( _hint->second ) ) );
         _hint++; //advance the iterator (it should reach "rightChunk"!)
         _ptrList.push_back( MemChunkPair ( &( _hint->first ), &( _hint->second ) ) );

         _hint = _thisMap.insert( _hint, typename BaseMap::value_type( rightLeftOver, NEW _Type( *_hint->second ) ) );
      }

      void insertEndOverlap()
      {
         MemoryChunk &leftChunk = const_cast< MemoryChunk & >( _hint->first );
         MemoryChunk &rightChunk = _key;
         MemoryChunk rightLeftOver;
         /*
          *   +=======================+
          *   |      leftChunk        |
          *   |   (already inserted)  |
          *   +=======================+
          *                +---------------------+
          *                |     rightChunk      |
          *                |  (to be inserted)   |
          *                +---------------------+
          *           == intersect ==
          *   +============+
          *   | leftChunk  |
          *   +============+
          *                +----------+............+
          *                |rightChunk|righLeftOver| <- not to be inserted yet.
          *                +----------+............+
          *   added to result-----^        ^------ not added to result (processed on next iterations)
          */

         MemoryChunk::intersect( leftChunk, rightChunk, rightLeftOver );

         //add the right chunk 
         _hint = _thisMap.insert( _hint, typename BaseMap::value_type( rightChunk, NEW _Type( *_hint->second ) ) );
         //result: right chunk
         _ptrList.push_back( MemChunkPair ( &( _hint->first ), &( _hint->second) ) );
         //leftover not added to result since more overlapping may exist
         _key = rightLeftOver;  //try to add what is left
      }

      void insertTotalOverlap()
      {
         _Type *ptr = (_Type *) NULL;
         MemoryChunk &leftChunk = _key;
         MemoryChunk &rightChunk = const_cast< MemoryChunk & >( _hint->first );
         _Type ** rightChunkDataPtr = &( _hint->second );
         MemoryChunk leftLeftOver;
         /*
          *                +=====================+
          *                |     rightChunk      |
          *                |  (already inserted) |
          *                +=====================+
          *   +-----------------------------------------------+
          *   |                  leftChunk                    |
          *   |               (to be inserted)                |
          *   +-----------------------------------------------+
          *                    == partition ==
          *                +=====================+
          *                |     rightChunk      |
          *                +=====================+
          *   +------------+        ^            +............+
          *   | leftChunk  |        |            |leftLeftOver|
          *   +------------+        |            +............+
          *         ^------  added to result          ^-- not added (keep processing)
          */

         MemoryChunk::partition( leftChunk, rightChunk, leftLeftOver );

         _hint = _thisMap.insert( _hint, typename BaseMap::value_type( leftChunk, ptr ) ); //add the left chunk, 
         _ptrList.push_back( MemChunkPair ( &( _hint->first ), &( _hint->second ) ) ); //return: leftChunk
         _ptrList.push_back( MemChunkPair ( &rightChunk, rightChunkDataPtr ) ); //return: rightChunk
         _key = leftLeftOver; //try to add what is left
      }

      void insertSubchunkOverlap()
      {
         MemoryChunk &leftChunk = const_cast< MemoryChunk & >( _hint->first );
         MemoryChunk &rightChunk = _key;
         MemoryChunk leftLeftOver;
         /*
          *   +===============================================+
          *   |                  leftChunk                    |
          *   |               (already inserted)              |
          *   +===============================================+
          *                +---------------------+
          *                |     rightChunk      |
          *                |  (to be inserted)   |
          *                +---------------------+
          *                    == partition ==
          *   +============+                     +............+
          *   | leftChunk  |                     |leftLeftOver|
          *   +============+                     +............+
          *                +---------------------+
          *                |     rightChunk      |
          *                +---------------------+
          *                   added to result
          */

         MemoryChunk::partition( leftChunk, rightChunk, leftLeftOver );

         _hint = _thisMap.insert( _hint, typename BaseMap::value_type( rightChunk, NEW _Type( *_hint->second ) ) ); //add the right chunk, 
         _ptrList.push_back( MemChunkPair ( &( _hint->first ), &( _hint->second ) ) ); //return: rightChunk
         _hint = _thisMap.insert( _hint, typename BaseMap::value_type( leftLeftOver, NEW _Type( *_hint->second ) ) ); //add the leftover
      }

      void insertTotalBeginOverlap()
      {
         MemoryChunk &leftChunk = const_cast< MemoryChunk & >( _hint->first );
         MemoryChunk &rightChunk = _key;
         /*
          *   +=====================+
          *   |      leftChunk      |
          *   |  (already inserted) |
          *   +=====================+
          *   +-----------------------------------------------+
          *   |                 rightChunk                    |
          *   |               (to be inserted)                |
          *   +-----------------------------------------------+
          *                    == partitionBeginAltB ==
          *   +=====================+
          *   |      leftChunk      |
          *   +=====================+
          *             ^           +.........................+
          *             |           |       rightChunk        |
          *             |           +.........................+
          *       added to result         ^-- not added (keep processing)
          */
         MemoryChunk::partitionBeginAltB( leftChunk, rightChunk );
         _ptrList.push_back( MemChunkPair ( &( _hint->first ), &( _hint->second ) ) ); //return: leftChunk
         _key = rightChunk; // redundant
      }

      void insertSubchunkBeginOverlap()
      {
         MemoryChunk &leftChunk = const_cast< MemoryChunk & >( _hint->first );
         MemoryChunk &rightChunk = _key;
         /*
          *   +===============================================+
          *   |                  leftChunk                    |
          *   |               (already inserted)              |
          *   +===============================================+
          *   +---------------------+
          *   |     rightChunk      |
          *   |  (to be inserted)   |
          *   +---------------------+
          *                == partitionBeginAgtB ==
          *                         +-------------------------+
          *                         |        rightChunk       |
          *                         +-------------------------+
          *   +=====================+
          *   |     leftchunk       |
          *   +=====================+
          *       added to result
          */
         MemoryChunk::partitionBeginAgtB( leftChunk, rightChunk );
         _ptrList.push_back( MemChunkPair ( &( _hint->first ), &( _hint->second ) ) ); //return: leftChunk
         _hint = _thisMap.insert( _hint, typename BaseMap::value_type( rightChunk, NEW _Type( *_hint->second ) ) ); //add the rightChunk
      }

      void insertTotalEndOverlap()
      {
         _Type *ptr = (_Type *) NULL;
         MemoryChunk &leftChunk = _key;
         MemoryChunk &rightChunk = const_cast< MemoryChunk & >( _hint->first );
         const MemoryChunk *rightChunkPtr = &( _hint->first );
         _Type ** rightChunkDataPtr = &( _hint->second );
         /*
          *                             +=====================+
          *                             |     rightChunk      |
          *                             |  (already inserted) |
          *                             +=====================+
          *   +-----------------------------------------------+
          *   |                  leftChunk                    |
          *   |               (to be inserted)                |
          *   +-----------------------------------------------+
          *                    == partition ==
          *                             +=====================+
          *                             |     rightChunk      |
          *                             +=====================+
          *   +-------------------------+        ^
          *   |        leftChunk        |        |
          *   +-------------------------+        |
          *         ^------  added to result ----+
          */
         MemoryChunk::partitionEnd( leftChunk, rightChunk );
         _hint = _thisMap.insert( _hint, typename BaseMap::value_type( rightChunk, ptr ) ); //add the leftChunk
         _ptrList.push_back( MemChunkPair ( &( _hint->first ), &( _hint->second ) ) ); //return: leftChunk
         _ptrList.push_back( MemChunkPair ( rightChunkPtr, rightChunkDataPtr ) ); //return: rightChunk
      }

      void insertSubchunkEndOverlap()
      {
         MemoryChunk &leftChunk = const_cast< MemoryChunk & >( _hint->first );
         MemoryChunk &rightChunk = _key;
         /*
          *   +===============================================+
          *   |                  leftChunk                    |
          *   |               (already inserted)              |
          *   +===============================================+
          *                             +---------------------+
          *                             |     rightChunk      |
          *                             |  (to be inserted)   |
          *                             +---------------------+
          *                    == partition ==
          *   +=========================+
          *   |        leftChunk        |
          *   +=========================+
          *                             +---------------------+
          *                             |     rightChunk      |
          *                             +---------------------+
          *                                added to result
          */
         MemoryChunk::partitionEnd( leftChunk, rightChunk );
         _hint = _thisMap.insert( _hint, typename BaseMap::value_type( rightChunk, NEW _Type( *_hint->second ) ) ); //add the rightChunk
         _ptrList.push_back( MemChunkPair ( &( _hint->first ), &( _hint->second ) ) ); //return: rightChunk
      }
   };
   // precondition: there is no exact match
   bool lastChunk = false;
   MemoryChunk iterKey = key;
   LocalFunctions local( *this, iterKey, hint, ptrList );
   if ( !this->empty() )
   {
      if ( hint != this->begin() )
         hint--;
      do {
         switch ( hint->first.checkOverlap( iterKey ) )
         {
            case MemoryChunk::NO_OVERLAP:
               if ( iterKey.getAddress() < hint->first.getAddress() )
               {
                  local.insertNoOverlap();
                  lastChunk = true;
               }
               break;
            case MemoryChunk::BEGIN_OVERLAP:
               local.insertBeginOverlap();
               lastChunk = true; //this is a final situation since the to-be-inserted chunk ends.
               break;
            case MemoryChunk::END_OVERLAP:
               local.insertEndOverlap();
               break;
            case MemoryChunk::TOTAL_OVERLAP:
               local.insertTotalOverlap();
               break;
            case MemoryChunk::SUBCHUNK_OVERLAP:
               local.insertSubchunkOverlap();
               lastChunk = true;
               break;
            case MemoryChunk::TOTAL_BEGIN_OVERLAP:
               local.insertTotalBeginOverlap();
               break;
            case MemoryChunk::SUBCHUNK_BEGIN_OVERLAP:
               local.insertSubchunkBeginOverlap();
               lastChunk = true;
               break;
            case MemoryChunk::TOTAL_END_OVERLAP:
               local.insertTotalEndOverlap();
               lastChunk = true;
               break;
            case MemoryChunk::SUBCHUNK_END_OVERLAP:
               local.insertSubchunkEndOverlap();
               lastChunk = true;
               break;
         }
         hint++;
      } while ( hint != this->end() && !lastChunk );
      if ( hint == this->end() && !lastChunk )
      {
         // no more chunks in the map but there is still "key" to insert
         hint = this->insert( hint, typename BaseMap::value_type( iterKey, ( _Type * ) NULL ) );
         ptrList.push_back( MemChunkPair ( &(hint->first), &(hint->second) ) );
      }
   } else {
      hint = this->insert( hint, typename BaseMap::value_type( key, ( _Type * ) NULL ) );
      ptrList.push_back( MemChunkPair ( &(hint->first), &(hint->second) ) );
   }
}

template < typename _Type >
void MemoryMap< _Type >::getWithOverlap( const MemoryChunk &key, const_iterator &hint, ConstMemChunkList &ptrList ) const
{
   class LocalFunctions {
      MemoryChunk          &_key;
      const_iterator       &_hint;
      ConstMemChunkList    &_ptrList;

      public:
      LocalFunctions( MemoryChunk &key, const_iterator &hint, ConstMemChunkList &ptrList ) :
         _key( key ), _hint( hint ), _ptrList( ptrList ) { }

      void getNoOverlap()
      {
         _ptrList.push_back( ConstMemChunkPair ( NEW MemoryChunk( _key ), NULL ) );
      }

      void getBeginOverlap( )
      {
         /*
          *                +=====================+
          *                |     rightChunk      |
          *                | (already inserted)  |
          *                +=====================+
          *   +-----------------------+
          *   |      leftChunk        |
          *   |    (to be inserted)   |
          *   +-----------------------+
          *           == intersect ==
          *   +------------+
          *   | leftChunk  |
          *   +------------+
          *        ^       +==========+............+
          *        |       |rightChunk|righLeftOver|
          *        |       +==========+............+
          *   added to result----^
          */
         MemoryChunk leftChunk = _key;
         MemoryChunk rightChunk = _hint->first ;
         MemoryChunk rightLeftOver;

         MemoryChunk::intersect( leftChunk, rightChunk, rightLeftOver );

         _ptrList.push_back( ConstMemChunkPair ( NEW MemoryChunk( leftChunk ) , NULL ) );
         _ptrList.push_back( ConstMemChunkPair ( NEW MemoryChunk( rightChunk ) , &(_hint->second) ) );
      }

      void getEndOverlap( )
      {
         MemoryChunk leftChunk = _hint->first ;
         MemoryChunk rightChunk = _key;
         MemoryChunk rightLeftOver;
         /*
          *   +=======================+
          *   |      leftChunk        |
          *   |   (already inserted)  |
          *   +=======================+
          *                +---------------------+
          *                |     rightChunk      |
          *                |  (to be inserted)   |
          *                +---------------------+
          *           == intersect ==
          *   +============+
          *   | leftChunk  |
          *   +============+
          *                +----------+............+
          *                |rightChunk|righLeftOver| <- not to be inserted yet.
          *                +----------+............+
          *   added to result-----^        ^------ not added to result (processed on next iterations)
          */

         MemoryChunk::intersect( leftChunk, rightChunk, rightLeftOver );

         //add the right chunk 
         //hint = this->insert( hint, typename BaseMap::value_type( rightChunk, NEW _Type( *hint->second ) ) );
         //result: right chunk
         _ptrList.push_back( ConstMemChunkPair ( NEW MemoryChunk( rightChunk ), &(_hint->second) ) );
         //leftover not added to result since more overlapping may exist
         _key = rightLeftOver;  //try to add what is left
      }

      void getTotalOverlap( )
      {
         //_Type *ptr = (_Type *) NULL;
         MemoryChunk leftChunk = _key;
         MemoryChunk rightChunk = _hint->first ;
         MemoryChunk leftLeftOver;
         /*
          *                +=====================+
          *                |     rightChunk      |
          *                |  (already inserted) |
          *                +=====================+
          *   +-----------------------------------------------+
          *   |                  leftChunk                    |
          *   |               (to be inserted)                |
          *   +-----------------------------------------------+
          *                    == partition ==
          *                +=====================+
          *                |     rightChunk      |
          *                +=====================+
          *   +------------+        ^            +............+
          *   | leftChunk  |        |            |leftLeftOver|
          *   +------------+        |            +............+
          *         ^------  added to result          ^-- not added (keep processing)
          */

         MemoryChunk::partition( leftChunk, rightChunk, leftLeftOver );

         _ptrList.push_back( ConstMemChunkPair ( NEW MemoryChunk ( leftChunk ), NULL ) ); //return: leftChunk
         _ptrList.push_back( ConstMemChunkPair ( NEW MemoryChunk ( rightChunk ), &(_hint->second) ) ); //return: rightChunk
         _key = leftLeftOver; //try to add what is left
      }

      void getSubchunkOverlap()
      {
         MemoryChunk leftChunk = _hint->first ;
         MemoryChunk rightChunk = _key;
         MemoryChunk leftLeftOver;
         /*
          *   +===============================================+
          *   |                  leftChunk                    |
          *   |               (already inserted)              |
          *   +===============================================+
          *                +---------------------+
          *                |     rightChunk      |
          *                |  (to be inserted)   |
          *                +---------------------+
          *                    == partition ==
          *   +============+                     +............+
          *   | leftChunk  |                     |leftLeftOver|
          *   +============+                     +............+
          *                +---------------------+
          *                |     rightChunk      |
          *                +---------------------+
          *                   added to result
          */

         MemoryChunk::partition( leftChunk, rightChunk, leftLeftOver );

         _ptrList.push_back( ConstMemChunkPair ( NEW MemoryChunk( rightChunk ), &(_hint->second) ) ); //return: rightChunk
      }

      void getTotalBeginOverlap()
      {
         MemoryChunk leftChunk = _hint->first ;
         MemoryChunk rightChunk = _key;
         /*
          *   +=====================+
          *   |      leftChunk      |
          *   |  (already inserted) |
          *   +=====================+
          *   +-----------------------------------------------+
          *   |                 rightChunk                    |
          *   |               (to be inserted)                |
          *   +-----------------------------------------------+
          *                    == partitionBeginAltB ==
          *   +=====================+
          *   |      leftChunk      |
          *   +=====================+
          *             ^           ...........................
          *             |           :       rightChunk        :
          *             |           :.........................:
          *       added to result         ^-- not added (keep processing)
          */
         MemoryChunk::partitionBeginAltB( leftChunk, rightChunk );
         _ptrList.push_back( ConstMemChunkPair ( NEW MemoryChunk ( rightChunk ), &(_hint->second) ) ); //return: leftChunk
         _key = rightChunk;
      }

      void getSubchunkBeginOverlap()
      {
         MemoryChunk leftChunk = _hint->first ;
         MemoryChunk rightChunk = _key;
         /*
          *   +===============================================+
          *   |                  leftChunk                    |
          *   |               (already inserted)              |
          *   +===============================================+
          *   +---------------------+
          *   |     rightChunk      |
          *   |  (to be inserted)   |
          *   +---------------------+
          *                == partitionBeginAgtB ==
          *                         +-------------------------+
          *                         |        rightChunk       |
          *                         +-------------------------+
          *   +=====================+
          *   |     leftchunk       |
          *   +=====================+
          *       added to result
          */
         MemoryChunk::partitionBeginAgtB( leftChunk, rightChunk );
         _ptrList.push_back( ConstMemChunkPair ( NEW MemoryChunk( leftChunk ), &(_hint->second) ) ); //return: leftChunk
      }

      void getTotalEndOverlap()
      {
         MemoryChunk leftChunk = _key;
         MemoryChunk rightChunk = _hint->first ;
         //const MemoryChunk *rightChunkPtr = &( hint->first );
         _Type * const * rightChunkDataPtr = &( _hint->second );
         /*
          *                             +=====================+
          *                             |     rightChunk      |
          *                             |  (already inserted) |
          *                             +=====================+
          *   +-----------------------------------------------+
          *   |                  leftChunk                    |
          *   |               (to be inserted)                |
          *   +-----------------------------------------------+
          *                    == partition ==
          *                             +=====================+
          *                             |     rightChunk      |
          *                             +=====================+
          *   +-------------------------+        ^
          *   |        leftChunk        |        |
          *   +-------------------------+        |
          *         ^------  added to result ----+
          */
         MemoryChunk::partitionEnd( leftChunk, rightChunk );
         _ptrList.push_back( ConstMemChunkPair ( NEW MemoryChunk ( leftChunk ), NULL ) ); //return: leftChunk
         _ptrList.push_back( ConstMemChunkPair ( NEW MemoryChunk ( rightChunk ), rightChunkDataPtr ) ); //return: rightChunk
      }

      void getSubchunkEndOverlap()
      {
         MemoryChunk leftChunk = _hint->first ;
         MemoryChunk rightChunk = _key;
         /*
          *   +===============================================+
          *   |                  leftChunk                    |
          *   |               (already inserted)              |
          *   +===============================================+
          *                             +---------------------+
          *                             |     rightChunk      |
          *                             |  (to be inserted)   |
          *                             +---------------------+
          *                    == partition ==
          *   +=========================+
          *   |        leftChunk        |
          *   +=========================+
          *                             +---------------------+
          *                             |     rightChunk      |
          *                             +---------------------+
          *                                added to result
          */
         MemoryChunk::partitionEnd( leftChunk, rightChunk );
         _ptrList.push_back( ConstMemChunkPair ( NEW MemoryChunk ( rightChunk ), &(_hint->second) ) ); //return: rightChunk
      }
   };

   // precondition: there is no exact match
   bool lastChunk = false;
   MemoryChunk iterKey = key;
   LocalFunctions local( iterKey, hint, ptrList );
   if ( !this->empty() )
   {
      if ( hint != this->begin() )
         hint--;
      do {
         //fprintf(stderr, "iterKey %d %d, hint %d %d : %s\n", iterKey.getAddress(), iterKey.getLength(), hint->first.getAddress(), hint->first.getLength(), MemoryChunk::strOverlap[ hint->first.checkOverlap( iterKey ) ]);
         switch ( hint->first.checkOverlap( iterKey ) )
         {
            case MemoryChunk::NO_OVERLAP:
               if ( iterKey.getAddress() < hint->first.getAddress() )
               {
                  local.getNoOverlap();
                  lastChunk = true;
               }
               break;
            case MemoryChunk::BEGIN_OVERLAP:
               local.getBeginOverlap();
               lastChunk = true; //this is a final situation since the to-be-inserted chunk ends.
               break;
            case MemoryChunk::END_OVERLAP:
               local.getEndOverlap();
               break;
            case MemoryChunk::TOTAL_OVERLAP:
               local.getTotalOverlap();
               break;
            case MemoryChunk::SUBCHUNK_OVERLAP:
               local.getSubchunkOverlap();
               lastChunk = true;
               break;
            case MemoryChunk::TOTAL_BEGIN_OVERLAP:
               local.getTotalBeginOverlap();
               break;
            case MemoryChunk::SUBCHUNK_BEGIN_OVERLAP:
               local.getSubchunkBeginOverlap();
               lastChunk = true;
               break;
            case MemoryChunk::TOTAL_END_OVERLAP:
               local.getTotalEndOverlap();
               lastChunk = true;
               break;
            case MemoryChunk::SUBCHUNK_END_OVERLAP:
               local.getSubchunkEndOverlap();
               lastChunk = true;
               break;
         }
         hint++;
      } while ( hint != this->end() && !lastChunk );
   } else {
      ptrList.push_back( ConstMemChunkPair ( NEW MemoryChunk(key), NULL ) );
   }
}

template < typename _Type >
void MemoryMap< _Type >::getOrAddChunk( uint64_t addr, std::size_t len, MemChunkList &resultEntries )
{
   MemoryChunk key( addr, len );

   typename BaseMap::iterator it = this->lower_bound( key );
   if ( it == this->end() || this->key_comp()( key, it->first ) || it->first.getLength() != len )
   {
      /* NOT EXACT ADDR FOUND: "addr" is higher than any odther addr in the map OR less than "it" */
      insertWithOverlap( key, it, resultEntries );
      //res = MEM_CHUNK_NOT_FOUND_BUT_ALLOCATED;
   }
   else
   {
      /* EXACT ADDR FOUND */
      resultEntries.push_back( MemChunkPair( &(it->first), &(it->second) ) );
   }
}

template < typename _Type >
void MemoryMap< _Type >::getChunk2( uint64_t addr, std::size_t len, ConstMemChunkList &resultEntries ) const
{
   MemoryChunk key( addr, len );

   //fprintf(stderr, "%s key requested addr %d, len %d\n", __FUNCTION__, key.getAddress(), key.getLength() );
   const_iterator it = this->lower_bound( key );
   if ( it == this->end() || this->key_comp()( key, it->first ) || it->first.getLength() != len )
   {
      /* NOT EXACT ADDR FOUND: "addr" is higher than any odther addr in the map OR less than "it" */
      getWithOverlap( key, it, resultEntries );
      //for (typename MemChunkList::iterator it = resultEntries.begin(); it != resultEntries.end(); it++)
      //{
      //   //fprintf(stderr, "result entry addr %d, len %d\n", it->first->getAddress(), it->first->getLength() );
      //   //if ( it->second != NULL ) 
      //   //   if ( *(it->second) != NULL ) 
      //   //      (*it->second)->print();
      //   //   else fprintf(stderr, "entry is null!! \n");
      //   //else fprintf(stderr, "entry ptr is null!! \n");
      //}
      if ( resultEntries.size() == 0 )
         fprintf(stderr, "result entry EMPTY!\n" );

   }
   else
   {
      /* EXACT ADDR FOUND */
      resultEntries.push_back( ConstMemChunkPair( NEW MemoryChunk( key ), &( it->second ) ) );
      //fprintf(stderr, "result entry addr %d, len %d\n", key.getAddress(), key.getLength() );
   }
}

template < typename _Type >
void MemoryMap< _Type >::print() const
{
   int i = 0;
   typename BaseMap::const_iterator it = this->begin();
   std::cerr << "printing memory chunks" << std::endl;
   for (; it != this->end(); it++)
   {
      std::cerr << "\tchunk: " << i++ << " addr=" << (void *) it->first.getAddress() <<"(" << (uint64_t)it->first.getAddress()<< ")" << " len=" << it->first.getLength() << " ptr val is " << it->second << " addr of ptr val is " << (void *) &(it->second) << " ";
      it->second->print();
   } 
   std::cerr << "end of memory chunks" << std::endl;
}

/* merge input MemoryMap mm into this */
template < typename _Type >
void MemoryMap< _Type >::merge( const MemoryMap< _Type > &mm )
{
   typename BaseMap::const_iterator inputIt = mm.begin();

   while ( inputIt != mm.end() )
   {
      MemChunkList result;
      typename MemChunkList::iterator resultIt;
      getOrAddChunk( inputIt->first.getAddress(), inputIt->first.getLength(), result);
      for ( resultIt = result.begin(); resultIt != result.end(); resultIt++ )
      {
         _Type **ent = resultIt->second;
         if ( *ent == NULL ) { *ent = NEW _Type ( *inputIt->second ); /*if ( sys.getNetwork()->getNodeNum()==0){fprintf(stderr, "[%p] New entry from merge tag %p\n", this, (void *)inputIt->first.getAddress() );}*/ }
         else { /*if ( sys.getNetwork()->getNodeNum()==0){ fprintf(stderr, "[%p] Already got that entry %p\n", this, (void *) inputIt->first.getAddress()); (*ent)->print(); inputIt->second->print();}*/ (*ent)->merge( *(inputIt->second) );}
      }
      inputIt++;
   }
}

/* merge input MemoryMap mm into this */
template < typename _Type >
void MemoryMap< _Type >::merge2( const MemoryMap< _Type > &mm )
{
   class LocalFunctions {
      MemoryMap<_Type> &_thisMap;

      public:
      LocalFunctions( MemoryMap<_Type> &thisMap ) : _thisMap( thisMap ) { }

      void tryToMergeWithPreviousEntry( iterator &thisIt ) {
         if ( thisIt != _thisMap.begin() ) {
            iterator prevIt = thisIt;
            prevIt--;
            if ( prevIt->first.getAddress() + prevIt->first.getLength() == thisIt->first.getAddress() ) {
               if ( prevIt->second->equal( *thisIt->second ) ) { 
                  MemoryChunk &prevNoConst = const_cast< MemoryChunk & >( prevIt->first );
                  prevNoConst.expandIncluding( thisIt->first );
                  _thisMap.erase( thisIt );
                  thisIt = _thisMap.find( prevNoConst );
               }
            }
         }
      }

      void expand( MemoryChunk &inputKey, _Type * const &inputData, iterator &thisIt )
      {
         class ExpandLocalFunctions {

            MemoryMap< _Type > &_thisMap;
            MemoryChunk        &_inputKey;
            _Type *      const &_inputData;
            iterator           &_thisIt;

            bool _thisAndInputDataAreEqual;
            bool _nextAndInputDataAreEqual;

            public:
            ExpandLocalFunctions( MemoryMap<_Type> &thisMap, MemoryChunk &inputKey, _Type * const &inputData, iterator &thisIt ) :
               _thisMap( thisMap ), _inputKey( inputKey ), _inputData( inputData ), _thisIt( thisIt ) {
                  _thisAndInputDataAreEqual = _thisIt->second->equal( *_inputData );
               }

            void expandNoOverlap() {
               if ( _thisAndInputDataAreEqual ) {
                  /* we can expand safetly, and we finish */
                  MemoryChunk &thisNoConst = const_cast< MemoryChunk & >( _thisIt->first );
                  thisNoConst.expandIncluding( _inputKey );
               } else {
                  _thisIt = _thisMap.insert( _thisIt, typename BaseMap::value_type( _inputKey, NEW _Type ( *_inputData ) ) );
               }
               _thisIt++;
            }

            void expandBeginOverlap( iterator &nextIt ) {
               const MemoryChunk &nextKey = nextIt->first;
               const MemoryChunk &thisKey = _thisIt->first;
               _nextAndInputDataAreEqual = nextIt->second->equal( *_inputData );
               if ( _thisAndInputDataAreEqual ) {
                  /* if the chunk Data is equivalent, we can expan all chunks and we finish */
                  if ( _nextAndInputDataAreEqual )
                  {
                     /* expand to fit [this ... inputKey ... next]
                      * then erase next entry
                      */
                     MemoryChunk &thisNoConst = const_cast< MemoryChunk & >( _thisIt->first );
                     thisNoConst.expandIncluding( nextKey );
                     delete nextIt->second;
                     _thisMap.erase( nextIt );
                     _thisIt = _thisMap.find( thisKey );
                  } else {
                     /* fit [this ... inputKey ... ]{next:overlap_with_input}(next:leftover)
                      */
                     MemoryChunk rightLeftOver;
                     MemoryChunk inputKeyCopy = _inputKey;
                     MemoryChunk &rightChunk = const_cast< MemoryChunk & >( nextIt->first );
                     /* we can not merge both already-inserted entries,
                      * expand "this" as much as possible and partition
                      * the leftover of "inputKey" that intersects with hintCopy
                      */
                     MemoryChunk &thisNoConst = const_cast< MemoryChunk & >( _thisIt->first );
                     thisNoConst.expandExcluding( nextIt->first );
                     MemoryChunk::intersect( inputKeyCopy, rightChunk, rightLeftOver ); //this modifies nextIt->first!!
                     _thisIt = _thisMap.insert( nextIt, typename BaseMap::value_type( rightLeftOver, NEW _Type( *nextIt->second ) ) );
                     _thisIt--;
                     _thisIt->second->merge( *_inputData );
                     _thisIt++;
                  }
               } else {
                  if ( _nextAndInputDataAreEqual )
                  {
                     /* expand to fit: (this)[... inputKey ... next]
                      */
                     std::pair<typename BaseMap::iterator, bool> insertResult;
                     _inputKey.expandIncluding( nextKey );
                     _Type *data = nextIt->second;
                     _thisMap.erase( nextIt );
                     insertResult = _thisMap.insert( typename BaseMap::value_type( _inputKey, data ) ); //reuse data entry
                     _thisIt = insertResult.first;
                  } else {
                     /* fit (this)[ ... inputKey ... ]{next:overlap_with_input}(next:leftover)
                      */
                     MemoryChunk rightLeftOver;
                     MemoryChunk inputKeyCopy = _inputKey;
                     MemoryChunk &rightChunk = const_cast< MemoryChunk & >( nextIt->first );
                     /* we can not merge both already-inserted entries,
                      * expand "this" as much as possible and partition
                      * the leftover of "inputKey" that intersects with hintCopy
                      */
                     MemoryChunk::intersect( inputKeyCopy, rightChunk, rightLeftOver ); //this modifies nextIt->first!!
                     _thisIt = _thisMap.insert( _thisIt, typename BaseMap::value_type( inputKeyCopy, NEW _Type( *_inputData ) ) );
                     _thisIt++;
                     _thisIt = _thisMap.insert( _thisIt, typename BaseMap::value_type( rightLeftOver, NEW _Type( *_thisIt->second ) ) );
                     _thisIt--;
                     _thisIt->second->merge( *_inputData ); //merge overlapped chunk
                     _thisIt++;
                  }
               }
            }

            void expandTotalOverlap( iterator &nextIt ) {
               const MemoryChunk &nextKey = nextIt->first;
               _nextAndInputDataAreEqual = nextIt->second->equal( *_inputData );
               if ( _thisAndInputDataAreEqual ) {
                  MemoryChunk &thisNoConst = const_cast< MemoryChunk & >( _thisIt->first );
                  if ( _nextAndInputDataAreEqual )
                  {
                     /* [this .. inputKey .. next]{..inputKey..} */
                     thisNoConst.expandIncluding( nextKey );
                     MemoryChunk key = _thisIt->first;
                     delete nextIt->second;
                     _thisMap.erase( nextIt );
                     _thisIt = nextIt = _thisMap.find( key );
                     //nextIt++;
                     _inputKey.cutAfter( _thisIt->first );
                  } else {
                     /* [this .. inputKey .. ][next]{..inputKey..} */
                     thisNoConst.expandExcluding( nextKey );
                     nextIt->second->merge( *_inputData );
                     _inputKey.cutAfter( nextKey );
                     _thisIt++; //increase thisIt to keep processing
                  }
               } else {
                  if ( _nextAndInputDataAreEqual )
                  {
                     /* (this)[.. inputKey .. next]{..inputKey..} */
                     std::pair<typename BaseMap::iterator, bool> insertResult;
                     MemoryChunk key = _inputKey;
                     key.expandIncluding( nextKey );
                     _Type *data = nextIt->second;
                     _thisMap.erase( nextIt );
                     insertResult = _thisMap.insert( typename BaseMap::value_type( _inputKey, data ) ); //reuse data entry
                     _thisIt = insertResult.first;
                     _inputKey.cutAfter( _thisIt->first );
                  } else {
                     /* (this)[.. inputKey .. ][next]{inputKey} */
                     MemoryChunk key = _inputKey;
                     key.expandExcluding( nextKey );
                     nextIt->second->merge( *_inputData );
                     _inputKey.cutAfter( nextKey );
                     _thisIt = _thisMap.insert( _thisIt, typename BaseMap::value_type( key, NEW _Type( *_inputData ) ) );
                     _thisIt++;
                  }
               }
            }

            void expandTotalBeginOverlap( iterator &nextIt ) {
               const MemoryChunk &nextKey = nextIt->first;
               const MemoryChunk &thisKey = _thisIt->first;
               _nextAndInputDataAreEqual = nextIt->second->equal( *_inputData );
               if ( _thisAndInputDataAreEqual ) {
                  if ( _nextAndInputDataAreEqual ) {
                     /* expand to fit [this][next]{next:leftover}
                      * then erase next entry
                      */
                     MemoryChunk &thisNoConst = const_cast< MemoryChunk & >( _thisIt->first );
                     thisNoConst.expandIncluding( nextKey );
                     delete nextIt->second;
                     _thisMap.erase( nextIt );
                     _thisIt = _thisMap.find( thisKey );
                  } else {
                     /* fit (this){next:partition_with_input}(next:leftover)
                      * merge partition, continue processing leftover
                      */
                     nextIt->second->merge( *_inputData );
                  }
               } else {
                  if ( _nextAndInputDataAreEqual ) {
                     /* expand to fit (this)[next]{next:leftover}
                      * since next and input are equal, no action is required
                      */
                  } else {
                     /* expand to fit (this)[next]{next:leftover}
                      * similar than before but we need to merge the information
                      * of "next" and "input".
                      */
                     nextIt->second->merge( *_inputData );
                  }
               }
               _inputKey.cutAfter( nextKey );
               _thisIt++;
            }

            void expandSubchunkBeginOverlap( iterator &nextIt )
            {
               MemoryChunk nextKeyCopy = nextIt->first;
               const MemoryChunk &thisKey = _thisIt->first;
               _Type *nextData = nextIt->second;
               _Type *thisData = _thisIt->second;
               _nextAndInputDataAreEqual = nextIt->second->equal( *_inputData );
               if ( _thisAndInputDataAreEqual ) {
                  if ( _nextAndInputDataAreEqual ) {
                     MemoryChunk &thisNoConst = const_cast< MemoryChunk & >( _thisIt->first );
                     thisNoConst.expandIncluding( nextKeyCopy );
                     _thisMap.erase( nextIt );
                  } else {
                     if ( !nextData->contains( *_inputData ) ) {
                        MemoryChunk &nextNoConst = const_cast< MemoryChunk & >( nextIt->first );
                        nextNoConst = _inputKey;
                        nextKeyCopy.cutAfter( _inputKey );
                        _thisIt = _thisMap.insert( nextIt, typename BaseMap::value_type( nextKeyCopy, NEW _Type ( *nextData ) ) );
                        nextData->merge( *_inputData ); //merge after insert to create a copy of the original "next" entry.
                     } //else No action is needed because merge would not create a new entry
                  }
               } else {
                  if ( _nextAndInputDataAreEqual ) {
                     //no expand with "this" and "next" already ok, do nothing
                  } else {
                     _Type * tmpData = NEW _Type ( *_inputData );
                     tmpData->merge( *nextData );
                     if ( tmpData->equal( *thisData ) ) {
                        if ( thisData->equal( *nextData ) ) { //merge Equal with BOTH next and this
                           MemoryChunk &thisNoConst = const_cast< MemoryChunk & >( _thisIt->first );
                           thisNoConst.expandIncluding( nextKeyCopy );
                           _thisMap.erase( nextIt );
                           _thisIt = _thisMap.find( thisKey );
                        } else {
                           std::pair<typename BaseMap::iterator, bool> insertResult;
                           MemoryChunk &thisNoConst = const_cast< MemoryChunk & >( _thisIt->first );
                           thisNoConst.expandIncluding( _inputKey );
                           _thisMap.erase( nextIt );
                           MemoryChunk::partitionBeginAgtB( nextKeyCopy, _inputKey );
                           insertResult = _thisMap.insert( typename BaseMap::value_type( _inputKey, nextData ) ); //reuse entry
                           _thisIt = insertResult.first;
                           delete tmpData;
                        }
                     } else {
                        MemoryChunk &nextNoConst = const_cast< MemoryChunk & >( nextIt->first );
                        nextNoConst = _inputKey;
                        nextKeyCopy.cutAfter( _inputKey );
                        _thisIt = _thisMap.insert( nextIt, typename BaseMap::value_type( nextKeyCopy, tmpData ) );
                     }
                  }
               }
            }

            void expandTotalEndOverlap( iterator &nextIt ) {
               const MemoryChunk &nextKey = nextIt->first;
               _nextAndInputDataAreEqual = nextIt->second->equal( *_inputData );
               if ( _thisAndInputDataAreEqual ) {
                  MemoryChunk &thisNoConst = const_cast< MemoryChunk & >( _thisIt->first );
                  if ( _nextAndInputDataAreEqual ) {
                     thisNoConst.expandIncluding( nextKey );
                     delete nextIt->second;
                     _thisMap.erase( nextIt );
                     _thisIt = _thisMap.find( thisNoConst );
                  } else {
                     thisNoConst.expandExcluding( nextKey );
                     nextIt->second->merge( *_inputData );
                  }
               } else {
                  if ( _nextAndInputDataAreEqual ) {
                     _thisIt = _thisMap.insert( _thisIt, typename BaseMap::value_type( _inputKey, nextIt->second ) );
                     _thisIt++;
                     _thisMap.erase( _thisIt );
                     _thisIt = _thisMap.find( _inputKey );
                  } else {
                     _inputKey.expandExcluding( nextKey );
                     nextIt->second->merge( *_inputData );
                     _thisIt = _thisMap.insert( _thisIt, typename BaseMap::value_type( _inputKey, NEW _Type ( *_inputData ) ) );
                  }
               }
               _thisIt++;
            }
         };

         ExpandLocalFunctions localexp ( _thisMap, inputKey, inputData, thisIt );

         iterator nextIt = thisIt;
         nextIt++;
         _Type *thisData = thisIt->second;
         bool thisAndInputDataAreEqual = thisData->equal( *inputData );
         //fprintf(stderr, "entry %s, this %d-%d  input %d-%d\n", __FUNCTION__, thisKey.getAddress(), thisKey.getLength(), inputKey.getAddress(), inputKey.getLength() );

         if ( nextIt == _thisMap.end() )
         {
            if ( thisAndInputDataAreEqual ) {
               /* we can expand safetly, and we finish */
               MemoryChunk &thisNoConst = const_cast< MemoryChunk & >( thisIt->first );
               thisNoConst.expandIncluding( inputKey );
            } else {
               thisIt = _thisMap.insert( thisIt, typename BaseMap::value_type( inputKey, NEW _Type ( *inputData ) ) );
            }
            thisIt++;
         } else {
            MemoryChunk nextKey = nextIt->first;
            _Type *nextData = nextIt->second;
            bool nextAndInputDataAreEqual = nextData->equal( *inputData );

            if ( nextKey.equal( inputKey ) ) {
               //fprintf(stderr, "EQ! entry %s, next %d-%d  input %d-%d\n", __FUNCTION__, nextKey.getAddress(), nextKey.getLength(), inputKey.getAddress(), inputKey.getLength() );
               if ( thisAndInputDataAreEqual && nextAndInputDataAreEqual ) 
                  fprintf(stderr, "ERROR: I think this is an error. input, and next are equal! data also (this, next and input)\n");
               if ( !nextAndInputDataAreEqual ) {
                  nextData->merge( *inputData );
               }
               tryToMergeWithPreviousEntry( nextIt );
               nextIt++;
               if ( nextIt != _thisMap.end() ) {
                  tryToMergeWithPreviousEntry( nextIt );
               }
            } else {
               fprintf(stderr, "entry %s, next %ld-%ld  input %ld-%ld, overlap is %s\n", __FUNCTION__, nextKey.getAddress(), nextKey.getLength(), inputKey.getAddress(), inputKey.getLength(), MemoryChunk::strOverlap[ nextKey.checkOverlap( inputKey ) ] );
               /* check overlap with the next item in the map! */
               switch ( nextKey.checkOverlap( inputKey ) ) {
                  case MemoryChunk::NO_OVERLAP:
                     localexp.expandNoOverlap();
                     break;
                  case MemoryChunk::BEGIN_OVERLAP:
                     localexp.expandBeginOverlap( nextIt );
                     break;
                  case MemoryChunk::END_OVERLAP:
                     /* ERROR, it is impossible to have End Overlap with hintNext */
                     fprintf(stderr, "error END_OVERLAP %s\n", __FUNCTION__ );
                     break;
                  case MemoryChunk::TOTAL_OVERLAP:
                     localexp.expandTotalOverlap( nextIt );
                     expand( inputKey, inputData, thisIt );
                     break;
                  case MemoryChunk::SUBCHUNK_OVERLAP:
                     /* ERROR, it is impossible to have Subchunk Overlap with hintNext */
                     fprintf(stderr, "error SUBCHUNK_OVERLAP %s\n", __FUNCTION__ );
                     break;
                  case MemoryChunk::TOTAL_BEGIN_OVERLAP:
                     localexp.expandTotalBeginOverlap( nextIt );
                     expand( inputKey, inputData, nextIt );
                     break;
                  case MemoryChunk::SUBCHUNK_BEGIN_OVERLAP:
                     localexp.expandSubchunkBeginOverlap( nextIt );
                     break;
                  case MemoryChunk::TOTAL_END_OVERLAP:
                     localexp.expandTotalEndOverlap( nextIt );
                     break;
                  case MemoryChunk::SUBCHUNK_END_OVERLAP:
                     /* ERROR, it is impossible to have Subchunk End Overlap with hintNext */
                     fprintf(stderr, "error SUBCHUNK_END_OVERLAP %s\n", __FUNCTION__ );
                     break;
               }
            }
         }
      }

      void mergeNoOverlap( iterator &thisIt, const_iterator &inputIt ) {
         const MemoryChunk &thisKey = thisIt->first;
         MemoryChunk inputKey = inputIt->first;
         _Type *inputData = inputIt->second;
         _Type *thisData = thisIt->second;
         bool thisAndInputDataAreEqual = thisData->equal( *inputData );
         if ( inputKey.getAddress() + inputKey.getLength() == thisKey.getAddress() && thisAndInputDataAreEqual ) {
            std::pair<iterator, bool> insertResult;
            inputKey.expandIncluding( thisKey );
            _thisMap.erase( thisIt );
            insertResult = _thisMap.insert( typename BaseMap::value_type( inputKey, thisData ) ); //reuse data
            thisIt = insertResult.first;
            tryToMergeWithPreviousEntry( thisIt );
            inputIt++;
         } else if ( thisKey.getAddress() > inputKey.getAddress() ) {
            thisIt = _thisMap.insert( thisIt, typename BaseMap::value_type( inputKey, NEW _Type( *inputData ) ) );
            tryToMergeWithPreviousEntry( thisIt );
            thisIt++;
            inputIt++;
         } else {
            thisIt++;
         }
      }

      void mergeBeginOverlap( iterator &thisIt, const_iterator &inputIt ) {
         const MemoryChunk &thisKey = thisIt->first;
         MemoryChunk inputKey = inputIt->first;
         _Type *inputData = inputIt->second;
         _Type *thisData = thisIt->second;
         bool thisAndInputDataAreEqual = thisData->equal( *inputData );
         if ( thisAndInputDataAreEqual ) {
            /* I CAN MERGE */
            std::pair<typename BaseMap::iterator, bool> insertResult;
            inputKey.expandIncluding( thisKey );
            _thisMap.erase( thisIt );
            insertResult = _thisMap.insert( typename BaseMap::value_type( inputKey, thisData ) ); //reuse data
            // insertResult.second must be true
            thisIt = insertResult.first;
         } else {
            MemoryChunk rightLeftOver;
            MemoryChunk &thisNoConst = const_cast< MemoryChunk & >( thisIt->first );
            MemoryChunk::intersect( inputKey, thisNoConst, rightLeftOver );
            thisIt = _thisMap.insert( thisIt, typename BaseMap::value_type( inputKey, NEW _Type( *inputData ) ));
            thisIt = _thisMap.insert( thisIt, typename BaseMap::value_type( rightLeftOver, NEW _Type( *thisData ) ));
            thisData->merge( *inputData );
         }
         inputIt++;
      }

      void mergeEndOverlap( iterator &thisIt, const_iterator &inputIt ) {
         MemoryChunk thisKey = thisIt->first;
         MemoryChunk inputKey = inputIt->first;
         _Type *inputData = inputIt->second;
         _Type *thisData = thisIt->second;
         MemoryChunk rightLeftOver;
         bool thisAndInputDataAreEqual = thisData->equal( *inputData );
         if ( thisAndInputDataAreEqual || thisData->contains( *inputData ) ) {
            MemoryChunk::intersect( thisKey, inputKey, rightLeftOver );
         }
         else {
            MemoryChunk &thisNoConst = const_cast< MemoryChunk & >( thisIt->first );
            MemoryChunk::intersect( thisNoConst, inputKey, rightLeftOver );
            thisIt = _thisMap.insert( thisIt, typename BaseMap::value_type( inputKey, NEW _Type( *thisData ) ));
            thisIt->second->merge( *inputData );
            tryToMergeWithPreviousEntry( thisIt );
         }
         expand( rightLeftOver, inputData, thisIt );
         inputIt++;
      }

      void mergeTotalOverlap( iterator &thisIt, const_iterator &inputIt ) {
         MemoryChunk thisKey = thisIt->first;
         MemoryChunk inputKey = inputIt->first;
         _Type *inputData = inputIt->second;
         _Type *thisData = thisIt->second;
         MemoryChunk leftLeftOver;
         bool thisAndInputDataAreEqual = thisData->equal( *inputData );
         MemoryChunk::partition( inputKey, thisKey, leftLeftOver );
         if ( thisAndInputDataAreEqual ) {
            std::pair<typename BaseMap::iterator, bool> insertResult;
            inputKey.expandIncluding( thisKey );
            _thisMap.erase( thisIt );
            insertResult = _thisMap.insert( typename BaseMap::value_type( inputKey, thisData ) ); // do not allocate new data, reuse old
            thisIt = insertResult.first;
         } else {
            thisIt->second->merge( *inputData );
            thisIt = _thisMap.insert( thisIt, typename BaseMap::value_type( inputKey, inputData ));
            thisIt++;
         }
         expand( leftLeftOver, inputData, thisIt );
         inputIt++;
      }

      void mergeSubchunkOverlap( iterator &thisIt, const_iterator &inputIt ) {
         MemoryChunk inputKey = inputIt->first;
         _Type *inputData = inputIt->second;
         _Type *thisData = thisIt->second;
         MemoryChunk leftLeftOver;
         bool thisAndInputDataAreEqual = thisData->equal( *inputData );
         if ( thisAndInputDataAreEqual || thisData->contains( *inputData ) ) {
            //do nothing
         } else {
            MemoryChunk &thisNoConst = const_cast< MemoryChunk & >( thisIt->first );
            MemoryChunk::partition( thisNoConst, inputKey, leftLeftOver );
            thisIt = _thisMap.insert( thisIt, typename BaseMap::value_type( inputKey, NEW _Type( *inputData ) ) );
            thisIt->second->merge( *thisData );
            thisIt = _thisMap.insert( thisIt, typename BaseMap::value_type( leftLeftOver, NEW _Type( *thisData ) ) );
         }
         inputIt++;
      }

      void mergeTotalBeginOverlap( iterator &thisIt, const_iterator &inputIt ) {
         MemoryChunk thisKey = thisIt->first;
         MemoryChunk inputKey = inputIt->first;
         _Type *inputData = inputIt->second;
         _Type *thisData = thisIt->second;
         bool thisAndInputDataAreEqual = thisData->equal( *inputData );
         MemoryChunk::partitionBeginAltB( thisKey, inputKey );
         if ( thisAndInputDataAreEqual ) { //do nothing
         } else {
            thisIt->second->merge( *inputData );
            tryToMergeWithPreviousEntry( thisIt );
         }
         inputKey.cutAfter( thisKey );
         expand( inputKey, inputData, thisIt );
         inputIt++;
      }

      void mergeSubchunkBeginOverlap( iterator &thisIt, const_iterator &inputIt ) {
         MemoryChunk inputKey = inputIt->first;
         _Type *inputData = inputIt->second;
         _Type *thisData = thisIt->second;
         bool thisAndInputDataAreEqual = thisData->equal( *inputData );
         if ( thisAndInputDataAreEqual || thisData->contains( *inputData ) ) {
            //do_nothing
         } else {
            iterator thisItCopy;
            MemoryChunk &thisNoConst = const_cast< MemoryChunk & >( thisIt->first );
            MemoryChunk::partitionBeginAgtB( thisNoConst, inputKey );
            thisIt = _thisMap.insert( thisIt, typename BaseMap::value_type( inputKey, NEW _Type( *thisData ) ) );
            thisItCopy = thisIt; thisItCopy--;
            thisData->merge( *inputData );
            tryToMergeWithPreviousEntry( thisItCopy );
         }
         inputIt++;
      }

      void mergeTotalEndOverlap( iterator &thisIt, const_iterator &inputIt ) {
         MemoryChunk inputKey = inputIt->first;
         _Type *inputData = inputIt->second;
         _Type *thisData = thisIt->second;
         bool thisAndInputDataAreEqual = thisData->equal( *inputData );
         if ( thisAndInputDataAreEqual ) {
            std::pair<typename BaseMap::iterator, bool> insertResult;
            _thisMap.erase( thisIt );
            insertResult = _thisMap.insert( typename BaseMap::value_type( inputKey, thisData ) ); //reuse data
            thisIt = insertResult.first;
         } else {
            MemoryChunk &thisNoConst = const_cast< MemoryChunk & >( thisIt->first );
            MemoryChunk::partitionEnd( inputKey, thisNoConst );
            thisIt = _thisMap.insert( thisIt, typename BaseMap::value_type( inputKey, NEW _Type ( *inputData ) ) );
            thisIt++;
            thisData->merge( *inputData );
            iterator nextIt = thisIt;
            nextIt++;
            if ( nextIt != _thisMap.end() ) {
               tryToMergeWithPreviousEntry( nextIt );
            }
         }
         thisIt++;
         inputIt++;
      }

      void mergeSubchunkEndOverlap( iterator &thisIt, const_iterator &inputIt ) {
         MemoryChunk inputKey = inputIt->first;
         _Type *inputData = inputIt->second;
         _Type *thisData = thisIt->second;
         bool thisAndInputDataAreEqual = thisData->equal( *inputData );
         if ( thisAndInputDataAreEqual ) {
            //do_nothing;
         } else {
            MemoryChunk &thisNoConst = const_cast< MemoryChunk & >( thisIt->first );
            MemoryChunk::partitionEnd( thisNoConst, inputKey );
            thisIt = _thisMap.insert( thisIt, typename BaseMap::value_type( inputKey, NEW _Type( *inputData ) ) );
            thisIt->second->merge( *thisData );
         }
         thisIt++;
         inputIt++;
      }

   };
   LocalFunctions local( *this );

   const_iterator inputIt = mm.begin();
   iterator thisIt = this->begin();

   while ( inputIt != mm.end() )
   {
      if ( thisIt == this->end() ) {
         //insert the rest of the input MemoryMap, because there are no more entries in "this"
         for (; inputIt != mm.end(); inputIt++ )
         {
            thisIt = this->insert( thisIt, typename BaseMap::value_type( inputIt->first, NEW _Type ( *( inputIt->second ) ) ) );
            local.tryToMergeWithPreviousEntry( thisIt );
         }
      } else {
         MemoryChunk thisKey = thisIt->first, inputKey = inputIt->first;
         _Type *thisData = thisIt->second, *inputData = inputIt->second;
         bool thisAndInputDataAreEqual = thisData->equal( *inputData );
         if ( thisKey.equal( inputKey ) ) {
            if ( !thisAndInputDataAreEqual )
            {
               thisIt->second->merge( *inputData );
               local.tryToMergeWithPreviousEntry( thisIt );
            }
            if ( thisKey.equal( thisIt->first) ) {
               thisIt++; //otherwhise it has been merged with prev
            }
            inputIt++;
         } else {
            switch ( thisKey.checkOverlap( inputKey ) )
            {
               case MemoryChunk::NO_OVERLAP:
                  local.mergeNoOverlap( thisIt, inputIt );
                  break;
               case MemoryChunk::BEGIN_OVERLAP:
                  local.mergeBeginOverlap( thisIt, inputIt );
                  break;
               case MemoryChunk::END_OVERLAP:
                  local.mergeEndOverlap( thisIt, inputIt );
                  break;
               case MemoryChunk::TOTAL_OVERLAP:
                  local.mergeTotalOverlap( thisIt, inputIt );
                  break;
               case MemoryChunk::SUBCHUNK_OVERLAP:
                  local.mergeSubchunkOverlap( thisIt, inputIt );
                  break;
               case MemoryChunk::TOTAL_BEGIN_OVERLAP:
                  local.mergeTotalBeginOverlap( thisIt, inputIt );
                  break;
               case MemoryChunk::SUBCHUNK_BEGIN_OVERLAP:
                  local.mergeSubchunkBeginOverlap( thisIt, inputIt );
                  break;
               case MemoryChunk::TOTAL_END_OVERLAP:
                  local.mergeTotalEndOverlap( thisIt, inputIt );
                  break;
               case MemoryChunk::SUBCHUNK_END_OVERLAP:
                  local.mergeSubchunkEndOverlap( thisIt, inputIt );
                  break;
            }
         }
      }
   }
   while ( thisIt != this->end() ) {
      local.tryToMergeWithPreviousEntry( thisIt );
      thisIt++;
   }
}

}

#endif /* MEMORYMAP_H */
