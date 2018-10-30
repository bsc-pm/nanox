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

#ifdef STANDALONE_TEST

#ifdef message
#undef message
#define message(x)
#else
#define message(x)
#endif
#ifdef ensure
#undef ensure
#define ensure(x,y)
#else
#define ensure(x,y)
#endif
#ifndef NEW
#define NEW new
#endif

#endif
#include "memorymap.hpp"

using namespace nanos;

const char* MemoryChunk::strOverlap[] = {
   "NO_OVERLAP",
   "BEGIN_OVERLAP",
   "END_OVERLAP",
   "TOTAL_OVERLAP",
   "SUBCHUNK_OVERLAP",
   "TOTAL_BEGIN_OVERLAP",
   "SUBCHUNK_BEGIN_OVERLAP",
   "TOTAL_END_OVERLAP",
   "SUBCHUNK_END_OVERLAP"
};

#if 1
void MemoryMap< uint64_t >::insertWithOverlapButNotGenerateIntersects( const MemoryChunk &key, iterator &hint, uint64_t data )
{
   bool lastChunk = false;
   MemoryChunk iterKey = key;
   bool reuseFirstPos = false;
   int firstCase = 0;

   if ( !this->empty() )
   {
      if ( hint != this->begin() )
         hint--;
      iterator firstPos = hint;

      MemoryChunk::OverlapType hintOverlap = hint->first.checkOverlap( iterKey );
      if ( hintOverlap == MemoryChunk::NO_OVERLAP && iterKey.getAddress() > hint->first.getAddress()) {
         hint++; firstPos++;
      }

      switch ( hintOverlap )
      {
         case MemoryChunk::NO_OVERLAP:
               firstCase = __LINE__;
               lastChunk = true;
            break;
         case MemoryChunk::BEGIN_OVERLAP:
            firstCase = __LINE__;
            lastChunk = true; //this is a final situation since the to-be-inserted chunk ends.
            break;
         case MemoryChunk::END_OVERLAP:
            firstCase = __LINE__;
            reuseFirstPos = true;
            break;
         case MemoryChunk::TOTAL_OVERLAP:
            break;
         case MemoryChunk::SUBCHUNK_OVERLAP:
            firstCase = __LINE__;
            reuseFirstPos = true;
            lastChunk = true;
            break;
         case MemoryChunk::TOTAL_BEGIN_OVERLAP:
            firstCase = __LINE__;
            reuseFirstPos = true;
            break;
         case MemoryChunk::SUBCHUNK_BEGIN_OVERLAP:
            firstCase = __LINE__;
            reuseFirstPos = true;
            lastChunk = true;
            break;
         case MemoryChunk::TOTAL_END_OVERLAP:
            firstCase = __LINE__;
            lastChunk = true;
            break;
         case MemoryChunk::SUBCHUNK_END_OVERLAP:
            firstCase = __LINE__;
            reuseFirstPos = true;
            lastChunk = true;
            break;
      }

      bool expandToLast = false;
      bool dontDeleteLast = false;

      while ( hint != this->end() && !lastChunk  )  {
         hint++;
         switch ( hint->first.checkOverlap( iterKey ) )
         {
            case MemoryChunk::NO_OVERLAP:
               if ( iterKey.getAddress() < hint->first.getAddress() ) {
                  lastChunk = true;
                  dontDeleteLast = true;
               } else {
                  std::cerr <<"ERROR @ "<< __FUNCTION__<<":"<<__LINE__ << std::endl;
               }
               break;
            case MemoryChunk::BEGIN_OVERLAP:
               expandToLast = true;
               lastChunk = true; //this is a final situation since the to-be-inserted chunk ends.
               break;
            case MemoryChunk::END_OVERLAP: //impossible here
               std::cerr <<"ERROR @ "<< __FUNCTION__<<":"<<__LINE__ << std::endl;
               break;
            case MemoryChunk::TOTAL_OVERLAP:
               break;
            case MemoryChunk::SUBCHUNK_OVERLAP:
               std::cerr <<"ERROR @ "<< __FUNCTION__<<":"<<__LINE__ <<" first case "<< firstCase<< std::endl;
               lastChunk = true;
               break;
            case MemoryChunk::TOTAL_BEGIN_OVERLAP:
               break;
            case MemoryChunk::SUBCHUNK_BEGIN_OVERLAP:
               expandToLast = true;
               lastChunk = true;
               break;
            case MemoryChunk::TOTAL_END_OVERLAP:
               lastChunk = true;
               break;
            case MemoryChunk::SUBCHUNK_END_OVERLAP:
               std::cerr <<"ERROR @ "<< __FUNCTION__<<":"<<__LINE__ << std::endl;
               lastChunk = true;
               break;
         }
      }

      if ( firstPos == hint ) { //range is [firstPos, firstPos)
         if ( reuseFirstPos ) { //overlap with the firstPos, and we must reuse it so don't insert.
            MemoryChunk &firstChunk = const_cast< MemoryChunk & >( hint->first );
            firstChunk.expandIncluding( iterKey );
         } else { //no overlap, and we can insert
            hint = this->insert( hint, BaseMap::value_type( iterKey, data ) );
         }
      } else { //range is [firstPos, hint)
         if ( reuseFirstPos ) { //don't delete first pos
            MemoryChunk &firstChunk = const_cast< MemoryChunk & >( firstPos->first );
            iterator secondPos = firstPos;
            secondPos++;
            firstChunk.expandIncluding( ( expandToLast ? hint->first : iterKey ) );
            if ( !dontDeleteLast ) hint++; //include hint into the deleted range
            this->erase( secondPos, hint );
         } else {
            if ( expandToLast ) iterKey.expandIncluding( hint->first );
            this->insert( firstPos, BaseMap::value_type( iterKey, data ) );
            if ( !dontDeleteLast ) hint++; //include hint into the deleted range
            this->erase( firstPos, hint );
         }
      }
   } else {
      hint = this->insert( hint, BaseMap::value_type( iterKey, data ) );
   }
}

void MemoryMap< uint64_t >::addChunk( uint64_t addr, std::size_t len, uint64_t data )
{
   MemoryChunk key( addr, len );

   iterator it = this->lower_bound( key );
   if ( it == this->end() || this->key_comp()( key, it->first ) || it->first.getLength() != len )
   {
      /* NOT EXACT ADDR FOUND: "addr" is higher than any odther addr in the map OR less than "it" */
      insertWithOverlapButNotGenerateIntersects( key, it, data );
   }
}


uint64_t MemoryMap< uint64_t >::getExactInsertIfNotFound( uint64_t addr, std::size_t len, uint64_t valIfNotFound, uint64_t valIfNotValid ) {
   uint64_t val;
   MemoryChunk key( addr, len );
   iterator it = this->lower_bound( key );
   if ( it == this->end() )
   {
      /* not exact match found, check it-- */
      if ( this->size() == 0 ) {
         it = this->insert( it, BaseMap::value_type( key, valIfNotFound ) );
         val = it->second;
      } else {
         it--;
         if ( it->first.checkOverlap( key ) == MemoryChunk::NO_OVERLAP ) {
            it = this->insert( it, BaseMap::value_type( key, valIfNotFound ) );
            val = it->second;
         } else {
            val = valIfNotValid;
         }
      }
   } else if ( this->key_comp()( key, it->first ) ) {
      /* less than addr */
      if ( it->first.checkOverlap( key ) == MemoryChunk::NO_OVERLAP ) {
         it = this->insert( it, BaseMap::value_type( key, valIfNotFound ) );
         val = it->second;
      } else {
         val = valIfNotValid;
      }
   } else if ( it->first.getLength() != len ) {
      /* same addr, wrong length */
      val = valIfNotValid;
   } else {
      /* exact address found */
      val = it->second;
   }
   return val;
}

uint64_t MemoryMap< uint64_t >::getExactByAddress( uint64_t addr, uint64_t valIfNotFound ) const {
   uint64_t val = valIfNotFound;
   MemoryChunk key( addr, 0 );
   const_iterator it = this->lower_bound( key );
   if ( it != this->end() && !this->key_comp()( key, it->first ) )
   {
      /* exact address found */
      val = it->second;
   }
   return val;
}


uint64_t MemoryMap< uint64_t >::getExactOrFullyOverlappingInsertIfNotFound( uint64_t addr, std::size_t len, bool &exact, uint64_t valIfNotFound, uint64_t valIfNotValid, uint64_t &conflictAddr, std::size_t &conflictSize ) {
   uint64_t val = valIfNotValid;
   MemoryChunk key( addr, len );
   iterator it = this->lower_bound( key );
   if ( it == this->end() )
   {
      /* not exact match found, check it-- */
      if ( this->size() == 0 ) {
         it = this->insert( it, BaseMap::value_type( key, valIfNotFound ) );
         val = it->second;
         exact = true;
      } else {
         it--;
         MemoryChunk::OverlapType ov = it->first.checkOverlap( key );
         if ( ov == MemoryChunk::NO_OVERLAP ) {
            it = this->insert( it, BaseMap::value_type( key, valIfNotFound ) );
            val = it->second;
            exact = true;
         } else if ( ov == MemoryChunk::SUBCHUNK_OVERLAP ||
               ov == MemoryChunk::SUBCHUNK_BEGIN_OVERLAP ||
               ov == MemoryChunk::SUBCHUNK_END_OVERLAP) {
            val = it->second;
            exact = false;
         } else {
            conflictAddr = it->first.getAddress();
            conflictSize = it->first.getLength();
            val = valIfNotValid;
            exact = false;
         }
      }
   } else if ( this->key_comp()( key, it->first ) ) {
      /* key less than it->first, not total overlap guaranteed */
      if ( it->first.checkOverlap( key ) == MemoryChunk::NO_OVERLAP ) {
         if ( it == this->begin() ) {
            it = this->insert( it, BaseMap::value_type( key, valIfNotFound ) );
            val = it->second;
            exact = true;
         } else {
            it--;
            MemoryChunk::OverlapType ov = it->first.checkOverlap( key );
            if ( ov == MemoryChunk::NO_OVERLAP ) {
               it = this->insert( it, BaseMap::value_type( key, valIfNotFound ) );
               val = it->second;
               exact = true;
            } else if ( ov == MemoryChunk::SUBCHUNK_OVERLAP ||
                  ov == MemoryChunk::SUBCHUNK_BEGIN_OVERLAP ||
                  ov == MemoryChunk::SUBCHUNK_END_OVERLAP) {
               val = it->second;
               exact = false;
            } else {
               conflictAddr = it->first.getAddress();
               conflictSize = it->first.getLength();
               val = valIfNotValid;
               exact = false;
            }
         }
      } else {
         conflictAddr = it->first.getAddress();
         conflictSize = it->first.getLength();
         val = valIfNotValid;
         exact = false;
      }
   } else if ( it->first.getLength() != len ) {
      /* same addr, wrong length */
      MemoryChunk::OverlapType ov = it->first.checkOverlap( key );
      if ( ov == MemoryChunk::SUBCHUNK_OVERLAP ||
            ov == MemoryChunk::SUBCHUNK_BEGIN_OVERLAP ||
            ov == MemoryChunk::SUBCHUNK_END_OVERLAP) {
         val = it->second;
         exact = false;
      } else {
         conflictAddr = it->first.getAddress();
         conflictSize = it->first.getLength();
         val = valIfNotValid;
         exact = false;
      }
   } else {
      /* exact address found */
      val = it->second;
      exact = true;
   }
   return val;
}

void MemoryMap< uint64_t >::eraseByAddress( uint64_t addr ) {
   MemoryChunk key( addr, 0 );
   iterator it = this->lower_bound( key );
   if ( it == this->end() || this->key_comp()( key, it->first ) )
   {
      std::cerr << "Could not erase, address not found." << std::endl;
      exit(-1);
   } else {
      this->erase( it );
   }
}

#endif
