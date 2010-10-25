/*************************************************************************************/
/*      Copyright 2009 Barcelona Supercomputing Center                               */
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
/*      along with NANOS++.  If not, see <http://www.gnu.org/licenses/>.             */
/*************************************************************************************/

#ifndef _NANOS_DIRECTORY_H
#define _NANOS_DIRECTORY_H

#include "directory_decl.hpp"
#include "hashmap.hpp"
#include "cache_decl.hpp"

using namespace nanos;

inline const Entry& Entry::operator= ( const Entry &ent )
{
   if ( this == &ent ) return *this;
   _tag = ent._tag;
   _version = ent._version;
   return *this;
}

inline uint64_t Entry::getTag() const { return _tag; }

inline void Entry::setTag( uint64_t tag) { _tag = tag; }

inline unsigned int Entry::getVersion() const { return _version.value(); }

inline void Entry::setVersion ( unsigned int version ) { _version = version; }

inline bool Entry::setVersionCS ( unsigned int version )
{
   Atomic< unsigned int > oldVal = _version.value();
   Atomic< unsigned int > newVal = version;
   return _version.cswap( oldVal, newVal );
}

inline void Entry::increaseVersion() { _version++; }

inline const DirectoryEntry& DirectoryEntry::operator= ( const DirectoryEntry &ent )
{
   if ( this == &ent ) return *this;
   setTag( ent.getTag() );
   setVersion( ent.getVersion() );
   _owner = ent._owner;
   return *this;
}

inline Cache * DirectoryEntry::getOwner() const { return _owner; }

inline void DirectoryEntry::setOwner( Cache *owner ) { _owner = owner; }

inline bool DirectoryEntry::isInvalidated()
{
   return _invalidated.value();
}

inline void DirectoryEntry::setInvalidated( bool invalid )
{
   _invalidated = invalid;
}

inline bool DirectoryEntry::trySetInvalidated()
{
   Atomic<bool> expected = false;
   Atomic<bool> value = true;
   return _invalidated.cswap( expected, value );
}

inline DirectoryEntry& Directory::insert( uint64_t tag, DirectoryEntry &ent,  bool &inserted )
{
   return _directory.insert( tag, ent, inserted );
}

inline DirectoryEntry& Directory::newEntry( uint64_t tag, unsigned int version, Cache* owner )
{
   DirectoryEntry& de = _directory[tag];
   de.setTag( tag );
   de.setVersion( version );
   de.setOwner( owner );
   return de;
}

inline DirectoryEntry* Directory::getEntry( uint64_t tag )
{
   return _directory.find( tag );
}

inline void Directory::registerAccess( uint64_t tag, size_t size, bool input, bool output )
{
   DirectoryEntry *de = _directory.find( tag );
   if ( de != NULL ) {
      if ( input ) {
         Cache *c = de->getOwner();
         if ( c != NULL )
            c->invalidate( tag, size, de );
      }
      if ( output ) {
         de->setVersion(de->getVersion()+1);
      }
   }
}

inline void Directory::waitInput( uint64_t tag )
{
   DirectoryEntry *de = _directory.find( tag );
   if ( de != NULL ) // The entry may have never been registered
      while ( de->getOwner() != NULL );
}

inline void Directory::synchronizeHost()
{
   std::list<DirectoryEntry *> flushings;
   DirectoryMap::iterator it = _directory.begin();
   while ( it != _directory.end() ) {
      DirectoryEntry &de = *it;
      Cache * c = de.getOwner();
      if ( c == NULL ) {
         // Invalidate all non-dirty copies
         de.setVersion( de.getVersion()+1 );
      } else {
         // Froce copy back
         c->invalidate( de.getTag(), &de );
         flushings.push_back( &de );
      }
      it++;
   }
   for ( std::list<DirectoryEntry *>::iterator deIt=flushings.begin(); deIt != flushings.end(); deIt++ ) {
      DirectoryEntry *de = *deIt;
      Cache *c = de->getOwner();
      if ( c != NULL ) {
         c->syncTransfer( de->getTag() );
         while (  de->getOwner() != NULL );
      }
   }
}

inline void Directory::synchronizeHost( std::list<uint64_t> syncTags )
{
   std::list<DirectoryEntry *> flushings;
   for ( std::list<uint64_t>::iterator it = syncTags.begin(); it != syncTags.end(); it++ ) {
      DirectoryEntry *de = _directory.find( *it );
      if ( de != NULL ) {
         Cache * c = de->getOwner();
         if ( c == NULL ) {
            // Invalidate all non-dirty copies
            de->setVersion( de->getVersion()+1 );
         } else {
            // Froce copy back
            c->invalidate( de->getTag(), de );
            flushings.push_back( de );
         }
      }
   }
   for ( std::list<DirectoryEntry *>::iterator deIt=flushings.begin(); deIt != flushings.end(); deIt++ ) {
      DirectoryEntry *de = *deIt;
      Cache *c = de->getOwner();
      if ( c != NULL ) {
         c->syncTransfer( de->getTag() );
         while (  de->getOwner() != NULL );
      }
   }
}

#endif
