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

using namespace nanos;

inline const Entry& Entry::operator= ( const Entry &ent )
{
   if ( this == &ent ) return *this;
   _tag = ent._tag;
   _version = ent._version;
}

inline uint64_t Entry::getTag() const { return _tag; }

inline void Entry::setTag( uint64_t tag) { _tag = tag; }

inline unsigned int Entry::getVersion() const { return _version; }

inline void Entry::setVersion ( unsigned int version ) { _version = version; }

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
   DirectoryMap::iterator it = _directory.find( tag );
   if ( it == _directory.end() )
      return NULL;
   DirectoryEntry& de = (*it).second;
   return &de;
}

#endif
