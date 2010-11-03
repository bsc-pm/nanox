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

#ifndef _NANOS_DIRECTORY_DECL_H
#define _NANOS_DIRECTORY_DECL_H

#include "compatibility.hpp"
#include "cache_fwd.hpp"
#include "hashmap_decl.hpp"
#include "atomic.hpp"

namespace nanos
{
   class Entry
   {
      private:

         uint64_t _tag;
         Atomic<unsigned int> _version;

      public:

         Entry() : _tag(0), _version(0) { }

         Entry ( uint64_t tag, unsigned int version ) : _tag( tag ), _version( version ) { }

         Entry ( const Entry &ent ) : _tag( ent._tag ), _version ( ent._version ) { }

         ~Entry () {}

         const Entry& operator= ( const Entry &ent );

         uint64_t getTag() const;

         void setTag( uint64_t tag);

         unsigned int getVersion() const;

         void setVersion( unsigned int version );

         void increaseVersion();

         bool setVersionCS( unsigned int version );
   };

   class DirectoryEntry : public Entry
   {
      private:

         //Atomic<Cache *> _owner;
         Cache * _owner;

         Lock _entryLock;

         Atomic<bool> _invalidated;

      public:

         DirectoryEntry() : Entry(), _owner( NULL ), _entryLock(), _invalidated( false ) { }

         DirectoryEntry( uint64_t tag, unsigned int version, Cache *c ) : Entry( tag, version ), _owner( c ), _entryLock(), _invalidated( false ) { }

         DirectoryEntry ( const DirectoryEntry &de) : Entry( de ), _owner( de._owner ), _entryLock(), _invalidated( false ) { }

         ~DirectoryEntry () {}

         const DirectoryEntry& operator= ( const DirectoryEntry &ent );

         Cache * getOwner() const;

         void setOwner( Cache *owner );

         bool isInvalidated();

         void setInvalidated( bool invalid );

         bool trySetInvalidated();

   };

   class Directory
   {
      private:

         typedef HashMap<uint64_t, DirectoryEntry> DirectoryMap;
         DirectoryMap _directory;

         // disable copy constructor and assignment operator
         Directory( const Directory &dir );
         const Directory & operator= ( const Directory &dir );

      public:

         Directory() : _directory() { }

         ~Directory() { }

         DirectoryEntry& insert( uint64_t tag, DirectoryEntry &ent, bool &inserted );

         DirectoryEntry& newEntry( uint64_t tag, unsigned int version, Cache* owner );

         DirectoryEntry* getEntry( uint64_t tag );

         void registerAccess( uint64_t tag, size_t size, bool input, bool output );

         void waitInput( uint64_t tag );

         void synchronizeHost();
         void synchronizeHost( std::list<uint64_t> syncTags );
   };

};

#endif
