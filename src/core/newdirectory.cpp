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

#else
#include "basethread.hpp"
#include "debug.hpp"
#endif

#include "newdirectory_decl.hpp"
#include "memorymap.hpp"
#include "atomic.hpp"

using namespace nanos;

New1dDirectory::New1dDirectory() : _directory(), _parent(NULL), _mergeLock(), _outputMergeLock() {}

New1dDirectory *New1dDirectory::_root = NULL;

void New1dDirectory::setParent( New1dDirectory *parent )
{
   _parent =  parent;
}

void New1dDirectory::registerAccess( uint64_t tag, std::size_t len, bool input, bool output, unsigned int memorySpaceId/*, DirectoryOps &ops*/ )
{
   MemoryMap< NewDirectoryEntryData >::MemChunkList resultListInput;
   MemoryMap< NewDirectoryEntryData >::MemChunkList resultList;
   MemoryMap< NewDirectoryEntryData >::MemChunkList::iterator it;

   _inputDirectory.getOrAddChunk( tag, len, resultListInput );
   unsigned int version = 1;

   for ( it = resultListInput.begin(); it != resultListInput.end(); it++ )
   {
      NewDirectoryEntryData **ent = it->second;
      if ( *ent == NULL )
      {
         if ( this->isRoot() ) {
            message("This is an error because _root directory is not supposed to use [registerAccess]");
         } else {
            message("lookup parent "<<(void*)&_parent->_inputDirectory << " tag "<<(void*)it->first->getAddress());
            MemoryMap< NewDirectoryEntryData >::ConstMemChunkList subchunkResults;
            MemoryMap< NewDirectoryEntryData >::ConstMemChunkList::iterator subIt;
            _parent->_inputDirectory.getChunk2( it->first->getAddress(), it->first->getLength(), subchunkResults );
            ensure(subchunkResults.size() > 0, "parent query result must be > 0");
            for (subIt = subchunkResults.begin(); subIt != subchunkResults.end(); subIt++ )
            {
               NewDirectoryEntryData * const *pEnt = subIt->second;
               if ( pEnt == NULL )
               {
                  if ( _parent->isRoot() )
                  {
                     *ent = NEW NewDirectoryEntryData( );
                     message("IVE CREATED AN ENTRY FOR TAG " << (void *) it->first->getAddress());
                  } else {
                     message("ERROR: invalid program! data spec not available.");
                  }
               }
               else
               {
                  message("FOUND on the parent");
                  *ent = NEW NewDirectoryEntryData( **pEnt ); //fixme assuming 1 return entry!!! fix it!
               }
               delete subIt->first;
            }
         }
      } else {
         message(" IVE GOT IT");
      }

      message("This version is " << (*ent)->getVersion() );
      if ( input && !output ) {
         message("I just want a replica to read " << (void *)tag );

         if ( (*ent)->hasWriteLocation() ) {
            message("Latest version is located in " << (*ent)->getWriteLocation() );
         } else { 
            message("No one has it for writing, is in 0? " << (*ent)->isLocatedIn(0) << ", in 1? " << (*ent)->isLocatedIn(1) );
         }
      } else if ( input && output ) {
         message("I want to be the master!! " << (void *)tag );
         if ( (*ent)->hasWriteLocation() ) {
            message("Latest version is located in " << (*ent)->getWriteLocation() );
         } else { 
            message("No one has it for writing, is in 0? " << (*ent)->isLocatedIn(0) << ", in 1? " << (*ent)->isLocatedIn(1) );
         }
      } else if ( !input && output ) {
         message("I want to generate some data " << (void *)tag );
         if ( (*ent)->hasWriteLocation() ) {
            message("Latest version is located in " << (*ent)->getWriteLocation() );
         } else { 
            message("No one has it for writing, is in 0? " << (*ent)->isLocatedIn(0) << ", in 1? " << (*ent)->isLocatedIn(1) );
         }
      } else { 
         message("WTF!!! no input, no output! " << (void *) tag );
      }
      version = (*ent)->getVersion() > version ? (*ent)->getVersion() : version;
   }

   /*
    * insert the data in the "output" directory, this acts as a merge, since the previous query could have been
    * fragmented, but in the final directory it will appear as a single entry.
    */
   _directory.getOrAddChunk( tag, len, resultList );
   ensure(resultList.size() == 1 , "It can not happen than a directory query returns more than one entry.");
   NewDirectoryEntryData **newEnt = resultList.front().second; 
   ensure( *newEnt == NULL, "It can not happen that a directory entry already existed ");
   
    *newEnt = NEW NewDirectoryEntryData( ); //fixme version
    (*newEnt)->setVersion( version );
    (*newEnt)->addAccess( memorySpaceId );
    if ( output ) { (*newEnt)->setWriteLocation( memorySpaceId ); (*newEnt)->increaseVersion();}
    else if ( input ) (*newEnt)->setWriteLocation( -1 );
    message("At the end this data is going to be in " << memorySpaceId );
}

void New1dDirectory::merge( const New1dDirectory &inputDir )
{
   message("merge " << &inputDir << " to input directory " << (void *)&_inputDirectory<< " read from " << (void *)&inputDir._directory );
   MemoryMap<NewDirectoryEntryData>::const_iterator it;
   _mergeLock.acquire();
   //_inputDirectory.insert( inputDir._directory.begin(), inputDir._directory.end() );
   _inputDirectory.merge2( inputDir._directory );
   _mergeLock.release();
   message("now dir has " << _inputDirectory.size());
}
void New1dDirectory::mergeOutput( const New1dDirectory &inputDir )
{
   message("merge " << &inputDir << " to output directory " << (void *)&_directory << " read from " << (void *)&inputDir._directory );
   _outputMergeLock.acquire();
   //_directory.insert( inputDir._directory.begin(), inputDir._directory.end() );
message("DIRECTORY TO MERGE");
   inputDir.print();
   _directory.merge2( inputDir._directory );
message("RESULT DIRECTORY");
   _directory.print();
   _outputMergeLock.release();
   message("now dir has " << _directory.size());
}

void New1dDirectory::setRoot() {
   _root = this;
}

bool New1dDirectory::isRoot() const {
   return ( _root == this );
}

void New1dDirectory::consolidate() {
   message( "consolidating directory..." );
 
   message( "_inputDir size is " << _inputDirectory.size() << " _dir size is " << _directory.size());
   _inputDirectory.insert( _directory.begin(), _directory.end() );
   _inputDirectory.print();
   //_directory.clear();
   message( "_inputDir ("<<(void*)&_inputDirectory<<") size is " << _inputDirectory.size() << " _dir size is " << _directory.size());
}

void New1dDirectory::print() const {
   fprintf(stderr, "printing inputDirectory:\n");
   _inputDirectory.print();
   fprintf(stderr, "printing directory:\n");
   _directory.print();
}

bool New1dDirectory::checkConsistency( uint64_t tag, std::size_t size, unsigned int memorySpaceId ) {
   bool ok = true;
   MemoryMap< NewDirectoryEntryData >::ConstMemChunkList subchunkResults;
   _inputDirectory.getChunk2( tag, size, subchunkResults );
   for ( MemoryMap< NewDirectoryEntryData >::ConstMemChunkList::iterator it = subchunkResults.begin(); it != subchunkResults.end() && ok; it++ ) {
      if ( it->second != NULL ) {
         if ( *(it->second) != NULL ) {
            ok = ok && (*(it->second))->isLocatedIn( memorySpaceId );
         } else {
            ok = false;
         }
      } else {
         ok = false;
      }
   }
   return ok;
}
