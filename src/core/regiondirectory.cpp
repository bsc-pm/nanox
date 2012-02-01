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

#include "regiondirectory_decl.hpp"
#include "regionbuilder.hpp"
#include "atomic.hpp"

using namespace nanos;

NewRegionDirectory::NewRegionDirectory() : _directory(), _parent(NULL), _mergeLock(), _outputMergeLock() {}

NewRegionDirectory *NewRegionDirectory::_root = NULL;

void NewRegionDirectory::setParent( NewRegionDirectory *parent )
{
   _parent =  parent;
}

void NewRegionDirectory::insertRegionIntoTree( RegionTree<NewDirectoryEntryData> &dir, Region const &reg, unsigned int memorySpaceId, bool setLoc, NewDirectoryEntryData const &ent )
{
   RegionTree<NewDirectoryEntryData>::iterator_list_t insertOuts;
   RegionTree<NewDirectoryEntryData>::iterator ret;
   ret = dir.findAndPopulate( reg, insertOuts );
   if ( !ret.isEmpty() ) insertOuts.push_back( ret );

   for ( RegionTree<NewDirectoryEntryData>::iterator_list_t::iterator it = insertOuts.begin();
         it != insertOuts.end();
         it++
   ) {
      RegionTree<NewDirectoryEntryData>::iterator &accessor = *it;
      NewDirectoryEntryData &nded = *accessor;
      if ( setLoc ) {
         std::cerr << "added loc "<< memorySpaceId << std::endl;
         nded.addAccess( memorySpaceId );
      } else {
         nded.merge( ent );
      }
   }
}

void NewRegionDirectory::registerAccess( Region reg, bool input, bool output, unsigned int memorySpaceId/*, DirectoryOps &ops*/ )
{
   bool skipParent = false;

   RegionTree<NewDirectoryEntryData>::iterator_list_t outs;
   _inputDirectory.find( reg, outs );
   if ( outs.empty () ) {
      message("partial ERROR: invalid program! data spec not available.");
   } else {
      //check if the region that we are registering is fully contained in the directory, if not, there is a programming error
      RegionTree<NewDirectoryEntryData>::iterator_list_t::iterator it = outs.begin();
      RegionTree<NewDirectoryEntryData>::iterator &firstAccessor = *it;
      Region tmpReg = firstAccessor.getRegion();
      bool combiningIsGoingOk = true;

      for ( ; ( it != outs.end() ) && ( combiningIsGoingOk ); it++) {
         RegionTree<NewDirectoryEntryData>::iterator &accessor = *it;
         combiningIsGoingOk = tmpReg.combine( accessor.getRegion(), tmpReg );
      }
      if ( combiningIsGoingOk ) {
         if ( tmpReg != reg && !tmpReg.contains( reg ) ) {
            message("partial ERROR: invalid program! data spec not available. Queried region unexpected.");
         } else { message("GOT IT: on mine"); skipParent = true;}
      } else {
         message("partial ERROR: invalid program! data spec not available. Non combined parts.");
      }
   }

   //warning! outs not cleared, I don't know yet if this can be problematic

   if ( !skipParent ) {
      _parent->_inputDirectory.find( reg, outs );

      if ( outs.empty () ) {
         message("ERROR: invalid program! data spec not available.");
      } else {
         //check if the region that we are registering is fully contained in the directory, if not, there is a programming error
         RegionTree<NewDirectoryEntryData>::iterator_list_t::iterator it = outs.begin();
         RegionTree<NewDirectoryEntryData>::iterator &firstAccessor = *it;
         Region tmpReg = firstAccessor.getRegion();
         bool combiningIsGoingOk = true;

         for ( ; ( it != outs.end() ) && ( combiningIsGoingOk ); it++) {
            RegionTree<NewDirectoryEntryData>::iterator &accessor = *it;
            combiningIsGoingOk = tmpReg.combine( accessor.getRegion(), tmpReg );
         }
         if ( combiningIsGoingOk ) {
            if ( tmpReg != reg && !tmpReg.contains( reg ) ) {
               message("ERROR: invalid program! data spec not available. Queried region unexpected.");
            } else { message("GOT IT: on the parent"); }
         } else {
            message("ERROR: invalid program! data spec not available. Non combined parts.");
         }
      }
   }

   //insertRegionIntoTree( _directory, reg, memorySpaceId, true, *((NewDirectoryEntryData * ) NULL) );
   insertRegionIntoTree( _directory, reg, 1, true, *((NewDirectoryEntryData * ) NULL) );
}

void NewRegionDirectory::_internal_merge( RegionTree<NewDirectoryEntryData> const &inputDir, RegionTree<NewDirectoryEntryData> &targetDir )
{
   //merge a predecessor wd (inputDir._directory) directory into mine (this._inputDirectory)
   nanos_region_dimension_internal_t wholeMemDim[1] = { { -1ULL, 0, -1ULL } };
   Region r = build_region( DataAccess((void *) 0, true, true, false, false, 1, wholeMemDim ) );
   RegionTree<NewDirectoryEntryData>::iterator_list_t outs;

   inputDir.find( r, outs );
   for ( RegionTree<NewDirectoryEntryData>::iterator_list_t::iterator it = outs.begin();
         it != outs.end();
         it++
   ) {
      RegionTree<NewDirectoryEntryData>::iterator_list_t insertOuts;
      RegionTree<NewDirectoryEntryData>::iterator &accessor = *it;
      NewDirectoryEntryData &nded = *accessor;

      insertRegionIntoTree( targetDir, accessor.getRegion(), /* not used */ 0, false, nded );
   }
}
void NewRegionDirectory::merge( const NewRegionDirectory &inputDir )
{
   //merge a predecessor wd (inputDir._directory) directory into mine (this._inputDirectory)

   _internal_merge( inputDir._directory , _inputDirectory );
   //message("merge " << &inputDir << " to input directory " << (void *)&_inputDirectory<< " read from " << (void *)&inputDir._directory );
}

void NewRegionDirectory::mergeOutput( const NewRegionDirectory &inputDir )
{
   //merge a finished wd (inputDir._directory) directory into mine (this._directory)

   _internal_merge( inputDir._directory , _directory );
   //message("merge " << &inputDir << " to output directory " << (void *)&_directory << " read from " << (void *)&inputDir._directory );
}

void NewRegionDirectory::setRoot() {
   _root = this;
   nanos_region_dimension_internal_t wholeMemDim[1] = { { -1ULL, 0, -1ULL } };
   Region r = build_region( DataAccess((void *) 0, true, true, false, false, 1, wholeMemDim ) );
   //Region r2 = build_region( DataAccess((void *) 0, true, true, false, false, 1, wholeMemDim ) );
   insertRegionIntoTree( _inputDirectory, r, 0, true, *((NewDirectoryEntryData * ) NULL) );
   //std::cerr << "Initialized whole Mem region, input " << (void *) &_inputDirectory << std::endl;
   //std::cerr << "==== Pre init =================  " << std::endl << _directory << "================================================" << std::endl;
   //insertRegionIntoTree( _directory, r, 0, true, *((NewDirectoryEntryData * ) NULL) );
   //std::cerr << "Initialized whole Mem region, directory " << (void *) &_directory << std::endl;
   //std::cerr << "==== Post init =================  " << std::endl << _directory << "================================================" << std::endl;
}

bool NewRegionDirectory::isRoot() const {
   return ( _root == this );
}

void NewRegionDirectory::consolidate() {
   message( "consolidating directory... " << (void *) &_directory  );
 
   //std::cerr << _directory << std::endl;
   //std::cerr << _inputDirectory << std::endl;
   //message( "consolidating directory... " << (void *) &_directory  );

   _internal_merge( _directory, _inputDirectory );

   nanos_region_dimension_internal_t wholeMemDim[1] = { { -1ULL, 0, -1ULL } };
   Region r = build_region( DataAccess((void *) 0, true, true, false, false, 1, wholeMemDim ) );
   RegionTree<NewDirectoryEntryData>::iterator_list_t outs;
   _directory.find( r, outs );
   _directory.removeMany( outs );

   //std::cerr << _directory << std::endl;
   //std::cerr << _inputDirectory << std::endl;
   //message( "consolidating directory... " << (void *) &_directory  );
}

void NewRegionDirectory::print() const {
   //fprintf(stderr, "printing inputDirectory:\n");
   //_inputDirectory.print();
   //fprintf(stderr, "printing directory:\n");
   //_directory.print();
}

bool NewRegionDirectory::checkConsistency( uint64_t tag, std::size_t size, unsigned int memorySpaceId ) {
   bool ok = true;
   //MemoryMap< NewDirectoryEntryData >::ConstMemChunkList subchunkResults;
   //_inputDirectory.getChunk2( tag, size, subchunkResults );
   //for ( MemoryMap< NewDirectoryEntryData >::ConstMemChunkList::iterator it = subchunkResults.begin(); it != subchunkResults.end() && ok; it++ ) {
   //   if ( it->second != NULL ) {
   //      if ( *(it->second) != NULL ) {
   //         ok = ok && (*(it->second))->isLocatedIn( memorySpaceId );
   //      } else {
   //         ok = false;
   //      }
   //   } else {
   //      ok = false;
   //   }
   //}
   return ok;
}

Region NewRegionDirectory::build_region( DataAccess const &dataAccess ) {
   // Find out the displacement due to the lower bounds and correct it in the address
   size_t base = 1UL;
   size_t displacement = 0L;
   for (short dimension = 0; dimension < dataAccess.dimension_count; dimension++) {
      nanos_region_dimension_internal_t const &dimensionData = dataAccess.dimensions[dimension];
      displacement = displacement + dimensionData.lower_bound * base;
      base = base * dimensionData.size;
   }
   size_t address = (size_t)dataAccess.address + displacement;

   // Build the Region

   // First dimension is base 1
   size_t additionalContribution = 0UL; // Contribution of the previous dimensions (necessary due to alignment issues)
   //std::cerr << "build region 0 len is " << dataAccess.dimensions[0].accessed_length << std::endl;
   Region region = RegionBuilder::build(address, 1UL, dataAccess.dimensions[0].accessed_length, additionalContribution);

   // Add the bits corresponding to the rest of the dimensions (base the previous one)
   base = 1 * dataAccess.dimensions[0].size;
   for (short dimension = 1; dimension < dataAccess.dimension_count; dimension++) {
      nanos_region_dimension_internal_t const &dimensionData = dataAccess.dimensions[dimension];

      //std::cerr << "build region " << dimension << " len is " << dimensionData.accessed_length << std::endl;
      region |= RegionBuilder::build(address, base, dimensionData.accessed_length, additionalContribution);
      base = base * dimensionData.size;
   }
   //std::cerr << "end build region n" << std::endl;

   return region;
}


std::ostream & nanos::operator<< (std::ostream &o, nanos::NewDirectoryEntryData const &ent)
{
   o << "WL: " << ent._writeLocation << " V: " << ent._version << " Locs: ";
   for ( std::set<int>::iterator it = ent._location.begin(); it != ent._location.end(); it++ ) {
      o << *it << " ";
   }
   return o;
}

