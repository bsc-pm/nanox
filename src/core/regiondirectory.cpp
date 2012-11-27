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

#include "regiondirectory.hpp"
#include "atomic.hpp"
#include "os.hpp"
#include "workdescriptor_decl.hpp"

using namespace nanos;

std::ostream & nanos::operator<< (std::ostream &o, nanos::NewDirectoryEntryData const &ent)
{
   o << "WL: " << ent._writeLocation << " V: " << ent._version << " Locs: ";
   for ( std::set<int>::iterator it = ent._location.begin(); it != ent._location.end(); it++ ) {
      o << *it << " ";
   }
   return o;
}

NewRegionDirectory::NewRegionDirectory() : _directory(), _parent(NULL), _mergeLock(), _outputMergeLock() {}

NewRegionDirectory *NewRegionDirectory::_root = NULL;

void NewRegionDirectory::setParent( NewRegionDirectory *parent )
{
   _parent = parent;
}

void NewRegionDirectory::insertRegionIntoTree( RegionTree<NewDirectoryEntryData> &dir, Region const &reg, unsigned int memorySpaceId, bool setLoc, NewDirectoryEntryData const &ent, unsigned int version )
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
         nded.addAccess( memorySpaceId, 0, version );
      } else {
         nded.merge( ent );
      }
   }
}

void NewRegionDirectory::masterGetLocation( Region const &reg, LocationInfoList &loc, unsigned int &version )
{
   version = 0;

   RegionTree<NewDirectoryEntryData>::iterator_list_t outs;
   RegionTree<NewDirectoryEntryData>::iterator_list_t outsParent;
   _directory.find( reg, outs );
   if ( outs.empty () ) {
      //message("partial ERROR: invalid program! data spec not available.");
   } else {
      //check if the region that we are registering is fully contained in the directory, if not, there is a programming error
      RegionTree<NewDirectoryEntryData>::iterator_list_t::iterator it = outs.begin();
      RegionTree<NewDirectoryEntryData>::iterator &firstAccessor = *it;
      Region tmpReg = firstAccessor.getRegion();
      bool combiningIsGoingOk = true; //FIXME: combining code does not work

      for ( ; ( it != outs.end() ) && ( combiningIsGoingOk ); it++) {
         RegionTree<NewDirectoryEntryData>::iterator &accessor = *it;
         combiningIsGoingOk = tmpReg.combine( accessor.getRegion(), tmpReg );
      }
      if ( combiningIsGoingOk ) {
         if ( tmpReg != reg && !tmpReg.contains( reg ) ) {
            //message("partial ERROR: invalid program! data spec not available. Queried region unexpected.");
         } else { /* message("GOT IT: on mine");*/  /*if(myThread->getId() == 0 )std::cerr << "THIS MAY BE A PROBLEM RIGHT NOW" << std::endl;*/}
      } else {
         //message("partial ERROR: invalid program! data spec not available. Non combined parts.");
      }
   }

   //outs contains the set of regions which contain the desired region
   RegionTree<NewDirectoryEntryData>::iterator_list_t::iterator it = outs.begin();
   for ( it = outs.begin() ; it != outs.end(); it++ ) {
	   RegionTree<NewDirectoryEntryData>::iterator &accessor = *it;
	   NewDirectoryEntryData &nded = *accessor;
	   //std::cerr << "Register dir, ret loc " << nded;
	   version = std::max( version, nded.getVersion() );
	   loc.push_back( std::make_pair( accessor.getRegion().intersect( reg ) , nded ) );
   }
}

void NewRegionDirectory::getLocation( Region const &reg, LocationInfoList &loc, unsigned int &version, WD const &wd )
{
   bool skipParent = false;
   version = 0;

         //std::cerr << "getLocation WD "<< wd.getId() <<" Parent (" << ((void *) _parent) << ") " << std::endl;
         //std::cerr << "getLocation Dp "<< wd.getDepth() << " this (" << ((void *) this) << ") " << std::endl;
         //std::cerr << "Region: "; reg.printSimple( std::cerr ); std::cerr << std::endl;
         //std::cerr << "This input dir: " << _inputDirectory << std::cerr;
         //if ( _parent ) std::cerr << "This parent dir: " << _parent->_directory << std::cerr;

   RegionTree<NewDirectoryEntryData>::iterator_list_t outs;
   RegionTree<NewDirectoryEntryData>::iterator_list_t outsParent;
   _inputDirectory.find( reg, outs );
   if ( outs.empty () ) {
      //message("partial ERROR: invalid program! data spec not available.");
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
            //message("partial ERROR: invalid program! data spec not available. Queried region unexpected.");
         } else { /* message("GOT IT: on mine");*/ skipParent = true; /*if(myThread->getId() == 0 )std::cerr << "THIS MAY BE A PROBLEM RIGHT NOW" << std::endl;*/}
      } else {
         //message("partial ERROR: invalid program! data spec not available. Non combined parts.");
      }
   }

   if ( !skipParent ) {
      ensure(_parent != NULL, "directory with unset parent.");
      outs.clear();
      _parent->_mergeLock.acquire();
      _parent->_directory.find( reg, outsParent );
      RegionTree<NewDirectoryEntryData>::iterator_list_t::iterator mergeIt;
      for ( mergeIt = outsParent.begin(); mergeIt != outsParent.end(); mergeIt++ )
         outs.push_back( *mergeIt ); //merge results from parent.

      if ( outs.empty () ) {
         message("ERROR: invalid program! data spec not available at node " << sys.getNetwork()->getNodeNum() << " addr: " << (void *) reg.getFirstValue() << " dir: " << (void *) _parent  );
      } else {
         //check if the region that we are registering is fully contained in the directory, if not, there is a programming error
         //FIXME: check if returned regions match the requested ones.

           //outs contains the set of regions which contain the desired region
         RegionTree<NewDirectoryEntryData>::iterator_list_t::iterator it = outs.begin();
         for ( it = outs.begin() ; it != outs.end(); it++ ) {
            RegionTree<NewDirectoryEntryData>::iterator &accessor = *it;
            NewDirectoryEntryData &nded = *accessor;
            version = std::max( version, nded.getVersion() );
            loc.push_back( std::make_pair( accessor.getRegion().intersect( reg ) , nded ) );
            //nded.addListener( wd );
   //std::cerr << ">>>> because of getLoc   >>>>>>>>>>>>>> ADD ACCESS TO _directory REG: " << std::endl;
   // accessor.getRegion().printSimple(std::cerr);
   //std::cerr << std::endl;
   // reg.printSimple(std::cerr);
   //std::cerr << std::endl;
            insertRegionIntoTree( _directory, accessor.getRegion(), /* not used */ 0, false, nded, /* not really used */ version );
         }
      }
      //   std::cerr << "WD "<< wd.getId() <<" Parent (" << ((void *) _parent) << "): " << std::endl << _parent->_directory << std::endl;
      //   std::cerr << "Dp "<< wd.getDepth() << " this (" << ((void *) this) << "): " << std::endl << _inputDirectory << std::endl;
      _parent->_mergeLock.release();
   } else {
      //outs contains the set of regions which contain the desired region
      RegionTree<NewDirectoryEntryData>::iterator_list_t::iterator it = outs.begin();
      for ( it = outs.begin() ; it != outs.end(); it++ ) {
         RegionTree<NewDirectoryEntryData>::iterator &accessor = *it;
         NewDirectoryEntryData &nded = *accessor;
         version = std::max( version, nded.getVersion() );
         loc.push_back( std::make_pair( accessor.getRegion().intersect( reg ) , nded ) );
         //nded.addListener( wd );
      }
   }
}


void NewRegionDirectory::addAccess(Region const &reg, unsigned int memorySpaceId, unsigned int version )
{
   //std::cerr << ">>>>>>>>>>>>>>>>>> ADD ACCESS TO _directory REG: " << reg << std::endl;
   //reg.printSimple( std::cerr );
   //std::cerr << std::endl << _directory;
   insertRegionIntoTree( _directory, reg, memorySpaceId, true, *((NewDirectoryEntryData * ) NULL), version );
   //std::cerr << std::endl << _directory << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"<< std::endl;
}

void NewRegionDirectory::_internal_merge( RegionTree<NewDirectoryEntryData> const &inputDir, RegionTree<NewDirectoryEntryData> &targetDir, bool print )
{
   //merge a predecessor wd (inputDir._directory) directory into mine (this._inputDirectory)
   nanos_region_dimension_internal_t wholeMemDim[1] = { { -1ULL, 0, -1ULL } };
   Region r = build_region( DataAccess((void *) 0, true, true, false, false, 1, wholeMemDim ) );
   RegionTree<NewDirectoryEntryData>::iterator_list_t outs;

   inputDir.find( r, outs );
   int n = 0;
   for ( RegionTree<NewDirectoryEntryData>::iterator_list_t::iterator it = outs.begin();
         it != outs.end();
         it++
   ) {
      RegionTree<NewDirectoryEntryData>::iterator_list_t insertOuts;
      RegionTree<NewDirectoryEntryData>::iterator &accessor = *it;
      NewDirectoryEntryData &nded = *accessor;
      n++;
      insertRegionIntoTree( targetDir, accessor.getRegion(), /* not used */ 0, false, nded, 0 );
   }
}

void NewRegionDirectory::merge( const NewRegionDirectory &inputDir )
{
   //merge a predecessor wd (inputDir._directory) directory into mine (this._inputDirectory)
   _mergeLock.acquire();
   _internal_merge( inputDir._directory , _inputDirectory );
   _mergeLock.release();
}

void NewRegionDirectory::mergeOutput( const NewRegionDirectory &inputDir )
{
   //merge a finished wd (inputDir._directory) directory into mine (this._directory)
   _mergeLock.acquire();
   //      std::cerr << "Merge output, to be merged "<< std::endl << inputDir._directory << std::endl;
   //      std::cerr << "Merge output, current "<< std::endl << _directory << std::endl;
   _internal_merge( inputDir._directory , _directory, true );
   //      std::cerr << "Merge output, after "<< std::endl << _directory << std::endl;
   _mergeLock.release();
}

void NewRegionDirectory::setRoot() {
   _root = this;
   message("SET ROOT DIR: " << (void *) this );
   nanos_region_dimension_internal_t wholeMemDim[1] = { { -1ULL, 0, -1ULL } };
   Region r = build_region( DataAccess((void *) 0, true, true, false, false, 1, wholeMemDim ) );
   insertRegionIntoTree( _inputDirectory, r, 0, true, *((NewDirectoryEntryData * ) NULL), 1 );
   insertRegionIntoTree( _directory, r, 0, true, *((NewDirectoryEntryData * ) NULL), 1 );
}

bool NewRegionDirectory::isRoot() const {
   return ( _root == this );
}

void NewRegionDirectory::consolidate( bool flushData ) {
   // commented because now we commit to input dir when a WD finishes (to not loss info to not really made dependencies)

   //nanos_region_dimension_internal_t wholeMemDim[1] = { { -1ULL, 0, -1ULL } };
   //Region r = build_region( DataAccess((void *) 0, true, true, false, false, 1, wholeMemDim ) );
   //RegionTree<NewDirectoryEntryData>::iterator_list_t outs;
   //_inputDirectory.find( r, outs );
   //_inputDirectory.removeMany( outs );
   //   std::cerr <<"Consolidate dir... "<< std::endl;
   //_internal_merge( _directory, _inputDirectory );
   //outs.clear();
   
   if ( flushData )
   {
      nanos_region_dimension_internal_t wholeMemDim[1] = { { -1ULL, 0, -1ULL } };
      Region r = build_region( DataAccess((void *) 0, true, true, false, false, 1, wholeMemDim ) );
      RegionTree<NewDirectoryEntryData>::iterator_list_t outs;
      //_inputDirectory.find( r, outs );
      _directory.find( r, outs );
      for (RegionTree<NewDirectoryEntryData>::iterator_list_t::iterator it = outs.begin(); it != outs.end(); it++) {
         RegionTree<NewDirectoryEntryData>::iterator &accessor = *it;
         NewDirectoryEntryData &nded = *accessor;
         Region const &reg = accessor.getRegion();
         if ( !nded.isLocatedIn( 0 ) ) {
            int loc = nded.getFirstLocation();
            sys.getCaches()[ loc ]->syncRegion( reg );
            nded.addAccess( 0, reg.getFirstValue(), nded.getVersion() );
         }
      }
   }
}

void NewRegionDirectory::lock() {
   _mergeLock.acquire();
}

void NewRegionDirectory::unlock() {
   _mergeLock.release();
}

void NewRegionDirectory::invalidate( RegionTree<CachedRegionStatus> *regions ) {
  std::cerr <<"IBVAL CODE "<< std::endl;
}
