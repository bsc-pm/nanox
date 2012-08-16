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

         void NewDirectoryEntryData::addAccess( int id, uint64_t address, unsigned int version ) {
            if ( version > _version ) {
               _location.clear();
               setWriteLocation( id );
            } else {
               // XXX notify an addition
               //std::cerr << " ++++++++++++++++++++++ MMMH " << _listeners.size() << std::endl;
               for ( std::set< WD * >::iterator it = _listeners.begin(); it != _listeners.end(); it++ ) {
                  std::cerr << "I have to notify wd " << (*it)->getId() << " that i have accessed this "<< std::endl;
               }
            }
            _version = version;
            _location.insert( LocationEntry( id, address ) );
         }
NewRegionDirectory::NewRegionDirectory() : _directory(), _parent(NULL), _mergeLock(), _outputMergeLock() {}

NewRegionDirectory *NewRegionDirectory::_root = NULL;

void NewRegionDirectory::setParent( NewRegionDirectory *parent )
{
   _parent =  parent;
}

void NewRegionDirectory::insertRegionIntoTree( RegionTree<NewDirectoryEntryData> &dir, Region const &reg, unsigned int memorySpaceId, bool setLoc, NewDirectoryEntryData const &ent, unsigned int version )
{
   //if(sys.getNetwork()->getNodeNum() == 0)std::cerr << "insert into " << &dir << std::endl;
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
         //if(sys.getNetwork()->getNodeNum() == 0 && reg.getMask() == 0xfffffffffe000000 ) std::cerr << "added loc "<< memorySpaceId << " version " << version << " to dir " << &dir << std::endl;
         nded.addAccess( memorySpaceId, 0, version );
         //nded.setVersion( version );
      } else {
         //if(sys.getNetwork()->getNodeNum() == 0 && reg.getMask() == 0xfffffffffe000000 )if ( nded.getVersion() == version ) { std::cerr << " have to merge from entry locs, addr " << (void *) reg.getFirstValue() << " to dir "<< &dir << std::endl; }
         nded.merge( ent );
      }
   }
}

void NewRegionDirectory::masterGetLocation( Region const &reg, LocationInfoList &loc, unsigned int &version )
{
   version = 0;

 //if (sys.getNetwork()->getNodeNum() == 0)if(myThread->getId() == 0 ) std::cerr << " register Loc " << memorySpaceId << " " << reg << std::endl;
 //if (sys.getNetwork()->getNodeNum() == 0)if(myThread->getId() == 0 ) { std::cerr << "this dir is " << &_inputDirectory << " parent is "  << ( ( _parent!= NULL) ? &(_parent->_directory) : NULL ) << std::endl;   }

   RegionTree<NewDirectoryEntryData>::iterator_list_t outs;
   RegionTree<NewDirectoryEntryData>::iterator_list_t outsParent;
   //_mergeLock.acquire();
   _directory.find( reg, outs );
   if ( outs.empty () ) {
      //message("partial ERROR: invalid program! data spec not available.");
      //if (sys.getNetwork()->getNodeNum() == 0)if(myThread->getId() == 0 )std::cerr << "+++++++++++++++++++++++++ "<< reg << " NO OUTS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! "<<std::endl;
   } else {
      //check if the region that we are registering is fully contained in the directory, if not, there is a programming error
      RegionTree<NewDirectoryEntryData>::iterator_list_t::iterator it = outs.begin();
      RegionTree<NewDirectoryEntryData>::iterator &firstAccessor = *it;
      Region tmpReg = firstAccessor.getRegion();
      bool combiningIsGoingOk = true;
         //{
         //   if (sys.getNetwork()->getNodeNum() == 0)if(myThread->getId() == 0 )std::cerr << "++++++++++++++++++++++++++ SYATY COMBINE ++++++++++++++) "<< &(_parent->_directory) << std::endl;
         //   if (sys.getNetwork()->getNodeNum() == 0)if(myThread->getId() == 0 )std::cerr << "+++++++++++++++++++++++++ "<< reg << std::endl;
         //   RegionTree<NewDirectoryEntryData>::iterator_list_t::iterator qrit = outs.begin();
         //   if (sys.getNetwork()->getNodeNum() == 0)if(myThread->getId() == 0 )std::cerr << "::::::::::::::: QUEry results "<< &(_parent->_directory) << std::endl;
         //   for ( ; qrit != outs.end(); qrit++ ) {
         //   RegionTree<NewDirectoryEntryData>::iterator &accessor = *qrit;
         //   if (sys.getNetwork()->getNodeNum() == 0)if(myThread->getId() == 0 )std::cerr << " ::out reg " << accessor.getRegion() << " <> " << *accessor << std::endl;
         //   
         //   }
         //   if (sys.getNetwork()->getNodeNum() == 0)if(myThread->getId() == 0 )std::cerr << "::::::::::::::: QUEry results END "<< &(_parent->_directory) << std::endl;
         //}

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
//if (sys.getNetwork()->getNodeNum() == 0)if(myThread->getId() == 0 )  std::cerr << "Reg " << reg << " registered for access with version " << (output ? ( version + 1 ) : version ) << " in location " << memorySpaceId << std::endl;
//if (sys.getNetwork()->getNodeNum() == 1) std::cerr << "Reg " << reg << " registered for access with version " << (output ? ( version + 1 ) : version ) << " in location " << memorySpaceId << std::endl;
   //insertRegionIntoTree( _directory, reg, memorySpaceId, devAddr, true, *((NewDirectoryEntryData * ) NULL), output ? ( version + 1 ) : version );
   //_mergeLock.release();
}

void NewRegionDirectory::getLocation( Region const &reg, LocationInfoList &loc, unsigned int &version, WD *wd )
{
   bool skipParent = false;
   version = 0;

   //bool track = false;
   //{
   //  unsigned int i;
   //  for ( i = 0; i < wd->getNumCopies() && !track; i++ ) {
   //    track = ( !wd->getCopies()[i].isOutput() && wd->getCopies()[i].isInput() && reg.getFirstValue() == (uint64_t) wd->getCopies()[i].getBaseAddress() );
   //  }
   //}
 //if (sys.getNetwork()->getNodeNum() == 0)if(myThread->getId() == 0 ) std::cerr << " register Loc " << memorySpaceId << " " << reg << std::endl;
 //if (sys.getNetwork()->getNodeNum() == 0)if(myThread->getId() == 0 ) { std::cerr << "this dir is " << &_inputDirectory << " parent is "  << ( ( _parent!= NULL) ? &(_parent->_directory) : NULL ) << std::endl;   }

   RegionTree<NewDirectoryEntryData>::iterator_list_t outs;
   RegionTree<NewDirectoryEntryData>::iterator_list_t outsParent;
      //if ( track) std::cerr <<" track: check input "<< wd->getId() << std::endl;
   _inputDirectory.find( reg, outs );
   if ( outs.empty () ) {
      //std::cerr <<"empty input dir " << wd->getId() << std::endl;
      //message("partial ERROR: invalid program! data spec not available.");
      //if (sys.getNetwork()->getNodeNum() == 0)if(myThread->getId() == 0 )std::cerr << "+++++++++++++++++++++++++ "<< reg << " NO OUTS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! "<<std::endl;
   } else {
      //check if the region that we are registering is fully contained in the directory, if not, there is a programming error
      RegionTree<NewDirectoryEntryData>::iterator_list_t::iterator it = outs.begin();
      RegionTree<NewDirectoryEntryData>::iterator &firstAccessor = *it;
      Region tmpReg = firstAccessor.getRegion();
      bool combiningIsGoingOk = true;
         //{
         //   if (sys.getNetwork()->getNodeNum() == 0)if(myThread->getId() == 0 )std::cerr << "++++++++++++++++++++++++++ SYATY COMBINE ++++++++++++++) "<< &(_parent->_directory) << std::endl;
         //   if (sys.getNetwork()->getNodeNum() == 0)if(myThread->getId() == 0 )std::cerr << "+++++++++++++++++++++++++ "<< reg << std::endl;
         //   RegionTree<NewDirectoryEntryData>::iterator_list_t::iterator qrit = outs.begin();
         //   if (sys.getNetwork()->getNodeNum() == 0)if(myThread->getId() == 0 )std::cerr << "::::::::::::::: QUEry results "<< &(_parent->_directory) << std::endl;
         //   for ( ; qrit != outs.end(); qrit++ ) {
         //   RegionTree<NewDirectoryEntryData>::iterator &accessor = *qrit;
         //   if (sys.getNetwork()->getNodeNum() == 0)if(myThread->getId() == 0 )std::cerr << " ::out reg " << accessor.getRegion() << " <> " << *accessor << std::endl;
         //   
         //   }
         //   if (sys.getNetwork()->getNodeNum() == 0)if(myThread->getId() == 0 )std::cerr << "::::::::::::::: QUEry results END "<< &(_parent->_directory) << std::endl;
         //}

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
      //if ( track) std::cerr <<" track: check parent " << wd->getId() << std::endl;
      _parent->_directory.find( reg, outsParent );
      RegionTree<NewDirectoryEntryData>::iterator_list_t::iterator mergeIt;
      //std::cerr << "Entries from parent: " << outsParent.size() << std::endl;
      for ( mergeIt = outsParent.begin(); mergeIt != outsParent.end(); mergeIt++ )
         outs.push_back( *mergeIt ); //merge results from parent.

      //std::cerr << "Entries in outs: " << outs.size() << std::endl;
      if ( outs.empty () ) {
         message("ERROR: invalid program! data spec not available at node " << sys.getNetwork()->getNodeNum());
         //if (sys.getNetwork()->getNodeNum() == 0)if(myThread->getId() == 0 )sys.printBt();
         //std::cerr << "!!!!!!!!!!!!!!!!!!!!!  this dir is " << &(_parent->_directory) << " " << (_parent->_directory) << std::endl;
      } else {
         //check if the region that we are registering is fully contained in the directory, if not, there is a programming error
#if 0
         {
            RegionTree<NewDirectoryEntryData>::iterator_list_t::iterator it = outs.begin();
            if(myThread->getId() == 0 )std::cerr << "::::::::::::::: QUEry results "<< &(_parent->_directory) << std::endl;
            for ( ; it != outs.end(); it++ ) {
            RegionTree<NewDirectoryEntryData>::iterator &accessor = *it;
            if(myThread->getId() == 0 )std::cerr << " ::out reg " << accessor.getRegion() << " <> " << *accessor << std::endl;
            
            }
            if(myThread->getId() == 0 )std::cerr << "::::::::::::::: QUEry results END "<< &(_parent->_directory) << std::endl;
         }
         RegionTree<NewDirectoryEntryData>::iterator_list_t::iterator it = outs.begin();
         RegionTree<NewDirectoryEntryData>::iterator &firstAccessor = *it;
         Region tmpReg = firstAccessor.getRegion().intersect( reg );
         bool combiningIsGoingOk = true;

            if(myThread->getId() == 0 )std::cerr << "++++++++++++++++++++++++++ SYATY COMBINE ++++++++++++++) "<< &(_parent->_directory) << std::endl;
            if(myThread->getId() == 0 )std::cerr << "+++++++++++++++++++++++++ "<< reg << std::endl;
         for ( ; ( it != outs.end() ) && ( combiningIsGoingOk ); it++) {
            RegionTree<NewDirectoryEntryData>::iterator &accessor = *it;
            if(myThread->getId() == 0 )std::cerr << " Region to combine is " << accessor.getRegion() << std::endl;
            if(myThread->getId() == 0 )std::cerr << " TMP reg " << tmpReg << std::endl;
            combiningIsGoingOk = tmpReg.combine( accessor.getRegion().intersect( reg ), tmpReg );
            //if(myThread->getId() == 0 )std::cerr << " Result " << combiningIsGoingOk << std::endl;
         }
         if ( combiningIsGoingOk ) {
            if ( tmpReg != reg && !tmpReg.contains( reg ) ) {
               message("ERROR: invalid program! data spec not available. Queried region unexpected.");
            } else { /*message("GOT IT: on the parent");*/ }
         } else {
            //if (sys.getNetwork()->getNodeNum() == 0) sys.printBt(); std::cerr << " LOL " << std::endl;
            message("ERROR: invalid program! data spec not available. Non combined parts.");
         }
           // if(myThread->getId() == 0 )std::cerr << "++++++++++++++++++++++++++ END SYATY COMBINE ++++++++++++++)" << std::endl;
#endif

           //outs contains the set of regions which contain the desired region
         RegionTree<NewDirectoryEntryData>::iterator_list_t::iterator it = outs.begin();
         for ( it = outs.begin() ; it != outs.end(); it++ ) {
            RegionTree<NewDirectoryEntryData>::iterator &accessor = *it;
            NewDirectoryEntryData &nded = *accessor;
            //std::cerr << "Register dir, ret loc " << nded;
      //if ( track) std::cerr <<" track: found parent " << wd->getId()<< std::endl;
            version = std::max( version, nded.getVersion() );
            //std::cerr << "[parent used] WD " << wd->getId() <<" add region FV " << (void *)  accessor.getRegion().getFirstValue() << nded << std::endl;
            loc.push_back( std::make_pair( accessor.getRegion().intersect( reg ) , nded ) );
            //nded.addListener( wd );
   insertRegionIntoTree( _directory, accessor.getRegion(), /* not used */ 0, false, nded, /* not really used */ version );
         }
      }
      _parent->_mergeLock.release();
   } else {
      //outs contains the set of regions which contain the desired region
      RegionTree<NewDirectoryEntryData>::iterator_list_t::iterator it = outs.begin();
      for ( it = outs.begin() ; it != outs.end(); it++ ) {
         RegionTree<NewDirectoryEntryData>::iterator &accessor = *it;
         NewDirectoryEntryData &nded = *accessor;
      //if ( track) std::cerr <<" track: found input mine "<< wd->getId() << std::endl;
         //std::cerr << "Register dir, ret loc " << nded;
         version = std::max( version, nded.getVersion() );
            //std::cerr << "[parent not used] WD " << wd->getId() <<" add region FV " << (void *) accessor.getRegion().getFirstValue() << std::endl;
         loc.push_back( std::make_pair( accessor.getRegion().intersect( reg ) , nded ) );
         //nded.addListener( wd );
      }
   }

//if (sys.getNetwork()->getNodeNum() == 0)if(myThread->getId() == 0 )  std::cerr << "Reg " << reg << " registered for access with version " << (output ? ( version + 1 ) : version ) << " in location " << memorySpaceId << std::endl;
//if (sys.getNetwork()->getNodeNum() == 1) std::cerr << "Reg " << reg << " registered for access with version " << (output ? ( version + 1 ) : version ) << " in location " << memorySpaceId << std::endl;
   //insertRegionIntoTree( _directory, reg, memorySpaceId, devAddr, true, *((NewDirectoryEntryData * ) NULL), output ? ( version + 1 ) : version );
}


void NewRegionDirectory::addAccess(Region const &reg, unsigned int memorySpaceId, unsigned int version )
{
   insertRegionIntoTree( _directory, reg, memorySpaceId, true, *((NewDirectoryEntryData * ) NULL), version );
}

void NewRegionDirectory::_internal_merge( RegionTree<NewDirectoryEntryData> const &inputDir, RegionTree<NewDirectoryEntryData> &targetDir, bool print )
{
   //double time1, time2, time3, time4;
   //merge a predecessor wd (inputDir._directory) directory into mine (this._inputDirectory)

   //time1 = OS::getMonotonicTime();

   nanos_region_dimension_internal_t wholeMemDim[1] = { { -1ULL, 0, -1ULL } };
   Region r = build_region( DataAccess((void *) 0, true, true, false, false, 1, wholeMemDim ) );
   RegionTree<NewDirectoryEntryData>::iterator_list_t outs;

   //time2 = OS::getMonotonicTime();

  //if(sys.getNetwork()->getNodeNum() > 0 ) std::cerr << "_internal_merge !!! " << std::endl;
   inputDir.find( r, outs );
   //time3 = OS::getMonotonicTime();
   int n = 0;
   for ( RegionTree<NewDirectoryEntryData>::iterator_list_t::iterator it = outs.begin();
         it != outs.end();
         it++
   ) {
      RegionTree<NewDirectoryEntryData>::iterator_list_t insertOuts;
      RegionTree<NewDirectoryEntryData>::iterator &accessor = *it;
      NewDirectoryEntryData &nded = *accessor;
      n++;

      //if(sys.getNetwork()->getNodeNum() == 0  && accessor.getRegion().getMask() == 0xfffffffffe000000 )std::cerr << "::::::merge into " << &targetDir <<", insert region "<<  accessor.getRegion() << " locinfo: " << nded << std::endl;
      insertRegionIntoTree( targetDir, accessor.getRegion(), /* not used */ 0, false, nded, 0 );
   }
      //if(sys.getNetwork()->getNodeNum() > 0 )std::cerr << "::::::merge, num of regions inserted: "<< n << std::endl;
   //time4 = OS::getMonotonicTime();
   //if ( sys.getNetwork()->getNodeNum() == 0 && print ) {
   //fprintf(stderr, "---------------------\n");
   //fprintf(stderr, "Init time: %f\n", time2-time1);
   //fprintf(stderr, "Find time: %f\n", time3-time2);
   //fprintf(stderr, "Merge time: %f - %d items\n", time4-time3, n);
   //fprintf(stderr, "---------------------\n");
   //}
  // std::cerr << "_internal_merge end !!! " << std::endl;
}
void NewRegionDirectory::merge( const NewRegionDirectory &inputDir )
{
   //merge a predecessor wd (inputDir._directory) directory into mine (this._inputDirectory)

   //std::cerr << "merge Output Dir into my input (succsessor to me)"<< std::endl;
   _mergeLock.acquire();
   //if(sys.getNetwork()->getNodeNum() == 0)std::cerr << "merge " << &inputDir<< " into " << this << " inD "<< (void *) &_inputDirectory << std::endl;
   _internal_merge( inputDir._directory , _inputDirectory );
   //if(sys.getNetwork()->getNodeNum() == 0)std::cerr << "merge Output Dir into my input (succsessor to me) COMPLETE "<< std::endl;
   _mergeLock.release();
   //message("DOME merge " << &inputDir << " to input directory " << (void *)&_inputDirectory<< " read from " << (void *)&inputDir._directory );
}

void NewRegionDirectory::mergeOutput( const NewRegionDirectory &inputDir )
{
   //merge a finished wd (inputDir._directory) directory into mine (this._directory)
   
   _mergeLock.acquire();

   //std::cerr << "merge Output Dir into my output (child finished) "<< std::endl;
   //if(sys.getNetwork()->getNodeNum() == 0)std::cerr << "mergeOut " <<  &inputDir._directory << " into " << &_inputDirectory << std::endl;
   //if(sys.getNetwork()->getNodeNum() == 0) std::cerr <<_inputDirectory << std::endl;
   //_internal_merge( inputDir._directory , _inputDirectory );
   _internal_merge( inputDir._directory , _directory, true );
   //if(sys.getNetwork()->getNodeNum() == 0)std::cerr << "mergeOut " <<  &inputDir._directory << " into " << &_inputDirectory << std::endl;
   //if(sys.getNetwork()->getNodeNum() == 0)std::cerr << "after merge " <<  &inputDir._directory << " into " << &_inputDirectory << std::endl;
   //if(sys.getNetwork()->getNodeNum() == 0) std::cerr  << _inputDirectory << std::endl;
   _mergeLock.release();
   //message("merge " << &inputDir << " to output directory " << (void *)&_directory << " read from " << (void *)&inputDir._directory );
}

void NewRegionDirectory::setRoot() {
   _root = this;
   nanos_region_dimension_internal_t wholeMemDim[1] = { { -1ULL, 0, -1ULL } };
   Region r = build_region( DataAccess((void *) 0, true, true, false, false, 1, wholeMemDim ) );
   //Region r2 = build_region( DataAccess((void *) 0, true, true, false, false, 1, wholeMemDim ) );
   insertRegionIntoTree( _inputDirectory, r, 0, true, *((NewDirectoryEntryData * ) NULL), 1 );
   //std::cerr << "Initialized whole Mem region, input " << (void *) &_inputDirectory << std::endl;
   //std::cerr << "==== Pre init =================  " << std::endl << _directory << "================================================" << std::endl;
   insertRegionIntoTree( _directory, r, 0, true, *((NewDirectoryEntryData * ) NULL), 1 );
   //std::cerr << "Initialized whole Mem region, directory " << (void *) &_directory << std::endl;
   //std::cerr << "==== Post init =================  " << std::endl << _directory << "================================================" << std::endl;
}

bool NewRegionDirectory::isRoot() const {
   return ( _root == this );
}

void NewRegionDirectory::consolidate( bool flushData ) {
   //message( "consolidating directory... " << (void *) &_directory  );
 
   //std::cerr << _directory << std::endl;
   //std::cerr << _inputDirectory << std::endl;
   //message( "consolidating directory... " << (void *) &_directory  );

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
      //std::cerr <<"Consolidate dir... "<< std::endl;
      _directory.find( r, outs );
      //for (RegionTree<NewDirectoryEntryData>::iterator_list_t::iterator it = outs.begin(); it != outs.end(); it++) {
      //   RegionTree<NewDirectoryEntryData>::iterator &accessor = *it;
      //   Region const &reg = accessor.getRegion();
      //   //std::cerr << (void *) reg.getFirstValue() << " ";
      //}
      //std::cerr << std::endl;
      for (RegionTree<NewDirectoryEntryData>::iterator_list_t::iterator it = outs.begin(); it != outs.end(); it++) {
         RegionTree<NewDirectoryEntryData>::iterator &accessor = *it;
         NewDirectoryEntryData &nded = *accessor;
         Region const &reg = accessor.getRegion();
         if ( !nded.isLocatedIn( 0 ) ) {
            int loc = nded.getFirstLocation();
            //std::cerr << " I have to sync " << reg << " from " << loc << std::endl;
            sys.getCaches()[ loc ]->syncRegion( reg /*, nded.getAddressOfLocation( loc )*/ );
            nded.addAccess( 0, reg.getFirstValue(), nded.getVersion() );
         } /*else {
            std::cerr << " I DO NOT have to sync " << reg << std::endl;
         }*/
      }
   }
   
   //std::cerr << _directory << std::endl;
   //std::cerr << _inputDirectory << std::endl;
   //message( "consolidating directory... done " << (void *) &_directory  );
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

void NewRegionDirectory::lock() { _mergeLock.acquire(); }
void NewRegionDirectory::unlock() { _mergeLock.release(); }

std::ostream & nanos::operator<< (std::ostream &o, nanos::NewDirectoryEntryData const &ent)
{
   o << "WL: " << ent._writeLocation << " V: " << ent._version << " Locs: ";
   for ( std::set<nanos::NewDirectoryEntryData::LocationEntry>::iterator it = ent._location.begin(); it != ent._location.end(); it++ ) {
      o << it->getMemorySpaceId() << ":" << (void *)it->getAddress() << " ";
   }
   return o;
}

