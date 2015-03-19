/*************************************************************************************/
/*      Copyright 2015 Barcelona Supercomputing Center                               */
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

#include <stdio.h>

#include "memcachecopy_decl.hpp"
#include "system_decl.hpp"
#include "memoryops_decl.hpp"
#include "deviceops.hpp"
#include "workdescriptor.hpp"
#include "basethread.hpp"

MemCacheCopy::MemCacheCopy() : 
   _version( 0 ), _childrenProducedVersion( 0 )
   , _reg( 0, (reg_key_t) NULL )
   , _locations()
   , _locationDataReady( false )
   , _chunk( NULL )
   , _policy( sys.getRegionCachePolicy() )
{
}

MemCacheCopy::MemCacheCopy( WD const &wd, unsigned int index/*, MemController &ccontrol*/ ) {
   sys.getHostMemory().getRegionId( wd.getCopies()[ index ], _reg, wd, index );
}

void MemCacheCopy::getVersionInfo() {
   if ( _version == 0 ) {
      unsigned int ver = 0;
      sys.getHostMemory().getVersionInfo( _reg, ver, _locations );
      setVersion( ver );
      _locationDataReady = true;
   }
}

void MemCacheCopy::generateInOps( BaseAddressSpaceInOps &ops, bool input, bool output, WD const &wd, unsigned int copyIdx ) {
   //NANOS_INSTRUMENT( InstrumentState inst4(NANOS_CC_CDIN_OP_GEN); );
   if ( input && output ) {
      //re read version, in case of this being a commutative or concurrent access
      if ( _reg.getVersion() > _version ) {
         *myThread->_file << "[!!!] WARNING: concurrent or commutative detected, wd " << wd.getId() << " " << (wd.getDescription()!=NULL?wd.getDescription():"[no desc]") << " index " << copyIdx << " _reg.getVersion() " << _reg.getVersion() << " _version " << _version << std::endl;
         _version = _reg.getVersion();
      }
   }
   if ( input ) {
      ops.copyInputData( *this, wd, copyIdx );
   } else if ( output ) {
      ops.allocateOutputMemory( _reg, _version, wd, copyIdx );
   } else {
      fprintf(stderr, "Error at %s.\n", __FUNCTION__);
   }
   //NANOS_INSTRUMENT( inst4.close(); );
}

void MemCacheCopy::generateOutOps( SeparateMemoryAddressSpace *from, SeparateAddressSpaceOutOps &ops, bool input, bool output, WD const &wd, unsigned int copyIdx ) {
   //std::cerr << __FUNCTION__ << std::endl;
   ops.copyOutputData( from, *this, output, wd, copyIdx );
}

unsigned int MemCacheCopy::getVersion() const {
   return _version;
}

void MemCacheCopy::setVersion( unsigned int version ) {
   _version = version;
}

bool MemCacheCopy::isRooted( memory_space_id_t &loc ) const {
   //bool result;
   //bool result2;
   bool result3;

   global_reg_t whole_obj( 1, _reg.key );
   result3 = whole_obj.isRooted();
   if ( result3 ) {
      loc = whole_obj.getFirstLocation();
   }

    //if ( _locations.size() > 0 ) {
    //   global_reg_t refReg( _locations.begin()->second, _reg.key );
    //   result = true;
    //   result2 = true;
    //   memory_space_id_t refloc = refReg.getFirstLocation();
    //   for ( NewLocationInfoList::const_iterator it = _locations.begin(); it != _locations.end() && result; it++ ) {
    //      global_reg_t thisReg( it->second, _reg.key );
    //      result = ( thisReg.isRooted() && thisReg.getFirstLocation() == refloc );
    //      result2 = ( thisReg.isRooted() && thisReg.getRootedLocation() == refloc );
    //   }
    //   if ( result ) loc = refloc;
    //} else {
    //   result = _reg.isRooted();
    //   if ( result ) {
    //      loc = _reg.getFirstLocation();
    //   }
    //}
   //if ( sys.getNetwork()->getNodeNum() == 0 ) {
   //   std::cerr << "whole object "; whole_obj.key->printRegion(std::cerr, whole_obj.id);
   //   std::cerr << std::endl << "real reg "; _reg.key->printRegion(std::cerr, _reg.id);
   //   std::cerr << std::endl << " copy root check result: " << result << " result2: " << result2 << " result3: " << result3 << std::endl;
   //}
   return result3;
}

void MemCacheCopy::printLocations( std::ostream &o ) const {
   for ( NewLocationInfoList::const_iterator it = _locations.begin(); it != _locations.end(); it++ ) {
      NewNewDirectoryEntryData *d = NewNewRegionDirectory::getDirectoryEntry( *(_reg.key), it->second );
      o << "   [ " << it->first << "," << it->second << " ] "; _reg.key->printRegion( o, it->first ); 
      if ( d ) o << " " << *d << std::endl; 
      else o << " dir entry n/a" << std::endl;
   }
}

void MemCacheCopy::setChildrenProducedVersion( unsigned int version ) {
   _childrenProducedVersion = version;
}

unsigned int MemCacheCopy::getChildrenProducedVersion() const {
   return _childrenProducedVersion;
}
