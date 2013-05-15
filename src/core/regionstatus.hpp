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

#ifndef _NANOS_REGION_STATUS_H
#define _NANOS_REGION_STATUS_H
#include <stdlib.h>
#include <list>
#include <algorithm>
#include "regionstatus_decl.hpp"
#include "dependableobject.hpp"
#include "dependenciesdomain.hpp"
#include "atomic.hpp"

using namespace nanos;

inline const RegionStatus & RegionStatus::operator= ( const RegionStatus &obj )
{
   _lastWriter = obj._lastWriter;
   return *this;
}

inline bool RegionStatus::hasLastWriter ( )
{
   return _lastWriter != NULL;
}

inline DependableObject* RegionStatus::getLastWriter ( ) const
{
   return _lastWriter;
}

inline void RegionStatus::setLastWriter ( DependableObject &depObj )
{
   SyncLockBlock lock( _writerLock );
   _lastWriter = &depObj;
}

inline void RegionStatus::deleteLastWriter ( DependableObject &depObj )
{
   if ( _lastWriter == &depObj ) {
      SyncLockBlock lock( _writerLock );
      if ( _lastWriter ==  &depObj ) {
         _lastWriter = NULL;
      }
   }
}

inline RegionStatus::DependableObjectList const & RegionStatus::getReaders ( ) const
{
   return _versionReaders;
}

inline RegionStatus::DependableObjectList & RegionStatus::getReaders ( )
{
   return _versionReaders;
}

inline void RegionStatus::setReader ( DependableObject &reader )
{
   _versionReaders.push_back( &reader );
}

inline bool RegionStatus::hasReader ( DependableObject &depObj )
{
   return ( find( _versionReaders.begin(), _versionReaders.end(), &depObj ) != _versionReaders.end() );
}

inline void RegionStatus::flushReaders ( )
{
   _versionReaders.clear();
}

inline void RegionStatus::deleteReader ( DependableObject &reader )
{
   _versionReaders.remove( &reader );
}

inline bool RegionStatus::hasReaders ()
{
   return !( _versionReaders.empty() );
}

inline Lock& RegionStatus::getReadersLock()
{
   return _readersLock;
}

inline CommutationDO* RegionStatus::getCommDO() const
{
   return _commDO;
}

inline void RegionStatus::setCommDO( CommutationDO *commDO )
{
   _commDO = commDO;
}

inline bool RegionStatus::isEmpty ()
{
   return ( _lastWriter == 0 ) && _versionReaders.empty() && ( _commDO == 0 );
}

inline bool RegionStatus::isOnHold () const
{
   return _hold;
}

inline void RegionStatus::hold ()
{
   _hold = true;
}

inline void RegionStatus::unhold ()
{
   _hold = false;
}

inline std::ostream & nanos::operator<<( std::ostream &o, nanos::RegionStatus const &regionStatus)
{
   //regionStatus._writerLock.lock();
   //regionStatus._readersLock.lock();
   o << "{"
      << "LastWriter: ";
   DependableObject *writer = regionStatus.getLastWriter();
   if ( writer ) {
      DependenciesDomain *domain = writer->getDependenciesDomain();
      if ( domain ) {
         o << " " << domain->getId() << "_" << writer->getId();
      } else {
         o << writer->getDescription();
      }
   } else {
      o << "-";
   }
   o << "|Readers:";
   
   RegionStatus::DependableObjectList const &readers = regionStatus.getReaders();
   for (RegionStatus::DependableObjectList::const_iterator it = readers.begin(); it != readers.end(); it++) {
      DependableObject *reader = *it;
      
      DependenciesDomain *domain = reader->getDependenciesDomain();
      if ( domain ) {
         o << " " << domain->getId() << "_" << reader->getId();
      } else {
         o << " " << reader->getDescription();
      }
   }
   
   o << "|CommutationDO: ";
   DependableObject *commutationDO = regionStatus.getCommDO();
   if ( commutationDO ) {
      DependenciesDomain *domain = commutationDO->getDependenciesDomain();
      if ( domain ) {
         o << " " << domain->getId() << "_" << commutationDO->getId();
      } else {
         o << commutationDO->getDescription();
      }
   } else {
      o << "-";
   }
   o << "}";
   
   //regionStatus._readersLock.unlock();
   //regionStatus._writerLock.unlock();
   
   return o;
}

#endif
