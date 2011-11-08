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

#ifndef _NANOS_SLICER
#define _NANOS_SLICER

#include "workdescriptor.hpp"
#include "schedule.hpp"
#include "nanos-int.h"
#include "slicer_decl.hpp"
#include <list>                                                                                                                                          

using namespace nanos;

inline void *Slicer::getSpecificData ( ) const
{
   return NULL;
}

inline Slicer * SlicedWD::getSlicer ( void ) const
{
   return &_slicer;
}

inline void SlicedWD::submit ()
{
   if ( _isSliceable ) _slicer.submit(*this);
   else WD::submit();
}

inline bool SlicedWD::dequeue ( WorkDescriptor **slice )
{
   if ( _isSliceable ) return _slicer.dequeue( this, slice );
   else return WD::dequeue (slice);
}

inline void SlicedWD::convertToRegularWD()
{
   _isSliceable=false;
}

#endif

