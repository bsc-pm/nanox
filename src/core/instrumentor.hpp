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
#ifndef __NANOS_INSTRUMENTOR_H
#define __NANOS_INSTRUMENTOR_H

#include "instrumentor_decl.hpp"

using namespace nanos;

nanos_event_type_t Instrumentor::Event::getType () const { return _type; }

nanos_event_state_value_t Instrumentor::Event::getState () { return _state; }

unsigned int Instrumentor::Event::getNumKVs () const { return _nkvs; }
Instrumentor::Event::ConstKVList Instrumentor::Event::getKVs () const { return _kvList; }

unsigned int Instrumentor::Event::getDomain ( void ) const { return _ptpDomain; }
unsigned int Instrumentor::Event::getId( void ) const { return _ptpId; }

void Instrumentor::Event::reverseType ( )
{
   switch ( _type )
   {
      case PTP_START: _type = PTP_END; break;
      case PTP_END: _type = PTP_START; break;
      case BURST_START: _type = BURST_END; break;
      case BURST_END: _type = BURST_START; break;
      default: break;
   }
}

#endif
