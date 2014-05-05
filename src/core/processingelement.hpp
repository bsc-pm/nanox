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

#ifndef _NANOS_PROCESSING_ELEMENT
#define _NANOS_PROCESSING_ELEMENT

#include <string.h>
#include "functors.hpp"
#include "processingelement_decl.hpp"
#include "workdescriptor_decl.hpp"

using namespace nanos;

inline ProcessingElement::~ProcessingElement()
{
   std::for_each(_threads.begin(),_threads.end(),deleter<BaseThread>);
}

inline int ProcessingElement::getId() const
{
   return _id;
}

//inline void ProcessingElement::setId( int id )
//{
//  _id = id;
//}

//inline int ProcessingElement::getUId() const
//{
//   return _uid;
//}
//
//inline void ProcessingElement::setUId( int uid )
//{
//  _uid = uid;
//}

inline const Device * ProcessingElement::getDeviceType () const
{
   return _device;
}

inline const Device * ProcessingElement::getSubDeviceType () const
{
   return _subDevice;
}
 
inline int ProcessingElement::getNUMANode() const{ return _numaNode; }

inline void ProcessingElement::setNUMANode( int node ){ _numaNode = node; }

inline std::size_t ProcessingElement::getNumThreads() const { return _threads.size(); }

#endif

