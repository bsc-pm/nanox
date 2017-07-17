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

#ifndef _NANOS_PROCESSING_ELEMENT
#define _NANOS_PROCESSING_ELEMENT

#include <string.h>
#include "functors.hpp"
#include "processingelement_decl.hpp"
#include "workdescriptor_decl.hpp"
#include "location.hpp"
#include "basethread.hpp"

namespace nanos {

inline ProcessingElement::~ProcessingElement()
{
   std::for_each(_threads.begin(),_threads.end(),deleter<BaseThread>);
}

inline int ProcessingElement::getId() const
{
   return _id;
}

inline ProcessingElement::ThreadList &ProcessingElement::getThreads() {
   return _threads;
}

inline std::vector<const Device*> const &ProcessingElement::getDeviceTypes () const
{
   return _devices;
}

inline void ProcessingElement::setActiveDevice(unsigned int devIdx) {
   _activeDevice = devIdx;
}

inline void ProcessingElement::setActiveDevice(const Device *dev) {
   unsigned int devIdx = 0;
   for ( ; devIdx < _devices.size() && dev != _devices[devIdx]; devIdx += 1 ) {}
   ensure( devIdx < _devices.size(), "Could not found the given device in the supported devices array");
   _activeDevice = devIdx;
}

inline unsigned int ProcessingElement::getActiveDevice() const {
   return _activeDevice;
}

inline std::size_t ProcessingElement::getNumThreads() const { return _threads.size(); }

} // namespace nanos

#endif

