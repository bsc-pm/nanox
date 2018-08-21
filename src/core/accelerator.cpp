/*************************************************************************************/
/*      Copyright 2009-2018 Barcelona Supercomputing Center                          */
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
/*      along with NANOS++.  If not, see <https://www.gnu.org/licenses/>.            */
/*************************************************************************************/

#include <algorithm>
#include "accelerator_decl.hpp"
#include "debug.hpp"
#include "schedule.hpp"
#include "copydata.hpp"
#include "instrumentation.hpp"
#include "system.hpp"
#include "functors.hpp"

using namespace nanos;

//#if LOCK_TRANSFER
//Lock Accelerator::_transferLock;
//#endif
//
//Accelerator::Accelerator ( const Device *arch, const Device *subArch, memory_space_id_t memId ) : ProcessingElement( arch, subArch, memId ) {}
//
//void Accelerator::waitInputs( WorkDescriptor &work )
//{
//   this->waitInputsDependent( work );
//}
//
//void Accelerator::copyDataOut( WorkDescriptor& work )
//{
//#if LOCK_TRANSFER
//   _transferLock.acquire();
//#endif
//#if LOCK_TRANSFER
//   _transferLock.release();
//#endif
//}
