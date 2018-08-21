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

#ifndef _NANOS_SMP_ULT
#define _NANOS_SMP_ULT

#include <stddef.h>
#include <stdint.h>
#include "workdescriptor_fwd.hpp"

extern "C"
{
// low-level routine to switch stacks
   void switchStacks( void *,void *,void *,void * );
}

void * initContext( void *stack, size_t stackSize, void (*wrapperFunction)(nanos::WD&), nanos::WD *wd,
                       void *cleanup, void *cleanupArg );

#ifndef SMP_SUPPORTS_ULT

extern "C" {
   inline void switchStacks( void *,void *,void *,void * ) {}
}

inline void * initContext( void *stack, size_t stackSize, void (*wrapperFunction)(nanos::WD&), nanos::WD *wd,
                       void *cleanup, void *cleanupArg ) { return 0; }

#endif

#endif

