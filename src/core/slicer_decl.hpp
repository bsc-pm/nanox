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

//!\file slicers_decl.hpp
//!\brief Slicer main classes declaration
//
//!\defgroup core_slicers Slicers module
//!\ingroup  core
//
/*!\page    core_slicers
 * \ingroup core_slicers
 *
 * \section slicer_wd Sliced Work Descriptor
 *
 * \section slicer_objects Slicer objects
 * Nanos++ defines several Slicer Objects. Each of them defines an specific behaviour for
 * submit() and dequeue() methods. We have currently implemented the following Slicer Objects:
 *
 * - \copybrief nanos::ext::SlicerStaticFor
 * - \copybrief nanos::ext::SlicerDynamicFor
 * - \copybrief nanos::ext::SlicerGuidedFor
 * - \copybrief nanos::ext::SlicerCompoundWD
 * - \copybrief nanos::ext::SlicerRepeatN
 * - \copybrief nanos::ext::SlicerReplicate
 *
 */

#ifndef _NANOS_SLICER_DECL_H
#define _NANOS_SLICER_DECL_H

#include "workdescriptor_decl.hpp"
#include "schedule_decl.hpp"
#include "nanos-int.h"
#include "slicer_fwd.hpp"
#include <list>                                                                                                                                          

namespace nanos {

   class Slicer
   {
      private:
         /*! \brief Slicer copy constructor (disabled)
          */
         Slicer ( const Slicer &s );
         /*! \brief Slicer copy assignment operator
          */
         Slicer & operator= ( const Slicer &s );
      public:
         /*! \brief Slicer default constructor
          */
         Slicer () { }
         /*! \brief Slicer destructor
          */
         virtual ~Slicer () { }
         /*! \brief Submit a WorkDescriptor (pure virtual)
          */
         virtual void submit ( WorkDescriptor &work ) = 0;
         /*! \brief Dequeue on a WorkDescriptor getting a slice (pure virtual)
          */
         virtual bool dequeue ( WorkDescriptor *wd, WorkDescriptor **slice ) = 0;
         /*! \brief Get Slicer specific data 
          */
         virtual void *getSpecificData ( ) const;
   };

} // namespace nanos

#endif

