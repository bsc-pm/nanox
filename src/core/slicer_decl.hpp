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
 * \copydoc nanos::SlicedWD
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
namespace nanos
{
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
         virtual void submit ( SlicedWD &work ) = 0;
         /*! \brief Dequeue on a WorkDescriptor getting a slice (pure virtual)
          */
         virtual bool dequeue ( SlicedWD *wd, WorkDescriptor **slice ) = 0;
         /*! \brief Get Slicer specific data 
          */
         virtual void *getSpecificData ( ) const;
   };

/*!\class SlicedWD
 * \brief A Sliced Work Descriptor is an specific class of WorkDescriptor which potentially can be
 * divided in smaller WorkDescriptor's
 *
 * A SlicedWD (Sliced Work Descriptor) is a specific class which derives from WorkDescriptor. Main
 * idea behind this class is to offer a mechanism which allow to decompose a WorkDescriptor in a
 * set of several WorkDescriptors. Initial implementation of this mechanism is related with the
 * ticket:96.
 *
 * A SlicedWD will be always related with:
 *
 * - a Slicer, which defines the work descriptor behaviour.
 * - a SlicerData, which keeps all the data needed for splitting the work.
 * - Slicer objects are common for all the SlicedWD of an specific type. In fact, the Slicer object
 *   determines the type of the SlicedWD. In the other hand, SlicerData objects are individual for
 *   each SlicedWD object.
 *
 * This mechanism is implemented as a derived class from WorkDescriptor: the SlicedWD. A SlicedWD
 * overrides the implementation of submit() and dequeue() methods which have been already defined
 * in the base class.
 *
 * In the base class, submit() method just call Scheduller::submit() method and dequeue() returns
 * the WD itself (meaning this is the work unit ready to be executed) and a boolean value (true,
 * meaning that it will be the last execution for this unit of work). Otherwise, derived class
 * SlicedWD will execute Slicer::submit() and Slicer::dequeue() respectively, giving the slicer
 * the responsibility of doing specific actions at submission or dequeuing time.
 *
 */
   class SlicedWD : public WorkDescriptor
   {
      private:
         Slicer      &_slicer;               /**< Related Slicer     */
         bool         _isSliceable;          /**< WD is sliceable (true by deafault) */
      private:
         /*! \brief SlicedWD default constructor (disabled)
          */
         SlicedWD ();
         /*! \brief SlicedWD copy constructor (disabled)
          */
         SlicedWD ( const SlicedWD &swd );
         /*! \brief SlicedWD copy assignment operator (disabled) 
          */
         SlicedWD & operator= ( const SlicedWD &swd );
      public:
         /*! \brief SlicedWD constructor - n devices
          */
          SlicedWD ( Slicer &slicer, int ndevices, DeviceData **devs, size_t data_size, int data_align,
                     void *wdata, size_t numCopies, CopyData *copies, char *desc )
                   : WorkDescriptor ( ndevices, devs, data_size, data_align, wdata, numCopies, copies, NULL, desc ),
                     _slicer(slicer), _isSliceable(true)  {}

         /*! \brief SlicedWD constructor - 1 device
          */
          SlicedWD ( Slicer &slicer, DeviceData *device, size_t data_size, int data_align, void *wdata,
                     size_t numCopies, CopyData* copies, char *desc )
                   : WorkDescriptor ( device, data_size, data_align, wdata, numCopies, copies, NULL, desc ),
                     _slicer(slicer), _isSliceable(true)  {}
         /*! \brief SlicedWD constructor - from wd
          */
          SlicedWD ( Slicer &slicer, WD &wd, DeviceData **device, CopyData *copies, void *wdata )
                   : WorkDescriptor ( wd, device, copies, wdata),
                     _slicer(slicer), _isSliceable(true)  {}

         /*! \brief SlicedWD destructor
          */
         ~SlicedWD  () {}
         /*! \brief Get related slicer
          */
         Slicer * getSlicer ( void ) const;
         /*! \brief WD submission
          *
          *  This function calls the specific code for WD submission which is
          *  implemented in the related slicer.
          */ 
         void submit ();
         /*! \brief WD dequeue
          *
          *  This function calls the specific code for WD dequeue which is
          *  implemented in the related slicer.
          *
          *  \param[in,out] slice : Resulting slice.
          *  \return  true if the resulting slice is the final slice and false otherwise.
          */ 
         bool dequeue ( WorkDescriptor **slice );
         /*! \brief Convert SlicedWD to a regular WD (changing the behaviour)
          *
          *  This functions change _isSliceable attribute which is used in
          *  submit and dequeue slicedWD function.
          */
         void convertToRegularWD();
   };

};

#endif

