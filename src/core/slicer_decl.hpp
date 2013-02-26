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

   class SlicedWD : public WD
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

