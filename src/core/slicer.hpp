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
#include "slicer_fwd.hpp"
#include <list>                                                                                                                                          

namespace nanos
{

   class Slicer
   {
      private:
      public:
         // constructor
         Slicer ( ) { }
         // destructor
         virtual ~Slicer ( ) { }

         virtual void submit ( SlicedWD & work ) = 0;
         virtual bool dequeue ( SlicedWD *wd, WorkDescriptor **slice ) = 0;
         virtual void *getSpecificData ( ) const { return NULL; }
   };

   class SlicerData
   {
      private:
      public:
         // constructor
         SlicerData ( ) { }
         // destructor
         ~SlicerData ( ) { }
   };

   class SlicedWD : public WD
   {
      private:
         Slicer     & _slicer;         /**< Related Slicer     */
         size_t       _slicerDataSize; /**< SlicerData size    */
         SlicerData & _slicerData;     /**< Related SlicerData */
      public:
          // constructors
          SlicedWD ( Slicer &slicer, size_t sdata_size, SlicerData &sdata, int ndevices, DeviceData **devs,
                     size_t data_size, void *wdata=0, size_t numCopies=0, CopyData *copies=NULL ) :
                     WorkDescriptor ( ndevices, devs, data_size, wdata, numCopies, copies ),
                     _slicer(slicer), _slicerDataSize(sdata_size), _slicerData(sdata)  {}

          SlicedWD ( Slicer &slicer, size_t sdata_size, SlicerData &sdata, DeviceData *device,
                     size_t data_size, void *wdata=0, size_t numCopies=0, CopyData* copies=NULL ) :
                      WorkDescriptor ( device, data_size, wdata, numCopies, copies ),
                     _slicer(slicer), _slicerDataSize(sdata_size), _slicerData(sdata)  {}

          SlicedWD ( Slicer &slicer, size_t sdata_size, SlicerData &sdata, WD &wd,
                      DeviceData **device, CopyData *copies, void *wdata=0 ) :
                      WorkDescriptor ( wd, device, copies, wdata),
                     _slicer(slicer), _slicerDataSize(sdata_size), _slicerData(sdata)  {}

         // destructor
         ~SlicedWD  ( ) { }

         // get/set functions
         Slicer * getSlicer ( void ) { return &_slicer; }
         void setSlicer ( Slicer &slicer ) { _slicer = slicer; }

         size_t getSlicerDataSize ( void ) { return _slicerDataSize; }
         void setSlicerDataSize ( size_t sdata_size ) { _slicerDataSize = sdata_size; }

         SlicerData * getSlicerData ( void ) { return &_slicerData; }
         void setSlicerData ( SlicerData &slicerData ) { _slicerData = slicerData; }

         /*! \brief WD submission
          *
          *  This function calls the specific code for WD submission which is
          *  implemented in the related slicer.
          */ 
         void submit () { _slicer.submit(*this); }

         /*! \brief WD dequeue
          *
          *  This function calls the specific code for WD dequeue which is
          *  implemented in the related slicer.
          *
          *  \param[in,out] slice : Resulting slice.
          *  \return  true if the resulting slice is the final slice and false otherwise.
          */ 
         bool dequeue ( WorkDescriptor **slice ) { return _slicer.dequeue( this, slice ); }
   };

   class SlicerDataRepeatN : public SlicerData
   {
      private:
         int _n; /**< Number of Repetitions */
      public:
         // constructor
         SlicerDataRepeatN ( int n) : _n (n) { }

         // destructor
         ~SlicerDataRepeatN ( ) { }

         // get/set functions
         void setN ( int n ) { _n = n; }
         int getN ( void ) { return _n; }

         /*! \brief Decrement internal counter by one
          *
          *  This function decrements the internal variable counter by one
          *
          *  \return Internal counter after decrementing its value
          */ 
         int decN () { return --_n; }
   };

   class SlicerDataFor : public nanos_slicer_data_for_internal_t, public SlicerData
   {
         /* int _lower: Loop lower bound */
         /* int _upper: Loop upper bound */
         /* int _step: Loop step */
         /* int _chunk: Slice chunk */
         /* int _sign: Loop sign 1 ascendant, -1 descendant */

      public:
         // constructor
         SlicerDataFor ( int lower, int upper, int step, int chunk = 1 )
         {
            _lower = lower;
            _upper = upper;
            _step = step;
            _chunk = chunk; 
            _sign = ( step < 0 ) ? -1 : +1;
         }
         // destructor
         ~SlicerDataFor ( ) { }

         // get/set functions
         void setLower ( int n ) { _lower = n; }
         void setUpper ( int n ) { _upper = n; }
         void setStep  ( int n ) {  _step = n; }
         void setChunk ( int n ) { _chunk = n; }
         void setSign  ( int n ) { _sign = n; }

         int getLower ( void ) { return _lower; }
         int getUpper ( void ) { return _upper; }
         int getStep  ( void ) { return _step; }
         int getChunk ( void ) { return _chunk; }
         int getSign  ( void ) { return _sign; }
   };

};

#endif

