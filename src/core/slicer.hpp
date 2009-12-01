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

namespace nanos
{

// Forward declarations

   class SlicedWD;

   class Slicer
   {
      private:
      public:
         // constructor
         Slicer ( ) { }
         // destructor
         virtual ~Slicer ( ) { }

         virtual void submit ( WorkDescriptor & work ) = 0;
         virtual bool dequeue ( SlicedWD *wd, WorkDescriptor **slice ) = 0;
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
         Slicer & _slicer;         /**< Related Slicer */
         SlicerData & _slicerData; /**< Related SlicerData */
      public:
          // constructors
          SlicedWD ( Slicer &slicer, SlicerData &sdata, int ndevices, DeviceData **devs,
                     size_t data_size, void *wdata=0 ) :
                     WorkDescriptor ( ndevices, devs, data_size, wdata),
                     _slicer(slicer), _slicerData(sdata)  {}

          SlicedWD ( Slicer &slicer, SlicerData &sdata, DeviceData *device,
                     size_t data_size, void *wdata=0 ) :
                      WorkDescriptor ( device, data_size, wdata),
                     _slicer(slicer), _slicerData(sdata)  {}

         // destructor
         ~SlicedWD  ( ) { }

         // get/set functions
         Slicer * getSlicer ( void ) { return &_slicer; }
         void setSlicer ( Slicer &slicer ) { _slicer = slicer; }
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

   class SlicerRepeatN: public Slicer
   {
      private:
      public:
         // constructor
         SlicerRepeatN ( ) { }

         // destructor
         ~SlicerRepeatN ( ) { }

         // headers (implemented in slicer.cpp)
         void submit ( WorkDescriptor & work ) ;
         bool dequeue ( SlicedWD *wd, WorkDescriptor **slice ) ;
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

   class Slicers
   {
      private:
         SlicerRepeatN _slicerRepeatN; /**< Repeat N slicer */
      public:
         // constructor
         Slicers ( ) { }

         // destructor
         ~Slicers ( ) { }

         // get functions
         Slicer & getSlicerRepeatN ( ) { return _slicerRepeatN; }
   };

};

#endif

