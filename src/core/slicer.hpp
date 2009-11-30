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
         Slicer ( ) { }
         virtual ~Slicer ( ) { }
         //virtual void submit ( WorkDescriptor & work )  { Scheduler::submit(work); }
         //virtual bool dequeue ( WorkDescriptor *wd, WorkDescriptor **slice ) { *slice = wd; return true; }
         virtual void submit ( WorkDescriptor & work ) = 0;
         virtual bool dequeue ( SlicedWD *wd, WorkDescriptor **slice ) = 0;
   };

   class SlicerData
   {
      private:
      public:
         SlicerData ( ) { }
         ~SlicerData ( ) { }
   };

   class SlicedWD : public WD
   {
      private:
         Slicer & _slicer;         /**< Related Slicer */
         SlicerData & _slicerData; /**< Related SlicerData */
      public:
          SlicedWD ( Slicer &slicer, SlicerData &sdata, int ndevices, DeviceData **devs, size_t data_size, void *wdata=0 ) :
            WorkDescriptor ( ndevices, devs, data_size, wdata), _slicer(slicer), _slicerData(sdata)  {}
          SlicedWD ( Slicer &slicer, SlicerData &sdata, DeviceData *device, size_t data_size, void *wdata=0 ) :
            WorkDescriptor ( device, data_size, wdata), _slicer(slicer), _slicerData(sdata)  {}

	 void submit () { _slicer.submit(*this); }
         bool dequeue ( WorkDescriptor **slice ) { return _slicer.dequeue( this, slice ); }
         Slicer * getSlicer ( void ) { return &_slicer; }
         SlicerData * getSlicerData ( void ) { return &_slicerData; }
   };

   class SlicerRepeatN: public Slicer
   {
      private:
      public:
         SlicerRepeatN ( ) { }
         ~SlicerRepeatN ( ) { }
         void submit ( WorkDescriptor & work ) ;
         bool dequeue ( SlicedWD *wd, WorkDescriptor **slice ) ;
   };

   class SlicerDataRepeatN : public SlicerData
   {
      private:
         int _n; /**< Number of Repetitions */
      public:
         SlicerDataRepeatN ( int n) : _n (n) { }
         ~SlicerDataRepeatN ( ) { }
         int decN () { return --_n; }
   };

   class Slicers
   {
      private:
         SlicerRepeatN _slicerRepeatN;
      public:
         Slicers ( ) { }
         ~Slicers ( ) { }

         Slicer & getSlicerRepeatN ( ) { return _slicerRepeatN; }
   };

};

#endif

