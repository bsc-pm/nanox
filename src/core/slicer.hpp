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

   class Slicer
   {
      private:
      public:
         Slicer ( ) { }
         virtual ~Slicer ( ) { }
         virtual void submit ( WorkDescriptor & work )  { Scheduler::submit(work); }
         virtual bool dequeue ( WorkDescriptor *wd, WorkDescriptor **slice ) { *slice = wd; return true; }
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
          SlicedWD ( Slicer &slicer, SlicerData &sdata, int ndevices, DeviceData **devs,void *wdata=0 ) :
            WorkDescriptor ( ndevices, devs, wdata), _slicer(slicer), _slicerData(sdata)  {}
          SlicedWD ( Slicer &slicer, SlicerData &sdata, DeviceData *device, void *wdata=0 ) :
            WorkDescriptor ( device, wdata), _slicer(slicer), _slicerData(sdata)  {}

	 void submit () { _slicer.submit(*this); }
         bool dequeue ( WorkDescriptor **slice ) { return _slicer.dequeue(this,slice); }
   };

   class SlicerStatic: public Slicer
   {
      private:
      public:
         SlicerStatic ( ) { }
         ~SlicerStatic ( ) { }
         void submit ( WorkDescriptor & work ) ;
         bool dequeue ( WorkDescriptor *wd, WorkDescriptor **slice ) ;
   };

   class SlicerDataStatic : public SlicerData
   {
      private:
      public:
         SlicerDataStatic ( ) { }
         ~SlicerDataStatic ( ) { }
   };

   class Slicers
   {
      private:
         SlicerStatic _slicerStatic;
      public:
         Slicers ( ) { }
         ~Slicers ( ) { }

         Slicer & getSlicerStatic ( ) { return _slicerStatic; }
   };

};

#endif

