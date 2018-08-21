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

#ifndef _NANOS_OpenCL_WD
#define _NANOS_OpenCL_WD

#include "workdescriptor_fwd.hpp"
#include "debug.hpp"
#include "opencldevice_decl.hpp"

namespace nanos {
namespace ext {
    
    

extern OpenCLDevice OpenCLDev;

// OpenCL back-end Device Description.
//
// Since the OpenCL back-end is slightly different from a normal back-end, we
// have a set of specialized Device Description. This is the root of the class
// hierarchy.
//
// Every Device Description must have the following layout in memory:
//
// +------------------+
// | Custom section   |
// +------------------+
// | Number of events |
// +------------------+
// | Input event 1    |
// | ...              |
// | Input event n    |
// +------------------+
// | Output event     |
// +------------------+ -+-
// | Start tick       |  | Profiling section
// | End tick         |  | (optional)
// +------------------+ -+-
//
// The OLCDD class contains the EventIterator class that can be use to iterate
// over events given the custom section size. It is Device Description builder
// responsibility to push arguments in the right order!

class OpenCLDD : public DD
   {
    
      private:
         int _oclStreamIdx;
    
      public:
         // constructors
         OpenCLDD( work_fct w ) : DD( &OpenCLDev, w ), _oclStreamIdx(-1) {}

         OpenCLDD() : DD( &OpenCLDev, NULL ), _oclStreamIdx(-1) {}

         // copy constructors
         OpenCLDD( const OpenCLDD &dd ) : DD( dd ), _oclStreamIdx( dd._oclStreamIdx ) {}

         // assignment operator
         const OpenCLDD & operator= ( const OpenCLDD &wd );

         // destructor
         virtual ~OpenCLDD() { }

         virtual void lazyInit (WD &wd, bool isUserLevelThread, WD *previous) { }
         virtual size_t size ( void ) { return sizeof(OpenCLDD); }
         virtual OpenCLDD *copyTo ( void *toAddr );
         virtual OpenCLDD *clone () const { return NEW OpenCLDD ( *this); }
         void setOpenclStreamIdx(int streamIdx);
         int getOpenCLStreamIdx();
   };

   inline const OpenCLDD & OpenCLDD::operator= ( const OpenCLDD &dd )
   {
      // self-assignment: ok
      if ( &dd == this ) return *this;

      DD::operator= ( dd );
      _oclStreamIdx= dd._oclStreamIdx;

      return *this;
   }
} // namespace ext
} // namespace nanos
#endif
