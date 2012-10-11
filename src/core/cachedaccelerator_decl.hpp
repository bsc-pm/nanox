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

#ifndef _NANOS_CACHED_ACCELERATOR_DECL
#define _NANOS_CACHED_ACCELERATOR_DECL

#include "accelerator_decl.hpp"
#include "system_decl.hpp"
#include "copydescriptor_decl.hpp"
#include "regioncache_decl.hpp"

namespace nanos
{

   class CachedAccelerator : public Accelerator
   {
      private:
        RegionCache             _newCache;

        /*! \brief CachedAccelerator default constructor (private)
         */
         CachedAccelerator ();
        /*! \brief CachedAccelerator copy constructor (private)
         */
         CachedAccelerator ( const CachedAccelerator &a );
        /*! \brief CachedAccelerator copy assignment operator (private)
         */
         const CachedAccelerator& operator= ( const CachedAccelerator &a );
      public:
        /*! \brief CachedAccelerator constructor - from 'newId' and 'arch'
         */
         CachedAccelerator( int newId, const Device *arch, const Device *subArch = NULL,
            Device *cacheArch = NULL, std::size_t cacheSize = 0 );

        /*! \brief CachedAccelerator destructor
         */
         virtual ~CachedAccelerator();

         void copyDataInDependent( WorkDescriptor &wd );
         void waitInputsDependent( WorkDescriptor &wd );
         virtual Device const *getCacheDeviceType() const;
   };

};

#endif
