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

#ifndef CACHEDREGIONSTATUS_DECL_HPP
#define CACHEDREGIONSTATUS_DECL_HPP

#include "version_decl.hpp"
#include "deviceops_decl.hpp"

namespace nanos {

   class CachedRegionStatus : public Version {
      private:
         DeviceOps _ops;
         bool _dirty;
      public:
         CachedRegionStatus();
         CachedRegionStatus( CachedRegionStatus const &rs );
         DeviceOps * getDeviceOps();

         CachedRegionStatus &operator=( CachedRegionStatus const &rs );
         CachedRegionStatus( CachedRegionStatus &rs );
         CachedRegionStatus &operator=( CachedRegionStatus &rs );

         bool isDirty() const;
         void setDirty();
         void clearDirty();
   };
} // namespace nanos

#endif /* CACHEDREGIONSTATUS_DECL_HPP */
