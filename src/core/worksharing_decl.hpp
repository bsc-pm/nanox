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

#include "nanos-int.h"

#ifndef _NANOS_WORK_SHARING_H
#define _NANOS_WORK_SHARING_H

namespace nanos {

   class WorkSharing {
      public:

         WorkSharing () {}

         virtual ~WorkSharing () {}

         //! \brief create a loop descriptor
         //! \return only one thread per loop will get 'true' (single like behaviour)
         virtual bool create( nanos_ws_desc_t **wsd, nanos_ws_info_t *info ) = 0;

         //! \brief Duplicates a WorkSharing Descriptor
         virtual void duplicateWS ( nanos_ws_desc_t *orig, nanos_ws_desc_t **copy) = 0;

         //! \brief Get next chunk of iterations
         //! \return if there are more iterations to execute
         virtual void nextItem( nanos_ws_desc_t *wsd, nanos_ws_item_t *wsi ) = 0 ;

         //! \brief Get the number of chunks that remain to be executed
         //! \return number of chunks
         virtual int64_t getItemsLeft( nanos_ws_desc_t *wsd ) = 0 ;

         //! \brief Get whether the WorkSharing needs to be fully instanced for all threads
         virtual bool instanceOnCreation() = 0;
   };

} // namespace nanos

#endif
