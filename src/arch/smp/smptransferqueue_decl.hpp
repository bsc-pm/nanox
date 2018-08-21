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

#ifndef SMPTRANSFERQUEUE_DECL
#define SMPTRANSFERQUEUE_DECL

#include <list>
#include "atomic_decl.hpp"
#include "deviceops_fwd.hpp"

namespace nanos {

class SMPTransfer {
   DeviceOps   *_ops;
   char        *_dst;
   char        *_src;
   std::size_t  _len;
   std::size_t  _count;
   std::size_t  _ld;
   bool         _in;
   public:
   SMPTransfer();
   SMPTransfer( DeviceOps *ops, char *dst, char *src, std::size_t len, std::size_t count, std::size_t ld, bool in );
   SMPTransfer( SMPTransfer const &s );
   SMPTransfer &operator=( SMPTransfer const &s );
   ~SMPTransfer();
   void execute();
};

class SMPTransferQueue {
   Lock _lock;
   std::list< SMPTransfer > _transfers;
   public:
   SMPTransferQueue();
   void addTransfer( DeviceOps *ops, char *dst, char *src, std::size_t len, std::size_t count, std::size_t ld, bool in );
   void tryExecuteOne();
};

} // namespace nanos

#endif
