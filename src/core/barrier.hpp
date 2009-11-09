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

#ifndef __NANOS_BARRIER_H
#define __NANOS_BARRIER_H

namespace nanos
{

   class Barrier
   {
      private:
         Barrier(const Barrier &);
         const Barrier operator= ( const Barrier & );
      public:
         Barrier() {}
         virtual ~Barrier() {}

         //removed init: it is not used in any barrier
         //virtual void init() = 0;

         virtual void barrier() = 0;
   };

   typedef Barrier * ( *barrFactory ) ();

}

#endif
