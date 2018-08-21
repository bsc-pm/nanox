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


#ifndef _NANOS_CONTAINER_TRAITS
#define _NANOS_CONTAINER_TRAITS


namespace nanos {


namespace traits_private {
   template <class CONTAINER_T>
   static long key_type_checker(typename CONTAINER_T::key_type *);
   template <class CONTAINER_T>
   static char key_type_checker(...);
}


template <class CONTAINER_T>
struct container_traits {
public:
   enum {
      is_associative = ( sizeof(traits_private::key_type_checker<CONTAINER_T>(0)) != sizeof(char) )
   };
};


} // namespace nanos


#endif // _NANOS_CONTAINER_TRAITS
