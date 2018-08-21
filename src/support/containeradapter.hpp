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


#ifndef _NANOS_CONTAINER_ADAPTER
#define _NANOS_CONTAINER_ADAPTER

#include "containeradapter_decl.hpp"
#include "containertraits.hpp"

namespace nanos {

template <class CONTAINER_T,bool IS_ASSOCIATIVE >
inline void ContainerAdapter<CONTAINER_T, IS_ASSOCIATIVE>::insert(CONTAINER_T &container, typename CONTAINER_T::value_type const &value)
{
   container.insert(value);
}

template <class CONTAINER_T,bool IS_ASSOCIATIVE >
inline typename CONTAINER_T::value_type ContainerAdapter<CONTAINER_T, IS_ASSOCIATIVE>::pop(CONTAINER_T &container)
{
   typename CONTAINER_T::iterator it = container.begin();
   typename CONTAINER_T::value_type value = *it;
   container.erase(it);
   return value;
}


template <class CONTAINER_T>
inline void ContainerAdapter<CONTAINER_T, false>::insert(CONTAINER_T &container, typename CONTAINER_T::value_type const &value)
{
   container.push_back(value);
}

template <class CONTAINER_T>
inline typename CONTAINER_T::value_type ContainerAdapter<CONTAINER_T, false>::pop(CONTAINER_T &container)
{
   typename CONTAINER_T::value_type value = container.back();
   container.pop_back();
   return value;
}

} // namespace nanos


#endif // _NANOS_CONTAINER_ADAPTER
