/*
	Cell/SMP Superscalar (CellSs/SMPSs): Easy programming the Cell BE/Shared Memory Processors
	Copyright (C) 2008 Barcelona Supercomputing Center - Centro Nacional de Supercomputacion
	
	This library is free software; you can redistribute it and/or
	modify it under the terms of the GNU Lesser General Public
	License as published by the Free Software Foundation; either
	version 2.1 of the License, or (at your option) any later version.
	
	This library is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
	Lesser General Public License for more details.
	
	You should have received a copy of the GNU Lesser General Public
	License along with this library; if not, write to the Free Software
	Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
	
	The GNU Lesser General Public License is contained in the file COPYING.
*/


#ifndef _NANOS_CONTAINER_ADAPTER
#define _NANOS_CONTAINER_ADAPTER

#include "containertraits.hpp"


namespace nanos {


template <class CONTAINER_T,bool IS_ASSOCIATIVE = container_traits<CONTAINER_T>::is_associative>
class ContainerAdapter {
public:
   static inline void insert(CONTAINER_T &container, typename CONTAINER_T::value_type const &value)
   {
      container.insert(value);
   }
   
   static inline typename CONTAINER_T::value_type pop(CONTAINER_T &container)
   {
      typename CONTAINER_T::iterator it = container.begin();
      typename CONTAINER_T::value_type value = *it;
      container.erase(it);
      return value;
   }
   
};


template <class CONTAINER_T>
class ContainerAdapter<CONTAINER_T, false> {
public:
   static inline void insert(CONTAINER_T &container, typename CONTAINER_T::value_type const &value)
   {
      container.push_back(value);
   }
   
   static inline typename CONTAINER_T::value_type pop(CONTAINER_T &container)
   {
      typename CONTAINER_T::value_type value = container.back();
      container.pop_back();
      return value;
   }
   
};


}


#endif // _NANOS_CONTAINER_ADAPTER
