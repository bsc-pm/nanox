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


}


#endif // _NANOS_CONTAINER_TRAITS
