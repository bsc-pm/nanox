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

#ifndef _NANOS_COMPATIBILITY_HPP
#define _NANOS_COMPATIBILITY_HPP

// compiler issues

#if __GXX_EXPERIMENTAL_CXX0X__

#include <unordered_map>

namespace TR1 = std;

namespace std {
#else

#include <tr1/unordered_map>

namespace TR1 = std::tr1;

namespace std { namespace tr1 {
#endif

/* Specialize hash for unsigned long long allows unordered_map<uint64_t, xxx> when compiling for 32 bits */
template<> struct hash<unsigned long long> : public std::unary_function<unsigned long long, std::size_t> { std::size_t operator()(unsigned long long val) const { return static_cast<std::size_t>(val); } };
}

#ifndef __GXX_EXPERIMENTAL_CXX0X__
}
#endif

#endif

