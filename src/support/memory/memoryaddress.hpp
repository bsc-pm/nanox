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

#ifndef ADDRESS_HPP
#define ADDRESS_HPP

#include <algorithm>
#include <stdint.h>
#include <ostream>

#ifdef HAVE_CXX11
#   define nanos_constexpr constexpr
#else
#   define nanos_constexpr
#endif

namespace nanos {
namespace utils {

/*
 * \brief Abstraction layer for memory addresses.
 * \defails Address provides an easy to use wrapper
 * for address manipulation. Very useful if pointer
 * arithmetic is need to be used.
 */
class Address {
	private:
		uintptr_t value; //!< Memory address

#ifndef HAVE_CXX11
		// Prior to C++11, deleted functions
		// did not exist.
		Address() : value( 0 ) {}
	public:
#else
	public:
		//! \brief Default constructor: avoid unintentional uninitialized addresses
		Address() = delete;
#endif

		/*! \brief Constructor by initialization.
		 *  \details Creates a new Address instance
		 *  using an unsigned integer.
		 */
		nanos_constexpr
		Address( uintptr_t v ) : value( v ) {}

		/*! \brief Constructor by initialization.
		 *  \details Creates a new Address instance
		 *  using a pointer's address.
		 */
		template< typename T >
		nanos_constexpr
		Address( T* v ) : value( reinterpret_cast<uintptr_t>(v) ) {}

		//! \brief Copy constructor
		nanos_constexpr
		Address( Address const& o ) : value(o.value) {}

		//! \brief Checks if two addresses are equal
		nanos_constexpr
		bool operator==( Address const& o ) {
			return value == o.value;
		}

		//! \brief Checks if two addresses differ
		nanos_constexpr
		bool operator!=( Address const& o ) {
			return value != o.value;
		}

		/*! \brief Calculate an address using
		 *  a base plus an offset.
		 *  @param[in] size Offset to be applied
		 *  @returns A new address object displaced size bytes
		 *  with respect to the value of this object.
		 */
		nanos_constexpr
		Address operator+( size_t size ) {
			return Address( value + size );
		}

		/*! \brief Calculate an address using
		 *  a base minus an offset.
		 *  @param[in] size Offset to be applied
		 *  @returns A new address object displaced size bytes
		 *  with respect to the value of this object.
		 */
		nanos_constexpr
		size_t operator-( Address const& base ) {
			return ((uintptr_t)base.value) - ((uintptr_t)value);
			//return reinterpret_cast<uintptr_t>(base.value)
			//	  - reinterpret_cast<uintptr_t>(value);
		}

		Address operator+=( size_t size ) {
			value += size;
			return *this;
		}

		Address operator-=( size_t size ) {
			value += size;
			return *this;
		}

		//! \returns if this address is smaller than the reference
		nanos_constexpr
		bool operator<( Address const& reference ) {
			return value < reference.value;
		}

		//! \returns if this address is greater than the reference
		nanos_constexpr
		bool operator>( Address const& reference ) {
			return value > reference.value;
		}

		//! @returns the integer representation of the address
		nanos_constexpr
		operator uintptr_t() {
			return value;
		}

		/*! @returns the pointer representation of the address
		 * using any type.
		 * \tparam T type of the represented pointer. Default: void
		 */
		template< typename T >
		operator T*() {
			return reinterpret_cast<T*>(value);
		}

		template< typename T >
		operator const T*() const {
			return reinterpret_cast<const T*>(value);
		}

		operator void*() {
			return reinterpret_cast<void*>(value);
		}

		operator const void*() const {
			return reinterpret_cast<const void*>(value);
		}

		/*! @returns whether this address fulfills an
		 * alignment restriction or not.
		 * @param[in] alignment_constraint the alignment
		 * restriction
		 */
		nanos_constexpr
		bool isAligned( size_t alignment_constraint ) {
			return ( value & (alignment_constraint-1)) == 0;
		}

		/*! @returns whether this address fulfills an
		 * alignment restriction or not.
		 * \tparam alignment_constraint the alignment
		 * restriction
		 */
		template< size_t alignment_constraint >
		nanos_constexpr
		bool isAligned() {
			return ( value & (alignment_constraint-1)) == 0;
		}

		/*! @returns returns an aligned address
		 * @param[in] alignment_constraint the alignment to be applied
		 */
		nanos_constexpr
		Address align( size_t alignment_constraint ) {
			return Address(
						value &
						~( alignment_constraint-1 )
						);
		}

		/*! @returns returns an aligned address
		 * @tparam alignment_constraint the alignment to be applied
		 */
		template< size_t alignment_constraint >
		nanos_constexpr
		Address align() {
			return Address(
						value &
						~( alignment_constraint-1 )
						);
		}

		/*! @returns returns an aligned address
		 * @param[in] lsb least significant bit of the aligned address
		 *
		 * \detail LSB is a common term for specifying the important
		 *         part of an address in an specific context.
		 *         For example, in virtual page management, lsb is
		 *         usually 12 ( 2^12: 4096 is the page size ).
		 *
		 *         Basically we have to build a mask where all the bits
		 *         in a position less significant than lsb are equal
		 *         to 0:
		 *          1) Create a number with a '1' value in the lsb-th
		 *             position.
		 *          2) Substract one: all bits below the lsb-th will
		 *             be '1'.
		 *          3) Perform a bitwise-NOT to finish the mask with
		 *             all 1s but in the non-significant bits.
		 */
		nanos_constexpr
		Address alignToLSB( short lsb ) {
			return Address(
						value &
						~( (1<<lsb)-1 )
						);
		}

		/*! @returns returns an aligned address
		 * @tparam alignment_constraint the alignment to be applied
		 * \sa alignUsingLSB"("short lsb")"
		 */
		template< short lsb >
		nanos_constexpr
		Address alignToLSB() {
			return Address(
						value &
						~( (1<<lsb)-1 )
						);
		}

		static Address uninitialized()
		{
#ifndef CXX11
			return Address(0);
#else
			return Address(nullptr);
#endif
		}

		friend std::ostream& operator<<(std::ostream& out, Address const &entry);
};

/*! \brief Checks that an alignment constraint is fulfilled
 */
#ifndef CXX11
nanos_constexpr
bool is_properly_aligned( Address address, size_t alignment_constraint );
#endif

nanos_constexpr
inline bool is_properly_aligned( Address address, size_t alignment_constraint )
{
   return ( static_cast<uintptr_t>(address) & (alignment_constraint-1) ) == 0;
}

/*! \brief Prints an address object to an output stream.
 *  \details String representation of an address in hexadecimal.
 */
std::ostream& operator<<(std::ostream& out, Address const &entry);

} // namespace utils
} // namespaec nanos

#endif // ADDRESS_HPP

