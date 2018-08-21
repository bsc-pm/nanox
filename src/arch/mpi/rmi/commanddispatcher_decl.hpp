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

#ifndef COMMAND_DISPATCHER_DECL_HPP
#define COMMAND_DISPATCHER_DECL_HPP

namespace nanos {
namespace mpi {
namespace command {
namespace detail {

/* Assumes Iterator's container can access elements randomly */
template < typename Iterator >
class iterator_range;

template < typename Iterator >
iterator_range<Iterator> make_range( Iterator const& begin, Iterator const& end );

template < typename Iterator >
iterator_range<Iterator> make_range( Iterator const& begin, size_t size );

} // namespace detail
} // namespace command
} // namespace mpi
} // namespace nanos

#endif // COMMAND_DISPATCHER_DECL_HPP
