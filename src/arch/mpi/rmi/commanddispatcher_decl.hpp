
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
