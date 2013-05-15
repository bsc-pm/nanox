#ifndef ADDRESSSPACE_H
#define ADDRESSSPACE_H
#include "addressspace_decl.hpp"
namespace nanos {

#if 0
template <>
template <>
void MemSpace< HostAddressSpace >::copy( MemSpace< SeparateAddressSpace > &from, TransferListType list, DeviceOps *ops ) {
   for ( TransferListType::iterator it = list.begin(); it != list.end(); it++ ) {
      if ( from.lockForTransfer( *it ) ) {
         doOp( from, *it, ops );
         from.releaseForTransfer( *it );
      } else {
         failToLock( from, *it, ops );
      }
   }
}
template <>
template < class T2 >
void MemSpace< SeparateAddressSpace >::copy( MemSpace< T2 > &from, TransferListType list, DeviceOps *ops ) {
   for ( TransferListType::iterator it = list.begin(); it != list.end(); it++ ) {
      if ( from.lockForTransfer( *it ) ) {
         doOp( from, *it, ops );
         from.releaseForTransfer( *it );
      } else {
         failToLock( from, *it, ops );
      }
   }
}
#else

template < class T>
void MemSpace< T >::copy( MemSpace< SeparateAddressSpace > &from, TransferListType list, WD const &wd ) {
   for ( TransferListType::iterator it = list.begin(); it != list.end(); it++ ) {
      if ( from.lockForTransfer( it->first, it->second ) ) {
         this->doOp( from, it->first, it->second, wd );
         //from.releaseForTransfer( it->first, it->second );
      } else {
         this->failToLock( from, it->first, it->second );
      }
   }
}

#endif

}
#endif /* ADDRESSSPACE_H */
