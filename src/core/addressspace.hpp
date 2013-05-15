#ifndef ADDRESSSPACE_H
#define ADDRESSSPACE_H
#include "addressspace_decl.hpp"
namespace nanos {

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

}
#endif /* ADDRESSSPACE_H */
