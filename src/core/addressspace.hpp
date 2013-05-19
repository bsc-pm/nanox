#ifndef ADDRESSSPACE_H
#define ADDRESSSPACE_H
#include "addressspace_decl.hpp"
namespace nanos {

template < class T >
void MemSpace< T >::copy( MemSpace< SeparateAddressSpace > &from, TransferList list, WD const &wd ) {
   for ( TransferList::const_iterator it = list.begin(); it != list.end(); it++ ) {
      if ( from.lockForTransfer( it->getRegion(), it->getVersion() ) ) {
         this->doOp( from, it->getRegion(), it->getVersion(), wd, it->getDeviceOps() );
         //from.releaseForTransfer( it->first, it->second );
      } else {
         this->failToLock( from, it->getRegion(), it->getVersion() );
      }
   }
}

}
#endif /* ADDRESSSPACE_H */
