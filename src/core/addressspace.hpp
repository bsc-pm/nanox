#ifndef ADDRESSSPACE_H
#define ADDRESSSPACE_H
#include "addressspace_decl.hpp"
namespace nanos {

template < class T >
void MemSpace< T >::copy( MemSpace< SeparateAddressSpace > &from, TransferList &list, WD const &wd, bool inval ) {
   for ( TransferList::const_iterator it = list.begin(); it != list.end(); it++ ) {
      this->doOp( from, it->getRegion(), it->getVersion(), wd, it->getCopyIndex(), it->getDeviceOps(), it->getChunk(), inval );
   }
}

}
#endif /* ADDRESSSPACE_H */
