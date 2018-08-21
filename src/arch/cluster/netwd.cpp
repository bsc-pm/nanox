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

#include "netwd_decl.hpp"
#include "workdescriptor.hpp"
#include "system_decl.hpp"
#ifdef OpenCL_DEV
#include "opencldd.hpp"
#include "opencldevice_decl.hpp"
#endif
#ifdef FPGA_DEV
#include "fpgadd.hpp"
#include "fpgadevice_decl.hpp"
#endif
#ifdef GPU_DEV
#include "gpudd.hpp"
#endif


using namespace nanos;

void * local_nanos_smp_factory( void *args );
void * local_nanos_smp_factory( void *args )
{
   nanos_smp_args_t *smp = ( nanos_smp_args_t * ) args;
   return ( void * )new ext::SMPDD( smp->outline );
}

#ifdef OpenCL_DEV
void * local_nanos_ocl_factory( void *args );
void * local_nanos_ocl_factory( void *args )
{
   nanos_smp_args_t *smp = ( nanos_smp_args_t * ) args;
   return ( void * )new ext::OpenCLDD( smp->outline );
}
#endif

#ifdef FPGA_DEV
void * local_nanos_fpga_factory( void *args );
void * local_nanos_fpga_factory( void *args )
{
   nanos_smp_args_t *smp = ( nanos_smp_args_t * ) args;
   return ( void * )new ext::FPGADD( smp->outline );
}
#endif

#ifdef GPU_DEV
void * local_nanos_gpu_factory( void *args );
void * local_nanos_gpu_factory( void *args )
{
   nanos_smp_args_t *smp = ( nanos_smp_args_t * ) args;
   return ( void * )new ext::GPUDD( smp->outline );
   //if ( prealloc != NULL )
   //{
   //   return ( void * )new (prealloc) ext::GPUDD( smp->outline );
   //}
   //else
   //{
   //   return ( void * ) new ext::GPUDD( smp->outline );
   //}
}
#endif

namespace nanos {
namespace ext {


std::size_t SerializedWDFields::getTotalSize( WD const &wd ) {
   unsigned int totalDimensions = 0;
   std::size_t total_size = 0;
   for (unsigned int i = 0; i < wd.getNumCopies(); i += 1) {
      totalDimensions += wd.getCopies()[i].getNumDimensions();
   }
   total_size = sizeof(SerializedWDFields) +
      wd.getNumCopies() * sizeof( CopyData ) + 
      totalDimensions * sizeof( nanos_region_dimension_t ) +
      wd.getDataSize();
   return total_size;
}

void SerializedWDFields::setup( WD const &wd ) {
   _wdId =  wd.getId();
   _outline = wd.getActiveDevice().getWorkFct();
   _xlate =  wd.getTranslateArgs();
   _dataSize = wd.getDataSize();
   _numCopies = wd.getNumCopies();
   _descriptionAddr = wd.getDescription();
   _wd = &wd;
   _totalDimensions = 0;
   for (unsigned int i = 0; i < wd.getNumCopies(); i += 1) {
      _totalDimensions += wd.getCopies()[i].getNumDimensions();
   }

   if ( wd.canRunIn( getSMPDevice() ) ) {
      _archId = 0;
   }
#ifdef GPU_DEV
   else if ( wd.canRunIn( GPU ) )
   {
      _archId = 1;
   }
#endif
#ifdef OpenCL_DEV
   else if ( wd.canRunIn( OpenCLDev ) )
   {
      _archId = 2;
   }
#endif
#ifdef FPGA_DEV
   else if ( wd.canRunIn( FPGA ) )
   {
      _archId = 3;
   }
#endif
   else {
      fatal("unsupported architecture");
   }
}

CopyData *SerializedWDFields::getCopiesAddr() const {
   char *addr = (char *)this;
   addr += sizeof( SerializedWDFields );
   return (CopyData *) addr;
}

nanos_region_dimension_internal_t *SerializedWDFields::getDimensionsAddr() const {
   return (nanos_region_dimension_internal_t *) (((char *) getCopiesAddr()) + _numCopies * sizeof( CopyData ));
}

char *SerializedWDFields::getDataAddr() const {
   return (((char *)getDimensionsAddr()) + _totalDimensions * sizeof( nanos_region_dimension_internal_t ));
}

std::size_t SerializedWDFields::getTotalDimensions() const {
   return _totalDimensions;
}

std::size_t SerializedWDFields::getDataSize() const {
   return _dataSize;
}

void (*SerializedWDFields::getXlateFunc() const)(void *, void*) {
   return _xlate;
}

unsigned int SerializedWDFields::getArchId() const {
   return _archId;
}

unsigned int SerializedWDFields::getWDId() const {
   return _wdId;
}

WD const *SerializedWDFields::getWDAddr() const {
   return _wd;
}

void (*SerializedWDFields::getOutline() const)(void *) {
   return _outline;
}

unsigned int SerializedWDFields::getNumCopies() const {
   return _numCopies;
}

const char *SerializedWDFields::getDescriptionAddr() const {
   return _descriptionAddr;
}

WD2Net::WD2Net( WD const &wd ) {
   _bufferSize = SerializedWDFields::getTotalSize( wd );
   _buffer = new char[ _bufferSize ];
   SerializedWDFields *swd = ( SerializedWDFields * ) _buffer;
   swd->setup( wd );

   if ( wd.getDataSize() > 0 )
   {
      ::memcpy( swd->getDataAddr(), wd.getData(), wd.getDataSize() );
   }

   CopyData *newCopies = ( CopyData * ) swd->getCopiesAddr();
   nanos_region_dimension_internal_t *dimensions = ( nanos_region_dimension_internal_t * ) swd->getDimensionsAddr();
   
   uintptr_t dimensionIndex = 0;
   for (unsigned int i = 0; i < wd.getNumCopies(); i += 1) {
      CopyData &cd = ( wd.getCopies()[i].getDeductedCD() != NULL ) 
         ? *(wd.getCopies()[i].getDeductedCD()) 
         : wd.getCopies()[i];
      new ( &newCopies[i] ) CopyData( cd );
      if ( newCopies[i].getDeductedCD() != NULL ) {
         std::cerr << "REGISTERED REG!!!!" << std::endl;
      }
      memcpy( &dimensions[ dimensionIndex ], cd.getDimensions(), sizeof( nanos_region_dimension_internal_t ) * cd.getNumDimensions());
      newCopies[i].setDimensions( ( nanos_region_dimension_internal_t *  ) dimensionIndex ); // This is the index because it makes no sense to send an address over the network
      newCopies[i].setHostBaseAddress( (uint64_t) cd.getBaseAddress() );
      newCopies[i].setRemoteHost( true );
      //newCopies[i].setBaseAddress( (void *) ( wd._ccontrol.getAddress( i ) - cd.getOffset() ) );
      newCopies[i].setBaseAddress( (void *) wd._mcontrol.getAddress( i ) );
      newCopies[i].setHostRegionId( wd._mcontrol._memCacheCopies[i]._reg.id );
      dimensionIndex += cd.getNumDimensions();
   }
}

WD2Net::~WD2Net() {
   delete[] _buffer;
   _buffer = NULL;
   _bufferSize = 0;
}

char *WD2Net::getBuffer() const {
   return _buffer;
}

std::size_t WD2Net::getBufferSize() const {
   return _bufferSize;
}

Net2WD::Net2WD( char *buffer, std::size_t buffer_size, RemoteWorkDescriptor **rwds ) {
   SerializedWDFields *swd = (SerializedWDFields *) buffer;

   nanos_smp_args_t smp_args = { swd->getOutline() };
   nanos_device_t dev = { NULL, (void *) &smp_args };
   switch (swd->getArchId()) {
      case 0: //SMP
         dev.factory = local_nanos_smp_factory; 
         break;
#ifdef GPU_DEV
      case 1: //CUDA
         dev.factory = local_nanos_gpu_factory;
         break;
#endif
#ifdef OpenCL_DEV
      case 2: //OpenCL
         dev.factory = local_nanos_ocl_factory;
         break;
#endif
#ifdef FPGA_DEV
      case 3: //FPGA
         dev.factory = local_nanos_fpga_factory;
         break;
#endif
      default: //WTF
         break;
   }

   int num_dimensions = swd->getTotalDimensions();
   nanos_region_dimension_internal_t *dimensions = NULL;
   nanos_region_dimension_internal_t **dimensions_ptr = ( num_dimensions > 0 ) ? &dimensions : NULL ;
   char *data = NULL;
   CopyData *newCopies = NULL;
   CopyData **newCopiesPtr = ( swd->getNumCopies() > 0 ) ? &newCopies : NULL ;

   _wd = NULL;
   sys.createWD( &_wd, (std::size_t) 1, &dev, swd->getDataSize(), (int) ( sizeof(void *) ), (void **) &data, rwds[swd->getArchId()], (nanos_wd_props_t *) NULL, (nanos_wd_dyn_props_t *) NULL, swd->getNumCopies(), newCopiesPtr, num_dimensions, dimensions_ptr, swd->getXlateFunc(), swd->getDescriptionAddr(), NULL );

   std::memcpy(data, swd->getDataAddr(), swd->getDataSize());

   // Set copies and dimensions, getDimensions() returns an index here, instead of a pointer,
   // the index is the position inside the dimension array that must be set as the base address for the dimensions
   CopyData *recvCopies = swd->getCopiesAddr();
   nanos_region_dimension_t *recvDimensions = swd->getDimensionsAddr();
   if ( swd->getNumCopies() > 0 ) {
      memcpy( *dimensions_ptr, recvDimensions, num_dimensions * sizeof(nanos_region_dimension_t) );
   }
   for (unsigned int i = 0; i < swd->getNumCopies(); i += 1)
   {
      new ( &newCopies[i] ) CopyData( recvCopies[i] );
      newCopies[i].setDimensions( (*dimensions_ptr) + ( ( uintptr_t ) recvCopies[i].getDimensions() ) );
   }

   _wd->setHostId( swd->getWDId() );
   _wd->setRemoteAddr( swd->getWDAddr() );
}

Net2WD::~Net2WD() {
}

WD *Net2WD::getWD() {
   return _wd;
}

}
}
