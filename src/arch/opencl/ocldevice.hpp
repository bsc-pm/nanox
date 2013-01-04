
#ifndef _OCL_DEVICE
#define _OCL_DEVICE

#include "ocldevice_decl.hpp"
#include "oclprocessor.hpp" 

using namespace nanos;
using namespace nanos::ext;

OCLDevice::OCLDevice( const char *name ) : Device( name ) { }

void *OCLDevice::allocate( size_t size, ProcessingElement *pe )
{
   if( OCLProcessor *proc = dynamic_cast<OCLProcessor *>( pe ) )
      return proc->allocate( size );


   fatal( "Can allocate only on OCLProcessor or OCLRemoteProcessor" );
}

void *OCLDevice::realloc( void * address,
                          size_t size,
                          size_t ceSize,
                          ProcessingElement *pe )
{
   if( OCLProcessor *proc = dynamic_cast<OCLProcessor *>( pe ) )
      return proc->realloc( address, size, ceSize );
   fatal( "Can reallocate only on OCLProcessor or OCLRemoteProcessor" );
}

void OCLDevice::free( void *address, ProcessingElement *pe )
{
   if( OCLProcessor *proc = dynamic_cast<OCLProcessor *>( pe ) )
      return proc->free( address );


   fatal( "Can free only on OCLProcessor or OCLRemoteProcessor" );
}

bool OCLDevice::copyIn( void *localDst,
                        CopyDescriptor &remoteSrc,
                        size_t size,
                        ProcessingElement *pe )
{
   if( OCLProcessor *proc = dynamic_cast<OCLProcessor *>( pe ) )
   {
      // Current thread is not the device owner: instead of doing the copy, add
      // it to the pending transfer list.
      if( myThread->runningOn() != pe )
         return proc->asyncCopyIn( localDst, remoteSrc, size );

      // We can do a synchronous copy.
      else
         return proc->copyIn( localDst, remoteSrc, size );
   }

   fatal( "Can copyIn only on OCLProcessor or OCLRemoteProcessor" );
}

bool OCLDevice::copyOut( CopyDescriptor &remoteDst,
                         void *localSrc,
                         size_t size,
                         ProcessingElement *pe )
{
   if( OCLProcessor *proc = dynamic_cast<OCLProcessor *>( pe ) )
   {
      // Current thread is not the device owner: instead of doing the copy, add
      // it to the pending transfer list.
      if( myThread->runningOn() != pe )
         return proc->asyncCopyOut( remoteDst, localSrc, size );

      // We can do a synchronous copy.
      else
         return proc->copyOut( remoteDst, localSrc, size );
   }

   fatal( "Can copyOut only on OCLProcessor or OCLRemoteProcessor" );
}

void OCLDevice::syncTransfer( uint64_t hostAddress, ProcessingElement *pe )
{
   if( OCLProcessor *proc = dynamic_cast<OCLProcessor *>( pe ) )
   {
      proc->syncTransfer( hostAddress );
      return;
   }

   fatal( "Can syncTransfer only on OCLProcessor or OCLRemoteProcessor" );
}

#endif // _OCL_DEVICE
