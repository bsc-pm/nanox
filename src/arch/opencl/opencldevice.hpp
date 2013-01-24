
#ifndef _OpenCL_DEVICE
#define _OpenCL_DEVICE

#include "opencldevice_decl.hpp"
#include "openclprocessor.hpp" 

using namespace nanos;
using namespace nanos::ext;

OpenCLDevice::OpenCLDevice( const char *name ) : Device( name ) { }

void *OpenCLDevice::allocate( size_t size, ProcessingElement *pe )
{
   if( OpenCLProcessor *proc = dynamic_cast<OpenCLProcessor *>( pe ) )
      return proc->allocate( size );


   fatal( "Can allocate only on OpenCLProcessor" );
}

void *OpenCLDevice::realloc( void * address,
                          size_t size,
                          size_t ceSize,
                          ProcessingElement *pe )
{
   if( OpenCLProcessor *proc = dynamic_cast<OpenCLProcessor *>( pe ) )
      return proc->realloc( address, size, ceSize );
   fatal( "Can reallocate only on OpenCLProcessor" );
}

void OpenCLDevice::free( void *address, ProcessingElement *pe )
{
   if( OpenCLProcessor *proc = dynamic_cast<OpenCLProcessor *>( pe ) )
      return proc->free( address );


   fatal( "Can free only on OpenCLProcessor" );
}

bool OpenCLDevice::copyIn( void *localDst,
                        CopyDescriptor &remoteSrc,
                        size_t size,
                        ProcessingElement *pe )
{
   if( OpenCLProcessor *proc = dynamic_cast<OpenCLProcessor *>( pe ) )
   {
      // Current thread is not the device owner: instead of doing the copy, add
      // it to the pending transfer list.
      if( myThread->runningOn() != pe )
         return proc->asyncCopyIn( localDst, remoteSrc, size );

      // We can do a synchronous copy.
      else
         return proc->copyIn( localDst, remoteSrc, size );
   }

   fatal( "Can copyIn only on OpenCLProcessor" );
}

bool OpenCLDevice::copyOut( CopyDescriptor &remoteDst,
                         void *localSrc,
                         size_t size,
                         ProcessingElement *pe )
{
   if( OpenCLProcessor *proc = dynamic_cast<OpenCLProcessor *>( pe ) )
   {
      // Current thread is not the device owner: instead of doing the copy, add
      // it to the pending transfer list.
      if( myThread->runningOn() != pe )
         return proc->asyncCopyOut( remoteDst, localSrc, size );

      // We can do a synchronous copy.
      else
         return proc->copyOut( remoteDst, localSrc, size );
   }

   fatal( "Can copyOut only on OpenCLProcessor" );
}

void OpenCLDevice::syncTransfer( uint64_t hostAddress, ProcessingElement *pe )
{
   if( OpenCLProcessor *proc = dynamic_cast<OpenCLProcessor *>( pe ) )
   {
      proc->syncTransfer( hostAddress );
      return;
   }

   fatal( "Can syncTransfer only on OpenCLProcessor" );
}

#endif // _OpenCL_DEVICE
