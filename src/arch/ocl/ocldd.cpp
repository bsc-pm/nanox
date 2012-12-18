
#include "ocldd.hpp"
#include "ocldevice.hpp"
#include "oclconfig.hpp"

using namespace nanos;
using namespace nanos::ext;

OCLDevice nanos::ext::OCLDev( "OCL" );

//
// OCLDD implementation.
//

void OCLDD::dump( void *data, size_t evOffset )
{
   std::cerr << "OCLDD(" << this << "):" << std::endl;

   std::cerr << "  _evs = [ ";
   for( event_iterator i = EventIterator::begin( data, evOffset),
                       e = EventIterator::end( data, evOffset );
                       i != e;
                       ++i )
      std::cerr << *i << " ";
   std::cerr << "]" << std::endl;
}

OCLNDRangeKernelStarSSDD::OCLNDRangeKernelStarSSDD( size_t workDim,
                                        size_t *globalWorkOffset,
                                        size_t *globalWorkSize,
                                        size_t *localWorkSize,
                                        void *dataOcl,
                                        bool profiled ) : OCLDD( profiled )
{
   OCLConfig::enableStarSSMode();
   if( workDim > MaxWorkDim )
      fatal0( "Unsupported work dimensions size" );

   _workDim = workDim;
   std::memcpy(_globalWorkOffset, globalWorkOffset, workDim * sizeof(size_t));
   std::memcpy(_globalWorkSize, globalWorkSize, workDim * sizeof(size_t));
   std::memcpy(_localWorkSize, localWorkSize, workDim * sizeof(size_t));
   
   //TODO not sure if needed,think a way to do mallocs with nanos-style (using Nanos API?)
   //Copying data (mallocs done here will be freed after WD is executed, using function freeAllocs)
   size_t sizeStruct=sizeof(Data);
   size_t sizeArgs=0;
   char *cur_pointer;
   for( OCLNDRangeKernelStarSSDD::arg_iterator j = OCLNDRangeKernelStarSSDD::ArgsIterator::begin(dataOcl); j != OCLNDRangeKernelStarSSDD::ArgsIterator::end(dataOcl); ++j ){
       Arg arg=*j;
       sizeStruct+=sizeof(Arg);
       if ((arg._size & 1)==0){           
             sizeArgs+=arg._size;
       }
   }
   //Allocate data for struct and pointed data info
   //copy struct content
   _oclData=NEW char[sizeStruct+sizeArgs];
   std::memcpy(_oclData, dataOcl, sizeStruct);
   cur_pointer=((char *)_oclData)+sizeStruct;
  
   //Copy data content (only for non-pointers)
   for( OCLNDRangeKernelStarSSDD::arg_iterator j = OCLNDRangeKernelStarSSDD::ArgsIterator::begin(_oclData); j != OCLNDRangeKernelStarSSDD::ArgsIterator::end(_oclData); ++j ){
       Arg arg=*j;
       char * origPointer= (char *)arg._ptr;
       //If not buffer, we copy it's content, if buffer its should be already copied to device (via cache)
       if ((arg._size & 1)==0){ 
           j->_ptr=cur_pointer;
           std::memcpy(cur_pointer, origPointer, arg._size);
           cur_pointer+=arg._size;
       }
   }
}

//Remove previously allocated data
void OCLNDRangeKernelStarSSDD::deleteData(void *data){
    delete[] (char*)data;
}