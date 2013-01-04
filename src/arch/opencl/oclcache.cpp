
#include "oclcache.hpp"
#include "oclconfig.hpp"
#include "oclprocessor.hpp"

using namespace nanos;
using namespace nanos::ext;

Lock OCLMemDump::_dumpLock;


//
// OCLMemDump implementation.
//

void OCLMemDump::dumpCopyIn(void *localDst,
        CopyDescriptor &remoteSrc,
        size_t size) {
    LockBlock lock(OCLMemDump::_dumpLock);

    std::cerr << "### COPY-IN ###" << std::endl
            << "  Dst = " << localDst
            << std::endl
            << "  Src = " << reinterpret_cast<void *> (remoteSrc.getTag())
            << std::endl;
}

void OCLMemDump::dumpCopyOut(CopyDescriptor &remoteDst,
        void *localSrc,
        size_t size) {
    LockBlock lock(OCLMemDump::_dumpLock);

    std::cerr << "### COPY-OUT ###" << std::endl
            << "  Dst = " << reinterpret_cast<void *> (remoteDst.getTag())
            << std::endl
            << "  Src = " << localSrc
            << std::endl;
}

//
// OCLCache implementation.
//

void OCLCache::initialize() {
    size_t pageSize = getPageSize();

    // Compute the amount of cache for the device.
    _devCacheSize = OCLConfig::getDevCacheSize();
    if (_devCacheSize == 0)
        _devCacheSize = _oclAdapter.getGlobalSize();
    _devCacheSize = roundUpToPageSize(_devCacheSize);

    // Initialize the device allocator. It manages virtual addresses above the
    // addresses managed by host allocator.
    _devAllocator.init(pageSize , _devCacheSize);
}

cl_mem OCLCache::toMemoryObjSizeSS(size_t size, void* addr) {
    cl_mem buf=NULL;
    //Find unused allocated buffer and remove from unused
    std::map<size_t, cl_mem>::iterator iter;
    for (iter = _allocBuffMappings.begin(); iter != _allocBuffMappings.end(); iter++) {
        if (iter->first == size) {
            buf = iter->second;
            _allocBuffMappings.erase(iter);
            break;
        }
    }
    _devBufAddrMappings[addr] = buf;
    return buf;
}

void *OCLCache::deviceAllocate(size_t size) {
    cl_mem buf;

    if (_oclAdapter.allocBuffer(size, buf) != CL_SUCCESS)
        fatal("Device allocation failed");

    void *addr = _devAllocator.allocate(size);

    if (!addr)
        fatal("Device cache allocation failed");

    _allocBuffMappings[size] = buf;
    _bufAddrMappings[addr] = buf;

    return addr;
}

void *OCLCache::deviceReallocate(void * addr, size_t size, size_t ceSize) {
    // TODO: checking wheter we need a more optimized solution.
    
    //TODO: in starSS mode we could remove a buffer from the address-asigned buffers cache
    //and place it on the size-allocated buffers cache (should be faster 
    //and apparently result would be the same)
    //instead of recreating the buffer
    deviceFree(addr);

    return deviceAllocate(size);
}

void OCLCache::deviceFree(void * addr) {
    void *allocAddr = addr;
    _devAllocator.free(allocAddr);
    cl_mem buf = _bufAddrMappings[allocAddr];
    for (std::map<void *, cl_mem>::iterator i = _devBufAddrMappings.begin(),
            e = _devBufAddrMappings.end();
            i != e;
            ++i) {
        if (i->second == buf) {
            if (_oclAdapter.freeBuffer(buf) != CL_SUCCESS)
                fatal("Cannot free device buffer");

            _devBufAddrMappings.erase(i);
            break;
        }
    }
}

bool OCLCache::deviceCopyIn(void *localDst,
        CopyDescriptor &remoteSrc,
        size_t size) {
    void *src = reinterpret_cast<void *> (remoteSrc.getTag());
    cl_int errCode;
    cl_mem buf;    
    //Find unused allocated buffer and remove from unused
    bool found=false;
    std::map<size_t, cl_mem>::iterator iter;
    for (iter = _allocBuffMappings.begin(); iter != _allocBuffMappings.end(); iter++) {
        if (iter->first == size) {
            buf = iter->second;
            _allocBuffMappings.erase(iter);
            found=true;
            break;
        }
    }
    if (found){   
            //This shouldn't be needed, if we found an allocated buffer search for this address in the map and remove it
            std::map<void*, cl_mem>::iterator iter2;
            for (iter2 = _devBufAddrMappings.begin(); iter2 != _devBufAddrMappings.end(); iter2++) {
                if (iter2->first == src) {
                    _devBufAddrMappings.erase(iter2);
                    break;
                }
            }
            _devBufAddrMappings[src] = buf;
    } else {            
            buf = _devBufAddrMappings[src];
    }
    errCode = _oclAdapter.writeBuffer(buf,
              src,
              0,
              size);
    if (errCode != CL_SUCCESS)
        fatal("Buffer writing failed");

    return true;
}

bool OCLCache::deviceCopyOut(CopyDescriptor &remoteDst,
        void *localSrc,
        size_t size) {
    void *dst = reinterpret_cast<void *> (remoteDst.getTag());

    // Local buffer, do not perform copy.
    if (!dst)
        return true;
    cl_int errCode;
    cl_mem buffer=_devBufAddrMappings[dst];
    
    errCode = _oclAdapter.readBuffer(buffer,
                dst,
                0,
                size);
    
    if (errCode != CL_SUCCESS) {
        fatal("Buffer reading failed");
    }

    return true;
}

//
// DMATransfer implementation.
//

namespace nanos {
    namespace ext {

        template <>
        void DMATransfer<void *, CopyDescriptor>::dump() const {
            std::cerr << *this;
        }

        template <>
        const uint64_t DMATransfer<void *, CopyDescriptor>::MaxId =
                std::numeric_limits<uint64_t>::max();

        template <>
        Atomic<uint64_t> DMATransfer<void *, CopyDescriptor>::_freeId = 0;

        template <>
        void DMATransfer<CopyDescriptor, void *>::dump() const {
            std::cerr << *this;
        }

        template <>
        const uint64_t DMATransfer<CopyDescriptor, void *>::MaxId =
                std::numeric_limits<uint64_t>::max();

        template <>
        Atomic<uint64_t> DMATransfer<CopyDescriptor, void *>::_freeId = 0;

    } // End namespace ext.
} // End namespace nanos.

//
// OCLDMA implementation.
//

bool OCLDMA::copyIn(void *localDst, CopyDescriptor &remoteSrc, size_t size) {
    LockBlock lock(_lock);

    _ins.push(DMAInTransfer(localDst, remoteSrc, size));

    return false;
}

bool OCLDMA::copyOut(CopyDescriptor &remoteDst, void *localSrc, size_t size) {
    LockBlock lock(_lock);

    _outs.push(DMAOutTransfer(remoteDst, localSrc, size));

    return false;
}

void OCLDMA::syncTransfer(uint64_t hostAddress) {
    LockBlock lock(_lock);

    _ins.prioritize(hostAddress);
    _outs.prioritize(hostAddress);
}

void OCLDMA::execTransfers() {
    LockBlock lock(_lock);

    // TODO: select a smarter policy.
    while (!_ins.empty()) {
        const DMAInTransfer &trans = _ins.top();
        CopyDescriptor &src = trans.getSource();

        _proc.copyIn(trans.getDestination(), src, trans.getSize());
        _proc.synchronize(src);

        _ins.pop();
    }

    // TODO: select a smarted policy.
    while (!_outs.empty()) {
        const DMAOutTransfer &trans = _outs.top();
        CopyDescriptor &dst = trans.getDestination();

        _proc.copyOut(dst, trans.getSource(), trans.getSize());
        _proc.synchronize(dst);

        _outs.pop();
    }
}

std::ostream &nanos::ext::operator<<(std::ostream &os, size_t size) {

    os << "### Size ###"
            << std::endl
            << "  RawSize: " 
            << std::endl
            << "  Operation: " << "DEVICE"
            << std::endl;

    return os;
}

std::ostream &
nanos::ext::operator<<(std::ostream &os,
        const DMATransfer<void *, CopyDescriptor> &trans) {
    const CopyDescriptor &src = trans.getSource();

    void *srcAddr = reinterpret_cast<void *> (src.getTag());
    unsigned version = src.getDirectoryVersion();

    os << "### DMA Input Transfer ###" << std::endl
            << "  To: " << trans.getDestination() << std::endl
            << "  From: [" << version << "] " << srcAddr << std::endl;

    return os;
}

std::ostream &
nanos::ext::operator<<(std::ostream &os,
        const DMATransfer<CopyDescriptor, void *> &trans) {
    const CopyDescriptor &dst = trans.getDestination();

    void *dstAddr = reinterpret_cast<void *> (dst.getTag());
    unsigned version = dst.getDirectoryVersion();

    os << "### DMA Output Transfer ###" << std::endl
            << "  To: [" << version << "] " << dstAddr << std::endl
            << "  From: " << trans.getSource() << std::endl;

    return os;
}
