
#include "oclcache.hpp"
#include "oclconfig.hpp"
#include "oclprocessor.hpp"

using namespace nanos;
using namespace nanos::ext;

const size_t Size::OperationMask = size_t(1);
const size_t Size::HostSizeMask = ~size_t(1);

const size_t Size::BufferIdMask = (1 << log2( getPageSize() ) ) - 2;
const size_t Size::DeviceSizeMask = ~(BufferIdMask | 1);
const size_t Size::DeviceLocalSizeMask = ~OperationMask;

Lock OCLMemDump::_dumpLock;

//
// Size implementation.
//

void Size::dump() {
    std::cerr << *this;
}

//
// OCLMemDump implementation.
//

void OCLMemDump::dumpCopyIn(void *localDst,
        CopyDescriptor &remoteSrc,
        size_t size) {
    dumpCopyIn(localDst, remoteSrc, Size(size));
}

void OCLMemDump::dumpCopyOut(CopyDescriptor &remoteDst,
        void *localSrc,
        size_t size) {
    dumpCopyOut(remoteDst, localSrc, Size(size));
}

void OCLMemDump::dumpCopyIn(void *localDst,
        CopyDescriptor &remoteSrc,
        Size size) {
    LockBlock lock(OCLMemDump::_dumpLock);

    std::cerr << "### COPY-IN ###" << std::endl
            << "  Dst = " << localDst
            << std::endl
            << "  Src = " << reinterpret_cast<void *> (remoteSrc.getTag())
            << std::endl
            << size;
}

void OCLMemDump::dumpCopyOut(CopyDescriptor &remoteDst,
        void *localSrc,
        Size size) {
    LockBlock lock(OCLMemDump::_dumpLock);

    std::cerr << "### COPY-OUT ###" << std::endl
            << "  Dst = " << reinterpret_cast<void *> (remoteDst.getTag())
            << std::endl
            << "  Src = " << localSrc
            << std::endl
            << size;
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

    // Compute the amount of cache for the host. If not given, use an arbitrary
    // big value -- we actually do not consume memory.
    _hostCacheSize = OCLConfig::getHostCacheSize();
    if (_hostCacheSize == 0)
        _hostCacheSize = std::numeric_limits<size_t>::max() -
        pageSize -
            _devCacheSize;
    _hostCacheSize = roundUpToPageSize(_hostCacheSize);

    // Initialize the host allocator.
    _hostAllocator.init(pageSize, _hostCacheSize);

    // Initialize the device allocator. It manages virtual addresses above the
    // addresses managed by host allocator.
    _devAllocator.init(pageSize + _hostCacheSize, _devCacheSize);
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

void *OCLCache::hostAllocate(Size size) {
    void *addr = _hostAllocator.allocate(size.getAllocSize());

    if (!addr)
        fatal("Host cache allocation failed");

    return addr;
}

void *OCLCache::deviceAllocate(Size size) {
    bool starSSMode = OCLConfig::getStarSSMode();
    cl_mem buf;

    size_t allocSize;
    if (starSSMode) {
        allocSize = size.getRaw();
    } else {
        allocSize = size.getAllocSize();
    }

    if (_oclAdapter.allocBuffer(allocSize, buf) != CL_SUCCESS)
        fatal("Device allocation failed");

    void *addr = _devAllocator.allocate(allocSize);

    if (!addr)
        fatal("Device cache allocation failed");

    if (starSSMode) {
        _allocBuffMappings[allocSize] = buf;
        _bufAddrMappings[addr] = buf;
    } else {
        _bufIdMappings[size.getId()] = buf;
        _bufAddrMappings[addr] = buf;
    }

    return addr;
}

void *OCLCache::hostReallocate(Address addr, Size size, Size ceSize) {
    // The host allocator is virtual. The straightforward approach is
    // sufficient for its needs.

    _hostAllocator.free(addr.getAllocAddr());

    return _hostAllocator.allocate(size.getAllocSize());
}

void *OCLCache::deviceReallocate(Address addr, Size size, Size ceSize) {
    // TODO: checking wheter we need a more optimized solution.
    
    //TODO: in starSS mode we could remove a buffer from the address-asigned buffers cache
    //and place it on the size-allocated buffers cache (should be faster 
    //and apparently result would be the same)
    //instead of recreating the buffer
    deviceFree(addr);

    return deviceAllocate(size);
}

void OCLCache::hostFree(Address addr) {
    _hostAllocator.free(addr.getAllocAddr());
}

void OCLCache::deviceFree(Address addr) {
    bool starSSMode = OCLConfig::getStarSSMode();
    

    if (starSSMode) {
        void *allocAddr = addr.getAllocAddr();
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

    } else {
        void *allocAddr = addr.getAllocAddr();
        cl_mem buf = _bufAddrMappings[allocAddr];
        for (std::map<unsigned, cl_mem>::iterator i = _bufIdMappings.begin(),
                e = _bufIdMappings.end();
                i != e;
                ++i) {
            if (i->second == buf) {
                if (_oclAdapter.freeBuffer(buf) != CL_SUCCESS)
                    fatal("Cannot free device buffer");

                _bufIdMappings.erase(i);
                break;
            }

            _devAllocator.free(allocAddr);
        }
    }

}

bool OCLCache::deviceCopyIn(void *localDst,
        CopyDescriptor &remoteSrc,
        Size size) {
    bool starSSMode = OCLConfig::getStarSSMode();
    void *src = reinterpret_cast<void *> (remoteSrc.getTag());

    unsigned id;

    size_t rightSize;
    if (starSSMode) {
        rightSize = size.getRaw();
    } else {
        id = size.getId();
        rightSize = size.getAllocSize();
    }


    cl_int errCode;
    cl_mem buf;
    if (starSSMode) {
        //Find unused allocated buffer and remove from unused
        bool found=false;
        std::map<size_t, cl_mem>::iterator iter;
        for (iter = _allocBuffMappings.begin(); iter != _allocBuffMappings.end(); iter++) {
            if (iter->first == rightSize) {
                buf = iter->second;
                _allocBuffMappings.erase(iter);
                found=true;
                break;
            }
        }
        if (found){        
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
    } else {
        buf=_bufIdMappings[id];
    }
    errCode = _oclAdapter.writeBuffer(buf,
              src,
              0,
              rightSize);
    if (errCode != CL_SUCCESS)
        fatal("Buffer writing failed");

    return true;
}

bool OCLCache::deviceCopyOut(CopyDescriptor &remoteDst,
        void *localSrc,
        Size size) {
    bool starSSMode = OCLConfig::getStarSSMode();
    void *dst = reinterpret_cast<void *> (remoteDst.getTag());

    // Local buffer, do not perform copy.
    if (!dst)
        return true;

    unsigned id;
    size_t rightSize;
    if (starSSMode) {
        rightSize = size.getRaw();
    } else {
        id = size.getId();
        rightSize = size.getAllocSize();
    }

    cl_int errCode;
    cl_mem buffer;
    if (starSSMode) {
        buffer=_devBufAddrMappings[dst];
    } else {
        buffer=_bufIdMappings[id];
    }
    
    errCode = _oclAdapter.readBuffer(buffer,
                dst,
                0,
                rightSize);
    
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

std::ostream &nanos::ext::operator<<(std::ostream &os, Size size) {
    bool hostOperation = size.isHostOperation();

    os << "### Size ###"
            << std::endl
            << "  RawSize: " << size.getRaw()
            << std::endl
            << "  Operation: " << (hostOperation ? "HOST" : "DEVICE")
            << std::endl
            << "  AllocSize: " << size.getAllocSize()
            << std::endl;

    if (!hostOperation)
        os << "  BufferId: " << size.getId() << std::endl;

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
            << "  From: [" << version << "] " << srcAddr << std::endl
            << "  Size: " << trans.getSize() << std::endl;

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
            << "  From: " << trans.getSource() << std::endl
            << "  Size: " << trans.getSize() << std::endl;

    return os;
}
