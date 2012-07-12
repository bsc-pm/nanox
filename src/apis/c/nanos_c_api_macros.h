#ifndef NANOS_API_MACROS_H
#define NANOS_API_MACROS_H

#define NANOS_API_DECL(Type, Name, Params) \
    extern Type Name##_ Params; \
    extern Type Name Params

#ifdef _NANOS_INTERNAL

   #define NANOS_API_DEF(Type, Name, Params) \
       __attribute__((alias(#Name))) Type Name##_ Params; \
       Type Name Params

#endif

#endif // NANOS_API_MACROS_H
