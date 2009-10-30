#ifndef __NANOS_CUTOFF_H
#define __NANOS_CUTOFF_H



namespace nanos
{

   class cutoff
   {

      public:
         cutoff() {};

         virtual void init() = 0;
         virtual bool cutoff_pred() = 0;
         virtual ~cutoff() {}
   };



}

#endif
