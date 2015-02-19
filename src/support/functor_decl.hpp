#ifndef FUNCTOR_DECL
#define FUNCTOR_DECL

class Functor {
   public:
      virtual ~Functor() { };
      virtual void operator()() = 0;
};

#endif /* FUNCTOR_DECL */
