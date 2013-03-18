#ifndef VERSION_DECL_HPP
#define VERSION_DECL_HPP
namespace nanos {
   class Version {
      private:
         unsigned int _version;
      public:
         Version();
         Version( Version const & ver );
         Version( unsigned int v );
         ~Version();
         Version &operator=( Version const & ver );
         unsigned int getVersion() const;
         unsigned int getVersion(bool increaseVersion);
         void setVersion( unsigned int version );
   };
}
#endif /* VERSION_DECL_HPP */
