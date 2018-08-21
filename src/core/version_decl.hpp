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
         void resetVersion();
   };

} // namespace nanos

#endif /* VERSION_DECL_HPP */
