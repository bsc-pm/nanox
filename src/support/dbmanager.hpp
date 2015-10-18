/*************************************************************************************/
/*      Copyright 2015 Barcelona Supercomputing Center                               */
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
/*      along with NANOS++.  If not, see <http://www.gnu.org/licenses/>.             */
/*************************************************************************************/

#ifndef DBMANAGER_HPP_
#define DBMANAGER_HPP_

#include <sqlite3.h>
#include <string>

namespace nanos {

const std::string defaultDbName = "nanos.db";

  class DbManager {
    sqlite3 *_db;
    bool _isOpen;
  public:
    DbManager() : _isOpen(false) {}
    ~DbManager();
    bool openConnection(std::string databaseName);
    bool openConnection();
    bool closeConnection();
    bool executeStatement(std::string stmt, int (*callback)(void*,int,char**,char**), void *data = NULL);
  };

}

#endif /* DBMANAGER_HPP_ */
