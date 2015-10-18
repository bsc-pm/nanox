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

#include "dbmanager.hpp"
#include "debug.hpp"
#include "os.hpp"
#include "config.hpp"

using namespace nanos;

DbManager::~DbManager()
{
  closeConnection();
}

bool DbManager::openConnection(std::string databaseName)
{
  if ( _isOpen )
    closeConnection();

  const int err = sqlite3_open(databaseName.c_str(), &_db);
  if ( err ) {
    std::string errMsg = "Can't open database: ";
    errMsg += sqlite3_errmsg(_db);
    fatal0(errMsg);
    return false;
  }

  _isOpen = true;
  return true;
}

bool DbManager::openConnection()
{
  return openConnection(defaultDbName);
}

bool DbManager::closeConnection()
{
  if ( _isOpen ) {
    const int err = sqlite3_close(_db);

    if ( err != SQLITE_OK ) {
        std::string errMsg = "Can't close database: ";
        errMsg += sqlite3_errmsg(_db);
        fatal0(errMsg);
        return false;
    }

    _isOpen = false;
    return true;
  } else {
    return false;
  }
}

bool DbManager::executeStatement(std::string stmt, int (*callback)(void*,int,char**,char**), void *data)
{
  if ( _isOpen ) {
    char *sqliteErrMsg = 0;
    const int err = sqlite3_exec(_db, stmt.c_str(), callback, data, &sqliteErrMsg);

    if ( err != SQLITE_OK ) {
      std::string errMsg = "Can't execute statement: ";
      errMsg += sqliteErrMsg;
      fatal0(errMsg);

      sqlite3_free(sqliteErrMsg);
      return false;
    }
    return true;
  } else {
    return false;
  }
}
