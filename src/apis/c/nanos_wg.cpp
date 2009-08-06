#include "workgroup.hpp"
#include "nanos.hpp"

using namespace nanos;

nanos_err_t nanos_wg_wait_completation ( nanos_wg_t wg )
{
  try {
    WG *wg = (WG *)uwg;
    wg->waitCompletation();
  } catch (...) {
    return NANOS_UNKNOWN_ERR;
  }
  return NANOS_OK;
}
