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

/*
<testinfo>
test_generator=gens/api-omp-generator
</testinfo>
*/

#include "nanos.h"
#include "omp.h"

struct  nanos_const_wd_definition_1
{
  nanos_const_wd_definition_t base;
  nanos_device_t devices[1];
};

struct  nanos_args_1_t
{
  int *i;
};

static void smp_ol_main_1(struct nanos_args_1_t *const args);

int main()
{
  int i;
  {
    nanos_err_t err;
    nanos_wd_dyn_props_t dyn_props;
    unsigned int nth_i;
    struct nanos_args_1_t imm_args;
    nanos_data_access_t dependences[1];
    static nanos_smp_args_t smp_ol_main_1_args = {.outline = (void (*)(void *))(void (*)(struct nanos_args_1_t *))&smp_ol_main_1};
    static struct nanos_const_wd_definition_1 nanos_wd_const_data = {.base = {.props = {.mandatory_creation = 1, .tied = 1, .clear_chunk = 0, .reserved0 = 0, .reserved1 = 0, .reserved2 = 0, .reserved3 = 0, .reserved4 = 0}, .data_alignment = __alignof__(struct nanos_args_1_t), .num_copies = 0, .num_devices = 1, .num_dimensions = 0, .description = 0}, .devices = {[0] = {.factory = &nanos_smp_factory, .arg = &smp_ol_main_1_args}}};
    unsigned int nanos_num_threads = nanos_omp_get_num_threads_next_parallel(0);
    nanos_team_t nanos_team = (nanos_team_t)0;
    nanos_thread_t nanos_team_threads[nanos_num_threads];
    err = nanos_create_team(&nanos_team, (nanos_sched_t)0, &nanos_num_threads, (nanos_constraint_t *)0, 1, nanos_team_threads, NULL );
    if (err != NANOS_OK)
      {
        nanos_handle_error(err);
      }
    dyn_props.tie_to = (nanos_thread_t)0;
    dyn_props.priority = 0;
    dyn_props.flags.is_final = 0;
    for (nth_i = 1; nth_i < nanos_num_threads; nth_i = nth_i + 1)
      {
        dyn_props.tie_to = nanos_team_threads[nth_i];
        struct nanos_args_1_t *ol_args = 0;
        nanos_wd_t nanos_wd_ = (nanos_wd_t)0;
        err = nanos_create_wd_compact(&nanos_wd_, &nanos_wd_const_data.base, &dyn_props, sizeof(struct nanos_args_1_t), (void **)&ol_args, nanos_current_wd(), (nanos_copy_data_t **)0, (nanos_region_dimension_internal_t **)0);
        if (err != NANOS_OK)
          {
            nanos_handle_error(err);
          }
        (*ol_args).i = &i;
        err = nanos_submit(nanos_wd_, 0, (nanos_data_access_t *)0, (nanos_team_t)0);
        if (err != NANOS_OK)
          {
            nanos_handle_error(err);
          }
      }
    dyn_props.tie_to = nanos_team_threads[0];
    imm_args.i = &i;
    err = nanos_create_wd_and_run_compact(&nanos_wd_const_data.base, &dyn_props, sizeof(struct nanos_args_1_t), &imm_args, 0, dependences, (nanos_copy_data_t *)0, (nanos_region_dimension_internal_t *)0, (nanos_translate_args_t)0);
    if (err != NANOS_OK)
      {
        nanos_handle_error(err);
      }
    err = nanos_end_team(nanos_team);
    if (err != NANOS_OK)
      {
        nanos_handle_error(err);
      }
  }
  return 0;
}

static void smp_ol_main_0_unpacked(nanos_ws_desc_t *wsd_1)
{
  int i;
  {
    {
      nanos_err_t err;
      err = nanos_omp_set_implicit(nanos_current_wd());
      if (err != NANOS_OK)
        {
          nanos_handle_error(err);
        }
    }
    {
      nanos_err_t err;
      nanos_ws_item_loop_t nanos_item_loop;
      err = nanos_worksharing_next_item(wsd_1, (void **)&nanos_item_loop);
      if (err != NANOS_OK)
        {
          nanos_handle_error(err);
        }
      while (nanos_item_loop.execute)
        {
          for (i = nanos_item_loop.lower; i <= nanos_item_loop.upper; i += 1)
            {
              {
              }
            }
          ;
          err = nanos_worksharing_next_item(wsd_1, (void **)&nanos_item_loop);
        }
    }
  }
}

struct  nanos_args_0_t
{
  nanos_ws_desc_t *wsd_1;
};

static void smp_ol_main_0(struct nanos_args_0_t *const args)
{
  {
    smp_ol_main_0_unpacked((*args).wsd_1);
  }
}

static void smp_ol_main_1_unpacked(int *const i)
{
  {
    nanos_err_t err;
    err = nanos_omp_set_implicit(nanos_current_wd());
    if (err != NANOS_OK)
      {
        nanos_handle_error(err);
      }
    err = nanos_enter_team();
    if (err != NANOS_OK)
      {
        nanos_handle_error(err);
      }
    {
      int nanos_chunk;
      nanos_ws_info_loop_t nanos_setup_info_loop;
      nanos_err_t err;
      nanos_ws_desc_t *wsd_1;
      _Bool single_guard;
      struct nanos_args_0_t imm_args;
      void *current_ws_policy = nanos_omp_find_worksharing(nanos_omp_sched_static);
      if (current_ws_policy == 0)
        {
          nanos_handle_error(NANOS_UNIMPLEMENTED);
        }
      nanos_chunk = 0;
      nanos_setup_info_loop.lower_bound = 0;
      nanos_setup_info_loop.upper_bound = 9;
      nanos_setup_info_loop.loop_step = 1;
      nanos_setup_info_loop.chunk_size = nanos_chunk;
      err = nanos_worksharing_create(&wsd_1, current_ws_policy, (void **)&nanos_setup_info_loop, &single_guard);
      if (err != NANOS_OK)
        {
          nanos_handle_error(err);
        }
      if (single_guard)
        {
          int sup_threads;
          err = nanos_team_get_num_supporting_threads(&sup_threads);
          if (err != NANOS_OK)
            {
              nanos_handle_error(err);
            }
          if (sup_threads > 0)
            {
              nanos_wd_dyn_props_t dyn_props;
              err = nanos_malloc((void **)&(*wsd_1).threads, sizeof(void *) * sup_threads, "", 0);
              if (err != NANOS_OK)
                {
                  nanos_handle_error(err);
                }
              err = nanos_team_get_supporting_threads(&(*wsd_1).nths, (*wsd_1).threads);
              if (err != NANOS_OK)
                {
                  nanos_handle_error(err);
                }
              struct nanos_args_0_t *ol_args = (struct nanos_args_0_t *)0;
              static nanos_smp_args_t smp_ol_main_0_args = {.outline = (void (*)(void *))(void (*)(struct nanos_args_0_t *))&smp_ol_main_0};
              static struct nanos_const_wd_definition_1 nanos_wd_const_data = {.base = {.props = {.mandatory_creation = 1, .tied = 1, .clear_chunk = 0, .reserved0 = 0, .reserved1 = 0, .reserved2 = 0, .reserved3 = 0, .reserved4 = 0}, .data_alignment = __alignof__(struct nanos_args_0_t), .num_copies = 0, .num_devices = 1, .num_dimensions = 0, .description = 0}, .devices = {[0] = {.factory = &nanos_smp_factory, .arg = &smp_ol_main_0_args}}};
              void *nanos_wd_ = (void *)0;
              dyn_props.tie_to = (void *)0;
              dyn_props.priority = 0;
              dyn_props.flags.is_final = 0;
              static void *replicate = (void *)0;
              if (replicate == (void *)0)
                {
                  replicate = nanos_find_slicer("replicate");
                }
              if (replicate == (void *)0)
                {
                  nanos_handle_error(NANOS_UNIMPLEMENTED);
                }
              err = nanos_create_sliced_wd(&nanos_wd_, nanos_wd_const_data.base.num_devices, nanos_wd_const_data.devices, (size_t)sizeof(struct nanos_args_0_t), nanos_wd_const_data.base.data_alignment, (void **)&ol_args, (void **)0, replicate, &nanos_wd_const_data.base.props, &dyn_props, 0, (nanos_copy_data_t **)0, 0, (nanos_region_dimension_internal_t **)0);
              if (err != NANOS_OK)
                {
                  nanos_handle_error(err);
                }
              (*ol_args).wsd_1 = wsd_1;
              err = nanos_submit(nanos_wd_, 0, (nanos_data_access_t *)0, (void *)0);
              if (err != NANOS_OK)
                {
                  nanos_handle_error(err);
                }
              err = nanos_free((*wsd_1).threads);
              if (err != NANOS_OK)
                {
                  nanos_handle_error(err);
                }
            }
        }
      imm_args.wsd_1 = wsd_1;
      smp_ol_main_0(&(imm_args));
    }
    err = nanos_omp_barrier();
    if (err != NANOS_OK)
      {
        nanos_handle_error(err);
      }
    err = nanos_leave_team();
    if (err != NANOS_OK)
      {
        nanos_handle_error(err);
      }
  }
}
static void smp_ol_main_1(struct nanos_args_1_t *const args)
{
  {
    smp_ol_main_1_unpacked((*args).i);
  }
}
