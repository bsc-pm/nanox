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
test_generator="gens/api-generator -d plain,regions,perfect-regions"
</testinfo>
*/
#include <nanos.h>

void task(int *a, int *b);
void task(int *a, int *b)
{
}
typedef struct _nx_data_env_0_t_tag
{
        int *__tmp_0_0;
        int *__tmp_1_0;
} _nx_data_env_0_t;
int main();
static void _smp__ol_main_0(_nx_data_env_0_t *_args)
{
    {
        task((_args->__tmp_0_0), (_args->__tmp_1_0));
    }
}

struct nanos_const_wd_definition_1
{
     nanos_const_wd_definition_t base;
     nanos_device_t devices[1];
};

struct nanos_const_wd_definition_1 const_data_1 = 
{
   {{
      .mandatory_creation = 1,
      .tied = 1
   },
   0,//__alignof__(section_data_1),
   0,
   1,
   0,NULL},
   {
      {
         nanos_smp_factory,
         0
      }
   }
};
int main ( int argc, char **argv )
{
    int a, b, i;
    for (i = 0;
        i < 100;
        i++)
    {
        {
            int *__tmp_0 = &a;
            int *__tmp_1 = &b;
            {
                /* SMP device descriptor */
                nanos_smp_args_t _ol_main_0_smp_args = {(void (*)(void *)) _smp__ol_main_0};
                /*nanos_device_t _ol_main_0_devices[] = {{
                    nanos_smp_factory,
                    &_ol_main_0_smp_args
                }};*/
                _nx_data_env_0_t *ol_args = (_nx_data_env_0_t *) 0;
                nanos_wd_t wd = (nanos_wd_t) 0;
                nanos_wd_dyn_props_t dyn_props = { .tie_to = 0, .priority = 0 };
                __builtin_memset(&const_data_1.base.props, 0, sizeof (const_data_1.base.props));
                const_data_1.base.data_alignment =  __alignof__(_nx_data_env_0_t);
                const_data_1.devices[0].arg = &_ol_main_0_smp_args;
                nanos_err_t err;
                err = nanos_create_wd_compact(&wd, &const_data_1.base, &dyn_props, sizeof(_nx_data_env_0_t), (void **) &ol_args, nanos_current_wd(), (nanos_copy_data_t **) 0, NULL);
                //err = nanos_create_wd(&wd, 1, _ol_main_0_devices, sizeof(_nx_data_env_0_t), __alignof__(_nx_data_env_0_t), (void **) &ol_args, nanos_current_wd(), &props, 0, (nanos_copy_data_t **) 0);
                if (err != NANOS_OK)
                    nanos_handle_error(err);
                if (wd != (nanos_wd_t) 0)
                {
                    ol_args->__tmp_0_0 = __tmp_0;
                    ol_args->__tmp_1_0 = __tmp_1;
                    nanos_region_dimension_t dimensions0[1] = {
                        {
                            ((char *) ((__tmp_0)) - (char *) ol_args->__tmp_0_0)*sizeof(int),
                            0,
                            ((char *) ((__tmp_0)) - (char *) ol_args->__tmp_0_0)*sizeof(int)
                        },
                    };
                    nanos_region_dimension_t dimensions1[1] = {
                        {
                            ((char *) ((__tmp_1)) - (char *) ol_args->__tmp_1_0)*sizeof(int),
                            0,
                            ((char *) ((__tmp_1)) - (char *) ol_args->__tmp_1_0)*sizeof(int)
                        },
                    };
                    nanos_data_access_t _data_accesses[2] = {
                        {
                            (void *) ol_args->__tmp_0_0,
                            {
                                1,
                                1,
                                1,
                                0,
                                0
                            },
                            1,
                            dimensions0,
                            0
                        },
                        {
                            (void *) ol_args->__tmp_1_0,
                            {
                                1,
                                1,
                                1,
                                0,
                                0
                            },
                            1,
                            dimensions1,
                            0
                        }
                    };
                    err = nanos_submit(wd, 2, (nanos_data_access_t *) _data_accesses, (nanos_team_t) 0);
                    if (err != NANOS_OK)
                        nanos_handle_error(err);
                }
                else
                {
                    _nx_data_env_0_t imm_args;
                    imm_args.__tmp_0_0 = __tmp_0;
                    imm_args.__tmp_1_0 = __tmp_1;
                    nanos_region_dimension_t dimensions0[1] = {
                        {
                            ((char *) ((__tmp_0)) - (char *) imm_args.__tmp_0_0)*sizeof(int),
                            0,
                            ((char *) ((__tmp_0)) - (char *) imm_args.__tmp_0_0)*sizeof(int)
                        },
                    };
                    nanos_region_dimension_t dimensions1[1] = {
                        {
                            ((char *) ((__tmp_1)) - (char *) imm_args.__tmp_1_0)*sizeof(int),
                            0,
                            ((char *) ((__tmp_1)) - (char *) imm_args.__tmp_1_0)*sizeof(int)
                        },
                    };
                    nanos_data_access_t _data_accesses[2] = {
                        {
                            (void *) &imm_args.__tmp_0_0,
                            {
                                1,
                                1,
                                1,
                                0,
                                0
                            },
                            1,
                            dimensions0,
                            0
                        },
                        {
                            (void *) &imm_args.__tmp_1_0,
                            {
                                1,
                                1,
                                1,
                                0,
                                0
                            },
                            1,
                            dimensions1,
                            0
                        }
                    };
                    err = nanos_create_wd_and_run_compact(&const_data_1.base, &dyn_props, sizeof(_nx_data_env_0_t), &imm_args, 2, (nanos_data_access_t *) _data_accesses, (nanos_copy_data_t *) 0, 0, NULL );
                    //err = nanos_create_wd_and_run(1, _ol_main_0_devices, sizeof(_nx_data_env_0_t), __alignof__(_nx_data_env_0_t),  &imm_args, &props, 0, (nanos_copy_data_t *) 0);
                    if (err != NANOS_OK)
                        nanos_handle_error(err);
                }
            }
        }
    }
    nanos_wg_wait_completion( nanos_current_wd(), 0 );
    return 0;
}
