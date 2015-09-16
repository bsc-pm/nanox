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

/*
<testinfo>
test_generator=gens/opencl-generator
test_generator_ENV='test_architecture=smp'
test_schedule=bf
</testinfo>
*/

struct _IO_FILE_plus;
extern struct _IO_FILE_plus _IO_2_1_stdin_;
extern struct _IO_FILE_plus _IO_2_1_stdout_;
extern struct _IO_FILE_plus _IO_2_1_stderr_;
struct _IO_FILE;
extern struct _IO_FILE *stdin;
extern struct _IO_FILE *stdout;
extern struct _IO_FILE *stderr;
extern int sys_nerr;
extern const char *const sys_errlist[];
unsigned int workgroup_size = 8;
struct  nanos_args_0_t
{
  int n;
  int *x;
  int mcc_ndrange_2_0;
  int mcc_ndrange_2_1;
  int mcc_ndrange_2_2;
};
enum mcc_enum_anon_5
{
  NANOS_OK = 0,
  NANOS_UNKNOWN_ERR = 1,
  NANOS_UNIMPLEMENTED = 2,
  NANOS_ENOMEM = 3,
  NANOS_INVALID_PARAM = 4,
  NANOS_INVALID_REQUEST = 5
};
typedef enum mcc_enum_anon_5 nanos_err_t;
typedef unsigned int nanos_copy_id_t;
typedef void *nanos_wd_t;
extern nanos_err_t nanos_get_addr(nanos_copy_id_t copy_id, void **addr, nanos_wd_t cwd);
extern void nanos_handle_error(nanos_err_t err);
static void nanos_xlate_fun_profilingtestc_0(struct nanos_args_0_t *const arg, void *wd)
{
  {
    void *device_base_address;
    nanos_err_t nanos_err;
    device_base_address = 0;
    nanos_err = nanos_get_addr(0, &device_base_address, wd);
    if (nanos_err != NANOS_OK)
      {
        nanos_handle_error(nanos_err);
      }
    (*arg).x = (int *)device_base_address;
  }
}
typedef unsigned long int size_t;
struct  mcc_struct_anon_19
{
  void (*outline)(void *);
};
typedef struct mcc_struct_anon_19 nanos_opencl_args_t;
static void ocl_ol_init_1(struct nanos_args_0_t *const args);
struct  mcc_struct_anon_11
{
  _Bool mandatory_creation:1;
  _Bool tied:1;
  _Bool clear_chunk:1;
  _Bool reserved0:1;
  _Bool reserved1:1;
  _Bool reserved2:1;
  _Bool reserved3:1;
  _Bool reserved4:1;
};
typedef struct mcc_struct_anon_11 nanos_wd_props_t;
struct  nanos_const_wd_definition_tag
{
  nanos_wd_props_t props;
  size_t data_alignment;
  size_t num_copies;
  size_t num_devices;
  size_t num_dimensions;
  const char *description;
};
typedef struct nanos_const_wd_definition_tag nanos_const_wd_definition_t;
struct  mcc_struct_anon_14
{
  void *(*factory)(void *);
  void *arg;
};
typedef struct mcc_struct_anon_14 nanos_device_t;
struct  nanos_const_wd_definition_1
{
  nanos_const_wd_definition_t base;
  nanos_device_t devices[1L];
};
extern void *nanos_opencl_factory(void *args);
struct  mcc_struct_anon_12
{
  _Bool is_final:1;
  _Bool is_recover:1;
  _Bool is_implicit:1;
  _Bool reserved3:1;
  _Bool reserved4:1;
  _Bool reserved5:1;
  _Bool reserved6:1;
  _Bool reserved7:1;
};
typedef struct mcc_struct_anon_12 nanos_wd_dyn_flags_t;
typedef void *nanos_thread_t;
struct  mcc_struct_anon_13
{
  nanos_wd_dyn_flags_t flags;
  nanos_thread_t tie_to;
  int priority;
};
typedef struct mcc_struct_anon_13 nanos_wd_dyn_props_t;
struct mcc_struct_anon_4;
typedef struct mcc_struct_anon_4 nanos_copy_data_internal_t;
typedef nanos_copy_data_internal_t nanos_copy_data_t;
struct mcc_struct_anon_0;
typedef struct mcc_struct_anon_0 nanos_region_dimension_internal_t;
typedef void *nanos_wg_t;
extern nanos_err_t nanos_create_wd_compact(nanos_wd_t *wd, nanos_const_wd_definition_t *const_data, nanos_wd_dyn_props_t *dyn_props, size_t data_size, void **data, nanos_wg_t wg, nanos_copy_data_t **copies, nanos_region_dimension_internal_t **dimensions);
extern nanos_wd_t nanos_current_wd(void);
struct  mcc_struct_anon_0
{
  size_t size;
  size_t lower_bound;
  size_t accessed_length;
};
typedef nanos_region_dimension_internal_t nanos_region_dimension_t;
struct  mcc_struct_anon_1
{
  _Bool input:1;
  _Bool output:1;
  _Bool can_rename:1;
  _Bool concurrent:1;
  _Bool commutative:1;
};
typedef struct mcc_struct_anon_1 nanos_access_type_internal_t;
typedef long int ptrdiff_t;
struct  mcc_struct_anon_2
{
  void *address;
  nanos_access_type_internal_t flags;
  short int dimension_count;
  const nanos_region_dimension_internal_t *dimensions;
  ptrdiff_t offset;
};
typedef struct mcc_struct_anon_2 nanos_data_access_internal_t;
typedef nanos_data_access_internal_t nanos_data_access_t;
enum mcc_enum_anon_0
{
  NANOS_PRIVATE = 0,
  NANOS_SHARED = 1
};
typedef enum mcc_enum_anon_0 nanos_sharing_t;
struct  mcc_struct_anon_5
{
  _Bool input:1;
  _Bool output:1;
};
typedef unsigned long int uint64_t;
typedef unsigned int memory_space_id_t;
struct  mcc_struct_anon_4
{
  void *address;
  nanos_sharing_t sharing;
  struct mcc_struct_anon_5 flags;
  short int dimension_count;
  nanos_region_dimension_internal_t *dimensions;
  ptrdiff_t offset;
  uint64_t host_base_address;
  memory_space_id_t host_region_id;
  _Bool remote_host;
};
typedef void (*nanos_translate_args_t)(void *, nanos_wd_t);
extern nanos_err_t nanos_set_translate_function(nanos_wd_t wd, nanos_translate_args_t translate_args);
typedef void *nanos_team_t;
extern nanos_err_t nanos_submit(nanos_wd_t wd, size_t num_data_accesses, nanos_data_access_t *data_accesses, nanos_team_t team);
extern nanos_err_t nanos_create_wd_and_run_compact(nanos_const_wd_definition_t *const_data, nanos_wd_dyn_props_t *dyn_props, size_t data_size, void *data, size_t num_data_accesses, nanos_data_access_t *data_accesses, nanos_copy_data_t *copies, nanos_region_dimension_internal_t *dimensions, nanos_translate_args_t translate_args);
extern nanos_err_t nanos_wg_wait_completion(nanos_wg_t wg, _Bool avoid_flush);
typedef struct _IO_FILE FILE;
extern int fprintf(FILE *__restrict __stream, const char *__restrict __format, ...);
extern struct _IO_FILE *stderr;
int main(int argc, char **argv)
{
  int i;
  size_t n = 1024;
  int x[n];
  {
    int mcc_ndrange_2_0 = n;
    int mcc_ndrange_2_1 = n;
    int mcc_ndrange_2_2 = n;
    int mcc_arg_0 = n;
    int *mcc_arg_1 = x;
    {
      static nanos_opencl_args_t ocl_ol_init_1_args;
      nanos_wd_dyn_props_t nanos_wd_dyn_props;
      struct nanos_args_0_t *ol_args;
      nanos_err_t nanos_err;
      struct nanos_args_0_t imm_args;
      nanos_region_dimension_t dimensions_0[1L];
      nanos_data_access_t dependences[1L];
       /* OpenCL device descriptor */ 
      ocl_ol_init_1_args.outline = (void (*)(void *))ocl_ol_init_1;
      static struct nanos_const_wd_definition_1 nanos_wd_const_data = {.base = {.props = {.mandatory_creation = 1, .tied = 0, .clear_chunk = 0, .reserved0 = 0, .reserved1 = 0, .reserved2 = 0, .reserved3 = 0, .reserved4 = 0}, .data_alignment = __alignof__(struct nanos_args_0_t), .num_copies = 1, .num_devices = 1, .num_dimensions = 1, .description = 0}, .devices = {[0] = {.factory = &nanos_opencl_factory, .arg = &ocl_ol_init_1_args}}};
      nanos_wd_dyn_props.tie_to = 0;
      nanos_wd_dyn_props.priority = 0;
      nanos_wd_dyn_props.flags.is_final = 0;
      nanos_wd_dyn_props.flags.is_implicit = 0;
      ol_args = (struct nanos_args_0_t *)0;
      nanos_wd_t nanos_wd_ = (void *)0;
      nanos_copy_data_t *ol_copy_data = (nanos_copy_data_t *)0;
      nanos_region_dimension_internal_t *ol_copy_dimensions = (nanos_region_dimension_internal_t *)0;
      nanos_err = nanos_create_wd_compact(&nanos_wd_, &nanos_wd_const_data.base, &nanos_wd_dyn_props, sizeof(struct nanos_args_0_t), (void **)&ol_args, nanos_current_wd(), &ol_copy_data, &ol_copy_dimensions);
      if (nanos_err != NANOS_OK)
        {
          nanos_handle_error(nanos_err);
        }
      dimensions_0[0].size = (((mcc_arg_0) - 1L - 0L) + 1L) * sizeof(int);
      dimensions_0[0].lower_bound = (0L - 0L) * sizeof(int);
      dimensions_0[0].accessed_length = ((mcc_arg_0) - 1L - 0L - (0L - 0L) + 1) * sizeof(int);
      dependences[0].offset = 0L;
      dependences[0].flags.input = 0;
      dependences[0].flags.output = 1;
      dependences[0].flags.can_rename = 0;
      dependences[0].flags.concurrent = 0;
      dependences[0].flags.commutative = 0;
      dependences[0].dimension_count = 1;
      dependences[0].address = (void *)mcc_arg_1;
      dependences[0].dimensions = dimensions_0;
      if (nanos_wd_ != (void *)0)
        {
          (*ol_args).n = mcc_arg_0;
          (*ol_args).x = mcc_arg_1;
          (*ol_args).mcc_ndrange_2_0 = mcc_ndrange_2_0;
          (*ol_args).mcc_ndrange_2_1 = mcc_ndrange_2_1;
          (*ol_args).mcc_ndrange_2_2 = mcc_ndrange_2_2;
          ol_copy_dimensions[0 + 0].size = (((mcc_arg_0) - 1L - 0L) + 1L) * sizeof(int);
          ol_copy_dimensions[0 + 0].lower_bound = (0L - 0L) * sizeof(int);
          ol_copy_dimensions[0 + 0].accessed_length = ((mcc_arg_0) - 1L - 0L - (0L - 0L) + 1) * sizeof(int);
          ol_copy_data[0].sharing = NANOS_SHARED;
          ol_copy_data[0].address = (void *)mcc_arg_1;
          ol_copy_data[0].flags.input = 0;
          ol_copy_data[0].flags.output = 1;
          ol_copy_data[0].dimension_count = (short int)1;
          ol_copy_data[0].dimensions = &ol_copy_dimensions[0];
          ol_copy_data[0].offset = 0L;
          nanos_err = nanos_set_translate_function(nanos_wd_, (void (*)(void *, nanos_wd_t))nanos_xlate_fun_profilingtestc_0);
          if (nanos_err != NANOS_OK)
            {
              nanos_handle_error(nanos_err);
            }
          nanos_err = nanos_submit(nanos_wd_, 1, &dependences[0], (void *)0);
          if (nanos_err != NANOS_OK)
            {
              nanos_handle_error(nanos_err);
            }
        }
      else
        {
          nanos_region_dimension_internal_t imm_copy_dimensions[1L];
          nanos_copy_data_t imm_copy_data[1L];
          imm_args.n = mcc_arg_0;
          imm_args.x = mcc_arg_1;
          imm_args.mcc_ndrange_2_0 = mcc_ndrange_2_0;
          imm_args.mcc_ndrange_2_1 = mcc_ndrange_2_1;
          imm_args.mcc_ndrange_2_2 = mcc_ndrange_2_2;
          imm_copy_dimensions[0 + 0].size = (((mcc_arg_0) - 1L - 0L) + 1L) * sizeof(int);
          imm_copy_dimensions[0 + 0].lower_bound = (0L - 0L) * sizeof(int);
          imm_copy_dimensions[0 + 0].accessed_length = ((mcc_arg_0) - 1L - 0L - (0L - 0L) + 1) * sizeof(int);
          imm_copy_data[0].sharing = NANOS_SHARED;
          imm_copy_data[0].address = (void *)mcc_arg_1;
          imm_copy_data[0].flags.input = 0;
          imm_copy_data[0].flags.output = 1;
          imm_copy_data[0].dimension_count = (short int)1;
          imm_copy_data[0].dimensions = &imm_copy_dimensions[0];
          imm_copy_data[0].offset = 0L;
          nanos_err = nanos_create_wd_and_run_compact(&nanos_wd_const_data.base, &nanos_wd_dyn_props, sizeof(struct nanos_args_0_t), &imm_args, 1, &dependences[0], imm_copy_data, imm_copy_dimensions, (void (*)(void *, nanos_wd_t))nanos_xlate_fun_profilingtestc_0);
          if (nanos_err != NANOS_OK)
            {
              nanos_handle_error(nanos_err);
            }
        }
    }
  }
  {
    nanos_err_t nanos_err;
    nanos_wd_t nanos_wd_ = nanos_current_wd();
    nanos_err = nanos_wg_wait_completion(nanos_wd_, 0);
    if (nanos_err != NANOS_OK)
      {
        nanos_handle_error(nanos_err);
      }
  }
  for (i = 0; i < 10; i++)
    {
      fprintf(stderr, "x[%d] == %d\n", i, x[i]);
    }
  return 0;
}
extern void *nanos_create_current_kernel(const char *kernel_name, const char *opencl_code, const char *compiler_opts);
extern nanos_err_t nanos_opencl_set_arg(void *opencl_kernel, int arg_num, size_t size, const void *pointer);
extern nanos_err_t nanos_opencl_set_bufferarg(void *opencl_kernel, int arg_num, const void *pointer);
extern nanos_err_t nanos_profile_exec_kernel(void *opencl_kernel, int work_dim, int range_size, size_t *ndr_offset, size_t *ndr_local_size, size_t *ndr_global_size);
static void ocl_ol_init_1_unpacked(int n, int *x, int mcc_ndrange_2_0, int mcc_ndrange_2_1, int mcc_ndrange_2_2)
{
  {
    nanos_err_t nanos_err;
    void *ompss_kernel_ocl = nanos_create_current_kernel("init", "kernel.cl", "");
    nanos_err = nanos_opencl_set_arg(ompss_kernel_ocl, 0, sizeof(int), &n);
    nanos_err = nanos_opencl_set_bufferarg(ompss_kernel_ocl, 1, x);
    int range_size = 3 - 1 + 1;
    int num_dim = 3;
    size_t offset_arr[num_dim];
    size_t local_size_arr[num_dim][range_size];
    size_t global_size_arr[num_dim][range_size];
    {
      int k;
      int i;
      for ((k = 0, i = 1); i <= 3; (i++, k++))
        {
          offset_arr[0] = 0;
          local_size_arr[0][k] = 1 << i;
          global_size_arr[0][k] = mcc_ndrange_2_0;
        }
    }
    {
      int k;
      int i;
      for ((k = 0, i = 1); i <= 3; (i++, k++))
        {
          offset_arr[1] = 0;
          local_size_arr[1][k] = 1 << i;
          global_size_arr[1][k] = mcc_ndrange_2_1;
        }
    }
    {
      int k;
      int i;
      for ((k = 0, i = 1); i <= 3; (i++, k++))
        {
          offset_arr[2] = 0;
          local_size_arr[2][k] = 1 << i;
          global_size_arr[2][k] = mcc_ndrange_2_2;
        }
    }
    {
      int i;
      for (i = 0; i < range_size; i++)
        {
          int k;
          for (k = 0; k < num_dim; k = k + 1)
            {
              if (global_size_arr[k][i] < local_size_arr[k][i])
                {
                  local_size_arr[k][i] = global_size_arr[k][i];
                }
              else
                {
                  if (global_size_arr[k][i] % local_size_arr[k][i] != 0)
                    {
                      global_size_arr[k][i] = global_size_arr[k][i] + (local_size_arr[k][i] - global_size_arr[k][i] % local_size_arr[k][i]);
                    }
                }
            }
        }
    }
    nanos_err = nanos_profile_exec_kernel(ompss_kernel_ocl, num_dim, range_size, offset_arr, (size_t *)local_size_arr, (size_t *)global_size_arr);
  }
}
static void ocl_ol_init_1(struct nanos_args_0_t *const args)
{
  {
    ocl_ol_init_1_unpacked((*args).n, (*args).x, (*args).mcc_ndrange_2_0, (*args).mcc_ndrange_2_1, (*args).mcc_ndrange_2_2);
  }
}
__attribute__((common)) char ompss_uses_opencl;
