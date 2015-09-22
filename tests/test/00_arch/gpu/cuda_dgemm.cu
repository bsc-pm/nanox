
/*
<testinfo>
test_generator=gens/gpu-generator
test_generator_ENV='NX_TEST_ARCH=smp'
test_schedule=bf
</testinfo>
*/

#include <nanos.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

void matmul_moved(int *A, int *B, int *C, int bs);
void gpu_ol_matmul_1_unpacked(int *&A, int *&B, int *&C, int &bs) throw();
// GPU kernel
__global__ void dgemm ( int * a , int * b, int * c );

extern "C"
{
  extern void *nanos_gpu_factory(void *args);
}

struct  nanos_const_wd_definition_1
{
    ::nanos_const_wd_definition_t base;
    ::nanos_device_t devices[1];
};

/***** GPU CODE *****/
__global__ void dgemm ( int * a , int * b, int * c )
{
   int i = (blockIdx.x * blockDim.x) + threadIdx.x;
   int j = (blockIdx.y * blockDim.y) + threadIdx.y;
   int rsize = gridDim.y * blockDim.y;
   //int idx = threadIdx.x * blockDim.x + threadIdx.y;
   //int i = threadIdx.x;
   //int j = threadIdx.y;

   unsigned int k;
   for ( k = 0; k < rsize; k++ ) {
      //c[idx] += a[i*blockDim.x + k] * b[k*blockDim.x + j];
      c[i*rsize+j] += a[i*rsize + k] * b[k*rsize + j];
   }
}
/***** END GPU CODE *****/


void matmul_moved(int *A, int *B, int *C, int bs)
{
  // Assuming bs is multiple of 16
  dim3 dimBlock(16, 16);
  dim3 dimGrid(bs/dimBlock.x, bs/dimBlock.y);
  dgemm<<<dimGrid, dimBlock>>>( A, B, C );


}

void gpu_ol_matmul_1_unpacked(int *&A, int *&B, int *&C, int &bs) throw()
{
  {
    ::matmul_moved(A, B, C, bs);
  }
}
struct  nanos_args_0_t
{
    int *A;
    int *B;
    int *C;
    int bs;
};
void gpu_ol_matmul_1_unpacked(int *&A, int *&B, int *&C, int &bs) throw();
static void gpu_ol_matmul_1(::nanos_args_0_t &args) throw()
{
  {
    ::gpu_ol_matmul_1_unpacked(args.A, args.B, args.C, args.bs);
  }
}
static void nanos_xlate_fun_dgemmcpp_0(::nanos_args_0_t &arg, void *wd) throw()
{
  {
    void *device_base_address;
    ::nanos_err_t err;
    device_base_address = 0;
    err = ::nanos_get_addr(0, &device_base_address, wd);
    if (err != ::NANOS_OK)
      {
        ::nanos_handle_error(err);
      }
    arg.A = (int *)device_base_address;
  }
  {
    void *device_base_address;
    ::nanos_err_t err;
    device_base_address = 0;
    err = ::nanos_get_addr(1, &device_base_address, wd);
    if (err != ::NANOS_OK)
      {
        ::nanos_handle_error(err);
      }
    arg.B = (int *)device_base_address;
  }
  {
    void *device_base_address;
    ::nanos_err_t err;
    device_base_address = 0;
    err = ::nanos_get_addr(2, &device_base_address, wd);
    if (err != ::NANOS_OK)
      {
        ::nanos_handle_error(err);
      }
    arg.C = (int *)device_base_address;
  }
}
int main(int argc, char **argv)
{
  const int n(1024);
  const int bs(128);
  const int nb(n / bs);
  int times(2);
  int **A;
  int **B;
  int **C;
  int i;
  int j;
  int k;
  int t;
  A = (int **)::malloc(nb * nb * sizeof(int *));
  B = (int **)::malloc(nb * nb * sizeof(int *));
  C = (int **)::malloc(nb * nb * sizeof(int *));
  for (i = 0; i < nb * nb; i++)
    {
      A[i] = (int *)::malloc(bs * bs * sizeof(int));
      B[i] = (int *)::malloc(bs * bs * sizeof(int));
      C[i] = (int *)::malloc(bs * bs * sizeof(int));
      for (j = 0; j < bs * bs; j++)
        {
          A[i][j] = 1;
          B[i][j] = 2;
          C[i][j] = 0;
        }
    }
  for (t = 0; t < times; t++)
    {
      for (i = 0; i < nb; i++)
        {
          for (j = 0; j < nb; j++)
            {
              for (k = 0; k < nb; k++)
                {
                  {
                    int *mcc_arg_0(A[i * nb + k]);
                    int *mcc_arg_1(B[k * nb + j]);
                    int *mcc_arg_2(C[i * nb + j]);
                    int mcc_arg_3(bs);
                    {
                       /* CUDA device descriptor */ 
                      static ::nanos_smp_args_t gpu_ol_matmul_1_args = { /* .::nanos_smp_args_t::outline =  */ (void (*)(void *))::gpu_ol_matmul_1};
                      static ::nanos_const_wd_definition_1 nanos_wd_const_data = { /* .::nanos_const_wd_definition_1::base =  */ { /* .::nanos_const_wd_definition_tag::props =  */ { /* .::nanos_wd_props_t::mandatory_creation =  */ 1,  /* .::nanos_wd_props_t::tied =  */ 1,  /* .::nanos_wd_props_t::clear_chunk =  */ 0,  /* .::nanos_wd_props_t::reserved0 =  */ 0,  /* .::nanos_wd_props_t::reserved1 =  */ 0,  /* .::nanos_wd_props_t::reserved2 =  */ 0,  /* .::nanos_wd_props_t::reserved3 =  */ 0,  /* .::nanos_wd_props_t::reserved4 =  */ 0},  /* .::nanos_const_wd_definition_tag::data_alignment =  */ __alignof__(::nanos_args_0_t),  /* .::nanos_const_wd_definition_tag::num_copies =  */ 3,  /* .::nanos_const_wd_definition_tag::num_devices =  */ 1,  /* .::nanos_const_wd_definition_tag::num_dimensions =  */ 3,  /* .::nanos_const_wd_definition_tag::description =  */ 0},  /* .::nanos_const_wd_definition_1::devices =  */ { /* [0] =  */ { /* .::nanos_device_t::factory =  */ &::nanos_gpu_factory,  /* .::nanos_device_t::arg =  */ &gpu_ol_matmul_1_args}}};
                      ::nanos_wd_dyn_props_t nanos_wd_dyn_props;
                      nanos_wd_dyn_props.tie_to = 0;
                      nanos_wd_dyn_props.priority = 0;
                      nanos_wd_dyn_props.flags.is_final = 0;
                      ::nanos_args_0_t *ol_args;
                      ol_args = (::nanos_args_0_t *)0;
                      ::nanos_args_0_t imm_args;
                      ::nanos_wd_t nanos_wd_((::nanos_wd_t)0);
                      ::nanos_copy_data_t *ol_copy_data((::nanos_copy_data_t *)0);
                      ::nanos_region_dimension_internal_t *ol_copy_dimensions((::nanos_region_dimension_internal_t *)0);
                      ::nanos_err_t err;
                      err = ::nanos_create_wd_compact(&nanos_wd_, &nanos_wd_const_data.base, &nanos_wd_dyn_props, sizeof(::nanos_args_0_t &), (void **)&ol_args, ::nanos_current_wd(), &ol_copy_data, &ol_copy_dimensions);
                      if (err != ::NANOS_OK)
                        {
                          ::nanos_handle_error(err);
                        }
                      ::nanos_region_dimension_t dimensions_0[1] = { /* [0] =  */ { /* .::nanos_region_dimension_internal_t::size =  */ sizeof(int) * (((mcc_arg_3 * mcc_arg_3) - 1 - 0) + 1),  /* .::nanos_region_dimension_internal_t::lower_bound =  */ sizeof(int) * 0,  /* .::nanos_region_dimension_internal_t::accessed_length =  */ sizeof(int) * (((mcc_arg_3 * mcc_arg_3) - 1 - 0) + 1)}};
                      ::nanos_region_dimension_t dimensions_1[1] = { /* [0] =  */ { /* .::nanos_region_dimension_internal_t::size =  */ sizeof(int) * (((mcc_arg_3 * mcc_arg_3) - 1 - 0) + 1),  /* .::nanos_region_dimension_internal_t::lower_bound =  */ sizeof(int) * 0,  /* .::nanos_region_dimension_internal_t::accessed_length =  */ sizeof(int) * (((mcc_arg_3 * mcc_arg_3) - 1 - 0) + 1)}};
                      ::nanos_region_dimension_t dimensions_2[1] = { /* [0] =  */ { /* .::nanos_region_dimension_internal_t::size =  */ sizeof(int) * (((mcc_arg_3 * mcc_arg_3) - 1 - 0) + 1),  /* .::nanos_region_dimension_internal_t::lower_bound =  */ sizeof(int) * 0,  /* .::nanos_region_dimension_internal_t::accessed_length =  */ sizeof(int) * (((mcc_arg_3 * mcc_arg_3) - 1 - 0) + 1)}};
                      ::nanos_data_access_t dependences[3] = { /* [0] =  */ { /* .::nanos_data_access_internal_t::address =  */ (void *)mcc_arg_0,  /* .::nanos_data_access_internal_t::flags =  */ { /* .::nanos_access_type_internal_t::input =  */ 1,  /* .::nanos_access_type_internal_t::output =  */ 0,  /* .::nanos_access_type_internal_t::can_rename =  */ 0,  /* .::nanos_access_type_internal_t::concurrent =  */ 0,  /* .::nanos_access_type_internal_t::commutative =  */ 0},  /* .::nanos_data_access_internal_t::dimension_count =  */ 1,  /* .::nanos_data_access_internal_t::dimensions =  */ dimensions_0,  /* .::nanos_data_access_internal_t::offset =  */ 0},  /* [1] =  */ { /* .::nanos_data_access_internal_t::address =  */ (void *)mcc_arg_1,  /* .::nanos_data_access_internal_t::flags =  */ { /* .::nanos_access_type_internal_t::input =  */ 1,  /* .::nanos_access_type_internal_t::output =  */ 0,  /* .::nanos_access_type_internal_t::can_rename =  */ 0,  /* .::nanos_access_type_internal_t::concurrent =  */ 0,  /* .::nanos_access_type_internal_t::commutative =  */ 0},  /* .::nanos_data_access_internal_t::dimension_count =  */ 1,  /* .::nanos_data_access_internal_t::dimensions =  */ dimensions_1,  /* .::nanos_data_access_internal_t::offset =  */ 0},  /* [2] =  */ { /* .::nanos_data_access_internal_t::address =  */ (void *)mcc_arg_2,  /* .::nanos_data_access_internal_t::flags =  */ { /* .::nanos_access_type_internal_t::input =  */ 1,  /* .::nanos_access_type_internal_t::output =  */ 1,  /* .::nanos_access_type_internal_t::can_rename =  */ 0,  /* .::nanos_access_type_internal_t::concurrent =  */ 0,  /* .::nanos_access_type_internal_t::commutative =  */ 0},  /* .::nanos_data_access_internal_t::dimension_count =  */ 1,  /* .::nanos_data_access_internal_t::dimensions =  */ dimensions_2,  /* .::nanos_data_access_internal_t::offset =  */ 0}};
                      ;
                      if (nanos_wd_ != (::nanos_wd_t)0)
                        {
                          (*ol_args).A = mcc_arg_0;
                          (*ol_args).B = mcc_arg_1;
                          (*ol_args).C = mcc_arg_2;
                          (*ol_args).bs = mcc_arg_3;
                          ol_copy_dimensions[0].size = (((mcc_arg_3 * mcc_arg_3) - 1 - 0) + 1) * sizeof(int);
                          ol_copy_dimensions[0].lower_bound = (0 - 0) * sizeof(int);
                          ol_copy_dimensions[0].accessed_length = ((mcc_arg_3 * mcc_arg_3) - 1 - 0 - (0 - 0) + 1) * sizeof(int);
                          ol_copy_data[0].sharing = ::NANOS_SHARED;
                          ol_copy_data[0].address = (void *)mcc_arg_0;
                          ol_copy_data[0].flags.input = 1;
                          ol_copy_data[0].flags.output = 0;
                          ol_copy_data[0].dimension_count = 1;
                          ol_copy_data[0].dimensions = &ol_copy_dimensions[0];
                          ol_copy_data[0].offset = 0;
                          ol_copy_dimensions[1].size = (((mcc_arg_3 * mcc_arg_3) - 1 - 0) + 1) * sizeof(int);
                          ol_copy_dimensions[1].lower_bound = (0 - 0) * sizeof(int);
                          ol_copy_dimensions[1].accessed_length = ((mcc_arg_3 * mcc_arg_3) - 1 - 0 - (0 - 0) + 1) * sizeof(int);
                          ol_copy_data[1].sharing = ::NANOS_SHARED;
                          ol_copy_data[1].address = (void *)mcc_arg_1;
                          ol_copy_data[1].flags.input = 1;
                          ol_copy_data[1].flags.output = 0;
                          ol_copy_data[1].dimension_count = 1;
                          ol_copy_data[1].dimensions = &ol_copy_dimensions[1];
                          ol_copy_data[1].offset = 0;
                          ol_copy_dimensions[2].size = (((mcc_arg_3 * mcc_arg_3) - 1 - 0) + 1) * sizeof(int);
                          ol_copy_dimensions[2].lower_bound = (0 - 0) * sizeof(int);
                          ol_copy_dimensions[2].accessed_length = ((mcc_arg_3 * mcc_arg_3) - 1 - 0 - (0 - 0) + 1) * sizeof(int);
                          ol_copy_data[2].sharing = ::NANOS_SHARED;
                          ol_copy_data[2].address = (void *)mcc_arg_2;
                          ol_copy_data[2].flags.input = 1;
                          ol_copy_data[2].flags.output = 1;
                          ol_copy_data[2].dimension_count = 1;
                          ol_copy_data[2].dimensions = &ol_copy_dimensions[2];
                          ol_copy_data[2].offset = 0;
                          err = ::nanos_set_translate_function(nanos_wd_, (::nanos_translate_args_t)::nanos_xlate_fun_dgemmcpp_0);
                          if (err != ::NANOS_OK)
                            {
                              ::nanos_handle_error(err);
                            }
                          err = ::nanos_submit(nanos_wd_, 3, dependences, (::nanos_team_t)0);
                          if (err != ::NANOS_OK)
                            {
                              ::nanos_handle_error(err);
                            }
                        }
                      else
                        {
                          imm_args.A = mcc_arg_0;
                          imm_args.B = mcc_arg_1;
                          imm_args.C = mcc_arg_2;
                          imm_args.bs = mcc_arg_3;
                          ::nanos_copy_data_t imm_copy_data[3];
                          ::nanos_region_dimension_internal_t imm_copy_dimensions[3];
                          imm_copy_dimensions[0].size = (((mcc_arg_3 * mcc_arg_3) - 1 - 0) + 1) * sizeof(int);
                          imm_copy_dimensions[0].lower_bound = (0 - 0) * sizeof(int);
                          imm_copy_dimensions[0].accessed_length = ((mcc_arg_3 * mcc_arg_3) - 1 - 0 - (0 - 0) + 1) * sizeof(int);
                          imm_copy_data[0].sharing = ::NANOS_SHARED;
                          imm_copy_data[0].address = (void *)mcc_arg_0;
                          imm_copy_data[0].flags.input = 1;
                          imm_copy_data[0].flags.output = 0;
                          imm_copy_data[0].dimension_count = 1;
                          imm_copy_data[0].dimensions = &imm_copy_dimensions[0];
                          imm_copy_data[0].offset = 0;
                          imm_copy_dimensions[1].size = (((mcc_arg_3 * mcc_arg_3) - 1 - 0) + 1) * sizeof(int);
                          imm_copy_dimensions[1].lower_bound = (0 - 0) * sizeof(int);
                          imm_copy_dimensions[1].accessed_length = ((mcc_arg_3 * mcc_arg_3) - 1 - 0 - (0 - 0) + 1) * sizeof(int);
                          imm_copy_data[1].sharing = ::NANOS_SHARED;
                          imm_copy_data[1].address = (void *)mcc_arg_1;
                          imm_copy_data[1].flags.input = 1;
                          imm_copy_data[1].flags.output = 0;
                          imm_copy_data[1].dimension_count = 1;
                          imm_copy_data[1].dimensions = &imm_copy_dimensions[1];
                          imm_copy_data[1].offset = 0;
                          imm_copy_dimensions[2].size = (((mcc_arg_3 * mcc_arg_3) - 1 - 0) + 1) * sizeof(int);
                          imm_copy_dimensions[2].lower_bound = (0 - 0) * sizeof(int);
                          imm_copy_dimensions[2].accessed_length = ((mcc_arg_3 * mcc_arg_3) - 1 - 0 - (0 - 0) + 1) * sizeof(int);
                          imm_copy_data[2].sharing = ::NANOS_SHARED;
                          imm_copy_data[2].address = (void *)mcc_arg_2;
                          imm_copy_data[2].flags.input = 1;
                          imm_copy_data[2].flags.output = 1;
                          imm_copy_data[2].dimension_count = 1;
                          imm_copy_data[2].dimensions = &imm_copy_dimensions[2];
                          imm_copy_data[2].offset = 0;
                          err = ::nanos_create_wd_and_run_compact(&nanos_wd_const_data.base, &nanos_wd_dyn_props, sizeof(::nanos_args_0_t &), &imm_args, 3, dependences, imm_copy_data, imm_copy_dimensions, (::nanos_translate_args_t)::nanos_xlate_fun_dgemmcpp_0);
                          if (err != ::NANOS_OK)
                            {
                              ::nanos_handle_error(err);
                            }
                        }
                    }
                  }
                }
            }
        }
    }
  {
    ::nanos_wd_t nanos_wd_(::nanos_current_wd());
    ::nanos_err_t err;
    err = ::nanos_wg_wait_completion(nanos_wd_, 0);
    if (err != ::NANOS_OK)
      {
        ::nanos_handle_error(err);
      }
  }
  int err(0);
  for (i = 0; i < nb * nb; i++)
    {
      for (j = 0; j < bs * bs; j++)
        {
          if (C[i][j] != 1 * 2 * n * times)
            {
              if (err < 5)
                {
                  ::printf("Expected %d but found %d at (%d, %d)\n", 1 * 2 * nb * times, C[i][j], i, j);
                }
              err++;
            }
        }
    }
  if (err != 0)
    {
      ::printf("Found %d errors\n", err);
    }
  return err;
}
__attribute__((weak)) char ompss_uses_cuda(1);
