/* ocl_kernels.cl */
__kernel void init(int n, int  __global * x)
{
    int i = get_global_id(0);
    if (i >= n)
        return;
    x[i] = i;
}

__kernel void increment(int n, int  __global * x)
{
    int i = get_global_id(0);
    if (i >= n)
        return;
    x[i]++;
}
