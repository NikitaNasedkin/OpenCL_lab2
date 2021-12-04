__kernel void saxpy(const int n, float a, __global float *input, 
	 const int incx, __global float *output,  const int incy) {
  int id = get_global_id(0);
  
  output[id * incx] = output[id * incx] + a * input[id * incy];
  
}

__kernel void daxpy(const int n, double a, __global double *input, 
	 const int incx, __global double *output,  const int incy) {
  int id = get_global_id(0);
  
  output[id * incx] = output[id * incx] + a * input[id * incy];
}
