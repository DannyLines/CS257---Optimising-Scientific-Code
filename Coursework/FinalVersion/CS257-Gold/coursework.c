#include "omp.h"
/**
 * The function to optimise as part of the coursework.
 *
 * l0, l1, l2 and l3 record the amount of time spent in each loop
 * and should not be optimised out. :)
 */
#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
void compute() {

	double t0, t1;
	
	// Loop 0.
	t0 = wtime();
	int i;
	int unroll = (int) (N/4)*4;
	__m128 loop0_vector = _mm_set1_ps(0.0f);
	for (i = 0; i < unroll; i+=4)
	{
		_mm_store_ps(ax+i, loop0_vector);
	}

	for(i = 0; i< unroll; i+=4)
	{
		_mm_store_ps(ay+i, loop0_vector);
	}

	for(i = 0; i < unroll; i+=4)
	{
		_mm_store_ps(az+i, loop0_vector);
	}

	for(;i<N;i++)
	{
		ax[i] = 0.0f;
		ay[i] = 0.0f;
		az[i] = 0.0f;
	}
	t1 = wtime();
	l0 += (t1 - t0);


 // Loop 1.
	t0 = wtime();
	float rx;
	float ry;
	float rz;
	float r2;
	float r2inv;
	float r6inv;
	float s;
	int j = 0;
	
	__m128 eps_vector = _mm_set1_ps(eps);
	if(N > 250)
	{
	  #pragma omp parallel for
	  for (int i = 0; i < unroll; i+=4)
	  {
		  __m128 xi_vector = _mm_load_ps(x+i);
		  __m128 yi_vector = _mm_load_ps(y+i);
		  __m128 zi_vector = _mm_load_ps(z+i);
		  for (j = 0; j < N; j++)
		  {
			  __m128 xj_vector = _mm_set1_ps(x[j]);
			  __m128 rx_vector = _mm_sub_ps(xj_vector, xi_vector);		
			  __m128 yj_vector = _mm_set1_ps(y[j]);
			  __m128 ry_vector = _mm_sub_ps(yj_vector, yi_vector);
			  __m128 zj_vector = _mm_set1_ps(z[j]);
			  __m128 rz_vector = _mm_sub_ps(zj_vector, zi_vector);	
			  __m128 r2inv_vector = _mm_rsqrt_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_mul_ps(rx_vector,rx_vector),_mm_mul_ps(ry_vector, ry_vector)),_mm_mul_ps(rz_vector, rz_vector)),eps_vector));
			  __m128 mj_vector = _mm_set1_ps(m[j]);
			  __m128 s_vector = _mm_mul_ps(mj_vector, _mm_mul_ps(_mm_mul_ps(r2inv_vector, r2inv_vector), r2inv_vector));
			  __m128 ax_vector = _mm_load_ps(ax+i);
			  _mm_store_ps(ax+i, _mm_add_ps(ax_vector,_mm_mul_ps(s_vector,rx_vector)));
			  __m128 ay_vector = _mm_load_ps(ay+i);
			  _mm_store_ps(ay+i, _mm_add_ps(ay_vector,_mm_mul_ps(s_vector,ry_vector)));
			  __m128 az_vector = _mm_load_ps(az+i);
			  _mm_store_ps(az+i, _mm_add_ps(az_vector,_mm_mul_ps(s_vector,rz_vector)));
		  }
	  }
	  for(;i<N;i++)
	  {
		  for(;j<N;j++)
		  {
			  rx = x[j] - x[i];
			  ry = y[j] - y[i];
			  rz = z[j] - z[i];
			  r2 = rx*rx + ry*ry + rz*rz + eps;
			  r2inv = 1.0f / sqrt(r2);
			  r6inv = r2inv * r2inv * r2inv;
			  s = m[j] * r6inv;
			  ax[i] += s * rx;
			  ay[i] += s * ry;
			  az[i] += s * rz;
		  }
	  }
	}
	else
	{
	  for (int i = 0; i < unroll; i+=4)
	  {
		  __m128 xi_vector = _mm_load_ps(x+i);
		  __m128 yi_vector = _mm_load_ps(y+i);
		  __m128 zi_vector = _mm_load_ps(z+i);
		  for (j = 0; j < N; j++)
		  {
			  __m128 xj_vector = _mm_set1_ps(x[j]);
			  __m128 rx_vector = _mm_sub_ps(xj_vector, xi_vector);		
			  __m128 yj_vector = _mm_set1_ps(y[j]);
			  __m128 ry_vector = _mm_sub_ps(yj_vector, yi_vector);
			  __m128 zj_vector = _mm_set1_ps(z[j]);
			  __m128 rz_vector = _mm_sub_ps(zj_vector, zi_vector);	
			  __m128 r2inv_vector = _mm_rsqrt_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_mul_ps(rx_vector,rx_vector),_mm_mul_ps(ry_vector, ry_vector)),_mm_mul_ps(rz_vector, rz_vector)),eps_vector));
			  __m128 mj_vector = _mm_set1_ps(m[j]);
			  __m128 s_vector = _mm_mul_ps(mj_vector, _mm_mul_ps(_mm_mul_ps(r2inv_vector, r2inv_vector), r2inv_vector));
			  __m128 ax_vector = _mm_load_ps(ax+i);
			  _mm_store_ps(ax+i, _mm_add_ps(ax_vector,_mm_mul_ps(s_vector,rx_vector)));
			  __m128 ay_vector = _mm_load_ps(ay+i);
			  _mm_store_ps(ay+i, _mm_add_ps(ay_vector,_mm_mul_ps(s_vector,ry_vector)));
			  __m128 az_vector = _mm_load_ps(az+i);
			  _mm_store_ps(az+i, _mm_add_ps(az_vector,_mm_mul_ps(s_vector,rz_vector)));
		  }
	  }
	  for(;i<N;i++)
	  {
		  for(;j<N;j++)
		  {
			  rx = x[j] - x[i];
			  ry = y[j] - y[i];
			  rz = z[j] - z[i];
			  r2 = rx*rx + ry*ry + rz*rz + eps;
			  r2inv = 1.0f / sqrt(r2);
			  r6inv = r2inv * r2inv * r2inv;
			  s = m[j] * r6inv;
			  ax[i] += s * rx;
			  ay[i] += s * ry;
			  az[i] += s * rz;
		  }
	  }
	}
	t1 = wtime();
	l1 += (t1 - t0);
	// Loop 2.
	t0 = wtime();
	float result = dmp*dt;
	__m128 result_vector = _mm_set1_ps(result);
	
	for (int i = 0; i < N; i+=4)
	{
	  _mm_store_ps(vx+i, _mm_add_ps(_mm_load_ps(vx+i), _mm_mul_ps(result_vector, _mm_load_ps(ax+i))));
		//vx[i] += dmp * (dt * ax[i]);
	}
	for(int i =0; i<N; i+=4)
	{
	  _mm_store_ps(vy+i, _mm_add_ps(_mm_load_ps(vy+i), _mm_mul_ps(result_vector, _mm_load_ps(ay+i))));
	}
	for(int i = 0; i < N; i+=4)
	{
	  _mm_store_ps(vz+i, _mm_add_ps(_mm_load_ps(vz+i), _mm_mul_ps(result_vector, _mm_load_ps(az+i))));
	}
	t1 = wtime();
	l2 += (t1 - t0);

	// Loop 3.
	t0 = wtime();
	
		for (int i = 0; i < N; i++) {
		x[i] += dt * vx[i];
		y[i] += dt * vy[i];
		z[i] += dt * vz[i];
		if (x[i] >= 1.0f || x[i] <= -1.0f) vx[i] *= -1.0f;
		if (y[i] >= 1.0f || y[i] <= -1.0f) vy[i] *= -1.0f;
		if (z[i] >= 1.0f || z[i] <= -1.0f) vz[i] *= -1.0f;				
		
	}
/*	
	for (int i = 0; i < N; i+=4)
	{
		__m128 x_vector = _mm_load_ps(x+i);
		__m128 dt_vector = _mm_set1_ps(dt);
		__m128 vx_vector = _mm_load_ps(vx+i);

		__m128 dtvx_vector = _mm_mul_ps(dt_vector, vx_vector);
		__m128 first_vector = _mm_add_ps(x_vector, dtvx_vector);
		
		_mm_store_ps(x+i, first_vector);
			
		__m128 positive_vector = _mm_set1_ps(1.0f);
		__m128 negative_vector = _mm_set1_ps(-1.0f);
		
		__m128 positive_flag_vector = _mm_cmpnle_ps(x_vector,positive_vector);
		__m128 negative_flag_vector = _mm_cmpnge_ps(x_vector,negative_vector);
				float test[4];
		_mm_store_ps(test, negative_flag_vector);
		
		
		__m128 result_flag_vector = _mm_or_ps(positive_flag_vector, negative_flag_vector);
		
		__m128 branch1_vector = _mm_mul_ps(x_vector, negative_vector);
		__m128 branch2_vector = _mm_mul_ps(x_vector, positive_vector);
		
		__m128 result = _mm_blendv_ps(branch2_vector, branch1_vector, result_flag_vector);
		
		_mm_store_ps(vx+i, result);
		
	}	
for (int i = 0; i < N; i++)
	{
		y[i] += dt * vy[i];
		if (y[i] >= 1.0f || y[i] <= -1.0f) vy[i] *= -1.0f;
		
		z[i] += dt * vz[i];
		if (z[i] >= 1.0f || z[i] <= -1.0f) vz[i] *= -1.0f;
	}
*/				
	t1 = wtime();
	l3 += (t1 - t0);
	
}
