 // Loop 1.
float rx;
float ry;
float rz;
float r2;
float r2inv;
float r6inv;
float s;
__m128 eps_vector = _mm_set1_ps(eps);
	//not unrolled but focusing on basics atm
	t0 = wtime();
	for (int i = 0; i < N; i+=4)
	{
		__m128 xi_vector = _mm_load_ps(x+i);
		__m128 yi_vector = _mm_load_ps(y+i);
		__m128 zi_vector = _mm_load_ps(z+i);
		//loads next four contents of x, y and z into vector
		for (int j = 0; j < N; j++)
		{
			__m128 xj_vector = _mm_set1_ps(x[j]);
			__m128 yj_vector = _mm_set1_ps(y[j]);
			__m128 zj_vector = _mm_set1_ps(z[j]);
			/*
				This is supposed to set all four contents of xj_vector and so on
				to be the current value of x[j], is erroring so am using this incorrectly
			*/	

			__m128 rx_vector = _mm_sub_ps(xj_vector, xi_vector);
			__m128 ry_vector = _mm_sub_ps(yj_vector, yi_vector);
			__m128 rz_vector = _mm_sub_ps(zj_vector, zi_vector);
			//Simple subtraction, x[j] - x[i] and so on.
			
			/*
			rx = x[j] - x[i];
			ry = y[j] - y[i];
			rz = z[j] - z[i];
			*/

			//Little confusing, just a bunch of multiply and add operations on vectors
			__m128 r2_vector = _mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_mul_ps(rx_vector, rx_vector), _mm_mul_ps(ry_vector, ry_vector)), _mm_mul_ps(rz_vector, rz_vector)), eps_vector);
			//r2 = rx*rx + ry*ry + rz*rz + eps;
			
			//Multiply mixed in with rsqrt oeprations
			__m128 r6inv_vector = _mm_mul_ps(_mm_mul_ps(_mm_rsqrt_ps(r2_vector),_mm_rsqrt_ps(r2_vector)), _mm_rsqrt_ps(r2_vector));
			/*
				r2inv = 1.0f / sqrt(r2);
				r6inv = r2inv * r2inv * r2inv;
			*/

			//Supposed to set all values in m_vector to be value m[j] in array, again wrong.
			__m128 m_vector = _mm_set1_ps(m[j]);
			__m128 s_vector = _mm_mul_ps(m_vector,r6inv_vector);
			//s = m[j] * r6inv;

			//Simple store operation, storing result of loop i, i+1, i+2 i+3 and current j into ax
			_mm_store_ps(ax+i,_mm_mul_ps(s_vector, rx_vector));
			_mm_store_ps(ay+i,_mm_mul_ps(s_vector, ry_vector));
			_mm_store_ps(az+i,_mm_mul_ps(s_vector, rz_vector));	
			/*
				ax[i] += s * rx;
				ay[i] += s * ry;
				az[i] += s * rz;
			*/
		}
	}
	t1 = wtime();
	l1 += (t1 - t0);
