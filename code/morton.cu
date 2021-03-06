// Copyright (C) 2016 Peter Zaspel
//
// This file is part of hmglib.
//
// hmglib is free software: you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License as published by the Free
// Software Foundation, either version 3 of the License, or (at your option) any
// later version.
//
// hmglib is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
// A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
// details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with hmglib.  If not, see <http://www.gnu.org/licenses/>.


#include "morton.h"

__device__ __host__ __forceinline__ uint64_t stretch(uint64_t x, int dim, int bits)
{
	if ((dim==3)&&(bits==20))
	{
		x &= 0xF00000000000FFFFul;         		// x = ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- jihg fedc ba98 7654 3210
		x = (x ^ (x << 32)) & 0x000F00000000FFFFul;	// x = ---- ---- ---- jihg ---- ---- ---- ---- ---- ---- ---- ---- fedc ba98 7654 3210
								// m = 0000 0000 0000 1111 0000 0000 0000 0000 0000 0000 0000 0000 1111 1111 1111 1111
		x = (x ^ (x << 16)) & 0x000F0000FF0000FFul;	// x = ---- ---- ---- jihg ---- ---- ---- ---- fedc ba98 ---- ---- ---- ---- 7654 3210
								// m = 0000 0000 0000 1111 0000 0000 0000 0000 1111 1111 0000 0000 0000 0000 1111 1111
		x = (x ^ (x <<  8)) & 0x000F00F00F00F00Ful;	// x = ---- ---- ---- jihg ---- ---- fedc ---- ---- ba98 ---- ---- 7654 ---- ---- 3210
								// m = 0000 0000 0000 1111 0000 0000 1111 0000 0000 1111 0000 0000 1111 0000 0000 1111
		x = (x ^ (x <<  4)) & 0x00c30c30c30c30c3ul;	// x = ---- ---- ji-- --hg ---- fe-- --dc ---- ba-- --98 ---- 76-- --54 ---- 32-- --10
								// m = 0000 0000 1100 0011 0000 1100 0011 0000 1100 0011 0000 1100 0011 0000 1100 0011
		x = (x ^ (x <<  2)) & 0x0249249249249249ul;	// x = ---- --j- -i-- h--g --f- -e-- d--c --b- -a-- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
								// m = 0000 0010 0100 1001 0010 0100 1001 0010 0100 1001 0010 0100 1001 0010 0100 1001
		return x;
	} else if ((dim==2)&&(bits==32))
	{
		x &=  0x00000000FFFFFFFFul; // mask		// x = ---- ---- ---- ---- ---- ---- ---- ---- vuts rqpo nmlk jihg fedc ba98 7654 3210
		x = (x | (x << 16)) & 0x0000FFFF0000FFFFul;	// x = ---- ---- ---- ---- vuts rqpo nmlk jihg ---- ---- ---- ---- fedc ba98 7654 3210
								// m = 0000 0000 0000 0000 1111 1111 1111 1111 0000 0000 0000 0000 1111 1111 1111 1111
		x = (x | (x << 8)) & 0x00FF00FF00FF00FFul;	// x = ---- ---- vuts rqpo ---- ---- nmlk jihg ---- ---- fedc ba98 ---- ---- 7654 3210
								// m = 0000 0000 1111 1111 0000 0000 1111 1111 0000 0000 1111 1111 0000 0000 1111 1111
		x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0Ful;	// x = ---- vuts ---- rqpo ---- nmlk ---- jihg ---- fedc ---- ba98 ---- 7654 ---- 3210
								// m = 0000 1111 0000 1111 0000 1111 0000 1111 0000 1111 0000 1111 0000 1111 0000 1111
		x = (x | (x << 2)) & 0x3333333333333333ul;	// x = --vu --ts --rq --po --nm --lk --ji --hg --fe --dc --ba --98 --76 --54 --32 --10
								// m = 0011 0011 0011 0011 0011 0011 0011 0011 0011 0011 0011 0011 0011 0011 0011 0011
		x = (x | (x << 1)) & 0x5555555555555555ul; 	// x = -v-u -t-s -r-q -p-o -n-m -l-k -j-i -h-g -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
								// m = 0101 0101 0101 0101 0101 0101 0101 0101 0101 0101 0101 0101 0101 0101 0101 0101

		return x;
	}
	else if ((dim==10)&&(bits==4))
	{
		x &=  0x000000000000000Ful; // mask		// x = ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 3210
		x = (x | (x << 16)) & 0x00000000000C0003ul;	// x = ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 32-- ---- ---- ---- --10
								// m = 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 1100 0000 0000 0000 0011
		x = (x | (x <<  8)) & 0x0000000008040201ul;	// x = ---- ---- ---- ---- ---- ---- ---- ---- ---- 3--- ---- -2-- ---- --1- ---- ---0
								// m = 0000 0000 0000 0000 0000 0000 0000 0000 0000 1000 0000 0100 0000 0010 0000 0001
		x = (x | (x <<  2)) & 0x0000000020100201ul;	// x = ---- ---- ---- ---- ---- ---- ---- ---- --3- ---- ---2 ---- ---- --1- ---- ---0
								// m = 0011 0011 0011 0011 0011 0011 0011 0011 0010 0000 0001 0000 0000 0010 0000 0001
		x = (x | (x <<  1)) & 0x0000000040100401ul; 	// x = ---- ---- ---- ---- ---- ---- ---- ---- -3-- ---- ---2 ---- ---- -1-- ---- ---0
								// m = 0000 0000 0000 0000 0000 0000 0000 0000 0100 0000 0001 0000 0000 0100 0000 0001
		return x;
	}
	else if ((dim==8)&&(bits==8))
	{
		x &=  0x00000000000000FFul; 
		x = (x | (x << 28)) & 0x0000000F0000000Ful;
		x = (x | (x << 14)) & 0x0003000300030003ul;
		x = (x | (x << 7)) & 0x0101010101010101ul;
		return x;
	}
	else return 0x0ul;
}



__device__ __host__ __forceinline__ uint64_t coord_to_fp(double coord, double minimum, double max_minus_min, int bits)
{
	double tmp = (coord - minimum) / max_minus_min;
	return min( (uint64_t)floor( tmp*(1ul << bits) ), (uint64_t)((1ul<<bits)-1) );
}



__global__ void get_morton_code_kernel(struct point_set* points, struct morton_code* morton)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= points->size) return;

	int dim = points->dim;
	int bits = morton->bits;

	uint64_t code_result = 0x0ul;
	for (int d=0; d<dim; d++)
	{
		double maximum = points->max_per_dim[d];
		double minimum = points->min_per_dim[d];
		double mmm = maximum-minimum;
		uint64_t tmp_code;

		// generate fixed-point representation of coordinate
		tmp_code = coord_to_fp( points->coords[d][idx], minimum, mmm, bits);
		// stretch binary representation
		tmp_code = stretch(tmp_code, dim, bits);
		// interleave
		code_result = code_result | (tmp_code << d);
	}
	
	morton->code[idx] = code_result;
	
	return;
		
}

void get_morton_code(struct point_set* points, struct morton_code* morton, int grid_size, int block_size)
{
	get_morton_code_kernel<<<grid_size, block_size>>>(points, morton);
}



