__constant__ const int BDIMX1=32;
__constant__ const int BDIMY1=16;
__constant__ const int radius1=6;
__constant__ const float pai=3.1415926;
__constant__ const int filter_scale=9;


__global__ void sum_poynting(float *poyn_px_d,float *poyn_pz_d,float *poyn_sx_d,float *poyn_sz_d,float *vxp1_d,float *vzp1_d,float *vxs1_d,float *vzs1_d,float *txx1_d,float *tzz1_d,float *txz1_d,float *tp1_d,int dimx,int dimz)
//sum_poynting<<<dimGrid,dimBlock>>>(poyn_rpx_d,poyn_rpz_d,poyn_rsx_d,poyn_rsz_d,rvxp1_d,rvzp1_d,rvxs1_d,rvzs1_d,rtxx1_d,rtzz1_d,rtxz1_d,rtp1_d,nx_append_radius,nz_append_radius);
//poyn_px_d,poyn_pz_d,poyn_sx_d,poyn_sz_d,vxp1_d,vzp1_d,vxs1_d,vzs1_d,txx1_d,tzz1_d,txz1_d,tp1_d,nx_append_radius1,nz_append_radius1
//poyn_rpx_d,poyn_rpz_d,poyn_rsx_d,poyn_rsz_d,rvxp2_d,rvzp2_d,rvxs2_d,rvzs2_d,rtxx2_d,rtzz2_d,rtxz2_d,rtp2_d,nx_append_radius1,nz_append_radius1
//(poyn_px_d,poyn_pz_d,poyn_sx_d,poyn_sz_d,vxp2_d,vzp2_d,vxs2_d,vzs2_d,txx2_d,tzz2_d,txz2_d,tp2_d,nx_append_radius1,nz_append_radius1);
{

		int ix=blockIdx.x*blockDim.x+threadIdx.x;
		int iz=blockIdx.y*blockDim.y+threadIdx.y;

		int in_idx;

		if(ix<dimx&&iz<dimz)
		{
			ix=ix+radius1;
			iz=iz+radius1;
			dimx=dimx+2*radius1;
			dimz=dimz+2*radius1;
			in_idx=ix*dimz+iz;
			
			poyn_px_d[in_idx]=-1*tp1_d[in_idx]*vxp1_d[in_idx];
			
			poyn_pz_d[in_idx]=-1*tp1_d[in_idx]*vzp1_d[in_idx];

			//poyn_px_d[in_idx]=-1*tp1_d[in_idx]*(vxp1_d[in_idx]+vxp1_d[in_idx+dimz])/2.0;
			
			//poyn_pz_d[in_idx]=-1*tp1_d[in_idx]*(vzp1_d[in_idx]+vzp1_d[in_idx+1])/2.0;
			
			//poyn_px_d[in_idx]=-1*tp1_d[in_idx]*(vxp1_d[in_idx]+vxp1_d[in_idx-dimz])/2.0;
			
			//poyn_pz_d[in_idx]=-1*tp1_d[in_idx]*(vzp1_d[in_idx]+vzp1_d[in_idx-1])/2.0;



			poyn_sx_d[in_idx]=-1*((txx1_d[in_idx]-tp1_d[in_idx])*vxs1_d[in_idx]+txz1_d[in_idx]*vzs1_d[in_idx]);
			
			poyn_sz_d[in_idx]=-1*((tzz1_d[in_idx]-tp1_d[in_idx])*vzs1_d[in_idx]+txz1_d[in_idx]*vxs1_d[in_idx]);
			
			//poyn_sx_d[in_idx]=-1*((txx1_d[in_idx]-tp1_d[in_idx])*(vxs1_d[in_idx]+vxs1_d[in_idx+dimz])/2.0+(txz1_d[in_idx]+txz1_d[in_idx+1]+txz1_d[in_idx+dimz]+txz1_d[in_idx+1+dimz])*(vzs1_d[in_idx]+vzs1_d[in_idx+1])/2.0/4.0);
			
			//poyn_sz_d[in_idx]=-1*((tzz1_d[in_idx]-tp1_d[in_idx])*(vzs1_d[in_idx]+vzs1_d[in_idx+1])/2.0+(txz1_d[in_idx]+txz1_d[in_idx+1]+txz1_d[in_idx+dimz]+txz1_d[in_idx+1+dimz])*(vxs1_d[in_idx]+vxs1_d[in_idx+dimz])/2.0/4.0);
			
			//poyn_sx_d[in_idx]=-1*((txx1_d[in_idx]-tp1_d[in_idx])*(vxs1_d[in_idx]+vxs1_d[in_idx-dimz])/2.0+(txz1_d[in_idx]+txz1_d[in_idx-1]+txz1_d[in_idx-dimz]+txz1_d[in_idx-1-dimz])*(vzs1_d[in_idx]+vzs1_d[in_idx-1])/2.0/4.0);
			
			//poyn_sz_d[in_idx]=-1*((tzz1_d[in_idx]-tp1_d[in_idx])*(vzs1_d[in_idx]+vzs1_d[in_idx-1])/2.0+(txz1_d[in_idx]+txz1_d[in_idx-1]+txz1_d[in_idx-dimz]+txz1_d[in_idx-1-dimz])*(vxs1_d[in_idx]+vxs1_d[in_idx-dimz])/2.0/4.0);
		}
}
__global__ void poynting(float *txx2_d,float *txz2_d,float *tzz2_d,float *vx2_d,float *vz2_d,float *pz_d,float *px_d,int dimx,int dimz)
//(txx2_d,txz2_d,tzz2_d,vx2_d,vz2_d,poyn_z_d,poyn_x_d,nx_append_radius1,nz_append_radius1);
{

		int ix=blockIdx.x*blockDim.x+threadIdx.x;
		int iz=blockIdx.y*blockDim.y+threadIdx.y;

		int in_idx;

		if(ix<dimx&&iz<dimz)
		{
			ix=ix+radius1;
			iz=iz+radius1;
			dimx=dimx+2*radius1;
			dimz=dimz+2*radius1;
			in_idx=ix*dimz+iz;
			
			//pz_d[in_idx]=-(txz2_d[in_idx]*vx2_d[in_idx]+tzz2_d[in_idx]*vz2_d[in_idx]);

			//px_d[in_idx]=-(txx2_d[in_idx]*vx2_d[in_idx]+txz2_d[in_idx]*vz2_d[in_idx]);
			
			pz_d[in_idx]=-((txz2_d[in_idx]+txz2_d[in_idx+1]+txz2_d[in_idx+dimz]+txz2_d[in_idx+1+dimz])*(vx2_d[in_idx]+vx2_d[in_idx+1])/2.0/4.0+tzz2_d[in_idx]*(vz2_d[in_idx]+vz2_d[in_idx+1])/2.0);

			px_d[in_idx]=-(txx2_d[in_idx]*(vx2_d[in_idx]+vx2_d[in_idx+dimz])/2.0+(txz2_d[in_idx]+txz2_d[in_idx+1]+txz2_d[in_idx+dimz]+txz2_d[in_idx+1+dimz])*(vz2_d[in_idx]+vz2_d[in_idx+1])/2.0/4.0);
			
			//pz_d[in_idx]=-((txz2_d[in_idx]+txz2_d[in_idx-1]+txz2_d[in_idx-dimz]+txz2_d[in_idx-1-dimz])*(vx2_d[in_idx]+vx2_d[in_idx-1])/2.0/4.0+tzz2_d[in_idx]*(vz2_d[in_idx]+vz2_d[in_idx-1])/2.0);

			//px_d[in_idx]=-(txx2_d[in_idx]*(vx2_d[in_idx]+vx2_d[in_idx-dimz])/2.0+(txz2_d[in_idx]+txz2_d[in_idx-1]+txz2_d[in_idx-dimz]+txz2_d[in_idx-1-dimz])*(vz2_d[in_idx]+vz2_d[in_idx-1])/2.0/4.0);
		}
}

__global__ void scalar_poynting(float *rpx_d,float *rpz_d,float *rvx2,float *rvz2,float *wf2_d,float *coe_d,int dimx,int dimz)
{
		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		if((ix<dimx)&&(iz<dimz))
		{
				dimx=dimx+2*radius1;
				dimz=dimz+2*radius1;
				ix=ix+radius1;
				iz=iz+radius1;
				in_idx=ix*dimz+iz;		

				rpx_d[in_idx]=-rvx2[in_idx]*wf2_d[in_idx];
				rpz_d[in_idx]=-rvz2[in_idx]*wf2_d[in_idx];
		}
}

__global__ void cal_direction_2D_elastic(float *poyn_px_d,float *poyn_pz_d,float *poyn_sx_d,float *poyn_sz_d,float *vxp1_d,float *vzp1_d,float *vxs1_d,float *vzs1_d,float *txx1_d,float *tzz1_d,float *txz1_d,float *tp1_d,int dimx,int dimz)
///cal_direction_2D_elastic<<<dimGrid,dimBlock>>>(direction_px_d,direction_pz_d,direction_sx_d,direction_sz_d,vxp2_d,vzp2_d,vxs2_d,vzs2_d,txx2_d,tzz2_d,txz2_d,tp2_d,nx_append_radius,nz_append_radius);
{
		int ix=blockIdx.x*blockDim.x+threadIdx.x;
		int iz=blockIdx.y*blockDim.y+threadIdx.y;

		int in_idx;
////cross product
////one is x component , another is z component
		if(ix<dimx&&iz<dimz)
		{
			ix=ix+radius1;
			iz=iz+radius1;
			dimx=dimx+2*radius1;
			dimz=dimz+2*radius1;
			in_idx=ix*dimz+iz;

			if(vzp1_d[in_idx]>=0)
			{
				poyn_px_d[in_idx]=vxp1_d[in_idx];
				
				poyn_pz_d[in_idx]=vzp1_d[in_idx];
			}

			else
			{
				poyn_px_d[in_idx]=-1.0*vxp1_d[in_idx];
				
				poyn_pz_d[in_idx]=-1.0*vzp1_d[in_idx];
			}

			if(vxs1_d[in_idx]>=0)
			{
				poyn_sx_d[in_idx]=vzs1_d[in_idx];
				
				poyn_sz_d[in_idx]=-1.0*vxs1_d[in_idx];
			}

			else
			{				
				poyn_sx_d[in_idx]=-1.0*vzs1_d[in_idx];
				
				poyn_sz_d[in_idx]=vxs1_d[in_idx];	
			}
		}
}

__global__ void divergence_old(float *p_d,float *p1_d,float *coe_d,float dx,float dz,int dimx,int dimz)
{
		__shared__ float s_data[BDIMY1+2*radius1][BDIMX1+2*radius1];

		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		int tx = threadIdx.x+radius1;
		int tz = threadIdx.y+radius1;
		s_data[tz][tx]=0.0;
		s_data[threadIdx.y][threadIdx.x]=0.0;
		s_data[BDIMY1+2*radius1-1-threadIdx.y][BDIMX1+2*radius1-1-threadIdx.x]=0.0;
		s_data[threadIdx.y][BDIMX1+2*radius1-1-threadIdx.x]=0.0;
		s_data[BDIMY1+2*radius1-1-threadIdx.y][threadIdx.x]=0.0;


		if((ix<dimx)&&(iz<dimz))
		{
				dimx=dimx+2*radius1;dimz=dimz+2*radius1;
				ix=ix+radius1;iz=iz+radius1;
				in_idx = ix*dimz+iz;//iz*dimx+ix;

				__syncthreads();

				s_data[tz][tx]=p_d[in_idx];

				if(threadIdx.y<radius1)
				{
						s_data[threadIdx.y][tx]=p_d[in_idx-radius1];//g_input[in_idx-radius1*dimx];//up
						s_data[threadIdx.y+BDIMY1+radius1][tx]=p_d[in_idx+BDIMY1];//g_input[in_idx+BDIMY1*dimx];//down
				}
				if(threadIdx.x<radius1)
				{
						s_data[tz][threadIdx.x]=p_d[in_idx-radius1*dimz];//g_input[in_idx-radius1];//left
						s_data[tz][threadIdx.x+BDIMX1+radius1]=p_d[in_idx+BDIMX1*dimz];//g_input[in_idx+BDIMX1];//right
				}

				__syncthreads();
			
				/*float    sumx=coe_d[1]*(s_data[tz][tx+1]-s_data[tz][tx]);
					sumx+=coe_d[2]*(s_data[tz][tx+2]-s_data[tz][tx-1]);
					sumx+=coe_d[3]*(s_data[tz][tx+3]-s_data[tz][tx-2]);
					sumx+=coe_d[4]*(s_data[tz][tx+4]-s_data[tz][tx-3]);
					sumx+=coe_d[5]*(s_data[tz][tx+5]-s_data[tz][tx-4]);
					sumx+=coe_d[6]*(s_data[tz][tx+6]-s_data[tz][tx-5]);*/

				/*float    sumz=coe_d[1]*(s_data[tz+1][tx]-s_data[tz][tx]);
					sumz+=coe_d[2]*(s_data[tz+2][tx]-s_data[tz-1][tx]);
					sumz+=coe_d[3]*(s_data[tz+3][tx]-s_data[tz-2][tx]);
					sumz+=coe_d[4]*(s_data[tz+4][tx]-s_data[tz-3][tx]);
					sumz+=coe_d[5]*(s_data[tz+5][tx]-s_data[tz-4][tx]);
					sumz+=coe_d[6]*(s_data[tz+6][tx]-s_data[tz-5][tx]);*/
////two choice
				float    sumx=coe_d[1]*(s_data[tz][tx+1]-s_data[tz][tx-1]);
					sumx+=coe_d[2]*(s_data[tz][tx+2]-s_data[tz][tx-2]);
					sumx+=coe_d[3]*(s_data[tz][tx+3]-s_data[tz][tx-3]);
					sumx+=coe_d[4]*(s_data[tz][tx+4]-s_data[tz][tx-4]);
					sumx+=coe_d[5]*(s_data[tz][tx+5]-s_data[tz][tx-5]);
					sumx+=coe_d[6]*(s_data[tz][tx+6]-s_data[tz][tx-6]);

				float    sumz=coe_d[1]*(s_data[tz+1][tx]-s_data[tz-1][tx]);
					sumz+=coe_d[2]*(s_data[tz+2][tx]-s_data[tz-2][tx]);
					sumz+=coe_d[3]*(s_data[tz+3][tx]-s_data[tz-3][tx]);
					sumz+=coe_d[4]*(s_data[tz+4][tx]-s_data[tz-4][tx]);
					sumz+=coe_d[5]*(s_data[tz+5][tx]-s_data[tz-5][tx]);
					sumz+=coe_d[6]*(s_data[tz+6][tx]-s_data[tz-6][tx]);			

				p1_d[in_idx]=-1*sumx/dx;

		}

}

__global__ void divergence_new(float *p_d,float *p1_d,float *normalx1_d,float *normalz1_d,float *coe_d,float dx,float dz,int dimx,int dimz)
{
		__shared__ float s_data[BDIMY1+2*radius1][BDIMX1+2*radius1];

		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		int tx = threadIdx.x+radius1;
		int tz = threadIdx.y+radius1;
		s_data[tz][tx]=0.0;
		s_data[threadIdx.y][threadIdx.x]=0.0;
		s_data[BDIMY1+2*radius1-1-threadIdx.y][BDIMX1+2*radius1-1-threadIdx.x]=0.0;
		s_data[threadIdx.y][BDIMX1+2*radius1-1-threadIdx.x]=0.0;
		s_data[BDIMY1+2*radius1-1-threadIdx.y][threadIdx.x]=0.0;

		if((ix<dimx)&&(iz<dimz))
		{
				dimx=dimx+2*radius1;dimz=dimz+2*radius1;
				ix=ix+radius1;iz=iz+radius1;
				in_idx = ix*dimz+iz;//iz*dimx+ix;

				__syncthreads();

				s_data[tz][tx]=p_d[in_idx];

				if(threadIdx.y<radius1)
				{
						s_data[threadIdx.y][tx]=p_d[in_idx-radius1];//g_input[in_idx-radius1*dimx];//up
						s_data[threadIdx.y+BDIMY1+radius1][tx]=p_d[in_idx+BDIMY1];//g_input[in_idx+BDIMY1*dimx];//down
				}
				if(threadIdx.x<radius1)
				{
						s_data[tz][threadIdx.x]=p_d[in_idx-radius1*dimz];//g_input[in_idx-radius1];//left
						s_data[tz][threadIdx.x+BDIMX1+radius1]=p_d[in_idx+BDIMX1*dimz];//g_input[in_idx+BDIMX1];//right
				}

				__syncthreads();
			
				/*float    sumx=coe_d[1]*(s_data[tz][tx+1]-s_data[tz][tx]);
					sumx+=coe_d[2]*(s_data[tz][tx+2]-s_data[tz][tx-1]);
					sumx+=coe_d[3]*(s_data[tz][tx+3]-s_data[tz][tx-2]);
					sumx+=coe_d[4]*(s_data[tz][tx+4]-s_data[tz][tx-3]);
					sumx+=coe_d[5]*(s_data[tz][tx+5]-s_data[tz][tx-4]);
					sumx+=coe_d[6]*(s_data[tz][tx+6]-s_data[tz][tx-5]);

				float    sumz=coe_d[1]*(s_data[tz+1][tx]-s_data[tz][tx]);
					sumz+=coe_d[2]*(s_data[tz+2][tx]-s_data[tz-1][tx]);
					sumz+=coe_d[3]*(s_data[tz+3][tx]-s_data[tz-2][tx]);
					sumz+=coe_d[4]*(s_data[tz+4][tx]-s_data[tz-3][tx]);
					sumz+=coe_d[5]*(s_data[tz+5][tx]-s_data[tz-4][tx]);
					sumz+=coe_d[6]*(s_data[tz+6][tx]-s_data[tz-5][tx]);*/
////two choice
				float    sumx=coe_d[1]*(s_data[tz][tx+1]-s_data[tz][tx-1]);
					sumx+=coe_d[2]*(s_data[tz][tx+2]-s_data[tz][tx-2]);
					sumx+=coe_d[3]*(s_data[tz][tx+3]-s_data[tz][tx-3]);
					sumx+=coe_d[4]*(s_data[tz][tx+4]-s_data[tz][tx-4]);
					sumx+=coe_d[5]*(s_data[tz][tx+5]-s_data[tz][tx-5]);
					sumx+=coe_d[6]*(s_data[tz][tx+6]-s_data[tz][tx-6]);

				float    sumz=coe_d[1]*(s_data[tz+1][tx]-s_data[tz-1][tx]);
					sumz+=coe_d[2]*(s_data[tz+2][tx]-s_data[tz-2][tx]);
					sumz+=coe_d[3]*(s_data[tz+3][tx]-s_data[tz-3][tx]);
					sumz+=coe_d[4]*(s_data[tz+4][tx]-s_data[tz-4][tx]);
					sumz+=coe_d[5]*(s_data[tz+5][tx]-s_data[tz-5][tx]);
					sumz+=coe_d[6]*(s_data[tz+6][tx]-s_data[tz-6][tx]);	

				p1_d[in_idx]=(sumx*normalz1_d[in_idx]-sumz*normalx1_d[in_idx]);

		}

}

__global__ void real_divergence(float *p_d,float *p_x_d,float *p_z_d,float dx,float dz,int dimx,int dimz,float *coe_d)
{
		__shared__ float s_data[BDIMY1+2*radius1][BDIMX1+2*radius1];

		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		int tx = threadIdx.x+radius1;
		int tz = threadIdx.y+radius1;
		s_data[tz][tx]=0.0;
		s_data[threadIdx.y][threadIdx.x]=0.0;
		s_data[BDIMY1+2*radius1-1-threadIdx.y][BDIMX1+2*radius1-1-threadIdx.x]=0.0;
		s_data[threadIdx.y][BDIMX1+2*radius1-1-threadIdx.x]=0.0;
		s_data[BDIMY1+2*radius1-1-threadIdx.y][threadIdx.x]=0.0;


		if((ix<dimx)&&(iz<dimz))
		{
				dimx=dimx+2*radius1;dimz=dimz+2*radius1;
				ix=ix+radius1;iz=iz+radius1;
				in_idx = ix*dimz+iz;//iz*dimx+ix;

				__syncthreads();

				s_data[tz][tx]=p_d[in_idx];

				if(threadIdx.y<radius1)
				{
						s_data[threadIdx.y][tx]=p_d[in_idx-radius1];//g_input[in_idx-radius1*dimx];//up
						s_data[threadIdx.y+BDIMY1+radius1][tx]=p_d[in_idx+BDIMY1];//g_input[in_idx+BDIMY1*dimx];//down
				}
				if(threadIdx.x<radius1)
				{
						s_data[tz][threadIdx.x]=p_d[in_idx-radius1*dimz];//g_input[in_idx-radius1];//left
						s_data[tz][threadIdx.x+BDIMX1+radius1]=p_d[in_idx+BDIMX1*dimz];//g_input[in_idx+BDIMX1];//right
				}

				__syncthreads();
////one choice
				float    sumx=coe_d[1]*(s_data[tz][tx+1]-s_data[tz][tx-1]);
					sumx+=coe_d[2]*(s_data[tz][tx+2]-s_data[tz][tx-2]);
					sumx+=coe_d[3]*(s_data[tz][tx+3]-s_data[tz][tx-3]);
					sumx+=coe_d[4]*(s_data[tz][tx+4]-s_data[tz][tx-4]);
					sumx+=coe_d[5]*(s_data[tz][tx+5]-s_data[tz][tx-5]);
					sumx+=coe_d[6]*(s_data[tz][tx+6]-s_data[tz][tx-6]);

				float    sumz=coe_d[1]*(s_data[tz+1][tx]-s_data[tz-1][tx]);
					sumz+=coe_d[2]*(s_data[tz+2][tx]-s_data[tz-2][tx]);
					sumz+=coe_d[3]*(s_data[tz+3][tx]-s_data[tz-3][tx]);
					sumz+=coe_d[4]*(s_data[tz+4][tx]-s_data[tz-4][tx]);
					sumz+=coe_d[5]*(s_data[tz+5][tx]-s_data[tz-5][tx]);
					sumz+=coe_d[6]*(s_data[tz+6][tx]-s_data[tz-6][tx]);
					
					p_x_d[in_idx]=sumx/dx;
					p_z_d[in_idx]=sumz/dz;					
		}
}
__global__ void curl_old(float *s_d,float *s1_d,float *coe_d,float dx,float dz,int dimx,int dimz)
{
		__shared__ float s_data[BDIMY1+2*radius1][BDIMX1+2*radius1];

		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		int tx = threadIdx.x+radius1;
		int tz = threadIdx.y+radius1;
		s_data[tz][tx]=0.0;
		s_data[threadIdx.y][threadIdx.x]=0.0;
		s_data[BDIMY1+2*radius1-1-threadIdx.y][BDIMX1+2*radius1-1-threadIdx.x]=0.0;
		s_data[threadIdx.y][BDIMX1+2*radius1-1-threadIdx.x]=0.0;
		s_data[BDIMY1+2*radius1-1-threadIdx.y][threadIdx.x]=0.0;

		if((ix<dimx)&&(iz<dimz))
		{
				dimx=dimx+2*radius1;dimz=dimz+2*radius1;
				ix=ix+radius1;iz=iz+radius1;
				in_idx = ix*dimz+iz;//iz*dimx+ix;

				__syncthreads();

				s_data[tz][tx]=s_d[in_idx];

				if(threadIdx.y<radius1)
				{
						s_data[threadIdx.y][tx]=s_d[in_idx-radius1];//g_input[in_idx-radius1*dimx];//up
						s_data[threadIdx.y+BDIMY1+radius1][tx]=s_d[in_idx+BDIMY1];//g_input[in_idx+BDIMY1*dimx];//down
				}
				if(threadIdx.x<radius1)
				{
						s_data[tz][threadIdx.x]=s_d[in_idx-radius1*dimz];//g_input[in_idx-radius1];//left
						s_data[tz][threadIdx.x+BDIMX1+radius1]=s_d[in_idx+BDIMX1*dimz];//g_input[in_idx+BDIMX1];//right
				}

				__syncthreads();
			
				/* float    sumx=coe_d[1]*(s_data[tz][tx+1]-s_data[tz][tx]);
					sumx+=coe_d[2]*(s_data[tz][tx+2]-s_data[tz][tx-1]);
					sumx+=coe_d[3]*(s_data[tz][tx+3]-s_data[tz][tx-2]);
					sumx+=coe_d[4]*(s_data[tz][tx+4]-s_data[tz][tx-3]);
					sumx+=coe_d[5]*(s_data[tz][tx+5]-s_data[tz][tx-4]);
					sumx+=coe_d[6]*(s_data[tz][tx+6]-s_data[tz][tx-5]);

				float    sumz=coe_d[1]*(s_data[tz+1][tx]-s_data[tz][tx]);
					sumz+=coe_d[2]*(s_data[tz+2][tx]-s_data[tz-1][tx]);
					sumz+=coe_d[3]*(s_data[tz+3][tx]-s_data[tz-2][tx]);
					sumz+=coe_d[4]*(s_data[tz+4][tx]-s_data[tz-3][tx]);
					sumz+=coe_d[5]*(s_data[tz+5][tx]-s_data[tz-4][tx]);
					sumz+=coe_d[6]*(s_data[tz+6][tx]-s_data[tz-5][tx]);*/

				float    sumx=coe_d[1]*(s_data[tz][tx+1]-s_data[tz][tx-1]);
					sumx+=coe_d[2]*(s_data[tz][tx+2]-s_data[tz][tx-2]);
					sumx+=coe_d[3]*(s_data[tz][tx+3]-s_data[tz][tx-3]);
					sumx+=coe_d[4]*(s_data[tz][tx+4]-s_data[tz][tx-4]);
					sumx+=coe_d[5]*(s_data[tz][tx+5]-s_data[tz][tx-5]);
					sumx+=coe_d[6]*(s_data[tz][tx+6]-s_data[tz][tx-6]);

				float    sumz=coe_d[1]*(s_data[tz+1][tx]-s_data[tz-1][tx]);
					sumz+=coe_d[2]*(s_data[tz+2][tx]-s_data[tz-2][tx]);
					sumz+=coe_d[3]*(s_data[tz+3][tx]-s_data[tz-3][tx]);
					sumz+=coe_d[4]*(s_data[tz+4][tx]-s_data[tz-4][tx]);
					sumz+=coe_d[5]*(s_data[tz+5][tx]-s_data[tz-5][tx]);
					sumz+=coe_d[6]*(s_data[tz+6][tx]-s_data[tz-6][tx]);

				s1_d[in_idx]=sumx/dx;

		}

}

__global__ void curl_new(float *s_d,float *s1_d,float *normalx1_d,float *normalz1_d,float *coe_d,float dx,float dz,int dimx,int dimz)
{
		__shared__ float s_data[BDIMY1+2*radius1][BDIMX1+2*radius1];

		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		int tx = threadIdx.x+radius1;
		int tz = threadIdx.y+radius1;
		s_data[tz][tx]=0.0;
		s_data[threadIdx.y][threadIdx.x]=0.0;
		s_data[BDIMY1+2*radius1-1-threadIdx.y][BDIMX1+2*radius1-1-threadIdx.x]=0.0;
		s_data[threadIdx.y][BDIMX1+2*radius1-1-threadIdx.x]=0.0;
		s_data[BDIMY1+2*radius1-1-threadIdx.y][threadIdx.x]=0.0;

		if((ix<dimx)&&(iz<dimz))
		{
				dimx=dimx+2*radius1;dimz=dimz+2*radius1;
				ix=ix+radius1;iz=iz+radius1;
				in_idx = ix*dimz+iz;//iz*dimx+ix;

				__syncthreads();

				s_data[tz][tx]=s_d[in_idx];

				if(threadIdx.y<radius1)
				{
						s_data[threadIdx.y][tx]=s_d[in_idx-radius1];//g_input[in_idx-radius1*dimx];//up
						s_data[threadIdx.y+BDIMY1+radius1][tx]=s_d[in_idx+BDIMY1];//g_input[in_idx+BDIMY1*dimx];//down
				}
				if(threadIdx.x<radius1)
				{
						s_data[tz][threadIdx.x]=s_d[in_idx-radius1*dimz];//g_input[in_idx-radius1];//left
						s_data[tz][threadIdx.x+BDIMX1+radius1]=s_d[in_idx+BDIMX1*dimz];//g_input[in_idx+BDIMX1];//right
				}
				
				__syncthreads();
			
				/* float    sumx=coe_d[1]*(s_data[tz][tx+1]-s_data[tz][tx]);
					sumx+=coe_d[2]*(s_data[tz][tx+2]-s_data[tz][tx-1]);
					sumx+=coe_d[3]*(s_data[tz][tx+3]-s_data[tz][tx-2]);
					sumx+=coe_d[4]*(s_data[tz][tx+4]-s_data[tz][tx-3]);
					sumx+=coe_d[5]*(s_data[tz][tx+5]-s_data[tz][tx-4]);
					sumx+=coe_d[6]*(s_data[tz][tx+6]-s_data[tz][tx-5]);

				float    sumz=coe_d[1]*(s_data[tz+1][tx]-s_data[tz][tx]);
					sumz+=coe_d[2]*(s_data[tz+2][tx]-s_data[tz-1][tx]);
					sumz+=coe_d[3]*(s_data[tz+3][tx]-s_data[tz-2][tx]);
					sumz+=coe_d[4]*(s_data[tz+4][tx]-s_data[tz-3][tx]);
					sumz+=coe_d[5]*(s_data[tz+5][tx]-s_data[tz-4][tx]);
					sumz+=coe_d[6]*(s_data[tz+6][tx]-s_data[tz-5][tx]);*/
					
				float    sumx=coe_d[1]*(s_data[tz][tx+1]-s_data[tz][tx-1]);
					sumx+=coe_d[2]*(s_data[tz][tx+2]-s_data[tz][tx-2]);
					sumx+=coe_d[3]*(s_data[tz][tx+3]-s_data[tz][tx-3]);
					sumx+=coe_d[4]*(s_data[tz][tx+4]-s_data[tz][tx-4]);
					sumx+=coe_d[5]*(s_data[tz][tx+5]-s_data[tz][tx-5]);
					sumx+=coe_d[6]*(s_data[tz][tx+6]-s_data[tz][tx-6]);

				float    sumz=coe_d[1]*(s_data[tz+1][tx]-s_data[tz-1][tx]);
					sumz+=coe_d[2]*(s_data[tz+2][tx]-s_data[tz-2][tx]);
					sumz+=coe_d[3]*(s_data[tz+3][tx]-s_data[tz-3][tx]);
					sumz+=coe_d[4]*(s_data[tz+4][tx]-s_data[tz-4][tx]);
					sumz+=coe_d[5]*(s_data[tz+5][tx]-s_data[tz-5][tx]);
					sumz+=coe_d[6]*(s_data[tz+6][tx]-s_data[tz-6][tx]);

				s1_d[in_idx]=sumx*normalz1_d[in_idx]-sumz*normalx1_d[in_idx];

		}

}

__global__ void real_curl(float *s_d,float *s_x_d,float *s_z_d,float dx,float dz,int dimx,int dimz,float *coe_d)
{
		__shared__ float s_data[BDIMY1+2*radius1][BDIMX1+2*radius1];

		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		int tx = threadIdx.x+radius1;
		int tz = threadIdx.y+radius1;
		s_data[tz][tx]=0.0;
		s_data[threadIdx.y][threadIdx.x]=0.0;
		s_data[BDIMY1+2*radius1-1-threadIdx.y][BDIMX1+2*radius1-1-threadIdx.x]=0.0;
		s_data[threadIdx.y][BDIMX1+2*radius1-1-threadIdx.x]=0.0;
		s_data[BDIMY1+2*radius1-1-threadIdx.y][threadIdx.x]=0.0;


		if((ix<dimx)&&(iz<dimz))
		{
				dimx=dimx+2*radius1;dimz=dimz+2*radius1;
				ix=ix+radius1;iz=iz+radius1;
				in_idx = ix*dimz+iz;//iz*dimx+ix;

				__syncthreads();

				s_data[tz][tx]=s_d[in_idx];

				if(threadIdx.y<radius1)
				{
						s_data[threadIdx.y][tx]=s_d[in_idx-radius1];//g_input[in_idx-radius1*dimx];//up
						s_data[threadIdx.y+BDIMY1+radius1][tx]=s_d[in_idx+BDIMY1];//g_input[in_idx+BDIMY1*dimx];//down
				}
				if(threadIdx.x<radius1)
				{
						s_data[tz][threadIdx.x]=s_d[in_idx-radius1*dimz];//g_input[in_idx-radius1];//left
						s_data[tz][threadIdx.x+BDIMX1+radius1]=s_d[in_idx+BDIMX1*dimz];//g_input[in_idx+BDIMX1];//right
				}

				__syncthreads();

				float    sumx=coe_d[1]*(s_data[tz][tx+1]-s_data[tz][tx-1]);
					sumx+=coe_d[2]*(s_data[tz][tx+2]-s_data[tz][tx-2]);
					sumx+=coe_d[3]*(s_data[tz][tx+3]-s_data[tz][tx-3]);
					sumx+=coe_d[4]*(s_data[tz][tx+4]-s_data[tz][tx-4]);
					sumx+=coe_d[5]*(s_data[tz][tx+5]-s_data[tz][tx-5]);
					sumx+=coe_d[6]*(s_data[tz][tx+6]-s_data[tz][tx-6]);

				float    sumz=coe_d[1]*(s_data[tz+1][tx]-s_data[tz-1][tx]);
					sumz+=coe_d[2]*(s_data[tz+2][tx]-s_data[tz-2][tx]);
					sumz+=coe_d[3]*(s_data[tz+3][tx]-s_data[tz-3][tx]);
					sumz+=coe_d[4]*(s_data[tz+4][tx]-s_data[tz-4][tx]);
					sumz+=coe_d[5]*(s_data[tz+5][tx]-s_data[tz-5][tx]);
					sumz+=coe_d[6]*(s_data[tz+6][tx]-s_data[tz-6][tx]);
					
					s_x_d[in_idx]=sumz/dx;
					s_z_d[in_idx]=sumx/dz;
					
		}
}

__global__ void decom(float *wf1_d,float *f1_d,float *wfp_d,float *wfs_d,float *coe_d,int nx_append,int nz_append,float dx,float dz)
{
		__shared__ float s_data[BDIMY1+2*radius1][BDIMX1+2*radius1];
		__shared__ float s_data1[BDIMY1+2*radius1][BDIMX1+2*radius1];

		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		int tx = threadIdx.x+radius1;
		int tz = threadIdx.y+radius1;

		s_data[tz][tx]=0.0;
		s_data[threadIdx.y][threadIdx.x]=0.0;
		s_data[BDIMY1+2*radius1-1-threadIdx.y][BDIMX1+2*radius1-1-threadIdx.x]=0.0;
		s_data[threadIdx.y][BDIMX1+2*radius1-1-threadIdx.x]=0.0;
		s_data[BDIMY1+2*radius1-1-threadIdx.y][threadIdx.x]=0.0;

		s_data1[tz][tx]=0.0;
		s_data1[threadIdx.y][threadIdx.x]=0.0;
		s_data1[BDIMY1+2*radius1-1-threadIdx.y][BDIMX1+2*radius1-1-threadIdx.x]=0.0;
		s_data1[threadIdx.y][BDIMX1+2*radius1-1-threadIdx.x]=0.0;
		s_data1[BDIMY1+2*radius1-1-threadIdx.y][threadIdx.x]=0.0;

		if(ix<nx_append-2*radius1&&iz<nz_append-2*radius1)
		{
			ix=ix+radius1;
			iz=iz+radius1;			
			in_idx=ix*nz_append+iz;	

			__syncthreads();

			s_data[tz][tx]=wf1_d[in_idx];
			s_data1[tz][tx]=f1_d[in_idx];
				
			if(threadIdx.y<radius1)
			{
				s_data[threadIdx.y][threadIdx.x]=wf1_d[in_idx-radius1-radius1*nz_append];//up
				s_data[threadIdx.y][threadIdx.x+2*radius1]=wf1_d[in_idx-radius1+radius1*nz_append];//up
				s_data[threadIdx.y+BDIMY1+radius1][threadIdx.x]=wf1_d[in_idx+BDIMY1-radius1*nz_append];//down
				s_data[threadIdx.y+BDIMY1+radius1][threadIdx.x+2*radius1]=wf1_d[in_idx+BDIMY1+radius1*nz_append];//down

				s_data1[threadIdx.y][threadIdx.x]=f1_d[in_idx-radius1-radius1*nz_append];//up
				s_data1[threadIdx.y][threadIdx.x+2*radius1]=f1_d[in_idx-radius1+radius1*nz_append];//up
				s_data1[threadIdx.y+BDIMY1+radius1][threadIdx.x]=f1_d[in_idx+BDIMY1-radius1*nz_append];//down
				s_data1[threadIdx.y+BDIMY1+radius1][threadIdx.x+2*radius1]=f1_d[in_idx+BDIMY1+radius1*nz_append];//down

			}
			if(threadIdx.x<radius1)
			{
				s_data[tz][threadIdx.x]=wf1_d[in_idx-radius1*nz_append];//g_input[in_idx-radius1];//left
				s_data[tz][threadIdx.x+BDIMX1+radius1]=wf1_d[in_idx+BDIMX1*nz_append];//g_input[in_idx+BDIMX1];//right
				s_data1[tz][threadIdx.x]=f1_d[in_idx-radius1*nz_append];//g_input[in_idx-radius1];//left
				s_data1[tz][threadIdx.x+BDIMX1+radius1]=f1_d[in_idx+BDIMX1*nz_append];//g_input[in_idx+BDIMX1];//right
			}
			
			__syncthreads();

//p wave
		float   sum=coe_d[1]*(s_data[tz][tx+1]-s_data[tz][tx]);		
		       sum+=coe_d[2]*(s_data[tz][tx+2]-s_data[tz][tx-1]);
		       sum+=coe_d[3]*(s_data[tz][tx+3]-s_data[tz][tx-2]);
		       sum+=coe_d[4]*(s_data[tz][tx+4]-s_data[tz][tx-3]);
		       sum+=coe_d[5]*(s_data[tz][tx+5]-s_data[tz][tx-4]);
		       sum+=coe_d[6]*(s_data[tz][tx+6]-s_data[tz][tx-5]);

		float  sum1=coe_d[1]*(s_data1[tz][tx]-  s_data1[tz-1][tx]);
		      sum1+=coe_d[2]*(s_data1[tz+1][tx]-s_data1[tz-2][tx]);
		      sum1+=coe_d[3]*(s_data1[tz+2][tx]-s_data1[tz-3][tx]);
		      sum1+=coe_d[4]*(s_data1[tz+3][tx]-s_data1[tz-4][tx]);
		      sum1+=coe_d[5]*(s_data1[tz+4][tx]-s_data1[tz-5][tx]);
		      sum1+=coe_d[6]*(s_data1[tz+5][tx]-s_data1[tz-6][tx]);
		wfp_d[in_idx]=(1.0/dx)*sum+(1.0/dz)*sum1;

//s wave
		float  sum3=coe_d[1]*(s_data[tz+1][tx]-s_data[tz][tx]);
		      sum3+=coe_d[2]*(s_data[tz+2][tx]-s_data[tz-1][tx]);
		      sum3+=coe_d[3]*(s_data[tz+3][tx]-s_data[tz-2][tx]);
		      sum3+=coe_d[4]*(s_data[tz+4][tx]-s_data[tz-3][tx]);
		      sum3+=coe_d[5]*(s_data[tz+5][tx]-s_data[tz-4][tx]);
		      sum3+=coe_d[6]*(s_data[tz+6][tx]-s_data[tz-5][tx]);

		float   sum2=coe_d[1]*(s_data1[tz][tx]-  s_data1[tz][tx-1]);		
		       sum2+=coe_d[2]*(s_data1[tz][tx+1]-s_data1[tz][tx-2]);
		       sum2+=coe_d[3]*(s_data1[tz][tx+2]-s_data1[tz][tx-3]);
		       sum2+=coe_d[4]*(s_data1[tz][tx+3]-s_data1[tz][tx-4]);
		       sum2+=coe_d[5]*(s_data1[tz][tx+4]-s_data1[tz][tx-5]);
		       sum2+=coe_d[6]*(s_data1[tz][tx+5]-s_data1[tz][tx-6]);
		       
		/*float  sum3=coe_d[1]*(s_data[tz][tx]-s_data[tz-1][tx]);
		      sum3+=coe_d[2]*(s_data[tz+1][tx]-s_data[tz-2][tx]);
		      sum3+=coe_d[3]*(s_data[tz+2][tx]-s_data[tz-3][tx]);
		      sum3+=coe_d[4]*(s_data[tz+3][tx]-s_data[tz-4][tx]);
		      sum3+=coe_d[5]*(s_data[tz+4][tx]-s_data[tz-5][tx]);
		      sum3+=coe_d[6]*(s_data[tz+5][tx]-s_data[tz-6][tx]);

		float   sum2=coe_d[1]*(s_data1[tz][tx+1]-  s_data1[tz][tx]);		
		       sum2+=coe_d[2]*(s_data1[tz][tx+2]-s_data1[tz][tx-1]);
		       sum2+=coe_d[3]*(s_data1[tz][tx+3]-s_data1[tz][tx-2]);
		       sum2+=coe_d[4]*(s_data1[tz][tx+4]-s_data1[tz][tx-3]);
		       sum2+=coe_d[5]*(s_data1[tz][tx+5]-s_data1[tz][tx-4]);
		       sum2+=coe_d[6]*(s_data1[tz][tx+6]-s_data1[tz][tx-5]);*/

		wfs_d[in_idx]=(1.0/dz)*sum3-(1.0/dx)*sum2;

		}
}

__global__ void decom_new(float *vx1_d,float *vz1_d,float *p_d,float *s_d,float *velocity_d,float *velocity1_d,float *coe_d,int dimx,int dimz,float dx,float dz)
{
		__shared__ float s_data1[BDIMY1+2*radius1][BDIMX1+2*radius1];
		__shared__ float s_data2[BDIMY1+2*radius1][BDIMX1+2*radius1];

		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;
		
		float s_velocity,s_velocity1;

		int tx = threadIdx.x+radius1;
		int tz = threadIdx.y+radius1;

		s_data1[tz][tx]=0.0;
		s_data1[threadIdx.y][threadIdx.x]=0.0;
		s_data1[BDIMY1+2*radius1-1-threadIdx.y][BDIMX1+2*radius1-1-threadIdx.x]=0.0;
		s_data1[threadIdx.y][BDIMX1+2*radius1-1-threadIdx.x]=0.0;
		s_data1[BDIMY1+2*radius1-1-threadIdx.y][threadIdx.x]=0.0;

		s_data2[tz][tx]=0.0;
		s_data2[threadIdx.y][threadIdx.x]=0.0;
		s_data2[BDIMY1+2*radius1-1-threadIdx.y][BDIMX1+2*radius1-1-threadIdx.x]=0.0;
		s_data2[threadIdx.y][BDIMX1+2*radius1-1-threadIdx.x]=0.0;
		s_data2[BDIMY1+2*radius1-1-threadIdx.y][threadIdx.x]=0.0;

		if((ix<dimx)&&(iz<dimz))
		{
				dimx=dimx+2*radius1;dimz=dimz+2*radius1;
				ix=ix+radius1;iz=iz+radius1;
				in_idx = ix*dimz+iz;//iz*dimx+ix;

				__syncthreads();

				s_data1[tz][tx]=vx1_d[in_idx];
				s_data2[tz][tx]=vz1_d[in_idx];

				if(threadIdx.y<radius1)
				{
						s_data1[threadIdx.y][tx]=vx1_d[in_idx-radius1];//g_input[in_idx-radius1*dimx];//up
						s_data1[threadIdx.y+BDIMY1+radius1][tx]=vx1_d[in_idx+BDIMY1];//g_input[in_idx+BDIMY1*dimx];//down

						s_data2[threadIdx.y][tx]=vz1_d[in_idx-radius1];//g_input[in_idx-radius1*dimx];//up
						s_data2[threadIdx.y+BDIMY1+radius1][tx]=vz1_d[in_idx+BDIMY1];//g_input[in_idx+BDIMY1*dimx];//down
				}
				if(threadIdx.x<radius1)
				{
						s_data1[tz][threadIdx.x]=vx1_d[in_idx-radius1*dimz];//g_input[in_idx-radius1];//left
						s_data1[tz][threadIdx.x+BDIMX1+radius1]=vx1_d[in_idx+BDIMX1*dimz];//g_input[in_idx+BDIMX1];//right

						s_data2[tz][threadIdx.x]=vz1_d[in_idx-radius1*dimz];//g_input[in_idx-radius1];//left
						s_data2[tz][threadIdx.x+BDIMX1+radius1]=vz1_d[in_idx+BDIMX1*dimz];//g_input[in_idx+BDIMX1];//right
				}
				
				s_velocity=velocity_d[in_idx];
				s_velocity1=(velocity1_d[in_idx]+velocity1_d[in_idx+1]+velocity1_d[in_idx+dimz]+velocity1_d[in_idx+1+dimz])/4.0;
				__syncthreads();
				
//p wave
		float   sum=coe_d[1]*(s_data1[tz][tx+1]-s_data1[tz][tx]);		
		       sum+=coe_d[2]*(s_data1[tz][tx+2]-s_data1[tz][tx-1]);
		       sum+=coe_d[3]*(s_data1[tz][tx+3]-s_data1[tz][tx-2]);
		       sum+=coe_d[4]*(s_data1[tz][tx+4]-s_data1[tz][tx-3]);
		       sum+=coe_d[5]*(s_data1[tz][tx+5]-s_data1[tz][tx-4]);
		       sum+=coe_d[6]*(s_data1[tz][tx+6]-s_data1[tz][tx-5]);

		float  sum1=coe_d[1]*(s_data2[tz][tx]-s_data2[tz-1][tx]);
		      sum1+=coe_d[2]*(s_data2[tz+1][tx]-s_data2[tz-2][tx]);
		      sum1+=coe_d[3]*(s_data2[tz+2][tx]-s_data2[tz-3][tx]);
		      sum1+=coe_d[4]*(s_data2[tz+3][tx]-s_data2[tz-4][tx]);
		      sum1+=coe_d[5]*(s_data2[tz+4][tx]-s_data2[tz-5][tx]);
		      sum1+=coe_d[6]*(s_data2[tz+5][tx]-s_data2[tz-6][tx]);
		     
		/*  float   sum=coe_d[1]*(s_data1[tz][tx]-s_data1[tz][tx-1]);		
		       sum+=coe_d[2]*(s_data1[tz][tx+1]-s_data1[tz][tx-2]);
		       sum+=coe_d[3]*(s_data1[tz][tx+2]-s_data1[tz][tx-3]);
		       sum+=coe_d[4]*(s_data1[tz][tx+3]-s_data1[tz][tx-4]);
		       sum+=coe_d[5]*(s_data1[tz][tx+4]-s_data1[tz][tx-5]);
		       sum+=coe_d[6]*(s_data1[tz][tx+5]-s_data1[tz][tx-6]);

		float  sum1=coe_d[1]*(s_data2[tz+1][tx]-s_data2[tz][tx]);
		      sum1+=coe_d[2]*(s_data2[tz+2][tx]-s_data2[tz-1][tx]);
		      sum1+=coe_d[3]*(s_data2[tz+3][tx]-s_data2[tz-2][tx]);
		      sum1+=coe_d[4]*(s_data2[tz+4][tx]-s_data2[tz-3][tx]);
		      sum1+=coe_d[5]*(s_data2[tz+5][tx]-s_data2[tz-4][tx]);
		      sum1+=coe_d[6]*(s_data2[tz+6][tx]-s_data2[tz-5][tx]);*/
		      
			p_d[in_idx]=s_velocity*(1.0/dx)*sum+s_velocity*(1.0/dz)*sum1;

//s wave
		float  sum3=coe_d[1]*(s_data1[tz+1][tx]-s_data1[tz][tx]);
		      sum3+=coe_d[2]*(s_data1[tz+2][tx]-s_data1[tz-1][tx]);
		      sum3+=coe_d[3]*(s_data1[tz+3][tx]-s_data1[tz-2][tx]);
		      sum3+=coe_d[4]*(s_data1[tz+4][tx]-s_data1[tz-3][tx]);
		      sum3+=coe_d[5]*(s_data1[tz+5][tx]-s_data1[tz-4][tx]);
		      sum3+=coe_d[6]*(s_data1[tz+6][tx]-s_data1[tz-5][tx]);

		float   sum2=coe_d[1]*(s_data2[tz][tx]-s_data2[tz][tx-1]);		
		       sum2+=coe_d[2]*(s_data2[tz][tx+1]-s_data2[tz][tx-2]);
		       sum2+=coe_d[3]*(s_data2[tz][tx+2]-s_data2[tz][tx-3]);
		       sum2+=coe_d[4]*(s_data2[tz][tx+3]-s_data2[tz][tx-4]);
		       sum2+=coe_d[5]*(s_data2[tz][tx+4]-s_data2[tz][tx-5]);
		       sum2+=coe_d[6]*(s_data2[tz][tx+5]-s_data2[tz][tx-6]);
		 
		/* float  sum3=coe_d[1]*(s_data1[tz][tx]-s_data1[tz-1][tx]);
		      sum3+=coe_d[2]*(s_data1[tz+1][tx]-s_data1[tz-2][tx]);
		      sum3+=coe_d[3]*(s_data1[tz+2][tx]-s_data1[tz-3][tx]);
		      sum3+=coe_d[4]*(s_data1[tz+3][tx]-s_data1[tz-4][tx]);
		      sum3+=coe_d[5]*(s_data1[tz+4][tx]-s_data1[tz-5][tx]);
		      sum3+=coe_d[6]*(s_data1[tz+5][tx]-s_data1[tz-6][tx]);*/

		/*float   sum2=coe_d[1]*(s_data2[tz][tx+1]-s_data2[tz][tx]);		
		       sum2+=coe_d[2]*(s_data2[tz][tx+2]-s_data2[tz][tx-1]);
		       sum2+=coe_d[3]*(s_data2[tz][tx+3]-s_data2[tz][tx-2]);
		       sum2+=coe_d[4]*(s_data2[tz][tx+4]-s_data2[tz][tx-3]);
		       sum2+=coe_d[5]*(s_data2[tz][tx+5]-s_data2[tz][tx-4]);
		       sum2+=coe_d[6]*(s_data2[tz][tx+6]-s_data2[tz][tx-5]);*/
		       
			s_d[in_idx]=s_velocity1*(1.0/dz)*sum3-s_velocity1*(1.0/dx)*sum2;	
		}		
}

__global__ void save_all_wavefield(float *fws_d,float *vx2_d,int it,int dimx,int dimz)
{

		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;

		int in_idx;

		if(ix<dimx&&iz<dimz)
		{
			ix=ix+radius1;iz=iz+radius1;
			dimx=dimx+2*radius1;dimz=dimz+2*radius1;		
			in_idx=ix*dimz+iz;
						
			fws_d[it*dimx*dimz+in_idx]=vx2_d[in_idx];
		}
}

__global__ void set_all_wavefield(float *fws_d,float *vx2_d,int it,int dimx,int dimz)
{

		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;

		int in_idx;

		if(ix<dimx&&iz<dimz)
		{
			ix=ix+radius1;iz=iz+radius1;
			dimx=dimx+2*radius1;dimz=dimz+2*radius1;		
			in_idx=ix*dimz+iz;
						
			vx2_d[in_idx]=fws_d[it*dimx*dimz+in_idx];
		}
}

__global__ void filter_sign_new(float *signx_d,float *filter_signx_d,int dimx,int dimz,int scale)
{
		__shared__ float s_data[filter_scale+2*radius1][filter_scale+2*radius1];

		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		int tx = threadIdx.x+radius1;
		int tz = threadIdx.y+radius1;
		s_data[tz][tx]=0.0;
		s_data[threadIdx.y][threadIdx.x]=0.0;
		s_data[threadIdx.y+2*radius1-1][threadIdx.x+2*radius1-1]=0.0;
		s_data[threadIdx.y+2*radius1-1][threadIdx.x]=0.0;
		s_data[threadIdx.y][threadIdx.x+2*radius1-1]=0.0;

		if(ix<dimx&&iz<dimz)
		{
				dimx=dimx+2*radius1;dimz=dimz+2*radius1;
				ix=ix+radius1;iz=iz+radius1;
				in_idx=ix*dimz+iz;
				filter_signx_d[in_idx]=0.0;

				__syncthreads();
				
				s_data[tz][tx]=signx_d[in_idx];
		
				if(threadIdx.y<radius1)
				{
					s_data[threadIdx.y][tx]=signx_d[in_idx-radius1];
					s_data[threadIdx.y+filter_scale+radius1][tx]=signx_d[in_idx+filter_scale];
				}
				if(threadIdx.x<radius1)
				{
					s_data[tz][threadIdx.x]=signx_d[in_idx-radius1*dimz];
					s_data[tz][threadIdx.x+filter_scale+radius1]=signx_d[in_idx+filter_scale*dimz];
				}
			
				__syncthreads();

				for(int m=-filter_scale/2;m<=filter_scale/2;m++)
					for(int n=-filter_scale/2;n<=filter_scale/2;n++)
						filter_signx_d[in_idx]+=s_data[tz-m][tx-n];////m is replaced by n
		}		
}

__global__ void filter_sign_new_share(float *signx_d,float *filter_signx_d,int dimx,int dimz,int scale)
{
		__shared__ float s_data1[BDIMY1+2*radius1][BDIMX1+2*radius1];
		
		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;
		int tx = threadIdx.x+radius1;
		int tz = threadIdx.y+radius1;
		s_data1[tz][tx]=0.0;
		s_data1[threadIdx.y][threadIdx.x]=0.0;
		s_data1[BDIMY1+2*radius1-1-threadIdx.y][BDIMX1+2*radius1-1-threadIdx.x]=0.0;
		s_data1[threadIdx.y][BDIMX1+2*radius1-1-threadIdx.x]=0.0;
		s_data1[BDIMY1+2*radius1-1-threadIdx.y][threadIdx.x]=0.0;

		if(ix<dimx&&iz<dimz)
		{
				dimx=dimx+2*radius1;dimz=dimz+2*radius1;
				ix=ix+radius1;iz=iz+radius1;
				in_idx=ix*dimz+iz;
				filter_signx_d[in_idx]=0.0;
				
				__syncthreads();

				s_data1[tz][tx]=signx_d[in_idx];

				if(threadIdx.y<radius1)
				{
					s_data1[threadIdx.y][threadIdx.x]=signx_d[in_idx-radius1-radius1*dimz];//up
					s_data1[threadIdx.y][threadIdx.x+2*radius1]=signx_d[in_idx-radius1+radius1*dimz];//up
					s_data1[threadIdx.y+BDIMY1+radius1][threadIdx.x]=signx_d[in_idx+BDIMY1-radius1*dimz];//down
					s_data1[threadIdx.y+BDIMY1+radius1][threadIdx.x+2*radius1]=signx_d[in_idx+BDIMY1+radius1*dimz];//down
				}
				if(threadIdx.x<radius1)
				{
					s_data1[tz][threadIdx.x]=signx_d[in_idx-radius1*dimz];//g_input[in_idx-radius1];//left
					s_data1[tz][threadIdx.x+BDIMX1+radius1]=signx_d[in_idx+BDIMX1*dimz];//g_input[in_idx+BDIMX1];//right
				}
				__syncthreads();

				for(int m=-4;m<=4;m++)
					for(int n=-4;n<=4;n++)
						filter_signx_d[in_idx]+=s_data1[tz+m][tx+n];////m is replaced by n
		}		
}

__global__ void compare_sign(float *filter_signx_d,float *filter_signy_d,float *filter_signz_d,float *sign_d,int dimx,int dimz)
{
		int ix=blockIdx.x*blockDim.x+threadIdx.x;
		int iz=blockIdx.y*blockDim.y+threadIdx.y;

		int in_idx;

		if(ix<dimx&&iz<dimz)
		{
			ix=ix+radius1;
			iz=iz+radius1;
			dimx=dimx+2*radius1;
			dimz=dimz+2*radius1;
			in_idx=ix*dimz+iz;
			if(filter_signx_d[in_idx]>=filter_signy_d[in_idx]&&filter_signx_d[in_idx]>=filter_signz_d[in_idx])	sign_d[in_idx]= 1;
			if(filter_signy_d[in_idx]>filter_signx_d[in_idx]&&filter_signy_d[in_idx]>filter_signz_d[in_idx])	sign_d[in_idx]=-1;
			if(filter_signz_d[in_idx]>filter_signx_d[in_idx]&&filter_signz_d[in_idx]>filter_signy_d[in_idx])	sign_d[in_idx]= 0;
		}
}

__global__ void set_sign_basedon_polarization_ps(float *vxp1_d,float *vzp1_d,float *rvxs1_d,float *rvzs1_d,float *signx_d,float *signy_d,float *signz_d,int dimx,int dimz)
{
		int ix=blockIdx.x*blockDim.x+threadIdx.x;
		int iz=blockIdx.y*blockDim.y+threadIdx.y;

		int in_idx;
		float set;
////cross product
////one is x component , another is z component
		if(ix<dimx&&iz<dimz)
		{
			ix=ix+radius1;
			iz=iz+radius1;
			dimx=dimx+2*radius1;
			dimz=dimz+2*radius1;
			in_idx=ix*dimz+iz;
			if(vzp1_d[in_idx]>=0&&rvxs1_d[in_idx]>=0)	set=-1*(vxp1_d[in_idx]*rvxs1_d[in_idx]+vzp1_d[in_idx]*rvzs1_d[in_idx]);
			if(vzp1_d[in_idx]>=0&&rvxs1_d[in_idx]<0)	set=1*(vxp1_d[in_idx]*rvxs1_d[in_idx]+vzp1_d[in_idx]*rvzs1_d[in_idx]);
			if(vzp1_d[in_idx]<0&&rvxs1_d[in_idx]>0)	set=1*(vxp1_d[in_idx]*rvxs1_d[in_idx]+vzp1_d[in_idx]*rvzs1_d[in_idx]);
			if(vzp1_d[in_idx]<0&&rvxs1_d[in_idx]<0)	set=-1*(vxp1_d[in_idx]*rvxs1_d[in_idx]+vzp1_d[in_idx]*rvzs1_d[in_idx]);
			
			if(set>0)	
				{
					signx_d[in_idx]=1;
					signy_d[in_idx]=0;
					signz_d[in_idx]=0;
				}
			if(set<0)	
				{
					signx_d[in_idx]=0;
					signy_d[in_idx]=1;
					signz_d[in_idx]=0;
				}
			if(set==0)	
				{
					signx_d[in_idx]=0;
					signy_d[in_idx]=0;
					signz_d[in_idx]=1;
				}
		}
}

__global__ void set_sign_basedon_polarization_sp(float *vxs1_d,float *vzs1_d,float *rvxp1_d,float *rvzp1_d,float *signx_d,float *signy_d,float *signz_d,int dimx,int dimz)
{
		int ix=blockIdx.x*blockDim.x+threadIdx.x;
		int iz=blockIdx.y*blockDim.y+threadIdx.y;

		int in_idx;
		float set;

		if(ix<dimx&&iz<dimz)
		{
			ix=ix+radius1;
			iz=iz+radius1;
			dimx=dimx+2*radius1;
			dimz=dimz+2*radius1;
			in_idx=ix*dimz+iz;
////cross product
////one is x component , another is z component
			if(vxs1_d[in_idx]<=0&&rvzp1_d[in_idx]<=0)	set=(vzs1_d[in_idx]*rvzp1_d[in_idx]+vxs1_d[in_idx]*rvxp1_d[in_idx]);
			if(vxs1_d[in_idx]<0&&rvzp1_d[in_idx]>0)	set=-1*(vzs1_d[in_idx]*rvzp1_d[in_idx]+vxs1_d[in_idx]*rvxp1_d[in_idx]);
			if(vxs1_d[in_idx]>0&&rvzp1_d[in_idx]<0)	set=-1*(vzs1_d[in_idx]*rvzp1_d[in_idx]+vxs1_d[in_idx]*rvxp1_d[in_idx]);
			if(vxs1_d[in_idx]>0&&rvzp1_d[in_idx]>0)	set=(vzs1_d[in_idx]*rvzp1_d[in_idx]+vxs1_d[in_idx]*rvxp1_d[in_idx]);
			if(set>0)	
				{
					signx_d[in_idx]=1;
					signy_d[in_idx]=0;
					signz_d[in_idx]=0;
				}
			if(set<0)	
				{
					signx_d[in_idx]=0;
					signy_d[in_idx]=1;
					signz_d[in_idx]=0;
				}
			if(set==0)	
				{
					signx_d[in_idx]=0;
					signy_d[in_idx]=0;
					signz_d[in_idx]=1;
				}
		}
}

//cross-product for sign
__global__ void set_sign_forps(float *poyn_x_d,float *poyn_z_d,float *poyn_rx_d,float *poyn_rz_d,float *signx_d,float *signy_d,float *signz_d,int dimx,int dimz)
{
		int ix=blockIdx.x*blockDim.x+threadIdx.x;
		int iz=blockIdx.y*blockDim.y+threadIdx.y;

		int in_idx;
		float set;

		if(ix<dimx&&iz<dimz)
		{
			ix=ix+radius1;
			iz=iz+radius1;
			dimx=dimx+2*radius1;
			dimz=dimz+2*radius1;
			in_idx=ix*dimz+iz;
			set=10000000*poyn_x_d[in_idx]*poyn_rz_d[in_idx]-10000000*poyn_z_d[in_idx]*poyn_rx_d[in_idx];
//cross-product for sign			
			if(set>0)	
				{
					signx_d[in_idx]=1;
					signy_d[in_idx]=0;
					signz_d[in_idx]=0;
				}
			if(set<0)	
				{
					signx_d[in_idx]=0;
					signy_d[in_idx]=1;
					signz_d[in_idx]=0;
				}
			if(set==0)	
				{
					signx_d[in_idx]=0;
					signy_d[in_idx]=0;
					signz_d[in_idx]=1;
				}
		}
}

__global__ void normalized(float *normalx_d,float *normalz_d,int dimx,int dimz)
{
		int ix=blockIdx.x*blockDim.x+threadIdx.x;
		int iz=blockIdx.y*blockDim.y+threadIdx.y;

		int in_idx;
		float sum;
		if(ix<dimx&&iz<dimz)
		{
			dimx=dimx+2*radius1;dimz=dimz+2*radius1;
			ix=ix+radius1;iz=iz+radius1;
			in_idx=ix*dimz+iz;
			sum=sqrt(normalx_d[in_idx]*normalx_d[in_idx]+normalz_d[in_idx]*normalz_d[in_idx]);
			if(sum!=0)
			{
				normalx_d[in_idx]=normalx_d[in_idx]/sum;
				normalz_d[in_idx]=normalz_d[in_idx]/sum;
			}
		}
}
__global__ void caculate_normal_basedon_resultpp(float *result_h_d,float *result_old_d,float *normalx1_d,float *normalz1_d,int nx,int nz,int dimx,int dimz,int boundary_left,int boundary_up,float *coe)
{
		int ix=blockIdx.x*blockDim.x+threadIdx.x;
		int iz=blockIdx.y*blockDim.y+threadIdx.y;

		int in_idx;
		int in_idx1;
		
		if(ix<nx-2*radius1&&iz<nz-2*radius1)
		{
			ix=ix+radius1;iz=iz+radius1;
			in_idx1=ix*nz+iz;
			in_idx=(ix+boundary_left)*dimz+boundary_up+iz;

			float  sumx_h=coe[1]*(result_h_d[in_idx1+1*dimz]-result_h_d[in_idx1]);
				sumx_h+=coe[2]*(result_h_d[in_idx1+2*dimz]-result_h_d[in_idx1-1*dimz]);
				sumx_h+=coe[3]*(result_h_d[in_idx1+3*dimz]-result_h_d[in_idx1-2*dimz]);
				sumx_h+=coe[4]*(result_h_d[in_idx1+4*dimz]-result_h_d[in_idx1-3*dimz]);
				sumx_h+=coe[5]*(result_h_d[in_idx1+5*dimz]-result_h_d[in_idx1-4*dimz]);
				sumx_h+=coe[6]*(result_h_d[in_idx1+6*dimz]-result_h_d[in_idx1-5*dimz]);
			float  sumz_h=coe[1]*(result_h_d[in_idx1+1]-result_h_d[in_idx1]);
				sumz_h+=coe[2]*(result_h_d[in_idx1+2]-result_h_d[in_idx1-1]);
				sumz_h+=coe[3]*(result_h_d[in_idx1+3]-result_h_d[in_idx1-2]);
				sumz_h+=coe[4]*(result_h_d[in_idx1+4]-result_h_d[in_idx1-3]);
				sumz_h+=coe[5]*(result_h_d[in_idx1+5]-result_h_d[in_idx1-4]);
				sumz_h+=coe[6]*(result_h_d[in_idx1+6]-result_h_d[in_idx1-5]);
			
			float  sumx_o=coe[1]*(result_old_d[in_idx1+1*dimz]-result_old_d[in_idx1]);
				sumx_o+=coe[2]*(result_old_d[in_idx1+2*dimz]-result_old_d[in_idx1-1*dimz]);
				sumx_o+=coe[3]*(result_old_d[in_idx1+3*dimz]-result_old_d[in_idx1-2*dimz]);
				sumx_o+=coe[4]*(result_old_d[in_idx1+4*dimz]-result_old_d[in_idx1-3*dimz]);
				sumx_o+=coe[5]*(result_old_d[in_idx1+5*dimz]-result_old_d[in_idx1-4*dimz]);
				sumx_o+=coe[6]*(result_old_d[in_idx1+6*dimz]-result_old_d[in_idx1-5*dimz]);
			float  sumz_o=coe[1]*(result_old_d[in_idx1+1]-result_old_d[in_idx1]);
				sumz_o+=coe[2]*(result_old_d[in_idx1+2]-result_old_d[in_idx1-1]);
				sumz_o+=coe[3]*(result_old_d[in_idx1+3]-result_old_d[in_idx1-2]);
				sumz_o+=coe[4]*(result_old_d[in_idx1+4]-result_old_d[in_idx1-3]);
				sumz_o+=coe[5]*(result_old_d[in_idx1+5]-result_old_d[in_idx1-4]);
				sumz_o+=coe[6]*(result_old_d[in_idx1+6]-result_old_d[in_idx1-5]);
				
			/*float  sumx_h=(result_h_d[in_idx1+1*dimz]-result_h_d[in_idx1]);
			float  sumz_h=(result_h_d[in_idx1+1]-result_h_d[in_idx1]);
			float  sumx_o=(result_old_d[in_idx1+1*dimz]-result_old_d[in_idx1]);
			float  sumz_o=(result_old_d[in_idx1+1]-result_old_d[in_idx1]);*/
			
				normalx1_d[in_idx]=(sumx_o*result_h_d[in_idx1]-sumx_h*result_old_d[in_idx1]);//(result_h_d[in_idx1]*result_h_d[in_idx1]+result_old_d[in_idx1]*result_old_d[in_idx1]);
				normalz1_d[in_idx]=(sumz_o*result_h_d[in_idx1]-sumz_h*result_old_d[in_idx1]);//(result_h_d[in_idx1]*result_h_d[in_idx1]+result_old_d[in_idx1]*result_old_d[in_idx1]);
		}
}

__global__ void caculate_normal_basedon_resultpp_new(float *result_h_d,float *result_old_d,float *normalx1_d,float *normalz1_d,int nx,int nz,int dimx,int dimz,int boundary_left,int boundary_up,float *coe)
{
		int ix=blockIdx.x*blockDim.x+threadIdx.x;
		int iz=blockIdx.y*blockDim.y+threadIdx.y;

		int in_idx;
		
		if(ix<dimx&&iz<dimz)
		{
			ix=ix+radius1;
			iz=iz+radius1;
			dimx=dimx+2*radius1;
			dimz=dimz+2*radius1;
			in_idx=ix*dimz+iz;
///////note that  :   1-0,2- -1,3- -2,4- -3;
			/*float  sumx_h=coe[1]*(result_h_d[in_idx+1*dimz]-result_h_d[in_idx]);
				sumx_h+=coe[2]*(result_h_d[in_idx+2*dimz]-result_h_d[in_idx-1*dimz]);
				sumx_h+=coe[3]*(result_h_d[in_idx+3*dimz]-result_h_d[in_idx-2*dimz]);
				sumx_h+=coe[4]*(result_h_d[in_idx+4*dimz]-result_h_d[in_idx-3*dimz]);
				sumx_h+=coe[5]*(result_h_d[in_idx+5*dimz]-result_h_d[in_idx-4*dimz]);
				sumx_h+=coe[6]*(result_h_d[in_idx+6*dimz]-result_h_d[in_idx-5*dimz]);
			float  sumz_h=coe[1]*(result_h_d[in_idx+1]-result_h_d[in_idx]);
				sumz_h+=coe[2]*(result_h_d[in_idx+2]-result_h_d[in_idx-1]);
				sumz_h+=coe[3]*(result_h_d[in_idx+3]-result_h_d[in_idx-2]);
				sumz_h+=coe[4]*(result_h_d[in_idx+4]-result_h_d[in_idx-3]);
				sumz_h+=coe[5]*(result_h_d[in_idx+5]-result_h_d[in_idx-4]);
				sumz_h+=coe[6]*(result_h_d[in_idx+6]-result_h_d[in_idx-5]);
			
			float  sumx_o=coe[1]*(result_old_d[in_idx+1*dimz]-result_old_d[in_idx]);
				sumx_o+=coe[2]*(result_old_d[in_idx+2*dimz]-result_old_d[in_idx-1*dimz]);
				sumx_o+=coe[3]*(result_old_d[in_idx+3*dimz]-result_old_d[in_idx-2*dimz]);
				sumx_o+=coe[4]*(result_old_d[in_idx+4*dimz]-result_old_d[in_idx-3*dimz]);
				sumx_o+=coe[5]*(result_old_d[in_idx+5*dimz]-result_old_d[in_idx-4*dimz]);
				sumx_o+=coe[6]*(result_old_d[in_idx+6*dimz]-result_old_d[in_idx-5*dimz]);
			float  sumz_o=coe[1]*(result_old_d[in_idx+1]-result_old_d[in_idx]);
				sumz_o+=coe[2]*(result_old_d[in_idx+2]-result_old_d[in_idx-1]);
				sumz_o+=coe[3]*(result_old_d[in_idx+3]-result_old_d[in_idx-2]);
				sumz_o+=coe[4]*(result_old_d[in_idx+4]-result_old_d[in_idx-3]);
				sumz_o+=coe[5]*(result_old_d[in_idx+5]-result_old_d[in_idx-4]);
				sumz_o+=coe[6]*(result_old_d[in_idx+6]-result_old_d[in_idx-5]);*/
			float sumx_h=coe[0]*result_h_d[in_idx];
				sumx_h+=coe[1]*(result_h_d[in_idx+1*dimz]-result_h_d[in_idx-1*dimz]);
				sumx_h+=coe[2]*(result_h_d[in_idx+2*dimz]-result_h_d[in_idx-2*dimz]);
				sumx_h+=coe[3]*(result_h_d[in_idx+3*dimz]-result_h_d[in_idx-3*dimz]);
				sumx_h+=coe[4]*(result_h_d[in_idx+4*dimz]-result_h_d[in_idx-4*dimz]);
				sumx_h+=coe[5]*(result_h_d[in_idx+5*dimz]-result_h_d[in_idx-5*dimz]);
				sumx_h+=coe[6]*(result_h_d[in_idx+6*dimz]-result_h_d[in_idx-6*dimz]);
				
			float sumz_h=coe[0]*result_h_d[in_idx];
				sumz_h+=coe[1]*(result_h_d[in_idx+1]-result_h_d[in_idx-1]);
				sumz_h+=coe[2]*(result_h_d[in_idx+2]-result_h_d[in_idx-2]);
				sumz_h+=coe[3]*(result_h_d[in_idx+3]-result_h_d[in_idx-3]);
				sumz_h+=coe[4]*(result_h_d[in_idx+4]-result_h_d[in_idx-4]);
				sumz_h+=coe[5]*(result_h_d[in_idx+5]-result_h_d[in_idx-5]);
				sumz_h+=coe[6]*(result_h_d[in_idx+6]-result_h_d[in_idx-6]);
			
			float sumx_o=coe[0]*result_old_d[in_idx];
				sumx_o+=coe[1]*(result_old_d[in_idx+1*dimz]-result_old_d[in_idx-1*dimz]);
				sumx_o+=coe[2]*(result_old_d[in_idx+2*dimz]-result_old_d[in_idx-2*dimz]);
				sumx_o+=coe[3]*(result_old_d[in_idx+3*dimz]-result_old_d[in_idx-3*dimz]);
				sumx_o+=coe[4]*(result_old_d[in_idx+4*dimz]-result_old_d[in_idx-4*dimz]);
				sumx_o+=coe[5]*(result_old_d[in_idx+5*dimz]-result_old_d[in_idx-5*dimz]);
				sumx_o+=coe[6]*(result_old_d[in_idx+6*dimz]-result_old_d[in_idx-6*dimz]);
				
			float sumz_o=coe[0]*result_old_d[in_idx];
				sumz_o+=coe[1]*(result_old_d[in_idx+1]-result_old_d[in_idx-1]);
				sumz_o+=coe[2]*(result_old_d[in_idx+2]-result_old_d[in_idx-2]);
				sumz_o+=coe[3]*(result_old_d[in_idx+3]-result_old_d[in_idx-3]);
				sumz_o+=coe[4]*(result_old_d[in_idx+4]-result_old_d[in_idx-4]);
				sumz_o+=coe[5]*(result_old_d[in_idx+5]-result_old_d[in_idx-5]);
				sumz_o+=coe[6]*(result_old_d[in_idx+6]-result_old_d[in_idx-6]);
				
			/*float  sumx_h=(result_h_d[in_idx+1*dimz]-result_h_d[in_idx-1*dimz]);
			float  sumz_h=(result_h_d[in_idx+1]-result_h_d[in_idx-1]);
			float  sumx_o=(result_old_d[in_idx+1*dimz]-result_old_d[in_idx-1*dimz]);
			float  sumz_o=(result_old_d[in_idx+1]-result_old_d[in_idx-1]);*/
			
				normalx1_d[in_idx]=(sumx_o*result_h_d[in_idx]-sumx_h*result_old_d[in_idx])/(result_h_d[in_idx]*result_h_d[in_idx]+result_old_d[in_idx]*result_old_d[in_idx]);
				normalz1_d[in_idx]=(sumz_o*result_h_d[in_idx]-sumz_h*result_old_d[in_idx])/(result_h_d[in_idx]*result_h_d[in_idx]+result_old_d[in_idx]*result_old_d[in_idx]);
		}
}

__global__ void caculate_normal(float *normalx_d,float *normalz_d,float *poyn_px_d,float *poyn_pz_d,float *poyn_rpx_d,float *poyn_rpz_d,int dimx,int dimz)
{
		int ix=blockIdx.x*blockDim.x+threadIdx.x;
		int iz=blockIdx.y*blockDim.y+threadIdx.y;

		int in_idx;

		if(ix<dimx&&iz<dimz)
		{
			ix=ix+radius;
			iz=iz+radius;
			dimx=dimx+2*radius;
			dimz=dimz+2*radius;
			in_idx=ix*dimz+iz;

			normalx_d[in_idx]=poyn_rpx_d[in_idx]-poyn_px_d[in_idx];
			normalz_d[in_idx]=poyn_rpz_d[in_idx]-poyn_pz_d[in_idx];
		}
}

__global__ void caculate_angle_open(float *angle_open_d,float *poyn_px_d,float *poyn_pz_d,float *poyn_rpx_d,float *poyn_rpz_d,int dimx,int dimz)
{
		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;
		double sum;

		if(ix<dimx&&iz<dimz)
		{
			dimx=dimx+2*radius1;dimz=dimz+2*radius1;
			ix=ix+radius1;iz=iz+radius1;
			in_idx=ix*dimz+iz;
			angle_open_d[in_idx]=0.0;
			sum=0.0;
					
			sum=(poyn_px_d[in_idx]*poyn_rpx_d[in_idx]+poyn_pz_d[in_idx]*poyn_rpz_d[in_idx]);
			
			angle_open_d[in_idx]=90*float(acos(sum))/pai;
		}
}

__global__ void caculate_angle_pp_real(float *angle_pp_d,float *normal_angle_d,int dimx,int dimz)
{
		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		if(ix<dimx&&iz<dimz)
		{
				dimx=dimx+2*radius1;dimz=dimz+2*radius1;
				ix=ix+radius1;iz=iz+radius1;
				in_idx=ix*dimz+iz;
				
				angle_pp_d[in_idx]=angle_pp_d[in_idx]-normal_angle_d[in_idx];
		}
}


__global__ void fwd_smooth(float *input_d,int dimx,int dimz)
{
		__shared__ float s_data[BDIMY1+2*radius1][BDIMX1+2*radius1];

		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		int tx = threadIdx.x+radius1;
		int tz = threadIdx.y+radius1;
		s_data[tz][tx]=0.0;
		s_data[threadIdx.y][threadIdx.x]=0.0;
		s_data[BDIMY1+2*radius1-1-threadIdx.y][BDIMX1+2*radius1-1-threadIdx.x]=0.0;
		s_data[threadIdx.y][BDIMX1+2*radius1-1-threadIdx.x]=0.0;
		s_data[BDIMY1+2*radius1-1-threadIdx.y][threadIdx.x]=0.0;

		if((ix<dimx)&&(iz<dimz))
		{
				dimx=dimx+2*radius1;dimz=dimz+2*radius1;
				ix=ix+radius1;iz=iz+radius1;
				in_idx = ix*dimz+iz;//iz*dimx+ix;

				__syncthreads();

				s_data[tz][tx]=input_d[in_idx];

				if(threadIdx.y<radius1)
				{
						s_data[threadIdx.y][tx]=input_d[in_idx-radius1];//g_input[in_idx-radius1*dimx];//up
						s_data[threadIdx.y+BDIMY1+radius1][tx]=input_d[in_idx+BDIMY1];//g_input[in_idx+BDIMY1*dimx];//down
				}
				if(threadIdx.x<radius1)
				{
						s_data[tz][threadIdx.x]=input_d[in_idx-radius1*dimz];//g_input[in_idx-radius1];//left
						s_data[tz][threadIdx.x+BDIMX1+radius1]=input_d[in_idx+BDIMX1*dimz];//g_input[in_idx+BDIMX1];//right
				}
				
				__syncthreads();

				input_d[in_idx]=(s_data[tz][tx]+s_data[tz+1][tx+1]+s_data[tz+1][tx-1]+s_data[tz-1][tx+1]+s_data[tz-1][tx-1])/5.0;
		}
}

__global__ void set_adcigs_imagingpp_angle(float *p_d,float *rp_d,float *p_adcigs_pp_d,float *n_adcigs_pp_d,float *imageup_d,float *angle_pp_d,float *angle_open_d,int source_x_cord,int nx,int nz,int nx_append,int nz_append,int boundary_up,int boundary_left,int angle_num,int dangle)
{
		int ix=blockIdx.x*blockDim.x+threadIdx.x;
		int iz=blockIdx.y*blockDim.y+threadIdx.y;

		int in_idx;
		int set;
		int r_angle;

		if(ix<nx&&iz<nz&&iz>10)
		//if(ix<nx&&iz<nz&&iz>10&&iz>0.8*(ix-source_x_cord)&&iz>0.8*(source_x_cord-ix))
		{
			in_idx=(boundary_left+ix)*nz_append+iz+boundary_up;
			imageup_d[ix*nz+iz]=0.0;
				
			//if(angle_open_d[in_idx]<=60)	imageup_d[ix*nz+iz]=p_d[in_idx]*rp_d[in_idx];
			//if(poyn_z_d[in_idx]>=0&&poyn_rz_d[in_idx]<=0)	imageup_d[ix*nz+iz]=p_d[in_idx]*rp_d[in_idx];
			//if(poyn_z_d[in_idx]<=0&&poyn_rz_d[in_idx]>=0)	imageup_d[ix*nz+iz]=p_d[in_idx]*rp_d[in_idx];	
			if(angle_open_d[in_idx]<=90)	imageup_d[ix*nz+iz]=p_d[in_idx]*rp_d[in_idx]*float(float(cos(double(pai*angle_open_d[in_idx]/180)))*float(cos(double(pai*angle_open_d[in_idx]/180)))*float(cos(double(pai*angle_open_d[in_idx]/180))));
			
			r_angle=int((fabs(angle_pp_d[in_idx])+0.5)/dangle);
///every time  only one angle need to caculate		
			set=r_angle*nx*nz+ix*nz+iz;
			if(r_angle<angle_num&&angle_pp_d[in_idx]>=0) p_adcigs_pp_d[set]+=imageup_d[ix*nz+iz]*exp(-(fabs(angle_pp_d[in_idx])-r_angle*dangle)*(fabs(angle_pp_d[in_idx])-r_angle*dangle)/8);
			if(r_angle<angle_num&&angle_pp_d[in_idx]<0) n_adcigs_pp_d[set]+=imageup_d[ix*nz+iz]*exp(-(fabs(angle_pp_d[in_idx])-r_angle*dangle)*(fabs(angle_pp_d[in_idx])-r_angle*dangle)/8);
		}
		
}

__global__ void set_adcigs_imagingps_angle(float *p_d,float *rs_d,float *p_adcigs_ps_d,float *n_adcigs_ps_d,float *imageup_d,float *angle_pp_d,float *angle_open_d,int source_x_cord,int nx,int nz,int nx_append,int nz_append,int boundary_up,int boundary_left,int angle_num,int dangle)
//angle_pp_d,p_d,rs_d,adcigs_ps_d,rimageupps_d,rimagedownps_d,angle_pp_d,nx,nz,nx_append,nz_append,boundary_up,boundary_left,angle_num,dangle
{
		int ix=blockIdx.x*blockDim.x+threadIdx.x;
		int iz=blockIdx.y*blockDim.y+threadIdx.y;

		int in_idx;
		int set;
		int r_angle;

		if(ix<nx&&iz<nz&&iz>10&&iz>0.8*(ix-source_x_cord)&&iz>0.8*(source_x_cord-ix))
		{
			in_idx=(boundary_left+ix)*nz_append+iz+boundary_up;
			imageup_d[ix*nz+iz]=0.0;
			
			if(angle_pp_d[in_idx]>=0)
			{
				imageup_d[ix*nz+iz]=p_d[in_idx]*rs_d[in_idx];
			}
			if(angle_pp_d[in_idx]<0)
			{
				imageup_d[ix*nz+iz]=-1*p_d[in_idx]*rs_d[in_idx];
			}	
			
			r_angle=int((fabs(angle_pp_d[in_idx])+0.5)/dangle);
	
			set=r_angle*nx*nz+ix*nz+iz;
			
			if(r_angle<angle_num&&angle_pp_d[in_idx]>=0) p_adcigs_ps_d[set]+=imageup_d[ix*nz+iz]*exp(-(fabs(angle_pp_d[in_idx])-r_angle*dangle)*(fabs(angle_pp_d[in_idx])-r_angle*dangle)/8);
			if(r_angle<angle_num&&angle_pp_d[in_idx]<0) n_adcigs_ps_d[set]+=imageup_d[ix*nz+iz]*exp(-(fabs(angle_pp_d[in_idx])-r_angle*dangle)*(fabs(angle_pp_d[in_idx])-r_angle*dangle)/8);
		
		}
}

__global__ void set_adcigs_imagingpp_angle_new(float *vxp_d,float *vzp_d,float *rvxp_d,float *rvzp_d,float *p_adcigs_pp_d,float *n_adcigs_pp_d,float *imageup_d,float *angle_pp_d,float *angle_open_d,int source_x_cord,int nx,int nz,int nx_append,int nz_append,int boundary_up,int boundary_left,int angle_num,int dangle)
//(vxp1_d,vzp1_d,rvxp1_d,rvzp1_d,p_adcigs_pp_d,n_adcigs_pp_d,rimageuppp_d,rimagedownpp_d,angle_pp_d,angle_open_d,nx,nz,nx_append,nz_append,boundary_up,boundary_left,angle_num,dangle);
{
		int ix=blockIdx.x*blockDim.x+threadIdx.x;
		int iz=blockIdx.y*blockDim.y+threadIdx.y;

		int in_idx;
		int set;
		int r_angle;
		
		//float sign;

		if(ix<nx&&iz<nz&&iz>10)
		//if(ix<nx&&iz<nz&&iz>10&&iz>0.8*(ix-source_x_cord)&&iz>0.8*(source_x_cord-ix))
		{
			in_idx=(boundary_left+ix)*nz_append+iz+boundary_up;
			imageup_d[ix*nz+iz]=0.0;
			
			//if(vzp_d[in_idx]*rvzp_d[in_idx]<=0)	sign=-1;
			//if(vzp_d[in_idx]*rvzp_d[in_idx]>0)	sign=+1;
				
			//if(angle_open_d[in_idx]<=90)	imageup_d[ix*nz+iz]=(fabs(vxp_d[in_idx]*rvxp_d[in_idx])*sign+vzp_d[in_idx]*rvzp_d[in_idx])*float(float(cos(double(pai*angle_open_d[in_idx]/180)))*float(cos(double(pai*angle_open_d[in_idx]/180)))*float(cos(double(pai*angle_open_d[in_idx]/180))));
			
			if(angle_open_d[in_idx]<=90)	imageup_d[ix*nz+iz]=(vxp_d[in_idx]*rvxp_d[in_idx]+vzp_d[in_idx]*rvzp_d[in_idx])*float(float(cos(double(pai*angle_open_d[in_idx]/180)))*float(cos(double(pai*angle_open_d[in_idx]/180)))*float(cos(double(pai*angle_open_d[in_idx]/180))));
			
			//if(angle_open_d[in_idx]<=90)	imageup_d[ix*nz+iz]=(vzp_d[in_idx]*rvzp_d[in_idx])*float(float(cos(double(pai*angle_open_d[in_idx]/180)))*float(cos(double(pai*angle_open_d[in_idx]/180)))*float(cos(double(pai*angle_open_d[in_idx]/180))));
			
			r_angle=int((fabs(angle_pp_d[in_idx])+0.5)/dangle);
///every time  only one angle need to caculate		
			set=r_angle*nx*nz+ix*nz+iz;
			if(r_angle<angle_num&&angle_pp_d[in_idx]>=0) p_adcigs_pp_d[set]+=imageup_d[ix*nz+iz]*exp(-(fabs(angle_pp_d[in_idx])-r_angle*dangle)*(fabs(angle_pp_d[in_idx])-r_angle*dangle)/8);
			if(r_angle<angle_num&&angle_pp_d[in_idx]<0) n_adcigs_pp_d[set]+=imageup_d[ix*nz+iz]*exp(-(fabs(angle_pp_d[in_idx])-r_angle*dangle)*(fabs(angle_pp_d[in_idx])-r_angle*dangle)/8);
		}
}

__global__ void set_adcigs_imagingps_angle_new(float *vxp_d,float *vzp_d,float *rvxs_d,float *rvzs_d,float *p_adcigs_ps_d,float *n_adcigs_ps_d,float *imageup_d,float *angle_pp_d,float *angle_open_d,int source_x_cord,int nx,int nz,int nx_append,int nz_append,int boundary_up,int boundary_left,int angle_num,int dangle)
//(vxp1_d,vzp1_d,rvxs1_d,rvzs1_d,p_adcigs_pp_d,n_adcigs_pp_d,rimageuppp_d,rimagedownpp_d,angle_pp_d,angle_open_d,nx,nz,nx_append,nz_append,boundary_up,boundary_left,angle_num,dangle);
{
		int ix=blockIdx.x*blockDim.x+threadIdx.x;
		int iz=blockIdx.y*blockDim.y+threadIdx.y;

		int in_idx;
		int set;
		int r_angle;
		//float sign;

		if(ix<nx&&iz<nz&&iz>10&&iz>0.8*(ix-source_x_cord)&&iz>0.8*(source_x_cord-ix))
		{
			in_idx=(boundary_left+ix)*nz_append+iz+boundary_up;
			imageup_d[ix*nz+iz]=0.0;
			
			//if(vxp_d[in_idx]*rvxs_d[in_idx]<=0)	sign=-1;
			//if(vxp_d[in_idx]*rvxs_d[in_idx]>0)	sign=+1;
			
			//imageup_d[ix*nz+iz]=vxp_d[in_idx]*rvxs_d[in_idx]+sign*fabs(vzp_d[in_idx]*rvzs_d[in_idx]);
			
			imageup_d[ix*nz+iz]=vxp_d[in_idx]*rvxs_d[in_idx]+vzp_d[in_idx]*rvzs_d[in_idx];
			
			//imageup_d[ix*nz+iz]=vxp_d[in_idx]*rvxs_d[in_idx];
				
			r_angle=int((fabs(angle_pp_d[in_idx])+0.5)/dangle);
	
			set=r_angle*nx*nz+ix*nz+iz;
			
			if(r_angle<angle_num&&angle_pp_d[in_idx]>=0) p_adcigs_ps_d[set]+=imageup_d[ix*nz+iz]*exp(-(fabs(angle_pp_d[in_idx])-r_angle*dangle)*(fabs(angle_pp_d[in_idx])-r_angle*dangle)/8);
			if(r_angle<angle_num&&angle_pp_d[in_idx]<0) n_adcigs_ps_d[set]+=imageup_d[ix*nz+iz]*exp(-(fabs(angle_pp_d[in_idx])-r_angle*dangle)*(fabs(angle_pp_d[in_idx])-r_angle*dangle)/8);
		
		}
}

__global__ void imagingadd_angle(float *adcigs_pp_d,float *imagedown_d,int nx,int nz,int angle_num,float average)
{
		int ix=blockIdx.x*blockDim.x+threadIdx.x;
		int iz=blockIdx.y*blockDim.y+threadIdx.y;
		int set;		

		if(ix<nx&&iz<nz)
		{
			for(int iangle=0;iangle<angle_num;iangle++)
				{
					set=iangle*nx*nz+ix*nz+iz;		
					adcigs_pp_d[set]=adcigs_pp_d[set]/(imagedown_d[ix*nz+iz]+average);
				}
		}
}

__global__ void imagingadd_angle_new(float *adcigs_pp_d,float *imagedown_d,int nx,int nz,int angle_num,float average)
{
		int ix=blockIdx.x*blockDim.x+threadIdx.x;
		int iz=blockIdx.y*blockDim.y+threadIdx.y;
		int set;		

		if(ix<nx&&iz<nz)
		{
			for(int iangle=0;iangle<2*angle_num;iangle++)
				{
					set=iangle*nx*nz+ix*nz+iz;		
					adcigs_pp_d[set]=adcigs_pp_d[set]/(imagedown_d[ix*nz+iz]+average);
				}
		}
}

__global__ void output_someangle(float *adcigs_d,float *wf_d,int angle_start,int angle_end,int nx,int nz)
{
		int ix=blockIdx.x*blockDim.x+threadIdx.x;
		int iz=blockIdx.y*blockDim.y+threadIdx.y;

		int in_idx;
		if(ix<nx&&iz<nz)
		{
			for(int iangle=angle_start;iangle<angle_end;iangle++)
			{				
				in_idx=iangle*nx*nz+ix*nz+iz;
				wf_d[ix*nz+iz]+=adcigs_d[in_idx];
			}
		}
}

__global__ void output_cdpangle(float *adcigs_d,float *adcigs_cdp_d,int cdp_location,int angle_num,int dangle,int nx,int nz)
{
		int iangle=blockIdx.x*blockDim.x+threadIdx.x;
		int iz=blockIdx.y*blockDim.y+threadIdx.y;

		int in_idx;
		if(iangle<angle_num&&iz<nz)
		{
			//in_idx=cdp_location*nz*angle_num+iangle*nz+iz;
			//adcigs_cdp_d[iangle*nz+iz]=adcigs_d[in_idx];

			in_idx=iangle*nx*nz+cdp_location*nz+iz;
			adcigs_cdp_d[iangle*nz+iz]=adcigs_d[in_idx];
		}
}

__global__ void caculate_angle_base_on_direction_least_square(float *angle_pp_d,float *poyn_px_d,float *poyn_pz_d,int dimx,int dimz,int scale)
//angle_pp_d,normal_x_d,normal_z_d,poyn_px_d,poyn_pz_d,nx_append_radius1,nz_append_radius1
{
		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;
		double sumx,sumz;
		int m,n;

		if(ix<dimx&&iz<dimz)
		{
				dimx=dimx+2*radius1;dimz=dimz+2*radius1;
				ix=ix+radius1;iz=iz+radius1;
				in_idx=ix*dimz+iz;
				angle_pp_d[in_idx]=0;
///note that  x/z
//least_square size of scale	  
				for(m=-4;m<=4;m++)
					for(n=-4;n<=4;n++)
						{
							sumx=sumx+poyn_px_d[in_idx+m+n*dimz]*poyn_pz_d[in_idx+m+n*dimz];
							sumz=sumz+poyn_pz_d[in_idx+m+n*dimz]*poyn_pz_d[in_idx+m+n*dimz];
						}
				if(sumz!=0)	angle_pp_d[in_idx]=float(atan(double(sumx*1.0/sumz)))*180/pai;			
		}
}

__global__ void caculate_angle_base_on_direction_least_square_share(float *angle_pp_d,float *poyn_px_d,float *poyn_pz_d,int dimx,int dimz,int scale)
//angle_pp_d,normal_x_d,normal_z_d,poyn_px_d,poyn_pz_d,nx_append_radius1,nz_append_radius1
{
		__shared__ float s_data1[BDIMY1+2*radius1][BDIMX1+2*radius1];
		__shared__ float s_data2[BDIMY1+2*radius1][BDIMX1+2*radius1];

		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;
		double sumx,sumz;
		int m,n;
		
		int tx = threadIdx.x+radius1;
		int tz = threadIdx.y+radius1;
		s_data1[tz][tx]=0.0;
		s_data1[threadIdx.y][threadIdx.x]=0.0;
		s_data1[BDIMY1+2*radius1-1-threadIdx.y][BDIMX1+2*radius1-1-threadIdx.x]=0.0;
		s_data1[threadIdx.y][BDIMX1+2*radius1-1-threadIdx.x]=0.0;
		s_data1[BDIMY1+2*radius1-1-threadIdx.y][threadIdx.x]=0.0;

		s_data2[tz][tx]=0.0;
		s_data2[threadIdx.y][threadIdx.x]=0.0;
		s_data2[BDIMY1+2*radius1-1-threadIdx.y][BDIMX1+2*radius1-1-threadIdx.x]=0.0;
		s_data2[threadIdx.y][BDIMX1+2*radius1-1-threadIdx.x]=0.0;
		s_data2[BDIMY1+2*radius1-1-threadIdx.y][threadIdx.x]=0.0;

		if(ix<dimx&&iz<dimz)
		{
				dimx=dimx+2*radius1;dimz=dimz+2*radius1;
				ix=ix+radius1;iz=iz+radius1;
				in_idx=ix*dimz+iz;
				sumx=0.0;
				sumz=0.0;
				angle_pp_d[in_idx]=0;
				
				__syncthreads();

				s_data1[tz][tx]=poyn_px_d[in_idx];
				s_data2[tz][tx]=poyn_pz_d[in_idx];

				if(threadIdx.y<radius1)
				{
					s_data1[threadIdx.y][threadIdx.x]=poyn_px_d[in_idx-radius1-radius1*dimz];//up
					s_data1[threadIdx.y][threadIdx.x+2*radius1]=poyn_px_d[in_idx-radius1+radius1*dimz];//up
					s_data1[threadIdx.y+BDIMY1+radius1][threadIdx.x]=poyn_px_d[in_idx+BDIMY1-radius1*dimz];//down
					s_data1[threadIdx.y+BDIMY1+radius1][threadIdx.x+2*radius1]=poyn_px_d[in_idx+BDIMY1+radius1*dimz];//down

					s_data2[threadIdx.y][threadIdx.x]=poyn_pz_d[in_idx-radius1-radius1*dimz];//up
					s_data2[threadIdx.y][threadIdx.x+2*radius1]=poyn_pz_d[in_idx-radius1+radius1*dimz];//up
					s_data2[threadIdx.y+BDIMY1+radius1][threadIdx.x]=poyn_pz_d[in_idx+BDIMY1-radius1*dimz];//down
					s_data2[threadIdx.y+BDIMY1+radius1][threadIdx.x+2*radius1]=poyn_pz_d[in_idx+BDIMY1+radius1*dimz];//down

				}
				if(threadIdx.x<radius1)
				{
					s_data1[tz][threadIdx.x]=poyn_px_d[in_idx-radius1*dimz];//g_input[in_idx-radius1];//left
					s_data1[tz][threadIdx.x+BDIMX1+radius1]=poyn_px_d[in_idx+BDIMX1*dimz];//g_input[in_idx+BDIMX1];//right
				
					s_data2[tz][threadIdx.x]=poyn_pz_d[in_idx-radius1*dimz];//g_input[in_idx-radius1];//left
					s_data2[tz][threadIdx.x+BDIMX1+radius1]=poyn_pz_d[in_idx+BDIMX1*dimz];//g_input[in_idx+BDIMX1];//right
				}
				__syncthreads();
///note that  x/z
//least_square size of scale	  
				for(m=-4;m<=4;m++)
					for(n=-4;n<=4;n++)
						{
							sumx=sumx+s_data1[tz+m][tx+n]*s_data2[tz+m][tx+n];
							sumz=sumz+s_data2[tz+m][tx+n]*s_data2[tz+m][tx+n];
						}
				if(sumz!=0)	angle_pp_d[in_idx]=float(atan(double(sumx*1.0/sumz)))*180/pai;			
		}
}

__global__ void caculate_normal_base_on_direction_least_square(float *poyn_px_d,float *poyn_pz_d,int dimx,int dimz,int scale)
//angle_pp_d,normal_x_d,normal_z_d,poyn_px_d,poyn_pz_d,nx_append_radius1,nz_append_radius1
{
		__shared__ float s_data1[BDIMY1+2*radius1][BDIMX1+2*radius1];
		__shared__ float s_data2[BDIMY1+2*radius1][BDIMX1+2*radius1];

		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;
		double sumx,sumz;
		int m,n;
		
		int tx = threadIdx.x+radius1;
		int tz = threadIdx.y+radius1;
		s_data1[tz][tx]=0.0;
		s_data1[threadIdx.y][threadIdx.x]=0.0;
		s_data1[BDIMY1+2*radius1-1-threadIdx.y][BDIMX1+2*radius1-1-threadIdx.x]=0.0;
		s_data1[threadIdx.y][BDIMX1+2*radius1-1-threadIdx.x]=0.0;
		s_data1[BDIMY1+2*radius1-1-threadIdx.y][threadIdx.x]=0.0;

		s_data2[tz][tx]=0.0;
		s_data2[threadIdx.y][threadIdx.x]=0.0;
		s_data2[BDIMY1+2*radius1-1-threadIdx.y][BDIMX1+2*radius1-1-threadIdx.x]=0.0;
		s_data2[threadIdx.y][BDIMX1+2*radius1-1-threadIdx.x]=0.0;
		s_data2[BDIMY1+2*radius1-1-threadIdx.y][threadIdx.x]=0.0;

		if(ix<dimx&&iz<dimz)
		{
				dimx=dimx+2*radius1;dimz=dimz+2*radius1;
				ix=ix+radius1;iz=iz+radius1;
				in_idx=ix*dimz+iz;
				
				__syncthreads();

				s_data1[tz][tx]=poyn_px_d[in_idx];
				s_data2[tz][tx]=poyn_pz_d[in_idx];

				if(threadIdx.y<radius1)
				{
					s_data1[threadIdx.y][threadIdx.x]=poyn_px_d[in_idx-radius1-radius1*dimz];//up
					s_data1[threadIdx.y][threadIdx.x+2*radius1]=poyn_px_d[in_idx-radius1+radius1*dimz];//up
					s_data1[threadIdx.y+BDIMY1+radius1][threadIdx.x]=poyn_px_d[in_idx+BDIMY1-radius1*dimz];//down
					s_data1[threadIdx.y+BDIMY1+radius1][threadIdx.x+2*radius1]=poyn_px_d[in_idx+BDIMY1+radius1*dimz];//down

					s_data2[threadIdx.y][threadIdx.x]=poyn_pz_d[in_idx-radius1-radius1*dimz];//up
					s_data2[threadIdx.y][threadIdx.x+2*radius1]=poyn_pz_d[in_idx-radius1+radius1*dimz];//up
					s_data2[threadIdx.y+BDIMY1+radius1][threadIdx.x]=poyn_pz_d[in_idx+BDIMY1-radius1*dimz];//down
					s_data2[threadIdx.y+BDIMY1+radius1][threadIdx.x+2*radius1]=poyn_pz_d[in_idx+BDIMY1+radius1*dimz];//down

				}
				if(threadIdx.x<radius1)
				{
					s_data1[tz][threadIdx.x]=poyn_px_d[in_idx-radius1*dimz];//g_input[in_idx-radius1];//left
					s_data1[tz][threadIdx.x+BDIMX1+radius1]=poyn_px_d[in_idx+BDIMX1*dimz];//g_input[in_idx+BDIMX1];//right
				
					s_data2[tz][threadIdx.x]=poyn_pz_d[in_idx-radius1*dimz];//g_input[in_idx-radius1];//left
					s_data2[tz][threadIdx.x+BDIMX1+radius1]=poyn_pz_d[in_idx+BDIMX1*dimz];//g_input[in_idx+BDIMX1];//right
				}
				__syncthreads();
				
				for(m=-4;m<=4;m++)
					for(n=-4;n<=4;n++)
						{
							sumx=sumx+s_data1[tz+m][tx+n];
						
							sumz=sumz+s_data2[tz+m][tx+n];
						}
				
				poyn_px_d[in_idx]=sumx;
				poyn_pz_d[in_idx]=sumz;		
		}
}



__global__ void set_adcigs_pp(float *p_d,float *rp_d,float *r_adcigs_pp_d,float *angle_pp_d,float *angle_open_d,int source_x_cord,int nx,int nz,int nx_append,int nz_append,int boundary_up,int boundary_left,int angle_num,int dangle)
{
		int ix=blockIdx.x*blockDim.x+threadIdx.x;
		int iz=blockIdx.y*blockDim.y+threadIdx.y;

		int in_idx;
		int set;
		int r_angle;
		float image;

		if(ix<nx&&iz<nz&&iz>10)
		//if(ix<nx&&iz<nz&&iz>10&&iz>0.8*(ix-source_x_cord)&&iz>0.8*(source_x_cord-ix))
		{
			in_idx=(boundary_left+ix)*nz_append+iz+boundary_up;
				
			//if(angle_open_d[in_idx]<=60)	image=p_d[in_idx]*rp_d[in_idx];	
			if(angle_open_d[in_idx]<=90)	image=p_d[in_idx]*rp_d[in_idx]*float(float(cos(double(pai*angle_open_d[in_idx]/180)))*float(cos(double(pai*angle_open_d[in_idx]/180)))*float(cos(double(pai*angle_open_d[in_idx]/180))));
			
			r_angle=int((fabs(angle_pp_d[in_idx])+0.5)/dangle);
///every time  only one angle need to caculate		
			set=r_angle*nx*nz+ix*nz+iz;
			if(r_angle<angle_num&&angle_pp_d[in_idx]>=0) r_adcigs_pp_d[angle_num*nx*nz+set]+=image*exp(-(fabs(angle_pp_d[in_idx])-r_angle*dangle)*(fabs(angle_pp_d[in_idx])-r_angle*dangle)/8);
			if(r_angle<angle_num&&angle_pp_d[in_idx]<0) r_adcigs_pp_d[angle_num*nx*nz-set]+=image*exp(-(fabs(angle_pp_d[in_idx])-r_angle*dangle)*(fabs(angle_pp_d[in_idx])-r_angle*dangle)/8);
		}
		
}

__global__ void set_adcigs_ps(float *p_d,float *rs_d,float *r_adcigs_ps_d,float *angle_pp_d,float *angle_open_d,int source_x_cord,int nx,int nz,int nx_append,int nz_append,int boundary_up,int boundary_left,int angle_num,int dangle)
//angle_pp_d,p_d,rs_d,adcigs_ps_d,rimageupps_d,rimagedownps_d,angle_pp_d,nx,nz,nx_append,nz_append,boundary_up,boundary_left,angle_num,dangle
{
		int ix=blockIdx.x*blockDim.x+threadIdx.x;
		int iz=blockIdx.y*blockDim.y+threadIdx.y;

		int in_idx;
		int set;
		int r_angle;
		float image;

		if(ix<nx&&iz<nz&&iz>10&&iz>0.8*(ix-source_x_cord)&&iz>0.8*(source_x_cord-ix))
		{
			in_idx=(boundary_left+ix)*nz_append+iz+boundary_up;
			
			if(angle_pp_d[in_idx]>=0)
			{
				image=p_d[in_idx]*rs_d[in_idx];
			}
			if(angle_pp_d[in_idx]<0)
			{
				image=-1*p_d[in_idx]*rs_d[in_idx];
			}	
			
			r_angle=int((fabs(angle_pp_d[in_idx])+0.5)/dangle);
	
			set=r_angle*nx*nz+ix*nz+iz;
			
			if(r_angle<angle_num&&angle_pp_d[in_idx]>=0) r_adcigs_ps_d[angle_num*nx*nz+set]+=image*exp(-(fabs(angle_pp_d[in_idx])-r_angle*dangle)*(fabs(angle_pp_d[in_idx])-r_angle*dangle)/8);
			if(r_angle<angle_num&&angle_pp_d[in_idx]<0) r_adcigs_ps_d[angle_num*nx*nz-set]+=image*exp(-(fabs(angle_pp_d[in_idx])-r_angle*dangle)*(fabs(angle_pp_d[in_idx])-r_angle*dangle)/8);
		
		}
}

__global__ void set_adcigs_pp_new(float *vxp_d,float *vzp_d,float *rvxp_d,float *rvzp_d,float *r_adcigs_pp_d,float *angle_pp_d,float *angle_open_d,int source_x_cord,int nx,int nz,int nx_append,int nz_append,int boundary_up,int boundary_left,int angle_num,int dangle)
//(vxp1_d,vzp1_d,rvxp1_d,rvzp1_d,p_adcigs_pp_d,n_adcigs_pp_d,rimageuppp_d,rimagedownpp_d,angle_pp_d,angle_open_d,nx,nz,nx_append,nz_append,boundary_up,boundary_left,angle_num,dangle);
{
		int ix=blockIdx.x*blockDim.x+threadIdx.x;
		int iz=blockIdx.y*blockDim.y+threadIdx.y;

		int in_idx;
		int set;
		int r_angle;
		
		//float sign;
		float image;

		if(ix<nx&&iz<nz&&iz>10)
		//if(ix<nx&&iz<nz&&iz>10&&iz>0.8*(ix-source_x_cord)&&iz>0.8*(source_x_cord-ix))
		{
			in_idx=(boundary_left+ix)*nz_append+iz+boundary_up;
			
			//if(vzp_d[in_idx]*rvzp_d[in_idx]<=0)	sign=-1;
			//if(vzp_d[in_idx]*rvzp_d[in_idx]>0)	sign=+1;
				
			//if(angle_open_d[in_idx]<=90)	image=(fabs(vxp_d[in_idx]*rvxp_d[in_idx])*sign+vzp_d[in_idx]*rvzp_d[in_idx])*float(float(cos(double(pai*angle_open_d[in_idx]/180)))*float(cos(double(pai*angle_open_d[in_idx]/180)))*float(cos(double(pai*angle_open_d[in_idx]/180))));
			
			if(angle_open_d[in_idx]<=90)	image=(vxp_d[in_idx]*rvxp_d[in_idx]+vzp_d[in_idx]*rvzp_d[in_idx])*float(float(cos(double(pai*angle_open_d[in_idx]/180)))*float(cos(double(pai*angle_open_d[in_idx]/180)))*float(cos(double(pai*angle_open_d[in_idx]/180))));
			
			//if(angle_open_d[in_idx]<=90)	image=(vzp_d[in_idx]*rvzp_d[in_idx])*float(float(cos(double(pai*angle_open_d[in_idx]/180)))*float(cos(double(pai*angle_open_d[in_idx]/180)))*float(cos(double(pai*angle_open_d[in_idx]/180))));
			
			r_angle=int((fabs(angle_pp_d[in_idx])+0.5)/dangle);
///every time  only one angle need to caculate		
			set=r_angle*nx*nz+ix*nz+iz;
			if(r_angle<angle_num&&angle_pp_d[in_idx]>=0) r_adcigs_pp_d[angle_num*nx*nz+set]+=image*exp(-(fabs(angle_pp_d[in_idx])-r_angle*dangle)*(fabs(angle_pp_d[in_idx])-r_angle*dangle)/8);
			if(r_angle<angle_num&&angle_pp_d[in_idx]<0) r_adcigs_pp_d[angle_num*nx*nz-set]+=image*exp(-(fabs(angle_pp_d[in_idx])-r_angle*dangle)*(fabs(angle_pp_d[in_idx])-r_angle*dangle)/8);
		}
}

__global__ void set_adcigs_ps_new(float *vxp_d,float *vzp_d,float *rvxs_d,float *rvzs_d,float *r_adcigs_ps_d,float *angle_pp_d,float *angle_open_d,int source_x_cord,int nx,int nz,int nx_append,int nz_append,int boundary_up,int boundary_left,int angle_num,int dangle)
//(vxp1_d,vzp1_d,rvxs1_d,rvzs1_d,p_adcigs_pp_d,n_adcigs_pp_d,rimageuppp_d,rimagedownpp_d,angle_pp_d,angle_open_d,nx,nz,nx_append,nz_append,boundary_up,boundary_left,angle_num,dangle);
{
		int ix=blockIdx.x*blockDim.x+threadIdx.x;
		int iz=blockIdx.y*blockDim.y+threadIdx.y;

		int in_idx;
		int set;
		int r_angle;
		//float sign;
		float image;

		if(ix<nx&&iz<nz&&iz>10&&iz>0.8*(ix-source_x_cord)&&iz>0.8*(source_x_cord-ix))
		{
			in_idx=(boundary_left+ix)*nz_append+iz+boundary_up;
			
			//if(vxp_d[in_idx]*rvxs_d[in_idx]<=0)	sign=-1;
			//if(vxp_d[in_idx]*rvxs_d[in_idx]>0)	sign=+1;
			
			//image=vxp_d[in_idx]*rvxs_d[in_idx]+sign*fabs(vzp_d[in_idx]*rvzs_d[in_idx]);
			
			image=vxp_d[in_idx]*rvxs_d[in_idx]+vzp_d[in_idx]*rvzs_d[in_idx];
			
			//image=vxp_d[in_idx]*rvxs_d[in_idx];
				
			r_angle=int((fabs(angle_pp_d[in_idx])+0.5)/dangle);
	
			set=r_angle*nx*nz+ix*nz+iz;
			
			if(r_angle<angle_num&&angle_pp_d[in_idx]>=0) r_adcigs_ps_d[angle_num*nx*nz+set]+=image*exp(-(fabs(angle_pp_d[in_idx])-r_angle*dangle)*(fabs(angle_pp_d[in_idx])-r_angle*dangle)/8);
			if(r_angle<angle_num&&angle_pp_d[in_idx]<0) r_adcigs_ps_d[angle_num*nx*nz-set]+=image*exp(-(fabs(angle_pp_d[in_idx])-r_angle*dangle)*(fabs(angle_pp_d[in_idx])-r_angle*dangle)/8);
		
		}
}
