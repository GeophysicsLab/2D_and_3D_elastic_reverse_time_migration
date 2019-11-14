__constant__ const int BDIMX3=32;
__constant__ const int BDIMY3=16;
__constant__ const int radius3=6;
__constant__ const float pai1=3.1415926;

__global__ void caculate_ex_amp_time(float *p_d,float *ex_amp_d,float *ex_time_d,int it,int dimx,int dimz)
{
		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		if(ix<dimx&&iz<dimz)
		{
				dimx=dimx+2*radius;dimz=dimz+2*radius;
				ix=ix+radius;iz=iz+radius;
				in_idx=ix*dimz+iz;
			/*if(ex_amp_d[in_idx]<p_d[in_idx]) 
				{			
					ex_time_d[in_idx]=it;
					ex_amp_d[in_idx]=p_d[in_idx];///////emphasize  fabs(p_d[in_idx]);
				}*/
			if(fabs(ex_amp_d[in_idx])<fabs(p_d[in_idx]))	
				{
					ex_time_d[in_idx]=it;
					ex_amp_d[in_idx]=fabs(p_d[in_idx]);
				}
		}
}

__global__ void caculate_ex_amp_time_new(float *vxp_d,float *vzp_d,float *ex_amp_d,float *ex_time_d,int it,int dimx,int dimz)
{
		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		if(ix<dimx&&iz<dimz)
		{
				dimx=dimx+2*radius;dimz=dimz+2*radius;
				ix=ix+radius;iz=iz+radius;
				in_idx=ix*dimz+iz;

			if(fabs(ex_amp_d[in_idx])<sqrt(vxp_d[in_idx]*vxp_d[in_idx]+vzp_d[in_idx]*vzp_d[in_idx]))	
				{
					ex_time_d[in_idx]=it;
					ex_amp_d[in_idx]=sqrt(vxp_d[in_idx]*vxp_d[in_idx]+vzp_d[in_idx]*vzp_d[in_idx]);
				}
		}
}

__global__ void caculate_ex_tp_time_new(float *tp1_d,float *ex_amp_d,float *ex_tp_time_d,int it,int dimx,int dimz)
//caculate_ex_tp_time_new<<<dimGrid,dimBlock>>>(tp1_d,ex_amp_tp_d,ex_tp_time_d,it-wavelet_half,nx_append_radius,nz_append_radius);
{
		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		if(ix<dimx&&iz<dimz)
		{
				dimx=dimx+2*radius;dimz=dimz+2*radius;
				ix=ix+radius;iz=iz+radius;
				in_idx=ix*dimz+iz;

			if(fabs(ex_amp_d[in_idx])<fabs(tp1_d[in_idx]))	
				{
					ex_tp_time_d[in_idx]=it;
					ex_amp_d[in_idx]=tp1_d[in_idx];
				}
		}
}

__global__ void caculate_ex_x_z(float *ex_amp_x_d,float *ex_amp_z_d,float *vxp_d,float *vzp_d,float *ex_time_d,int it,int dimx,int dimz)
{
		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		if(ix<dimx&&iz<dimz)
		{
				dimx=dimx+2*radius;dimz=dimz+2*radius;
				ix=ix+radius;iz=iz+radius;
				in_idx=ix*dimz+iz;

				if(it==ex_time_d[in_idx])
				{
					ex_amp_x_d[in_idx]=vxp_d[in_idx];
					ex_amp_z_d[in_idx]=vzp_d[in_idx];
				}
		}
}

__global__ void caculate_ex_x_z_new(float *ex_amp_x_d,float *ex_amp_z_d,float *vxp_d,float *vzp_d,float *ex_time_d,int it,int dimx,int dimz)
{
		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		if(ix<dimx&&iz<dimz)
		{
				dimx=dimx+2*radius;dimz=dimz+2*radius;
				ix=ix+radius;iz=iz+radius;
				in_idx=ix*dimz+iz;

				if(it==ex_time_d[in_idx])
				{
					ex_amp_x_d[in_idx]=vxp_d[in_idx];
					ex_amp_z_d[in_idx]=1.0*(vzp_d[in_idx+1]+vzp_d[in_idx-1]+vzp_d[in_idx+dimz]+vzp_d[in_idx-dimz])/4.0;
				}
		}
}

__global__ void caculate_ex_angle(float *ex_angle1_d,float *angle_pp_d,float *ex_time_d,int it,int dimx,int dimz)
{
		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		if(ix<dimx&&iz<dimz)
		{
				dimx=dimx+2*radius;dimz=dimz+2*radius;
				ix=ix+radius;iz=iz+radius;
				in_idx=ix*dimz+iz;

				if(it==ex_time_d[in_idx])
					ex_angle1_d[in_idx]=angle_pp_d[in_idx];
					/*{	
						if(angle_pp_d[in_idx]>0)	ex_angle1_d[in_idx]=int(angle_pp_d[in_idx]+0.5);
						if(angle_pp_d[in_idx]<=0)	ex_angle1_d[in_idx]=int(angle_pp_d[in_idx]-0.5);
					}*/
		}
}

__global__ void caculate_ex_angle_pp_only_RTM(float *angle_pp_d,float *poyn_px_d,float *poyn_pz_d,float *ex_time_d,int it,int dimx,int dimz)
{
		__shared__ float s_data1[BDIMY3+2*radius3][BDIMX3+2*radius3];
		__shared__ float s_data2[BDIMY3+2*radius3][BDIMX3+2*radius3];

		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;
		double sumx=0.0,sumz=0.0;
		int m,n;
		
		int tx = threadIdx.x+radius3;
		int tz = threadIdx.y+radius3;
		s_data1[tz][tx]=0.0;
		s_data1[threadIdx.y][threadIdx.x]=0.0;
		s_data1[BDIMY3+2*radius3-1-threadIdx.y][BDIMX3+2*radius3-1-threadIdx.x]=0.0;
		s_data1[threadIdx.y][BDIMX3+2*radius3-1-threadIdx.x]=0.0;
		s_data1[BDIMY3+2*radius3-1-threadIdx.y][threadIdx.x]=0.0;

		s_data2[tz][tx]=0.0;
		s_data2[threadIdx.y][threadIdx.x]=0.0;
		s_data2[BDIMY3+2*radius3-1-threadIdx.y][BDIMX3+2*radius3-1-threadIdx.x]=0.0;
		s_data2[threadIdx.y][BDIMX3+2*radius3-1-threadIdx.x]=0.0;
		s_data2[BDIMY3+2*radius3-1-threadIdx.y][threadIdx.x]=0.0;

		if(ix<dimx&&iz<dimz)
		{
				dimx=dimx+2*radius;dimz=dimz+2*radius;
				ix=ix+radius;iz=iz+radius;
				in_idx=ix*dimz+iz;

				__syncthreads();				

				s_data1[tz][tx]=poyn_px_d[in_idx];
				s_data2[tz][tx]=poyn_pz_d[in_idx];

				if(threadIdx.y<radius3)
				{
					s_data1[threadIdx.y][threadIdx.x]=poyn_px_d[in_idx-radius3-radius3*dimz];//up
					s_data1[threadIdx.y][threadIdx.x+2*radius3]=poyn_px_d[in_idx-radius3+radius3*dimz];//up
					s_data1[threadIdx.y+BDIMY3+radius3][threadIdx.x]=poyn_px_d[in_idx+BDIMY3-radius3*dimz];//down
					s_data1[threadIdx.y+BDIMY3+radius3][threadIdx.x+2*radius3]=poyn_px_d[in_idx+BDIMY3+radius3*dimz];//down


					s_data2[threadIdx.y][threadIdx.x]=poyn_pz_d[in_idx-radius3-radius3*dimz];//up
					s_data2[threadIdx.y][threadIdx.x+2*radius3]=poyn_pz_d[in_idx-radius3+radius3*dimz];//up
					s_data2[threadIdx.y+BDIMY3+radius3][threadIdx.x]=poyn_pz_d[in_idx+BDIMY3-radius3*dimz];//down
					s_data2[threadIdx.y+BDIMY3+radius3][threadIdx.x+2*radius3]=poyn_pz_d[in_idx+BDIMY3+radius3*dimz];//down

				}
				if(threadIdx.x<radius3)
				{
					s_data1[tz][threadIdx.x]=poyn_px_d[in_idx-radius3*dimz];//g_input[in_idx-radius3];//left
					s_data1[tz][threadIdx.x+BDIMX3+radius3]=poyn_px_d[in_idx+BDIMX3*dimz];//g_input[in_idx+BDIMX3];//right
				
					s_data2[tz][threadIdx.x]=poyn_pz_d[in_idx-radius3*dimz];//g_input[in_idx-radius3];//left
					s_data2[tz][threadIdx.x+BDIMX3+radius3]=poyn_pz_d[in_idx+BDIMX3*dimz];//g_input[in_idx+BDIMX3];//right
				}

				__syncthreads();
///note that  x/z
//least_square size of scale	 
				if(it==ex_time_d[in_idx])
				{ 
				
					for(m=-2;m<=+2;m++)
						for(n=-2;n<=+2;n++)
							{
								sumx=sumx+1.0*s_data1[tz+m][tx+n]*s_data2[tz+m][tx+n];

								sumz=sumz+1.0*s_data2[tz+m][tx+n]*s_data2[tz+m][tx+n];
							}
					if(sumz!=0)	angle_pp_d[in_idx]=float(atan(double(sumx*1.0/sumz)))*180/pai1;
				}
		}
}

__global__ void caculate_ex_angle_rp_only_RTM(float *angle_pp_d,float *poyn_px_d,float *poyn_pz_d,float *ex_time_d,int it,int dimx,int dimz)
//caculate_ex_angle_rp_only_RTM<<<dimGrid,dimBlock>>>(ex_angle_rpp_d,poyn_rpx_d,poyn_rpz_d,ex_time_d,it,nx_append_radius,nz_append_radius);
{
		__shared__ float s_data1[BDIMY3+2*radius3][BDIMX3+2*radius3];
		__shared__ float s_data2[BDIMY3+2*radius3][BDIMX3+2*radius3];

		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;
		double sumx=0.0,sumz=0.0;
		int m=0,n=0;
		
		int tx = threadIdx.x+radius3;
		int tz = threadIdx.y+radius3;
		s_data1[tz][tx]=0.0;
		s_data1[threadIdx.y][threadIdx.x]=0.0;
		s_data1[BDIMY3+2*radius3-1-threadIdx.y][BDIMX3+2*radius3-1-threadIdx.x]=0.0;
		s_data1[threadIdx.y][BDIMX3+2*radius3-1-threadIdx.x]=0.0;
		s_data1[BDIMY3+2*radius3-1-threadIdx.y][threadIdx.x]=0.0;

		s_data2[tz][tx]=0.0;
		s_data2[threadIdx.y][threadIdx.x]=0.0;
		s_data2[BDIMY3+2*radius3-1-threadIdx.y][BDIMX3+2*radius3-1-threadIdx.x]=0.0;
		s_data2[threadIdx.y][BDIMX3+2*radius3-1-threadIdx.x]=0.0;
		s_data2[BDIMY3+2*radius3-1-threadIdx.y][threadIdx.x]=0.0;

		if(ix<dimx&&iz<dimz)
		{
				dimx=dimx+2*radius;dimz=dimz+2*radius;
				ix=ix+radius;iz=iz+radius;
				in_idx=ix*dimz+iz;

				__syncthreads();				

				s_data1[tz][tx]=poyn_px_d[in_idx];
				s_data2[tz][tx]=poyn_pz_d[in_idx];

				if(threadIdx.y<radius3)
				{
					s_data1[threadIdx.y][threadIdx.x]=poyn_px_d[in_idx-radius3-radius3*dimz];//up
					s_data1[threadIdx.y][threadIdx.x+2*radius3]=poyn_px_d[in_idx-radius3+radius3*dimz];//up
					s_data1[threadIdx.y+BDIMY3+radius3][threadIdx.x]=poyn_px_d[in_idx+BDIMY3-radius3*dimz];//down
					s_data1[threadIdx.y+BDIMY3+radius3][threadIdx.x+2*radius3]=poyn_px_d[in_idx+BDIMY3+radius3*dimz];//down

					s_data2[threadIdx.y][threadIdx.x]=poyn_pz_d[in_idx-radius3-radius3*dimz];//up
					s_data2[threadIdx.y][threadIdx.x+2*radius3]=poyn_pz_d[in_idx-radius3+radius3*dimz];//up
					s_data2[threadIdx.y+BDIMY3+radius3][threadIdx.x]=poyn_pz_d[in_idx+BDIMY3-radius3*dimz];//down
					s_data2[threadIdx.y+BDIMY3+radius3][threadIdx.x+2*radius3]=poyn_pz_d[in_idx+BDIMY3+radius3*dimz];//down

				}
				if(threadIdx.x<radius3)
				{
					s_data1[tz][threadIdx.x]=poyn_px_d[in_idx-radius3*dimz];//g_input[in_idx-radius3];//left
					s_data1[tz][threadIdx.x+BDIMX3+radius3]=poyn_px_d[in_idx+BDIMX3*dimz];//g_input[in_idx+BDIMX3];//right
				
					s_data2[tz][threadIdx.x]=poyn_pz_d[in_idx-radius3*dimz];//g_input[in_idx-radius3];//left
					s_data2[tz][threadIdx.x+BDIMX3+radius3]=poyn_pz_d[in_idx+BDIMX3*dimz];//g_input[in_idx+BDIMX3];//right
				}

				__syncthreads();
///note that  x/z
//least_square size of scale	 
				if(it==ex_time_d[in_idx])
				{ 
				
					for(m=-4;m<=+4;m++)
						for(n=-4;n<=+4;n++)
							{
								sumx=sumx+1.0*s_data1[tz+m][tx+n]*s_data2[tz+m][tx+n];

								sumz=sumz+1.0*s_data2[tz+m][tx+n]*s_data2[tz+m][tx+n];
							}
					if(sumz!=0)	angle_pp_d[in_idx]=float(atan(double(sumx*1.0/sumz)))*180/pai1;
				}
		}
}

__global__ void caculate_ex_angle_new(float *ex_angle_d,float *angle_pp_d,float *normal_angle_d,float *poyn_px_d,float *poyn_pz_d,float *ex_time_d,int it,int dimx,int dimz)
{
		__shared__ float s_data1[BDIMY3+2*radius3][BDIMX3+2*radius3];
		__shared__ float s_data2[BDIMY3+2*radius3][BDIMX3+2*radius3];

		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;
		double sumx=0.0,sumz=0.0;
		int m,n;
		
		int tx = threadIdx.x+radius3;
		int tz = threadIdx.y+radius3;
		s_data1[tz][tx]=0.0;
		s_data1[threadIdx.y][threadIdx.x]=0.0;
		s_data1[BDIMY3+2*radius3-1-threadIdx.y][BDIMX3+2*radius3-1-threadIdx.x]=0.0;
		s_data1[threadIdx.y][BDIMX3+2*radius3-1-threadIdx.x]=0.0;
		s_data1[BDIMY3+2*radius3-1-threadIdx.y][threadIdx.x]=0.0;

		s_data2[tz][tx]=0.0;
		s_data2[threadIdx.y][threadIdx.x]=0.0;
		s_data2[BDIMY3+2*radius3-1-threadIdx.y][BDIMX3+2*radius3-1-threadIdx.x]=0.0;
		s_data2[threadIdx.y][BDIMX3+2*radius3-1-threadIdx.x]=0.0;
		s_data2[BDIMY3+2*radius3-1-threadIdx.y][threadIdx.x]=0.0;

		if(ix<dimx&&iz<dimz)
		{
				dimx=dimx+2*radius;dimz=dimz+2*radius;
				ix=ix+radius;iz=iz+radius;
				in_idx=ix*dimz+iz;

				__syncthreads();				

				s_data1[tz][tx]=poyn_px_d[in_idx];
				s_data2[tz][tx]=poyn_pz_d[in_idx];

				if(threadIdx.y<radius3)
				{
					s_data1[threadIdx.y][threadIdx.x]=poyn_px_d[in_idx-radius3-radius3*dimz];//up
					s_data1[threadIdx.y][threadIdx.x+2*radius3]=poyn_px_d[in_idx-radius3+radius3*dimz];//up

					s_data1[threadIdx.y+BDIMY3+radius3][threadIdx.x]=poyn_px_d[in_idx+BDIMY3-radius3*dimz];//down
					s_data1[threadIdx.y+BDIMY3+radius3][threadIdx.x+2*radius3]=poyn_px_d[in_idx+BDIMY3+radius3*dimz];//down

					s_data2[threadIdx.y][threadIdx.x]=poyn_pz_d[in_idx-radius3-radius3*dimz];//up
					s_data2[threadIdx.y][threadIdx.x+2*radius3]=poyn_pz_d[in_idx-radius3+radius3*dimz];//up
					s_data2[threadIdx.y+BDIMY3+radius3][threadIdx.x]=poyn_pz_d[in_idx+BDIMY3-radius3*dimz];//down
					s_data2[threadIdx.y+BDIMY3+radius3][threadIdx.x+2*radius3]=poyn_pz_d[in_idx+BDIMY3+radius3*dimz];//down

				}
				if(threadIdx.x<radius3)
				{
					s_data1[tz][threadIdx.x]=poyn_px_d[in_idx-radius3*dimz];//g_input[in_idx-radius3];//left
					s_data1[tz][threadIdx.x+BDIMX3+radius3]=poyn_px_d[in_idx+BDIMX3*dimz];//g_input[in_idx+BDIMX3];//right
				
					s_data2[tz][threadIdx.x]=poyn_pz_d[in_idx-radius3*dimz];//g_input[in_idx-radius3];//left
					s_data2[tz][threadIdx.x+BDIMX3+radius3]=poyn_pz_d[in_idx+BDIMX3*dimz];//g_input[in_idx+BDIMX3];//right
				}

				__syncthreads();
///note that  x/z
//least_square size of scale	 
				if(it==ex_time_d[in_idx])
				{ 
				
					for(m=-2;m<=+2;m++)
						for(n=-2;n<=+2;n++)
							{
								sumx=sumx+1.0*s_data1[tz+m][tx+n]*s_data2[tz+m][tx+n];

								sumz=sumz+1.0*s_data2[tz+m][tx+n]*s_data2[tz+m][tx+n];
							}
					if(sumz!=0)	angle_pp_d[in_idx]=float(atan(double(sumx*1.0/sumz)))*180/pai1;	
					
					ex_angle_d[in_idx]=angle_pp_d[in_idx]-normal_angle_d[in_idx];

				}
		}
}

__global__ void imaging_ex(float *rimageup1_d,float *ex_amp_d,float *ex_time_d,float *rp_d,int it,int nx,int nz,int dimx,int dimz,int boundary_up,int boundary_left)
{
		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		if(ix<nx&&iz<nz)
		{
			in_idx=(ix+boundary_left)*dimz+iz+boundary_up;
			if(it==ex_time_d[in_idx])
			rimageup1_d[ix*nz+iz]=rp_d[in_idx]*1.0/fabs(ex_amp_d[in_idx]);
		}
}

__global__ void imaging_ex_for_xxzz(float *rimageup1_d,float *ex_amp_d,float *ex_time_d,float *rvxp1_d,float *rvzp1_d,int it,int source_x_cord,int nx,int nz,int dimx,int dimz,int boundary_up,int boundary_left,float average)
{
		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;
		int sign=1;

		//if(ix<nx&&iz<nz&&iz>50)
		if(ix<nx&&iz<nz&&iz>50&&iz>0.8*(ix-source_x_cord)&&iz>0.8*(source_x_cord-ix))
		{
			in_idx=(ix+boundary_left)*dimz+iz+boundary_up;
			if(ex_amp_d[in_idx]<0)	sign=-1;

			if(it==ex_time_d[in_idx])
			{
				if(fabs(ex_amp_d[in_idx])>average)	rimageup1_d[ix*nz+iz]=(rvxp1_d[in_idx]+rvzp1_d[in_idx])*1.0/ex_amp_d[in_idx];
				
				if(fabs(ex_amp_d[in_idx])<=average)	rimageup1_d[ix*nz+iz]=(rvxp1_d[in_idx]+rvzp1_d[in_idx])*1.0/average*sign;
			}
		}
}	

__global__ void imaging_ex_new(float *rimageup1_d,float *ex_amp_d,float *ex_time_d,float *rp_d,int it,int source_x_cord,int nx,int nz,int dimx,int dimz,int boundary_up,int boundary_left)
{
		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		//if(ix<nx&&iz<nz&&iz>50)
		if(ix<nx&&iz<nz&&iz>50&&iz>1.2*(ix-source_x_cord)&&iz>1.2*(source_x_cord-ix))
		{
			in_idx=(ix+boundary_left)*dimz+iz+boundary_up;
			if(it==ex_time_d[in_idx])
			rimageup1_d[ix*nz+iz]=rp_d[in_idx]*1.0/fabs(ex_amp_d[in_idx]);
		}
}

__global__ void imaging_ex_new_average(float *rimageup1_d,float *ex_amp_d,float *ex_time_d,float *rp_d,int it,int source_x_cord,int nx,int nz,int dimx,int dimz,int boundary_up,int boundary_left,float average)
{
		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;
		int sign=1;

		//if(ix<nx&&iz<nz&&iz>50)
		if(ix<nx&&iz<nz&&iz>50&&iz>1.2*(ix-source_x_cord)&&iz>1.2*(source_x_cord-ix))
		{
			in_idx=(ix+boundary_left)*dimz+iz+boundary_up;
			if(ex_amp_d[in_idx]<0)	sign=-1;

			if(it==ex_time_d[in_idx])
			{
				if(fabs(ex_amp_d[in_idx])>average)	rimageup1_d[ix*nz+iz]=rp_d[in_idx]*1.0/ex_amp_d[in_idx];
				
				if(fabs(ex_amp_d[in_idx])<=average)	rimageup1_d[ix*nz+iz]=rp_d[in_idx]*1.0/average*sign;
			}
		}
}

__global__ void imaging_ex_correlation(float *rimageup1_d,float *ex_amp_d,float *ex_time_d,float *rp_d,int it,int source_x_cord,int nx,int nz,int dimx,int dimz,int boundary_up,int boundary_left)
//image_npp_xx_d,ex_amp1_x_d,ex_time1_d,rvxp1_d,it,source_x_cord[ishot]-receiver_x_cord[ishot],imaging_size[0],nz,nx_append,nz_append,boundary_up,boundary_left
{
		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		//if(ix<nx&&iz<nz&&iz>50)
		if(ix<nx&&iz<nz&&iz>50&&iz>1.2*(ix-source_x_cord)&&iz>1.2*(source_x_cord-ix))
		{
			in_idx=(ix+boundary_left)*dimz+iz+boundary_up;

			if(it==ex_time_d[in_idx])
			{
				rimageup1_d[ix*nz+iz]=rp_d[in_idx]*1.0*ex_amp_d[in_idx];
			}
		}
}

__global__ void imaging_ex_ps(float *rimageup1_d,float *ex_amp_d,float *ex_time_d,float *rs_d,float *ex_angle_d,int it,int nx,int nz,int dimx,int dimz,int boundary_up,int boundary_left)
{
		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		if(ix<nx&&iz<nz)
		{
			in_idx=(ix+boundary_left)*dimz+iz+boundary_up;
			if(it==ex_time_d[in_idx])
				{
					if(ex_angle_d[in_idx]>0)	rimageup1_d[ix*nz+iz]=1.0*rs_d[in_idx]/fabs(ex_amp_d[in_idx]);
					if(ex_angle_d[in_idx]<=0)	rimageup1_d[ix*nz+iz]=-1.0*rs_d[in_idx]/fabs(ex_amp_d[in_idx]);
				}
		}
}

__global__ void imageing_after_correcting(float *rimageup1_d,float *ex_angle_d,int nx,int nz,int dimx,int dimz,int boundary_up,int boundary_left)
{
		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		if(ix<nx&&iz<nz)
		{
			in_idx=(ix+boundary_left)*dimz+iz+boundary_up;
			
			if(ex_angle_d[in_idx]<=0)	rimageup1_d[ix*nz+iz]=-1*rimageup1_d[ix*nz+iz];
		}
}

__global__ void set_adcigs_for_ex(float *r_adcigs_pp_d,float *image_d,float *ex_angle_d,int source_x_cord,int nx,int nz,int dimx,int dimz,int boundary_up,int boundary_left,int angle_num,int dangle)
//(r_adcigs_pp_d,image_pp_du_d,ex_angle_d,source_x_cord[ishot]-receiver_x_cord[ishot],imaging_size[0],nz,nx_append,nz_append,boundary_up,boundary_left,angle_num,dangle)
{
		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;
		int set;
		int r_angle;

		//if(ix<nx&&iz<nz&&iz>50)
		if(ix<nx&&iz<nz&&iz>50&&iz>1.2*(ix-source_x_cord)&&iz>1.2*(source_x_cord-ix))
		{
			in_idx=(ix+boundary_left)*dimz+iz+boundary_up;	
			
			r_angle=int((fabs(ex_angle_d[in_idx])+0.5)/dangle);

			set=r_angle*nx*nz+ix*nz+iz;

			if(r_angle<angle_num&&ex_angle_d[in_idx]>=0) r_adcigs_pp_d[angle_num*nx*nz+set]+=image_d[ix*nz+iz];//*exp(-(fabs(ex_angle_d[in_idx])-r_angle*dangle)*(fabs(ex_angle_d[in_idx])-r_angle*dangle)/8);
			if(r_angle<angle_num&&ex_angle_d[in_idx]<0) r_adcigs_pp_d[(angle_num-r_angle)*nx*nz+ix*nz+iz]+=image_d[ix*nz+iz];//*exp(-(fabs(ex_angle_d[in_idx])-r_angle*dangle)*(fabs(ex_angle_d[in_idx])-r_angle*dangle)/8);
		}
}


__global__ void set_value_a_to_b(float *angle_pp_d,float *angle_pp1_d,int dimx,int dimz)
{
		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		if(ix<dimx&&iz<dimz)
		{
			dimx=dimx+2*radius;dimz=dimz+2*radius;
			ix=ix+radius;iz=iz+radius;
			in_idx=ix*dimz+iz;

			angle_pp1_d[in_idx]=angle_pp_d[in_idx];
		}
}

__global__ void sum_image1_and_image2(float *image1_d,float *image2_d,float *image3_d,int nx,int nz)
{
		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		if(ix<nx&&iz<nz)
		{
			in_idx=ix*nz+iz;

			image1_d[in_idx]=image2_d[in_idx]+image3_d[in_idx];
		}
}
__global__ void imaging_correlation_ex_2D(float *ex_resulttp_d,float *ex_amp_d,float *ex_tp_time_d,float *rtp2_d,int nx,int nz,int dimz,int boundary_up,int boundary_left,float *max,int it)
//imaging_correlation_ex_2D<<<dimGrid,dimBlock>>>(ex_result_tp_d,ex_amp_tp_d,ex_tp_time_d,rtp2_d,nx_size,nz,nz_append,boundary_up,boundary_left,&para_max_d[0],it);
{
		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		if(ix<nx&&iz<nz)
		{
			in_idx=(ix+boundary_left)*dimz+iz+boundary_up;
			if(it==ex_tp_time_d[in_idx])
			{
				ex_resulttp_d[ix*nz+iz]=1.0*rtp2_d[in_idx]/ex_amp_d[in_idx];
				//ex_resulttp_d[ix*nz+iz]=1.0*rtp2_d[in_idx]/(ex_amp_d[in_idx]+0.00000000001*(*max));
			}
		}
}
__global__ void imaging_inner_product_ex_2D(float *ex_vresult_pp_d,float *ex_amp_d,float *ex_amp_x_d,float *ex_amp_z_d,float *ex_time_d,float *rvxp2_d,float *rvzp2_d,int nx,int nz,int dimz,int boundary_up,int boundary_left,float *max,int it)
///imaging_inner_product_ex_2D<<<dimGrid,dimBlock>>>(ex_vresultpp_d,ex_amp_d,ex_amp_x_d,ex_amp_z_d,ex_time_d,rvxp2_d,rvzp2_d,nx_size,nz,nz_append,boundary_up,boundary_left,&para_max_d[1],it);
{
		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int idx;

		if(ix<nx&&iz<nz)
		{
			idx=(ix+boundary_left)*dimz+iz+boundary_up;
			if(it==ex_time_d[idx])
			{
				float molecular=ex_amp_x_d[idx]*rvxp2_d[idx]+ex_amp_z_d[idx]*rvzp2_d[idx];

				float denominator=ex_amp_x_d[idx]*ex_amp_x_d[idx]+ex_amp_z_d[idx]*ex_amp_z_d[idx];

				ex_vresult_pp_d[ix*nz+iz]=molecular*1.0/(denominator);
				//ex_vresult_pp_d[ix*nz+iz]=molecular*1.0/(denominator+0.0000000000001*(*max));
				
			}
		}
}

__global__ void imaging_inner_product_ex_2D_new(float *ex_vresult_pp_d,float *ex_amp_d,float *ex_amp_x_d,float *ex_amp_z_d,float *ex_time_d,float *rvxp2_d,float *rvzp2_d,int nx,int nz,int dimz,int boundary_up,int boundary_left,float *max,int it)
///imaging_inner_product_ex_2D<<<dimGrid,dimBlock>>>(ex_vresultpp_d,ex_amp_d,ex_amp_x_d,ex_amp_z_d,ex_time_d,rvxp2_d,rvzp2_d,nx_size,nz,nz_append,boundary_up,boundary_left,&para_max_d[1],it);
{
		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int idx;

		if(ix<nx&&iz<nz)
		{
			idx=(ix+boundary_left)*dimz+iz+boundary_up;
			if(it==ex_time_d[idx])
			{
				float change_rpz=1.0*(rvzp2_d[idx+1]+rvzp2_d[idx-1]+rvzp2_d[idx+dimz]+rvzp2_d[idx-dimz])/4.0;

				float molecular=ex_amp_x_d[idx]*rvxp2_d[idx]+ex_amp_z_d[idx]*change_rpz;

				float denominator=ex_amp_x_d[idx]*ex_amp_x_d[idx]+ex_amp_z_d[idx]*ex_amp_z_d[idx];

				ex_vresult_pp_d[ix*nz+iz]=molecular*1.0/(denominator);
				//ex_vresult_pp_d[ix*nz+iz]=molecular*1.0/(denominator+0.0000000000001*(*max));
				
			}
		}
}

__global__ void imaging_correlation_for_xxzz(float *vxp1_d,float *vxs1_d,float *rvxp1_d,float *rvxs1_d,float *resultxx_d,int nx,int nz,int dimz,int boundary_up,int boundary_left)
//imaging_correlation_for_xxzz<<<dimGrid,dimBlock>>>(vxp1_d,vxs1_d,rvxp1_d,rvxs1_d,resultxx_d,nx_size,nz,nz_append,boundary_up,boundary_left);
{
		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		if(ix<nx&&iz<nz)
		{
			in_idx=(ix+boundary_left)*dimz+iz+boundary_up;
			
			resultxx_d[ix*nz+iz]=resultxx_d[ix*nz+iz]+1.0*(vxp1_d[in_idx]+vxs1_d[in_idx])*(rvxp1_d[in_idx]+rvxs1_d[in_idx]);			
		}
}

__global__ void imaging_pp_compensate_dependent_angle_2D(float *ex_angle_pp_d,float *ex_angle_rpp_d,float *com_ex_vresultpp_d,float *ex_vresultpp_d,float *ex_time_d,int nx,int nz,int dimz,int boundary_up,int boundary_left,int it)
//imaging_pp_compensate_dependent_angle_2D<<<dimGrid,dimBlock>>>(ex_angle_pp_d,ex_angle_rpp_d,com_ex_vresultpp_d,ex_vresultpp_d,ex_time_d,nx_size,nz,nz_append,boundary_up,boundary_left,it);
{
		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int idx;
		float com=0.0;
		float denominrator=0.0;

		if(ix<nx&&iz<nz)
		{
			idx=(ix+boundary_left)*dimz+iz+boundary_up;
			if(it==ex_time_d[idx])
			{
				com=ex_angle_pp_d[idx]-ex_angle_rpp_d[idx];

				if(fabs(com)>=87.5&&fabs(com)<=92.5)	denominrator=0.1;

				else				denominrator=float(cos(1.0*com*pai1/180.0));

				com_ex_vresultpp_d[ix*nz+iz]=ex_vresultpp_d[ix*nz+iz]/(1.0*denominrator+0.0001);

				//ex_angle_rpp_d[idx]=com;
				ex_angle_rpp_d[idx]=ex_angle_pp_d[idx]-ex_angle_rpp_d[idx];	
			}
		}
}

__global__ void imaging_ps_compensate_dependent_angle_2D(float *ex_angle_pp_d,float *ex_angle_rpp_d,float *com_ex_vresultpp_d,float *ex_vresultpp_d,float *ex_time_d,int nx,int nz,int dimz,int boundary_up,int boundary_left,int it)
{
		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int idx;
		float com=0.0;
		float denominrator=0.0;

		if(ix<nx&&iz<nz)
		{
			idx=(ix+boundary_left)*dimz+iz+boundary_up;
			if(it==ex_time_d[idx])
			{
				com=ex_angle_pp_d[idx]-ex_angle_rpp_d[idx];

				if(fabs(com)<=2.5)	denominrator=0.1;

				else			denominrator=float(sin(fabs(1.0*com*pai1/180.0)));

				com_ex_vresultpp_d[ix*nz+iz]=ex_vresultpp_d[ix*nz+iz]/(1.0*denominrator+0.0001);

				//ex_angle_rpp_d[idx]=com;
				ex_angle_rpp_d[idx]=ex_angle_pp_d[idx]-ex_angle_rpp_d[idx];				
			}
		}
}


__global__ void caculate_ex_open_pp_ps(float *ex_open_pp1_d,float *ex_angle_pp1_d,float *ex_angle_rpp1_d,int nx,int nz,int dimx,int dimz,int boundary_up,int boundary_left,int it,float *ex_time_d)
//caculate_ex_open_pp_ps<<<dimGrid,dimBlock>>>(ex_open_pp_d,ex_angle_pp_d,ex_angle_rpp_d,nx_size,nz,nx_append,nz_append,boundary_up,boundary_left,it,ex_time_d);
{
		
		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		if(ix<dimx&&iz<dimz)
		{
				//dimx=dimx+2*radius;dimz=dimz+2*radius;
				//ix=ix+radius;iz=iz+radius;
				in_idx=ix*dimz+iz;

				if(it==ex_time_d[in_idx])
				{
					ex_open_pp1_d[in_idx]=ex_angle_pp1_d[in_idx]-ex_angle_rpp1_d[in_idx];
				} 
		}
}
__global__ void cuda_ex_com_pp_ps_sign(float *ex_com_pp_sign_d,float *ex_open_pp_d,int nx,int nz,int dimx,int dimz,int mark)
//cuda_ex_com_pp_ps_sign<<<dimGrid,dimBlock>>>(ex_com_pp_sign_d,ex_open_pp1_d,nx_size,nz,nx_append,nz_append,0);
{		
		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		if(ix<dimx&&iz<dimz)
		{
				//dimx=dimx+2*radius;dimz=dimz+2*radius;
				//ix=ix+radius;iz=iz+radius;
				in_idx=ix*dimz+iz;

				if(mark==0)	ex_com_pp_sign_d[in_idx]=1.0*cos(1.0*ex_open_pp_d[in_idx]*pai1/180.0);///pp

				if(mark==1)	ex_com_pp_sign_d[in_idx]=fabs(1.0*sin(1.0*ex_open_pp_d[in_idx]*pai1/180.0));///ps
		}
}

__global__ void imaging_pp_compensate_dependent_angle_2D_new(float *ex_open_pp1_d,float *ex_com_pp_sign_d,float *com_ex_vresultpp_d,float *ex_vresultpp_d,float *ex_time_d,int nx,int nz,int dimz,int boundary_up,int boundary_left)
//imaging_pp_compensate_dependent_angle_2D_new<<<dimGrid,dimBlock>>>(ex_open_pp1_d,ex_com_pp_sign_d,com_ex_vresultpp_d,ex_vresultpp_d,ex_time_d,nx_size,nz,nz_append,boundary_up,boundary_left);
{
		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int idx;
		float denominrator=0.0;

		if(ix<nx&&iz<nz)
		{
			idx=(ix+boundary_left)*dimz+iz+boundary_up;
			
			//if(fabs(ex_open_ps1_d[idx])>=87.5&&fabs(ex_open_ps1_d[idx])<=92.5)	denominrator=0.001;			

			denominrator=ex_com_pp_sign_d[idx];//+0.01;

			//if(denominrator!=0)	com_ex_vresultpp_d[ix*nz+iz]=1.0*ex_vresultpp_d[ix*nz+iz]/denominrator;

			if(denominrator<=0)		com_ex_vresultpp_d[ix*nz+iz]=-1.0*ex_vresultpp_d[ix*nz+iz];

			if(denominrator>0)		com_ex_vresultpp_d[ix*nz+iz]=1.0*ex_vresultpp_d[ix*nz+iz];

			com_ex_vresultpp_d[ix*nz+iz]=-1.0*com_ex_vresultpp_d[ix*nz+iz]/(fabs(denominrator)+0.03);
		}
}

__global__ void imaging_ps_compensate_dependent_angle_2D_new(float *ex_open_ps1_d,float *ex_com_ps_sign_d,float *com_ex_vresultps_d,float *ex_vresultps_d,float *ex_time_d,int nx,int nz,int dimz,int boundary_up,int boundary_left)
//imaging_ps_compensate_dependent_angle_2D_new<<<dimGrid,dimBlock>>>(ex_open_pp1_d,ex_com_ps_sign_d,com_ex_vresultps_d,ex_vresultps_d,ex_time_d,nx_size,nz,nz_append,boundary_up,boundary_left);
{
		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int idx;
		float denominrator=0.0;

		if(ix<nx&&iz<nz)
		{
			idx=(ix+boundary_left)*dimz+iz+boundary_up;
			
			//if(fabs(ex_open_ps1_d[idx])<=2.5)	denominrator=0.001;			

			denominrator=fabs(ex_com_ps_sign_d[idx])+0.1;

			if(denominrator!=0)	com_ex_vresultps_d[ix*nz+iz]=-1.0*ex_vresultps_d[ix*nz+iz]/denominrator;
		}
}
