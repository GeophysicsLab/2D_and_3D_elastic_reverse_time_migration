__constant__ const int BDIMX2=32;
__constant__ const int BDIMY2=16;
__constant__ const int radius2=6;
__constant__ const int Block_Size=512;
#define EPS 1e-30

__global__ void cuda_mul_error_random(float *obs_shot_x_d,float *error_random_d,int receiver_interval,int receiver_num,int lt)
{
	int ix = blockIdx.x*blockDim.x+threadIdx.x;
	int iz = blockIdx.y*blockDim.y+threadIdx.y;
	int in_idx;

	if((ix<receiver_num)&&(iz<lt))
	{
		in_idx=ix*lt+iz;
			
		obs_shot_x_d[in_idx]=obs_shot_x_d[in_idx]*error_random_d[ix];
	}
}

__global__ void cuda_mul_shot_scale(float *obs_shot_x_d,int ishot,int shot_num,int shot_scale,int receiver_num,int lt)
//cuda_mul_shot_scale<<<dimGrid_lt,dimBlock>>>(obs_shot_x_d,ishot,shot_num,shot_scale,receiver_num,lt);
{
	int ix = blockIdx.x*blockDim.x+threadIdx.x;
	int iz = blockIdx.y*blockDim.y+threadIdx.y;
	int in_idx;

	if((ix<receiver_num)&&(iz<lt))
	{
		in_idx=ix*lt+iz;
			
		if(ishot<shot_num/2)	obs_shot_x_d[in_idx]=obs_shot_x_d[in_idx]*1.0/6;

		else			obs_shot_x_d[in_idx]=obs_shot_x_d[in_idx]*6.0;
	}
}

__global__ void cuda_mul_shot_scale_new(float *obs_shot_x_d,int ishot,int shot_num,float *shot_scale_d,int receiver_num,int lt)
//cuda_mul_shot_scale<<<dimGrid_lt,dimBlock>>>(obs_shot_x_d,ishot,shot_num,shot_scale,receiver_num,lt);
{
	int ix = blockIdx.x*blockDim.x+threadIdx.x;
	int iz = blockIdx.y*blockDim.y+threadIdx.y;
	int in_idx;

	if((ix<receiver_num)&&(iz<lt))
	{
		in_idx=ix*lt+iz;
			
			obs_shot_x_d[in_idx]=obs_shot_x_d[in_idx]*shot_scale_d[ishot];
	}
}

__global__ void cuda_laplace(float *input,float *output,float *velocity_d,int nx,int nz,int dimx,int dimz,int boundary_left,int boundary_up,float dx,float dz,int mark,int laplace)
{
		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;
		int id;
		float s_velocity;	
		float up1,down1,left1,right1,self,result;	

		if(ix>0&&ix<nx-1&&iz>0&&iz<nz-1)
		{
			id=ix*nz+iz;

			in_idx=(ix+boundary_left)*dimz+iz+boundary_up;//iz*nz+ix;

			__syncthreads();

			self=-1.0*input[id];
			up1=-1.0*input[id-1];
			down1=-1.0*input[id+1];

			left1=-1.0*input[id-nz];
			right1=-1.0*input[id+nz];
			s_velocity=velocity_d[in_idx]*velocity_d[in_idx];
			
			result=(right1+left1-2*self)*1.0/dx/dx+(up1+down1-2*self)*1.0/dz/dz;			

			__syncthreads();

			if(mark==0)	output[id]=s_velocity*result/4.0;

			if(mark==1)	output[id]=2500*2500/4*result;
		}		
}

__global__ void cuda_lap(float *wf_d,int nx,int nz,int laplace)
{
		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int id;

		if(ix==0||ix==nx-1||iz==0||iz==nz-1)
		{
			
				id=ix*nz+iz;	

				/*if(ix<=1)	wf_d[id]=wf_d[id+1*nz];

				if(ix>=nx-1)	wf_d[id]=wf_d[id-1*nz];

				if(iz<=1)	wf_d[id]=wf_d[id+1];

				if(iz>=nz-1)	wf_d[id]=wf_d[id-1];*/
				////////////mabye  some artifact may be introduced into boundary.2017年03月23日 星期四 21时25分10秒 

				if(ix==0)	wf_d[id]=wf_d[id+1*nz];

				if(ix==nx-1)	wf_d[id]=wf_d[id-1*nz];

				if(iz==0)	wf_d[id]=wf_d[id+1];

				if(iz==nz-1)	wf_d[id]=wf_d[id-1];

				/*if(ix==0)	wf_d[id]=wf_d[id+1*nz]*0.6666;

				if(ix==nx-1)	wf_d[id]=wf_d[id-1*nz]*0.6666;

				if(iz==0)	wf_d[id]=wf_d[id+1]*0.6666;

				if(iz==nz-1)	wf_d[id]=wf_d[id-1]*0.6666;*/
			
		}

}

/*__global__ void cuda_laplace(float *input,float *velocity_d,int nx,int nz,int dimx,int dimz,int boundary_left,int boundary_up,float dx,float dz,int mark)
{
		__shared__ float s_data1[BDIMY2+2][BDIMX2+2];
		
		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;
		int id;

		float s_velocity;
		float result;

		int tx = threadIdx.x+1;
		int tz = threadIdx.y+1;

		s_data1[tz][tx]=0.0;
		s_data1[threadIdx.y][threadIdx.x]=0.0;
		s_data1[BDIMY2+2*1-threadIdx.y][BDIMX2+2*1-threadIdx.x]=0.0;
		s_data1[threadIdx.y][BDIMX2+2*1-threadIdx.x]=0.0;
		s_data1[BDIMY2+2*1-threadIdx.y][threadIdx.x]=0.0;

		if(ix<nx-2&&iz<nz-2)
		{		
			ix=ix+1;iz=iz+1;
			in_idx=ix*nz+iz;//iz*dimx+ix;

			id=(ix+boundary_left)*dimz+iz+boundary_up;

			__syncthreads();

			s_data1[tz][tx]=input[in_idx];
			

			if(threadIdx.y<1)
			{
					s_data1[threadIdx.y][tx]=input[in_idx-1];//g_input[in_idx-radius2*dimx];//up
					s_data1[threadIdx.y+BDIMY2+1][tx]=input[in_idx+BDIMY2];//g_input[in_idx+BDIMY2*dimx];//down
			}
			if(threadIdx.x<1)
			{
					s_data1[tz][threadIdx.x]=input[in_idx-1*dimz];//g_input[in_idx-radius2];//left
					s_data1[tz][threadIdx.x+BDIMX2+1]=input[in_idx+BDIMX2*dimz];//g_input[in_idx+BDIMX2];//right
			}

			s_velocity=1;//velocity_d[id]*velocity_d[id];
				
			__syncthreads();

			result=(s_data1[tz+1][tx]+s_data1[tz-1][tx]-2.0*s_data1[tz][tx])/(dz*dz*1.0)+(s_data1[tz][tx+1]+s_data1[tz][tx-1]-2.0*s_data1[tz][tx])/(dx*dx*1.0);

			if(mark==0)	input[in_idx]=-1.0*result*s_velocity/4.0;

			if(mark==1)	input[in_idx]=-1.0*result*2500*2500/4.0;

		}
}*/

__global__ void cal_sum_a_b_to_c(float *vx_x_d,float *vz_z_d,float *wf_append_d,int dimx,int dimz)
///cal_sum_a_b_to_c<<<dimGrid_lt, dimBlock>>>(cal_shot_x_d,cal_shot_z_d,cal_shot_all_d,receiver_num,lt);
{
		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		if((ix<dimx)&&(iz<dimz))
		{
			in_idx = ix*dimz+iz;//iz*dimx+ix;
			
			wf_append_d[in_idx]=vx_x_d[in_idx]+vz_z_d[in_idx];
		}
}

__global__ void cal_sub_a_b_to_c(float *vx_x_d,float *vz_z_d,float *wf_append_d,int dimx,int dimz)
{
		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		if((ix<dimx)&&(iz<dimz))
		{
			in_idx = ix*dimz+iz;//iz*dimx+ix;
			
			wf_append_d[in_idx]=vx_x_d[in_idx]-vz_z_d[in_idx];
		}
}

__global__ void cal_mul_a_b_to_c(float *vx_x_d,float *vz_z_d,float *wf_append_d,int dimx,int dimz)
{
		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		if((ix<dimx)&&(iz<dimz))
		{
			in_idx = ix*dimz+iz;//iz*dimx+ix;
			
			wf_append_d[in_idx]=1.0*vx_x_d[in_idx]*vz_z_d[in_idx];
		}
}

//////2016年10月20日 星期四 00时23分35秒 弹性波反偏移算子 based on Zhou 2012 and Ren 2016/// Zongcai Feng, Gerard T:five equation!!!!
/////similar with the first-order velocity-stress equation
__global__ void demig_fwd_vx(float *vx2,float *vx1,float *txx1,float *txz1,float *attenuation_d,float dx,float dz,float dt,float *coe_d,float coe_x,float coe_z,int dimx,int dimz,float *density_d)//,float *vx,float *perturb_density,int mark_density)
//vx2_d,vx1_d,txx1_d,txz1_d,attenuation_d,dx,dz,dt,coe_opt_d,coe_x,coe_z,nx_append_radius2,nz_append_radius2
{
		__shared__ float s_data1[BDIMY2+2*radius2][BDIMX2+2*radius2];
		__shared__ float s_data2[BDIMY2+2*radius2][BDIMX2+2*radius2];
		float dt_real=dt/1000;
	//	float s_velocity;
		float s_attenuation;

		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		int tx = threadIdx.x+radius2;
		int tz = threadIdx.y+radius2;
		s_data1[tz][tx]=0.0;
		s_data1[threadIdx.y][threadIdx.x]=0.0;
		s_data1[BDIMY2+2*radius2-1-threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data1[threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data1[BDIMY2+2*radius2-1-threadIdx.y][threadIdx.x]=0.0;

		s_data2[tz][tx]=0.0;
		s_data2[threadIdx.y][threadIdx.x]=0.0;
		s_data2[BDIMY2+2*radius2-1-threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data2[threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data2[BDIMY2+2*radius2-1-threadIdx.y][threadIdx.x]=0.0;

		if((ix<dimx)&&(iz<dimz))
		{
				dimx=dimx+2*radius2;dimz=dimz+2*radius2;
				ix=ix+radius2;iz=iz+radius2;
				in_idx = ix*dimz+iz;//iz*dimx+ix;

				__syncthreads();

				s_data1[tz][tx]=txx1[in_idx];
				s_data2[tz][tx]=txz1[in_idx];

				if(threadIdx.y<radius2)
				{
						s_data1[threadIdx.y][tx]=txx1[in_idx-radius2];//g_input[in_idx-radius2*dimx];//up
						s_data1[threadIdx.y+BDIMY2+radius2][tx]=txx1[in_idx+BDIMY2];//g_input[in_idx+BDIMY2*dimx];//down

						s_data2[threadIdx.y][tx]=txz1[in_idx-radius2];//g_input[in_idx-radius2*dimx];//up
						s_data2[threadIdx.y+BDIMY2+radius2][tx]=txz1[in_idx+BDIMY2];//g_input[in_idx+BDIMY2*dimx];//down
				}
				if(threadIdx.x<radius2)
				{
						s_data1[tz][threadIdx.x]=txx1[in_idx-radius2*dimz];//g_input[in_idx-radius2];//left
						s_data1[tz][threadIdx.x+BDIMX2+radius2]=txx1[in_idx+BDIMX2*dimz];//g_input[in_idx+BDIMX2];//right

						s_data2[tz][threadIdx.x]=txz1[in_idx-radius2*dimz];//g_input[in_idx-radius2];//left
						s_data2[tz][threadIdx.x+BDIMX2+radius2]=txz1[in_idx+BDIMX2*dimz];//g_input[in_idx+BDIMX2];//right
				}

				//s_velocity=velocity_d[in_idx]*velocity_d[in_idx];
				s_attenuation=attenuation_d[in_idx];
				__syncthreads();

				float    sumx=coe_d[1]*(s_data1[tz][tx]-s_data1[tz][tx-1]);
					sumx+=coe_d[2]*(s_data1[tz][tx+1]-s_data1[tz][tx-2]);
					sumx+=coe_d[3]*(s_data1[tz][tx+2]-s_data1[tz][tx-3]);
					sumx+=coe_d[4]*(s_data1[tz][tx+3]-s_data1[tz][tx-4]);
					sumx+=coe_d[5]*(s_data1[tz][tx+4]-s_data1[tz][tx-5]);
					sumx+=coe_d[6]*(s_data1[tz][tx+5]-s_data1[tz][tx-6]);

				float    sumz=coe_d[1]*(s_data2[tz][tx]-s_data2[tz-1][tx]);
					sumz+=coe_d[2]*(s_data2[tz+1][tx]-s_data2[tz-2][tx]);
					sumz+=coe_d[3]*(s_data2[tz+2][tx]-s_data2[tz-3][tx]);
					sumz+=coe_d[4]*(s_data2[tz+3][tx]-s_data2[tz-4][tx]);
					sumz+=coe_d[5]*(s_data2[tz+4][tx]-s_data2[tz-5][tx]);
					sumz+=coe_d[6]*(s_data2[tz+5][tx]-s_data2[tz-6][tx]);

				vx2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*
						((1.0-s_attenuation*dt_real/2.0)*vx1[in_idx]+1.0/density_d[in_idx]*(sumx*coe_x+sumz*coe_z));
		}
}

__global__ void demig_fwd_vx_new(float *vx2,float *vx1,float *txx1,float *txz1,float *attenuation_d,float dx,float dz,float dt,float *coe_d,float coe_x,float coe_z,int dimx,int dimz,float *density_d,float *perturb_density_d,float *vx_t_d)//,float *vx,float *perturb_density,int mark_density)
//vx2_d,vx1_d,txx1_d,txz1_d,attenuation_d,dx,dz,dt,coe_opt_d,coe_x,coe_z,nx_append_radius2,nz_append_radius2
{
		__shared__ float s_data1[BDIMY2+2*radius2][BDIMX2+2*radius2];
		__shared__ float s_data2[BDIMY2+2*radius2][BDIMX2+2*radius2];
		float dt_real=dt/1000;
	//	float s_velocity;
		float s_attenuation;
		float perturb_density;

		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		int tx = threadIdx.x+radius2;
		int tz = threadIdx.y+radius2;
		s_data1[tz][tx]=0.0;
		s_data1[threadIdx.y][threadIdx.x]=0.0;
		s_data1[BDIMY2+2*radius2-1-threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data1[threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data1[BDIMY2+2*radius2-1-threadIdx.y][threadIdx.x]=0.0;

		s_data2[tz][tx]=0.0;
		s_data2[threadIdx.y][threadIdx.x]=0.0;
		s_data2[BDIMY2+2*radius2-1-threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data2[threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data2[BDIMY2+2*radius2-1-threadIdx.y][threadIdx.x]=0.0;

		if((ix<dimx)&&(iz<dimz))
		{
				dimx=dimx+2*radius2;dimz=dimz+2*radius2;
				ix=ix+radius2;iz=iz+radius2;
				in_idx = ix*dimz+iz;//iz*dimx+ix;

				__syncthreads();

				s_data1[tz][tx]=txx1[in_idx];
				s_data2[tz][tx]=txz1[in_idx];

				if(threadIdx.y<radius2)
				{
						s_data1[threadIdx.y][tx]=txx1[in_idx-radius2];//g_input[in_idx-radius2*dimx];//up
						s_data1[threadIdx.y+BDIMY2+radius2][tx]=txx1[in_idx+BDIMY2];//g_input[in_idx+BDIMY2*dimx];//down

						s_data2[threadIdx.y][tx]=txz1[in_idx-radius2];//g_input[in_idx-radius2*dimx];//up
						s_data2[threadIdx.y+BDIMY2+radius2][tx]=txz1[in_idx+BDIMY2];//g_input[in_idx+BDIMY2*dimx];//down
				}
				if(threadIdx.x<radius2)
				{
						s_data1[tz][threadIdx.x]=txx1[in_idx-radius2*dimz];//g_input[in_idx-radius2];//left
						s_data1[tz][threadIdx.x+BDIMX2+radius2]=txx1[in_idx+BDIMX2*dimz];//g_input[in_idx+BDIMX2];//right

						s_data2[tz][threadIdx.x]=txz1[in_idx-radius2*dimz];//g_input[in_idx-radius2];//left
						s_data2[tz][threadIdx.x+BDIMX2+radius2]=txz1[in_idx+BDIMX2*dimz];//g_input[in_idx+BDIMX2];//right
				}

				//s_velocity=velocity_d[in_idx]*velocity_d[in_idx];
				perturb_density=perturb_density_d[in_idx];
				s_attenuation=attenuation_d[in_idx];
				__syncthreads();

				float    sumx=coe_d[1]*(s_data1[tz][tx]-s_data1[tz][tx-1]);
					sumx+=coe_d[2]*(s_data1[tz][tx+1]-s_data1[tz][tx-2]);
					sumx+=coe_d[3]*(s_data1[tz][tx+2]-s_data1[tz][tx-3]);
					sumx+=coe_d[4]*(s_data1[tz][tx+3]-s_data1[tz][tx-4]);
					sumx+=coe_d[5]*(s_data1[tz][tx+4]-s_data1[tz][tx-5]);
					sumx+=coe_d[6]*(s_data1[tz][tx+5]-s_data1[tz][tx-6]);

				float    sumz=coe_d[1]*(s_data2[tz][tx]-s_data2[tz-1][tx]);
					sumz+=coe_d[2]*(s_data2[tz+1][tx]-s_data2[tz-2][tx]);
					sumz+=coe_d[3]*(s_data2[tz+2][tx]-s_data2[tz-3][tx]);
					sumz+=coe_d[4]*(s_data2[tz+3][tx]-s_data2[tz-4][tx]);
					sumz+=coe_d[5]*(s_data2[tz+4][tx]-s_data2[tz-5][tx]);
					sumz+=coe_d[6]*(s_data2[tz+5][tx]-s_data2[tz-6][tx]);

				vx2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*
						((1.0-s_attenuation*dt_real/2.0)*vx1[in_idx]+1.0/density_d[in_idx]*(sumx*coe_x+sumz*coe_z)-perturb_density*vx_t_d[in_idx]);
		}
}

__global__ void demig_fwd_vz(float *vz2,float *vz1,float *tzz1,float *txz1,float *attenuation_d,float dx,float dz,float dt,float *coe_d,float coe_x,float coe_z,int dimx,int dimz,float *density_d)
{
		__shared__ float s_data1[BDIMY2+2*radius2][BDIMX2+2*radius2];
		__shared__ float s_data2[BDIMY2+2*radius2][BDIMX2+2*radius2];
		float dt_real=dt/1000;
	//	float s_velocity;
		float s_attenuation;

		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		int tx = threadIdx.x+radius2;
		int tz = threadIdx.y+radius2;
		s_data1[tz][tx]=0.0;
		s_data1[threadIdx.y][threadIdx.x]=0.0;
		s_data1[BDIMY2+2*radius2-1-threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data1[threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data1[BDIMY2+2*radius2-1-threadIdx.y][threadIdx.x]=0.0;

		s_data2[tz][tx]=0.0;
		s_data2[threadIdx.y][threadIdx.x]=0.0;
		s_data2[BDIMY2+2*radius2-1-threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data2[threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data2[BDIMY2+2*radius2-1-threadIdx.y][threadIdx.x]=0.0;

		if((ix<dimx)&&(iz<dimz))
		{
				dimx=dimx+2*radius2;dimz=dimz+2*radius2;
				ix=ix+radius2;iz=iz+radius2;
				in_idx = ix*dimz+iz;//iz*dimx+ix;

				__syncthreads();

				s_data1[tz][tx]=tzz1[in_idx];
				s_data2[tz][tx]=txz1[in_idx];

				if(threadIdx.y<radius2)
				{
						s_data1[threadIdx.y][tx]=tzz1[in_idx-radius2];//g_input[in_idx-radius2*dimx];//up
						s_data1[threadIdx.y+BDIMY2+radius2][tx]=tzz1[in_idx+BDIMY2];//g_input[in_idx+BDIMY2*dimx];//down
						s_data2[threadIdx.y][tx]=txz1[in_idx-radius2];//g_input[in_idx-radius2*dimx];//up
						s_data2[threadIdx.y+BDIMY2+radius2][tx]=txz1[in_idx+BDIMY2];//g_input[in_idx+BDIMY2*dimx];//down
				}
				if(threadIdx.x<radius2)
				{
						s_data1[tz][threadIdx.x]=tzz1[in_idx-radius2*dimz];//g_input[in_idx-radius2];//left
						s_data1[tz][threadIdx.x+BDIMX2+radius2]=tzz1[in_idx+BDIMX2*dimz];//g_input[in_idx+BDIMX2];//right
						s_data2[tz][threadIdx.x]=txz1[in_idx-radius2*dimz];//g_input[in_idx-radius2];//left
						s_data2[tz][threadIdx.x+BDIMX2+radius2]=txz1[in_idx+BDIMX2*dimz];//g_input[in_idx+BDIMX2];//right
				}

				//s_velocity=velocity_d[in_idx]*velocity_d[in_idx];
				s_attenuation=attenuation_d[in_idx];
				__syncthreads();

				float    sumx=coe_d[1]*(s_data2[tz][tx+1]-s_data2[tz][tx]);
					sumx+=coe_d[2]*(s_data2[tz][tx+2]-s_data2[tz][tx-1]);
					sumx+=coe_d[3]*(s_data2[tz][tx+3]-s_data2[tz][tx-2]);
					sumx+=coe_d[4]*(s_data2[tz][tx+4]-s_data2[tz][tx-3]);
					sumx+=coe_d[5]*(s_data2[tz][tx+5]-s_data2[tz][tx-4]);
					sumx+=coe_d[6]*(s_data2[tz][tx+6]-s_data2[tz][tx-5]);

				float    sumz=coe_d[1]*(s_data1[tz+1][tx]-s_data1[tz][tx]);
					sumz+=coe_d[2]*(s_data1[tz+2][tx]-s_data1[tz-1][tx]);
					sumz+=coe_d[3]*(s_data1[tz+3][tx]-s_data1[tz-2][tx]);
					sumz+=coe_d[4]*(s_data1[tz+4][tx]-s_data1[tz-3][tx]);
					sumz+=coe_d[5]*(s_data1[tz+5][tx]-s_data1[tz-4][tx]);
					sumz+=coe_d[6]*(s_data1[tz+6][tx]-s_data1[tz-5][tx]);

				vz2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*
						((1.0-s_attenuation*dt_real/2.0)*vz1[in_idx]+1.0/density_d[in_idx]*(sumx*coe_x+sumz*coe_z));
		}
}

__global__ void demig_fwd_vz_new(float *vz2,float *vz1,float *tzz1,float *txz1,float *attenuation_d,float dx,float dz,float dt,float *coe_d,float coe_x,float coe_z,int dimx,int dimz,float *density_d,float *perturb_density_d,float *vz_t_d)
{
		__shared__ float s_data1[BDIMY2+2*radius2][BDIMX2+2*radius2];
		__shared__ float s_data2[BDIMY2+2*radius2][BDIMX2+2*radius2];
		float dt_real=dt/1000;
	//	float s_velocity;
		float s_attenuation;
		float perturb_density;

		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		int tx = threadIdx.x+radius2;
		int tz = threadIdx.y+radius2;
		s_data1[tz][tx]=0.0;
		s_data1[threadIdx.y][threadIdx.x]=0.0;
		s_data1[BDIMY2+2*radius2-1-threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data1[threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data1[BDIMY2+2*radius2-1-threadIdx.y][threadIdx.x]=0.0;

		s_data2[tz][tx]=0.0;
		s_data2[threadIdx.y][threadIdx.x]=0.0;
		s_data2[BDIMY2+2*radius2-1-threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data2[threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data2[BDIMY2+2*radius2-1-threadIdx.y][threadIdx.x]=0.0;

		if((ix<dimx)&&(iz<dimz))
		{
				dimx=dimx+2*radius2;dimz=dimz+2*radius2;
				ix=ix+radius2;iz=iz+radius2;
				in_idx = ix*dimz+iz;//iz*dimx+ix;

				__syncthreads();

				s_data1[tz][tx]=tzz1[in_idx];
				s_data2[tz][tx]=txz1[in_idx];

				if(threadIdx.y<radius2)
				{
						s_data1[threadIdx.y][tx]=tzz1[in_idx-radius2];//g_input[in_idx-radius2*dimx];//up
						s_data1[threadIdx.y+BDIMY2+radius2][tx]=tzz1[in_idx+BDIMY2];//g_input[in_idx+BDIMY2*dimx];//down
						s_data2[threadIdx.y][tx]=txz1[in_idx-radius2];//g_input[in_idx-radius2*dimx];//up
						s_data2[threadIdx.y+BDIMY2+radius2][tx]=txz1[in_idx+BDIMY2];//g_input[in_idx+BDIMY2*dimx];//down
				}
				if(threadIdx.x<radius2)
				{
						s_data1[tz][threadIdx.x]=tzz1[in_idx-radius2*dimz];//g_input[in_idx-radius2];//left
						s_data1[tz][threadIdx.x+BDIMX2+radius2]=tzz1[in_idx+BDIMX2*dimz];//g_input[in_idx+BDIMX2];//right
						s_data2[tz][threadIdx.x]=txz1[in_idx-radius2*dimz];//g_input[in_idx-radius2];//left
						s_data2[tz][threadIdx.x+BDIMX2+radius2]=txz1[in_idx+BDIMX2*dimz];//g_input[in_idx+BDIMX2];//right
				}

				//s_velocity=velocity_d[in_idx]*velocity_d[in_idx];
				perturb_density=perturb_density_d[in_idx];
				s_attenuation=attenuation_d[in_idx];
				__syncthreads();

				float    sumx=coe_d[1]*(s_data2[tz][tx+1]-s_data2[tz][tx]);
					sumx+=coe_d[2]*(s_data2[tz][tx+2]-s_data2[tz][tx-1]);
					sumx+=coe_d[3]*(s_data2[tz][tx+3]-s_data2[tz][tx-2]);
					sumx+=coe_d[4]*(s_data2[tz][tx+4]-s_data2[tz][tx-3]);
					sumx+=coe_d[5]*(s_data2[tz][tx+5]-s_data2[tz][tx-4]);
					sumx+=coe_d[6]*(s_data2[tz][tx+6]-s_data2[tz][tx-5]);

				float    sumz=coe_d[1]*(s_data1[tz+1][tx]-s_data1[tz][tx]);
					sumz+=coe_d[2]*(s_data1[tz+2][tx]-s_data1[tz-1][tx]);
					sumz+=coe_d[3]*(s_data1[tz+3][tx]-s_data1[tz-2][tx]);
					sumz+=coe_d[4]*(s_data1[tz+4][tx]-s_data1[tz-3][tx]);
					sumz+=coe_d[5]*(s_data1[tz+5][tx]-s_data1[tz-4][tx]);
					sumz+=coe_d[6]*(s_data1[tz+6][tx]-s_data1[tz-5][tx]);

				vz2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*
						((1.0-s_attenuation*dt_real/2.0)*vz1[in_idx]+1.0/density_d[in_idx]*(sumx*coe_x+sumz*coe_z)-perturb_density*vz_t_d[in_idx]);
		}
}

__global__ void demig_fwd_txxzzxz(float *txx2,float *txx1,float *tzz2,float *tzz1,float *txz2,float *txz1,float *vx2,float *vz2,float *velocity_d,float *velocity1_d,float *attenuation_d,float dt,float *coe_d,float coe_x,float coe_z,int dimx,int dimz,float *density_d,float *perturb_lame1_d,float *perturb_lame2_d,float *vx_x_d,float *vx_z_d,float *vz_x_d,float *vz_z_d)
//demig_fwd_txxzzxz<<<dimGrid,dimBlock>>>(rtxx2_d,rtxx1_d,rtzz2_d,rtzz1_d,rtxz2_d,rtxz1_d,rvx1_d,rvz1_d,s_velocity_d,s_velocity1_d,attenuation_d,dt,coe_opt_d,coe_x,coe_z,nx_append_radius2,nz_append_radius2,s_density_d,tmp_perturb_lame1_d,tmp_perturb_lame2_d,vx_x_d,vx_z_d,vz_x_d,vz_z_d);
{
		__shared__ float s_data1[BDIMY2+2*radius2][BDIMX2+2*radius2];
		__shared__ float s_data2[BDIMY2+2*radius2][BDIMX2+2*radius2];
		float dt_real=dt/1000;
		float s_velocity;
		float s_velocity1;
		float perturb_lame1;
		float perturb_lame2;
		float s_attenuation;

		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		int tx = threadIdx.x+radius2;
		int tz = threadIdx.y+radius2;
		s_data1[tz][tx]=0.0;
		s_data1[threadIdx.y][threadIdx.x]=0.0;
		s_data1[BDIMY2+2*radius2-1-threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data1[threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data1[BDIMY2+2*radius2-1-threadIdx.y][threadIdx.x]=0.0;

		s_data2[tz][tx]=0.0;
		s_data2[threadIdx.y][threadIdx.x]=0.0;
		s_data2[BDIMY2+2*radius2-1-threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data2[threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data2[BDIMY2+2*radius2-1-threadIdx.y][threadIdx.x]=0.0;

		if((ix<dimx)&&(iz<dimz))
		{
				dimx=dimx+2*radius2;dimz=dimz+2*radius2;
				ix=ix+radius2;iz=iz+radius2;
				in_idx = ix*dimz+iz;//iz*dimx+ix;

				__syncthreads();

				s_data1[tz][tx]=vx2[in_idx];
				s_data2[tz][tx]=vz2[in_idx];

				if(threadIdx.y<radius2)
				{
						s_data1[threadIdx.y][tx]=vx2[in_idx-radius2];//g_input[in_idx-radius2*dimx];//up
						s_data1[threadIdx.y+BDIMY2+radius2][tx]=vx2[in_idx+BDIMY2];//g_input[in_idx+BDIMY2*dimx];//down

						s_data2[threadIdx.y][tx]=vz2[in_idx-radius2];//g_input[in_idx-radius2*dimx];//up
						s_data2[threadIdx.y+BDIMY2+radius2][tx]=vz2[in_idx+BDIMY2];//g_input[in_idx+BDIMY2*dimx];//down
				}
				if(threadIdx.x<radius2)
				{
						s_data1[tz][threadIdx.x]=vx2[in_idx-radius2*dimz];//g_input[in_idx-radius2];//left
						s_data1[tz][threadIdx.x+BDIMX2+radius2]=vx2[in_idx+BDIMX2*dimz];//g_input[in_idx+BDIMX2];//right

						s_data2[tz][threadIdx.x]=vz2[in_idx-radius2*dimz];//g_input[in_idx-radius2];//left
						s_data2[tz][threadIdx.x+BDIMX2+radius2]=vz2[in_idx+BDIMX2*dimz];//g_input[in_idx+BDIMX2];//right
				}

				s_velocity=velocity_d[in_idx]*velocity_d[in_idx];
				s_velocity1=velocity1_d[in_idx]*velocity1_d[in_idx];

				perturb_lame1=perturb_lame1_d[in_idx];
				perturb_lame2=perturb_lame2_d[in_idx];

				s_attenuation=attenuation_d[in_idx];
				__syncthreads();
		

				float    sumx=coe_d[1]*(s_data1[tz][tx+1]-s_data1[tz][tx]);
					sumx+=coe_d[2]*(s_data1[tz][tx+2]-s_data1[tz][tx-1]);
					sumx+=coe_d[3]*(s_data1[tz][tx+3]-s_data1[tz][tx-2]);
					sumx+=coe_d[4]*(s_data1[tz][tx+4]-s_data1[tz][tx-3]);
					sumx+=coe_d[5]*(s_data1[tz][tx+5]-s_data1[tz][tx-4]);
					sumx+=coe_d[6]*(s_data1[tz][tx+6]-s_data1[tz][tx-5]);

				float    sumz=coe_d[1]*(s_data2[tz][tx]-  s_data2[tz-1][tx]);
					sumz+=coe_d[2]*(s_data2[tz+1][tx]-s_data2[tz-2][tx]);
					sumz+=coe_d[3]*(s_data2[tz+2][tx]-s_data2[tz-3][tx]);
					sumz+=coe_d[4]*(s_data2[tz+3][tx]-s_data2[tz-4][tx]);
					sumz+=coe_d[5]*(s_data2[tz+4][tx]-s_data2[tz-5][tx]);
					sumz+=coe_d[6]*(s_data2[tz+5][tx]-s_data2[tz-6][tx]);


				txx2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*txx1[in_idx]+
						s_velocity*density_d[in_idx]*sumx*coe_x+(s_velocity-2*s_velocity1)*density_d[in_idx]*sumz*coe_z+
						(perturb_lame1+2*perturb_lame2)*vx_x_d[in_idx]*dt_real+perturb_lame1*vz_z_d[in_idx]*dt_real);//s_velocity  and  s_velocity1

				tzz2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*tzz1[in_idx]+
						(s_velocity-2*s_velocity1)*density_d[in_idx]*sumx*coe_x+s_velocity*density_d[in_idx]*sumz*coe_z+
						(perturb_lame1+2*perturb_lame2)*vz_z_d[in_idx]*dt_real+perturb_lame1*vx_x_d[in_idx]*dt_real);//sumx  and  sumz 
							
				float    sumx1=coe_d[1]*(s_data2[tz][tx]-  s_data2[tz][tx-1]);
					sumx1+=coe_d[2]*(s_data2[tz][tx+1]-s_data2[tz][tx-2]);
					sumx1+=coe_d[3]*(s_data2[tz][tx+2]-s_data2[tz][tx-3]);
					sumx1+=coe_d[4]*(s_data2[tz][tx+3]-s_data2[tz][tx-4]);
					sumx1+=coe_d[5]*(s_data2[tz][tx+4]-s_data2[tz][tx-5]);
					sumx1+=coe_d[6]*(s_data2[tz][tx+5]-s_data2[tz][tx-6]);

				float    sumz1=coe_d[1]*(s_data1[tz+1][tx]-s_data1[tz][tx]);
					sumz1+=coe_d[2]*(s_data1[tz+2][tx]-s_data1[tz-1][tx]);
					sumz1+=coe_d[3]*(s_data1[tz+3][tx]-s_data1[tz-2][tx]);
					sumz1+=coe_d[4]*(s_data1[tz+4][tx]-s_data1[tz-3][tx]);
					sumz1+=coe_d[5]*(s_data1[tz+5][tx]-s_data1[tz-4][tx]);
					sumz1+=coe_d[6]*(s_data1[tz+6][tx]-s_data1[tz-5][tx]);

				txz2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*
						((1.0-s_attenuation*dt_real/2.0)*txz1[in_idx]+s_velocity1*density_d[in_idx]*(sumx1*coe_x+sumz1*coe_z)+
						perturb_lame2*vx_z_d[in_idx]*dt_real+perturb_lame2*vz_x_d[in_idx]*dt_real);
		}
}
//////2017年01月04日 星期三 09时24分28秒 弹性波反偏移算子 based on Zhou 2012 and Ren 2016/// Zongcai Feng, Gerard T:five equation!!!!
/////similar with the first-order velocity-stress equation
__global__ void demig_fwd_vx_mul(float *vx2,float *vx1,float *txx1,float *txz1,float *attenuation_d,float dx,float dz,float dt,float *coe_d,float coe_x,float coe_z,int dimx,int dimz,float *density_d,float *d1_d,float *d2_d,float *d3_d,float *d4_d,float *d5_d)
//demig_fwd_vx_mul<<<dimGrid,dimBlock>>>(rvx2_d,rvx1_d,rtxx2_d,rtxz2_d,attenuation_d,dx,dz,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,s_density_d,dem_p1_d,dem_p2_d,dem_p3_d,dem_p4_d,dem_p5_d);
{
		__shared__ float s_data1[BDIMY2+2*radius2][BDIMX2+2*radius2];
		__shared__ float s_data2[BDIMY2+2*radius2][BDIMX2+2*radius2];
		float dt_real=dt/1000;
	//	float s_velocity;
		float s_attenuation;

		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		int tx = threadIdx.x+radius2;
		int tz = threadIdx.y+radius2;
		s_data1[tz][tx]=0.0;
		s_data1[threadIdx.y][threadIdx.x]=0.0;
		s_data1[BDIMY2+2*radius2-1-threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data1[threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data1[BDIMY2+2*radius2-1-threadIdx.y][threadIdx.x]=0.0;

		s_data2[tz][tx]=0.0;
		s_data2[threadIdx.y][threadIdx.x]=0.0;
		s_data2[BDIMY2+2*radius2-1-threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data2[threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data2[BDIMY2+2*radius2-1-threadIdx.y][threadIdx.x]=0.0;

		if((ix<dimx)&&(iz<dimz))
		{
				dimx=dimx+2*radius2;dimz=dimz+2*radius2;
				ix=ix+radius2;iz=iz+radius2;
				in_idx = ix*dimz+iz;//iz*dimx+ix;

				__syncthreads();

				s_data1[tz][tx]=txx1[in_idx];
				s_data2[tz][tx]=txz1[in_idx];

				if(threadIdx.y<radius2)
				{
						s_data1[threadIdx.y][tx]=txx1[in_idx-radius2];//g_input[in_idx-radius2*dimx];//up
						s_data1[threadIdx.y+BDIMY2+radius2][tx]=txx1[in_idx+BDIMY2];//g_input[in_idx+BDIMY2*dimx];//down

						s_data2[threadIdx.y][tx]=txz1[in_idx-radius2];//g_input[in_idx-radius2*dimx];//up
						s_data2[threadIdx.y+BDIMY2+radius2][tx]=txz1[in_idx+BDIMY2];//g_input[in_idx+BDIMY2*dimx];//down
				}
				if(threadIdx.x<radius2)
				{
						s_data1[tz][threadIdx.x]=txx1[in_idx-radius2*dimz];//g_input[in_idx-radius2];//left
						s_data1[tz][threadIdx.x+BDIMX2+radius2]=txx1[in_idx+BDIMX2*dimz];//g_input[in_idx+BDIMX2];//right

						s_data2[tz][threadIdx.x]=txz1[in_idx-radius2*dimz];//g_input[in_idx-radius2];//left
						s_data2[tz][threadIdx.x+BDIMX2+radius2]=txz1[in_idx+BDIMX2*dimz];//g_input[in_idx+BDIMX2];//right
				}

				//s_velocity=velocity_d[in_idx]*velocity_d[in_idx];
				s_attenuation=attenuation_d[in_idx];
				__syncthreads();

				float    sumx=coe_d[1]*(s_data1[tz][tx]-s_data1[tz][tx-1]);
					sumx+=coe_d[2]*(s_data1[tz][tx+1]-s_data1[tz][tx-2]);
					sumx+=coe_d[3]*(s_data1[tz][tx+2]-s_data1[tz][tx-3]);
					sumx+=coe_d[4]*(s_data1[tz][tx+3]-s_data1[tz][tx-4]);
					sumx+=coe_d[5]*(s_data1[tz][tx+4]-s_data1[tz][tx-5]);
					sumx+=coe_d[6]*(s_data1[tz][tx+5]-s_data1[tz][tx-6]);

				float    sumz=coe_d[1]*(s_data2[tz][tx]-s_data2[tz-1][tx]);
					sumz+=coe_d[2]*(s_data2[tz+1][tx]-s_data2[tz-2][tx]);
					sumz+=coe_d[3]*(s_data2[tz+2][tx]-s_data2[tz-3][tx]);
					sumz+=coe_d[4]*(s_data2[tz+3][tx]-s_data2[tz-4][tx]);
					sumz+=coe_d[5]*(s_data2[tz+4][tx]-s_data2[tz-5][tx]);
					sumz+=coe_d[6]*(s_data2[tz+5][tx]-s_data2[tz-6][tx]);

				vx2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*
						((1.0-s_attenuation*dt_real/2.0)*vx1[in_idx]+1.0/density_d[in_idx]*(sumx*coe_x+sumz*coe_z)+1.0*d1_d[in_idx]*dt_real/density_d[in_idx]);
		}
}

__global__ void demig_fwd_vz_mul(float *vz2,float *vz1,float *tzz1,float *txz1,float *attenuation_d,float dx,float dz,float dt,float *coe_d,float coe_x,float coe_z,int dimx,int dimz,float *density_d,float *d1_d,float *d2_d,float *d3_d,float *d4_d,float *d5_d)
//demig_fwd_vz_mul<<<dimGrid,dimBlock>>>(rvz2_d,rvz1_d,rtzz2_d,rtxz2_d,attenuation_d,dx,dz,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,s_density_d,dem_p1_d,dem_p2_d,dem_p3_d,dem_p4_d,dem_p5_d);
{
		__shared__ float s_data1[BDIMY2+2*radius2][BDIMX2+2*radius2];
		__shared__ float s_data2[BDIMY2+2*radius2][BDIMX2+2*radius2];
		float dt_real=dt/1000;
	//	float s_velocity;
		float s_attenuation;

		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		int tx = threadIdx.x+radius2;
		int tz = threadIdx.y+radius2;
		s_data1[tz][tx]=0.0;
		s_data1[threadIdx.y][threadIdx.x]=0.0;
		s_data1[BDIMY2+2*radius2-1-threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data1[threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data1[BDIMY2+2*radius2-1-threadIdx.y][threadIdx.x]=0.0;

		s_data2[tz][tx]=0.0;
		s_data2[threadIdx.y][threadIdx.x]=0.0;
		s_data2[BDIMY2+2*radius2-1-threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data2[threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data2[BDIMY2+2*radius2-1-threadIdx.y][threadIdx.x]=0.0;

		if((ix<dimx)&&(iz<dimz))
		{
				dimx=dimx+2*radius2;dimz=dimz+2*radius2;
				ix=ix+radius2;iz=iz+radius2;
				in_idx = ix*dimz+iz;//iz*dimx+ix;

				__syncthreads();

				s_data1[tz][tx]=tzz1[in_idx];
				s_data2[tz][tx]=txz1[in_idx];

				if(threadIdx.y<radius2)
				{
						s_data1[threadIdx.y][tx]=tzz1[in_idx-radius2];//g_input[in_idx-radius2*dimx];//up
						s_data1[threadIdx.y+BDIMY2+radius2][tx]=tzz1[in_idx+BDIMY2];//g_input[in_idx+BDIMY2*dimx];//down
						s_data2[threadIdx.y][tx]=txz1[in_idx-radius2];//g_input[in_idx-radius2*dimx];//up
						s_data2[threadIdx.y+BDIMY2+radius2][tx]=txz1[in_idx+BDIMY2];//g_input[in_idx+BDIMY2*dimx];//down
				}
				if(threadIdx.x<radius2)
				{
						s_data1[tz][threadIdx.x]=tzz1[in_idx-radius2*dimz];//g_input[in_idx-radius2];//left
						s_data1[tz][threadIdx.x+BDIMX2+radius2]=tzz1[in_idx+BDIMX2*dimz];//g_input[in_idx+BDIMX2];//right
						s_data2[tz][threadIdx.x]=txz1[in_idx-radius2*dimz];//g_input[in_idx-radius2];//left
						s_data2[tz][threadIdx.x+BDIMX2+radius2]=txz1[in_idx+BDIMX2*dimz];//g_input[in_idx+BDIMX2];//right
				}

				//s_velocity=velocity_d[in_idx]*velocity_d[in_idx];
				s_attenuation=attenuation_d[in_idx];
				__syncthreads();

				float    sumx=coe_d[1]*(s_data2[tz][tx+1]-s_data2[tz][tx]);
					sumx+=coe_d[2]*(s_data2[tz][tx+2]-s_data2[tz][tx-1]);
					sumx+=coe_d[3]*(s_data2[tz][tx+3]-s_data2[tz][tx-2]);
					sumx+=coe_d[4]*(s_data2[tz][tx+4]-s_data2[tz][tx-3]);
					sumx+=coe_d[5]*(s_data2[tz][tx+5]-s_data2[tz][tx-4]);
					sumx+=coe_d[6]*(s_data2[tz][tx+6]-s_data2[tz][tx-5]);

				float    sumz=coe_d[1]*(s_data1[tz+1][tx]-s_data1[tz][tx]);
					sumz+=coe_d[2]*(s_data1[tz+2][tx]-s_data1[tz-1][tx]);
					sumz+=coe_d[3]*(s_data1[tz+3][tx]-s_data1[tz-2][tx]);
					sumz+=coe_d[4]*(s_data1[tz+4][tx]-s_data1[tz-3][tx]);
					sumz+=coe_d[5]*(s_data1[tz+5][tx]-s_data1[tz-4][tx]);
					sumz+=coe_d[6]*(s_data1[tz+6][tx]-s_data1[tz-5][tx]);

				vz2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*
						((1.0-s_attenuation*dt_real/2.0)*vz1[in_idx]+1.0/density_d[in_idx]*(sumx*coe_x+sumz*coe_z)+1.0*d2_d[in_idx]*dt_real/density_d[in_idx]);
		}
}

__global__ void demig_fwd_txxzzxz_mul(float *txx2,float *txx1,float *tzz2,float *tzz1,float *txz2,float *txz1,float *vx2,float *vz2,float *velocity_d,float *velocity1_d,float *attenuation_d,float dt,float *coe_d,float coe_x,float coe_z,int dimx,int dimz,float *density_d,float *d1_d,float *d2_d,float *d3_d,float *d4_d,float *d5_d)
//demig_fwd_txxzzxz_mul<<<dimGrid,dimBlock>>>(rtxx2_d,rtxx1_d,rtzz2_d,rtzz1_d,rtxz2_d,rtxz1_d,rvx1_d,rvz1_d,s_velocity_d,s_velocity1_d,attenuation_d,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,s_density_d,dem_p1_d,dem_p2_d,dem_p3_d,dem_p4_d,dem_p5_d);
{
		__shared__ float s_data1[BDIMY2+2*radius2][BDIMX2+2*radius2];
		__shared__ float s_data2[BDIMY2+2*radius2][BDIMX2+2*radius2];
		float dt_real=dt/1000;
		float s_velocity;
		float s_velocity1;

		float s_attenuation;

		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		int tx = threadIdx.x+radius2;
		int tz = threadIdx.y+radius2;
		s_data1[tz][tx]=0.0;
		s_data1[threadIdx.y][threadIdx.x]=0.0;
		s_data1[BDIMY2+2*radius2-1-threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data1[threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data1[BDIMY2+2*radius2-1-threadIdx.y][threadIdx.x]=0.0;

		s_data2[tz][tx]=0.0;
		s_data2[threadIdx.y][threadIdx.x]=0.0;
		s_data2[BDIMY2+2*radius2-1-threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data2[threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data2[BDIMY2+2*radius2-1-threadIdx.y][threadIdx.x]=0.0;

		if((ix<dimx)&&(iz<dimz))
		{
				dimx=dimx+2*radius2;dimz=dimz+2*radius2;
				ix=ix+radius2;iz=iz+radius2;
				in_idx = ix*dimz+iz;//iz*dimx+ix;

				__syncthreads();

				s_data1[tz][tx]=vx2[in_idx];
				s_data2[tz][tx]=vz2[in_idx];

				if(threadIdx.y<radius2)
				{
						s_data1[threadIdx.y][tx]=vx2[in_idx-radius2];//g_input[in_idx-radius2*dimx];//up
						s_data1[threadIdx.y+BDIMY2+radius2][tx]=vx2[in_idx+BDIMY2];//g_input[in_idx+BDIMY2*dimx];//down

						s_data2[threadIdx.y][tx]=vz2[in_idx-radius2];//g_input[in_idx-radius2*dimx];//up
						s_data2[threadIdx.y+BDIMY2+radius2][tx]=vz2[in_idx+BDIMY2];//g_input[in_idx+BDIMY2*dimx];//down
				}
				if(threadIdx.x<radius2)
				{
						s_data1[tz][threadIdx.x]=vx2[in_idx-radius2*dimz];//g_input[in_idx-radius2];//left
						s_data1[tz][threadIdx.x+BDIMX2+radius2]=vx2[in_idx+BDIMX2*dimz];//g_input[in_idx+BDIMX2];//right

						s_data2[tz][threadIdx.x]=vz2[in_idx-radius2*dimz];//g_input[in_idx-radius2];//left
						s_data2[tz][threadIdx.x+BDIMX2+radius2]=vz2[in_idx+BDIMX2*dimz];//g_input[in_idx+BDIMX2];//right
				}

				s_velocity=velocity_d[in_idx]*velocity_d[in_idx];
				s_velocity1=velocity1_d[in_idx]*velocity1_d[in_idx];

				s_attenuation=attenuation_d[in_idx];
				__syncthreads();
		

				float    sumx=coe_d[1]*(s_data1[tz][tx+1]-s_data1[tz][tx]);
					sumx+=coe_d[2]*(s_data1[tz][tx+2]-s_data1[tz][tx-1]);
					sumx+=coe_d[3]*(s_data1[tz][tx+3]-s_data1[tz][tx-2]);
					sumx+=coe_d[4]*(s_data1[tz][tx+4]-s_data1[tz][tx-3]);
					sumx+=coe_d[5]*(s_data1[tz][tx+5]-s_data1[tz][tx-4]);
					sumx+=coe_d[6]*(s_data1[tz][tx+6]-s_data1[tz][tx-5]);

				float    sumz=coe_d[1]*(s_data2[tz][tx]-  s_data2[tz-1][tx]);
					sumz+=coe_d[2]*(s_data2[tz+1][tx]-s_data2[tz-2][tx]);
					sumz+=coe_d[3]*(s_data2[tz+2][tx]-s_data2[tz-3][tx]);
					sumz+=coe_d[4]*(s_data2[tz+3][tx]-s_data2[tz-4][tx]);
					sumz+=coe_d[5]*(s_data2[tz+4][tx]-s_data2[tz-5][tx]);
					sumz+=coe_d[6]*(s_data2[tz+5][tx]-s_data2[tz-6][tx]);


				txx2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*txx1[in_idx]+
						s_velocity*density_d[in_idx]*sumx*coe_x+(s_velocity-2*s_velocity1)*density_d[in_idx]*sumz*coe_z+
						d3_d[in_idx]*dt_real);//s_velocity  and  s_velocity1

				tzz2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*tzz1[in_idx]+
						(s_velocity-2*s_velocity1)*density_d[in_idx]*sumx*coe_x+s_velocity*density_d[in_idx]*sumz*coe_z+
						d4_d[in_idx]*dt_real);//sumx  and  sumz 
							
				float    sumx1=coe_d[1]*(s_data2[tz][tx]-  s_data2[tz][tx-1]);
					sumx1+=coe_d[2]*(s_data2[tz][tx+1]-s_data2[tz][tx-2]);
					sumx1+=coe_d[3]*(s_data2[tz][tx+2]-s_data2[tz][tx-3]);
					sumx1+=coe_d[4]*(s_data2[tz][tx+3]-s_data2[tz][tx-4]);
					sumx1+=coe_d[5]*(s_data2[tz][tx+4]-s_data2[tz][tx-5]);
					sumx1+=coe_d[6]*(s_data2[tz][tx+5]-s_data2[tz][tx-6]);

				float    sumz1=coe_d[1]*(s_data1[tz+1][tx]-s_data1[tz][tx]);
					sumz1+=coe_d[2]*(s_data1[tz+2][tx]-s_data1[tz-1][tx]);
					sumz1+=coe_d[3]*(s_data1[tz+3][tx]-s_data1[tz-2][tx]);
					sumz1+=coe_d[4]*(s_data1[tz+4][tx]-s_data1[tz-3][tx]);
					sumz1+=coe_d[5]*(s_data1[tz+5][tx]-s_data1[tz-4][tx]);
					sumz1+=coe_d[6]*(s_data1[tz+6][tx]-s_data1[tz-5][tx]);

				txz2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*
						((1.0-s_attenuation*dt_real/2.0)*txz1[in_idx]+s_velocity1*density_d[in_idx]*(sumx1*coe_x+sumz1*coe_z)+
						d5_d[in_idx]*dt_real);
		}
}


///////2016年11月20日 星期日 05时59分52秒 
__global__ void cuda_cal_dem_parameter(float *dem_p1_d,float *dem_p2_d,float *dem_p3_d,float *dem_p4_d,float *dem_p5_d,float *vx_x_d,float *vx_z_d,float *vz_x_d,float *vz_z_d,float *vx_t_d,float *vz_t_d,float *tmp_perturb_lame1_d,float *tmp_perturb_lame2_d,float *tmp_perturb_den_d,int dimx,int dimz)
//cuda_cal_dem_parameter<<<dimGrid,dimBlock>>>(dem_p1_d,dem_p2_d,dem_p3_d,dem_p4_d,dem_p5_d,vx_x_d,vx_z_d,vz_x_d,vz_z_d,vx_t_d,vz_t_d,tmp_perturb_lame1_d,tmp_perturb_lame2_d,tmp_perturb_den_d,nx_append_radius,nz_append_radius);
{
		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		if((ix<dimx)&&(iz<dimz))
		{
				dimx=dimx+2*radius2;dimz=dimz+2*radius2;
				ix=ix+radius2;iz=iz+radius2;
				in_idx = ix*dimz+iz;//iz*dimx+ix;

				dem_p1_d[in_idx]=-1.0*tmp_perturb_den_d[in_idx]*vx_t_d[in_idx];
				dem_p2_d[in_idx]=-1.0*tmp_perturb_den_d[in_idx]*vz_t_d[in_idx];

				dem_p3_d[in_idx]=((tmp_perturb_lame1_d[in_idx]+2*tmp_perturb_lame2_d[in_idx])*vx_x_d[in_idx]+tmp_perturb_lame1_d[in_idx]*vz_z_d[in_idx]);

				dem_p4_d[in_idx]=((tmp_perturb_lame1_d[in_idx]+2*tmp_perturb_lame2_d[in_idx])*vz_z_d[in_idx]+tmp_perturb_lame1_d[in_idx]*vx_x_d[in_idx]);

				dem_p5_d[in_idx]=(tmp_perturb_lame2_d[in_idx]*vx_z_d[in_idx]+tmp_perturb_lame2_d[in_idx]*vz_x_d[in_idx]);
		}
}
/////2016年11月28日 星期一 05时14分07秒 
__global__ void cuda_cal_dem_parameter_new(float *dem_p1_d,float *dem_p2_d,float *dem_p3_d,float *dem_p4_d,float *dem_p5_d,float *vx1_d,float *vz1_d,float *txx1_d,float *txz1_d,float *tzz1_d,float *tmp_perturb_lame1_d,float *tmp_perturb_lame2_d,float *tmp_perturb_den_d,float *velocity_d,float *velocity1_d,float *s_density_d,int dimx,int dimz)
//<<<dimGrid,dimBlock>>>(dem_p1_d,dem_p2_d,dem_p3_d,dem_p4_d,dem_p5_d,vx1_d,vz1_d,txx1_d,tzz1_d,txz1_d,tmp_perturb_lame1_d,tmp_perturb_lame2_d,tmp_perturb_den_d,nx_append_radius,nz_append_radius);
{
		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;
		float s_velocity,s_velocity1;
		float d_x=0,d_z=0;


		if((ix<dimx)&&(iz<dimz))
		{
				dimx=dimx+2*radius2;dimz=dimz+2*radius2;
				ix=ix+radius2;iz=iz+radius2;
				in_idx = ix*dimz+iz;//iz*dimx+ix;
			
				s_velocity=velocity_d[in_idx]*velocity_d[in_idx];
				s_velocity1=velocity1_d[in_idx]*velocity1_d[in_idx];

				d_x=(s_velocity*s_density_d[in_idx]*txx1_d[in_idx]-(s_velocity-2*s_velocity1)*s_density_d[in_idx]*tzz1_d[in_idx])/(4*s_velocity1*s_density_d[in_idx]*(s_velocity-s_velocity1)*s_density_d[in_idx]);

				d_z=(s_velocity*s_density_d[in_idx]*tzz1_d[in_idx]-(s_velocity-2*s_velocity1)*s_density_d[in_idx]*txx1_d[in_idx])/(4*s_velocity1*s_density_d[in_idx]*(s_velocity-s_velocity1)*s_density_d[in_idx]);

				dem_p1_d[in_idx]=tmp_perturb_den_d[in_idx]*vx1_d[in_idx];
				dem_p2_d[in_idx]=tmp_perturb_den_d[in_idx]*vz1_d[in_idx];

				dem_p3_d[in_idx]=-1.0*((tmp_perturb_lame1_d[in_idx]+2*tmp_perturb_lame2_d[in_idx])*d_x+tmp_perturb_lame1_d[in_idx]*d_z);

				dem_p4_d[in_idx]=-1.0*((tmp_perturb_lame1_d[in_idx]+2*tmp_perturb_lame2_d[in_idx])*d_z+tmp_perturb_lame1_d[in_idx]*d_x);

				dem_p5_d[in_idx]=-1.0*tmp_perturb_lame2_d[in_idx]*txz1_d[in_idx]/(s_velocity1*s_density_d[in_idx]);
		}
}

__global__ void cuda_cal_dem_parameter_lame(float *dem_p1_d,float *dem_p2_d,float *dem_p3_d,float *dem_p4_d,float *dem_p5_d,float *vx_x_d,float *vx_z_d,float *vz_x_d,float *vz_z_d,float *vx_t_d,float *vz_t_d,float *tmp_perturb_lame1_d,float *tmp_perturb_lame2_d,float *tmp_perturb_den_d,int dimx,int dimz,float *s_velocity_d,float *s_velocity1_d,float *s_density_d)
//cuda_cal_dem_parameter_lame<<<dimGrid,dimBlock>>>(dem_p1_d,dem_p2_d,dem_p3_d,dem_p4_d,dem_p5_d,vx_x_d,vx_z_d,vz_x_d,vz_z_d,vx_t_d,vz_t_d,tmp_perturb_lame1_d,tmp_perturb_lame2_d,tmp_perturb_den_d,nx_append_radius,nz_append_radius,s_velocity_d,s_velocity1_d,s_density_d);
{
		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		float lame1,lame2;

		if((ix<dimx)&&(iz<dimz))
		{
				dimx=dimx+2*radius2;dimz=dimz+2*radius2;
				ix=ix+radius2;iz=iz+radius2;
				in_idx = ix*dimz+iz;//iz*dimx+ix;

				lame1=s_density_d[in_idx]*s_velocity_d[in_idx]*s_velocity_d[in_idx]-2*s_density_d[in_idx]*s_velocity1_d[in_idx]*s_velocity1_d[in_idx];
				lame2=s_density_d[in_idx]*s_velocity1_d[in_idx]*s_velocity1_d[in_idx];

				dem_p1_d[in_idx]=-1.0*tmp_perturb_den_d[in_idx]*vx_t_d[in_idx]*s_density_d[in_idx];
				dem_p2_d[in_idx]=-1.0*tmp_perturb_den_d[in_idx]*vz_t_d[in_idx]*s_density_d[in_idx];

				//dem_p3_d[in_idx]=((tmp_perturb_lame1_d[in_idx]+2*tmp_perturb_lame2_d[in_idx])*vx_x_d[in_idx]+tmp_perturb_lame1_d[in_idx]*vz_z_d[in_idx]);
				dem_p3_d[in_idx]=1.0*lame1*(vx_x_d[in_idx]+vz_z_d[in_idx])*tmp_perturb_lame1_d[in_idx]+2.0*lame2*vx_x_d[in_idx]*tmp_perturb_lame2_d[in_idx];

				dem_p4_d[in_idx]=1.0*lame1*(vx_x_d[in_idx]+vz_z_d[in_idx])*tmp_perturb_lame1_d[in_idx]+2.0*lame2*vz_z_d[in_idx]*tmp_perturb_lame2_d[in_idx];

				dem_p5_d[in_idx]=1.0*lame2*(vx_z_d[in_idx]+vz_x_d[in_idx])*tmp_perturb_lame2_d[in_idx];
		}
}

__global__ void cuda_cal_dem_parameter_velocity(float *dem_p1_d,float *dem_p2_d,float *dem_p3_d,float *dem_p4_d,float *dem_p5_d,float *vx_x_d,float *vx_z_d,float *vz_x_d,float *vz_z_d,float *vx_t_d,float *vz_t_d,float *tmp_perturb_vp_d,float *tmp_perturb_vs_d,float *tmp_perturb_density_d,int dimx,int dimz,float *s_velocity_d,float *s_velocity1_d,float *s_density_d)
//cuda_cal_dem_parameter_velocity<<<dimGrid,dimBlock>>>(dem_p1_d,dem_p2_d,dem_p3_d,dem_p4_d,dem_p5_d,vx_x_d,vx_z_d,vz_x_d,vz_z_d,vx_t_d,vz_t_d,tmp_perturb_vp_d,tmp_perturb_vs_d,tmp_perturb_density_d,nx_append_radius,nz_append_radius,s_velocity_d,s_velocity1_d,s_density_d);
{
		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		float lame1,lame2;

		if((ix<dimx)&&(iz<dimz))
		{
				dimx=dimx+2*radius2;dimz=dimz+2*radius2;
				ix=ix+radius2;iz=iz+radius2;
				in_idx = ix*dimz+iz;//iz*dimx+ix;

				lame1=s_density_d[in_idx]*s_velocity_d[in_idx]*s_velocity_d[in_idx]-2*s_density_d[in_idx]*s_velocity1_d[in_idx]*s_velocity1_d[in_idx];
				lame2=s_density_d[in_idx]*s_velocity1_d[in_idx]*s_velocity1_d[in_idx];

				dem_p1_d[in_idx]=-1.0*tmp_perturb_density_d[in_idx]*vx_t_d[in_idx]*s_density_d[in_idx];
				dem_p2_d[in_idx]=-1.0*tmp_perturb_density_d[in_idx]*vz_t_d[in_idx]*s_density_d[in_idx];

				dem_p3_d[in_idx]=1.0*((lame1+2.0*lame2)*vx_x_d[in_idx]+lame1*vz_z_d[in_idx])*tmp_perturb_density_d[in_idx]
						+2.0*(lame1+2.0*lame2)*(vx_x_d[in_idx]+vz_z_d[in_idx])*tmp_perturb_vp_d[in_idx]
						-4.0*lame2*vz_z_d[in_idx]*tmp_perturb_vs_d[in_idx];

				dem_p4_d[in_idx]=1.0*((lame1+2.0*lame2)*vz_z_d[in_idx]+lame1*vx_x_d[in_idx])*tmp_perturb_density_d[in_idx]
						+2.0*(lame1+2.0*lame2)*(vx_x_d[in_idx]+vz_z_d[in_idx])*tmp_perturb_vp_d[in_idx]
						-4.0*lame2*vx_x_d[in_idx]*tmp_perturb_vs_d[in_idx];

				dem_p5_d[in_idx]=1.0*lame2*(vx_z_d[in_idx]+vz_x_d[in_idx])*tmp_perturb_density_d[in_idx]
						//+2.0*(vx_z_d[in_idx]+vz_x_d[in_idx])*tmp_perturb_vs_d[in_idx];/////这个错误，找了一天，2017年01月10日 星期二 11时10分09秒 
						+2.0*lame2*(vx_z_d[in_idx]+vz_x_d[in_idx])*tmp_perturb_vs_d[in_idx];/////this error spend one day!!!!!，
		}
}

__global__ void cuda_cal_dem_parameter_impedance(float *dem_p1_d,float *dem_p2_d,float *dem_p3_d,float *dem_p4_d,float *dem_p5_d,float *vx_x_d,float *vx_z_d,float *vz_x_d,float *vz_z_d,float *vx_t_d,float *vz_t_d,float *tmp_perturb_vp_d,float *tmp_perturb_vs_d,float *tmp_perturb_density_d,int dimx,int dimz,float *s_velocity_d,float *s_velocity1_d,float *s_density_d)
//cuda_cal_dem_parameter<<<dimGrid,dimBlock>>>(dem_p1_d,dem_p2_d,dem_p3_d,dem_p4_d,dem_p5_d,vx_x_d,vx_z_d,vz_x_d,vz_z_d,vx_t_d,vz_t_d,tmp_perturb_lame1_d,tmp_perturb_lame2_d,tmp_perturb_den_d,nx_append_radius,nz_append_radius);
{
		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		float lame1,lame2;

		if((ix<dimx)&&(iz<dimz))
		{
				dimx=dimx+2*radius2;dimz=dimz+2*radius2;
				ix=ix+radius2;iz=iz+radius2;
				in_idx = ix*dimz+iz;//iz*dimx+ix;

				lame1=s_density_d[in_idx]*s_velocity_d[in_idx]*s_velocity_d[in_idx]-2*s_density_d[in_idx]*s_velocity1_d[in_idx]*s_velocity1_d[in_idx];
				lame2=s_density_d[in_idx]*s_velocity1_d[in_idx]*s_velocity1_d[in_idx];

				dem_p1_d[in_idx]=-1.0*tmp_perturb_density_d[in_idx]*vx_t_d[in_idx]*s_density_d[in_idx];
				dem_p2_d[in_idx]=-1.0*tmp_perturb_density_d[in_idx]*vz_t_d[in_idx]*s_density_d[in_idx];

				/*dem_p3_d[in_idx]=1.0*((lame1+2.0*lame2)*vx_x_d[in_idx]+lame1*vz_z_d[in_idx])*tmp_perturb_density_d[in_idx]
						+2.0*(lame1+2.0*lame2)*(vx_x_d[in_idx]+vz_z_d[in_idx])*tmp_perturb_vp_d[in_idx]
						-4.0*lame2*vz_z_d[in_idx]*tmp_perturb_vs_d[in_idx];*/

				dem_p3_d[in_idx]=-1.0*((lame1+2.0*lame2)*vx_x_d[in_idx]+lame1*vz_z_d[in_idx])*tmp_perturb_density_d[in_idx]
						+2.0*(lame1+2.0*lame2)*(vx_x_d[in_idx]+vz_z_d[in_idx])*tmp_perturb_vp_d[in_idx]
						-4.0*lame2*vz_z_d[in_idx]*tmp_perturb_vs_d[in_idx];

				/*dem_p4_d[in_idx]=1.0*((lame1+2.0*lame2)*vz_z_d[in_idx]+lame1*vx_x_d[in_idx])*tmp_perturb_density_d[in_idx]
						+2.0*(lame1+2.0*lame2)*(vx_x_d[in_idx]+vz_z_d[in_idx])*tmp_perturb_vp_d[in_idx]
						-4.0*lame2*vx_x_d[in_idx]*tmp_perturb_vs_d[in_idx];*/

				dem_p4_d[in_idx]=-1.0*((lame1+2.0*lame2)*vz_z_d[in_idx]+lame1*vx_x_d[in_idx])*tmp_perturb_density_d[in_idx]
						+2.0*(lame1+2.0*lame2)*(vx_x_d[in_idx]+vz_z_d[in_idx])*tmp_perturb_vp_d[in_idx]
						-4.0*lame2*vx_x_d[in_idx]*tmp_perturb_vs_d[in_idx];

				/*dem_p5_d[in_idx]=1.0*lame2*(vx_z_d[in_idx]+vz_x_d[in_idx])*tmp_perturb_density_d[in_idx]
						//+2.0*(vx_z_d[in_idx]+vz_x_d[in_idx])*tmp_perturb_vs_d[in_idx];/////这个错误，找了一天，2017年01月10日 星期二 11时10分09秒 
						+2.0*lame2*(vx_z_d[in_idx]+vz_x_d[in_idx])*tmp_perturb_vs_d[in_idx];/////this error spend one day!!!!!，*/

				dem_p5_d[in_idx]=-1.0*lame2*(vx_z_d[in_idx]+vz_x_d[in_idx])*tmp_perturb_density_d[in_idx]
						//+2.0*(vx_z_d[in_idx]+vz_x_d[in_idx])*tmp_perturb_vs_d[in_idx];/////这个错误，找了一天，2017年01月10日 星期二 11时10分09秒 
						+2.0*lame2*(vx_z_d[in_idx]+vz_x_d[in_idx])*tmp_perturb_vs_d[in_idx];/////this error spend one day!!!!!，
		}
}

//////2016年10月08日 星期六 09时53分37秒   一阶速度应力方程的伴随状态方程
__global__ void adjoint_fwd_vx(float *vx2,float *vx1,float *txx1,float *txz1,float *tzz1,float *attenuation_d,float dx,float dz,float dt,float *coe_d,float coe_x,float coe_z,int dimx,int dimz,float *velocity_d,float *velocity1_d,float *density_d)
//adjoint_fwd_vx<<<dimGrid,dimBlock>>>(rvx1_d,rvx2_d,rtxx1_d,rtxz1_d,rtzz1_d,attenuation_d,dx,dz,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,s_velocity_d,s_velocity1_d,s_density_d);
{
		__shared__ float s_data1[BDIMY2+2*radius2][BDIMX2+2*radius2];
		__shared__ float s_data2[BDIMY2+2*radius2][BDIMX2+2*radius2];
		__shared__ float s_data3[BDIMY2+2*radius2][BDIMX2+2*radius2];
		float dt_real=dt/1000;
		float s_velocity;
		float s_velocity1;
		float s_attenuation;

		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		int tx = threadIdx.x+radius2;
		int tz = threadIdx.y+radius2;
		s_data1[tz][tx]=0.0;
		s_data1[threadIdx.y][threadIdx.x]=0.0;
		s_data1[BDIMY2+2*radius2-1-threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data1[threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data1[BDIMY2+2*radius2-1-threadIdx.y][threadIdx.x]=0.0;

		s_data2[tz][tx]=0.0;
		s_data2[threadIdx.y][threadIdx.x]=0.0;
		s_data2[BDIMY2+2*radius2-1-threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data2[threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data2[BDIMY2+2*radius2-1-threadIdx.y][threadIdx.x]=0.0;

		s_data3[tz][tx]=0.0;
		s_data3[threadIdx.y][threadIdx.x]=0.0;
		s_data3[BDIMY2+2*radius2-1-threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data3[threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data3[BDIMY2+2*radius2-1-threadIdx.y][threadIdx.x]=0.0;

		if((ix<dimx)&&(iz<dimz))
		{
				dimx=dimx+2*radius2;dimz=dimz+2*radius2;
				ix=ix+radius2;iz=iz+radius2;
				in_idx = ix*dimz+iz;//iz*dimx+ix;

				__syncthreads();

				s_data1[tz][tx]=txx1[in_idx];
				s_data2[tz][tx]=txz1[in_idx];
				s_data3[tz][tx]=tzz1[in_idx];

				if(threadIdx.y<radius2)
				{
						s_data1[threadIdx.y][tx]=txx1[in_idx-radius2];//g_input[in_idx-radius2*dimx];//up
						s_data1[threadIdx.y+BDIMY2+radius2][tx]=txx1[in_idx+BDIMY2];//g_input[in_idx+BDIMY2*dimx];//down

						s_data2[threadIdx.y][tx]=txz1[in_idx-radius2];//g_input[in_idx-radius2*dimx];//up
						s_data2[threadIdx.y+BDIMY2+radius2][tx]=txz1[in_idx+BDIMY2];//g_input[in_idx+BDIMY2*dimx];//down

						s_data3[threadIdx.y][tx]=tzz1[in_idx-radius2];//g_input[in_idx-radius2*dimx];//up
						s_data3[threadIdx.y+BDIMY2+radius2][tx]=tzz1[in_idx+BDIMY2];//g_input[in_idx+BDIMY2*dimx];//down
				}
				if(threadIdx.x<radius2)
				{
						s_data1[tz][threadIdx.x]=txx1[in_idx-radius2*dimz];//g_input[in_idx-radius2];//left
						s_data1[tz][threadIdx.x+BDIMX2+radius2]=txx1[in_idx+BDIMX2*dimz];//g_input[in_idx+BDIMX2];//right

						s_data2[tz][threadIdx.x]=txz1[in_idx-radius2*dimz];//g_input[in_idx-radius2];//left
						s_data2[tz][threadIdx.x+BDIMX2+radius2]=txz1[in_idx+BDIMX2*dimz];//g_input[in_idx+BDIMX2];//right

						s_data3[tz][threadIdx.x]=tzz1[in_idx-radius2*dimz];//g_input[in_idx-radius2];//left
						s_data3[tz][threadIdx.x+BDIMX2+radius2]=tzz1[in_idx+BDIMX2*dimz];//g_input[in_idx+BDIMX2];//right
				}

				s_velocity=velocity_d[in_idx]*velocity_d[in_idx];
				s_velocity1=velocity1_d[in_idx]*velocity1_d[in_idx];//////s_velocity1=velocity1_d[in_idx]*velocity_d[in_idx];     error//气人
//////////////注意伴随状态方程左边存在密度，所以用来反传计算伴随波场跟密度没有关系？？？？？？？？？？

				s_attenuation=attenuation_d[in_idx];
				__syncthreads();
/////data1:txx1///////data2:txz1///////data3:tzz1///////
////sumx:the derivation of x direction of txx
				float    sumx=coe_d[1]*(s_data1[tz][tx]-s_data1[tz][tx-1]);
					sumx+=coe_d[2]*(s_data1[tz][tx+1]-s_data1[tz][tx-2]);
					sumx+=coe_d[3]*(s_data1[tz][tx+2]-s_data1[tz][tx-3]);
					sumx+=coe_d[4]*(s_data1[tz][tx+3]-s_data1[tz][tx-4]);
					sumx+=coe_d[5]*(s_data1[tz][tx+4]-s_data1[tz][tx-5]);
					sumx+=coe_d[6]*(s_data1[tz][tx+5]-s_data1[tz][tx-6]);

////sumxz:the derivation of z direction of txz
				float    sumxz=coe_d[1]*(s_data2[tz][tx]-s_data2[tz-1][tx]);
					sumxz+=coe_d[2]*(s_data2[tz+1][tx]-s_data2[tz-2][tx]);
					sumxz+=coe_d[3]*(s_data2[tz+2][tx]-s_data2[tz-3][tx]);
					sumxz+=coe_d[4]*(s_data2[tz+3][tx]-s_data2[tz-4][tx]);
					sumxz+=coe_d[5]*(s_data2[tz+4][tx]-s_data2[tz-5][tx]);
					sumxz+=coe_d[6]*(s_data2[tz+5][tx]-s_data2[tz-6][tx]);
////sumx1:the derivation of x direction of tzz
				float    sumx1=coe_d[1]*(s_data3[tz][tx]-s_data3[tz][tx-1]);
					sumx1+=coe_d[2]*(s_data3[tz][tx+1]-s_data3[tz][tx-2]);
					sumx1+=coe_d[3]*(s_data3[tz][tx+2]-s_data3[tz][tx-3]);
					sumx1+=coe_d[4]*(s_data3[tz][tx+3]-s_data3[tz][tx-4]);
					sumx1+=coe_d[5]*(s_data3[tz][tx+4]-s_data3[tz][tx-5]);
					sumx1+=coe_d[6]*(s_data3[tz][tx+5]-s_data3[tz][tx-6]);

				vx2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*
				((1.0-s_attenuation*dt_real/2.0)*vx1[in_idx]+(s_velocity*sumx*coe_x+(s_velocity-2*s_velocity1)*sumx1*coe_x+s_velocity1*sumxz*coe_z));
		}
}

__global__ void adjoint_fwd_vx_illum(float *r_d_illum,float *vx2,float *vx1,float *txx1,float *txz1,float *tzz1,float *attenuation_d,float dx,float dz,float dt,float *coe_d,float coe_x,float coe_z,int dimx,int dimz,float *velocity_d,float *velocity1_d,float *density_d)
//adjoint_fwd_vx<<<dimGrid,dimBlock>>>(rvx2_d,rvx1_d,rtxx1_d,rtxz1_d,rtzz1_d,attenuation_d,dx,dz,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,s_velocity_d,s_velocity1_d,s_density_d);
{
		__shared__ float s_data1[BDIMY2+2*radius2][BDIMX2+2*radius2];
		__shared__ float s_data2[BDIMY2+2*radius2][BDIMX2+2*radius2];
		__shared__ float s_data3[BDIMY2+2*radius2][BDIMX2+2*radius2];
		float dt_real=dt/1000;
		float s_velocity;
		float s_velocity1;
		float s_attenuation;

		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		int tx = threadIdx.x+radius2;
		int tz = threadIdx.y+radius2;
		s_data1[tz][tx]=0.0;
		s_data1[threadIdx.y][threadIdx.x]=0.0;
		s_data1[BDIMY2+2*radius2-1-threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data1[threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data1[BDIMY2+2*radius2-1-threadIdx.y][threadIdx.x]=0.0;

		s_data2[tz][tx]=0.0;
		s_data2[threadIdx.y][threadIdx.x]=0.0;
		s_data2[BDIMY2+2*radius2-1-threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data2[threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data2[BDIMY2+2*radius2-1-threadIdx.y][threadIdx.x]=0.0;

		s_data3[tz][tx]=0.0;
		s_data3[threadIdx.y][threadIdx.x]=0.0;
		s_data3[BDIMY2+2*radius2-1-threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data3[threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data3[BDIMY2+2*radius2-1-threadIdx.y][threadIdx.x]=0.0;

		if((ix<dimx)&&(iz<dimz))
		{
				dimx=dimx+2*radius2;dimz=dimz+2*radius2;
				ix=ix+radius2;iz=iz+radius2;
				in_idx = ix*dimz+iz;//iz*dimx+ix;

				__syncthreads();

				s_data1[tz][tx]=txx1[in_idx];
				s_data2[tz][tx]=txz1[in_idx];
				s_data3[tz][tx]=tzz1[in_idx];

				if(threadIdx.y<radius2)
				{
						s_data1[threadIdx.y][tx]=txx1[in_idx-radius2];//g_input[in_idx-radius2*dimx];//up
						s_data1[threadIdx.y+BDIMY2+radius2][tx]=txx1[in_idx+BDIMY2];//g_input[in_idx+BDIMY2*dimx];//down

						s_data2[threadIdx.y][tx]=txz1[in_idx-radius2];//g_input[in_idx-radius2*dimx];//up
						s_data2[threadIdx.y+BDIMY2+radius2][tx]=txz1[in_idx+BDIMY2];//g_input[in_idx+BDIMY2*dimx];//down

						s_data3[threadIdx.y][tx]=tzz1[in_idx-radius2];//g_input[in_idx-radius2*dimx];//up
						s_data3[threadIdx.y+BDIMY2+radius2][tx]=tzz1[in_idx+BDIMY2];//g_input[in_idx+BDIMY2*dimx];//down
				}
				if(threadIdx.x<radius2)
				{
						s_data1[tz][threadIdx.x]=txx1[in_idx-radius2*dimz];//g_input[in_idx-radius2];//left
						s_data1[tz][threadIdx.x+BDIMX2+radius2]=txx1[in_idx+BDIMX2*dimz];//g_input[in_idx+BDIMX2];//right

						s_data2[tz][threadIdx.x]=txz1[in_idx-radius2*dimz];//g_input[in_idx-radius2];//left
						s_data2[tz][threadIdx.x+BDIMX2+radius2]=txz1[in_idx+BDIMX2*dimz];//g_input[in_idx+BDIMX2];//right

						s_data3[tz][threadIdx.x]=tzz1[in_idx-radius2*dimz];//g_input[in_idx-radius2];//left
						s_data3[tz][threadIdx.x+BDIMX2+radius2]=tzz1[in_idx+BDIMX2*dimz];//g_input[in_idx+BDIMX2];//right
				}

				s_velocity=velocity_d[in_idx]*velocity_d[in_idx];
				s_velocity1=velocity1_d[in_idx]*velocity1_d[in_idx];//////s_velocity1=velocity1_d[in_idx]*velocity_d[in_idx];     error//气人
//////////////注意伴随状态方程左边存在密度，所以用来反传计算伴随波场跟密度没有关系？？？？？？？？？？

				s_attenuation=attenuation_d[in_idx];
				__syncthreads();
/////data1:txx1///////data2:txz1///////data3:tzz1///////
////sumx:the derivation of x direction of txx
				float    sumx=coe_d[1]*(s_data1[tz][tx]-s_data1[tz][tx-1]);
					sumx+=coe_d[2]*(s_data1[tz][tx+1]-s_data1[tz][tx-2]);
					sumx+=coe_d[3]*(s_data1[tz][tx+2]-s_data1[tz][tx-3]);
					sumx+=coe_d[4]*(s_data1[tz][tx+3]-s_data1[tz][tx-4]);
					sumx+=coe_d[5]*(s_data1[tz][tx+4]-s_data1[tz][tx-5]);
					sumx+=coe_d[6]*(s_data1[tz][tx+5]-s_data1[tz][tx-6]);

////sumxz:the derivation of z direction of txz
				float    sumxz=coe_d[1]*(s_data2[tz][tx]-s_data2[tz-1][tx]);
					sumxz+=coe_d[2]*(s_data2[tz+1][tx]-s_data2[tz-2][tx]);
					sumxz+=coe_d[3]*(s_data2[tz+2][tx]-s_data2[tz-3][tx]);
					sumxz+=coe_d[4]*(s_data2[tz+3][tx]-s_data2[tz-4][tx]);
					sumxz+=coe_d[5]*(s_data2[tz+4][tx]-s_data2[tz-5][tx]);
					sumxz+=coe_d[6]*(s_data2[tz+5][tx]-s_data2[tz-6][tx]);
////sumx1:the derivation of x direction of tzz
				float    sumx1=coe_d[1]*(s_data3[tz][tx]-s_data3[tz][tx-1]);
					sumx1+=coe_d[2]*(s_data3[tz][tx+1]-s_data3[tz][tx-2]);
					sumx1+=coe_d[3]*(s_data3[tz][tx+2]-s_data3[tz][tx-3]);
					sumx1+=coe_d[4]*(s_data3[tz][tx+3]-s_data3[tz][tx-4]);
					sumx1+=coe_d[5]*(s_data3[tz][tx+4]-s_data3[tz][tx-5]);
					sumx1+=coe_d[6]*(s_data3[tz][tx+5]-s_data3[tz][tx-6]);

				vx2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*
				((1.0-s_attenuation*dt_real/2.0)*vx1[in_idx]+(s_velocity*sumx*coe_x+(s_velocity-2*s_velocity1)*sumx1*coe_x+s_velocity1*sumxz*coe_z));

				r_d_illum[in_idx]+=vx2[in_idx]*vx2[in_idx];
		}
}

__global__ void adjoint_fwd_vz(float *vz2,float *vz1,float *txx1,float *txz1,float *tzz1,float *attenuation_d,float dx,float dz,float dt,float *coe_d,float coe_x,float coe_z,int dimx,int dimz,float *velocity_d,float *velocity1_d,float *density_d)
//adjoint_fwd_vz<<<dimGrid,dimBlock>>>(rvz1_d,rvz2_d,rtxx1_d,rtxz1_d,rtzz1_d,attenuation_d,dx,dz,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,s_velocity_d,s_velocity1_d,s_density_d);
{
		__shared__ float s_data1[BDIMY2+2*radius2][BDIMX2+2*radius2];
		__shared__ float s_data2[BDIMY2+2*radius2][BDIMX2+2*radius2];
		__shared__ float s_data3[BDIMY2+2*radius2][BDIMX2+2*radius2];
		float dt_real=dt/1000;
		float s_velocity;
		float s_velocity1;
		float s_attenuation;

		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		int tx = threadIdx.x+radius2;
		int tz = threadIdx.y+radius2;
		s_data1[tz][tx]=0.0;
		s_data1[threadIdx.y][threadIdx.x]=0.0;
		s_data1[BDIMY2+2*radius2-1-threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data1[threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data1[BDIMY2+2*radius2-1-threadIdx.y][threadIdx.x]=0.0;

		s_data2[tz][tx]=0.0;
		s_data2[threadIdx.y][threadIdx.x]=0.0;
		s_data2[BDIMY2+2*radius2-1-threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data2[threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data2[BDIMY2+2*radius2-1-threadIdx.y][threadIdx.x]=0.0;

		s_data3[tz][tx]=0.0;
		s_data3[threadIdx.y][threadIdx.x]=0.0;
		s_data3[BDIMY2+2*radius2-1-threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data3[threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data3[BDIMY2+2*radius2-1-threadIdx.y][threadIdx.x]=0.0;

		if((ix<dimx)&&(iz<dimz))
		{
				dimx=dimx+2*radius2;dimz=dimz+2*radius2;
				ix=ix+radius2;iz=iz+radius2;
				in_idx = ix*dimz+iz;//iz*dimx+ix;

				__syncthreads();

				s_data1[tz][tx]=txx1[in_idx];
				s_data2[tz][tx]=txz1[in_idx];
				s_data3[tz][tx]=tzz1[in_idx];

				if(threadIdx.y<radius2)
				{
						s_data1[threadIdx.y][tx]=txx1[in_idx-radius2];//g_input[in_idx-radius2*dimx];//up
						s_data1[threadIdx.y+BDIMY2+radius2][tx]=txx1[in_idx+BDIMY2];//g_input[in_idx+BDIMY2*dimx];//down

						s_data2[threadIdx.y][tx]=txz1[in_idx-radius2];//g_input[in_idx-radius2*dimx];//up
						s_data2[threadIdx.y+BDIMY2+radius2][tx]=txz1[in_idx+BDIMY2];//g_input[in_idx+BDIMY2*dimx];//down

						s_data3[threadIdx.y][tx]=tzz1[in_idx-radius2];//g_input[in_idx-radius2*dimx];//up
						s_data3[threadIdx.y+BDIMY2+radius2][tx]=tzz1[in_idx+BDIMY2];//g_input[in_idx+BDIMY2*dimx];//down
				}
				if(threadIdx.x<radius2)
				{
						s_data1[tz][threadIdx.x]=txx1[in_idx-radius2*dimz];//g_input[in_idx-radius2];//left
						s_data1[tz][threadIdx.x+BDIMX2+radius2]=txx1[in_idx+BDIMX2*dimz];//g_input[in_idx+BDIMX2];//right

						s_data2[tz][threadIdx.x]=txz1[in_idx-radius2*dimz];//g_input[in_idx-radius2];//left
						s_data2[tz][threadIdx.x+BDIMX2+radius2]=txz1[in_idx+BDIMX2*dimz];//g_input[in_idx+BDIMX2];//right

						s_data3[tz][threadIdx.x]=tzz1[in_idx-radius2*dimz];//g_input[in_idx-radius2];//left
						s_data3[tz][threadIdx.x+BDIMX2+radius2]=tzz1[in_idx+BDIMX2*dimz];//g_input[in_idx+BDIMX2];//right
				}

				s_velocity=velocity_d[in_idx]*velocity_d[in_idx];
				s_velocity1=velocity1_d[in_idx]*velocity1_d[in_idx];//////s_velocity1=velocity1_d[in_idx]*velocity_d[in_idx];     error//气人

				s_attenuation=attenuation_d[in_idx];
				__syncthreads();
/////data1:txx1///////data2:txz1///////data3:tzz1///////
////sumz:the derivation of z direction of tzz   ///////data3:tzz1/////// 
				float    sumz=coe_d[1]*(s_data3[tz+1][tx]-s_data3[tz][tx]);
					sumz+=coe_d[2]*(s_data3[tz+2][tx]-s_data3[tz-1][tx]);
					sumz+=coe_d[3]*(s_data3[tz+3][tx]-s_data3[tz-2][tx]);
					sumz+=coe_d[4]*(s_data3[tz+4][tx]-s_data3[tz-3][tx]);
					sumz+=coe_d[5]*(s_data3[tz+5][tx]-s_data3[tz-4][tx]);
					sumz+=coe_d[6]*(s_data3[tz+6][tx]-s_data3[tz-5][tx]);

////sumz1:the derivation of z direction of txx/////data1:txx1////
				float    sumz1=coe_d[1]*(s_data1[tz+1][tx]-s_data1[tz][tx]);
					sumz1+=coe_d[2]*(s_data1[tz+2][tx]-s_data1[tz-1][tx]);
					sumz1+=coe_d[3]*(s_data1[tz+3][tx]-s_data1[tz-2][tx]);
					sumz1+=coe_d[4]*(s_data1[tz+4][tx]-s_data1[tz-3][tx]);
					sumz1+=coe_d[5]*(s_data1[tz+5][tx]-s_data1[tz-4][tx]);
					sumz1+=coe_d[6]*(s_data1[tz+6][tx]-s_data1[tz-5][tx]);

////sumx:the derivation of x direction of txz/////data2:txz1////					
				float    sumx=coe_d[1]*(s_data2[tz][tx+1]-s_data2[tz][tx]);
					sumx+=coe_d[2]*(s_data2[tz][tx+2]-s_data2[tz][tx-1]);
					sumx+=coe_d[3]*(s_data2[tz][tx+3]-s_data2[tz][tx-2]);
					sumx+=coe_d[4]*(s_data2[tz][tx+4]-s_data2[tz][tx-3]);
					sumx+=coe_d[5]*(s_data2[tz][tx+5]-s_data2[tz][tx-4]);
					sumx+=coe_d[6]*(s_data2[tz][tx+6]-s_data2[tz][tx-5]);

				vz2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*
				((1.0-s_attenuation*dt_real/2.0)*vz1[in_idx]+(s_velocity*sumz*coe_z+(s_velocity-2*s_velocity1)*sumz1*coe_z+s_velocity1*sumx*coe_x));
		}
}

__global__ void adjoint_fwd_vz_illum(float *r_d_illum,float *vz2,float *vz1,float *txx1,float *txz1,float *tzz1,float *attenuation_d,float dx,float dz,float dt,float *coe_d,float coe_x,float coe_z,int dimx,int dimz,float *velocity_d,float *velocity1_d,float *density_d)
//adjoint_fwd_vz<<<dimGrid,dimBlock>>>(rvz2_d,rvz1_d,rtxx2_d,rtxz2_d,rtzz2_d,attenuation_d,dx,dz,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,s_velocity_d,s_velocity1_d,s_density_d);
{
		__shared__ float s_data1[BDIMY2+2*radius2][BDIMX2+2*radius2];
		__shared__ float s_data2[BDIMY2+2*radius2][BDIMX2+2*radius2];
		__shared__ float s_data3[BDIMY2+2*radius2][BDIMX2+2*radius2];
		float dt_real=dt/1000;
		float s_velocity;
		float s_velocity1;
		float s_attenuation;

		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		int tx = threadIdx.x+radius2;
		int tz = threadIdx.y+radius2;
		s_data1[tz][tx]=0.0;
		s_data1[threadIdx.y][threadIdx.x]=0.0;
		s_data1[BDIMY2+2*radius2-1-threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data1[threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data1[BDIMY2+2*radius2-1-threadIdx.y][threadIdx.x]=0.0;

		s_data2[tz][tx]=0.0;
		s_data2[threadIdx.y][threadIdx.x]=0.0;
		s_data2[BDIMY2+2*radius2-1-threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data2[threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data2[BDIMY2+2*radius2-1-threadIdx.y][threadIdx.x]=0.0;

		s_data3[tz][tx]=0.0;
		s_data3[threadIdx.y][threadIdx.x]=0.0;
		s_data3[BDIMY2+2*radius2-1-threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data3[threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data3[BDIMY2+2*radius2-1-threadIdx.y][threadIdx.x]=0.0;

		if((ix<dimx)&&(iz<dimz))
		{
				dimx=dimx+2*radius2;dimz=dimz+2*radius2;
				ix=ix+radius2;iz=iz+radius2;
				in_idx = ix*dimz+iz;//iz*dimx+ix;

				__syncthreads();

				s_data1[tz][tx]=txx1[in_idx];
				s_data2[tz][tx]=txz1[in_idx];
				s_data3[tz][tx]=tzz1[in_idx];

				if(threadIdx.y<radius2)
				{
						s_data1[threadIdx.y][tx]=txx1[in_idx-radius2];//g_input[in_idx-radius2*dimx];//up
						s_data1[threadIdx.y+BDIMY2+radius2][tx]=txx1[in_idx+BDIMY2];//g_input[in_idx+BDIMY2*dimx];//down

						s_data2[threadIdx.y][tx]=txz1[in_idx-radius2];//g_input[in_idx-radius2*dimx];//up
						s_data2[threadIdx.y+BDIMY2+radius2][tx]=txz1[in_idx+BDIMY2];//g_input[in_idx+BDIMY2*dimx];//down

						s_data3[threadIdx.y][tx]=tzz1[in_idx-radius2];//g_input[in_idx-radius2*dimx];//up
						s_data3[threadIdx.y+BDIMY2+radius2][tx]=tzz1[in_idx+BDIMY2];//g_input[in_idx+BDIMY2*dimx];//down
				}
				if(threadIdx.x<radius2)
				{
						s_data1[tz][threadIdx.x]=txx1[in_idx-radius2*dimz];//g_input[in_idx-radius2];//left
						s_data1[tz][threadIdx.x+BDIMX2+radius2]=txx1[in_idx+BDIMX2*dimz];//g_input[in_idx+BDIMX2];//right

						s_data2[tz][threadIdx.x]=txz1[in_idx-radius2*dimz];//g_input[in_idx-radius2];//left
						s_data2[tz][threadIdx.x+BDIMX2+radius2]=txz1[in_idx+BDIMX2*dimz];//g_input[in_idx+BDIMX2];//right

						s_data3[tz][threadIdx.x]=tzz1[in_idx-radius2*dimz];//g_input[in_idx-radius2];//left
						s_data3[tz][threadIdx.x+BDIMX2+radius2]=tzz1[in_idx+BDIMX2*dimz];//g_input[in_idx+BDIMX2];//right
				}

				s_velocity=velocity_d[in_idx]*velocity_d[in_idx];
				s_velocity1=velocity1_d[in_idx]*velocity1_d[in_idx];//////s_velocity1=velocity1_d[in_idx]*velocity_d[in_idx];     error//气人

				s_attenuation=attenuation_d[in_idx];
				__syncthreads();
/////data1:txx1///////data2:txz1///////data3:tzz1///////
////sumz:the derivation of z direction of tzz   ///////data3:tzz1/////// 
				float    sumz=coe_d[1]*(s_data3[tz+1][tx]-s_data3[tz][tx]);
					sumz+=coe_d[2]*(s_data3[tz+2][tx]-s_data3[tz-1][tx]);
					sumz+=coe_d[3]*(s_data3[tz+3][tx]-s_data3[tz-2][tx]);
					sumz+=coe_d[4]*(s_data3[tz+4][tx]-s_data3[tz-3][tx]);
					sumz+=coe_d[5]*(s_data3[tz+5][tx]-s_data3[tz-4][tx]);
					sumz+=coe_d[6]*(s_data3[tz+6][tx]-s_data3[tz-5][tx]);

////sumz1:the derivation of z direction of txx/////data1:txx1////
				float    sumz1=coe_d[1]*(s_data1[tz+1][tx]-s_data1[tz][tx]);
					sumz1+=coe_d[2]*(s_data1[tz+2][tx]-s_data1[tz-1][tx]);
					sumz1+=coe_d[3]*(s_data1[tz+3][tx]-s_data1[tz-2][tx]);
					sumz1+=coe_d[4]*(s_data1[tz+4][tx]-s_data1[tz-3][tx]);
					sumz1+=coe_d[5]*(s_data1[tz+5][tx]-s_data1[tz-4][tx]);
					sumz1+=coe_d[6]*(s_data1[tz+6][tx]-s_data1[tz-5][tx]);

////sumx:the derivation of x direction of txz/////data2:txz1////					
				float    sumx=coe_d[1]*(s_data2[tz][tx+1]-s_data2[tz][tx]);
					sumx+=coe_d[2]*(s_data2[tz][tx+2]-s_data2[tz][tx-1]);
					sumx+=coe_d[3]*(s_data2[tz][tx+3]-s_data2[tz][tx-2]);
					sumx+=coe_d[4]*(s_data2[tz][tx+4]-s_data2[tz][tx-3]);
					sumx+=coe_d[5]*(s_data2[tz][tx+5]-s_data2[tz][tx-4]);
					sumx+=coe_d[6]*(s_data2[tz][tx+6]-s_data2[tz][tx-5]);

				vz2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*
				((1.0-s_attenuation*dt_real/2.0)*vz1[in_idx]+(s_velocity*sumz*coe_z+(s_velocity-2*s_velocity1)*sumz1*coe_z+s_velocity1*sumx*coe_x));

				r_d_illum[in_idx]+=vz2[in_idx]*vz2[in_idx];
		}
}


__global__ void adjoint_fwd_txxzzxz(float *txx2,float *txx1,float *tzz2,float *tzz1,float *txz2,float *txz1,float *vx2,float *vz2,float *velocity_d,float *velocity1_d,float *attenuation_d,float dt,float *coe_d,float coe_x,float coe_z,int dimx,int dimz,float *density_d)
//adjoint_fwd_txxzzxz<<<dimGrid,dimBlock>>>(rtxx2_d,rtxx1_d,rtzz2_d,rtzz1_d,rtxz2_d,rtxz1_d,rvx1_d,rvz1_d,s_velocity_d,s_velocity1_d,attenuation_d,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,s_density_d);
{
		__shared__ float s_data1[BDIMY2+2*radius2][BDIMX2+2*radius2];
		__shared__ float s_data2[BDIMY2+2*radius2][BDIMX2+2*radius2];
		float dt_real=dt/1000;
		//float s_velocity;
		//float s_velocity1;
		float s_attenuation;

		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		int tx = threadIdx.x+radius2;
		int tz = threadIdx.y+radius2;
		s_data1[tz][tx]=0.0;
		s_data1[threadIdx.y][threadIdx.x]=0.0;
		s_data1[BDIMY2+2*radius2-1-threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data1[threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data1[BDIMY2+2*radius2-1-threadIdx.y][threadIdx.x]=0.0;

		s_data2[tz][tx]=0.0;
		s_data2[threadIdx.y][threadIdx.x]=0.0;
		s_data2[BDIMY2+2*radius2-1-threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data2[threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data2[BDIMY2+2*radius2-1-threadIdx.y][threadIdx.x]=0.0;

		if((ix<dimx)&&(iz<dimz))
		{
				dimx=dimx+2*radius2;dimz=dimz+2*radius2;
				ix=ix+radius2;iz=iz+radius2;
				in_idx = ix*dimz+iz;//iz*dimx+ix;

				__syncthreads();

				s_data1[tz][tx]=vx2[in_idx];
				s_data2[tz][tx]=vz2[in_idx];

				if(threadIdx.y<radius2)
				{
						s_data1[threadIdx.y][tx]=vx2[in_idx-radius2];//g_input[in_idx-radius2*dimx];//up
						s_data1[threadIdx.y+BDIMY2+radius2][tx]=vx2[in_idx+BDIMY2];//g_input[in_idx+BDIMY2*dimx];//down

						s_data2[threadIdx.y][tx]=vz2[in_idx-radius2];//g_input[in_idx-radius2*dimx];//up
						s_data2[threadIdx.y+BDIMY2+radius2][tx]=vz2[in_idx+BDIMY2];//g_input[in_idx+BDIMY2*dimx];//down
				}
				if(threadIdx.x<radius2)
				{
						s_data1[tz][threadIdx.x]=vx2[in_idx-radius2*dimz];//g_input[in_idx-radius2];//left
						s_data1[tz][threadIdx.x+BDIMX2+radius2]=vx2[in_idx+BDIMX2*dimz];//g_input[in_idx+BDIMX2];//right

						s_data2[tz][threadIdx.x]=vz2[in_idx-radius2*dimz];//g_input[in_idx-radius2];//left
						s_data2[tz][threadIdx.x+BDIMX2+radius2]=vz2[in_idx+BDIMX2*dimz];//g_input[in_idx+BDIMX2];//right
				}

				//s_velocity=velocity_d[in_idx]*velocity_d[in_idx];
				//s_velocity1=velocity1_d[in_idx]*velocity1_d[in_idx];
				s_attenuation=attenuation_d[in_idx];
				__syncthreads();
		
/////////sumx:the derivation of x direction of vx
				float    sumx=coe_d[1]*(s_data1[tz][tx+1]-s_data1[tz][tx]);
					sumx+=coe_d[2]*(s_data1[tz][tx+2]-s_data1[tz][tx-1]);
					sumx+=coe_d[3]*(s_data1[tz][tx+3]-s_data1[tz][tx-2]);
					sumx+=coe_d[4]*(s_data1[tz][tx+4]-s_data1[tz][tx-3]);
					sumx+=coe_d[5]*(s_data1[tz][tx+5]-s_data1[tz][tx-4]);
					sumx+=coe_d[6]*(s_data1[tz][tx+6]-s_data1[tz][tx-5]);
/////////sumz:the derivation of z direction of vz
				float    sumz=coe_d[1]*(s_data2[tz][tx]-s_data2[tz-1][tx]);
					sumz+=coe_d[2]*(s_data2[tz+1][tx]-s_data2[tz-2][tx]);
					sumz+=coe_d[3]*(s_data2[tz+2][tx]-s_data2[tz-3][tx]);
					sumz+=coe_d[4]*(s_data2[tz+3][tx]-s_data2[tz-4][tx]);
					sumz+=coe_d[5]*(s_data2[tz+4][tx]-s_data2[tz-5][tx]);
					sumz+=coe_d[6]*(s_data2[tz+5][tx]-s_data2[tz-6][tx]);

				txx2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*txx1[in_idx]+sumx*coe_x);
				tzz2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*tzz1[in_idx]+sumz*coe_z);

/////////sumx1:the derivation of x direction of vz							
				float    sumx1=coe_d[1]*(s_data2[tz][tx]-s_data2[tz][tx-1]);
					sumx1+=coe_d[2]*(s_data2[tz][tx+1]-s_data2[tz][tx-2]);
					sumx1+=coe_d[3]*(s_data2[tz][tx+2]-s_data2[tz][tx-3]);
					sumx1+=coe_d[4]*(s_data2[tz][tx+3]-s_data2[tz][tx-4]);
					sumx1+=coe_d[5]*(s_data2[tz][tx+4]-s_data2[tz][tx-5]);
					sumx1+=coe_d[6]*(s_data2[tz][tx+5]-s_data2[tz][tx-6]);
/////////sumz1:the derivation of z direction of vx
				float    sumz1=coe_d[1]*(s_data1[tz+1][tx]-s_data1[tz][tx]);
					sumz1+=coe_d[2]*(s_data1[tz+2][tx]-s_data1[tz-1][tx]);
					sumz1+=coe_d[3]*(s_data1[tz+3][tx]-s_data1[tz-2][tx]);
					sumz1+=coe_d[4]*(s_data1[tz+4][tx]-s_data1[tz-3][tx]);
					sumz1+=coe_d[5]*(s_data1[tz+5][tx]-s_data1[tz-4][tx]);
					sumz1+=coe_d[6]*(s_data1[tz+6][tx]-s_data1[tz-5][tx]);

				txz2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*
						((1.0-s_attenuation*dt_real/2.0)*txz1[in_idx]+sumx1*coe_x+sumz1*coe_z);
		}
}

__global__ void adjoint_fwd_txxzzxzpp(float *tp2,float *tp1,float *txx2,float *txx1,float *tzz2,float *tzz1,float *txz2,float *txz1,float *vx2,float *vz2,float *velocity_d,float *velocity1_d,float *attenuation_d,float dt,float *coe_d,float coe_x,float coe_z,int dimx,int dimz,float *density_d)
//adjoint_fwd_txxzzxzpp<<<dimGrid,dimBlock>>>(rtp1_d,rtp2_d,rtxx1_d,rtxx2_d,rtzz1_d,rtzz2_d,rtxz1_d,rtxz2_d,rvx2_d,rvz2_d,s_velocity_d,s_velocity1_d,attenuation_d,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,s_density_d);
{
		__shared__ float s_data1[BDIMY2+2*radius2][BDIMX2+2*radius2];
		__shared__ float s_data2[BDIMY2+2*radius2][BDIMX2+2*radius2];
		float dt_real=dt/1000;
		float s_velocity;
		//float s_velocity1;
		float s_attenuation;

		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		int tx = threadIdx.x+radius2;
		int tz = threadIdx.y+radius2;
		s_data1[tz][tx]=0.0;
		s_data1[threadIdx.y][threadIdx.x]=0.0;
		s_data1[BDIMY2+2*radius2-1-threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data1[threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data1[BDIMY2+2*radius2-1-threadIdx.y][threadIdx.x]=0.0;

		s_data2[tz][tx]=0.0;
		s_data2[threadIdx.y][threadIdx.x]=0.0;
		s_data2[BDIMY2+2*radius2-1-threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data2[threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data2[BDIMY2+2*radius2-1-threadIdx.y][threadIdx.x]=0.0;

		if((ix<dimx)&&(iz<dimz))
		{
				dimx=dimx+2*radius2;dimz=dimz+2*radius2;
				ix=ix+radius2;iz=iz+radius2;
				in_idx = ix*dimz+iz;//iz*dimx+ix;

				__syncthreads();

				s_data1[tz][tx]=vx2[in_idx];
				s_data2[tz][tx]=vz2[in_idx];

				if(threadIdx.y<radius2)
				{
						s_data1[threadIdx.y][tx]=vx2[in_idx-radius2];//g_input[in_idx-radius2*dimx];//up
						s_data1[threadIdx.y+BDIMY2+radius2][tx]=vx2[in_idx+BDIMY2];//g_input[in_idx+BDIMY2*dimx];//down

						s_data2[threadIdx.y][tx]=vz2[in_idx-radius2];//g_input[in_idx-radius2*dimx];//up
						s_data2[threadIdx.y+BDIMY2+radius2][tx]=vz2[in_idx+BDIMY2];//g_input[in_idx+BDIMY2*dimx];//down
				}
				if(threadIdx.x<radius2)
				{
						s_data1[tz][threadIdx.x]=vx2[in_idx-radius2*dimz];//g_input[in_idx-radius2];//left
						s_data1[tz][threadIdx.x+BDIMX2+radius2]=vx2[in_idx+BDIMX2*dimz];//g_input[in_idx+BDIMX2];//right

						s_data2[tz][threadIdx.x]=vz2[in_idx-radius2*dimz];//g_input[in_idx-radius2];//left
						s_data2[tz][threadIdx.x+BDIMX2+radius2]=vz2[in_idx+BDIMX2*dimz];//g_input[in_idx+BDIMX2];//right
				}

				s_velocity=velocity_d[in_idx]*velocity_d[in_idx];
				//s_velocity1=velocity1_d[in_idx]*velocity1_d[in_idx];
				s_attenuation=attenuation_d[in_idx];
				__syncthreads();
		
/////////sumx:the derivation of x direction of vx
				float    sumx=coe_d[1]*(s_data1[tz][tx+1]-s_data1[tz][tx]);
					sumx+=coe_d[2]*(s_data1[tz][tx+2]-s_data1[tz][tx-1]);
					sumx+=coe_d[3]*(s_data1[tz][tx+3]-s_data1[tz][tx-2]);
					sumx+=coe_d[4]*(s_data1[tz][tx+4]-s_data1[tz][tx-3]);
					sumx+=coe_d[5]*(s_data1[tz][tx+5]-s_data1[tz][tx-4]);
					sumx+=coe_d[6]*(s_data1[tz][tx+6]-s_data1[tz][tx-5]);
/////////sumz:the derivation of z direction of vz
				float    sumz=coe_d[1]*(s_data2[tz][tx]-s_data2[tz-1][tx]);
					sumz+=coe_d[2]*(s_data2[tz+1][tx]-s_data2[tz-2][tx]);
					sumz+=coe_d[3]*(s_data2[tz+2][tx]-s_data2[tz-3][tx]);
					sumz+=coe_d[4]*(s_data2[tz+3][tx]-s_data2[tz-4][tx]);
					sumz+=coe_d[5]*(s_data2[tz+4][tx]-s_data2[tz-5][tx]);
					sumz+=coe_d[6]*(s_data2[tz+5][tx]-s_data2[tz-6][tx]);

				txx2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*txx1[in_idx]+sumx*coe_x);
				tzz2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*tzz1[in_idx]+sumz*coe_z);

				//tp2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*tp1[in_idx]+sumx*coe_x+sumz*coe_z);	
				tp2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*tp1[in_idx]+s_velocity*density_d[in_idx]*sumx*coe_x+s_velocity*density_d[in_idx]*sumz*coe_z);

/////////sumx1:the derivation of x direction of vz							
				float    sumx1=coe_d[1]*(s_data2[tz][tx]-s_data2[tz][tx-1]);
					sumx1+=coe_d[2]*(s_data2[tz][tx+1]-s_data2[tz][tx-2]);
					sumx1+=coe_d[3]*(s_data2[tz][tx+2]-s_data2[tz][tx-3]);
					sumx1+=coe_d[4]*(s_data2[tz][tx+3]-s_data2[tz][tx-4]);
					sumx1+=coe_d[5]*(s_data2[tz][tx+4]-s_data2[tz][tx-5]);
					sumx1+=coe_d[6]*(s_data2[tz][tx+5]-s_data2[tz][tx-6]);
/////////sumz1:the derivation of z direction of vx
				float    sumz1=coe_d[1]*(s_data1[tz+1][tx]-s_data1[tz][tx]);
					sumz1+=coe_d[2]*(s_data1[tz+2][tx]-s_data1[tz-1][tx]);
					sumz1+=coe_d[3]*(s_data1[tz+3][tx]-s_data1[tz-2][tx]);
					sumz1+=coe_d[4]*(s_data1[tz+4][tx]-s_data1[tz-3][tx]);
					sumz1+=coe_d[5]*(s_data1[tz+5][tx]-s_data1[tz-4][tx]);
					sumz1+=coe_d[6]*(s_data1[tz+6][tx]-s_data1[tz-5][tx]);

				txz2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*
						((1.0-s_attenuation*dt_real/2.0)*txz1[in_idx]+sumx1*coe_x+sumz1*coe_z);
		}
}

//////2016年11月03日 星期四 19时12分09秒 add new gradient_x and_z for lame1 lame2    一阶速度应力方程的伴随状态方程
__global__ void adjoint_fwd_vx_new(float *vx2,float *vx1,float *txx1,float *txz1,float *tzz1,float *attenuation_d,float dx,float dz,float dt,float *coe_d,float coe_x,float coe_z,int dimx,int dimz,float *velocity_d,float *velocity1_d,float *density_d)
//adjoint_fwd_vx<<<dimGrid,dimBlock>>>(rvx2_d,rvx1_d,rtxx1_d,rtxz1_d,rtzz1_d,attenuation_d,dx,dz,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,s_velocity_d,s_velocity1_d,s_density_d);
{
		__shared__ float s_data1[BDIMY2+2*radius2][BDIMX2+2*radius2];
		__shared__ float s_data2[BDIMY2+2*radius2][BDIMX2+2*radius2];
		__shared__ float s_data3[BDIMY2+2*radius2][BDIMX2+2*radius2];
		float dt_real=dt/1000;
		//float s_velocity;
		//float s_velocity1;
		float s_density;
		float s_attenuation;

		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;
		int up,down;
		int left,right;

		int tx = threadIdx.x+radius2;
		int tz = threadIdx.y+radius2;
		s_data1[tz][tx]=0.0;
		s_data1[threadIdx.y][threadIdx.x]=0.0;
		s_data1[BDIMY2+2*radius2-1-threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data1[threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data1[BDIMY2+2*radius2-1-threadIdx.y][threadIdx.x]=0.0;

		s_data2[tz][tx]=0.0;
		s_data2[threadIdx.y][threadIdx.x]=0.0;
		s_data2[BDIMY2+2*radius2-1-threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data2[threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data2[BDIMY2+2*radius2-1-threadIdx.y][threadIdx.x]=0.0;

		s_data3[tz][tx]=0.0;
		s_data3[threadIdx.y][threadIdx.x]=0.0;
		s_data3[BDIMY2+2*radius2-1-threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data3[threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data3[BDIMY2+2*radius2-1-threadIdx.y][threadIdx.x]=0.0;

		if((ix<dimx)&&(iz<dimz))
		{
				dimx=dimx+2*radius2;dimz=dimz+2*radius2;
				ix=ix+radius2;iz=iz+radius2;
				in_idx = ix*dimz+iz;//iz*dimx+ix;

				__syncthreads();
//////////////////////////////////////////////////////////lame1:(velocity_d[in_idx]*velocity_d[in_idx]-2*velocity1_d[in_idx]*velocity1_d[in_idx])/density_d[in_idx]
//////////////////////////////////////////////////////////lame2:velocity1_d[in_idx]*velocity1_d[in_idx]/density_d[in_idx]
//////////////////////////////////////////////////////////lame1+2*lame2:velocity_d[in_idx]*velocity_d[in_idx]/density_d[in_idx]
				s_data1[tz][tx]=velocity_d[in_idx]*velocity_d[in_idx]/density_d[in_idx]*txx1[in_idx];
				s_data2[tz][tx]=velocity1_d[in_idx]*velocity1_d[in_idx]/density_d[in_idx]*txz1[in_idx];
				s_data3[tz][tx]=(velocity_d[in_idx]*velocity_d[in_idx]-2*velocity1_d[in_idx]*velocity1_d[in_idx])/density_d[in_idx]*tzz1[in_idx];

				if(threadIdx.y<radius2)
				{
						up=in_idx-radius2;
						down=in_idx+BDIMY2;
						s_data1[threadIdx.y][tx]=velocity_d[up]*velocity_d[up]/density_d[up]*txx1[up];//g_input[in_idx-radius2*dimx];//up
						s_data1[threadIdx.y+BDIMY2+radius2][tx]=velocity_d[down]*velocity_d[down]/density_d[down]*txx1[down];//g_input[in_idx+BDIMY2*dimx];//down
						s_data2[threadIdx.y][tx]=velocity1_d[up]*velocity1_d[up]/density_d[up]*txz1[up];//g_input[in_idx-radius2*dimx];//up
						s_data2[threadIdx.y+BDIMY2+radius2][tx]=velocity1_d[down]*velocity1_d[down]/density_d[down]*txz1[down];//g_input[in_idx+BDIMY2*dimx];//down
						s_data3[threadIdx.y][tx]=(velocity_d[up]*velocity_d[up]-2*velocity1_d[up]*velocity1_d[up])/density_d[up]*tzz1[up];//g_input[in_idx-radius2*dimx];//up
						s_data3[threadIdx.y+BDIMY2+radius2][tx]=(velocity_d[down]*velocity_d[down]-2*velocity1_d[down]*velocity1_d[down])/density_d[down]*tzz1[down];//g_input[in_idx+BDIMY2*dimx];//down
				}
				if(threadIdx.x<radius2)
				{
						left=in_idx-radius2*dimz;
						right=in_idx+BDIMX2*dimz;
						s_data1[tz][threadIdx.x]=velocity_d[left]*velocity_d[left]/density_d[left]*txx1[left];//g_input[in_idx-radius2];//left
						s_data1[tz][threadIdx.x+BDIMX2+radius2]=velocity_d[right]*velocity_d[right]/density_d[right]*txx1[right];//g_input[in_idx+BDIMX2];//right

						s_data2[tz][threadIdx.x]=velocity1_d[left]*velocity1_d[left]/density_d[left]*txz1[left];//g_input[in_idx-radius2];//left
						s_data2[tz][threadIdx.x+BDIMX2+radius2]=velocity1_d[right]*velocity1_d[right]/density_d[right]*txz1[right];//g_input[in_idx+BDIMX2];//right

						s_data3[tz][threadIdx.x]=(velocity_d[left]*velocity_d[left]-2*velocity1_d[left]*velocity1_d[left])/density_d[left]*tzz1[left];//g_input[in_idx-radius2];//left
						s_data3[tz][threadIdx.x+BDIMX2+radius2]=(velocity_d[right]*velocity_d[right]-2*velocity1_d[right]*velocity1_d[right])/density_d[right]*tzz1[right];//g_input[in_idx+BDIMX2];//right
				}

				//s_velocity=velocity_d[in_idx]*velocity_d[in_idx];
				//s_velocity1=velocity1_d[in_idx]*velocity1_d[in_idx];//////s_velocity1=velocity1_d[in_idx]*velocity_d[in_idx];     error//气人
				s_density=density_d[in_idx];
//////////////注意伴随状态方程左边存在密度，所以用来反传计算伴随波场跟密度没有关系？？？？？？？？？？

				s_attenuation=attenuation_d[in_idx];
				__syncthreads();
/////data1:txx1///////data2:txz1///////data3:tzz1///////
////sumx:the derivation of x direction of txx
				float    sumx=coe_d[1]*(s_data1[tz][tx]-s_data1[tz][tx-1]);
					sumx+=coe_d[2]*(s_data1[tz][tx+1]-s_data1[tz][tx-2]);
					sumx+=coe_d[3]*(s_data1[tz][tx+2]-s_data1[tz][tx-3]);
					sumx+=coe_d[4]*(s_data1[tz][tx+3]-s_data1[tz][tx-4]);
					sumx+=coe_d[5]*(s_data1[tz][tx+4]-s_data1[tz][tx-5]);
					sumx+=coe_d[6]*(s_data1[tz][tx+5]-s_data1[tz][tx-6]);

////sumxz:the derivation of z direction of txz
				float    sumxz=coe_d[1]*(s_data2[tz][tx]-s_data2[tz-1][tx]);
					sumxz+=coe_d[2]*(s_data2[tz+1][tx]-s_data2[tz-2][tx]);
					sumxz+=coe_d[3]*(s_data2[tz+2][tx]-s_data2[tz-3][tx]);
					sumxz+=coe_d[4]*(s_data2[tz+3][tx]-s_data2[tz-4][tx]);
					sumxz+=coe_d[5]*(s_data2[tz+4][tx]-s_data2[tz-5][tx]);
					sumxz+=coe_d[6]*(s_data2[tz+5][tx]-s_data2[tz-6][tx]);
////sumx1:the derivation of x direction of tzz
				float    sumx1=coe_d[1]*(s_data3[tz][tx]-s_data3[tz][tx-1]);
					sumx1+=coe_d[2]*(s_data3[tz][tx+1]-s_data3[tz][tx-2]);
					sumx1+=coe_d[3]*(s_data3[tz][tx+2]-s_data3[tz][tx-3]);
					sumx1+=coe_d[4]*(s_data3[tz][tx+3]-s_data3[tz][tx-4]);
					sumx1+=coe_d[5]*(s_data3[tz][tx+4]-s_data3[tz][tx-5]);
					sumx1+=coe_d[6]*(s_data3[tz][tx+5]-s_data3[tz][tx-6]);

				vx2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*
				((1.0-s_attenuation*dt_real/2.0)*vx1[in_idx]+(s_density*sumx*coe_x+s_density*sumx1*coe_x+s_density*sumxz*coe_z));
		}
}

__global__ void adjoint_fwd_vz_new(float *vz2,float *vz1,float *txx1,float *txz1,float *tzz1,float *attenuation_d,float dx,float dz,float dt,float *coe_d,float coe_x,float coe_z,int dimx,int dimz,float *velocity_d,float *velocity1_d,float *density_d)
//vx2_d,vx1_d,txx1_d,txz1_d,attenuation_d,dx,dz,dt,coe_opt_d,coe_x,coe_z,nx_append_radius2,nz_append_radius2
{
		__shared__ float s_data1[BDIMY2+2*radius2][BDIMX2+2*radius2];
		__shared__ float s_data2[BDIMY2+2*radius2][BDIMX2+2*radius2];
		__shared__ float s_data3[BDIMY2+2*radius2][BDIMX2+2*radius2];
		float dt_real=dt/1000;
		//float s_velocity;
		//float s_velocity1;
		float s_density;
		float s_attenuation;

		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;
		int up,down;
		int left,right;

		int tx = threadIdx.x+radius2;
		int tz = threadIdx.y+radius2;
		s_data1[tz][tx]=0.0;
		s_data1[threadIdx.y][threadIdx.x]=0.0;
		s_data1[BDIMY2+2*radius2-1-threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data1[threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data1[BDIMY2+2*radius2-1-threadIdx.y][threadIdx.x]=0.0;

		s_data2[tz][tx]=0.0;
		s_data2[threadIdx.y][threadIdx.x]=0.0;
		s_data2[BDIMY2+2*radius2-1-threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data2[threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data2[BDIMY2+2*radius2-1-threadIdx.y][threadIdx.x]=0.0;

		s_data3[tz][tx]=0.0;
		s_data3[threadIdx.y][threadIdx.x]=0.0;
		s_data3[BDIMY2+2*radius2-1-threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data3[threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data3[BDIMY2+2*radius2-1-threadIdx.y][threadIdx.x]=0.0;

		if((ix<dimx)&&(iz<dimz))
		{
				dimx=dimx+2*radius2;dimz=dimz+2*radius2;
				ix=ix+radius2;iz=iz+radius2;
				in_idx = ix*dimz+iz;//iz*dimx+ix;

				__syncthreads();
//////////////////////////////////////////////////////////lame1:(velocity_d[in_idx]*velocity_d[in_idx]-2*velocity1_d[in_idx]*velocity1_d[in_idx])/density_d[in_idx]
//////////////////////////////////////////////////////////lame2:velocity1_d[in_idx]*velocity1_d[in_idx]/density_d[in_idx]
//////////////////////////////////////////////////////////lame1+2*lame2:velocity_d[in_idx]*velocity_d[in_idx]/density_d[in_idx]
				s_data1[tz][tx]=(velocity_d[in_idx]*velocity_d[in_idx]-2*velocity1_d[in_idx]*velocity1_d[in_idx])/density_d[in_idx]*txx1[in_idx];
				s_data2[tz][tx]=velocity1_d[in_idx]*velocity1_d[in_idx]/density_d[in_idx]*txz1[in_idx];
				s_data3[tz][tx]=velocity_d[in_idx]*velocity_d[in_idx]/density_d[in_idx]*tzz1[in_idx];

				if(threadIdx.y<radius2)
				{
						up=in_idx-radius2;
						down=in_idx+BDIMY2;
						s_data1[threadIdx.y][tx]=(velocity_d[up]*velocity_d[up]-2*velocity1_d[up]*velocity1_d[up])/density_d[up]*txx1[up];//g_input[in_idx-radius2*dimx];//up
						s_data1[threadIdx.y+BDIMY2+radius2][tx]=(velocity_d[down]*velocity_d[down]-2*velocity1_d[down]*velocity1_d[down])/density_d[down]*txx1[down];//g_input[in_idx+BDIMY2*dimx];//down

						s_data2[threadIdx.y][tx]=velocity1_d[up]*velocity1_d[up]/density_d[up]*txz1[up];//g_input[in_idx-radius2*dimx];//up
						s_data2[threadIdx.y+BDIMY2+radius2][tx]=velocity1_d[down]*velocity1_d[down]/density_d[down]*txz1[down];//g_input[in_idx+BDIMY2*dimx];//down

						s_data3[threadIdx.y][tx]=velocity_d[up]*velocity_d[up]/density_d[up]*tzz1[up];//g_input[in_idx-radius2*dimx];//up
						s_data3[threadIdx.y+BDIMY2+radius2][tx]=velocity_d[down]*velocity_d[down]/density_d[down]*tzz1[down];//g_input[in_idx+BDIMY2*dimx];//down
				}
				if(threadIdx.x<radius2)
				{
						left=in_idx-radius2*dimz;
						right=in_idx+BDIMX2*dimz;
						s_data1[tz][threadIdx.x]=(velocity_d[left]*velocity_d[left]-2*velocity1_d[left]*velocity1_d[left])/density_d[left]*txx1[left];//g_input[in_idx-radius2];//left
						s_data1[tz][threadIdx.x+BDIMX2+radius2]=(velocity_d[right]*velocity_d[right]-2*velocity1_d[right]*velocity1_d[right])/density_d[right]*txx1[right];//g_input[in_idx+BDIMX2];//right

						s_data2[tz][threadIdx.x]=velocity1_d[left]*velocity1_d[left]/density_d[left]*txz1[left];//g_input[in_idx-radius2];//left
						s_data2[tz][threadIdx.x+BDIMX2+radius2]=velocity1_d[right]*velocity1_d[right]/density_d[right]*txz1[right];//g_input[in_idx+BDIMX2];//right

						s_data3[tz][threadIdx.x]=velocity_d[left]*velocity_d[left]/density_d[left]*tzz1[left];//g_input[in_idx-radius2];//left
						s_data3[tz][threadIdx.x+BDIMX2+radius2]=velocity_d[right]*velocity_d[right]/density_d[right]*tzz1[right];//g_input[in_idx+BDIMX2];//right
				}

				//s_velocity=velocity_d[in_idx]*velocity_d[in_idx];
				//s_velocity1=velocity1_d[in_idx]*velocity1_d[in_idx];//////s_velocity1=velocity1_d[in_idx]*velocity_d[in_idx];     error//气人
				s_density=density_d[in_idx];

				s_attenuation=attenuation_d[in_idx];
				__syncthreads();
/////data1:txx1///////data2:txz1///////data3:tzz1///////
////sumz:the derivation of z direction of tzz   ///////data3:tzz1/////// 
				float    sumz=coe_d[1]*(s_data3[tz+1][tx]-s_data3[tz][tx]);
					sumz+=coe_d[2]*(s_data3[tz+2][tx]-s_data3[tz-1][tx]);
					sumz+=coe_d[3]*(s_data3[tz+3][tx]-s_data3[tz-2][tx]);
					sumz+=coe_d[4]*(s_data3[tz+4][tx]-s_data3[tz-3][tx]);
					sumz+=coe_d[5]*(s_data3[tz+5][tx]-s_data3[tz-4][tx]);
					sumz+=coe_d[6]*(s_data3[tz+6][tx]-s_data3[tz-5][tx]);

////sumz1:the derivation of z direction of txx/////data1:txx1////
				float    sumz1=coe_d[1]*(s_data1[tz+1][tx]-s_data1[tz][tx]);
					sumz1+=coe_d[2]*(s_data1[tz+2][tx]-s_data1[tz-1][tx]);
					sumz1+=coe_d[3]*(s_data1[tz+3][tx]-s_data1[tz-2][tx]);
					sumz1+=coe_d[4]*(s_data1[tz+4][tx]-s_data1[tz-3][tx]);
					sumz1+=coe_d[5]*(s_data1[tz+5][tx]-s_data1[tz-4][tx]);
					sumz1+=coe_d[6]*(s_data1[tz+6][tx]-s_data1[tz-5][tx]);

////sumx:the derivation of x direction of txz/////data2:txz1////					
				float    sumx=coe_d[1]*(s_data2[tz][tx+1]-s_data2[tz][tx]);
					sumx+=coe_d[2]*(s_data2[tz][tx+2]-s_data2[tz][tx-1]);
					sumx+=coe_d[3]*(s_data2[tz][tx+3]-s_data2[tz][tx-2]);
					sumx+=coe_d[4]*(s_data2[tz][tx+4]-s_data2[tz][tx-3]);
					sumx+=coe_d[5]*(s_data2[tz][tx+5]-s_data2[tz][tx-4]);
					sumx+=coe_d[6]*(s_data2[tz][tx+6]-s_data2[tz][tx-5]);

				vz2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*
				((1.0-s_attenuation*dt_real/2.0)*vz1[in_idx]+(s_density*sumz*coe_z+s_density*sumz1*coe_z+s_density*sumx*coe_x));
		}
}

//cal_derivation_x<<<dimGrid,dimBlock>>>(vx1_d,vx_x_d,coe_opt_d,dx,dz,nx_append_radius,nz_append_radius,mark);
__global__ void cal_derivation_x(float *vx1_d,float *vx_x_d,float *coe_d,float dx,float dz,int dimx,int dimz,int mark)
{
		__shared__ float s_data1[BDIMY2+2*radius2][BDIMX2+2*radius2];

		float sum=0;
		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		int tx = threadIdx.x+radius2;
		int tz = threadIdx.y+radius2;

		s_data1[tz][tx]=0.0;
		s_data1[threadIdx.y][threadIdx.x]=0.0;
		s_data1[BDIMY2+2*radius2-1-threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data1[threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data1[BDIMY2+2*radius2-1-threadIdx.y][threadIdx.x]=0.0;

		if((ix<dimx)&&(iz<dimz))
		{
				dimx=dimx+2*radius2;dimz=dimz+2*radius2;
				ix=ix+radius2;iz=iz+radius2;
				in_idx = ix*dimz+iz;//iz*dimx+ix;

				__syncthreads();

				s_data1[tz][tx]=vx1_d[in_idx];


				if(threadIdx.y<radius2)
				{
						s_data1[threadIdx.y][tx]=vx1_d[in_idx-radius2];//g_input[in_idx-radius2*dimx];//up
						s_data1[threadIdx.y+BDIMY2+radius2][tx]=vx1_d[in_idx+BDIMY2];//g_input[in_idx+BDIMY2*dimx];//down
				}
				if(threadIdx.x<radius2)
				{
						s_data1[tz][threadIdx.x]=vx1_d[in_idx-radius2*dimz];//g_input[in_idx-radius2];//left
						s_data1[tz][threadIdx.x+BDIMX2+radius2]=vx1_d[in_idx+BDIMX2*dimz];//g_input[in_idx+BDIMX2];//right
				}
				__syncthreads();
				
	if(mark==0)
	{
		       sum=coe_d[1]*(s_data1[tz][tx+1]-s_data1[tz][tx]);		
		       sum+=coe_d[2]*(s_data1[tz][tx+2]-s_data1[tz][tx-1]);
		       sum+=coe_d[3]*(s_data1[tz][tx+3]-s_data1[tz][tx-2]);
		       sum+=coe_d[4]*(s_data1[tz][tx+4]-s_data1[tz][tx-3]);
		       sum+=coe_d[5]*(s_data1[tz][tx+5]-s_data1[tz][tx-4]);
		       sum+=coe_d[6]*(s_data1[tz][tx+6]-s_data1[tz][tx-5]);
	}

	
	if(mark==1)	
	{		     
		       sum=coe_d[1]*(s_data1[tz][tx]-s_data1[tz][tx-1]);		
		       sum+=coe_d[2]*(s_data1[tz][tx+1]-s_data1[tz][tx-2]);
		       sum+=coe_d[3]*(s_data1[tz][tx+2]-s_data1[tz][tx-3]);
		       sum+=coe_d[4]*(s_data1[tz][tx+3]-s_data1[tz][tx-4]);
		       sum+=coe_d[5]*(s_data1[tz][tx+4]-s_data1[tz][tx-5]);
		       sum+=coe_d[6]*(s_data1[tz][tx+5]-s_data1[tz][tx-6]);
	}
		      
			vx_x_d[in_idx]=(1.0/dx)*sum;
		}		
}

__global__ void cal_derivation_z(float *vz1_d,float *vz_z_d,float *coe_d,float dx,float dz,int dimx,int dimz,int mark)
{
		__shared__ float s_data2[BDIMY2+2*radius2][BDIMX2+2*radius2];

		float sum1=0;
		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;
		
		int tx = threadIdx.x+radius2;
		int tz = threadIdx.y+radius2;

		s_data2[tz][tx]=0.0;
		s_data2[threadIdx.y][threadIdx.x]=0.0;
		s_data2[BDIMY2+2*radius2-1-threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data2[threadIdx.y][BDIMX2+2*radius2-1-threadIdx.x]=0.0;
		s_data2[BDIMY2+2*radius2-1-threadIdx.y][threadIdx.x]=0.0;

		if((ix<dimx)&&(iz<dimz))
		{
				dimx=dimx+2*radius2;dimz=dimz+2*radius2;
				ix=ix+radius2;iz=iz+radius2;
				in_idx = ix*dimz+iz;//iz*dimx+ix;

				__syncthreads();

				s_data2[tz][tx]=vz1_d[in_idx];

				if(threadIdx.y<radius2)
				{
						s_data2[threadIdx.y][tx]=vz1_d[in_idx-radius2];//g_input[in_idx-radius2*dimx];//up
						s_data2[threadIdx.y+BDIMY2+radius2][tx]=vz1_d[in_idx+BDIMY2];//g_input[in_idx+BDIMY2*dimx];//down
				}
				if(threadIdx.x<radius2)
				{
						s_data2[tz][threadIdx.x]=vz1_d[in_idx-radius2*dimz];//g_input[in_idx-radius2];//left
						s_data2[tz][threadIdx.x+BDIMX2+radius2]=vz1_d[in_idx+BDIMX2*dimz];//g_input[in_idx+BDIMX2];//right
				}
		
				__syncthreads();
	
	if(mark==0)
	{
		      sum1=coe_d[1]*(s_data2[tz][tx]-s_data2[tz-1][tx]);
		      sum1+=coe_d[2]*(s_data2[tz+1][tx]-s_data2[tz-2][tx]);
		      sum1+=coe_d[3]*(s_data2[tz+2][tx]-s_data2[tz-3][tx]);
		      sum1+=coe_d[4]*(s_data2[tz+3][tx]-s_data2[tz-4][tx]);
		      sum1+=coe_d[5]*(s_data2[tz+4][tx]-s_data2[tz-5][tx]);
		      sum1+=coe_d[6]*(s_data2[tz+5][tx]-s_data2[tz-6][tx]);
		     
	}
	if(mark==1)
	{
		      sum1=coe_d[1]*(s_data2[tz+1][tx]-s_data2[tz][tx]);
		      sum1+=coe_d[2]*(s_data2[tz+2][tx]-s_data2[tz-1][tx]);
		      sum1+=coe_d[3]*(s_data2[tz+3][tx]-s_data2[tz-2][tx]);
		      sum1+=coe_d[4]*(s_data2[tz+4][tx]-s_data2[tz-3][tx]);
		      sum1+=coe_d[5]*(s_data2[tz+5][tx]-s_data2[tz-4][tx]);
		      sum1+=coe_d[6]*(s_data2[tz+6][tx]-s_data2[tz-5][tx]);
	}	      
			vz_z_d[in_idx]=(1.0/dz)*sum1;
		}		
}


__global__ void cuda_cal_objective(float *obj, float *err, int ng)
/*< calculate the value of objective function: obj >*/
{
  	__shared__ float  sdata[Block_Size];
    	int tid=threadIdx.x;
    	sdata[tid]=0.0f;
	for(int s=0; s<(ng+Block_Size-1)/Block_Size; s++)
	{
		int id=s*blockDim.x+threadIdx.x;
		float a=(id<ng)?err[id]:0.0f;
		sdata[tid] += a*a;	
	} 
    	__syncthreads();

    	/* do reduction in shared mem */
    	for(int s=blockDim.x/2; s>32; s>>=1) 
    	{
		if (threadIdx.x < s) sdata[tid] += sdata[tid + s]; __syncthreads();
    	}
   	if (tid < 32)
   	{
		if (blockDim.x >=  64) { sdata[tid] += sdata[tid + 32]; }
		if (blockDim.x >=  32) { sdata[tid] += sdata[tid + 16]; }
		if (blockDim.x >=  16) { sdata[tid] += sdata[tid +  8]; }
		if (blockDim.x >=   8) { sdata[tid] += sdata[tid +  4]; }
		if (blockDim.x >=   4) { sdata[tid] += sdata[tid +  2]; }
		if (blockDim.x >=   2) { sdata[tid] += sdata[tid +  1]; }
    	}
     
    	if (tid == 0) { *obj=sdata[0]; }
}


__global__ void cuda_cal_correlation_objective(float *obj, float *obj_parameter_d)
/*< calculate the value of objective function: obj >*/
{
	*obj=float(-1.0*obj_parameter_d[2]/sqrt(obj_parameter_d[0])/sqrt(obj_parameter_d[1]));
}


__global__ void cuda_adj_shot(float *adj_shot_x_d,float *tmp_shot_x_d,float *obs_shot_x_d,int receiver_num,int lt,float *correlation_parameter_d)
{
	
	int ix = blockIdx.x*blockDim.x+threadIdx.x;
	int iz = blockIdx.y*blockDim.y+threadIdx.y;
	int in_idx;

	float a,b,c;

	a=correlation_parameter_d[0];///tmp*tmp	
	b=correlation_parameter_d[1];///obs*obs
	c=correlation_parameter_d[2];///tmp*obs

	if((ix<receiver_num)&&(iz<lt))
	{
		in_idx=ix*lt+iz;
			
		adj_shot_x_d[in_idx]=(1.0/sqrt(a*b))*(1.0*c/a*tmp_shot_x_d[in_idx]-obs_shot_x_d[in_idx]);
	}

}

__global__ void cal_gradient_in_elastic_media(float *grad_lame1_d,float *grad_lame2_d,float *grad_den_d,float *vx_t_d,float *vz_t_d,float *vx_x_d,float *vz_z_d,float *vx_z_d,float *vz_x_d,float *rvx1_d,float *rvz1_d,float *rtxx1_d,float *rtxz1_d,float *rtzz1_d,int boundary_left,int boundary_up,int nx,int nz,int dimx,int dimz,float *s_velocity_d,float *s_velocity1_d,float *s_density_d)
//cal_gradient_in_elastic_media<<<dimGrid,dimBlock>>>(grad_lame11_d,grad_lame22_d,grad_den1_d,vx_t_d,vz_t_d,vx_x_d,vz_z_d,vx_z_d,vz_x_d,rvx2_d,rvz2_d,rtxx2_d,rtxz2_d,rtzz2_d,boundary_left,boundary_up,nx_size,nz,nx_append,nz_append,s_velocity_d,s_velocity1_d,s_density_d);
{
		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;
		int in_idx1;

		float lame1;
		float lame2;

		if((ix<nx)&&(iz<nz))
		{
			in_idx=ix*nz+iz;//iz*nz+ix;
			in_idx1=(ix+boundary_left)*dimz+iz+boundary_up;//iz*nz+ix;

			lame1=s_density_d[in_idx1]*s_velocity_d[in_idx1]*s_velocity_d[in_idx1]-2.0*s_density_d[in_idx1]*s_velocity1_d[in_idx1]*s_velocity1_d[in_idx1];
			lame2=s_density_d[in_idx1]*s_velocity1_d[in_idx1]*s_velocity1_d[in_idx1];

			grad_den_d[in_idx]=grad_den_d[in_idx]+s_density_d[in_idx1]*(rvx1_d[in_idx1]*vx_t_d[in_idx1]+rvz1_d[in_idx1]*vz_t_d[in_idx1]);

			grad_lame1_d[in_idx]=grad_lame1_d[in_idx]+lame1*(rtxx1_d[in_idx1]+rtzz1_d[in_idx1])*(vx_x_d[in_idx1]+vz_z_d[in_idx1])*(-1.0);

			grad_lame2_d[in_idx]=grad_lame2_d[in_idx]+lame2*(2.0*rtxx1_d[in_idx1]*vx_x_d[in_idx1]+2.0*rtzz1_d[in_idx1]*vz_z_d[in_idx1]+rtxz1_d[in_idx1]*(vx_z_d[in_idx1]+vz_x_d[in_idx1]))*(-1.0);
		}
}

__global__ void cal_gradient_in_elastic_media_new(float *grad_lame1_d,float *grad_lame2_d,float *grad_den_d,float *vx_t_d,float *vz_t_d,float *vx_x_d,float *vz_z_d,float *vx_z_d,float *vz_x_d,float *rvx1_d,float *rvz1_d,float *rtxx1_d,float *rtxz1_d,float *rtzz1_d,int boundary_left,int boundary_up,int nx,int nz,int dimx,int dimz,float *s_velocity_d,float *s_velocity1_d,float *s_density_d)
//cal_gradient_in_elastic_media_new<<<dimGrid,dimBlock>>>(grad_lame11_d,grad_lame22_d,grad_den1_d,vx_t_d,vz_t_d,vx_x_d,vz_z_d,vx_z_d,vz_x_d,rvx2_d,rvz2_d,rtxx2_d,rtxz2_d,rtzz2_d,boundary_left,boundary_up,nx_size,nz,nx_append,nz_append,s_velocity_d,s_velocity1_d,s_density_d);
{
		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;
		int in_idx1;

		if((ix<nx)&&(iz<nz))
		{
			in_idx=ix*nz+iz;//iz*nz+ix;
			in_idx1=(ix+boundary_left)*dimz+iz+boundary_up;//iz*nz+ix;

			grad_den_d[in_idx]=grad_den_d[in_idx]+(rvx1_d[in_idx1]*vx_t_d[in_idx1]+rvz1_d[in_idx1]*vz_t_d[in_idx1]);

			grad_lame1_d[in_idx]=grad_lame1_d[in_idx]+(rtxx1_d[in_idx1]+rtzz1_d[in_idx1])*(vx_x_d[in_idx1]+vz_z_d[in_idx1])*(-1.0);

			grad_lame2_d[in_idx]=grad_lame2_d[in_idx]+(2.0*rtxx1_d[in_idx1]*vx_x_d[in_idx1]+2.0*rtzz1_d[in_idx1]*vz_z_d[in_idx1]+rtxz1_d[in_idx1]*(vx_z_d[in_idx1]+vz_x_d[in_idx1]))*(-1.0);
		}
}

__global__ void cal_gradient_for_den_old(float *grad_density_d,float *vx1_d,float *vx2_d,float *vz1_d,float *vz2_d,float *rvx1_d,float *rvz1_d,float dt,int boundary_left,int boundary_up,int nx,int nz,int dimx,int dimz)
{
		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;
		int in_idx1;
		float dt_real;
		dt_real=dt/1000;


		if((ix<nx)&&(iz<nz))
		{
			in_idx=ix*nz+iz;//iz*nz+ix;
			in_idx1=(ix+boundary_left)*dimz+iz+boundary_up;//iz*nz+ix;
			
			grad_density_d[in_idx]+=(rvx1_d[in_idx1]*(vx2_d[in_idx1]-vx1_d[in_idx1])/dt_real+rvz1_d[in_idx1]*(vz2_d[in_idx1]-vz1_d[in_idx1])/dt_real)*(-1.0);
		}
}

///////////////////////the following  cal_gradient_for_den cal_gradient_for_lame1 cal_gradient_for_lame2 - + +////+ - -
__global__ void cal_gradient_for_den(float *grad_den_d,float *vx_t_d,float *vz_t_d,float *rvx1_d,float *rvz1_d,float dt,int boundary_left,int boundary_up,int nx,int nz,int dimx,int dimz)
//cal_gradient_for_den<<<dimGrid,dimBlock>>>(grad_den_d,vx_t_d,vz_t_d,rvx1_d,rvz1_d,dt,boundary_left,boundary_up,nx,nz,nx_append,nz_append);
{
		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;
		int in_idx1;
		//float dt_real;
		//dt_real=dt/1000;

		if((ix<nx)&&(iz<nz))
		{
			in_idx=ix*nz+iz;//iz*nz+ix;
			in_idx1=(ix+boundary_left)*dimz+iz+boundary_up;//iz*nz+ix;
			
			grad_den_d[in_idx]=grad_den_d[in_idx]+(rvx1_d[in_idx1]*vx_t_d[in_idx1]+rvz1_d[in_idx1]*vz_t_d[in_idx1]);
			
			//grad_den_d[in_idx]=grad_den_d[in_idx]+(rvx1_d[in_idx1]*vx_t_d[in_idx1]+rvz1_d[in_idx1]*vz_t_d[in_idx1])*(-1.0);
		}
}

__global__ void cal_gradient_for_lame1(float *grad_lame1_d,float *rtxx1_d,float *rtzz1_d,float *vx_x_d,float *vz_z_d,int boundary_left,int boundary_up,int nx,int nz,int dimx,int dimz)
//cal_gradient_for_lame1<<<dimGrid,dimBlock>>>(grad_lame1_d,rtxx1_d,rtzz1_d,vx_x_d,vz_z_d,boundary_left,boundary_up,nx,nz,nx_append,nz_append);
{
		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;
		int in_idx1;

		if((ix<nx)&&(iz<nz))
		{
			in_idx=ix*nz+iz;//iz*nz+ix;
			in_idx1=(ix+boundary_left)*dimz+iz+boundary_up;//iz*nz+ix;

			grad_lame1_d[in_idx]=grad_lame1_d[in_idx]+(rtxx1_d[in_idx1]+rtzz1_d[in_idx1])*(vx_x_d[in_idx1]+vz_z_d[in_idx1])*(-1.0);

			//grad_lame1_d[in_idx]=grad_lame1_d[in_idx]+(rtxx1_d[in_idx1]+rtzz1_d[in_idx1])*(vx_x_d[in_idx1]+vz_z_d[in_idx1]);
		}
}

__global__ void cal_gradient_for_lame2(float *grad_lame2_d,float *rtxx1_d,float *rtxz1_d,float *rtzz1_d,float *vx_x_d,float *vz_z_d,float *vx_z_d,float *vz_x_d,int boundary_left,int boundary_up,int nx,int nz,int dimx,int dimz)
//cal_gradient_for_lame2<<<dimGrid,dimBlock>>>(grad_lame2_d,rtxx1_d,rtxz1_d,rtzz1_d,vx_x_d,vz_z_d,vx_z_d,vz_x_d,boundary_left,boundary_up,nx,nz,nx_append,nz_append);
{
		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;
		int in_idx1;

		if((ix<nx)&&(iz<nz))
		{
			in_idx=ix*nz+iz;//iz*nz+ix;
			in_idx1=(ix+boundary_left)*dimz+iz+boundary_up;//iz*nz+ix;	

			grad_lame2_d[in_idx]=grad_lame2_d[in_idx]+(2*rtxx1_d[in_idx1]*vx_x_d[in_idx1]+2*rtzz1_d[in_idx1]*vz_z_d[in_idx1]+rtxz1_d[in_idx1]*(vx_z_d[in_idx1]+vz_x_d[in_idx1]))*(-1.0);
			//grad_lame2_d[in_idx]=grad_lame2_d[in_idx]+(2*rtxx1_d[in_idx1]*vx_x_d[in_idx1]+2*rtzz1_d[in_idx1]*vz_z_d[in_idx1]+rtxz1_d[in_idx1]*(vx_z_d[in_idx1]+vz_x_d[in_idx1]));
		}
}

/////
///////////////////////the following  cal_gradient_for_d_mul cal_gradient_for_lam_mul cal_gradient_for_lam_mul - + +////+ - -
__global__ void cal_gradient_for_den_mul(float *grad_den_d,float *vx_t_d,float *vz_t_d,float *rvx1_d,float *rvz1_d,float dt,int boundary_left,int boundary_up,int nx,int nz,int dimx,int dimz,float *s_velocity_d,float *s_velocity1_d,float *s_density_d)
//cal_gradient_for_den_mul<<<dimGrid,dimBlock>>>(grad_den1_d,vx_t_d,vz_t_d,rvx2_d,rvz2_d,dt,boundary_left,boundary_up,nx,nz,nx_append,nz_append,s_velocity_d,s_velocity1_d,s_density_d);
{
		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;
		int in_idx1;
		//float dt_real;
		//dt_real=dt/1000;

		if((ix<nx)&&(iz<nz))
		{
			in_idx=ix*nz+iz;//iz*nz+ix;
			in_idx1=(ix+boundary_left)*dimz+iz+boundary_up;//iz*nz+ix;
			
			grad_den_d[in_idx]=grad_den_d[in_idx]+s_density_d[in_idx1]*(rvx1_d[in_idx1]*vx_t_d[in_idx1]+rvz1_d[in_idx1]*vz_t_d[in_idx1]);
			
			//grad_den_d[in_idx]=grad_den_d[in_idx]+s_density_d[in_idx1]*(rvx1_d[in_idx1]*vx_t_d[in_idx1]+rvz1_d[in_idx1]*vz_t_d[in_idx1])*(-1.0);
		}
}

__global__ void cal_gradient_for_lame1_mul(float *grad_lame1_d,float *rtxx1_d,float *rtzz1_d,float *vx_x_d,float *vz_z_d,int boundary_left,int boundary_up,int nx,int nz,int dimx,int dimz,float *s_velocity_d,float *s_velocity1_d,float *s_density_d)
//cal_gradient_for_lame1_mul<<<dimGrid,dimBlock>>>(grad_lame11_d,rtxx2_d,rtzz2_d,vx_x_d,vz_z_d,boundary_left,boundary_up,nx_size,nz,nx_append,nz_append,s_velocity_d,s_velocity1_d,s_density_d);
{
		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;
		int in_idx1;
		float lame1;

		if((ix<nx)&&(iz<nz))
		{
			in_idx=ix*nz+iz;//iz*nz+ix;
			in_idx1=(ix+boundary_left)*dimz+iz+boundary_up;//iz*nz+ix;

			lame1=s_density_d[in_idx1]*s_velocity_d[in_idx1]*s_velocity_d[in_idx1]-2.0*s_density_d[in_idx1]*s_velocity1_d[in_idx1]*s_velocity1_d[in_idx1];

			grad_lame1_d[in_idx]=grad_lame1_d[in_idx]+lame1*(rtxx1_d[in_idx1]+rtzz1_d[in_idx1])*(vx_x_d[in_idx1]+vz_z_d[in_idx1])*(-1.0);

			//grad_lame1_d[in_idx]=grad_lame1_d[in_idx]+lame1*(rtxx1_d[in_idx1]+rtzz1_d[in_idx1])*(vx_x_d[in_idx1]+vz_z_d[in_idx1]);
		}
}

__global__ void cal_gradient_for_lame2_mul(float *grad_lame2_d,float *rtxx1_d,float *rtxz1_d,float *rtzz1_d,float *vx_x_d,float *vz_z_d,float *vx_z_d,float *vz_x_d,int boundary_left,int boundary_up,int nx,int nz,int dimx,int dimz,float *s_velocity_d,float *s_velocity1_d,float *s_density_d)
//cal_gradient_for_lame2_mul<<<dimGrid,dimBlock>>>(grad_lame22_d,rtxx2_d,rtxz2_d,rtzz2_d,vx_x_d,vz_z_d,vx_z_d,vz_x_d,boundary_left,boundary_up,nx_size,nz,nx_append,nz_append,s_velocity_d,s_velocity1_d,s_density_d);
{
		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;
		int in_idx1;

		float lame2;

		if((ix<nx)&&(iz<nz))
		{
			in_idx=ix*nz+iz;//iz*nz+ix;
			in_idx1=(ix+boundary_left)*dimz+iz+boundary_up;//iz*nz+ix;
			
			lame2=s_density_d[in_idx1]*s_velocity1_d[in_idx1]*s_velocity1_d[in_idx1];	

			grad_lame2_d[in_idx]=grad_lame2_d[in_idx]+lame2*(2.0*rtxx1_d[in_idx1]*vx_x_d[in_idx1]+2.0*rtzz1_d[in_idx1]*vz_z_d[in_idx1]+rtxz1_d[in_idx1]*(vx_z_d[in_idx1]+vz_x_d[in_idx1]))*(-1.0);
			//grad_lame2_d[in_idx]=grad_lame2_d[in_idx]+lame2*(2*rtxx1_d[in_idx1]*vx_x_d[in_idx1]+2*rtzz1_d[in_idx1]*vz_z_d[in_idx1]+rtxz1_d[in_idx1]*(vx_z_d[in_idx1]+vz_x_d[in_idx1]));
		}
}

///////////////////////the following  cal_gradient_for_den cal_gradient_for_lame1 cal_gradient_for_lame2 - + +////+ - -
__global__ void cal_gradient_for_den_new(float *grad_den_d,float *vx_t_d,float *vz_t_d,float *rvx1_d,float *rvz1_d,float dt,int boundary_left,int boundary_up,int nx,int nz,int dimx,int dimz)
//cal_gradient_for_den<<<dimGrid,dimBlock>>>(grad_den_d,vx_t_d,vz_t_d,rvx1_d,rvz1_d,dt,boundary_left,boundary_up,nx,nz,nx_append,nz_append);
{
		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;
		int in_idx1;
		//float dt_real;
		//dt_real=dt/1000;

		if((ix<nx)&&(iz<nz))
		{
			in_idx=ix*nz+iz;//iz*nz+ix;
			in_idx1=(ix+boundary_left)*dimz+iz+boundary_up;//iz*nz+ix;
			
			grad_den_d[in_idx]=grad_den_d[in_idx]+(rvx1_d[in_idx1]*vx_t_d[in_idx1]+rvz1_d[in_idx1]*vz_t_d[in_idx1]);
			
			//grad_den_d[in_idx]=grad_den_d[in_idx]+(rvx1_d[in_idx1]*vx_t_d[in_idx1]+rvz1_d[in_idx1]*vz_t_d[in_idx1])*(-1.0);
		}
}

__global__ void cal_gradient_for_lame1_new(float *grad_lame1_d,float *rtxx1_d,float *rtzz1_d,float *txx1_d,float *tzz1_d,float *velocity_d,float *velocity1_d,float *s_density_d,int boundary_left,int boundary_up,int nx,int nz,int dimx,int dimz)
//cal_gradient_for_lame1_new<<<dimGrid,dimBlock>>>(grad_lame11_d,rtxx2_d,rtzz2_d,txx1_d,tzz1_d,s_velocity_d,s_velocity1_d,s_density_d,boundary_left,boundary_up,nx,nz,nx_append,nz_append);
{
		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;
		int id;
		float lame1,lame2;

		if((ix<nx)&&(iz<nz))
		{
			in_idx=(ix+boundary_left)*dimz+iz+boundary_up;//iz*nz+ix;
			id=ix*nz+iz;
			
			lame1=s_density_d[in_idx]*velocity_d[in_idx]*velocity_d[in_idx]-2*s_density_d[in_idx]*velocity1_d[in_idx]*velocity1_d[in_idx];
			lame2=s_density_d[in_idx]*velocity1_d[in_idx]*velocity1_d[in_idx];
			
			grad_lame1_d[id]=grad_lame1_d[id]+(txx1_d[in_idx]+tzz1_d[in_idx])*(rtxx1_d[in_idx]+rtzz1_d[in_idx])/(4*(lame1+lame2)*(lame1+lame2))*(-1.0);
		}
}

__global__ void cal_gradient_for_lame2_new(float *grad_lame2_d,float *rtxx1_d,float *rtxz1_d,float *rtzz1_d,float *txx1_d,float *tzz1_d,float *txz1_d,float *velocity_d,float *velocity1_d,float *s_density_d,int boundary_left,int boundary_up,int nx,int nz,int dimx,int dimz)
//cal_gradient_for_lame2_new<<<dimGrid,dimBlock>>>(grad_lame22_d,rtxx1_d,rtxz1_d,rtzz2_d,txx1_d,tzz1_d,txz1_d,s_velocity_d,s_velocity1_d,s_density_d,boundary_left,boundary_up,nx,nz,nx_append,nz_append);
{
		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;
		int id;
		float lame1,lame2;

		if((ix<nx)&&(iz<nz))
		{
			in_idx=(ix+boundary_left)*dimz+iz+boundary_up;//iz*nz+ix;
			id=ix*nz+iz;
			
			lame1=s_density_d[in_idx]*velocity_d[in_idx]*velocity_d[in_idx]-2*s_density_d[in_idx]*velocity1_d[in_idx]*velocity1_d[in_idx];
			lame2=s_density_d[in_idx]*velocity1_d[in_idx]*velocity1_d[in_idx];
			
			grad_lame2_d[id]=grad_lame2_d[id]+(1.0*txz1_d[in_idx]*rtxz1_d[in_idx]/(lame2*lame2)+(txx1_d[in_idx]+tzz1_d[in_idx])*(rtxx1_d[in_idx]+rtzz1_d[in_idx])/(4.0*(lame1+lame2)*(lame1+lame2))+(txx1_d[in_idx]-tzz1_d[in_idx])*(rtxx1_d[in_idx]-rtzz1_d[in_idx])/(4.0*lame2*lame2))*(-1.0);			
		}
}

__global__ void cal_gradient_for_vp(float *grad_vp_d,float *grad_lame1_d,float *grad_lame2_d,float *grad_den_d,float *s_velocity_d,float *s_velocity1_d,float *s_density_d,int nx,int nz,int dimx,int dimz,int boundary_left,int boundary_up)
//vp<<<dimGrid,dimBlock>>>(grad_vp1_d,grad_lame1_d,grad_lame2_d,grad_den_d,s_velocity_d,s_velocity1_d,nx,nz,nx_append,nz_append,boundary_left,boundary_up);
{
		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;
		int in_idx1;

		if((ix<nx)&&(iz<nz))
		{
			in_idx=ix*nz+iz;//iz*nz+ix;
			in_idx1=(ix+boundary_left)*dimz+iz+boundary_up;//iz*nz+ix;
			
			 grad_vp_d[in_idx]=2*s_density_d[in_idx1]*s_velocity_d[in_idx1]*grad_lame1_d[in_idx];
		}
}

__global__ void cal_gradient_for_vs(float *grad_vs_d,float *grad_lame1_d,float *grad_lame2_d,float *grad_den_d,float *s_velocity_d,float *s_velocity1_d,float *s_density_d,int nx,int nz,int dimx,int dimz,int boundary_left,int boundary_up)
//cal_gradient_for_vs<<<dimGrid,dimBlock>>>(grad_vs1_d,grad_lame1_d,grad_lame2_d,grad_den_d,s_velocity_d,s_velocity1_d,s_density_d,nx,nz,nx_append,nz_append,boundary_left,boundary_up);
{
		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;
		int in_idx1;

		if((ix<nx)&&(iz<nz))
		{
			in_idx=ix*nz+iz;//iz*nz+ix;
			in_idx1=(ix+boundary_left)*dimz+iz+boundary_up;//iz*nz+ix;
		
			grad_vs_d[in_idx]=(-4*s_density_d[in_idx1]*s_velocity1_d[in_idx1]*grad_lame1_d[in_idx]+2*s_density_d[in_idx1]*s_velocity1_d[in_idx1]*grad_lame2_d[in_idx]);
		}
}

__global__ void cal_gradient_for_density(float *grad_density_d,float *grad_lame1_d,float *grad_lame2_d,float *grad_den_d,float *s_velocity_d,float *s_velocity1_d,float *s_density_d,int nx,int nz,int dimx,int dimz,int boundary_left,int boundary_up)
//cal_gradient_for_density<<<dimGrid,dimBlock>>>(grad_density1_d,grad_lame11_d,grad_lame22_d,grad_den1_d,s_velocity_d,s_velocity1_d,s_density_d,nx,nz,nx_append,nz_append,boundary_left,boundary_up);
{
		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;
		int in_idx1;

		if((ix<nx)&&(iz<nz))
		{
			in_idx=ix*nz+iz;//iz*nz+ix;
			in_idx1=(ix+boundary_left)*dimz+iz+boundary_up;//iz*nz+ix;
		
			grad_density_d[in_idx]=((s_velocity_d[in_idx1]*s_velocity_d[in_idx1]-2*s_velocity1_d[in_idx1]*s_velocity1_d[in_idx1])*grad_lame1_d[in_idx]+s_velocity1_d[in_idx1]*s_velocity1_d[in_idx1]*grad_lame2_d[in_idx]+grad_den_d[in_idx]);
		}
}

__global__ void invert_lame_to_vp(float *grad_vp1_d,float *grad_lame11_d,float *grad_lame22_d,float *s_velocity_d,float *s_velocity1_d,float *s_density_d,int nx,int nz,int nx_append,int nz_append,int boundary_left,int boundary_up)
//invert_lame_to_vp<<<dimGrid,dimBlock>>>(grad_vp1_d,grad_lame11_d,grad_lame22_d,s_velocity_d,s_velocity1_d,s_density_d,nx,nz,nx_append,nz_append,boundary_left,boundary_up);
{
	int ix = blockIdx.x*blockDim.x+threadIdx.x;
	int iz = blockIdx.y*blockDim.y+threadIdx.y;  	
	int id,id1;

	if (ix<nx && iz<nz)
	{
		id=ix*nz+iz;
		id1=(ix+boundary_left)*nz_append+iz+boundary_up;
		
		grad_vp1_d[id]=2*s_density_d[id1]*s_velocity_d[id1]*grad_lame11_d[id];
	}

}

__global__ void invert_lame_to_vs(float *grad_vs1_d,float *grad_lame11_d,float *grad_lame22_d,float *s_velocity_d,float *s_velocity1_d,float *s_density_d,int nx,int nz,int nx_append,int nz_append,int boundary_left,int boundary_up)
//invert_lame_to_vs<<<dimGrid,dimBlock>>>(grad_vs1_d,grad_lame11_d,grad_lame22_d,s_velocity_d,s_velocity1_d,s_density_d,nx,nz,nx_append,nz_append,boundary_left,boundary_up);
{
	int ix = blockIdx.x*blockDim.x+threadIdx.x;
	int iz = blockIdx.y*blockDim.y+threadIdx.y;  	
	int id,id1;

	if (ix<nx && iz<nz)
	{
		id=ix*nz+iz;
		id1=(ix+boundary_left)*nz_append+iz+boundary_up;
		
		grad_vs1_d[id]=-4*s_density_d[id1]*s_velocity1_d[id1]*grad_lame11_d[id]+2*s_density_d[id1]*s_velocity1_d[id1]*grad_lame22_d[id];
	}

}

__global__ void invert_lame_to_density(float *grad_density1_d,float *grad_lame11_d,float *grad_lame22_d,float *grad_den1_d,float *s_velocity_d,float *s_velocity1_d,float *s_density_d,int nx,int nz,int nx_append,int nz_append,int boundary_left,int boundary_up)
//invert_lame_to_density<<<dimGrid,dimBlock>>>(grad_density1_d,grad_lame11_d,grad_lame22_d,grad_den1_d,s_velocity_d,s_velocity1_d,s_density_d,nx,nz,nx_append,nz_append,boundary_left,boundary_up);
{
	int ix = blockIdx.x*blockDim.x+threadIdx.x;
	int iz = blockIdx.y*blockDim.y+threadIdx.y;  	
	int id,id1;

	if (ix<nx && iz<nz)
	{
		id=ix*nz+iz;
		id1=(ix+boundary_left)*nz_append+iz+boundary_up;
		
		grad_density1_d[id]=((s_velocity_d[id1]*s_velocity_d[id1]-2*s_velocity1_d[id1]*s_velocity1_d[id1])*grad_lame11_d[id]+s_velocity1_d[id1]*s_velocity1_d[id1]*grad_lame22_d[id]+grad_den1_d[id]);
	}

}

__global__ void invert_lame_to_velocity_para_new(float *grad_vp1_d,float *grad_vs1_d,float *grad_density1_d,float *grad_lame11_d,float *grad_lame22_d,float *grad_den1_d,float *s_velocity_d,float *s_velocity1_d,float *s_density_d,int nx,int nz,int nx_append,int nz_append,int boundary_left,int boundary_up)
//invert_lame_to_velocity_para<<<dimGrid_new,dimBlock>>>(all_grad_vp1_d,all_grad_vs1_d,all_grad_density1_d,all_grad_lame11_d,all_grad_lame22_d,all_grad_den1_d,s_velocity_all_d,s_velocity1_all_d,s_density_all_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up);
{
	int ix = blockIdx.x*blockDim.x+threadIdx.x;
	int iz = blockIdx.y*blockDim.y+threadIdx.y;  	
	int id,id1;

	if (ix<nx && iz<nz)
	{
		id=ix*nz+iz;
		id1=(ix+boundary_left)*nz_append+iz+boundary_up;


		grad_vp1_d[id]=2*s_velocity_d[id1]*s_density_d[id1]*grad_lame11_d[id];

		grad_vs1_d[id]=-4.0*s_velocity1_d[id1]*s_density_d[id1]*grad_lame11_d[id]+2.0*s_velocity1_d[id1]*s_density_d[id1]*grad_lame22_d[id];
		
		grad_density1_d[id]=(-2.0*s_velocity1_d[id1]*s_velocity1_d[id1]+s_velocity_d[id1]*s_velocity_d[id1])*grad_lame11_d[id]+s_velocity1_d[id1]*s_velocity1_d[id1]*grad_lame22_d[id]+grad_den1_d[id];
	}
}

__global__ void invert_lame_to_velocity_para(float *grad_vp1_d,float *grad_vs1_d,float *grad_density1_d,float *grad_lame11_d,float *grad_lame22_d,float *grad_den1_d,float *s_velocity_d,float *s_velocity1_d,float *s_density_d,int nx,int nz,int nx_append,int nz_append,int boundary_left,int boundary_up)
//invert_lame_to_velocity_para<<<dimGrid_new,dimBlock>>>(all_grad_vp1_d,all_grad_vs1_d,all_grad_density1_d,all_grad_lame11_d,all_grad_lame22_d,all_grad_den1_d,s_velocity_all_d,s_velocity1_all_d,s_density_all_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up);
{
	int ix = blockIdx.x*blockDim.x+threadIdx.x;
	int iz = blockIdx.y*blockDim.y+threadIdx.y;  	
	int id,id1;

	float lame1,lame2;

	if(ix<nx&&iz<nz)
	{
		id=ix*nz+iz;
		id1=(ix+boundary_left)*nz_append+iz+boundary_up;

		lame1=s_density_d[id1]*s_velocity_d[id1]*s_velocity_d[id1]-2*s_density_d[id1]*s_velocity1_d[id1]*s_velocity1_d[id1];
		lame2=s_density_d[id1]*s_velocity1_d[id1]*s_velocity1_d[id1];

		//grad_vp1_d[id]=2*(lame1+2*lame2)*1.0/lame1*grad_lame11_d[id];

		//grad_vs1_d[id]=-4*lame2*1.0/lame1*grad_lame11_d[id]+2*grad_lame22_d[id];

		//grad_density1_d[id]=-1.0*grad_lame11_d[id]-1.0*grad_lame22_d[id]+grad_den1_d[id];

		grad_vp1_d[id]=2.0*(lame1+2*lame2)*grad_lame11_d[id]*1.0/lame1;

		grad_vs1_d[id]=-4.0*lame2*grad_lame11_d[id]*1.0/lame1+2.0*grad_lame22_d[id];
		
		grad_density1_d[id]=grad_lame11_d[id]+grad_lame22_d[id]+grad_den1_d[id];
	}

}

__global__ void invert_lame_to_velocity_vp(float *grad_vp1_d,float *grad_lame11_d,float *grad_lame22_d,float *grad_den1_d,float *s_velocity_d,float *s_velocity1_d,float *s_density_d,int nx,int nz,int nx_append,int nz_append,int boundary_left,int boundary_up)
///invert_lame_to_velocity_para<<<dimGrid,dimBlock>>>(grad_vp1_d,grad_vs1_d,grad_density1_d,grad_lame11_d,grad_lame22_d,grad_den1_d,s_velocity_d,s_velocity1_d,s_density_d,nx,nz,nx_append,nz_append,boundary_left,boundary_up);
{
	int ix = blockIdx.x*blockDim.x+threadIdx.x;
	int iz = blockIdx.y*blockDim.y+threadIdx.y;  	
	int id,id1;

	float lame1,lame2;

	if(ix<nx&&iz<nz)
	{
		id=ix*nz+iz;
		id1=(ix+boundary_left)*nz_append+iz+boundary_up;

		lame1=s_density_d[id1]*s_velocity_d[id1]*s_velocity_d[id1]-2*s_density_d[id1]*s_velocity1_d[id1]*s_velocity1_d[id1];
		lame2=s_density_d[id1]*s_velocity1_d[id1]*s_velocity1_d[id1];

		grad_vp1_d[id]=2.0*(lame1+2*lame2)*grad_lame11_d[id]*1.0/lame1;
	}

}

__global__ void invert_lame_to_velocity_vs(float *grad_vs1_d,float *grad_lame11_d,float *grad_lame22_d,float *grad_den1_d,float *s_velocity_d,float *s_velocity1_d,float *s_density_d,int nx,int nz,int nx_append,int nz_append,int boundary_left,int boundary_up)
///invert_lame_to_velocity_para<<<dimGrid,dimBlock>>>(grad_vp1_d,grad_vs1_d,grad_density1_d,grad_lame11_d,grad_lame22_d,grad_den1_d,s_velocity_d,s_velocity1_d,s_density_d,nx,nz,nx_append,nz_append,boundary_left,boundary_up);
{
	int ix = blockIdx.x*blockDim.x+threadIdx.x;
	int iz = blockIdx.y*blockDim.y+threadIdx.y;  	
	int id,id1;

	float lame1,lame2;

	if(ix<nx&&iz<nz)
	{
		id=ix*nz+iz;
		id1=(ix+boundary_left)*nz_append+iz+boundary_up;

		lame1=s_density_d[id1]*s_velocity_d[id1]*s_velocity_d[id1]-2*s_density_d[id1]*s_velocity1_d[id1]*s_velocity1_d[id1];
		lame2=s_density_d[id1]*s_velocity1_d[id1]*s_velocity1_d[id1];

		grad_vs1_d[id]=-4.0*lame2*grad_lame11_d[id]*1.0/lame1+2.0*grad_lame22_d[id];
	}

}

__global__ void invert_lame_to_velocity_density(float *grad_density1_d,float *grad_lame11_d,float *grad_lame22_d,float *grad_den1_d,float *s_velocity_d,float *s_velocity1_d,float *s_density_d,int nx,int nz,int nx_append,int nz_append,int boundary_left,int boundary_up)
///invert_lame_to_velocity_para<<<dimGrid,dimBlock>>>(grad_vp1_d,grad_vs1_d,grad_density1_d,grad_lame11_d,grad_lame22_d,grad_den1_d,s_velocity_d,s_velocity1_d,s_density_d,nx,nz,nx_append,nz_append,boundary_left,boundary_up);
{
	int ix = blockIdx.x*blockDim.x+threadIdx.x;
	int iz = blockIdx.y*blockDim.y+threadIdx.y;  	
	int id;
	//int id1;

	//float lame1,lame2;

	if(ix<nx&&iz<nz)
	{
		id=ix*nz+iz;
		//id1=(ix+boundary_left)*nz_append+iz+boundary_up;

		//lame1=s_density_d[id1]*s_velocity_d[id1]*s_velocity_d[id1]-2*s_density_d[id1]*s_velocity1_d[id1]*s_velocity1_d[id1];
		//lame2=s_density_d[id1]*s_velocity1_d[id1]*s_velocity1_d[id1];
		
		grad_density1_d[id]=grad_lame11_d[id]+grad_lame22_d[id]+grad_den1_d[id];
	}

}

__global__ void invert_lame_to_impedance_para_new(float *grad_vp1_d,float *grad_vs1_d,float *grad_density1_d,float *grad_lame11_d,float *grad_lame22_d,float *grad_den1_d,float *s_velocity_d,float *s_velocity1_d,float *s_density_d,int nx,int nz,int nx_append,int nz_append,int boundary_left,int boundary_up)
//invert_lame_to_impedance_para_new<<<dimGrid_new,dimBlock>>>(all_grad_vp1_d,all_grad_vs1_d,all_grad_density1_d,all_grad_lame11_d,all_grad_lame22_d,all_grad_den1_d,s_velocity_all_d,s_velocity1_all_d,s_density_all_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up);
{
	int ix = blockIdx.x*blockDim.x+threadIdx.x;
	int iz = blockIdx.y*blockDim.y+threadIdx.y;  	
	int id,id1;

	if (ix<nx && iz<nz)
	{
		id=ix*nz+iz;
		id1=(ix+boundary_left)*nz_append+iz+boundary_up;


		grad_vp1_d[id]=2*s_velocity_d[id1]*grad_lame11_d[id];

		grad_vs1_d[id]=-4.0*s_velocity1_d[id1]*grad_lame11_d[id]+2.0*s_velocity1_d[id1]*grad_lame22_d[id];
		
		grad_density1_d[id]=(2.0*s_velocity1_d[id1]*s_velocity1_d[id1]-s_velocity_d[id1]*s_velocity_d[id1])*grad_lame11_d[id]-s_velocity1_d[id1]*s_velocity1_d[id1]*grad_lame22_d[id]+grad_den1_d[id];
	}
}


__global__ void invert_lame_to_impedance_para(float *grad_vp1_d,float *grad_vs1_d,float *grad_density1_d,float *grad_lame11_d,float *grad_lame22_d,float *grad_den1_d,float *s_velocity_d,float *s_velocity1_d,float *s_density_d,int nx,int nz,int nx_append,int nz_append,int boundary_left,int boundary_up)
///invert_lame_to_impedance_para<<<dimGrid,dimBlock>>>(grad_vp1_d,grad_vs1_d,grad_density1_d,grad_lame11_d,grad_lame22_d,grad_den1_d,s_velocity_d,s_velocity1_d,s_density_d,nx,nz,nx_append,nz_append,boundary_left);;
{
	int ix = blockIdx.x*blockDim.x+threadIdx.x;
	int iz = blockIdx.y*blockDim.y+threadIdx.y;  	
	int id,id1;

	float lame1,lame2;

	if (ix<nx && iz<nz)
	{
		id=ix*nz+iz;
		id1=(ix+boundary_left)*nz_append+iz+boundary_up;

		lame1=s_density_d[id1]*s_velocity_d[id1]*s_velocity_d[id1]-2*s_density_d[id1]*s_velocity1_d[id1]*s_velocity1_d[id1];
		lame2=s_density_d[id1]*s_velocity1_d[id1]*s_velocity1_d[id1];

		grad_vp1_d[id]=2*(lame1+2*lame2)*1.0/lame1*grad_lame11_d[id];

		grad_vs1_d[id]=-4*lame2*1.0/lame1*grad_lame11_d[id]+2*grad_lame22_d[id];
		
		grad_density1_d[id]=-1.0*grad_lame11_d[id]-1.0*grad_lame22_d[id]+grad_den1_d[id];
	}
}

__global__ void invert_lame_to_impedance_vp(float *grad_vp1_d,float *grad_lame11_d,float *grad_lame22_d,float *grad_den1_d,float *s_velocity_d,float *s_velocity1_d,float *s_density_d,int nx,int nz,int nx_append,int nz_append,int boundary_left,int boundary_up)
///invert_lame_to_impedance_para<<<dimGrid,dimBlock>>>(grad_vp1_d,grad_vs1_d,grad_density1_d,grad_lame11_d,grad_lame22_d,grad_den1_d,s_velocity_d,s_velocity1_d,s_density_d,nx,nz,nx_append,nz_append,boundary_left);;
{
	int ix = blockIdx.x*blockDim.x+threadIdx.x;
	int iz = blockIdx.y*blockDim.y+threadIdx.y;  	
	int id,id1;

	float lame1,lame2;

	if (ix<nx && iz<nz)
	{
		id=ix*nz+iz;
		id1=(ix+boundary_left)*nz_append+iz+boundary_up;

		lame1=s_density_d[id1]*s_velocity_d[id1]*s_velocity_d[id1]-2*s_density_d[id1]*s_velocity1_d[id1]*s_velocity1_d[id1];
		lame2=s_density_d[id1]*s_velocity1_d[id1]*s_velocity1_d[id1];

		grad_vp1_d[id]=2*(lame1+2*lame2)*1.0/lame1*grad_lame11_d[id];
	}
}

__global__ void invert_lame_to_impedance_vs(float *grad_vs1_d,float *grad_lame11_d,float *grad_lame22_d,float *grad_den1_d,float *s_velocity_d,float *s_velocity1_d,float *s_density_d,int nx,int nz,int nx_append,int nz_append,int boundary_left,int boundary_up)
///invert_lame_to_impedance_para<<<dimGrid,dimBlock>>>(grad_vp1_d,grad_vs1_d,grad_density1_d,grad_lame11_d,grad_lame22_d,grad_den1_d,s_velocity_d,s_velocity1_d,s_density_d,nx,nz,nx_append,nz_append,boundary_left);;
{
	int ix = blockIdx.x*blockDim.x+threadIdx.x;
	int iz = blockIdx.y*blockDim.y+threadIdx.y;  	
	int id,id1;

	float lame1,lame2;

	if (ix<nx && iz<nz)
	{
		id=ix*nz+iz;
		id1=(ix+boundary_left)*nz_append+iz+boundary_up;

		lame1=s_density_d[id1]*s_velocity_d[id1]*s_velocity_d[id1]-2*s_density_d[id1]*s_velocity1_d[id1]*s_velocity1_d[id1];
		lame2=s_density_d[id1]*s_velocity1_d[id1]*s_velocity1_d[id1];

		grad_vs1_d[id]=-4*lame2*1.0/lame1*grad_lame11_d[id]+2*grad_lame22_d[id];
	}
}

__global__ void invert_lame_to_impedance_density(float *grad_density1_d,float *grad_lame11_d,float *grad_lame22_d,float *grad_den1_d,float *s_velocity_d,float *s_velocity1_d,float *s_density_d,int nx,int nz,int nx_append,int nz_append,int boundary_left,int boundary_up)
///invert_lame_to_impedance_para<<<dimGrid,dimBlock>>>(grad_vp1_d,grad_vs1_d,grad_density1_d,grad_lame11_d,grad_lame22_d,grad_den1_d,s_velocity_d,s_velocity1_d,s_density_d,nx,nz,nx_append,nz_append,boundary_left);;
{
	int ix = blockIdx.x*blockDim.x+threadIdx.x;
	int iz = blockIdx.y*blockDim.y+threadIdx.y;  	
	int id;
	//int id1;

	//float lame1,lame2;

	if (ix<nx && iz<nz)
	{
		id=ix*nz+iz;
		//id1=(ix+boundary_left)*nz_append+iz+boundary_up;

		//lame1=s_density_d[id1]*s_velocity_d[id1]*s_velocity_d[id1]-2*s_density_d[id1]*s_velocity1_d[id1]*s_velocity1_d[id1];
		//lame2=s_density_d[id1]*s_velocity1_d[id1]*s_velocity1_d[id1];
		
		grad_density1_d[id]=-1.0*grad_lame11_d[id]-1.0*grad_lame22_d[id]+grad_den1_d[id];
	}
}

__global__ void cuda_cal_residuals_new(float *res_shot_x_d,float *cal_shot_x_d,float *obs_shot_x_d,int receiver_num,int lt)
///cuda_cal_residuals_new<<<dimGrid_lt,dimBlock>>>(res_shot_x_d,cal_shot_x_d,obs_shot_x_d,receiver_num,lt);
{
		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		if((ix<receiver_num)&&(iz<lt))
		{
			in_idx=ix*lt+iz;
			
			res_shot_x_d[in_idx]=cal_shot_x_d[in_idx]-obs_shot_x_d[in_idx];
		}
}

__global__ void scale_cal_shot(float *cal_shot_x_d,float *cal_max,float *obs_max,int receiver_num,int lt)
{
		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		if((ix<receiver_num)&&(iz<lt))
		{
			in_idx=ix*lt+iz;

			if(cal_max[0]!=0)	cal_shot_x_d[in_idx]=cal_shot_x_d[in_idx]*obs_max[0]/cal_max[0]*1.0;
		}
}

__global__ void cuda_cal_beta(float *beta, float *g0, float *g1, float *cg, int N)
/*< calculate beta for nonlinear conjugate gradient algorithm 
configuration requirement: <<<1,Block_Size>>> >*/
{
    	__shared__ float sdata[Block_Size];
	__shared__ float tdata[Block_Size];
	__shared__ float rdata[Block_Size];
    	int tid = threadIdx.x;
    	sdata[tid] = 0.0f;
	tdata[tid] = 0.0f;
	rdata[tid] = 0.0f;
	for(int s=0; s<(N+Block_Size-1)/Block_Size; s++)
	{
		int id=s*blockDim.x+threadIdx.x;
		float a=(id<N)?g0[id]:0.0f;
		float b=(id<N)?g1[id]:0.0f;
		//float c=(id<N)?cg[id]:0.0f;

		/* HS: Hestenses-Stiefel NLCG algorithm */
/*
		sdata[tid] += b*(b-a);	// numerator of HS
		tdata[tid] += c*(b-a);	// denominator of HS,DY
		rdata[tid] += b*b;	// numerator of DY
*/
  	
		// PRP: Polark-Ribiere-Polyar NLCG algorithm 

		sdata[tid] += b*(b-a);	// numerator
		tdata[tid] += a*a;	// denominator

		// HS: Hestenses-Stiefel NLCG algorithm 
/*
		sdata[tid] += b*(b-a);	// numerator
		tdata[tid] += c*(b-a);	// denominator
*/
		// FR: Fletcher-Reeves NLCG algorithm 
/*
		sdata[tid] += b*b;	// numerator
		tdata[tid] += a*a;	// denominator
*/
		// PRP: Polark-Ribiere-Polyar NLCG algorithm 
/*
		sdata[tid] += b*(b-a);	// numerator
		tdata[tid] += a*a;	// denominator
*/
		// CD: Fletcher NLCG algorithm  
/*
		sdata[tid] += b*b;	// numerator
		tdata[tid] -= c*a;	// denominator
*/
		// DY: Dai-Yuan NLCG algorithm 
/*
		sdata[tid] += b*b;	// numerator
		tdata[tid] += c*(b-a);	// denominator
*/
	} 
    	__syncthreads();

    	/* do reduction in shared mem */
    	for(int s=blockDim.x/2; s>32; s>>=1) 
    	{
		if (threadIdx.x < s)	{ sdata[tid]+=sdata[tid+s]; tdata[tid]+=tdata[tid+s]; rdata[tid]+=rdata[tid+s];}
		__syncthreads();
    	}     
   	if (tid < 32)
   	{
		if (blockDim.x >=64) { sdata[tid]+=sdata[tid+32]; tdata[tid]+=tdata[tid+32]; rdata[tid]+=rdata[tid+32];}
		if (blockDim.x >=32) { sdata[tid]+=sdata[tid+16]; tdata[tid]+=tdata[tid+16]; rdata[tid]+=rdata[tid+16];}
		if (blockDim.x >=16) { sdata[tid]+=sdata[tid+ 8]; tdata[tid]+=tdata[tid+ 8]; rdata[tid]+=rdata[tid+ 8];}
		if (blockDim.x >= 8) { sdata[tid]+=sdata[tid+ 4]; tdata[tid]+=tdata[tid+ 4]; rdata[tid]+=rdata[tid+ 4];}
		if (blockDim.x >= 4) { sdata[tid]+=sdata[tid+ 2]; tdata[tid]+=tdata[tid+ 2]; rdata[tid]+=rdata[tid+ 2];}
		if (blockDim.x >= 2) { sdata[tid]+=sdata[tid+ 1]; tdata[tid]+=tdata[tid+ 1]; rdata[tid]+=rdata[tid+ 1];}
    	}
     
	if (tid == 0) 
	{ 
		//float beta_HS=0.0;
		//float beta_DY=0.0;
		float beta_PRP=0.0;
		if(fabsf(tdata[0])>EPS) 
		{
			//beta_HS=sdata[0]/tdata[0]; 
			//beta_DY=rdata[0]/tdata[0];

			beta_PRP=sdata[0]/tdata[0];
		} 
		//*beta=max(0.0, min(beta_HS, beta_DY));/* Hybrid HS-DY method combined with iteration restart */

		*beta=beta_PRP;/* Hybrid HS-DY method combined with iteration restart */
	}	
}

__global__ void cuda_cal_beta_new(float *beta, float *g0, float *g1, float *cg, int N,int mark)
///cuda_cal_beta_new<<<1, Block_Size>>>(beta_d, grad_lame1_d, grad_lame11_d, conj_lame1_d, nxnz,0);
/*< calculate beta for nonlinear conjugate gradient algorithm 
configuration requirement: <<<1,Block_Size>>> >*/
{
    	__shared__ float sdata[Block_Size];
	__shared__ float tdata[Block_Size];
	__shared__ float rdata[Block_Size];
    	int tid = threadIdx.x;
    	sdata[tid] = 0.0f;
	tdata[tid] = 0.0f;
	rdata[tid] = 0.0f;

	for(int s=0; s<(N+Block_Size-1)/Block_Size; s++)
	{
		int id=s*blockDim.x+threadIdx.x;
		float a=(id<N)?g0[id]:0.0f;
		float b=(id<N)?g1[id]:0.0f;
		float c=(id<N)?cg[id]:0.0f;

		/* HS: Hestenses-Stiefel NLCG algorithm */

		sdata[tid] += b*(b-a);	// numerator of HS
		tdata[tid] += c*(b-a);	// denominator of HS,DY
		rdata[tid] += b*b;	// numerator of DY

		
/*   	
		// PRP: Polark-Ribiere-Polyar NLCG algorithm 
		sdata[tid] += b*(b-a);	// numerator
		tdata[tid] += a*a;	// denominator
		// HS: Hestenses-Stiefel NLCG algorithm 
		sdata[tid] += b*(b-a);	// numerator
		tdata[tid] += c*(b-a);	// denominator
		// FR: Fletcher-Reeves NLCG algorithm 
		sdata[tid] += b*b;	// numerator
		tdata[tid] += a*a;	// denominator
*/

/*
		// PRP: Polark-Ribiere-Polyar NLCG algorithm 
		sdata[tid] += b*(b-a);	// numerator
		tdata[tid] += a*a;	// denominator
*/

/*
		// CD: Fletcher NLCG algorithm  
		sdata[tid] += b*b;	// numerator
		tdata[tid] -= c*a;	// denominator
*/
/*
		// DY: Dai-Yuan NLCG algorithm 
		sdata[tid] += b*b;	// numerator
		tdata[tid] += c*(b-a);	// denominator
*/
	} 
    	__syncthreads();

    	/* do reduction in shared mem */
    	for(int s=blockDim.x/2; s>32; s>>=1) 
    	{
		if (threadIdx.x < s)	{ sdata[tid]+=sdata[tid+s]; tdata[tid]+=tdata[tid+s]; rdata[tid]+=rdata[tid+s];}
		__syncthreads();
    	}     
   	if (tid < 32)
   	{
		if (blockDim.x >=64) { sdata[tid]+=sdata[tid+32]; tdata[tid]+=tdata[tid+32]; rdata[tid]+=rdata[tid+32];}
		if (blockDim.x >=32) { sdata[tid]+=sdata[tid+16]; tdata[tid]+=tdata[tid+16]; rdata[tid]+=rdata[tid+16];}
		if (blockDim.x >=16) { sdata[tid]+=sdata[tid+ 8]; tdata[tid]+=tdata[tid+ 8]; rdata[tid]+=rdata[tid+ 8];}
		if (blockDim.x >= 8) { sdata[tid]+=sdata[tid+ 4]; tdata[tid]+=tdata[tid+ 4]; rdata[tid]+=rdata[tid+ 4];}
		if (blockDim.x >= 4) { sdata[tid]+=sdata[tid+ 2]; tdata[tid]+=tdata[tid+ 2]; rdata[tid]+=rdata[tid+ 2];}
		if (blockDim.x >= 2) { sdata[tid]+=sdata[tid+ 1]; tdata[tid]+=tdata[tid+ 1]; rdata[tid]+=rdata[tid+ 1];}
    	}
     
	if (tid == 0) 
	{ 
		float beta_HS=0.0;
		float beta_DY=0.0;
		//float beta_PRP=0.0;
		if(fabsf(tdata[0])>EPS) 
		{
			beta_HS=sdata[0]/tdata[0]; 
			beta_DY=rdata[0]/tdata[0];
			//beta_PRP=sdata[0]/tdata[0];
		}
		//*beta=max(0.0, min(beta_HS, beta_DY));/* Hybrid HS-DY method combined with iteration restart */ 
		beta[mark]=max(0.0, min(beta_HS, beta_DY));/* Hybrid HS-DY method combined with iteration restart */

		//beta[1]=beta_PRP;/* PRP method combined with iteration restart */
		//beta[mark]=beta_PRP;
	}	
}

__global__ void cuda_cal_conjgrad(float *g1, float *cg, float beta, int nz, int nx)
/*< calculate nonlinear conjugate gradient >*/
{
	int i1=blockIdx.x*blockDim.x+threadIdx.x;
	int i2=blockIdx.y*blockDim.y+threadIdx.y;
	int id=i1+i2*nz;

	if (i1<nz && i2<nx) cg[id]=-g1[id]+beta*cg[id];
}

__global__ void cuda_cal_conjgrad_new(float *g1, float *cg, float *beta, int nx, int nz,int mark)
///cuda_cal_conjgrad_new<<<dimGrid,dimBlock>>>(grad_lame22_d, conj_lame2_d, beta_d, nx, nz,1);
//cuda_cal_conjgrad_new<<<dimGrid,dimBlock>>>(grad_vp1_d, conj_vp_d, beta_d, nx, nz,0);
/*< calculate nonlinear conjugate gradient >*/
{
	int ix=blockIdx.x*blockDim.x+threadIdx.x;
	int iz=blockIdx.y*blockDim.y+threadIdx.y;

	int id=ix*nz+iz;

	if (ix<nx && iz<nz) cg[id]=-1.0*g1[id]+beta[mark]*cg[id];
}

__global__ void cuda_cal_window(float *s_velocity_d,float *v_window_d,int nx,int nz,int nx_append,int nz_append,int boundary_left,int boundary_up)
////cuda_cal_window<<<dimGrid,dimBlock>>>(expand_perturb_lame1_d,v_window_d,nx,nz,nx_append,nz_append,boundary_left,boundary_up);
{
	int ix=blockIdx.x*blockDim.x+threadIdx.x;
	int iz=blockIdx.y*blockDim.y+threadIdx.y;

	int id=ix*nz+iz;
	int id1=(ix+boundary_left)*nz_append+iz+boundary_up;

	if (ix<nx && iz<nz) 	v_window_d[id]=s_velocity_d[id1];
}

//cuda_cal_expand<<<dimGrid,dimBlock>>>(s_velocity_d,v_window_d,nx,nz,nx_append,nz_append,boundary_left,boundary_up);
__global__ void cuda_cal_expand(float *s_velocity_d,float *v_window_d,int nx,int nz,int nx_append,int nz_append,int boundary_left,int boundary_up)
{
	int ix=blockIdx.x*blockDim.x+threadIdx.x;
	int iz=blockIdx.y*blockDim.y+threadIdx.y;

	//int id=ix*nz+iz;

	int id0=(ix-boundary_left)*nz+iz-boundary_up;

	//int id1=(ix+boundary_left)*nz_append+iz+boundary_up;

	int id2=ix*nz_append+iz;

	if (ix<nx_append && iz<nz_append)
		{
			if(ix>=boundary_left&&ix<boundary_left+nx&&iz>=boundary_up&&iz<boundary_up+nz)	s_velocity_d[id2]=v_window_d[id0];
///up		
			if(ix>=boundary_left&&ix<boundary_left+nx&&iz<boundary_up)			s_velocity_d[id2]=v_window_d[(ix-boundary_left)*nz];
///down
			if(ix>=boundary_left&&ix<boundary_left+nx&&iz>=nz+boundary_up)			s_velocity_d[id2]=v_window_d[(ix-boundary_left)*nz+nz-1];
//left
			if(ix<boundary_left&&iz>=boundary_up&&iz<boundary_up+nz)				s_velocity_d[id2]=v_window_d[0*nz+iz-boundary_up];
//right
			if(ix>=nx+boundary_left&&iz>=boundary_up&&iz<boundary_up+nz)			s_velocity_d[id2]=v_window_d[(nx-1)*nz+iz-boundary_up];
//up left
			if(ix<boundary_left&&iz<boundary_up)						s_velocity_d[id2]=v_window_d[0];
//up right
			if(ix>=nx+boundary_left&&iz<boundary_up)						s_velocity_d[id2]=v_window_d[(nx-1)*nz];
//down left
			if(ix<boundary_left&&iz>=nz+boundary_up)						s_velocity_d[id2]=v_window_d[nz-1];
//down right
			if(ix>=nx+boundary_left&&iz>=nz+boundary_up)					s_velocity_d[id2]=v_window_d[(nx-1)*nz+nz-1];
		}
}

__global__ void cuda_cal_epsilon(float *vv, float *cg, float *epsil, int N)
/*< calculate estimated stepsize (epsil) according to Taratola's method
configuration requirement: <<<1, Block_Size>>> >*/ 
{
    	__shared__ float sdata[Block_Size];/* find max(|vv(:)|) */
	__shared__ float tdata[Block_Size];/* find max(|cg(:)|) */
    	int tid = threadIdx.x;
    	sdata[tid] = 0.0f;
    	tdata[tid] = 0.0f;
	for(int s=0; s<(N+Block_Size-1)/Block_Size; s++)
	{
		int id=s*blockDim.x+threadIdx.x;
		float a=(id<N)?fabsf(vv[id]):0.0f;
		float b=(id<N)?fabsf(cg[id]):0.0f;
		sdata[tid]= max(sdata[tid], a);
		tdata[tid]= max(tdata[tid], b);
	} 
    	__syncthreads();

    	/* do reduction in shared mem */
    	for(int s=blockDim.x/2; s>32; s>>=1) 
    	{
		if (threadIdx.x < s)	{sdata[tid]=max(sdata[tid], sdata[tid+s]);tdata[tid]=max(tdata[tid], tdata[tid+s]);} 
		__syncthreads();
    	}  
   	if (tid < 32)
   	{
		if (blockDim.x >=  64) { sdata[tid] =max(sdata[tid],sdata[tid + 32]);tdata[tid]=max(tdata[tid], tdata[tid+32]);}
		if (blockDim.x >=  32) { sdata[tid] =max(sdata[tid],sdata[tid + 16]);tdata[tid]=max(tdata[tid], tdata[tid+16]);}
		if (blockDim.x >=  16) { sdata[tid] =max(sdata[tid],sdata[tid + 8]);tdata[tid]=max(tdata[tid], tdata[tid+8]);}
		if (blockDim.x >=   8) { sdata[tid] =max(sdata[tid],sdata[tid + 4]);tdata[tid]=max(tdata[tid], tdata[tid+4]);}
		if (blockDim.x >=   4) { sdata[tid] =max(sdata[tid],sdata[tid + 2]);tdata[tid]=max(tdata[tid], tdata[tid+2]);}
		if (blockDim.x >=   2) { sdata[tid] =max(sdata[tid],sdata[tid + 1]);tdata[tid]=max(tdata[tid], tdata[tid+1]);}
    	}

    	if (tid == 0) { if(tdata[0]>EPS) *epsil=0.01*sdata[0]/tdata[0]; else *epsil=0.0;}
}


__global__ void cuda_cal_epsilon_new(float *vv, float *cg, float *epsil, int N,int mark)
/*< calculate estimated stepsize (epsil) according to Taratola's method
configuration requirement: <<<1, Block_Size>>> >*/ 
{
    	__shared__ float sdata[Block_Size];/* find max(|vv(:)|) */
	__shared__ float tdata[Block_Size];/* find max(|cg(:)|) */
    	int tid = threadIdx.x;
    	sdata[tid] = 0.0f;
    	tdata[tid] = 0.0f;
	for(int s=0; s<(N+Block_Size-1)/Block_Size; s++)
	{
		int id=s*blockDim.x+threadIdx.x;
		float a=(id<N)?fabsf(vv[id]):0.0f;
		float b=(id<N)?fabsf(cg[id]):0.0f;
		sdata[tid]= max(sdata[tid], a);
		tdata[tid]= max(tdata[tid], b);
	} 
    	__syncthreads();

    	/* do reduction in shared mem */
    	for(int s=blockDim.x/2; s>32; s>>=1) 
    	{
		if (threadIdx.x < s)	{sdata[tid]=max(sdata[tid], sdata[tid+s]);tdata[tid]=max(tdata[tid], tdata[tid+s]);} 
		__syncthreads();
    	}  
   	if (tid < 32)
   	{
		if (blockDim.x >=  64) { sdata[tid] =max(sdata[tid],sdata[tid + 32]);tdata[tid]=max(tdata[tid], tdata[tid+32]);}
		if (blockDim.x >=  32) { sdata[tid] =max(sdata[tid],sdata[tid + 16]);tdata[tid]=max(tdata[tid], tdata[tid+16]);}
		if (blockDim.x >=  16) { sdata[tid] =max(sdata[tid],sdata[tid + 8]);tdata[tid]=max(tdata[tid], tdata[tid+8]);}
		if (blockDim.x >=   8) { sdata[tid] =max(sdata[tid],sdata[tid + 4]);tdata[tid]=max(tdata[tid], tdata[tid+4]);}
		if (blockDim.x >=   4) { sdata[tid] =max(sdata[tid],sdata[tid + 2]);tdata[tid]=max(tdata[tid], tdata[tid+2]);}
		if (blockDim.x >=   2) { sdata[tid] =max(sdata[tid],sdata[tid + 1]);tdata[tid]=max(tdata[tid], tdata[tid+1]);}
    	}

    	if (tid == 0) { if(tdata[0]>EPS) epsil[mark]=0.01*sdata[0]/tdata[0]; else epsil[mark]=0.0;}

	//if (tid == 0) { if(tdata[0]>EPS) epsil[mark]=0.0; else epsil[mark]=0.0;}
}

__global__ void cuda_cal_vtmp(float *vtmp, float *vv, float *cg, float epsil, int nz, int nx, int window_vel)
/*< calculate temporary velocity >*/ 
{
	int i1=threadIdx.x+blockIdx.x*blockDim.x;
	int i2=threadIdx.y+blockIdx.y*blockDim.x;
	int id=i1+i2*nz;

	//if (i1<nz && i1>=window_vel && i2<nx)	vtmp[id]=vv[id]+epsil*cg[id];

	if (i1<nz && i2<nx)	vtmp[id]=vv[id]+epsil*cg[id];
}

__global__ void cuda_cal_vtmp_new(float *vtmp, float *vv, float *cg, float *epsil, int nx, int nz,int nx_append,int nz_append,int boundary_left,int boundary_up,int mark)
//cuda_cal_vtmp_new<<<dimGrid,dimBlock>>>(v_window_d,s_velocity_d, conj_vp_d, epsil_d,nx,nz,nx_append,nz_append,boundary_left,boundary_up,0);
//cuda_cal_vtmp_new<<<dimGrid,dimBlock>>>(v_window_d,s_density_d,conj_density_d,epsil_d,nx,nz,nx_append,nz_append,boundary_left,boundary_up,1);
/*< calculate temporary velocity >*/ 
{
	//int i1=threadIdx.x+blockIdx.x*blockDim.x;
	//int i2=threadIdx.y+blockIdx.y*blockDim.x;
	int ix = blockIdx.x*blockDim.x+threadIdx.x;
	int iz = blockIdx.y*blockDim.y+threadIdx.y;  	
	int id,id1;

	if (ix<nx && iz<nz)
	{
		id=ix*nz+iz;
		id1=(ix+boundary_left)*nz_append+iz+boundary_up;

		vtmp[id]=vv[id1]+epsil[mark]*cg[id];

		if(iz<1)	vtmp[id]=vv[id1];
	}
}

__global__ void cuda_cal_vtmp_new_new(float *vtmp, float *vv, float *cg, float *epsil, int nx, int nz,int nx_append,int nz_append,int boundary_left,int boundary_up)
/*< calculate temporary velocity >*/ 
{
	//int i1=threadIdx.x+blockIdx.x*blockDim.x;
	//int i2=threadIdx.y+blockIdx.y*blockDim.x;
	int ix = blockIdx.x*blockDim.x+threadIdx.x;
	int iz = blockIdx.y*blockDim.y+threadIdx.y;  	
	int id,id1;

	if (ix<nx && iz<nz)
	{
		id=ix*nz+iz;
		id1=(ix+boundary_left)*nz_append+iz+boundary_up;

		if(epsil[1]>epsil[0])	vtmp[id]=vv[id1]+cg[id];

		if(iz<1)	vtmp[id]=vv[id1];
	}
}

__global__ void cuda_cal_vtmp_fixed(float *vtmp, float *vv, float *cg, int nx, int nz,int nx_append,int nz_append,int boundary_left,int boundary_up,float mark)
//cuda_cal_vtmp_fixed<<<dimGrid,dimBlock>>>(v_window_d,s_velocity_d, conj_vp_d, epsil_d,nx,nz,nx_append,nz_append,boundary_left,boundary_up,0);
/*< calculate temporary velocity >*/ 
{
	//int i1=threadIdx.x+blockIdx.x*blockDim.x;
	//int i2=threadIdx.y+blockIdx.y*blockDim.x;
	int ix = blockIdx.x*blockDim.x+threadIdx.x;
	int iz = blockIdx.y*blockDim.y+threadIdx.y;  	
	int id,id1;

	if (ix<nx && iz<nz)
	{
		id=ix*nz+iz;
		id1=(ix+boundary_left)*nz_append+iz+boundary_up;

		vtmp[id]=vv[id1]+mark*cg[id];

		if(iz<1)	vtmp[id]=vv[id1];
	}
}


__global__ void cuda_update_vel_new(float *vv, float *cg, float *alpha, int nx, int nz, int mark)
/*< update velocity model with obtained stepsize (alpha) >*/
{
	//int ix=threadIdx.x+blockIdx.x*blockDim.x;
	//int iz=threadIdx.y+blockIdx.y*blockDim.x;
	int ix = blockIdx.x*blockDim.x+threadIdx.x;
	int iz = blockIdx.y*blockDim.y+threadIdx.y;  
	int id=iz+ix*nz;

	//if (i1<nz && i1>=window_vel && i2<nx) vv[id]=vv[id]+alpha*cg[id];
	if (ix<nx && iz<nz&&iz>1) vv[id]=vv[id]+alpha[mark]*cg[id];
}

__global__ void cuda_sum_alpha12(float *alpha1, float *alpha2, float *dcaltmp, float *dobs, float *derr, int ng)
//cuda_sum_alpha12<<<(ng+511)/512, 512>>>(d_alpha1, d_alpha2, d_dcal, &d_dobs[it*ng], &d_derr[it*ng], ng);
/*< calculate the numerator and denominator of alpha
	alpha1: numerator; length=ng
	alpha2: denominator; length=ng >*/
{
	int id=threadIdx.x+blockDim.x*blockIdx.x;
	if(id<ng) { 
		float c=derr[id];
		float a=dobs[id]+c;/* since f(mk)-dobs[id]=derr[id], thus f(mk)=b+c; */
		float b=dcaltmp[id]-a;/* f(mk+epsil*cg)-f(mk) */
		alpha1[id]-=b*c; alpha2[id]+=b*b; 
	}
}

__global__ void cuda_sum_alpha12_new(float *alpha1, float *alpha2, float *dcaltmp, float *dobs, float *derr, int ng,int lt)
//cuda_sum_alpha12_new<<<dimGrid_lt,dimBlock>>>(d_alpha1,d_alpha2,cal_shot_d,obs_shot_d,res_shot_d,receiver_num,lt);
/*< calculate the numerator and denominator of alpha
	alpha1: numerator; length=ng
	alpha2: denominator; length=ng >*/
{
	int id;
	//int id=threadIdx.x+blockDim.x*blockIdx.x;
	//int ix=threadIdx.x+blockIdx.x*blockDim.x;
	//int iz=threadIdx.y+blockIdx.y*blockDim.x;
	int ix = blockIdx.x*blockDim.x+threadIdx.x;
	int iz = blockIdx.y*blockDim.y+threadIdx.y;  
	if(ix<ng&&iz<lt) 
	{ 
		id=ix*lt+iz;		

		float c=derr[id];
		float a=dobs[id]+c;/* since f(mk)-dobs[id]=derr[id], thus f(mk)=b+c; */
		float b=dcaltmp[id]-a;/* f(mk+epsil*cg)-f(mk) */
		alpha1[id]+=b*c; alpha2[id]+=b*b; 
	}
}

__global__ void cuda_sum_alpha12_new_for_lsrtm(float *alpha1, float *alpha2, float *dcaltmp, float *dobs, float *derr, int ng,int lt)
/*< calculate the numerator and denominator of alpha
	alpha1: numerator; length=ng
	alpha2: denominator; length=ng >*/
{
	int id;
	//int id=threadIdx.x+blockDim.x*blockIdx.x;
	//int ix=threadIdx.x+blockIdx.x*blockDim.x;
	//int iz=threadIdx.y+blockIdx.y*blockDim.x;
	int ix = blockIdx.x*blockDim.x+threadIdx.x;
	int iz = blockIdx.y*blockDim.y+threadIdx.y;  
	if(ix<ng&&iz<lt) 
	{ 
		id=ix*lt+iz;		

		float c=derr[id];
		//float a=dobs[id]+c;/* since f(mk)-dobs[id]=derr[id], thus f(mk)=b+c; */
		float b=dcaltmp[id];/* f(mk+epsil*cg)-f(mk) */
		alpha1[id]-=b*c; alpha2[id]+=b*b; 
	}
}

__global__ void cuda_cal_alpha(float *alpha, float *alpha1, float *alpha2, float epsil, int ng)
/*< calculate searched stepsize (alpha) according to Taratola's method
configuration requirement: <<<1, Block_Size>>> >*/ 
{
  	__shared__ float sdata[Block_Size];
	__shared__ float tdata[Block_Size];
    	int tid=threadIdx.x;
    	sdata[tid]=0.0f;
	tdata[tid]=0.0f;
	for(int s=0; s<(ng+Block_Size-1)/Block_Size; s++)
	{
		int id=s*blockDim.x+threadIdx.x;
		float a=(id<ng)?alpha1[id]:0.0f;
		float b=(id<ng)?alpha2[id]:0.0f;
		sdata[tid] +=a;	
		tdata[tid] +=b;	
	} 
    	__syncthreads();

    	/* do reduction in shared mem */
    	for(int s=blockDim.x/2; s>32; s>>=1) 
    	{
		if (threadIdx.x < s) { sdata[tid] += sdata[tid + s];tdata[tid] += tdata[tid + s]; } __syncthreads();
    	}
   	if (tid < 32)
   	{
		if (blockDim.x >=  64) { sdata[tid] += sdata[tid + 32]; tdata[tid] += tdata[tid + 32];}
		if (blockDim.x >=  32) { sdata[tid] += sdata[tid + 16]; tdata[tid] += tdata[tid + 16];}
		if (blockDim.x >=  16) { sdata[tid] += sdata[tid +  8]; tdata[tid] += tdata[tid +  8];}
		if (blockDim.x >=   8) { sdata[tid] += sdata[tid +  4]; tdata[tid] += tdata[tid +  4];}
		if (blockDim.x >=   4) { sdata[tid] += sdata[tid +  2]; tdata[tid] += tdata[tid +  2];}
		if (blockDim.x >=   2) { sdata[tid] += sdata[tid +  1]; tdata[tid] += tdata[tid +  1];}
    	}
     
    	if (tid == 0) { if(tdata[0]>EPS) *alpha=epsil*sdata[0]/(tdata[0]+EPS); else *alpha=0.0;}
}

__global__ void cuda_cal_alpha_new(float *alpha, float *alpha1, float *alpha2, float *epsil, int ng,int mark)
/*< calculate searched stepsize (alpha) according to Taratola's method
configuration requirement: <<<1, Block_Size>>> >*/ 
{
  	__shared__ float sdata[Block_Size];
	__shared__ float tdata[Block_Size];
    	int tid=threadIdx.x;
    	sdata[tid]=0.0f;
	tdata[tid]=0.0f;
	for(int s=0; s<(ng+Block_Size-1)/Block_Size; s++)
	{
		int id=s*blockDim.x+threadIdx.x;
		float a=(id<ng)?alpha1[id]:0.0f;
		float b=(id<ng)?alpha2[id]:0.0f;
		sdata[tid] +=a;	
		tdata[tid] +=b;	
	} 
    	__syncthreads();

    	/* do reduction in shared mem */
    	for(int s=blockDim.x/2; s>32; s>>=1) 
    	{
		if (threadIdx.x < s) { sdata[tid] += sdata[tid + s];tdata[tid] += tdata[tid + s]; } __syncthreads();
    	}
   	if (tid < 32)
   	{
		if (blockDim.x >=  64) { sdata[tid] += sdata[tid + 32]; tdata[tid] += tdata[tid + 32];}
		if (blockDim.x >=  32) { sdata[tid] += sdata[tid + 16]; tdata[tid] += tdata[tid + 16];}
		if (blockDim.x >=  16) { sdata[tid] += sdata[tid +  8]; tdata[tid] += tdata[tid +  8];}
		if (blockDim.x >=   8) { sdata[tid] += sdata[tid +  4]; tdata[tid] += tdata[tid +  4];}
		if (blockDim.x >=   4) { sdata[tid] += sdata[tid +  2]; tdata[tid] += tdata[tid +  2];}
		if (blockDim.x >=   2) { sdata[tid] += sdata[tid +  1]; tdata[tid] += tdata[tid +  1];}
    	}
     
    	if (tid == 0) { if(tdata[0]>EPS) alpha[mark]=-1.0*epsil[mark]*sdata[0]/(tdata[0]+EPS); else alpha[mark]=0.0;}
}

__global__ void cuda_cal_alpha_new_for_lsrtm(float *alpha, float *alpha1, float *alpha2, float *epsil, int ng,int mark)
/*< calculate searched stepsize (alpha) according to Taratola's method
configuration requirement: <<<1, Block_Size>>> >*/ 
{
  	__shared__ float sdata[Block_Size];
	__shared__ float tdata[Block_Size];
    	int tid=threadIdx.x;
    	sdata[tid]=0.0f;
	tdata[tid]=0.0f;
	for(int s=0; s<(ng+Block_Size-1)/Block_Size; s++)
	{
		int id=s*blockDim.x+threadIdx.x;
		float a=(id<ng)?alpha1[id]:0.0f;
		float b=(id<ng)?alpha2[id]:0.0f;
		sdata[tid] +=a;	
		tdata[tid] +=b;	
	} 
    	__syncthreads();

    	/* do reduction in shared mem */
    	for(int s=blockDim.x/2; s>32; s>>=1) 
    	{
		if (threadIdx.x < s) { sdata[tid] += sdata[tid + s];tdata[tid] += tdata[tid + s]; } __syncthreads();
    	}
   	if (tid < 32)
   	{
		if (blockDim.x >=  64) { sdata[tid] += sdata[tid + 32]; tdata[tid] += tdata[tid + 32];}
		if (blockDim.x >=  32) { sdata[tid] += sdata[tid + 16]; tdata[tid] += tdata[tid + 16];}
		if (blockDim.x >=  16) { sdata[tid] += sdata[tid +  8]; tdata[tid] += tdata[tid +  8];}
		if (blockDim.x >=   8) { sdata[tid] += sdata[tid +  4]; tdata[tid] += tdata[tid +  4];}
		if (blockDim.x >=   4) { sdata[tid] += sdata[tid +  2]; tdata[tid] += tdata[tid +  2];}
		if (blockDim.x >=   2) { sdata[tid] += sdata[tid +  1]; tdata[tid] += tdata[tid +  1];}
    	}
     
    	//if (tid == 0) { if(tdata[0]>EPS) alpha[mark]=sdata[0]/(tdata[0]+EPS); else *alpha=0.0;}
	
	if (tid == 0) {alpha[mark]=1.0*sdata[0]/(tdata[0]);}
}

__global__ void cuda_cal_alpha_new_for_correlation_lsrtm(float *alpha,float *correlation_parameter_d,int mark)
{
	float a,b;
	float tmp_tmp,obs_obs,cal_cal,tmp_obs,tmp_cal,cal_obs;

	tmp_tmp=correlation_parameter_d[0];///////////////#p*#p

	obs_obs=correlation_parameter_d[1];///////////////p_obs*p_obs

	cal_cal=correlation_parameter_d[2];///////////////p_p*p_p

	tmp_obs=correlation_parameter_d[3];///////////////#p*p_obs

	tmp_cal=correlation_parameter_d[4];///////////////#p*p_p

	cal_obs=correlation_parameter_d[5];///////////////p_p*_obs

	
	//a=1.0/obs_obs/tmp_tmp*(	(2.0*cal_obs*tmp_cal+tmp_obs*cal_cal)/tmp_tmp-3.0*tmp_obs*tmp_cal*tmp_cal/tmp_tmp/tmp_tmp	);

	//b=1.0/obs_obs/tmp_tmp*(tmp_obs*tmp_cal/tmp_tmp-cal_obs);

	a=1.0/sqrt(1.0*obs_obs*tmp_tmp)*(	(2.0*cal_obs*tmp_cal+tmp_obs*cal_cal)/tmp_tmp-3.0*tmp_obs*tmp_cal*tmp_cal/tmp_tmp/tmp_tmp	);

	b=1.0/sqrt(1.0*obs_obs*tmp_tmp)*(tmp_obs*tmp_cal/tmp_tmp-cal_obs);


	alpha[mark]=-1.0*b/a;
}

__global__ void cal_hydrid_conj(float *hydrid_conj_d,float *conj_vp_d,float *conj_vs_d,float *conj_density_d,float *beta_step,int nx,int nz)
{
	int id;
	//int id=threadIdx.x+blockDim.x*blockIdx.x;
	//int ix=threadIdx.x+blockIdx.x*blockDim.x;
	//int iz=threadIdx.y+blockIdx.y*blockDim.x;
	int ix = blockIdx.x*blockDim.x+threadIdx.x;
	int iz = blockIdx.y*blockDim.y+threadIdx.y;  
	if(ix<nx&&iz<nz) 
	{ 
		id=ix*nz+iz;	
		hydrid_conj_d[id]=beta_step[0]*conj_vp_d[id]+beta_step[1]*conj_vs_d[id]+beta_step[2]*conj_density_d[id];
	}
}

__global__ void cuda_update_vel(float *vv, float *cg, float alpha, int nx, int nz)
/*< update velocity model with obtained stepsize (alpha) >*/
{
	//int ix=threadIdx.x+blockIdx.x*blockDim.x;
	//int iz=threadIdx.y+blockIdx.y*blockDim.x;
	int ix = blockIdx.x*blockDim.x+threadIdx.x;
	int iz = blockIdx.y*blockDim.y+threadIdx.y;  
	int id=iz+ix*nz;

	//if (i1<nz && i1>=window_vel && i2<nx) vv[id]=vv[id]+alpha*cg[id];
	if (ix<nx && iz<nz) vv[id]=vv[id]+alpha*cg[id];
}

__global__ void cuda_update_shots(float *res_shot_x_d,float *cal_shot_x_d,float *beta_step_d,int receiver_num,int lt,int mark)
{
	int ix = blockIdx.x*blockDim.x+threadIdx.x;
	int iz = blockIdx.y*blockDim.y+threadIdx.y;  	
	int id;

	if (ix<receiver_num && iz<lt)
	{
		id=ix*lt+iz;
		
		res_shot_x_d[id]=res_shot_x_d[id]+beta_step_d[mark]*cal_shot_x_d[id];
	}
}

__global__ void cuda_update_shots_new(float *res_shot_x_d,float *cal_shot_x_d,float *beta_step_d,int receiver_num,int lt,int mark,int precon_z2)
{
	int ix = blockIdx.x*blockDim.x+threadIdx.x;
	int iz = blockIdx.y*blockDim.y+threadIdx.y;  	
	int id;

	if (ix<receiver_num && iz<lt)
	{
		id=ix*lt+iz;
		if(iz>precon_z2)	res_shot_x_d[id]=1.0*res_shot_x_d[id]+beta_step_d[mark]*cal_shot_x_d[id];

		else			res_shot_x_d[id]=0.01*res_shot_x_d[id]+beta_step_d[mark]*cal_shot_x_d[id];
	}
}

__global__ void cuda_update_tmp_shots(float *tmp_shot_x_d,float *cal_shot_x_d,float *beta_step_d,int receiver_num,int lt,int mark)
{
	int ix = blockIdx.x*blockDim.x+threadIdx.x;
	int iz = blockIdx.y*blockDim.y+threadIdx.y;  	
	int id;

	if (ix<receiver_num && iz<lt)
	{
		id=ix*lt+iz;
		
		tmp_shot_x_d[id]=tmp_shot_x_d[id]+beta_step_d[mark]*cal_shot_x_d[id];
	}
}

__global__ void cuda_update_shots_and_image(float *cal_shot_x_d,float *res_shot_x1_d,float *res_shot_x2_d,float *cg_parameter_d,int receiver_num,int lt,int mark)
//cuda_update_shots_and_image<<<dimGrid_lt,dimBlock>>>(cal_shot_x_d,res_shot_x1_d,res_shot_x2_d,cg_parameter_d,receiver_num,lt);
//cuda_update_shots_and_image<<<dimGrid_lt,dimBlock>>>(cal_shot_z_d,res_shot_z1_d,res_shot_z2_d,cg_parameter_d,receiver_num,lt);
//cuda_update_shots_and_image<<<dimGrid,dimBlock>>>(grad_lame11_d,perturb_lame1_d,perturb_lame11_d,cg_parameter_d,nx,nz);						
//cuda_update_shots_and_image<<<dimGrid,dimBlock>>>(grad_lame22_d,perturb_lame2_d,perturb_lame22_d,cg_parameter_d,nx,nz);

{
	int ix = blockIdx.x*blockDim.x+threadIdx.x;
	int iz = blockIdx.y*blockDim.y+threadIdx.y;  	
	int id;

	if (ix<receiver_num && iz<lt)
	{
		id=ix*lt+iz;
		if(mark==0)
		{
			res_shot_x2_d[id]=(cg_parameter_d[5]*cal_shot_x_d[id]+cg_parameter_d[6]*res_shot_x1_d[id]);
		}
		if(mark==1)
		{
			res_shot_x2_d[id]=-1.0*(cg_parameter_d[5]*cal_shot_x_d[id]+cg_parameter_d[6]*res_shot_x1_d[id]);
		}
		
	}
}
__global__ void cuda_scale_gradient_new(float *grad_vp1_d,float *d_illum,int nx,int nz,int dimx,int dimz,int boundary_left,int boundary_up,int precon)
{
	int ix = blockIdx.x*blockDim.x+threadIdx.x;
	int iz = blockIdx.y*blockDim.y+threadIdx.y;
	int in_idx;
	int in_idx1;
	if (ix<nx && iz<nz)
	{
		in_idx=ix*nz+iz;
		in_idx1=(ix+boundary_left)*dimz+iz+boundary_up;

		//grad_vp1_d[in_idx]=grad_vp1_d[in_idx]/(sqrt(d_illum[in_idx1]+EPS));

		grad_vp1_d[in_idx]=grad_vp1_d[in_idx]/(d_illum[in_idx1]+EPS);
	}
}

__global__ void cuda_scale_gradient_new_1(float *grad_vp1_d,float *d_illum,int nx,int nz,int dimx,int dimz,int boundary_left,int boundary_up,int precon,int z1,int z2,float scale)
{
	int ix = blockIdx.x*blockDim.x+threadIdx.x;
	int iz = blockIdx.y*blockDim.y+threadIdx.y;
	int in_idx;
	int in_idx1;

	//float m=0.0;

	if (ix<nx && iz<nz)
	{
		in_idx=ix*nz+iz;
		in_idx1=(ix+boundary_left)*dimz+iz+boundary_up;

		//grad_vp1_d[in_idx]=grad_vp1_d[in_idx]/(sqrtf(d_illum[in_idx1]+EPS));

		grad_vp1_d[in_idx]=scale*grad_vp1_d[in_idx]/(d_illum[in_idx1]+EPS);

		/*if(z1!=z2)		
		{
			if(iz<z1)	grad_vp1_d[in_idx]=0.0;
////Ren 2016 Kohn 2012
			if(iz>=z1&&iz<=z2)
			{		
				m=2.0*3.0*(iz-z1-(z2-z1))/(z2-z1);			
				grad_vp1_d[in_idx]=grad_vp1_d[in_idx]*float(exp(-0.5*m*m));
			}
		}*/
	}
}

__global__ void cuda_attenuation_after_lap(float *grad_vp1_d,int nx,int nz,int dimx,int dimz,int boundary_left,int boundary_up,int z1,int z2)
{
/////////////////////2017年03月25日 星期六 21时32分15秒 仔细想一想，应该先laplace  再衰减
	int ix = blockIdx.x*blockDim.x+threadIdx.x;
	int iz = blockIdx.y*blockDim.y+threadIdx.y;
	int in_idx;
	//int in_idx1;

	float m=0.0;

	if (ix<nx && iz<nz)
	{
		in_idx=ix*nz+iz;
		//in_idx1=(ix+boundary_left)*dimz+iz+boundary_up;

		if(z1!=z2&&z2!=0)		
		{
			if(iz<z1)	grad_vp1_d[in_idx]=0.0;
////Ren 2016 Kohn 2012
			if(iz>=z1&&iz<=z2)
			{		
				m=2.0*3.0*(iz-z1-(z2-z1))/(z2-z1);			
				grad_vp1_d[in_idx]=grad_vp1_d[in_idx]*float(exp(-0.5*m*m));
			}
		}
	}
}

__global__ void cuda_attenuation_after_lap_new(float *grad_vp1_d,int nx,int nz,int dimx,int dimz,int boundary_left,int boundary_up,int z1,int z2)
{
/////////////////////2017年03月25日 星期六 21时32分15秒 仔细想一想，应该先laplace  再衰减
	int ix = blockIdx.x*blockDim.x+threadIdx.x;
	int iz = blockIdx.y*blockDim.y+threadIdx.y;
	int in_idx;
	//int in_idx1;

	float m=0.0;

	if (ix<nx && iz<nz)
	{
		in_idx=ix*nz+iz;
		//in_idx1=(ix+boundary_left)*dimz+iz+boundary_up;

		if(z1!=z2&&z2!=0)		
		{
			if(iz<z1)	grad_vp1_d[in_idx]=0.0;
////Ren 2016 Kohn 2012
			if(iz>=z1&&iz<=z2)
			{		
				m=2.0*3.0*(iz-z1-(z2-z1))/(z2-z1);			
				grad_vp1_d[in_idx]=grad_vp1_d[in_idx]*float(exp(-0.5*m*m));
			}
		}

		if(ix==0)	grad_vp1_d[in_idx]=grad_vp1_d[in_idx]*0.6666;

		if(ix==nx-1)	grad_vp1_d[in_idx]=grad_vp1_d[in_idx]*0.6666;
		
		if(iz==0)	grad_vp1_d[in_idx]=grad_vp1_d[in_idx]*0.6666;

		if(iz==nz-1)	grad_vp1_d[in_idx]=grad_vp1_d[in_idx]*0.6666;	
	}
}

__global__ void cuda_attenuation_after_lap_new1(float *grad_vp1_d,int nx,int nz,int dimx,int dimz,int boundary_left,int boundary_up,int z1,int z2)
{
/////////////////////2017年03月25日 星期六 21时32分15秒 仔细想一想，应该先laplace  再衰减
	int ix = blockIdx.x*blockDim.x+threadIdx.x;
	int iz = blockIdx.y*blockDim.y+threadIdx.y;
	int in_idx;
	//int in_idx1;

	float m=0.0;
	double change;

	if (ix<nx && iz<nz)
	{
		in_idx=ix*nz+iz;
		//in_idx1=(ix+boundary_left)*dimz+iz+boundary_up;

		
		if(iz<=z1)
		{
			grad_vp1_d[in_idx]=0.0;
		}

		if(iz>=z1&&iz<=z2)
		{		
			m=1.0*(z2-iz)/(z2-z1);
			change=pow(cos(pai/2*m),3);
				
			grad_vp1_d[in_idx]=grad_vp1_d[in_idx]*change*1.0;
		}
		

		if(ix==0)	grad_vp1_d[in_idx]=grad_vp1_d[in_idx]*0.6666;

		if(ix==nx-1)	grad_vp1_d[in_idx]=grad_vp1_d[in_idx]*0.6666;
		
		if(iz==0)	grad_vp1_d[in_idx]=grad_vp1_d[in_idx]*0.6666;

		if(iz==nz-1)	grad_vp1_d[in_idx]=grad_vp1_d[in_idx]*0.6666;	
	}
}

__global__ void cuda_attenuation_after_lap_new2(float *grad_vp1_d,int nx,int nz,int dimx,int dimz,int boundary_left,int boundary_up,int z1,int z2)
{
/////////////////////2017年03月25日 星期六 21时32分15秒 仔细想一想，应该先laplace  再衰减
	int ix = blockIdx.x*blockDim.x+threadIdx.x;
	int iz = blockIdx.y*blockDim.y+threadIdx.y;
	int in_idx;
	//int in_idx1;

	float m=0.0;
	double change;

	if (ix<nx && iz<nz)
	{
		in_idx=ix*nz+iz;
		//in_idx1=(ix+boundary_left)*dimz+iz+boundary_up;

		if(z1==0)
		{
			if(iz<=z1)
			{
				grad_vp1_d[in_idx]=0.0;
			}

			if(iz>=z1&&iz<=z2)
			{		
				m=1.0*(z2-iz)/(z2-z1);
				change=pow(cos(pai/2*m),3);
				
				grad_vp1_d[in_idx]=grad_vp1_d[in_idx]*change*1.0;
			}
		}

		else
		{
			if(z1!=z2&&z2!=0)		
			{
				if(iz<z1)	grad_vp1_d[in_idx]=0.0;
////Ren 2016 Kohn 2012
				if(iz>=z1&&iz<=z2)
				{		
					m=2.0*3.0*(iz-z1-(z2-z1))/(z2-z1);			
					grad_vp1_d[in_idx]=grad_vp1_d[in_idx]*float(exp(-0.5*m*m));
				}
			}
		}

		if(ix==0)	grad_vp1_d[in_idx]=grad_vp1_d[in_idx]*0.6666;

		if(ix==nx-1)	grad_vp1_d[in_idx]=grad_vp1_d[in_idx]*0.6666;
		
		if(iz==0)	grad_vp1_d[in_idx]=grad_vp1_d[in_idx]*0.6666;

		if(iz==nz-1)	grad_vp1_d[in_idx]=grad_vp1_d[in_idx]*0.6666;	
	}
}

__global__ void cuda_attenuation_adj(float *adj_shot_x_d,int receiver_num,int lt,int offset_left,int offset_right,int receiver_offset)
{
	int ix = blockIdx.x*blockDim.x+threadIdx.x;
	int iz = blockIdx.y*blockDim.y+threadIdx.y;
	int in_idx;

	float m0=0.0;
	
	double change;

	int distance_left;
	int distance_right;

	/*float p0=0.1,p1=0.9;

	float angle0,angle1;

	angle0=float(acos(sqrt(1.0*p0)));
	angle1=float(acos(sqrt(1.0*p1)));*/

	float pd;

	int off0,off1,off2,off3;

	if (ix<receiver_num && iz<lt)
	{
		in_idx=ix*lt+iz;
		
		if((offset_left=receiver_offset)||(offset_right=receiver_offset))
		{
				off0=0;
				off1=int(offset_left/3);
				off2=int(receiver_num-offset_right/3);
				off3=receiver_num;
				
				if(ix>=off0&&ix<off1)
				{
					//m0=-1.0*angle0+(ix-off0)*(angle0/(off1-off0));

					m0=1.0*(off1-ix)/(off1-off0)*pai/2;

					pd=float(pow(cos(m0*1.0),3.0));

					adj_shot_x_d[in_idx]=adj_shot_x_d[in_idx]*pd*1.0;
				}

				if(ix>=off2&&ix<off3)
				{
					//m0=1.0*(ix-off2)*(angle1/(off3-off2));

					m0=1.0*(ix-off2)/(off3-off2)*pai/2;

					pd=float(pow(cos(m0*1.0),3.0));
					adj_shot_x_d[in_idx]=adj_shot_x_d[in_idx]*pd*1.0;
				}			
		}

		if(offset_left>receiver_offset)
		{
			if(ix>=0&&ix<(offset_left-receiver_offset))
				adj_shot_x_d[in_idx]=0;

			distance_left=int(receiver_offset/3);

			if(ix>=(offset_left-receiver_offset)&&ix<(offset_left-receiver_offset+distance_left)&&offset_left!=0)
			{
				m0=1.0*(offset_left-receiver_offset+distance_left-ix);

				change=pow(cos(pai/2*m0/distance_left),3);				

				adj_shot_x_d[in_idx]=adj_shot_x_d[in_idx]*change;
			}
		}

		if(offset_right>receiver_offset)
		{
			if(ix>=(offset_left+receiver_offset)&&ix<receiver_num)
				adj_shot_x_d[in_idx]=0;
			
			distance_right=int(receiver_offset/3);

			if(ix>=(offset_left+receiver_offset-distance_right)&&ix<(offset_left+receiver_offset)&&offset_right!=0)
			{
				m0=1.0*(ix-(offset_left+receiver_offset-distance_right));

				change=pow(cos(pai/2*m0/distance_right),3);

				adj_shot_x_d[in_idx]=adj_shot_x_d[in_idx]*change;
			}
		}

	}
}

__global__ void cuda_scale_gradient_2(float *grad_vp1_d,int nx,int nz,int dimx,int dimz,int boundary_left,int boundary_up,int precon,int z1,int z2,float scale)
{
	int ix = blockIdx.x*blockDim.x+threadIdx.x;
	int iz = blockIdx.y*blockDim.y+threadIdx.y;
	int in_idx;
	
	float m=0.0;

	if (ix<nx && iz<nz)
	{
		in_idx=ix*nz+iz;

		if(iz<z1)	grad_vp1_d[in_idx]=0.0;
////Ren 2016 Kohn 2012
		if(z1!=z2&&z2!=0)		
		{
			if(iz>=z1&&iz<=z2)
			{		
				m=2.0*3.0*(iz-z1-(z2-z1))/(z2-z1);			
				grad_vp1_d[in_idx]=grad_vp1_d[in_idx]*float(exp(-0.5*m*m))*scale;
			}

			if(iz>z2)	grad_vp1_d[in_idx]=(iz*1.0/z2)*grad_vp1_d[in_idx]*scale;
		}
	}
}

__global__ void cuda_scale_gradient_acqusition(float *grad_vp1_d,float *d_illum,int nx,int nz,int dimx,int dimz,int boundary_left,int boundary_up,int precon,int z1,int z2,float scale)
{
	int ix = blockIdx.x*blockDim.x+threadIdx.x;
	int iz = blockIdx.y*blockDim.y+threadIdx.y;
	int in_idx;
	int in_idx1;

	float m=0.0;

	if (ix<nx && iz<nz)
	{
		in_idx=ix*nz+iz;

		in_idx1=(ix+boundary_left)*dimz+iz+boundary_up;

		if(precon==1)
		{
			//grad_vp1_d[in_idx]=grad_vp1_d[in_idx]/(sqrtf(d_illum[in_idx1]+EPS));
	
			grad_vp1_d[in_idx]=1.0*scale*grad_vp1_d[in_idx]/(d_illum[in_idx1]+EPS);
		}

		if(precon==2)
		{
			if(iz<z1)	grad_vp1_d[in_idx]=0.0;
////Ren 2016 Kohn 2012
			if(z1!=z2&&z2!=0)		
			{
				if(iz>=z1&&iz<=z2)
				{		
					m=2.0*3.0*(iz-z1-(z2-z1))/(z2-z1);			
					grad_vp1_d[in_idx]=grad_vp1_d[in_idx]*float(exp(-0.5*m*m))*scale;
				}

				if(iz>z2)	grad_vp1_d[in_idx]=(iz*1.0/z2)*grad_vp1_d[in_idx]*scale;
			}
		}
		
	}
}


__global__ void cuda_scale_gradient_acqusition_only_RTM(float *grad_vp1_d,float *d_illum,int nx,int nz,int dimx,int dimz,int boundary_left,int boundary_up,int precon,int z1,int z2,float scale,int offset_left,int offset_right,int receiver_offset,int offset_attenuation)
///cuda_scale_gradient_acqusition_only_RTM<<<dimGrid,dimBlock>>>(vresultppz_d,d_illum,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up,precon,precon_z1,precon_z2,0.0000001,offset_left[ishot],offset_right[ishot],receiver_offset);
{
	int ix = blockIdx.x*blockDim.x+threadIdx.x;
	int iz = blockIdx.y*blockDim.y+threadIdx.y;
	int in_idx;

	int distance_left;
	int distance_right;

	float m=0.0;
	double change;

	if (ix<nx && iz<nz)
	{
		in_idx=ix*nz+iz;

		if(precon==1)
		{
			//grad_vp1_d[in_idx]=grad_vp1_d[in_idx]/(sqrtf(d_illum[in_idx]+EPS));
			//grad_vp1_d[in_idx]=1.0*scale*grad_vp1_d[in_idx]/(d_illum[in_idx]+EPS);

			if(d_illum[in_idx]!=0)	grad_vp1_d[in_idx]=1.0*scale*grad_vp1_d[in_idx]/(d_illum[in_idx]);
		}

		if(precon==2)
		{
			if(iz<z1)	grad_vp1_d[in_idx]=0.0;
////Ren 2016 Kohn 2012
			if(z1!=z2&&z2!=0)		
			{
				if(iz>=z1&&iz<=z2)
				{		
					m=2.0*3.0*(iz-z1-(z2-z1))/(z2-z1);			
					grad_vp1_d[in_idx]=grad_vp1_d[in_idx]*float(exp(-0.5*m*m))*scale;
				}

				if(iz>z2)	grad_vp1_d[in_idx]=(iz*1.0/z2)*grad_vp1_d[in_idx]*scale;
			}
		}

		if(receiver_offset!=0&&offset_attenuation!=0)
		{

			if(offset_left<=receiver_offset||offset_right<=receiver_offset)
			{
				distance_left=int(offset_left/offset_attenuation);
				distance_right=int(offset_right/offset_attenuation);

				//if(distance_left<=2)	distance_left=3;
				//if(distance_right<=2)	distance_right=3;///2017年09月05日 星期二 08时44分25秒  it is important,when I join offset_attenuation

				if(ix<=distance_left&&offset_left!=0&&distance_left!=0)
				{
					m=1.0*(distance_left-ix);

					change=pow(cos(pai/2*m/distance_left),3);

					grad_vp1_d[in_idx]=grad_vp1_d[in_idx]*change;
				}

				if(ix>=nx-distance_right&&offset_right!=0&&distance_right!=0)
				{
					m=1.0*(ix-nx+distance_right);

					change=pow(cos(pai/2*m/distance_right),3);

					grad_vp1_d[in_idx]=grad_vp1_d[in_idx]*change;
				}
			}

			if(offset_left>receiver_offset)
			{
				if(ix>=0&&ix<(offset_left-receiver_offset))
					grad_vp1_d[in_idx]=0;

				distance_left=int(receiver_offset/offset_attenuation);

				//if(distance_left<=2)		distance_left=3;///2017年09月05日 星期二 08时44分25秒  it is important,when I join offset_attenuation

				if(ix>=(offset_left-receiver_offset)&&ix<(offset_left-receiver_offset+distance_left)&&offset_left!=0&&distance_left!=0)
				{
					m=1.0*(offset_left-receiver_offset+distance_left-ix);

					change=pow(cos(pai/2*m/distance_left),3);				

					grad_vp1_d[in_idx]=grad_vp1_d[in_idx]*change;
				}
			}

			if(offset_right>receiver_offset)
			{
				if(ix>=(offset_left+receiver_offset)&&ix<nx)
					grad_vp1_d[in_idx]=0;
				
				distance_right=int(receiver_offset/offset_attenuation);

				//if(distance_right<=2)	distance_right=3;///2017年09月05日 星期二 08时44分25秒  it is important,when I join offset_attenuation

				if(ix>=(offset_left+receiver_offset-distance_right)&&ix<(offset_left+receiver_offset)&&offset_right!=0&&distance_right!=0)
				{
					m=1.0*(ix-(offset_left+receiver_offset-distance_right));

					change=pow(cos(pai/2*m/distance_right),3);

					grad_vp1_d[in_idx]=grad_vp1_d[in_idx]*change;
				}
			}	
		}	
	}
}

__global__ void cuda_scale_gradient_acqusition_only_RTM_ex_amp(float *grad_vp1_d,float *d_illum,int nx,int nz,int dimx,int dimz,int boundary_left,int boundary_up,int precon,int z1,int z2,float scale,int offset_left,int offset_right,int receiver_offset,int offset_attenuation)
///cuda_scale_gradient_acqusition_only_RTM<<<dimGrid,dimBlock>>>(vresultppz_d,d_illum,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up,precon,precon_z1,precon_z2,0.0000001,offset_left[ishot],offset_right[ishot],receiver_offset);
{
	int ix = blockIdx.x*blockDim.x+threadIdx.x;
	int iz = blockIdx.y*blockDim.y+threadIdx.y;
	int in_idx;

	int distance_left;
	int distance_right;

	float m=0.0;
	double change;

	if (ix<nx && iz<nz)
	{
		in_idx=ix*nz+iz;

		if(receiver_offset!=0&&offset_attenuation!=0)
		{

			if(offset_left<=receiver_offset||offset_right<=receiver_offset)
			{
				distance_left=int(offset_left/offset_attenuation);
				distance_right=int(offset_right/offset_attenuation);

				//if(distance_left<=2)	distance_left=3;
				//if(distance_right<=2)	distance_right=3;///2017年09月05日 星期二 08时44分25秒  it is important,when I join offset_attenuation

				if(ix<=distance_left&&offset_left!=0&&distance_left!=0)
				{
					m=1.0*(distance_left-ix);

					change=pow(cos(pai/2*m/distance_left),3);

					grad_vp1_d[in_idx]=grad_vp1_d[in_idx]*change;
				}

				if(ix>=nx-distance_right&&offset_right!=0&&distance_right!=0)
				{
					m=1.0*(ix-nx+distance_right);

					change=pow(cos(pai/2*m/distance_right),3);

					grad_vp1_d[in_idx]=grad_vp1_d[in_idx]*change;
				}
			}

			if(offset_left>receiver_offset)
			{
				if(ix>=0&&ix<(offset_left-receiver_offset))
					grad_vp1_d[in_idx]=0;

				distance_left=int(receiver_offset/offset_attenuation);

				//if(distance_left<=2)		distance_left=3;///2017年09月05日 星期二 08时44分25秒  it is important,when I join offset_attenuation

				if(ix>=(offset_left-receiver_offset)&&ix<(offset_left-receiver_offset+distance_left)&&offset_left!=0&&distance_left!=0)
				{
					m=1.0*(offset_left-receiver_offset+distance_left-ix);

					change=pow(cos(pai/2*m/distance_left),3);				

					grad_vp1_d[in_idx]=grad_vp1_d[in_idx]*change;
				}
			}

			if(offset_right>receiver_offset)
			{
				if(ix>=(offset_left+receiver_offset)&&ix<nx)
					grad_vp1_d[in_idx]=0;
				
				distance_right=int(receiver_offset/offset_attenuation);

				//if(distance_right<=2)	distance_right=3;///2017年09月05日 星期二 08时44分25秒  it is important,when I join offset_attenuation

				if(ix>=(offset_left+receiver_offset-distance_right)&&ix<(offset_left+receiver_offset)&&offset_right!=0&&distance_right!=0)
				{
					m=1.0*(ix-(offset_left+receiver_offset-distance_right));

					change=pow(cos(pai/2*m/distance_right),3);

					grad_vp1_d[in_idx]=grad_vp1_d[in_idx]*change;
				}
			}	
		}	
	}
}

///////////////attenuation two point
__global__ void cuda_scale_gradient_acqusition_new(float *grad_vp1_d,float *d_illum,int nx,int nz,int dimx,int dimz,int boundary_left,int boundary_up,int precon,int z1,int z2,float scale,int offset_left,int offset_right,int receiver_offset,int offset_attenuation)
///cuda_scale_gradient_acqusition_new<<<dimGrid,dimBlock>>>(vresultppz_d,d_illum,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up,precon,precon_z1,precon_z2,0.0000001,offset_left[ishot],offset_right[ishot],receiver_offset);
{
	int ix = blockIdx.x*blockDim.x+threadIdx.x;
	int iz = blockIdx.y*blockDim.y+threadIdx.y;
	int in_idx;
	int in_idx1;

	int distance_left;
	int distance_right;

	float m=0.0;
	double change;

	if (ix<nx && iz<nz)
	{
		in_idx=ix*nz+iz;

		in_idx1=(ix+boundary_left)*dimz+iz+boundary_up;

		if(precon==1)
		{
			//grad_vp1_d[in_idx]=grad_vp1_d[in_idx]/(sqrtf(d_illum[in_idx1]+EPS));
			grad_vp1_d[in_idx]=scale*grad_vp1_d[in_idx]/(d_illum[in_idx1]+EPS);
		}

		if(precon==2)
		{
			if(iz<z1)	grad_vp1_d[in_idx]=0.0;
////Ren 2016 Kohn 2012
			if(z1!=z2&&z2!=0)		
			{
				if(iz>=z1&&iz<=z2)
				{		
					m=2.0*3.0*(iz-z1-(z2-z1))/(z2-z1);			
					grad_vp1_d[in_idx]=grad_vp1_d[in_idx]*float(exp(-0.5*m*m))*scale;
				}

				if(iz>z2)	grad_vp1_d[in_idx]=(iz*1.0/z2)*grad_vp1_d[in_idx]*scale;
			}
		}

		if(receiver_offset!=0&&offset_attenuation!=0)
		{

			if(offset_left<=receiver_offset||offset_right<=receiver_offset)
			{
				distance_left=int(offset_left/offset_attenuation);
				distance_right=int(offset_right/offset_attenuation);

				//if(distance_left<=2)	distance_left=3;
				//if(distance_right<=2)	distance_right=3;///2017年09月05日 星期二 08时44分25秒  it is important,when I join offset_attenuation

				if(ix<=distance_left&&offset_left!=0&&distance_left!=0)
				{
					m=1.0*(distance_left-ix);

					change=pow(cos(pai/2*m/distance_left),3);

					grad_vp1_d[in_idx]=grad_vp1_d[in_idx]*change;
				}

				if(ix>=nx-distance_right&&offset_right!=0&&distance_right!=0)
				{
					m=1.0*(ix-nx+distance_right);

					change=pow(cos(pai/2*m/distance_right),3);

					grad_vp1_d[in_idx]=grad_vp1_d[in_idx]*change;
				}
			}

			if(offset_left>receiver_offset)
			{
				if(ix>=0&&ix<(offset_left-receiver_offset))
					grad_vp1_d[in_idx]=0;

				distance_left=int(receiver_offset/offset_attenuation);

				//if(distance_left<=2)		distance_left=3;///2017年09月05日 星期二 08时44分25秒  it is important,when I join offset_attenuation

				if(ix>=(offset_left-receiver_offset)&&ix<(offset_left-receiver_offset+distance_left)&&offset_left!=0&&distance_left!=0)
				{
					m=1.0*(offset_left-receiver_offset+distance_left-ix);

					change=pow(cos(pai/2*m/distance_left),3);				

					grad_vp1_d[in_idx]=grad_vp1_d[in_idx]*change;
				}
			}

			if(offset_right>receiver_offset)
			{
				if(ix>=(offset_left+receiver_offset)&&ix<nx)
					grad_vp1_d[in_idx]=0;
				
				distance_right=int(receiver_offset/offset_attenuation);

				//if(distance_right<=2)	distance_right=3;///2017年09月05日 星期二 08时44分25秒  it is important,when I join offset_attenuation

				if(ix>=(offset_left+receiver_offset-distance_right)&&ix<(offset_left+receiver_offset)&&offset_right!=0&&distance_right!=0)
				{
					m=1.0*(ix-(offset_left+receiver_offset-distance_right));

					change=pow(cos(pai/2*m/distance_right),3);

					grad_vp1_d[in_idx]=grad_vp1_d[in_idx]*change;
				}
			}	
		}	
	}
}


__global__ void cuda_scale_gradient_acqusition_new_old(float *grad_vp1_d,float *d_illum,int nx,int nz,int dimx,int dimz,int boundary_left,int boundary_up,int precon,int z1,int z2,float scale,int offset_left,int offset_right,int receiver_offset)
///cuda_scale_gradient_acqusition_new<<<dimGrid,dimBlock>>>(vresultppz_d,d_illum,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up,precon,precon_z1,precon_z2,0.0000001,offset_left[ishot],offset_right[ishot],receiver_offset);
{
	int ix = blockIdx.x*blockDim.x+threadIdx.x;
	int iz = blockIdx.y*blockDim.y+threadIdx.y;
	int in_idx;
	int in_idx1;

	int distance_left;
	int distance_right;

	float m=0.0;
	double change;

	if (ix<nx && iz<nz)
	{
		in_idx=ix*nz+iz;

		in_idx1=(ix+boundary_left)*dimz+iz+boundary_up;

		if(precon==1)
		{
			//grad_vp1_d[in_idx]=grad_vp1_d[in_idx]/(sqrtf(d_illum[in_idx1]+EPS));
			grad_vp1_d[in_idx]=scale*grad_vp1_d[in_idx]/(d_illum[in_idx1]+EPS);
		}

		if(precon==2)
		{
			if(iz<z1)	grad_vp1_d[in_idx]=0.0;
////Ren 2016 Kohn 2012
			if(z1!=z2&&z2!=0)		
			{
				if(iz>=z1&&iz<=z2)
				{		
					m=2.0*3.0*(iz-z1-(z2-z1))/(z2-z1);			
					grad_vp1_d[in_idx]=grad_vp1_d[in_idx]*float(exp(-0.5*m*m))*scale;
				}

				if(iz>z2)	grad_vp1_d[in_idx]=(iz*1.0/z2)*grad_vp1_d[in_idx]*scale;
			}
		}

		distance_left=int(offset_left/3);
		distance_right=int(offset_right/3);

		if(receiver_offset!=0)
		{
			if(ix<=distance_left&&offset_left!=0)
			{
				m=1.0*(distance_left-ix);

				change=pow(cos(pai/2*m/distance_left),3);

				//grad_vp1_d[in_idx]=grad_vp1_d[in_idx]*float(exp(-0.1*m*m));

				grad_vp1_d[in_idx]=grad_vp1_d[in_idx]*change;
			}

			if(ix>=nx-distance_right&&offset_right!=0)
			{
				m=1.0*(ix-nx+distance_right);

				change=pow(cos(pai/2*m/distance_right),3);

				//grad_vp1_d[in_idx]=grad_vp1_d[in_idx]*float(exp(-0.1*m*m));

				grad_vp1_d[in_idx]=grad_vp1_d[in_idx]*change;
			}
		}	
	}
}

__global__ void cuda_vsp_precondition(float *grad_vp1_d,int nx,int nz,int nx_append,int nz_append,int boundary_left,int boundary_up,int receiver_num,int receiver_x_cord,int receiver_interval,int receiver_z_cord,int receiver_z_interval)
{
	int ix = blockIdx.x*blockDim.x+threadIdx.x;
	int iz = blockIdx.y*blockDim.y+threadIdx.y;
	int in_idx;
	//int id;

	if (ix<nx&&iz<nz)
	{
		in_idx=ix*nz+iz;
		if(iz>=receiver_z_cord&&iz<(receiver_z_cord+receiver_num*receiver_z_interval))
		{
			if(receiver_x_cord<3)
			{
				if(ix<receiver_x_cord+2)				grad_vp1_d[in_idx]=0.0;
			}

			if(receiver_x_cord>=3&&receiver_x_cord<=nx-3)
			{
				if(ix>=receiver_x_cord-2&&ix<=receiver_x_cord+2)	grad_vp1_d[in_idx]=0.0;
			}	

			if(receiver_x_cord>nx-3)
			{
				if(ix>receiver_x_cord)				grad_vp1_d[in_idx]=0.0;
			}
		}
	}
}

/////////////////////////precondition:cut the top artifact
__global__ void cuda_precon_cut(float *grad_vp1_d,int nx,int nz,int dimx,int dimz,int boundary_left,int boundary_up)
{
	int ix = blockIdx.x*blockDim.x+threadIdx.x;
	int iz = blockIdx.y*blockDim.y+threadIdx.y;
	int in_idx;
	int in_idx1;
	int zzzz=40;
	float change;
	if(ix<nx&&iz<nz)
	{
		in_idx=ix*nz+iz;
		in_idx1=ix*nz+zzzz;
		change=cos((zzzz-iz)/zzzz*pai/2);

		//change=exp(-1.0*(zzzz-iz));

		//if(iz<zzzz)	grad_vp1_d[in_idx]=grad_vp1_d[in_idx1];
		
		if(iz<zzzz)	grad_vp1_d[in_idx]=grad_vp1_d[in_idx1]*change*change*change;
	}
}

__global__ void cuda_bell_smoothz(float *g, float *smg, int rbell, int nz, int nx)
/*< smoothing with gaussian function >*/
{
	int i;
	int i1=threadIdx.x+blockIdx.x*blockDim.x;
	int i2=threadIdx.y+blockIdx.y*blockDim.y;
	int id=i1+i2*nz;
	if(i1<nz && i2<nx)
	{
		float s=0;
		for(i=-rbell; i<=rbell; i++) if(i1+i>=0 && i1+i<nz) s+=expf(-(2.0*i*i)/rbell)*g[id+i];
		smg[id]=s;
	}
}

__global__ void cuda_bell_smoothx(float *g, float *smg, int rbell, int nz, int nx)
/*< smoothing with gaussian function >*/
{
	int i;
	int i1=threadIdx.x+blockIdx.x*blockDim.x;
	int i2=threadIdx.y+blockIdx.y*blockDim.y;
	int id=i1+i2*nz;
	if(i1<nz && i2<nx)
	{
		float s=0;
		for(i=-rbell; i<=rbell; i++) if(i2+i>=0 && i2+i<nx) s+=expf(-(2.0*i*i)/rbell)*g[id+nz*i];
		smg[id]=s;
	}
}

__global__ void cuda_cal_max(float *obs_max,float *obs_shot_x_d,int N)
////configuration requirement: <<<1, Block_Size>>> >*/ 
{
    	__shared__ float sdata[Block_Size];/* find max(|vv(:)|) */
	
    	int tid = threadIdx.x;
    	sdata[tid] = 0.0f;
    
	for(int s=0; s<(N+Block_Size-1)/Block_Size; s++)
	{
		int id=s*blockDim.x+threadIdx.x;
		float a=(id<N)?fabsf(obs_shot_x_d[id]):0.0f;
		//float b=(id<N)?fabsf(cg[id]):0.0f;
		sdata[tid]= max(sdata[tid], a);
		//tdata[tid]= max(tdata[tid], b);
	} 
    	__syncthreads();

    	/* do reduction in shared mem */
    	for(int s=blockDim.x/2; s>32; s>>=1) 
    	{
		if (threadIdx.x < s)	{sdata[tid]=max(sdata[tid], sdata[tid+s]);} 
		__syncthreads();
    	}  
   	if (tid < 32)
   	{
		if (blockDim.x >=  64) { sdata[tid] =max(sdata[tid],sdata[tid + 32]);}
		if (blockDim.x >=  32) { sdata[tid] =max(sdata[tid],sdata[tid + 16]);}
		if (blockDim.x >=  16) { sdata[tid] =max(sdata[tid],sdata[tid + 8]);}
		if (blockDim.x >=   8) { sdata[tid] =max(sdata[tid],sdata[tid + 4]);}
		if (blockDim.x >=   4) { sdata[tid] =max(sdata[tid],sdata[tid + 2]);}
		if (blockDim.x >=   2) { sdata[tid] =max(sdata[tid],sdata[tid + 1]);}
    	}

    	if (tid == 0) {*obs_max=sdata[0]; }
}

__global__ void cuda_dot(float *matrix1,float *matrix2,int ng,float *dot)
{
	__shared__ float  sdata[Block_Size];
    	int tid=threadIdx.x;
    	sdata[tid]=0.0f;
	for(int s=0; s<(ng+Block_Size-1)/Block_Size; s++)
	{
		int id=s*blockDim.x+threadIdx.x;
		float a=(id<ng)?matrix1[id]:0.0f;
		float b=(id<ng)?matrix2[id]:0.0f;
		sdata[tid] += a*b;	
	} 
    	__syncthreads();

    	/* do reduction in shared mem */
    	for(int s=blockDim.x/2; s>32; s>>=1) 
    	{
		if (threadIdx.x < s) sdata[tid] += sdata[tid + s]; __syncthreads();
    	}
   	if (tid < 32)
   	{
		if (blockDim.x >=  64) { sdata[tid] += sdata[tid + 32]; }
		if (blockDim.x >=  32) { sdata[tid] += sdata[tid + 16]; }
		if (blockDim.x >=  16) { sdata[tid] += sdata[tid +  8]; }
		if (blockDim.x >=   8) { sdata[tid] += sdata[tid +  4]; }
		if (blockDim.x >=   4) { sdata[tid] += sdata[tid +  2]; }
		if (blockDim.x >=   2) { sdata[tid] += sdata[tid +  1]; }
    	}
     
    	if (tid == 0) { *dot=sdata[0]; }
}

__global__ void cuda_dot_sum(float *matrix1,float *matrix2,int ng,float *dot_sum )
{
	__shared__ float  sdata[Block_Size];
    	int tid=threadIdx.x;
    	sdata[tid]=0.0f;
	for(int s=0; s<(ng+Block_Size-1)/Block_Size; s++)
	{
		int id=s*blockDim.x+threadIdx.x;
		float a=(id<ng)?matrix1[id]:0.0f;
		float b=(id<ng)?matrix2[id]:0.0f;
		sdata[tid] += a*b;	
	} 
    	__syncthreads();

    	/* do reduction in shared mem */
    	for(int s=blockDim.x/2; s>32; s>>=1) 
    	{
		if (threadIdx.x < s) sdata[tid] += sdata[tid + s]; __syncthreads();
    	}
   	if (tid < 32)
   	{
		if (blockDim.x >=  64) { sdata[tid] += sdata[tid + 32]; }
		if (blockDim.x >=  32) { sdata[tid] += sdata[tid + 16]; }
		if (blockDim.x >=  16) { sdata[tid] += sdata[tid +  8]; }
		if (blockDim.x >=   8) { sdata[tid] += sdata[tid +  4]; }
		if (blockDim.x >=   4) { sdata[tid] += sdata[tid +  2]; }
		if (blockDim.x >=   2) { sdata[tid] += sdata[tid +  1]; }
    	}
     
    	if (tid == 0) { *dot_sum+=sdata[0]; }
}

__global__ void cuda_cal_alpha_and_beta(float *cg_parameter_d)
{
		
	float gg=cg_parameter_d[0];
	float ss=cg_parameter_d[1];
	float gr=-1.0*cg_parameter_d[2];
	float gs=cg_parameter_d[3];
	float sr=-1.0*cg_parameter_d[4];

	float denominator=gg*ss*max(1.0-(gs/gg)*(gs/ss),EPS);
	//float denominator=gg*ss*(1.0-(gs/gg)*(gs/ss));

	cg_parameter_d[5]=(ss*gr-gs*sr)/denominator;

	cg_parameter_d[6]=(gg*sr-gs*gr)/denominator;

	if(ss==0)//////////the steepest decline	
	{
		cg_parameter_d[5]=gr/gg;
		cg_parameter_d[6]=0;
	}
}

__global__ void cuda_cal_alpha_and_beta_old(float *cg_parameter_d)
{
		
	float gg=cg_parameter_d[0];
	float ss=cg_parameter_d[1];
	float gr=cg_parameter_d[2];
	float gs=cg_parameter_d[3];
	float sr=cg_parameter_d[4];

	float denominator=gg*ss*max(1.0-(gs/gg)*(gs/ss),EPS);
	//float denominator=gg*ss*(1.0-(gs/gg)*(gs/ss));

	cg_parameter_d[5]=(gs*sr-ss*gr)/denominator;

	cg_parameter_d[6]=(gs*gr-gg*sr)/denominator;

	if(ss==0)//////////the steepest decline	
	{
		cg_parameter_d[5]=-1.0*gr/gg;
		cg_parameter_d[6]=0;
	}
}

__global__ void cuda_cal_lame_to_velocity(float *tmp_lame1_d,float *tmp_lame2_d,float *tmp_density_d,float *tmp_velocity_d,float *tmp_velocity1_d,int dimx,int dimz)
//cuda_cal_lame_to_velocity<<<dimGrid,dimBlock>>>(expand_perturb_lame1_d,expand_perturb_lame2_d,s_density_d,expand_perturb_vp_d,expand_perturb_vs_d,nx_append,nz_append);
{
	int ix = blockIdx.x*blockDim.x+threadIdx.x;
	int iz = blockIdx.y*blockDim.y+threadIdx.y;  
	int id=iz+ix*dimz;

	if(ix<dimx&&iz<dimz)
	{
///////////note:::sqrt(A)          A must >0!!!!
		tmp_velocity_d[id]=(tmp_lame1_d[id]+2*tmp_lame2_d[id])/tmp_density_d[id]*1.0;	
		tmp_velocity1_d[id]=tmp_lame2_d[id]/tmp_density_d[id];

		//tmp_velocity_d[id]=float(sqrt((tmp_lame1_d[id]+2*tmp_lame2_d[id])/tmp_density_d[id]*1.0));
		//tmp_velocity1_d[id]=float(sqrt(tmp_lame2_d[id]/tmp_density_d[id]));	
	}
}

__global__ void cuda_cal_velocity_to_lame(float *tmp_lame1_d,float *tmp_lame2_d,float *tmp_density_d,float *tmp_velocity_d,float *tmp_velocity1_d,int dimx,int dimz)
{
	int ix = blockIdx.x*blockDim.x+threadIdx.x;
	int iz = blockIdx.y*blockDim.y+threadIdx.y;  
	int id=iz+ix*dimz;

	if(ix<dimx&&iz<dimz)
	{
		tmp_lame2_d[id]=tmp_density_d[id]*tmp_velocity1_d[id]*tmp_velocity1_d[id];

		tmp_lame1_d[id]=tmp_density_d[id]*tmp_velocity_d[id]*tmp_velocity_d[id]-2*tmp_lame2_d[id];		
	}
}

__global__ void sum_integral(float *vxp_integral_d,float *vxp_d,int dimx,int dimz)
{
	int ix = blockIdx.x*blockDim.x+threadIdx.x;
	int iz = blockIdx.y*blockDim.y+threadIdx.y;
	int in_idx;

	if((ix<dimx)&&(iz<dimz))
	{
		dimx=dimx+2*radius;dimz=dimz+2*radius;
		ix=ix+radius;iz=iz+radius;
		in_idx = ix*dimz+iz;//iz*dimx+ix;

		vxp_integral_d[in_idx]+=vxp_d[in_idx];/////////////vxp_integral_d  :every shot for zero
	}
}


__global__ void calcualte_hydrid_grad(float *hydrid_grad_d,float *grad_lame11_d,int nx,int nz,int mark)
{
	int ix = blockIdx.x*blockDim.x+threadIdx.x;
	int iz = blockIdx.y*blockDim.y+threadIdx.y;
	int in_idx;

	if((ix<nx)&&(iz<nz))
	{
		in_idx=ix*nz+iz;//iz*dimx+ix;

		hydrid_grad_d[mark*nx*nz+in_idx]=grad_lame11_d[in_idx];/////////////vxp_integral_d  :every shot for zero
	}
}



__global__ void cuda_cal_dem_parameter_elastic_media(float *dem_p1_d,float *dem_p2_d,float *dem_p3_d,float *dem_p4_d,float *dem_p5_d,float *vx_x_d,float *vx_z_d,float *vz_x_d,float *vz_z_d,float *vx_t_d,float *vz_t_d,float *tmp_perturb_lame1_d,float *tmp_perturb_lame2_d,float *tmp_perturb_den_d,float *tmp_perturb_vp_d,float *tmp_perturb_vs_d,float *tmp_perturb_density_d,int dimx,int dimz,float *s_velocity_d,float *s_velocity1_d,float *s_density_d,int inversion_para)
//cuda_cal_dem_parameter_elastic_media<<<dimGrid,dimBlock>>>(dem_p1_d,dem_p2_d,dem_p3_d,dem_p4_d,dem_p5_d,vx_x_d,vx_z_d,vz_x_d,vz_z_d,vx_t_d,vz_t_d,tmp_perturb_lame1_d,tmp_perturb_lame2_d,tmp_perturb_den_d,tmp_perturb_vp_d,tmp_perturb_vs_d,tmp_perturb_density_d,nx_append_radius,nz_append_radius,s_velocity_d,s_velocity1_d,s_density_d,inversion_para);
{
		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;
		
		//float dt_real=dt/1000;
		float lame1,lame2;

		float p1,p2,p3;
		float p4,p5,p6;

		if((ix<dimx)&&(iz<dimz))
		{
				dimx=dimx+2*radius2;dimz=dimz+2*radius2;
				ix=ix+radius2;iz=iz+radius2;
				in_idx = ix*dimz+iz;//iz*dimx+ix;			

				lame1=s_density_d[in_idx]*s_velocity_d[in_idx]*s_velocity_d[in_idx]-2.0*s_density_d[in_idx]*s_velocity1_d[in_idx]*s_velocity1_d[in_idx];
				lame2=s_density_d[in_idx]*s_velocity1_d[in_idx]*s_velocity1_d[in_idx];
				
				if(inversion_para==1)
				{
					p1=tmp_perturb_lame1_d[in_idx]*lame1*1.0;

					p2=tmp_perturb_lame2_d[in_idx]*lame2*1.0;

					p3=tmp_perturb_den_d[in_idx]*s_density_d[in_idx]*1.0;				
				}
				
				if(inversion_para==2)
				{
					p4=tmp_perturb_vp_d[in_idx]*s_velocity_d[in_idx]*1.0;

					p5=tmp_perturb_vs_d[in_idx]*s_velocity1_d[in_idx]*1.0;

					p6=tmp_perturb_density_d[in_idx]*s_density_d[in_idx]*1.0;
				}

				if(inversion_para==3)
				{
					p4=tmp_perturb_vp_d[in_idx]*s_velocity_d[in_idx]*s_density_d[in_idx]*1.0;

					p5=tmp_perturb_vs_d[in_idx]*s_velocity1_d[in_idx]*s_density_d[in_idx]*1.0;

					p6=tmp_perturb_density_d[in_idx]*s_density_d[in_idx]*1.0;
				}

				if(inversion_para==1)
				{
					p1=1.0*p1;

					p2=1.0*p2;

					p3=1.0*p3;
				}

				if(inversion_para==2)
				{
					p1=2.0*s_density_d[in_idx]*s_velocity_d[in_idx]*p4-4.0*s_density_d[in_idx]*s_velocity1_d[in_idx]*p5+1.0*(1.0*s_velocity_d[in_idx]*s_velocity_d[in_idx]-2.0*s_velocity1_d[in_idx]*s_velocity1_d[in_idx])*p6;

					//p2=-2.0*s_density_d[in_idx]*s_velocity1_d[in_idx]*p5+1.0*s_velocity1_d[in_idx]*s_velocity1_d[in_idx]*p6;
					p2=2.0*s_density_d[in_idx]*s_velocity1_d[in_idx]*p5+1.0*s_velocity1_d[in_idx]*s_velocity1_d[in_idx]*p6;

					p3=p6;
				}

				if(inversion_para==3)
				{
					p1=2*s_velocity_d[in_idx]*p4-4*s_velocity1_d[in_idx]*p5+(-1.0*s_velocity_d[in_idx]*s_velocity_d[in_idx]+2*s_velocity1_d[in_idx]*s_velocity1_d[in_idx])*p6;

					p2=2.0*s_velocity1_d[in_idx]*p5-1.0*s_velocity1_d[in_idx]*s_velocity1_d[in_idx]*p6;

					p3=p6;
				}

					dem_p1_d[in_idx]=(-1.0)*p3*vx_t_d[in_idx];
					dem_p2_d[in_idx]=(-1.0)*p3*vz_t_d[in_idx];

					dem_p3_d[in_idx]=(p1+2*p2)*vx_x_d[in_idx]+p1*vz_z_d[in_idx];
		
					dem_p4_d[in_idx]=(p1+2*p2)*vz_z_d[in_idx]+p1*vx_x_d[in_idx];

					dem_p5_d[in_idx]=p2*(vx_z_d[in_idx]+vz_x_d[in_idx]);				
		}
}
