///////////////QQQQQQQQQQQQQQQ2017年07月27日 星期四 10时11分40秒 
	
__global__ void save_and_set_wavefiled(float *vx2_d,float *save_vx_d,int nx,int nz,int nx_append,int nz_append,int boundary_left,int boundary_up,int mark)
//save_and_set_wavefiled<<<dimGrid,dimBlock>>>(vx2_d,&save_vx_d[it*nx*nz],nx,nz,nx_append,nz_append,boundary_left,boundary_up,0);
{

		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;
		int in_idx1;

		if((ix<nx)&&(iz<nz))
		{	
			in_idx = ix*nz+iz;
			in_idx1=(ix+boundary_left)*nz_append+iz+boundary_up;

			if(mark==0)	save_vx_d[in_idx]=vx2_d[in_idx1];

			if(mark==1)	vx2_d[in_idx1]=save_vx_d[in_idx];
		}
}
__global__ void cuda_cal_viscoelastic(float *modul_p_d,float *modul_s_d,float *qp_d,float *qs_d,float *tao_d,float *strain_p_d,float *strain_s_d,float freq,float *velocity_d,float *velocity1_d,float *density_d,int dimx,int dimz)
//cuda_cal_viscoelastic<<<dimGrid,dimBlock>>>(modul_p_d,modul_s_d,qp_d,qs_d,tao_d,strain_p_d,strain_s_d,freq,velocity_d,velocity1_d,density_d,nx_append,nz_append);
{
//////////////we note that  linear solid theroy  L=1
		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		float w;
		
		w=2*pai*freq;

		if((ix<dimx)&&(iz<dimz))
		{
				//dimx=dimx+2*radius;dimz=dimz+2*radius;
				//ix=ix+radius;iz=iz+radius;
				in_idx = ix*dimz+iz;//iz*dimx+ix;

				//if(qp_d[in_idx]>1088)		qp_d[in_idx]=qp_d[in_idx]*0.015;
				//if(qs_d[in_idx]>1088/sqrt(3.0))	qs_d[in_idx]=qs_d[in_idx]*0.015*sqrt(3.0);
				
				tao_d[in_idx]=1.0/w*(1.0*sqrt(1.0+(1.0/qp_d[in_idx]/qp_d[in_idx]))-1.0/qp_d[in_idx]);
				
				strain_p_d[in_idx]=1.0/(1.0*w*w*tao_d[in_idx]);

				strain_s_d[in_idx]=1.0*(1+1.0*w*tao_d[in_idx]*qs_d[in_idx])/(1.0*w*qs_d[in_idx]-1.0*w*w*tao_d[in_idx]);

				//modul_p_d[in_idx]=1.0*velocity_d[in_idx]*velocity_d[in_idx]*density_d[in_idx]*(strain_p_d[in_idx]*1.0/tao_d[in_idx]);

				//modul_s_d[in_idx]=1.0*velocity1_d[in_idx]*velocity1_d[in_idx]*density_d[in_idx]*(strain_s_d[in_idx]*1.0/tao_d[in_idx]);

				///in order to moditfy elastic program at least
				modul_p_d[in_idx]=1.0*velocity_d[in_idx]*velocity_d[in_idx]*(strain_p_d[in_idx]*1.0/tao_d[in_idx]);

				modul_s_d[in_idx]=1.0*velocity1_d[in_idx]*velocity1_d[in_idx]*(strain_s_d[in_idx]*1.0/tao_d[in_idx]);
		}
}


__global__ void fwd_txxzzxzpp_viscoelastic_and_memory(float *tp2,float *tp1,float *txx2,float *txx1,float *tzz2,float *tzz1,float *txz2,float *txz1,float *vx2,float *vz2,float *modul_p_d,float *modul_s_d,float *attenuation_d,float dt,float *coe_d,float coe_x,float coe_z,int dimx,int dimz,float *density_d,float *mem_p2_d,float *mem_p1_d,float *mem_xx2_d,float *mem_xx1_d,float *mem_zz2_d,float *mem_zz1_d,float *mem_xz2_d,float *mem_xz1_d,float *velocity_d,float *velocity1_d,float *tao_d,float *strain_p_d,float *strain_s_d)
//fwd_txxzzxzpp_viscoelastic_and_memory<<<dimGrid,dimBlock>>>(tp2_d,tp1_d,txx2_d,txx1_d,tzz2_d,tzz1_d,txz2_d,txz1_d,vx1_d,vz1_d,modul_p_d,modul_s_d,attenuation_d,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,density_d,mem_p2_d,mem_p1_d,mem_xx2_d,mem_xx1_d,mem_zz2_d,mem_zz1_d,mem_xz2_d,mem_xz1_d,velocity_d,velocity1_d,tao_d,strain_p_d,strain_s_d);
{
		__shared__ float s_data1[BDIMY+2*radius][BDIMX+2*radius];
		__shared__ float s_data2[BDIMY+2*radius][BDIMX+2*radius];
		float dt_real=dt/1000;
		float s_velocity;
		float s_velocity1;
	
		float pppp=0,ssss=0;
		
		//float s_velocity3;
		//float s_velocity4;
		float s_attenuation;

		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		int tx = threadIdx.x+radius;
		int tz = threadIdx.y+radius;
		s_data1[tz][tx]=0.0;
		s_data1[threadIdx.y][threadIdx.x]=0.0;
		s_data1[BDIMY+2*radius-1-threadIdx.y][BDIMX+2*radius-1-threadIdx.x]=0.0;
		s_data1[threadIdx.y][BDIMX+2*radius-1-threadIdx.x]=0.0;
		s_data1[BDIMY+2*radius-1-threadIdx.y][threadIdx.x]=0.0;

		s_data2[tz][tx]=0.0;
		s_data2[threadIdx.y][threadIdx.x]=0.0;
		s_data2[BDIMY+2*radius-1-threadIdx.y][BDIMX+2*radius-1-threadIdx.x]=0.0;
		s_data2[threadIdx.y][BDIMX+2*radius-1-threadIdx.x]=0.0;
		s_data2[BDIMY+2*radius-1-threadIdx.y][threadIdx.x]=0.0;

		if((ix<dimx)&&(iz<dimz))
		{
				dimx=dimx+2*radius;dimz=dimz+2*radius;
				ix=ix+radius;iz=iz+radius;
				in_idx = ix*dimz+iz;//iz*dimx+ix;

				__syncthreads();

				s_data1[tz][tx]=vx2[in_idx];
				s_data2[tz][tx]=vz2[in_idx];

				if(threadIdx.y<radius)
				{
						s_data1[threadIdx.y][tx]=vx2[in_idx-radius];//g_input[in_idx-radius*dimx];//up
						s_data1[threadIdx.y+BDIMY+radius][tx]=vx2[in_idx+BDIMY];//g_input[in_idx+BDIMY*dimx];//down

						s_data2[threadIdx.y][tx]=vz2[in_idx-radius];//g_input[in_idx-radius*dimx];//up
						s_data2[threadIdx.y+BDIMY+radius][tx]=vz2[in_idx+BDIMY];//g_input[in_idx+BDIMY*dimx];//down
				}
				if(threadIdx.x<radius)
				{
						s_data1[tz][threadIdx.x]=vx2[in_idx-radius*dimz];//g_input[in_idx-radius];//left
						s_data1[tz][threadIdx.x+BDIMX+radius]=vx2[in_idx+BDIMX*dimz];//g_input[in_idx+BDIMX];//right

						s_data2[tz][threadIdx.x]=vz2[in_idx-radius*dimz];//g_input[in_idx-radius];//left
						s_data2[tz][threadIdx.x+BDIMX+radius]=vz2[in_idx+BDIMX*dimz];//g_input[in_idx+BDIMX];//right
				}
				
				s_velocity=velocity_d[in_idx]*velocity_d[in_idx];
				s_velocity1=velocity1_d[in_idx]*velocity1_d[in_idx];

				//s_velocity=(velocity_d[in_idx]*velocity_d[in_idx]+velocity_d[in_idx+dimz]*velocity_d[in_idx+dimz])/2.0;
				//s_velocity1=(velocity1_d[in_idx]*velocity1_d[in_idx]+velocity1_d[in_idx+dimz]*velocity1_d[in_idx+dimz])/2.0;
				
				//s_velocity3=(velocity_d[in_idx]*velocity_d[in_idx]+velocity_d[in_idx+1]*velocity_d[in_idx+1])/2.0;
				//s_velocity4=(velocity1_d[in_idx]*velocity1_d[in_idx]+velocity1_d[in_idx+1]*velocity1_d[in_idx+1])/2.0;
				s_attenuation=attenuation_d[in_idx];
				__syncthreads();
				
				pppp=(1.0-(1.0*strain_p_d[in_idx]/tao_d[in_idx]));
				ssss=(1.0-(1.0*strain_s_d[in_idx]/tao_d[in_idx]));
				

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

				mem_p2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*mem_p1_d[in_idx]-1.0*dt_real*1.0/tao_d[in_idx]*mem_p1_d[in_idx]+s_velocity*density_d[in_idx]/tao_d[in_idx]*pppp*(sumx*coe_x+sumz*coe_z));//s_velocity  and  s_velocity1

				mem_xx2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*mem_xx1_d[in_idx]-1.0*dt_real*1.0/tao_d[in_idx]*mem_xx1_d[in_idx]+s_velocity*density_d[in_idx]/tao_d[in_idx]*pppp*(sumx*coe_x+sumz*coe_z)-2.0*s_velocity1*density_d[in_idx]/tao_d[in_idx]*ssss*sumz*coe_z);//s_velocity  and  s_velocity1

				mem_zz2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*mem_zz1_d[in_idx]-1.0*dt_real*1.0/tao_d[in_idx]*mem_zz1_d[in_idx]+s_velocity*density_d[in_idx]/tao_d[in_idx]*pppp*(sumx*coe_x+sumz*coe_z)-2.0*s_velocity1*density_d[in_idx]/tao_d[in_idx]*ssss*sumx*coe_x);//s_velocity  and  s_velocity1


				txx2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*txx1[in_idx]+modul_p_d[in_idx]*density_d[in_idx]*sumx*coe_x+(modul_p_d[in_idx]-2*modul_s_d[in_idx])*density_d[in_idx]*sumz*coe_z+dt_real*mem_xx2_d[in_idx]);//s_velocity  and  s_velocity1

				tzz2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*tzz1[in_idx]+(modul_p_d[in_idx]-2*modul_s_d[in_idx])*density_d[in_idx]*sumx*coe_x+modul_p_d[in_idx]*density_d[in_idx]*sumz*coe_z+dt_real*+mem_zz2_d[in_idx]);//sumx  and  sumz 
				
				tp2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*tp1[in_idx]+modul_p_d[in_idx]*density_d[in_idx]*sumx*coe_x+modul_p_d[in_idx]*density_d[in_idx]*sumz*coe_z+dt_real*mem_p2_d[in_idx]);	
					
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

				mem_xz2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*mem_xz1_d[in_idx]-1.0*dt_real*1.0/tao_d[in_idx]*mem_xz1_d[in_idx]+s_velocity1*density_d[in_idx]/tao_d[in_idx]*ssss*(sumx1*coe_x+sumz1*coe_z));//s_velocity  and  s_velocity1

				txz2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*txz1[in_idx]+modul_s_d[in_idx]*density_d[in_idx]*(sumx1*coe_x+sumz1*coe_z)+dt_real*mem_xz2_d[in_idx]);
		}
}

__global__ void fwd_txxzzxzpp_viscoelastic_and_memory_3parameterization(float *tp2_d,float *tp1_d,float *txx2_d,float *txx1_d,float *tzz2_d,float *tzz1_d,float *txz2_d,float *txz1_d,float *vx2_d,float *vz2_d,float *modul_p_d,float *modul_s_d,float *attenuation_d,float dt,float *coe_d,float coe_x,float coe_z,int dimx,int dimz,float *density_d,float *mem_p2_d,float *mem_p1_d,float *mem_xx2_d,float *mem_xx1_d,float *mem_zz2_d,float *mem_zz1_d,float *mem_xz2_d,float *mem_xz1_d,float *velocity_d,float *velocity1_d,float *tao_d,float *strain_p_d,float *strain_s_d)
//fwd_txxzzxzpp_viscoelastic_and_memory<<<dimGrid,dimBlock>>>(tp2_d,tp1_d,txx2_d,txx1_d,tzz2_d,tzz1_d,txz2_d,txz1_d,vx1_d,vz1_d,modul_p_d,modul_s_d,attenuation_d,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,density_d,mem_p2_d,mem_p1_d,mem_xx2_d,mem_xx1_d,mem_zz2_d,mem_zz1_d,mem_xz2_d,mem_xz1_d,velocity_d,velocity1_d,tao_d,strain_p_d,strain_s_d);
{
		__shared__ float s_data1[BDIMY+2*radius][BDIMX+2*radius];
		__shared__ float s_data2[BDIMY+2*radius][BDIMX+2*radius];
		float dt_real=dt/1000;
		float s_velocity;
		float s_velocity1;
	
		float pppp=0,ssss=0;
		float s_attenuation;

		float density,tao;

		float modul_p,modul_s;

		float mem_p1,mem_xx1,mem_zz1,mem_xz1;

		//float tp1;

		float txx1,tzz1,txz1;


		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		int tx = threadIdx.x+radius;
		int tz = threadIdx.y+radius;
		s_data1[tz][tx]=0.0;
		s_data1[threadIdx.y][threadIdx.x]=0.0;
		s_data1[BDIMY+2*radius-1-threadIdx.y][BDIMX+2*radius-1-threadIdx.x]=0.0;
		s_data1[threadIdx.y][BDIMX+2*radius-1-threadIdx.x]=0.0;
		s_data1[BDIMY+2*radius-1-threadIdx.y][threadIdx.x]=0.0;

		s_data2[tz][tx]=0.0;
		s_data2[threadIdx.y][threadIdx.x]=0.0;
		s_data2[BDIMY+2*radius-1-threadIdx.y][BDIMX+2*radius-1-threadIdx.x]=0.0;
		s_data2[threadIdx.y][BDIMX+2*radius-1-threadIdx.x]=0.0;
		s_data2[BDIMY+2*radius-1-threadIdx.y][threadIdx.x]=0.0;

		if((ix<dimx)&&(iz<dimz))
		{
				dimx=dimx+2*radius;dimz=dimz+2*radius;
				ix=ix+radius;iz=iz+radius;
				in_idx = ix*dimz+iz;//iz*dimx+ix;

				__syncthreads();

				s_data1[tz][tx]=vx2_d[in_idx];
				s_data2[tz][tx]=vz2_d[in_idx];

				if(threadIdx.y<radius)
				{
						s_data1[threadIdx.y][tx]=vx2_d[in_idx-radius];//g_input[in_idx-radius*dimx];//up
						s_data1[threadIdx.y+BDIMY+radius][tx]=vx2_d[in_idx+BDIMY];//g_input[in_idx+BDIMY*dimx];//down

						s_data2[threadIdx.y][tx]=vz2_d[in_idx-radius];//g_input[in_idx-radius*dimx];//up
						s_data2[threadIdx.y+BDIMY+radius][tx]=vz2_d[in_idx+BDIMY];//g_input[in_idx+BDIMY*dimx];//down
				}
				if(threadIdx.x<radius)
				{
						s_data1[tz][threadIdx.x]=vx2_d[in_idx-radius*dimz];//g_input[in_idx-radius];//left
						s_data1[tz][threadIdx.x+BDIMX+radius]=vx2_d[in_idx+BDIMX*dimz];//g_input[in_idx+BDIMX];//right

						s_data2[tz][threadIdx.x]=vz2_d[in_idx-radius*dimz];//g_input[in_idx-radius];//left
						s_data2[tz][threadIdx.x+BDIMX+radius]=vz2_d[in_idx+BDIMX*dimz];//g_input[in_idx+BDIMX];//right
				}
				
				s_velocity=velocity_d[in_idx]*velocity_d[in_idx];
				s_velocity1=velocity1_d[in_idx]*velocity1_d[in_idx];

				pppp=(1.0-(1.0*strain_p_d[in_idx]/tao_d[in_idx]));
				ssss=(1.0-(1.0*strain_s_d[in_idx]/tao_d[in_idx]));

				density=density_d[in_idx];

				tao=tao_d[in_idx];

				modul_p=modul_p_d[in_idx];
				modul_s=modul_s_d[in_idx];
		
				mem_p1=mem_p1_d[in_idx];
				mem_xx1=mem_xx1_d[in_idx];
				mem_zz1=mem_zz1_d[in_idx];
				mem_xz1=mem_xz1_d[in_idx];

				//tp1=tp1_d[in_idx];
				txx1=txx1_d[in_idx];
				tzz1=tzz1_d[in_idx];
				txz1=txz1_d[in_idx];

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

				mem_p2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*mem_p1-1.0*dt_real*1.0/tao*mem_p1+s_velocity*density/tao*pppp*(sumx*coe_x+sumz*coe_z));//s_velocity  and  s_velocity1

				mem_xx2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*mem_xx1-1.0*dt_real*1.0/tao*mem_xx1-2.0*s_velocity1*density/tao*ssss*sumz*coe_z);//s_velocity  and  s_velocity1

				mem_zz2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*mem_zz1-1.0*dt_real*1.0/tao*mem_zz1-2.0*s_velocity1*density/tao*ssss*sumx*coe_x);//s_velocity  and  s_velocity1


				txx2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*txx1+modul_p*density*sumx*coe_x+(modul_p-2*modul_s)*density*sumz*coe_z+dt_real*(mem_p2_d[in_idx]+mem_xx2_d[in_idx]));//s_velocity  and  s_velocity1

				tzz2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*tzz1+(modul_p-2*modul_s)*density*sumx*coe_x+modul_p*density*sumz*coe_z+dt_real*(mem_p2_d[in_idx]+mem_zz2_d[in_idx]));//sumx  and  sumz 
				
				//tp2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*tp1+modul_p*density*sumx*coe_x+modul_p*density*sumz*coe_z+dt_real*mem_p2_d[in_idx]);	
					
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

				mem_xz2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*mem_xz1-1.0*dt_real*1.0/tao*mem_xz1+s_velocity1*density/tao*ssss*(sumx1*coe_x+sumz1*coe_z));//s_velocity  and  s_velocity1

				txz2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*txz1+modul_s*density*(sumx1*coe_x+sumz1*coe_z)+dt_real*mem_xz2_d[in_idx]);
		}
}


__global__ void demig_fwd_txxzzxzpp_viscoelastic_and_memory_3parameterization(float *tp2_d,float *tp1_d,float *txx2_d,float *txx1_d,float *tzz2_d,float *tzz1_d,float *txz2_d,float *txz1_d,float *vx2_d,float *vz2_d,float *modul_p_d,float *modul_s_d,float *attenuation_d,float dt,float *coe_d,float coe_x,float coe_z,int dimx,int dimz,float *density_d,float *mem_p2_d,float *mem_p1_d,float *mem_xx2_d,float *mem_xx1_d,float *mem_zz2_d,float *mem_zz1_d,float *mem_xz2_d,float *mem_xz1_d,float *velocity_d,float *velocity1_d,float *tao_d,float *strain_p_d,float *strain_s_d,float *dem_p_all_d)
//demig_fwd_txxzzxzpp_viscoelastic_and_memory_3parameterization(float *tp2,float *tp1,float *txx2,float *txx1,float *tzz2,float *tzz1,float *txz2,float *txz1,float *vx2,float *vz2,float *modul_p_d,float *modul_s_d,float *attenuation_d,float dt,float *coe_d,float coe_x,float coe_z,int dimx,int dimz,float *density_d,float *mem_p2_d,float *mem_p1_d,float *mem_xx2_d,float *mem_xx1_d,float *mem_zz2_d,float *mem_zz1_d,float *mem_xz2_d,float *mem_xz1_d,float *velocity_d,float *velocity1_d,float *tao_d,float *strain_p_d,float *strain_s_d,float *dem_p_all_d)
{
		__shared__ float s_data1[BDIMY+2*radius][BDIMX+2*radius];
		__shared__ float s_data2[BDIMY+2*radius][BDIMX+2*radius];
		float dt_real=dt/1000;
		float s_velocity;
		float s_velocity1;
	
		float pppp=0,ssss=0;
		float s_attenuation;

		float density,tao;

		float modul_p,modul_s;

		float mem_p1,mem_xx1,mem_zz1,mem_xz1;

		//float tp1;

		float txx1,tzz1,txz1;

		float dem_p2,dem_p3,dem_p4,dem_p5,dem_p6,dem_p7;

		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		int tx = threadIdx.x+radius;
		int tz = threadIdx.y+radius;
		s_data1[tz][tx]=0.0;
		s_data1[threadIdx.y][threadIdx.x]=0.0;
		s_data1[BDIMY+2*radius-1-threadIdx.y][BDIMX+2*radius-1-threadIdx.x]=0.0;
		s_data1[threadIdx.y][BDIMX+2*radius-1-threadIdx.x]=0.0;
		s_data1[BDIMY+2*radius-1-threadIdx.y][threadIdx.x]=0.0;

		s_data2[tz][tx]=0.0;
		s_data2[threadIdx.y][threadIdx.x]=0.0;
		s_data2[BDIMY+2*radius-1-threadIdx.y][BDIMX+2*radius-1-threadIdx.x]=0.0;
		s_data2[threadIdx.y][BDIMX+2*radius-1-threadIdx.x]=0.0;
		s_data2[BDIMY+2*radius-1-threadIdx.y][threadIdx.x]=0.0;

		if((ix<dimx)&&(iz<dimz))
		{
				dimx=dimx+2*radius;dimz=dimz+2*radius;
				ix=ix+radius;iz=iz+radius;
				in_idx = ix*dimz+iz;//iz*dimx+ix;

				__syncthreads();

				s_data1[tz][tx]=vx2_d[in_idx];
				s_data2[tz][tx]=vz2_d[in_idx];

				if(threadIdx.y<radius)
				{
						s_data1[threadIdx.y][tx]=vx2_d[in_idx-radius];//g_input[in_idx-radius*dimx];//up
						s_data1[threadIdx.y+BDIMY+radius][tx]=vx2_d[in_idx+BDIMY];//g_input[in_idx+BDIMY*dimx];//down

						s_data2[threadIdx.y][tx]=vz2_d[in_idx-radius];//g_input[in_idx-radius*dimx];//up
						s_data2[threadIdx.y+BDIMY+radius][tx]=vz2_d[in_idx+BDIMY];//g_input[in_idx+BDIMY*dimx];//down
				}
				if(threadIdx.x<radius)
				{
						s_data1[tz][threadIdx.x]=vx2_d[in_idx-radius*dimz];//g_input[in_idx-radius];//left
						s_data1[tz][threadIdx.x+BDIMX+radius]=vx2_d[in_idx+BDIMX*dimz];//g_input[in_idx+BDIMX];//right

						s_data2[tz][threadIdx.x]=vz2_d[in_idx-radius*dimz];//g_input[in_idx-radius];//left
						s_data2[tz][threadIdx.x+BDIMX+radius]=vz2_d[in_idx+BDIMX*dimz];//g_input[in_idx+BDIMX];//right
				}
				
				s_velocity=velocity_d[in_idx]*velocity_d[in_idx];
				s_velocity1=velocity1_d[in_idx]*velocity1_d[in_idx];

				pppp=(1.0-(1.0*strain_p_d[in_idx]/tao_d[in_idx]));
				ssss=(1.0-(1.0*strain_s_d[in_idx]/tao_d[in_idx]));

				density=density_d[in_idx];

				tao=tao_d[in_idx];

				modul_p=modul_p_d[in_idx];
				modul_s=modul_s_d[in_idx];
		
				mem_p1=mem_p1_d[in_idx];
				mem_xx1=mem_xx1_d[in_idx];
				mem_zz1=mem_zz1_d[in_idx];
				mem_xz1=mem_xz1_d[in_idx];

				//tp1=tp1_d[in_idx];
				txx1=txx1_d[in_idx];
				tzz1=tzz1_d[in_idx];
				txz1=txz1_d[in_idx];

				dem_p2=dem_p_all_d[2*dimx*dimz+in_idx];
				dem_p3=dem_p_all_d[3*dimx*dimz+in_idx];
				dem_p4=dem_p_all_d[4*dimx*dimz+in_idx];
				dem_p5=dem_p_all_d[5*dimx*dimz+in_idx];
				dem_p6=dem_p_all_d[6*dimx*dimz+in_idx];
				dem_p7=dem_p_all_d[7*dimx*dimz+in_idx];

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

				mem_p2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*mem_p1-1.0*dt_real*1.0/tao*mem_p1+s_velocity*density/tao*pppp*(sumx*coe_x+sumz*coe_z));//s_velocity  and  s_velocity1

				mem_xx2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*mem_xx1-1.0*dt_real*1.0/tao*mem_xx1-2.0*s_velocity1*density/tao*ssss*sumz*coe_z+dem_p5);//s_velocity  and  s_velocity1

				mem_zz2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*mem_zz1-1.0*dt_real*1.0/tao*mem_zz1-2.0*s_velocity1*density/tao*ssss*sumx*coe_x+dem_p6);//s_velocity  and  s_velocity1


				txx2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*txx1+modul_p*density*sumx*coe_x+(modul_p-2*modul_s)*density*sumz*coe_z+dt_real*(mem_p2_d[in_idx]+mem_xx2_d[in_idx])+dem_p2);//s_velocity  and  s_velocity1

				tzz2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*tzz1+(modul_p-2*modul_s)*density*sumx*coe_x+modul_p*density*sumz*coe_z+dt_real*(mem_p2_d[in_idx]+mem_zz2_d[in_idx])+dem_p3);//sumx  and  sumz 
				
				//tp2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*tp1+modul_p*density*sumx*coe_x+modul_p*density*sumz*coe_z+dt_real*mem_p2_d[in_idx]);	
					
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

				mem_xz2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*mem_xz1-1.0*dt_real*1.0/tao*mem_xz1+s_velocity1*density/tao*ssss*(sumx1*coe_x+sumz1*coe_z)+dem_p7);//s_velocity  and  s_velocity1

				txz2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*txz1+modul_s*density*(sumx1*coe_x+sumz1*coe_z)+dt_real*mem_xz2_d[in_idx]+dem_p4);
		}
}


/*__global__ void demig_fwd_txxzzxzpp_viscoelastic_and_memory_3parameterization(float *tp2,float *tp1,float *txx2,float *txx1,float *tzz2,float *tzz1,float *txz2,float *txz1,float *vx2,float *vz2,float *modul_p_d,float *modul_s_d,float *attenuation_d,float dt,float *coe_d,float coe_x,float coe_z,int dimx,int dimz,float *density_d,float *mem_p2_d,float *mem_p1_d,float *mem_xx2_d,float *mem_xx1_d,float *mem_zz2_d,float *mem_zz1_d,float *mem_xz2_d,float *mem_xz1_d,float *velocity_d,float *velocity1_d,float *tao_d,float *strain_p_d,float *strain_s_d,float *dem_p_all_d)
//fwd_txxzzxzpp_viscoelastic_and_memory<<<dimGrid,dimBlock>>>(tp2_d,tp1_d,txx2_d,txx1_d,tzz2_d,tzz1_d,txz2_d,txz1_d,vx1_d,vz1_d,modul_p_d,modul_s_d,attenuation_d,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,density_d,mem_p2_d,mem_p1_d,mem_xx2_d,mem_xx1_d,mem_zz2_d,mem_zz1_d,mem_xz2_d,mem_xz1_d,velocity_d,velocity1_d,tao_d,strain_p_d,strain_s_d);
{
		__shared__ float s_data1[BDIMY+2*radius][BDIMX+2*radius];
		__shared__ float s_data2[BDIMY+2*radius][BDIMX+2*radius];
		float dt_real=dt/1000;
		float s_velocity;
		float s_velocity1;
	
		float pppp=0,ssss=0;
		
		//float s_velocity3;
		//float s_velocity4;
		float s_attenuation;

		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		int tx = threadIdx.x+radius;
		int tz = threadIdx.y+radius;
		s_data1[tz][tx]=0.0;
		s_data1[threadIdx.y][threadIdx.x]=0.0;
		s_data1[BDIMY+2*radius-1-threadIdx.y][BDIMX+2*radius-1-threadIdx.x]=0.0;
		s_data1[threadIdx.y][BDIMX+2*radius-1-threadIdx.x]=0.0;
		s_data1[BDIMY+2*radius-1-threadIdx.y][threadIdx.x]=0.0;

		s_data2[tz][tx]=0.0;
		s_data2[threadIdx.y][threadIdx.x]=0.0;
		s_data2[BDIMY+2*radius-1-threadIdx.y][BDIMX+2*radius-1-threadIdx.x]=0.0;
		s_data2[threadIdx.y][BDIMX+2*radius-1-threadIdx.x]=0.0;
		s_data2[BDIMY+2*radius-1-threadIdx.y][threadIdx.x]=0.0;

		if((ix<dimx)&&(iz<dimz))
		{
				dimx=dimx+2*radius;dimz=dimz+2*radius;
				ix=ix+radius;iz=iz+radius;
				in_idx = ix*dimz+iz;//iz*dimx+ix;

				__syncthreads();

				s_data1[tz][tx]=vx2[in_idx];
				s_data2[tz][tx]=vz2[in_idx];

				if(threadIdx.y<radius)
				{
						s_data1[threadIdx.y][tx]=vx2[in_idx-radius];//g_input[in_idx-radius*dimx];//up
						s_data1[threadIdx.y+BDIMY+radius][tx]=vx2[in_idx+BDIMY];//g_input[in_idx+BDIMY*dimx];//down

						s_data2[threadIdx.y][tx]=vz2[in_idx-radius];//g_input[in_idx-radius*dimx];//up
						s_data2[threadIdx.y+BDIMY+radius][tx]=vz2[in_idx+BDIMY];//g_input[in_idx+BDIMY*dimx];//down
				}
				if(threadIdx.x<radius)
				{
						s_data1[tz][threadIdx.x]=vx2[in_idx-radius*dimz];//g_input[in_idx-radius];//left
						s_data1[tz][threadIdx.x+BDIMX+radius]=vx2[in_idx+BDIMX*dimz];//g_input[in_idx+BDIMX];//right

						s_data2[tz][threadIdx.x]=vz2[in_idx-radius*dimz];//g_input[in_idx-radius];//left
						s_data2[tz][threadIdx.x+BDIMX+radius]=vz2[in_idx+BDIMX*dimz];//g_input[in_idx+BDIMX];//right
				}
				
				s_velocity=velocity_d[in_idx]*velocity_d[in_idx];
				s_velocity1=velocity1_d[in_idx]*velocity1_d[in_idx];

				//s_velocity=(velocity_d[in_idx]*velocity_d[in_idx]+velocity_d[in_idx+dimz]*velocity_d[in_idx+dimz])/2.0;
				//s_velocity1=(velocity1_d[in_idx]*velocity1_d[in_idx]+velocity1_d[in_idx+dimz]*velocity1_d[in_idx+dimz])/2.0;
				
				//s_velocity3=(velocity_d[in_idx]*velocity_d[in_idx]+velocity_d[in_idx+1]*velocity_d[in_idx+1])/2.0;
				//s_velocity4=(velocity1_d[in_idx]*velocity1_d[in_idx]+velocity1_d[in_idx+1]*velocity1_d[in_idx+1])/2.0;
				s_attenuation=attenuation_d[in_idx];
				__syncthreads();
				
				pppp=(1.0-(1.0*strain_p_d[in_idx]/tao_d[in_idx]));
				ssss=(1.0-(1.0*strain_s_d[in_idx]/tao_d[in_idx]));
				

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

				//mem_p2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*mem_p1_d[in_idx]-1.0*dt_real*1.0/tao_d[in_idx]*mem_p1_d[in_idx]+s_velocity*density_d[in_idx]/tao_d[in_idx]*pppp*(sumx*coe_x+sumz*coe_z));//s_velocity  and  s_velocity1

				mem_xx2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*mem_xx1_d[in_idx]-1.0*dt_real*1.0/tao_d[in_idx]*mem_xx1_d[in_idx]-2.0*s_velocity1*density_d[in_idx]/tao_d[in_idx]*ssss*sumz*coe_z+dem_p_all_d[5*dimx*dimz+in_idx]);//s_velocity  and  s_velocity1

				mem_zz2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*mem_zz1_d[in_idx]-1.0*dt_real*1.0/tao_d[in_idx]*mem_zz1_d[in_idx]-2.0*s_velocity1*density_d[in_idx]/tao_d[in_idx]*ssss*sumx*coe_x+dem_p_all_d[6*dimx*dimz+in_idx]);//s_velocity  and  s_velocity1


				txx2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*txx1[in_idx]+modul_p_d[in_idx]*density_d[in_idx]*sumx*coe_x+(modul_p_d[in_idx]-2*modul_s_d[in_idx])*density_d[in_idx]*sumz*coe_z+dt_real*(mem_p2_d[in_idx]+mem_xx2_d[in_idx])+dem_p_all_d[2*dimx*dimz+in_idx]);//s_velocity  and  s_velocity1

				tzz2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*tzz1[in_idx]+(modul_p_d[in_idx]-2*modul_s_d[in_idx])*density_d[in_idx]*sumx*coe_x+modul_p_d[in_idx]*density_d[in_idx]*sumz*coe_z+dt_real*(mem_p2_d[in_idx]+mem_zz2_d[in_idx])+dem_p_all_d[3*dimx*dimz+in_idx]);//sumx  and  sumz 
				
				//tp2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*tp1[in_idx]+modul_p_d[in_idx]*density_d[in_idx]*sumx*coe_x+modul_p_d[in_idx]*density_d[in_idx]*sumz*coe_z+dt_real*mem_p2_d[in_idx]);	
					
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

				mem_xz2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*mem_xz1_d[in_idx]-1.0*dt_real*1.0/tao_d[in_idx]*mem_xz1_d[in_idx]+s_velocity1*density_d[in_idx]/tao_d[in_idx]*ssss*(sumx1*coe_x+sumz1*coe_z)+dem_p_all_d[7*dimx*dimz+in_idx]);//s_velocity  and  s_velocity1

				txz2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*txz1[in_idx]+modul_s_d[in_idx]*density_d[in_idx]*(sumx1*coe_x+sumz1*coe_z)+dt_real*mem_xz2_d[in_idx]+dem_p_all_d[4*dimx*dimz+in_idx]);
		}
}*/

__global__ void cuda_packaging(float *packaging_d,float dx,float dz,float dt,float coe_x,float coe_z,int dimx,int dimz,float *coe_d)
{
		int ix;
		packaging_d[0]=dx;
		packaging_d[1]=dz;
		packaging_d[2]=dt;
		packaging_d[3]=coe_x;
		packaging_d[4]=coe_z;
		packaging_d[5]=dimx;
		packaging_d[6]=dimz;

		for(ix=0;ix<radius+1;ix++)
		{
			packaging_d[7+ix]=coe_d[ix];
		}
}

//float *vx_x_d,float *vx_z_d,float *vz_x_d,float *vz_z_d,
__global__ void fwd_txxzzxzpp_viscoelastic_and_memory_3parameterization_new(float *vx_x_d,float *vx_z_d,float *vz_x_d,float *vz_z_d,float *tp2_d,float *tp1_d,float *txx2_d,float *txx1_d,float *tzz2_d,float *tzz1_d,float *txz2_d,float *txz1_d,float *vx2_d,float *vz2_d,float *modul_p_d,float *modul_s_d,float *attenuation_d,float *density_d,float *mem_p2_d,float *mem_p1_d,float *mem_xx2_d,float *mem_xx1_d,float *mem_zz2_d,float *mem_zz1_d,float *mem_xz2_d,float *mem_xz1_d,float *velocity_d,float *velocity1_d,float *tao_d,float *strain_p_d,float *strain_s_d,float *packaging_d)
//__global__ void fwd_txxzzxzpp_viscoelastic_and_memory_3parameterization_new<<<dimGrid,dimBlock>>>(vx_x_d,vx_z_d,vz_x_d,vz_z_d,tp2_d,tp1_d,txx2_d,txx1_d,tzz2_d,tzz1_d,txz2_d,txz1_d,vx2_d,vz2_d,modul_p_d,modul_s_d,attenuation_d,s_density_d,mem_p2_d,mem_p1_d,mem_xx2_d,mem_xx1_d,mem_zz2_d,mem_zz1_d,mem_xz2_d,mem_xz1_d,s_velocity_d,s_velocity1_d,tao_d,strain_p_d,strain_s_d,packaging_d);
{
		__shared__ float s_data1[BDIMY+2*radius][BDIMX+2*radius];
		__shared__ float s_data2[BDIMY+2*radius][BDIMX+2*radius];

		__shared__ float coe_d[radius+1];

		float dx,dz,dt,coe_x,coe_z;
		int ir,dimx,dimz;
		
		dx=packaging_d[0];
		dz=packaging_d[1];
		dt=packaging_d[2];
		coe_x=packaging_d[3];
		coe_z=packaging_d[4];
		dimx=packaging_d[5];
		dimz=packaging_d[6];

		for(ir=0;ir<radius+1;ir++)
		{
			coe_d[ir]=packaging_d[7+ir];
		}

		float dt_real=dt/1000;
		float s_velocity;
		float s_velocity1;
	
		float pppp=0,ssss=0;
		float s_attenuation;

		float density,tao;

		float modul_p,modul_s;

		float mem_p1,mem_xx1,mem_zz1,mem_xz1;

		float tp1,txx1,tzz1,txz1;


		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		int tx = threadIdx.x+radius;
		int tz = threadIdx.y+radius;
		s_data1[tz][tx]=0.0;
		s_data1[threadIdx.y][threadIdx.x]=0.0;
		s_data1[BDIMY+2*radius-1-threadIdx.y][BDIMX+2*radius-1-threadIdx.x]=0.0;
		s_data1[threadIdx.y][BDIMX+2*radius-1-threadIdx.x]=0.0;
		s_data1[BDIMY+2*radius-1-threadIdx.y][threadIdx.x]=0.0;

		s_data2[tz][tx]=0.0;
		s_data2[threadIdx.y][threadIdx.x]=0.0;
		s_data2[BDIMY+2*radius-1-threadIdx.y][BDIMX+2*radius-1-threadIdx.x]=0.0;
		s_data2[threadIdx.y][BDIMX+2*radius-1-threadIdx.x]=0.0;
		s_data2[BDIMY+2*radius-1-threadIdx.y][threadIdx.x]=0.0;

		if((ix<dimx)&&(iz<dimz))
		{
				dimx=dimx+2*radius;dimz=dimz+2*radius;
				ix=ix+radius;iz=iz+radius;
				in_idx = ix*dimz+iz;//iz*dimx+ix;

				__syncthreads();

				s_data1[tz][tx]=vx2_d[in_idx];
				s_data2[tz][tx]=vz2_d[in_idx];

				if(threadIdx.y<radius)
				{
						s_data1[threadIdx.y][tx]=vx2_d[in_idx-radius];//g_input[in_idx-radius*dimx];//up
						s_data1[threadIdx.y+BDIMY+radius][tx]=vx2_d[in_idx+BDIMY];//g_input[in_idx+BDIMY*dimx];//down

						s_data2[threadIdx.y][tx]=vz2_d[in_idx-radius];//g_input[in_idx-radius*dimx];//up
						s_data2[threadIdx.y+BDIMY+radius][tx]=vz2_d[in_idx+BDIMY];//g_input[in_idx+BDIMY*dimx];//down
				}
				if(threadIdx.x<radius)
				{
						s_data1[tz][threadIdx.x]=vx2_d[in_idx-radius*dimz];//g_input[in_idx-radius];//left
						s_data1[tz][threadIdx.x+BDIMX+radius]=vx2_d[in_idx+BDIMX*dimz];//g_input[in_idx+BDIMX];//right

						s_data2[tz][threadIdx.x]=vz2_d[in_idx-radius*dimz];//g_input[in_idx-radius];//left
						s_data2[tz][threadIdx.x+BDIMX+radius]=vz2_d[in_idx+BDIMX*dimz];//g_input[in_idx+BDIMX];//right
				}
				
				s_velocity=velocity_d[in_idx]*velocity_d[in_idx];
				s_velocity1=velocity1_d[in_idx]*velocity1_d[in_idx];

				pppp=(1.0-(1.0*strain_p_d[in_idx]/tao_d[in_idx]));
				ssss=(1.0-(1.0*strain_s_d[in_idx]/tao_d[in_idx]));

				density=density_d[in_idx];

				tao=tao_d[in_idx];

				modul_p=modul_p_d[in_idx];
				modul_s=modul_s_d[in_idx];
		
				mem_p1=mem_p1_d[in_idx];
				mem_xx1=mem_xx1_d[in_idx];
				mem_zz1=mem_zz1_d[in_idx];
				mem_xz1=mem_xz1_d[in_idx];

				tp1=tp1_d[in_idx];
				txx1=txx1_d[in_idx];
				tzz1=tzz1_d[in_idx];
				txz1=txz1_d[in_idx];

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

				vx_x_d[in_idx]=sumx*1.0/dx;

				vz_z_d[in_idx]=sumz*1.0/dz;

				mem_p2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*mem_p1-1.0*dt_real*1.0/tao*mem_p1+s_velocity*density/tao*pppp*(sumx*coe_x+sumz*coe_z));//s_velocity  and  s_velocity1

				mem_xx2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*mem_xx1-1.0*dt_real*1.0/tao*mem_xx1-2.0*s_velocity1*density/tao*ssss*sumz*coe_z);//s_velocity  and  s_velocity1

				mem_zz2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*mem_zz1-1.0*dt_real*1.0/tao*mem_zz1-2.0*s_velocity1*density/tao*ssss*sumx*coe_x);//s_velocity  and  s_velocity1


				txx2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*txx1+modul_p*density*sumx*coe_x+(modul_p-2*modul_s)*density*sumz*coe_z+dt_real*(mem_p2_d[in_idx]+mem_xx2_d[in_idx]));//s_velocity  and  s_velocity1

				tzz2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*tzz1+(modul_p-2*modul_s)*density*sumx*coe_x+modul_p*density*sumz*coe_z+dt_real*(mem_p2_d[in_idx]+mem_zz2_d[in_idx]));//sumx  and  sumz 
				
				tp2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*tp1+modul_p*density*sumx*coe_x+modul_p*density*sumz*coe_z+dt_real*mem_p2_d[in_idx]);	
					
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

				/////vx_z_d[in_idx]=sumx1*1.0/dz;//This is a fault, which leads to the distortion of the graident of the vs

				////vz_x_d[in_idx]=sumz1*1.0/dx;//This is a fault, which leads to the distortion of the graident of the vs

				vx_z_d[in_idx]=sumz1*1.0/dz;

				vz_x_d[in_idx]=sumx1*1.0/dx;

				mem_xz2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*mem_xz1-1.0*dt_real*1.0/tao*mem_xz1+s_velocity1*density/tao*ssss*(sumx1*coe_x+sumz1*coe_z));//s_velocity  and  s_velocity1

				txz2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*txz1+modul_s*density*(sumx1*coe_x+sumz1*coe_z)+dt_real*mem_xz2_d[in_idx]);
		}
}


//float *vx_x_d,float *vx_z_d,float *vz_x_d,float *vz_z_d,
/*__global__ void fwd_txxzzxzpp_viscoelastic_and_memory_3parameterization_new(float *vx_x_d,float *vx_z_d,float *vz_x_d,float *vz_z_d,float *tp2,float *tp1,float *txx2,float *txx1,float *tzz2,float *tzz1,float *txz2,float *txz1,float *vx2,float *vz2,float *modul_p_d,float *modul_s_d,float *attenuation_d,float *density_d,float *mem_p2_d,float *mem_p1_d,float *mem_xx2_d,float *mem_xx1_d,float *mem_zz2_d,float *mem_zz1_d,float *mem_xz2_d,float *mem_xz1_d,float *velocity_d,float *velocity1_d,float *tao_d,float *strain_p_d,float *strain_s_d,float *packaging_d)
//fwd_txxzzxzpp_viscoelastic_and_memory<<<dimGrid,dimBlock>>>(tp2_d,tp1_d,txx2_d,txx1_d,tzz2_d,tzz1_d,txz2_d,txz1_d,vx1_d,vz1_d,modul_p_d,modul_s_d,attenuation_d,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,density_d,mem_p2_d,mem_p1_d,mem_xx2_d,mem_xx1_d,mem_zz2_d,mem_zz1_d,mem_xz2_d,mem_xz1_d,velocity_d,velocity1_d,tao_d,strain_p_d,strain_s_d);
{
		__shared__ float s_data1[BDIMY+2*radius][BDIMX+2*radius];
		__shared__ float s_data2[BDIMY+2*radius][BDIMX+2*radius];
		__shared__ float coe_d[radius+1];

		float dx,dz,dt,coe_x,coe_z;
		int ir,dimx,dimz;
		
		dx=packaging_d[0];
		dz=packaging_d[1];
		dt=packaging_d[2];
		coe_x=packaging_d[3];
		coe_z=packaging_d[4];
		dimx=packaging_d[5];
		dimz=packaging_d[6];

		for(ir=0;ir<radius+1;ir++)
		{
			coe_d[ir]=packaging_d[7+ir];
		}
		
		float dt_real=dt/1000;
		float s_velocity;
		float s_velocity1;
	
		float pppp=0,ssss=0;
		
		//float s_velocity3;
		//float s_velocity4;
		float s_attenuation;

		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		int tx = threadIdx.x+radius;
		int tz = threadIdx.y+radius;
		s_data1[tz][tx]=0.0;
		s_data1[threadIdx.y][threadIdx.x]=0.0;
		s_data1[BDIMY+2*radius-1-threadIdx.y][BDIMX+2*radius-1-threadIdx.x]=0.0;
		s_data1[threadIdx.y][BDIMX+2*radius-1-threadIdx.x]=0.0;
		s_data1[BDIMY+2*radius-1-threadIdx.y][threadIdx.x]=0.0;

		s_data2[tz][tx]=0.0;
		s_data2[threadIdx.y][threadIdx.x]=0.0;
		s_data2[BDIMY+2*radius-1-threadIdx.y][BDIMX+2*radius-1-threadIdx.x]=0.0;
		s_data2[threadIdx.y][BDIMX+2*radius-1-threadIdx.x]=0.0;
		s_data2[BDIMY+2*radius-1-threadIdx.y][threadIdx.x]=0.0;

		if((ix<dimx)&&(iz<dimz))
		{
				dimx=dimx+2*radius;dimz=dimz+2*radius;
				ix=ix+radius;iz=iz+radius;
				in_idx = ix*dimz+iz;//iz*dimx+ix;

				__syncthreads();

				s_data1[tz][tx]=vx2[in_idx];
				s_data2[tz][tx]=vz2[in_idx];

				if(threadIdx.y<radius)
				{
						s_data1[threadIdx.y][tx]=vx2[in_idx-radius];//g_input[in_idx-radius*dimx];//up
						s_data1[threadIdx.y+BDIMY+radius][tx]=vx2[in_idx+BDIMY];//g_input[in_idx+BDIMY*dimx];//down

						s_data2[threadIdx.y][tx]=vz2[in_idx-radius];//g_input[in_idx-radius*dimx];//up
						s_data2[threadIdx.y+BDIMY+radius][tx]=vz2[in_idx+BDIMY];//g_input[in_idx+BDIMY*dimx];//down
				}
				if(threadIdx.x<radius)
				{
						s_data1[tz][threadIdx.x]=vx2[in_idx-radius*dimz];//g_input[in_idx-radius];//left
						s_data1[tz][threadIdx.x+BDIMX+radius]=vx2[in_idx+BDIMX*dimz];//g_input[in_idx+BDIMX];//right

						s_data2[tz][threadIdx.x]=vz2[in_idx-radius*dimz];//g_input[in_idx-radius];//left
						s_data2[tz][threadIdx.x+BDIMX+radius]=vz2[in_idx+BDIMX*dimz];//g_input[in_idx+BDIMX];//right
				}
				
				s_velocity=velocity_d[in_idx]*velocity_d[in_idx];
				s_velocity1=velocity1_d[in_idx]*velocity1_d[in_idx];

				//s_velocity=(velocity_d[in_idx]*velocity_d[in_idx]+velocity_d[in_idx+dimz]*velocity_d[in_idx+dimz])/2.0;
				//s_velocity1=(velocity1_d[in_idx]*velocity1_d[in_idx]+velocity1_d[in_idx+dimz]*velocity1_d[in_idx+dimz])/2.0;
				
				//s_velocity3=(velocity_d[in_idx]*velocity_d[in_idx]+velocity_d[in_idx+1]*velocity_d[in_idx+1])/2.0;
				//s_velocity4=(velocity1_d[in_idx]*velocity1_d[in_idx]+velocity1_d[in_idx+1]*velocity1_d[in_idx+1])/2.0;
				s_attenuation=attenuation_d[in_idx];
				__syncthreads();
				
				pppp=(1.0-(1.0*strain_p_d[in_idx]/tao_d[in_idx]));
				ssss=(1.0-(1.0*strain_s_d[in_idx]/tao_d[in_idx]));
				

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

				vx_x_d[in_idx]=sumx*1.0/dx;

				vz_z_d[in_idx]=sumz*1.0/dz;

				mem_p2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*mem_p1_d[in_idx]-1.0*dt_real*1.0/tao_d[in_idx]*mem_p1_d[in_idx]+s_velocity*density_d[in_idx]/tao_d[in_idx]*pppp*(sumx*coe_x+sumz*coe_z));//s_velocity  and  s_velocity1

				mem_xx2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*mem_xx1_d[in_idx]-1.0*dt_real*1.0/tao_d[in_idx]*mem_xx1_d[in_idx]-2.0*s_velocity1*density_d[in_idx]/tao_d[in_idx]*ssss*sumz*coe_z);//s_velocity  and  s_velocity1

				mem_zz2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*mem_zz1_d[in_idx]-1.0*dt_real*1.0/tao_d[in_idx]*mem_zz1_d[in_idx]-2.0*s_velocity1*density_d[in_idx]/tao_d[in_idx]*ssss*sumx*coe_x);//s_velocity  and  s_velocity1


				txx2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*txx1[in_idx]+modul_p_d[in_idx]*density_d[in_idx]*sumx*coe_x+(modul_p_d[in_idx]-2*modul_s_d[in_idx])*density_d[in_idx]*sumz*coe_z+dt_real*(mem_p2_d[in_idx]+mem_xx2_d[in_idx]));//s_velocity  and  s_velocity1

				tzz2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*tzz1[in_idx]+(modul_p_d[in_idx]-2*modul_s_d[in_idx])*density_d[in_idx]*sumx*coe_x+modul_p_d[in_idx]*density_d[in_idx]*sumz*coe_z+dt_real*(mem_p2_d[in_idx]+mem_zz2_d[in_idx]));//sumx  and  sumz 
				
				tp2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*tp1[in_idx]+modul_p_d[in_idx]*density_d[in_idx]*sumx*coe_x+modul_p_d[in_idx]*density_d[in_idx]*sumz*coe_z+dt_real*mem_p2_d[in_idx]);	
					
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

				/////vx_z_d[in_idx]=sumx1*1.0/dz;//This is a fault, which leads to the distortion of the graident of the vs

				////vz_x_d[in_idx]=sumz1*1.0/dx;//This is a fault, which leads to the distortion of the graident of the vs

				vx_z_d[in_idx]=sumz1*1.0/dz;

				vz_x_d[in_idx]=sumx1*1.0/dx;

				mem_xz2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*mem_xz1_d[in_idx]-1.0*dt_real*1.0/tao_d[in_idx]*mem_xz1_d[in_idx]+s_velocity1*density_d[in_idx]/tao_d[in_idx]*ssss*(sumx1*coe_x+sumz1*coe_z));//s_velocity  and  s_velocity1

				txz2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*txz1[in_idx]+modul_s_d[in_idx]*density_d[in_idx]*(sumx1*coe_x+sumz1*coe_z)+dt_real*mem_xz2_d[in_idx]);
		}
}*/

__global__ void fwd_txxzzxzpp_viscoelastic(float *tp2,float *tp1,float *txx2,float *txx1,float *tzz2,float *tzz1,float *txz2,float *txz1,float *vx2,float *vz2,float *velocity_d,float *velocity1_d,float *attenuation_d,float dt,float *coe_d,float coe_x,float coe_z,int dimx,int dimz,float *density_d,float *mem_p1_d,float *mem_xx1_d,float *mem_zz1_d,float *mem_xz1_d)
//fwd_txxzzxzpp_viscoelastic<<<dimGrid,dimBlock>>>(tp2_d,tp1_d,txx2_d,txx1_d,tzz2_d,tzz1_d,txz2_d,txz1_d,vx1_d,vz1_d,modul_p_d,modul_s_d,attenuation_d,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,density_d,mem_p1_d,mem_xx1_d,mem_zz1_d,mem_xz1_d);
{
		__shared__ float s_data1[BDIMY+2*radius][BDIMX+2*radius];
		__shared__ float s_data2[BDIMY+2*radius][BDIMX+2*radius];
		float dt_real=dt/1000;
		float s_velocity;
		float s_velocity1;
	
		
		//float s_velocity3;
		//float s_velocity4;
		float s_attenuation;

		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		int tx = threadIdx.x+radius;
		int tz = threadIdx.y+radius;
		s_data1[tz][tx]=0.0;
		s_data1[threadIdx.y][threadIdx.x]=0.0;
		s_data1[BDIMY+2*radius-1-threadIdx.y][BDIMX+2*radius-1-threadIdx.x]=0.0;
		s_data1[threadIdx.y][BDIMX+2*radius-1-threadIdx.x]=0.0;
		s_data1[BDIMY+2*radius-1-threadIdx.y][threadIdx.x]=0.0;

		s_data2[tz][tx]=0.0;
		s_data2[threadIdx.y][threadIdx.x]=0.0;
		s_data2[BDIMY+2*radius-1-threadIdx.y][BDIMX+2*radius-1-threadIdx.x]=0.0;
		s_data2[threadIdx.y][BDIMX+2*radius-1-threadIdx.x]=0.0;
		s_data2[BDIMY+2*radius-1-threadIdx.y][threadIdx.x]=0.0;

		if((ix<dimx)&&(iz<dimz))
		{
				dimx=dimx+2*radius;dimz=dimz+2*radius;
				ix=ix+radius;iz=iz+radius;
				in_idx = ix*dimz+iz;//iz*dimx+ix;

				__syncthreads();

				s_data1[tz][tx]=vx2[in_idx];
				s_data2[tz][tx]=vz2[in_idx];

				if(threadIdx.y<radius)
				{
						s_data1[threadIdx.y][tx]=vx2[in_idx-radius];//g_input[in_idx-radius*dimx];//up
						s_data1[threadIdx.y+BDIMY+radius][tx]=vx2[in_idx+BDIMY];//g_input[in_idx+BDIMY*dimx];//down

						s_data2[threadIdx.y][tx]=vz2[in_idx-radius];//g_input[in_idx-radius*dimx];//up
						s_data2[threadIdx.y+BDIMY+radius][tx]=vz2[in_idx+BDIMY];//g_input[in_idx+BDIMY*dimx];//down
				}
				if(threadIdx.x<radius)
				{
						s_data1[tz][threadIdx.x]=vx2[in_idx-radius*dimz];//g_input[in_idx-radius];//left
						s_data1[tz][threadIdx.x+BDIMX+radius]=vx2[in_idx+BDIMX*dimz];//g_input[in_idx+BDIMX];//right

						s_data2[tz][threadIdx.x]=vz2[in_idx-radius*dimz];//g_input[in_idx-radius];//left
						s_data2[tz][threadIdx.x+BDIMX+radius]=vz2[in_idx+BDIMX*dimz];//g_input[in_idx+BDIMX];//right
				}
				
				//s_velocity=velocity_d[in_idx]*velocity_d[in_idx];
				//s_velocity1=velocity1_d[in_idx]*velocity1_d[in_idx];

				s_velocity=velocity_d[in_idx];
				s_velocity1=velocity1_d[in_idx];//2017年07月27日 星期四 14时40分31秒 

				//s_velocity=(velocity_d[in_idx]*velocity_d[in_idx]+velocity_d[in_idx+dimz]*velocity_d[in_idx+dimz])/2.0;
				//s_velocity1=(velocity1_d[in_idx]*velocity1_d[in_idx]+velocity1_d[in_idx+dimz]*velocity1_d[in_idx+dimz])/2.0;
				
				//s_velocity3=(velocity_d[in_idx]*velocity_d[in_idx]+velocity_d[in_idx+1]*velocity_d[in_idx+1])/2.0;
				//s_velocity4=(velocity1_d[in_idx]*velocity1_d[in_idx]+velocity1_d[in_idx+1]*velocity1_d[in_idx+1])/2.0;
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


				txx2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*txx1[in_idx]+s_velocity*density_d[in_idx]*sumx*coe_x+(s_velocity-2*s_velocity1)*density_d[in_idx]*sumz*coe_z+dt_real*(mem_p1_d[in_idx]+mem_xx1_d[in_idx]));//s_velocity  and  s_velocity1

				tzz2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*tzz1[in_idx]+(s_velocity-2*s_velocity1)*density_d[in_idx]*sumx*coe_x+s_velocity*density_d[in_idx]*sumz*coe_z+dt_real*(mem_p1_d[in_idx]+mem_zz1_d[in_idx]));//sumx  and  sumz 
				
				tp2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*tp1[in_idx]+s_velocity*density_d[in_idx]*sumx*coe_x+s_velocity*density_d[in_idx]*sumz*coe_z+dt_real*mem_p1_d[in_idx]);	
					
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

				txz2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*txz1[in_idx]+s_velocity1*density_d[in_idx]*(sumx1*coe_x+sumz1*coe_z)+dt_real*mem_xz1_d[in_idx]);
		}
}

__global__ void fwd_memory(float *mem_p2_d,float *mem_p1_d,float *mem_xx2_d,float *mem_xx1_d,float *mem_zz2_d,float *mem_zz1_d,float *mem_xz2_d,float *mem_xz1_d,float *vx2,float *vz2,float *attenuation_d,float dx,float dz,float dt,float *coe_d,float coe_x,float coe_z,int dimx,int dimz,float *density_d,float *velocity_d,float *velocity1_d,float *tao_d,float *strain_p_d,float *strain_s_d)
//fwd_memory<<<dimGrid,dimBlock>>>(mem_p2_d,mem_p1_d,mem_xx2_d,mem_xx1_d,mem_zz2_d,mem_zz1_d,mem_xz2_d,mem_xz1_d,vx2_d,vz2_d,attenuation_d,dx,dz,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,density_d,velocity_d,velocity1_d,tao_d,strain_p_d,strain_s_d);
{
		__shared__ float s_data1[BDIMY+2*radius][BDIMX+2*radius];
		__shared__ float s_data2[BDIMY+2*radius][BDIMX+2*radius];
		float dt_real=dt/1000;
		float s_velocity;
		float s_velocity1;
		float pppp=0,ssss=0;
		
		float density;

		float tao;

		float s_attenuation;

		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		int tx = threadIdx.x+radius;
		int tz = threadIdx.y+radius;
		s_data1[tz][tx]=0.0;
		s_data1[threadIdx.y][threadIdx.x]=0.0;
		s_data1[BDIMY+2*radius-1-threadIdx.y][BDIMX+2*radius-1-threadIdx.x]=0.0;
		s_data1[threadIdx.y][BDIMX+2*radius-1-threadIdx.x]=0.0;
		s_data1[BDIMY+2*radius-1-threadIdx.y][threadIdx.x]=0.0;

		s_data2[tz][tx]=0.0;
		s_data2[threadIdx.y][threadIdx.x]=0.0;
		s_data2[BDIMY+2*radius-1-threadIdx.y][BDIMX+2*radius-1-threadIdx.x]=0.0;
		s_data2[threadIdx.y][BDIMX+2*radius-1-threadIdx.x]=0.0;
		s_data2[BDIMY+2*radius-1-threadIdx.y][threadIdx.x]=0.0;

		if((ix<dimx)&&(iz<dimz))
		{
				dimx=dimx+2*radius;dimz=dimz+2*radius;
				ix=ix+radius;iz=iz+radius;
				in_idx = ix*dimz+iz;//iz*dimx+ix;

				__syncthreads();

				s_data1[tz][tx]=vx2[in_idx];
				s_data2[tz][tx]=vz2[in_idx];

				if(threadIdx.y<radius)
				{
						s_data1[threadIdx.y][tx]=vx2[in_idx-radius];//g_input[in_idx-radius*dimx];//up
						s_data1[threadIdx.y+BDIMY+radius][tx]=vx2[in_idx+BDIMY];//g_input[in_idx+BDIMY*dimx];//down

						s_data2[threadIdx.y][tx]=vz2[in_idx-radius];//g_input[in_idx-radius*dimx];//up
						s_data2[threadIdx.y+BDIMY+radius][tx]=vz2[in_idx+BDIMY];//g_input[in_idx+BDIMY*dimx];//down
				}
				if(threadIdx.x<radius)
				{
						s_data1[tz][threadIdx.x]=vx2[in_idx-radius*dimz];//g_input[in_idx-radius];//left
						s_data1[tz][threadIdx.x+BDIMX+radius]=vx2[in_idx+BDIMX*dimz];//g_input[in_idx+BDIMX];//right

						s_data2[tz][threadIdx.x]=vz2[in_idx-radius*dimz];//g_input[in_idx-radius];//left
						s_data2[tz][threadIdx.x+BDIMX+radius]=vz2[in_idx+BDIMX*dimz];//g_input[in_idx+BDIMX];//right
				}
				
				s_velocity=velocity_d[in_idx]*velocity_d[in_idx];
				s_velocity1=velocity1_d[in_idx]*velocity1_d[in_idx];

				pppp=(1.0-(1.0*strain_p_d[in_idx]/tao_d[in_idx]));
				ssss=(1.0-(1.0*strain_s_d[in_idx]/tao_d[in_idx]));	

				density=density_d[in_idx];

				tao=tao_d[in_idx];
				
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


				mem_p2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*mem_p1_d[in_idx]-1.0*dt_real*1.0/tao*mem_p1_d[in_idx]+s_velocity*density/tao*pppp*(sumx*coe_x+sumz*coe_z));//s_velocity  and  s_velocity1

				mem_xx2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*mem_xx1_d[in_idx]-1.0*dt_real*1.0/tao*mem_xx1_d[in_idx]-2.0*s_velocity1*density/tao*ssss*sumz*coe_z);//s_velocity  and  s_velocity1

				mem_zz2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*mem_zz1_d[in_idx]-1.0*dt_real*1.0/tao*mem_zz1_d[in_idx]-2.0*s_velocity1*density/tao*ssss*sumx*coe_x);//s_velocity  and  s_velocity1
	
					
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

				mem_xz2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*mem_xz1_d[in_idx]-1.0*dt_real*1.0/tao*mem_xz1_d[in_idx]+s_velocity1*density/tao*ssss*(sumx1*coe_x+sumz1*coe_z));//s_velocity  and  s_velocity1
		}
}


/*__global__ void fwd_memory(float *mem_p2_d,float *mem_p1_d,float *mem_xx2_d,float *mem_xx1_d,float *mem_zz2_d,float *mem_zz1_d,float *mem_xz2_d,float *mem_xz1_d,float *vx2,float *vz2,float *attenuation_d,float dx,float dz,float dt,float *coe_d,float coe_x,float coe_z,int dimx,int dimz,float *density_d,float *velocity_d,float *velocity1_d,float *tao_d,float *strain_p_d,float *strain_s_d)
//fwd_memory<<<dimGrid,dimBlock>>>(mem_p2_d,mem_p1_d,mem_xx2_d,mem_xx1_d,mem_zz2_d,mem_zz1_d,mem_xz2_d,mem_xz1_d,vx2_d,vz2_d,attenuation_d,dx,dz,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,density_d,velocity_d,velocity1_d,tao_d,strain_p_d,strain_s_d);
{
		__shared__ float s_data1[BDIMY+2*radius][BDIMX+2*radius];
		__shared__ float s_data2[BDIMY+2*radius][BDIMX+2*radius];
		float dt_real=dt/1000;
		float s_velocity;
		float s_velocity1;
		float pppp=0,ssss=0;
		
		//float s_velocity3;
		//float s_velocity4;
		float s_attenuation;

		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		int tx = threadIdx.x+radius;
		int tz = threadIdx.y+radius;
		s_data1[tz][tx]=0.0;
		s_data1[threadIdx.y][threadIdx.x]=0.0;
		s_data1[BDIMY+2*radius-1-threadIdx.y][BDIMX+2*radius-1-threadIdx.x]=0.0;
		s_data1[threadIdx.y][BDIMX+2*radius-1-threadIdx.x]=0.0;
		s_data1[BDIMY+2*radius-1-threadIdx.y][threadIdx.x]=0.0;

		s_data2[tz][tx]=0.0;
		s_data2[threadIdx.y][threadIdx.x]=0.0;
		s_data2[BDIMY+2*radius-1-threadIdx.y][BDIMX+2*radius-1-threadIdx.x]=0.0;
		s_data2[threadIdx.y][BDIMX+2*radius-1-threadIdx.x]=0.0;
		s_data2[BDIMY+2*radius-1-threadIdx.y][threadIdx.x]=0.0;

		if((ix<dimx)&&(iz<dimz))
		{
				dimx=dimx+2*radius;dimz=dimz+2*radius;
				ix=ix+radius;iz=iz+radius;
				in_idx = ix*dimz+iz;//iz*dimx+ix;

				__syncthreads();

				s_data1[tz][tx]=vx2[in_idx];
				s_data2[tz][tx]=vz2[in_idx];

				if(threadIdx.y<radius)
				{
						s_data1[threadIdx.y][tx]=vx2[in_idx-radius];//g_input[in_idx-radius*dimx];//up
						s_data1[threadIdx.y+BDIMY+radius][tx]=vx2[in_idx+BDIMY];//g_input[in_idx+BDIMY*dimx];//down

						s_data2[threadIdx.y][tx]=vz2[in_idx-radius];//g_input[in_idx-radius*dimx];//up
						s_data2[threadIdx.y+BDIMY+radius][tx]=vz2[in_idx+BDIMY];//g_input[in_idx+BDIMY*dimx];//down
				}
				if(threadIdx.x<radius)
				{
						s_data1[tz][threadIdx.x]=vx2[in_idx-radius*dimz];//g_input[in_idx-radius];//left
						s_data1[tz][threadIdx.x+BDIMX+radius]=vx2[in_idx+BDIMX*dimz];//g_input[in_idx+BDIMX];//right

						s_data2[tz][threadIdx.x]=vz2[in_idx-radius*dimz];//g_input[in_idx-radius];//left
						s_data2[tz][threadIdx.x+BDIMX+radius]=vz2[in_idx+BDIMX*dimz];//g_input[in_idx+BDIMX];//right
				}
				
				s_velocity=velocity_d[in_idx]*velocity_d[in_idx];
				s_velocity1=velocity1_d[in_idx]*velocity1_d[in_idx];

				//s_velocity=(velocity_d[in_idx]*velocity_d[in_idx]+velocity_d[in_idx+dimz]*velocity_d[in_idx+dimz])/2.0;
				//s_velocity1=(velocity1_d[in_idx]*velocity1_d[in_idx]+velocity1_d[in_idx+dimz]*velocity1_d[in_idx+dimz])/2.0;
				
				//s_velocity3=(velocity_d[in_idx]*velocity_d[in_idx]+velocity_d[in_idx+1]*velocity_d[in_idx+1])/2.0;
				//s_velocity4=(velocity1_d[in_idx]*velocity1_d[in_idx]+velocity1_d[in_idx+1]*velocity1_d[in_idx+1])/2.0;
				s_attenuation=attenuation_d[in_idx];
				__syncthreads();

				pppp=(1.0-(1.0*strain_p_d[in_idx]/tao_d[in_idx]));
				ssss=(1.0-(1.0*strain_s_d[in_idx]/tao_d[in_idx]));	
		

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


				mem_p2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*mem_p1_d[in_idx]-1.0*dt_real*1.0/tao_d[in_idx]*mem_p1_d[in_idx]+s_velocity*density_d[in_idx]/tao_d[in_idx]*pppp*(sumx*coe_x+sumz*coe_z));//s_velocity  and  s_velocity1

				mem_xx2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*mem_xx1_d[in_idx]-1.0*dt_real*1.0/tao_d[in_idx]*mem_xx1_d[in_idx]-2.0*s_velocity1*density_d[in_idx]/tao_d[in_idx]*ssss*sumz*coe_z);//s_velocity  and  s_velocity1

				mem_zz2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*mem_zz1_d[in_idx]-1.0*dt_real*1.0/tao_d[in_idx]*mem_zz1_d[in_idx]-2.0*s_velocity1*density_d[in_idx]/tao_d[in_idx]*ssss*sumx*coe_x);//s_velocity  and  s_velocity1
	
					
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

				mem_xz2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*mem_xz1_d[in_idx]-1.0*dt_real*1.0/tao_d[in_idx]*mem_xz1_d[in_idx]+s_velocity1*density_d[in_idx]/tao_d[in_idx]*ssss*(sumx1*coe_x+sumz1*coe_z));//s_velocity  and  s_velocity1
		}
}*/


//////////////source and receiver back propagation based on viscoelastic modeling
__global__ void rfwd_txxzzxzpp_viscoelastic_and_memory_old(float *tp2,float *tp1,float *txx2,float *txx1,float *tzz2,float *tzz1,float *txz2,float *txz1,float *vx2,float *vz2,float *modul_p_d,float *modul_s_d,float *attenuation_d,float dt,float *coe_d,float coe_x,float coe_z,int dimx,int dimz,float *density_d,float *mem_p2_d,float *mem_p1_d,float *mem_xx2_d,float *mem_xx1_d,float *mem_zz2_d,float *mem_zz1_d,float *mem_xz2_d,float *mem_xz1_d,float *velocity_d,float *velocity1_d,float *tao_d,float *strain_p_d,float *strain_s_d)
//fwd_txxzzxzpp_viscoelastic_and_memory<<<dimGrid,dimBlock>>>(tp2_d,tp1_d,txx2_d,txx1_d,tzz2_d,tzz1_d,txz2_d,txz1_d,vx1_d,vz1_d,modul_p_d,modul_s_d,attenuation_d,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,density_d,mem_p2_d,mem_p1_d,mem_xx2_d,mem_xx1_d,mem_zz2_d,mem_zz1_d,mem_xz2_d,mem_xz1_d,velocity_d,velocity1_d,tao_d,strain_p_d,strain_s_d);
{
		__shared__ float s_data1[BDIMY+2*radius][BDIMX+2*radius];
		__shared__ float s_data2[BDIMY+2*radius][BDIMX+2*radius];
		float dt_real=dt/1000;
		float s_velocity;
		float s_velocity1;
	
		float pppp=0,ssss=0;
		
		//float s_velocity3;
		//float s_velocity4;
		float s_attenuation;

		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		int tx = threadIdx.x+radius;
		int tz = threadIdx.y+radius;
		s_data1[tz][tx]=0.0;
		s_data1[threadIdx.y][threadIdx.x]=0.0;
		s_data1[BDIMY+2*radius-1-threadIdx.y][BDIMX+2*radius-1-threadIdx.x]=0.0;
		s_data1[threadIdx.y][BDIMX+2*radius-1-threadIdx.x]=0.0;
		s_data1[BDIMY+2*radius-1-threadIdx.y][threadIdx.x]=0.0;

		s_data2[tz][tx]=0.0;
		s_data2[threadIdx.y][threadIdx.x]=0.0;
		s_data2[BDIMY+2*radius-1-threadIdx.y][BDIMX+2*radius-1-threadIdx.x]=0.0;
		s_data2[threadIdx.y][BDIMX+2*radius-1-threadIdx.x]=0.0;
		s_data2[BDIMY+2*radius-1-threadIdx.y][threadIdx.x]=0.0;

		if((ix<dimx)&&(iz<dimz))
		{
				dimx=dimx+2*radius;dimz=dimz+2*radius;
				ix=ix+radius;iz=iz+radius;
				in_idx = ix*dimz+iz;//iz*dimx+ix;

				__syncthreads();

				s_data1[tz][tx]=vx2[in_idx];
				s_data2[tz][tx]=vz2[in_idx];

				if(threadIdx.y<radius)
				{
						s_data1[threadIdx.y][tx]=vx2[in_idx-radius];//g_input[in_idx-radius*dimx];//up
						s_data1[threadIdx.y+BDIMY+radius][tx]=vx2[in_idx+BDIMY];//g_input[in_idx+BDIMY*dimx];//down

						s_data2[threadIdx.y][tx]=vz2[in_idx-radius];//g_input[in_idx-radius*dimx];//up
						s_data2[threadIdx.y+BDIMY+radius][tx]=vz2[in_idx+BDIMY];//g_input[in_idx+BDIMY*dimx];//down
				}
				if(threadIdx.x<radius)
				{
						s_data1[tz][threadIdx.x]=vx2[in_idx-radius*dimz];//g_input[in_idx-radius];//left
						s_data1[tz][threadIdx.x+BDIMX+radius]=vx2[in_idx+BDIMX*dimz];//g_input[in_idx+BDIMX];//right

						s_data2[tz][threadIdx.x]=vz2[in_idx-radius*dimz];//g_input[in_idx-radius];//left
						s_data2[tz][threadIdx.x+BDIMX+radius]=vz2[in_idx+BDIMX*dimz];//g_input[in_idx+BDIMX];//right
				}
				
				s_velocity=velocity_d[in_idx]*velocity_d[in_idx];
				s_velocity1=velocity1_d[in_idx]*velocity1_d[in_idx];

				//s_velocity=(velocity_d[in_idx]*velocity_d[in_idx]+velocity_d[in_idx+dimz]*velocity_d[in_idx+dimz])/2.0;
				//s_velocity1=(velocity1_d[in_idx]*velocity1_d[in_idx]+velocity1_d[in_idx+dimz]*velocity1_d[in_idx+dimz])/2.0;
				
				//s_velocity3=(velocity_d[in_idx]*velocity_d[in_idx]+velocity_d[in_idx+1]*velocity_d[in_idx+1])/2.0;
				//s_velocity4=(velocity1_d[in_idx]*velocity1_d[in_idx]+velocity1_d[in_idx+1]*velocity1_d[in_idx+1])/2.0;
				s_attenuation=attenuation_d[in_idx];
				__syncthreads();
				
				pppp=(1.0-(1.0*strain_p_d[in_idx]/tao_d[in_idx]));
				ssss=(1.0-(1.0*strain_s_d[in_idx]/tao_d[in_idx]));
				

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

				/*mem_p2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*mem_p1_d[in_idx]-1.0*dt_real*1.0/tao_d[in_idx]*mem_p1_d[in_idx]+s_velocity*density_d[in_idx]/tao_d[in_idx]*pppp*(sumx*coe_x+sumz*coe_z));//s_velocity  and  s_velocity1

				mem_xx2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*mem_xx1_d[in_idx]-1.0*dt_real*1.0/tao_d[in_idx]*mem_xx1_d[in_idx]-2.0*s_velocity1*density_d[in_idx]/tao_d[in_idx]*ssss*sumz*coe_z);//s_velocity  and  s_velocity1

				mem_zz2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*mem_zz1_d[in_idx]-1.0*dt_real*1.0/tao_d[in_idx]*mem_zz1_d[in_idx]-2.0*s_velocity1*density_d[in_idx]/tao_d[in_idx]*ssss*sumx*coe_x);//s_velocity  and  s_velocity1*/

				mem_p2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*mem_p1_d[in_idx]-1.0*dt_real*1.0/tao_d[in_idx]*mem_p1_d[in_idx]+s_velocity*density_d[in_idx]/tao_d[in_idx]*pppp*(sumx*coe_x+sumz*coe_z));//s_velocity  and  s_velocity1

				mem_xx2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*mem_xx1_d[in_idx]-1.0*dt_real*1.0/tao_d[in_idx]*mem_xx1_d[in_idx]-2.0*s_velocity1*density_d[in_idx]/tao_d[in_idx]*ssss*sumz*coe_z);//s_velocity  and  s_velocity1

				mem_zz2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*mem_zz1_d[in_idx]-1.0*dt_real*1.0/tao_d[in_idx]*mem_zz1_d[in_idx]-2.0*s_velocity1*density_d[in_idx]/tao_d[in_idx]*ssss*sumx*coe_x);//s_velocity  and  s_velocity1

				
				/*txx2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*txx1[in_idx]+modul_p_d[in_idx]*density_d[in_idx]*sumx*coe_x+(modul_p_d[in_idx]-2*modul_s_d[in_idx])*density_d[in_idx]*sumz*coe_z-dt_real*(mem_p2_d[in_idx]+mem_xx2_d[in_idx]));//s_velocity  and  s_velocity1

				tzz2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*tzz1[in_idx]+(modul_p_d[in_idx]-2*modul_s_d[in_idx])*density_d[in_idx]*sumx*coe_x+modul_p_d[in_idx]*density_d[in_idx]*sumz*coe_z)-dt_real*(mem_p2_d[in_idx]+mem_zz2_d[in_idx]);//sumx  and  sumz 
				
				tp2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*tp1[in_idx]+modul_p_d[in_idx]*density_d[in_idx]*sumx*coe_x+modul_p_d[in_idx]*density_d[in_idx]*sumz*coe_z-dt_real*mem_p2_d[in_idx]);*/

				txx2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*txx1[in_idx]-modul_p_d[in_idx]*density_d[in_idx]*sumx*coe_x-(modul_p_d[in_idx]-2*modul_s_d[in_idx])*density_d[in_idx]*sumz*coe_z-dt_real*(mem_p2_d[in_idx]+mem_xx2_d[in_idx]));//s_velocity  and  s_velocity1

				tzz2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*tzz1[in_idx]-(modul_p_d[in_idx]-2*modul_s_d[in_idx])*density_d[in_idx]*sumx*coe_x-modul_p_d[in_idx]*density_d[in_idx]*sumz*coe_z-dt_real*(mem_p2_d[in_idx]+mem_zz2_d[in_idx]));//sumx  and  sumz 
				
				tp2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*tp1[in_idx]-modul_p_d[in_idx]*density_d[in_idx]*sumx*coe_x-modul_p_d[in_idx]*density_d[in_idx]*sumz*coe_z-dt_real*mem_p2_d[in_idx]);	
					
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

				/*mem_xz2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*mem_xz1_d[in_idx]-1.0*dt_real*1.0/tao_d[in_idx]*mem_xz1_d[in_idx]+s_velocity1*density_d[in_idx]/tao_d[in_idx]*ssss*(sumx1*coe_x+sumz1*coe_z));//s_velocity  and  s_velocity1

				txz2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*txz1[in_idx]+modul_s_d[in_idx]*density_d[in_idx]*(sumx1*coe_x+sumz1*coe_z)-dt_real*mem_xz2_d[in_idx]);*/

				mem_xz2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*mem_xz1_d[in_idx]-1.0*dt_real*1.0/tao_d[in_idx]*mem_xz1_d[in_idx]+s_velocity1*density_d[in_idx]/tao_d[in_idx]*ssss*(sumx1*coe_x+sumz1*coe_z));//s_velocity  and  s_velocity1

				txz2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*txz1[in_idx]-modul_s_d[in_idx]*density_d[in_idx]*(sumx1*coe_x+sumz1*coe_z)-dt_real*mem_xz2_d[in_idx]);
		}
}

//////////////source and receiver back propagation based on viscoelastic modeling
__global__ void rfwd_txxzzxzpp_viscoelastic_and_memory(float *tp2,float *tp1,float *txx2,float *txx1,float *tzz2,float *tzz1,float *txz2,float *txz1,float *vx2,float *vz2,float *modul_p_d,float *modul_s_d,float *attenuation_d,float dt,float *coe_d,float coe_x,float coe_z,int dimx,int dimz,float *density_d,float *mem_p2_d,float *mem_p1_d,float *mem_xx2_d,float *mem_xx1_d,float *mem_zz2_d,float *mem_zz1_d,float *mem_xz2_d,float *mem_xz1_d,float *velocity_d,float *velocity1_d,float *tao_d,float *strain_p_d,float *strain_s_d)
//fwd_txxzzxzpp_viscoelastic_and_memory<<<dimGrid,dimBlock>>>(tp2_d,tp1_d,txx2_d,txx1_d,tzz2_d,tzz1_d,txz2_d,txz1_d,vx1_d,vz1_d,modul_p_d,modul_s_d,attenuation_d,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,density_d,mem_p2_d,mem_p1_d,mem_xx2_d,mem_xx1_d,mem_zz2_d,mem_zz1_d,mem_xz2_d,mem_xz1_d,velocity_d,velocity1_d,tao_d,strain_p_d,strain_s_d);
{
		__shared__ float s_data1[BDIMY+2*radius][BDIMX+2*radius];
		__shared__ float s_data2[BDIMY+2*radius][BDIMX+2*radius];
		float dt_real=dt/1000;
		float s_velocity;
		float s_velocity1;
	
		float pppp=0,ssss=0;
		
		//float s_velocity3;
		//float s_velocity4;
		float s_attenuation;

		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		int tx = threadIdx.x+radius;
		int tz = threadIdx.y+radius;
		s_data1[tz][tx]=0.0;
		s_data1[threadIdx.y][threadIdx.x]=0.0;
		s_data1[BDIMY+2*radius-1-threadIdx.y][BDIMX+2*radius-1-threadIdx.x]=0.0;
		s_data1[threadIdx.y][BDIMX+2*radius-1-threadIdx.x]=0.0;
		s_data1[BDIMY+2*radius-1-threadIdx.y][threadIdx.x]=0.0;

		s_data2[tz][tx]=0.0;
		s_data2[threadIdx.y][threadIdx.x]=0.0;
		s_data2[BDIMY+2*radius-1-threadIdx.y][BDIMX+2*radius-1-threadIdx.x]=0.0;
		s_data2[threadIdx.y][BDIMX+2*radius-1-threadIdx.x]=0.0;
		s_data2[BDIMY+2*radius-1-threadIdx.y][threadIdx.x]=0.0;

		if((ix<dimx)&&(iz<dimz))
		{
				dimx=dimx+2*radius;dimz=dimz+2*radius;
				ix=ix+radius;iz=iz+radius;
				in_idx = ix*dimz+iz;//iz*dimx+ix;

				__syncthreads();

				s_data1[tz][tx]=vx2[in_idx];
				s_data2[tz][tx]=vz2[in_idx];

				if(threadIdx.y<radius)
				{
						s_data1[threadIdx.y][tx]=vx2[in_idx-radius];//g_input[in_idx-radius*dimx];//up
						s_data1[threadIdx.y+BDIMY+radius][tx]=vx2[in_idx+BDIMY];//g_input[in_idx+BDIMY*dimx];//down

						s_data2[threadIdx.y][tx]=vz2[in_idx-radius];//g_input[in_idx-radius*dimx];//up
						s_data2[threadIdx.y+BDIMY+radius][tx]=vz2[in_idx+BDIMY];//g_input[in_idx+BDIMY*dimx];//down
				}
				if(threadIdx.x<radius)
				{
						s_data1[tz][threadIdx.x]=vx2[in_idx-radius*dimz];//g_input[in_idx-radius];//left
						s_data1[tz][threadIdx.x+BDIMX+radius]=vx2[in_idx+BDIMX*dimz];//g_input[in_idx+BDIMX];//right

						s_data2[tz][threadIdx.x]=vz2[in_idx-radius*dimz];//g_input[in_idx-radius];//left
						s_data2[tz][threadIdx.x+BDIMX+radius]=vz2[in_idx+BDIMX*dimz];//g_input[in_idx+BDIMX];//right
				}
				
				s_velocity=velocity_d[in_idx]*velocity_d[in_idx];
				s_velocity1=velocity1_d[in_idx]*velocity1_d[in_idx];

				//s_velocity=(velocity_d[in_idx]*velocity_d[in_idx]+velocity_d[in_idx+dimz]*velocity_d[in_idx+dimz])/2.0;
				//s_velocity1=(velocity1_d[in_idx]*velocity1_d[in_idx]+velocity1_d[in_idx+dimz]*velocity1_d[in_idx+dimz])/2.0;
				
				//s_velocity3=(velocity_d[in_idx]*velocity_d[in_idx]+velocity_d[in_idx+1]*velocity_d[in_idx+1])/2.0;
				//s_velocity4=(velocity1_d[in_idx]*velocity1_d[in_idx]+velocity1_d[in_idx+1]*velocity1_d[in_idx+1])/2.0;
				s_attenuation=attenuation_d[in_idx];
				__syncthreads();
				
				pppp=(1.0-(1.0*strain_p_d[in_idx]/tao_d[in_idx]));
				ssss=(1.0-(1.0*strain_s_d[in_idx]/tao_d[in_idx]));
				

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

				mem_p2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*mem_p1_d[in_idx]-1.0*dt_real*1.0/tao_d[in_idx]*mem_p1_d[in_idx]+s_velocity*density_d[in_idx]/tao_d[in_idx]*pppp*(sumx*coe_x+sumz*coe_z));//s_velocity  and  s_velocity1

				mem_xx2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*mem_xx1_d[in_idx]-1.0*dt_real*1.0/tao_d[in_idx]*mem_xx1_d[in_idx]-2.0*s_velocity1*density_d[in_idx]/tao_d[in_idx]*ssss*sumz*coe_z);//s_velocity  and  s_velocity1

				mem_zz2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*mem_zz1_d[in_idx]-1.0*dt_real*1.0/tao_d[in_idx]*mem_zz1_d[in_idx]-2.0*s_velocity1*density_d[in_idx]/tao_d[in_idx]*ssss*sumx*coe_x);//s_velocity  and  s_velocity1

				/*mem_p2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*mem_p1_d[in_idx]+1.0/(1.0-dt_real*1.0/tao_d[in_idx])*(mem_p1_d[in_idx]-s_velocity*density_d[in_idx]/tao_d[in_idx]*pppp*(sumx*coe_x+sumz*coe_z)));//s_velocity  and  s_velocity1

				mem_xx2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*mem_xx1_d[in_idx]+1.0/(1.0-dt_real*1.0/tao_d[in_idx])*(mem_xx1_d[in_idx]+2.0*s_velocity1*density_d[in_idx]/tao_d[in_idx]*ssss*sumz*coe_z));//s_velocity  and  s_velocity1

				mem_zz2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*mem_zz1_d[in_idx]+1.0/(1.0-dt_real*1.0/tao_d[in_idx])*(mem_zz1_d[in_idx]+2.0*s_velocity1*density_d[in_idx]/tao_d[in_idx]*ssss*sumx*coe_x));//s_velocity  and  s_velocity1*/

				
				txx2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*txx1[in_idx]+modul_p_d[in_idx]*density_d[in_idx]*sumx*coe_x+(modul_p_d[in_idx]-2*modul_s_d[in_idx])*density_d[in_idx]*sumz*coe_z-dt_real*(mem_p2_d[in_idx]+mem_xx2_d[in_idx]));//s_velocity  and  s_velocity1

				tzz2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*tzz1[in_idx]+(modul_p_d[in_idx]-2*modul_s_d[in_idx])*density_d[in_idx]*sumx*coe_x+modul_p_d[in_idx]*density_d[in_idx]*sumz*coe_z)-dt_real*(mem_p2_d[in_idx]+mem_zz2_d[in_idx]);//sumx  and  sumz 
				
				tp2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*tp1[in_idx]+modul_p_d[in_idx]*density_d[in_idx]*sumx*coe_x+modul_p_d[in_idx]*density_d[in_idx]*sumz*coe_z-dt_real*mem_p2_d[in_idx]);

				/*txx2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*txx1[in_idx]-modul_p_d[in_idx]*density_d[in_idx]*sumx*coe_x-(modul_p_d[in_idx]-2*modul_s_d[in_idx])*density_d[in_idx]*sumz*coe_z-dt_real*(mem_p2_d[in_idx]+mem_xx2_d[in_idx]));//s_velocity  and  s_velocity1

				tzz2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*tzz1[in_idx]-(modul_p_d[in_idx]-2*modul_s_d[in_idx])*density_d[in_idx]*sumx*coe_x-modul_p_d[in_idx]*density_d[in_idx]*sumz*coe_z-dt_real*(mem_p2_d[in_idx]+mem_zz2_d[in_idx]));//sumx  and  sumz 
				
				tp2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*tp1[in_idx]-modul_p_d[in_idx]*density_d[in_idx]*sumx*coe_x-modul_p_d[in_idx]*density_d[in_idx]*sumz*coe_z-dt_real*mem_p2_d[in_idx]);*/	
					
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

				mem_xz2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*mem_xz1_d[in_idx]-1.0*dt_real*1.0/tao_d[in_idx]*mem_xz1_d[in_idx]+s_velocity1*density_d[in_idx]/tao_d[in_idx]*ssss*(sumx1*coe_x+sumz1*coe_z));//s_velocity  and  s_velocity1

				txz2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*txz1[in_idx]+modul_s_d[in_idx]*density_d[in_idx]*(sumx1*coe_x+sumz1*coe_z)-dt_real*mem_xz2_d[in_idx]);

				/*mem_xz2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*mem_xz1_d[in_idx]+1.0/(1.0-dt_real*1.0/tao_d[in_idx])*(mem_xz1_d[in_idx]-s_velocity1*density_d[in_idx]/tao_d[in_idx]*ssss*(sumx1*coe_x+sumz1*coe_z)));//s_velocity  and  s_velocity1

				txz2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*txz1[in_idx]-modul_s_d[in_idx]*density_d[in_idx]*(sumx1*coe_x+sumz1*coe_z)-dt_real*mem_xz2_d[in_idx]);*/
		}
}

__global__ void receiver_fwd_txxzzxzpp_viscoelastic_and_memory(float *tp2,float *tp1,float *txx2,float *txx1,float *tzz2,float *tzz1,float *txz2,float *txz1,float *vx2,float *vz2,float *modul_p_d,float *modul_s_d,float *attenuation_d,float dt,float *coe_d,float coe_x,float coe_z,int dimx,int dimz,float *density_d,float *mem_p2_d,float *mem_p1_d,float *mem_xx2_d,float *mem_xx1_d,float *mem_zz2_d,float *mem_zz1_d,float *mem_xz2_d,float *mem_xz1_d,float *velocity_d,float *velocity1_d,float *tao_d,float *strain_p_d,float *strain_s_d)
//fwd_txxzzxzpp_viscoelastic_and_memory<<<dimGrid,dimBlock>>>(tp2_d,tp1_d,txx2_d,txx1_d,tzz2_d,tzz1_d,txz2_d,txz1_d,vx1_d,vz1_d,modul_p_d,modul_s_d,attenuation_d,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,density_d,mem_p2_d,mem_p1_d,mem_xx2_d,mem_xx1_d,mem_zz2_d,mem_zz1_d,mem_xz2_d,mem_xz1_d,velocity_d,velocity1_d,tao_d,strain_p_d,strain_s_d);
{
		__shared__ float s_data1[BDIMY+2*radius][BDIMX+2*radius];
		__shared__ float s_data2[BDIMY+2*radius][BDIMX+2*radius];
		float dt_real=dt/1000;
		float s_velocity;
		float s_velocity1;
	
		float pppp=0,ssss=0;
		
		//float s_velocity3;
		//float s_velocity4;
		float s_attenuation;

		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		int tx = threadIdx.x+radius;
		int tz = threadIdx.y+radius;
		s_data1[tz][tx]=0.0;
		s_data1[threadIdx.y][threadIdx.x]=0.0;
		s_data1[BDIMY+2*radius-1-threadIdx.y][BDIMX+2*radius-1-threadIdx.x]=0.0;
		s_data1[threadIdx.y][BDIMX+2*radius-1-threadIdx.x]=0.0;
		s_data1[BDIMY+2*radius-1-threadIdx.y][threadIdx.x]=0.0;

		s_data2[tz][tx]=0.0;
		s_data2[threadIdx.y][threadIdx.x]=0.0;
		s_data2[BDIMY+2*radius-1-threadIdx.y][BDIMX+2*radius-1-threadIdx.x]=0.0;
		s_data2[threadIdx.y][BDIMX+2*radius-1-threadIdx.x]=0.0;
		s_data2[BDIMY+2*radius-1-threadIdx.y][threadIdx.x]=0.0;

		if((ix<dimx)&&(iz<dimz))
		{
				dimx=dimx+2*radius;dimz=dimz+2*radius;
				ix=ix+radius;iz=iz+radius;
				in_idx = ix*dimz+iz;//iz*dimx+ix;

				__syncthreads();

				s_data1[tz][tx]=vx2[in_idx];
				s_data2[tz][tx]=vz2[in_idx];

				if(threadIdx.y<radius)
				{
						s_data1[threadIdx.y][tx]=vx2[in_idx-radius];//g_input[in_idx-radius*dimx];//up
						s_data1[threadIdx.y+BDIMY+radius][tx]=vx2[in_idx+BDIMY];//g_input[in_idx+BDIMY*dimx];//down

						s_data2[threadIdx.y][tx]=vz2[in_idx-radius];//g_input[in_idx-radius*dimx];//up
						s_data2[threadIdx.y+BDIMY+radius][tx]=vz2[in_idx+BDIMY];//g_input[in_idx+BDIMY*dimx];//down
				}
				if(threadIdx.x<radius)
				{
						s_data1[tz][threadIdx.x]=vx2[in_idx-radius*dimz];//g_input[in_idx-radius];//left
						s_data1[tz][threadIdx.x+BDIMX+radius]=vx2[in_idx+BDIMX*dimz];//g_input[in_idx+BDIMX];//right

						s_data2[tz][threadIdx.x]=vz2[in_idx-radius*dimz];//g_input[in_idx-radius];//left
						s_data2[tz][threadIdx.x+BDIMX+radius]=vz2[in_idx+BDIMX*dimz];//g_input[in_idx+BDIMX];//right
				}
				
				s_velocity=velocity_d[in_idx]*velocity_d[in_idx];
				s_velocity1=velocity1_d[in_idx]*velocity1_d[in_idx];

				//s_velocity=(velocity_d[in_idx]*velocity_d[in_idx]+velocity_d[in_idx+dimz]*velocity_d[in_idx+dimz])/2.0;
				//s_velocity1=(velocity1_d[in_idx]*velocity1_d[in_idx]+velocity1_d[in_idx+dimz]*velocity1_d[in_idx+dimz])/2.0;
				
				//s_velocity3=(velocity_d[in_idx]*velocity_d[in_idx]+velocity_d[in_idx+1]*velocity_d[in_idx+1])/2.0;
				//s_velocity4=(velocity1_d[in_idx]*velocity1_d[in_idx]+velocity1_d[in_idx+1]*velocity1_d[in_idx+1])/2.0;
				s_attenuation=attenuation_d[in_idx];
				__syncthreads();
				
				pppp=(1.0-(1.0*strain_p_d[in_idx]/tao_d[in_idx]));
				ssss=(1.0-(1.0*strain_s_d[in_idx]/tao_d[in_idx]));
				

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

				mem_p2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*mem_p1_d[in_idx]-1.0*dt_real*1.0/tao_d[in_idx]*mem_p1_d[in_idx]+s_velocity*density_d[in_idx]/tao_d[in_idx]*pppp*(sumx*coe_x+sumz*coe_z));//s_velocity  and  s_velocity1

				mem_xx2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*mem_xx1_d[in_idx]-1.0*dt_real*1.0/tao_d[in_idx]*mem_xx1_d[in_idx]-2.0*s_velocity1*density_d[in_idx]/tao_d[in_idx]*ssss*sumz*coe_z);//s_velocity  and  s_velocity1

				mem_zz2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*mem_zz1_d[in_idx]-1.0*dt_real*1.0/tao_d[in_idx]*mem_zz1_d[in_idx]-2.0*s_velocity1*density_d[in_idx]/tao_d[in_idx]*ssss*sumx*coe_x);//s_velocity  and  s_velocity1


				txx2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*txx1[in_idx]+modul_p_d[in_idx]*density_d[in_idx]*sumx*coe_x+(modul_p_d[in_idx]-2*modul_s_d[in_idx])*density_d[in_idx]*sumz*coe_z-dt_real*(mem_p2_d[in_idx]+mem_xx2_d[in_idx]));//s_velocity  and  s_velocity1

				tzz2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*tzz1[in_idx]+(modul_p_d[in_idx]-2*modul_s_d[in_idx])*density_d[in_idx]*sumx*coe_x+modul_p_d[in_idx]*density_d[in_idx]*sumz*coe_z-dt_real*(mem_p2_d[in_idx]+mem_zz2_d[in_idx]));//sumx  and  sumz 
				
				tp2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*tp1[in_idx]+modul_p_d[in_idx]*density_d[in_idx]*sumx*coe_x+modul_p_d[in_idx]*density_d[in_idx]*sumz*coe_z-dt_real*mem_p2_d[in_idx]);	
					
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

				mem_xz2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*mem_xz1_d[in_idx]-1.0*dt_real*1.0/tao_d[in_idx]*mem_xz1_d[in_idx]+s_velocity1*density_d[in_idx]/tao_d[in_idx]*ssss*(sumx1*coe_x+sumz1*coe_z));//s_velocity  and  s_velocity1

				txz2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*txz1[in_idx]+modul_s_d[in_idx]*density_d[in_idx]*(sumx1*coe_x+sumz1*coe_z)-dt_real*mem_xz2_d[in_idx]);
		}
}


__global__ void receiver_fwd_txxzzxzpp_viscoelastic_and_memory_3parameterization(float *tp2,float *tp1,float *txx2,float *txx1,float *tzz2,float *tzz1,float *txz2,float *txz1,float *vx2,float *vz2,float *modul_p_d,float *modul_s_d,float *attenuation_d,float dt,float *coe_d,float coe_x,float coe_z,int dimx,int dimz,float *density_d,float *mem_p2_d,float *mem_p1_d,float *mem_xx2_d,float *mem_xx1_d,float *mem_zz2_d,float *mem_zz1_d,float *mem_xz2_d,float *mem_xz1_d,float *velocity_d,float *velocity1_d,float *tao_d,float *strain_p_d,float *strain_s_d)
//fwd_txxzzxzpp_viscoelastic_and_memory<<<dimGrid,dimBlock>>>(tp2_d,tp1_d,txx2_d,txx1_d,tzz2_d,tzz1_d,txz2_d,txz1_d,vx1_d,vz1_d,modul_p_d,modul_s_d,attenuation_d,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,density_d,mem_p2_d,mem_p1_d,mem_xx2_d,mem_xx1_d,mem_zz2_d,mem_zz1_d,mem_xz2_d,mem_xz1_d,velocity_d,velocity1_d,tao_d,strain_p_d,strain_s_d);
{
		__shared__ float s_data1[BDIMY+2*radius][BDIMX+2*radius];
		__shared__ float s_data2[BDIMY+2*radius][BDIMX+2*radius];
		float dt_real=dt/1000;
		float s_velocity;
		float s_velocity1;
	
		float pppp=0,ssss=0;
		
		//float s_velocity3;
		//float s_velocity4;
		float s_attenuation;

		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		int tx = threadIdx.x+radius;
		int tz = threadIdx.y+radius;
		s_data1[tz][tx]=0.0;
		s_data1[threadIdx.y][threadIdx.x]=0.0;
		s_data1[BDIMY+2*radius-1-threadIdx.y][BDIMX+2*radius-1-threadIdx.x]=0.0;
		s_data1[threadIdx.y][BDIMX+2*radius-1-threadIdx.x]=0.0;
		s_data1[BDIMY+2*radius-1-threadIdx.y][threadIdx.x]=0.0;

		s_data2[tz][tx]=0.0;
		s_data2[threadIdx.y][threadIdx.x]=0.0;
		s_data2[BDIMY+2*radius-1-threadIdx.y][BDIMX+2*radius-1-threadIdx.x]=0.0;
		s_data2[threadIdx.y][BDIMX+2*radius-1-threadIdx.x]=0.0;
		s_data2[BDIMY+2*radius-1-threadIdx.y][threadIdx.x]=0.0;

		if((ix<dimx)&&(iz<dimz))
		{
				dimx=dimx+2*radius;dimz=dimz+2*radius;
				ix=ix+radius;iz=iz+radius;
				in_idx = ix*dimz+iz;//iz*dimx+ix;

				__syncthreads();

				s_data1[tz][tx]=vx2[in_idx];
				s_data2[tz][tx]=vz2[in_idx];

				if(threadIdx.y<radius)
				{
						s_data1[threadIdx.y][tx]=vx2[in_idx-radius];//g_input[in_idx-radius*dimx];//up
						s_data1[threadIdx.y+BDIMY+radius][tx]=vx2[in_idx+BDIMY];//g_input[in_idx+BDIMY*dimx];//down

						s_data2[threadIdx.y][tx]=vz2[in_idx-radius];//g_input[in_idx-radius*dimx];//up
						s_data2[threadIdx.y+BDIMY+radius][tx]=vz2[in_idx+BDIMY];//g_input[in_idx+BDIMY*dimx];//down
				}
				if(threadIdx.x<radius)
				{
						s_data1[tz][threadIdx.x]=vx2[in_idx-radius*dimz];//g_input[in_idx-radius];//left
						s_data1[tz][threadIdx.x+BDIMX+radius]=vx2[in_idx+BDIMX*dimz];//g_input[in_idx+BDIMX];//right

						s_data2[tz][threadIdx.x]=vz2[in_idx-radius*dimz];//g_input[in_idx-radius];//left
						s_data2[tz][threadIdx.x+BDIMX+radius]=vz2[in_idx+BDIMX*dimz];//g_input[in_idx+BDIMX];//right
				}
				
				s_velocity=velocity_d[in_idx]*velocity_d[in_idx];
				s_velocity1=velocity1_d[in_idx]*velocity1_d[in_idx];

				//s_velocity=(velocity_d[in_idx]*velocity_d[in_idx]+velocity_d[in_idx+dimz]*velocity_d[in_idx+dimz])/2.0;
				//s_velocity1=(velocity1_d[in_idx]*velocity1_d[in_idx]+velocity1_d[in_idx+dimz]*velocity1_d[in_idx+dimz])/2.0;
				
				//s_velocity3=(velocity_d[in_idx]*velocity_d[in_idx]+velocity_d[in_idx+1]*velocity_d[in_idx+1])/2.0;
				//s_velocity4=(velocity1_d[in_idx]*velocity1_d[in_idx]+velocity1_d[in_idx+1]*velocity1_d[in_idx+1])/2.0;
				s_attenuation=attenuation_d[in_idx];
				__syncthreads();
				
				pppp=(1.0-(1.0*strain_p_d[in_idx]/tao_d[in_idx]));
				ssss=(1.0-(1.0*strain_s_d[in_idx]/tao_d[in_idx]));
				

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

				mem_p2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*mem_p1_d[in_idx]-1.0*dt_real*1.0/tao_d[in_idx]*mem_p1_d[in_idx]+s_velocity*density_d[in_idx]/tao_d[in_idx]*pppp*(sumx*coe_x+sumz*coe_z));//s_velocity  and  s_velocity1

				mem_xx2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*mem_xx1_d[in_idx]-1.0*dt_real*1.0/tao_d[in_idx]*mem_xx1_d[in_idx]-2.0*s_velocity1*density_d[in_idx]/tao_d[in_idx]*ssss*sumz*coe_z);//s_velocity  and  s_velocity1

				mem_zz2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*mem_zz1_d[in_idx]-1.0*dt_real*1.0/tao_d[in_idx]*mem_zz1_d[in_idx]-2.0*s_velocity1*density_d[in_idx]/tao_d[in_idx]*ssss*sumx*coe_x);//s_velocity  and  s_velocity1


				txx2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*txx1[in_idx]+modul_p_d[in_idx]*density_d[in_idx]*sumx*coe_x+(modul_p_d[in_idx]-2*modul_s_d[in_idx])*density_d[in_idx]*sumz*coe_z-dt_real*(mem_p2_d[in_idx]+mem_xx2_d[in_idx]));//s_velocity  and  s_velocity1

				tzz2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*tzz1[in_idx]+(modul_p_d[in_idx]-2*modul_s_d[in_idx])*density_d[in_idx]*sumx*coe_x+modul_p_d[in_idx]*density_d[in_idx]*sumz*coe_z-dt_real*(mem_p2_d[in_idx]+mem_zz2_d[in_idx]));//sumx  and  sumz 
				
				tp2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*tp1[in_idx]+modul_p_d[in_idx]*density_d[in_idx]*sumx*coe_x+modul_p_d[in_idx]*density_d[in_idx]*sumz*coe_z-dt_real*mem_p2_d[in_idx]);	
					
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

				mem_xz2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*mem_xz1_d[in_idx]-1.0*dt_real*1.0/tao_d[in_idx]*mem_xz1_d[in_idx]+s_velocity1*density_d[in_idx]/tao_d[in_idx]*ssss*(sumx1*coe_x+sumz1*coe_z));//s_velocity  and  s_velocity1

				txz2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*txz1[in_idx]+modul_s_d[in_idx]*density_d[in_idx]*(sumx1*coe_x+sumz1*coe_z)-dt_real*mem_xz2_d[in_idx]);
		}
}



///////////////2017年07月31日 星期一 20时42分28秒 
__global__ void adjoint_fwd_vx_viscoelastic(float *vx2,float *vx1,float *txx1,float *txz1,float *tzz1,float *attenuation_d,float dx,float dz,float dt,float *coe_d,float coe_x,float coe_z,int dimx,int dimz,float *modul_p_d,float *modul_s_d,float *density_d,float *mem_p1_d,float *mem_xx1_d,float *mem_zz1_d,float *mem_xz1_d,float *velocity_d,float *velocity1_d,float *tao_d,float *strain_p_d,float *strain_s_d)
//adjoint_fwd_vx_viscoelastic<<<dimGrid,dimBlock>>>(rvx2_d,rvx1_d,rtxx2_d,rtxz2_d,rtzz2_d,attenuation_d,dx,dz,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,modul_p_d,modul_s_d,s_density_d,rmem_p2_d,rmem_xx2_d,rmem_zz2_d,rmem_xz2_d,s_velocity_d,s_velocity1_d,tao_d,strain_p_d,strain_s_d);
{
		__shared__ float s_data1[BDIMY2+2*radius2][BDIMX2+2*radius2];
		__shared__ float s_data2[BDIMY2+2*radius2][BDIMX2+2*radius2];
		__shared__ float s_data3[BDIMY2+2*radius2][BDIMX2+2*radius2];
		float dt_real=dt/1000;
		float s_velocity;
		float s_velocity1;

		float tao;
		float modul_p;
		float modul_s;
		
		float s_attenuation;

		float pppp=0,ssss=0;

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

				tao=tao_d[in_idx];

				modul_p=modul_p_d[in_idx];
				modul_s=modul_s_d[in_idx];
//////////////注意伴随状态方程左边存在密度，所以用来反传计算伴随波场跟密度没有关系？？？？？？？？？？

				pppp=1.0/tao*(1.0-(1.0*strain_p_d[in_idx]/tao))*s_velocity;

				ssss=1.0/tao*(1.0-(1.0*strain_s_d[in_idx]/tao))*s_velocity1;

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

////sumx:the derivation of x direction of rxx
				float    sumrxx=coe_d[1]*(mem_xx1_d[in_idx]-mem_xx1_d[in_idx-1*dimz]);
					sumrxx+=coe_d[2]*(mem_xx1_d[in_idx+1*dimz]-mem_xx1_d[in_idx-2*dimz]);
					sumrxx+=coe_d[3]*(mem_xx1_d[in_idx+2*dimz]-mem_xx1_d[in_idx-3*dimz]);
					sumrxx+=coe_d[4]*(mem_xx1_d[in_idx+3*dimz]-mem_xx1_d[in_idx-4*dimz]);
					sumrxx+=coe_d[5]*(mem_xx1_d[in_idx+4*dimz]-mem_xx1_d[in_idx-5*dimz]);
					sumrxx+=coe_d[6]*(mem_xx1_d[in_idx+5*dimz]-mem_xx1_d[in_idx-6*dimz]);

////sumx1:the derivation of x direction of rzz
				float    sumrzz=coe_d[1]*(mem_zz1_d[in_idx]-mem_zz1_d[in_idx-1*dimz]);
					sumrzz+=coe_d[2]*(mem_zz1_d[in_idx+1*dimz]-mem_zz1_d[in_idx-2*dimz]);
					sumrzz+=coe_d[3]*(mem_zz1_d[in_idx+2*dimz]-mem_zz1_d[in_idx-3*dimz]);
					sumrzz+=coe_d[4]*(mem_zz1_d[in_idx+3*dimz]-mem_zz1_d[in_idx-4*dimz]);
					sumrzz+=coe_d[5]*(mem_zz1_d[in_idx+4*dimz]-mem_zz1_d[in_idx-5*dimz]);
					sumrzz+=coe_d[6]*(mem_zz1_d[in_idx+5*dimz]-mem_zz1_d[in_idx-6*dimz]);

////sumxz:the derivation of z direction of rxz
				float    sumrxz=coe_d[1]*(mem_xz1_d[in_idx]-mem_xz1_d[in_idx-1]);
					sumrxz+=coe_d[2]*(mem_xz1_d[in_idx+1]-mem_xz1_d[in_idx-2]);
					sumrxz+=coe_d[3]*(mem_xz1_d[in_idx+2]-mem_xz1_d[in_idx-3]);
					sumrxz+=coe_d[4]*(mem_xz1_d[in_idx+3]-mem_xz1_d[in_idx-4]);
					sumrxz+=coe_d[5]*(mem_xz1_d[in_idx+4]-mem_xz1_d[in_idx-5]);
					sumrxz+=coe_d[6]*(mem_xz1_d[in_idx+5]-mem_xz1_d[in_idx-6]);

				//vx2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*vx1[in_idx]+(s_velocity*sumx*coe_x+(s_velocity-2*s_velocity1)*sumx1*coe_x+s_velocity1*sumxz*coe_z));

				vx2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*vx1[in_idx]+modul_p*sumx*coe_x+(modul_p-2*modul_s)*sumx1*coe_x+modul_s*sumxz*coe_z+pppp*sumrxx*coe_x+(pppp-2.0*ssss)*sumrzz*coe_x+ssss*sumrxz*coe_z);
		}
}


__global__ void adjoint_fwd_vz_viscoelastic(float *vz2,float *vz1,float *txx1,float *txz1,float *tzz1,float *attenuation_d,float dx,float dz,float dt,float *coe_d,float coe_x,float coe_z,int dimx,int dimz,float *modul_p_d,float *modul_s_d,float *density_d,float *mem_p1_d,float *mem_xx1_d,float *mem_zz1_d,float *mem_xz1_d,float *velocity_d,float *velocity1_d,float *tao_d,float *strain_p_d,float *strain_s_d)
//adjoint_fwd_vz_viscoelastic<<<dimGrid,dimBlock>>>(rvz2_d,rvz1_d,rtxx2_d,rtxz2_d,rtzz2_d,attenuation_d,dx,dz,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,modul_p_d,modul_s_d,s_density_d,rmem_p1_d,rmem_xx1_d,rmem_zz1_d,rmem_xz1_d,s_velocity_d,s_velocity1_d,tao_d,strain_p_d,strain_s_d);
{
		__shared__ float s_data1[BDIMY2+2*radius2][BDIMX2+2*radius2];
		__shared__ float s_data2[BDIMY2+2*radius2][BDIMX2+2*radius2];
		__shared__ float s_data3[BDIMY2+2*radius2][BDIMX2+2*radius2];
		float dt_real=dt/1000;
		float s_velocity;
		float s_velocity1;
		float s_attenuation;

		float tao;
		float modul_p;
		float modul_s;

		float pppp=0,ssss=0;

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

				tao=tao_d[in_idx];

				modul_p=modul_p_d[in_idx];
				modul_s=modul_s_d[in_idx];

				pppp=1.0/tao*(1.0-(1.0*strain_p_d[in_idx]/tao))*s_velocity;

				ssss=1.0/tao*(1.0-(1.0*strain_s_d[in_idx]/tao))*s_velocity1;

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

////sumx:the derivation of z direction of rxx
				float    sumrxx=coe_d[1]*(mem_xx1_d[in_idx+1]-mem_xx1_d[in_idx]);
					sumrxx+=coe_d[2]*(mem_xx1_d[in_idx+2]-mem_xx1_d[in_idx-1]);
					sumrxx+=coe_d[3]*(mem_xx1_d[in_idx+3]-mem_xx1_d[in_idx-2]);
					sumrxx+=coe_d[4]*(mem_xx1_d[in_idx+4]-mem_xx1_d[in_idx-3]);
					sumrxx+=coe_d[5]*(mem_xx1_d[in_idx+5]-mem_xx1_d[in_idx-4]);
					sumrxx+=coe_d[6]*(mem_xx1_d[in_idx+6]-mem_xx1_d[in_idx-5]);

////sumx1:the derivation of z direction of rzz
				float    sumrzz=coe_d[1]*(mem_zz1_d[in_idx+1]-mem_zz1_d[in_idx]);
					sumrzz+=coe_d[2]*(mem_zz1_d[in_idx+2]-mem_zz1_d[in_idx-1]);
					sumrzz+=coe_d[3]*(mem_zz1_d[in_idx+3]-mem_zz1_d[in_idx-2]);
					sumrzz+=coe_d[4]*(mem_zz1_d[in_idx+4]-mem_zz1_d[in_idx-3]);
					sumrzz+=coe_d[5]*(mem_zz1_d[in_idx+5]-mem_zz1_d[in_idx-4]);
					sumrzz+=coe_d[6]*(mem_zz1_d[in_idx+6]-mem_zz1_d[in_idx-5]);

////sumxz:the derivation of x direction of rxz
				float    sumrxz=coe_d[1]*(mem_xz1_d[in_idx+1*dimz]-mem_xz1_d[in_idx]);
					sumrxz+=coe_d[2]*(mem_xz1_d[in_idx+2*dimz]-mem_xz1_d[in_idx-1*dimz]);
					sumrxz+=coe_d[3]*(mem_xz1_d[in_idx+3*dimz]-mem_xz1_d[in_idx-2*dimz]);
					sumrxz+=coe_d[4]*(mem_xz1_d[in_idx+4*dimz]-mem_xz1_d[in_idx-3*dimz]);
					sumrxz+=coe_d[5]*(mem_xz1_d[in_idx+5*dimz]-mem_xz1_d[in_idx-4*dimz]);
					sumrxz+=coe_d[6]*(mem_xz1_d[in_idx+6*dimz]-mem_xz1_d[in_idx-5*dimz]);

				//vz2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*vz1[in_idx]+(s_velocity*sumz*coe_z+(s_velocity-2*s_velocity1)*sumz1*coe_z+s_velocity1*sumx*coe_x));

				vz2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*vz1[in_idx]+modul_p*sumz*coe_z+(modul_p-2*modul_s)*sumz1*coe_z+modul_s*sumx*coe_x+(pppp-2.0*ssss)*sumrxx*coe_z+pppp*sumrzz*coe_z+ssss*sumrxz*coe_x);
		}
}


__global__ void adjoint_fwd_memory(float *mem_p2_d,float *mem_p1_d,float *mem_xx2_d,float *mem_xx1_d,float *mem_zz2_d,float *mem_zz1_d,float *mem_xz2_d,float *mem_xz1_d,float *tp1,float *txx1,float *tzz1,float *txz1,float *attenuation_d,float dx,float dz,float dt,float *coe_d,float coe_x,float coe_z,int dimx,int dimz,float *density_d,float *velocity_d,float *velocity1_d,float *tao_d,float *strain_p_d,float *strain_s_d)
//adjoint_fwd_memory<<<dimGrid,dimBlock>>>(rmem_p2_d,rmem_p1_d,rmem_xx2_d,rmem_xx1_d,rmem_zz2_d,rmem_zz1_d,rmem_xz2_d,rmem_xz1_d,rtp2_d,rtxx2_d,rtzz2_d,rtxz2_d,attenuation_d,dx,dz,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,s_density_d,s_velocity_d,s_velocity1_d,tao_d,strain_p_d,strain_s_d);
{
		float s_attenuation;

		float dt_real=dt/1000;

		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		if((ix<dimx)&&(iz<dimz))
		{
				dimx=dimx+2*radius2;dimz=dimz+2*radius2;
				ix=ix+radius2;iz=iz+radius2;
				in_idx = ix*dimz+iz;//iz*dimx+ix;

				__syncthreads();

				s_attenuation=attenuation_d[in_idx];

				__syncthreads();

				mem_p2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*mem_p1_d[in_idx]-1.0*dt_real*1.0/tao_d[in_idx]*mem_p1_d[in_idx]+1.0*dt_real*tp1[in_idx]);

				mem_xx2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*mem_xx1_d[in_idx]-1.0*dt_real*1.0/tao_d[in_idx]*mem_xx1_d[in_idx]+1.0*dt_real*txx1[in_idx]);

				mem_zz2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*mem_zz1_d[in_idx]-1.0*dt_real*1.0/tao_d[in_idx]*mem_zz1_d[in_idx]+1.0*dt_real*tzz1[in_idx]);
	
				mem_xz2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*mem_xz1_d[in_idx]-1.0*dt_real*1.0/tao_d[in_idx]*mem_xz1_d[in_idx]+1.0*dt_real*txz1[in_idx]);
		}
}

__global__ void adjoint_fwd_memory_new(float *mem_p2_d,float *mem_p1_d,float *mem_xx2_d,float *mem_xx1_d,float *mem_zz2_d,float *mem_zz1_d,float *mem_xz2_d,float *mem_xz1_d,float *tp1_d,float *txx1_d,float *tzz1_d,float *txz1_d,float *attenuation_d,float dx,float dz,float dt,float *coe_d,float coe_x,float coe_z,int dimx,int dimz,float *density_d,float *velocity_d,float *velocity1_d,float *tao_d,float *strain_p_d,float *strain_s_d)
//adjoint_fwd_memory<<<dimGrid,dimBlock>>>(rmem_p2_d,rmem_p1_d,rmem_xx2_d,rmem_xx1_d,rmem_zz2_d,rmem_zz1_d,rmem_xz2_d,rmem_xz1_d,rtp2_d,rtxx2_d,rtzz2_d,rtxz2_d,attenuation_d,dx,dz,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,s_density_d,s_velocity_d,s_velocity1_d,tao_d,strain_p_d,strain_s_d);
{
		float s_attenuation;

		float dt_real=dt/1000;

		float mem_p1,mem_xx1,mem_zz1,mem_xz1;
		float tp1,txx1,tzz1,txz1;
		float tao;

		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;

		int in_idx;

		if((ix<dimx)&&(iz<dimz))
		{
				dimx=dimx+2*radius2;dimz=dimz+2*radius2;
				ix=ix+radius2;iz=iz+radius2;
				in_idx = ix*dimz+iz;//iz*dimx+ix;

				__syncthreads();

				s_attenuation=attenuation_d[in_idx];

				mem_p1=mem_p1_d[in_idx],mem_xx1=mem_xx1_d[in_idx],mem_zz1=mem_zz1_d[in_idx],mem_xz1=mem_xz1_d[in_idx];
				tp1=tp1_d[in_idx],txx1=txx1_d[in_idx],tzz1=tzz1_d[in_idx],txz1=txz1_d[in_idx];
				tao=tao_d[in_idx];

				__syncthreads();

				mem_p2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*mem_p1-1.0*dt_real*1.0/tao*mem_p1+1.0*dt_real*tp1);

				mem_xx2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*mem_xx1-1.0*dt_real*1.0/tao*mem_xx1+1.0*dt_real*txx1);

				mem_zz2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*mem_zz1-1.0*dt_real*1.0/tao*mem_zz1+1.0*dt_real*tzz1);
	
				mem_xz2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*mem_xz1-1.0*dt_real*1.0/tao*mem_xz1+1.0*dt_real*txz1);
		}
}

__global__ void cal_gradient_in_viscoelastic_media(float *grad_lame1_d,float *grad_lame2_d,float *grad_den_d,float *vx_t_d,float *vz_t_d,float *vx_x_d,float *vz_z_d,float *vx_z_d,float *vz_x_d,float *rvx1_d,float *rvz1_d,float *rtxx1_d,float *rtxz1_d,float *rtzz1_d,float *rmem_xx1_d,float *rmem_xz1_d,float *rmem_zz1_d,int boundary_left,int boundary_up,int nx,int nz,int dimx,int dimz,float *s_velocity_d,float *s_velocity1_d,float *s_density_d,float *tao_d,float *strain_p_d,float *strain_s_d)
//cal_gradient_in_viscoelastic_media<<<dimGrid,dimBlock>>>(grad_lame11_d,grad_lame22_d,grad_den1_d,vx_t_d,vz_t_d,vx_x_d,vz_z_d,vx_z_d,vz_x_d,rvx2_d,rvz2_d,rtxx2_d,rtxz2_d,rtzz2_d,rmem_xx2_d,rmem_xz2_d,rmem_zz2_d,boundary_left,boundary_up,nx_size,nz,nx_append,nz_append,s_velocity_d,s_velocity1_d,s_density_d,tao_d,strain_p_d,strain_s_d);
{
		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;
		int in_idx1;
		float lame1;
		float lame2;
		float mp=0.0;
		float ms=0.0;
		float np=0.0;
		float ns=0.0;
		float A=0.0,B=0.0,C=0.0;
		float D=0.0,E=0.0,F=0.0;

		if((ix<nx)&&(iz<nz))
		{
			in_idx=ix*nz+iz;//iz*nz+ix;
			in_idx1=(ix+boundary_left)*dimz+iz+boundary_up;//iz*nz+ix;

			lame1=s_density_d[in_idx1]*s_velocity_d[in_idx1]*s_velocity_d[in_idx1]-2.0*s_density_d[in_idx1]*s_velocity1_d[in_idx1]*s_velocity1_d[in_idx1];
			lame2=s_density_d[in_idx1]*s_velocity1_d[in_idx1]*s_velocity1_d[in_idx1];

			mp=1.0*strain_p_d[in_idx1]/tao_d[in_idx1];
			ms=1.0*strain_s_d[in_idx1]/tao_d[in_idx1];

			np=1.0/tao_d[in_idx1]*(1-mp);
			ns=1.0/tao_d[in_idx1]*(1-ms);


			grad_den_d[in_idx]=grad_den_d[in_idx]+1.0*s_density_d[in_idx1]*(rvx1_d[in_idx1]*vx_t_d[in_idx1]+rvz1_d[in_idx1]*vz_t_d[in_idx1]);

			grad_lame1_d[in_idx]=grad_lame1_d[in_idx]+(-1.0)*lame1*(1.0*mp*(rtxx1_d[in_idx1]+rtzz1_d[in_idx1])*(vx_x_d[in_idx1]+vz_z_d[in_idx1])+1.0*np*(rmem_xx1_d[in_idx1]+rmem_zz1_d[in_idx1])*(vx_x_d[in_idx1]+vz_z_d[in_idx1]));		
			

			A=(1.0*mp*vx_x_d[in_idx1]+(mp-ms)*vz_z_d[in_idx1])*rtxx1_d[in_idx1];
			B=(1.0*mp*vz_z_d[in_idx1]+(mp-ms)*vx_x_d[in_idx1])*rtzz1_d[in_idx1];
			C=1.0*ms*(vx_z_d[in_idx1]+vz_x_d[in_idx1])*rtxz1_d[in_idx1];

			D=(1.0*np*vx_x_d[in_idx1]+(np-ns)*vz_z_d[in_idx1])*rmem_xx1_d[in_idx1];
			E=(1.0*np*vz_z_d[in_idx1]+(np-ns)*vx_x_d[in_idx1])*rmem_zz1_d[in_idx1];
			F=1.0*ns*(vx_z_d[in_idx1]+vz_x_d[in_idx1])*rmem_xz1_d[in_idx1];

			grad_lame2_d[in_idx]=grad_lame2_d[in_idx]+(-1.0)*lame2*(2.0*A+2.0*B+C+2.0*D+2.0*E+F);
		}
}

__global__ void cal_gradient_in_viscoelastic_media_new(float *grad_lame1_d,float *grad_lame2_d,float *grad_den_d,float *vx_t_d,float *vz_t_d,float *vx_x_d,float *vz_z_d,float *vx_z_d,float *vz_x_d,float *rvx1_d,float *rvz1_d,float *rtxx1_d,float *rtxz1_d,float *rtzz1_d,float *rmem_xx1_d,float *rmem_xz1_d,float *rmem_zz1_d,int boundary_left,int boundary_up,int nx,int nz,int dimx,int dimz,float *s_velocity_d,float *s_velocity1_d,float *s_density_d,float *tao_d,float *strain_p_d,float *strain_s_d)
//cal_gradient_in_viscoelastic_media<<<dimGrid,dimBlock>>>(grad_lame11_d,grad_lame22_d,grad_den1_d,vx_t_d,vz_t_d,vx_x_d,vz_z_d,vx_z_d,vz_x_d,rvx2_d,rvz2_d,rtxx2_d,rtxz2_d,rtzz2_d,rmem_xx2_d,rmem_xz2_d,rmem_zz2_d,boundary_left,boundary_up,nx,nz,nx_append,nz_append,s_velocity_d,s_velocity1_d,s_density_d,tao_d,strain_p_d,strain_s_d);
{
		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;
		int in_idx1;
		//float lame1;
		//float lame2;
		float mp=0.0;
		float ms=0.0;
		float np=0.0;
		float ns=0.0;
		float A=0.0,B=0.0,C=0.0;
		float D=0.0,E=0.0,F=0.0;

		if((ix<nx)&&(iz<nz))
		{
			in_idx=ix*nz+iz;//iz*nz+ix;
			in_idx1=(ix+boundary_left)*dimz+iz+boundary_up;//iz*nz+ix;

			//lame1=s_density_d[in_idx1]*s_velocity_d[in_idx1]*s_velocity_d[in_idx1]-2.0*s_density_d[in_idx1]*s_velocity1_d[in_idx1]*s_velocity1_d[in_idx1];
			//lame2=s_density_d[in_idx1]*s_velocity1_d[in_idx1]*s_velocity1_d[in_idx1];

			mp=1.0*strain_p_d[in_idx1]/tao_d[in_idx1];
			ms=1.0*strain_s_d[in_idx1]/tao_d[in_idx1];

			np=1.0/tao_d[in_idx1]*(1-mp);
			ns=1.0/tao_d[in_idx1]*(1-ms);


			grad_den_d[in_idx]=grad_den_d[in_idx]+1.0*(rvx1_d[in_idx1]*vx_t_d[in_idx1]+rvz1_d[in_idx1]*vz_t_d[in_idx1]);

			grad_lame1_d[in_idx]=grad_lame1_d[in_idx]+(-1.0)*(1.0*mp*(rtxx1_d[in_idx1]+rtzz1_d[in_idx1])*(vx_x_d[in_idx1]+vz_z_d[in_idx1])+1.0*np*(rmem_xx1_d[in_idx1]+rmem_zz1_d[in_idx1])*(vx_x_d[in_idx1]+vz_z_d[in_idx1]));		
			

			A=(1.0*mp*vx_x_d[in_idx1]+(mp-ms)*vz_z_d[in_idx1])*rtxx1_d[in_idx1];
			B=(1.0*mp*vz_z_d[in_idx1]+(mp-ms)*vx_x_d[in_idx1])*rtzz1_d[in_idx1];
			C=1.0*ms*(vx_z_d[in_idx1]+vz_x_d[in_idx1])*rtxz1_d[in_idx1];

			D=(1.0*np*vx_x_d[in_idx1]+(np-ns)*vz_z_d[in_idx1])*rmem_xx1_d[in_idx1];
			E=(1.0*np*vz_z_d[in_idx1]+(np-ns)*vx_x_d[in_idx1])*rmem_zz1_d[in_idx1];
			F=1.0*ns*(vx_z_d[in_idx1]+vz_x_d[in_idx1])*rmem_xz1_d[in_idx1];

			grad_lame2_d[in_idx]=grad_lame2_d[in_idx]+(-1.0)*(2.0*A+2.0*B+C+2.0*D+2.0*E+F);
		}

}


//////2017年08月02日 星期三 11时48分46秒 
__global__ void cuda_cal_dem_parameter_viscoelastic_media(float *dem_p1_d,float *dem_p2_d,float *dem_p3_d,float *dem_p4_d,float *dem_p5_d,float *dem_p6_d,float *dem_p7_d,float *dem_p8_d,float *vx_x_d,float *vx_z_d,float *vz_x_d,float *vz_z_d,float *vx_t_d,float *vz_t_d,float *tmp_perturb_lame1_d,float *tmp_perturb_lame2_d,float *tmp_perturb_den_d,float *tmp_perturb_vp_d,float *tmp_perturb_vs_d,float *tmp_perturb_density_d,int dimx,int dimz,float *s_velocity_d,float *s_velocity1_d,float *s_density_d,float *tao_d,float *strain_p_d,float *strain_s_d,float dt,int inversion_para)
//cuda_cal_dem_parameter_viscoelastic_media<<<dimGrid,dimBlock>>>(dem_p1_d,dem_p2_d,dem_p3_d,dem_p4_d,dem_p5_d,dem_p6_d,dem_p7_d,dem_p8_d,vx_x_d,vx_z_d,vz_x_d,vz_z_d,vx_t_d,vz_t_d,tmp_perturb_lame1_d,tmp_perturb_lame2_d,tmp_perturb_den_d,tmp_perturb_vp_d,tmp_perturb_vs_d,tmp_perturb_density_d,nx_append_radius,nz_append_radius,s_velocity_d,s_velocity1_d,s_density_d,tao_d,strain_p_d,strain_s_d,dt,inversion_para);
{
		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;
		
		float dt_real=dt/1000;
		float lame1,lame2;
		float mp=0.0;
		float ms=0.0;
		float np=0.0;
		float ns=0.0;

		float p1,p2,p3;
		float p4,p5,p6;

		if((ix<dimx)&&(iz<dimz))
		{
				dimx=dimx+2*radius2;dimz=dimz+2*radius2;
				ix=ix+radius2;iz=iz+radius2;
				in_idx = ix*dimz+iz;//iz*dimx+ix;
			
				mp=1.0*strain_p_d[in_idx]/tao_d[in_idx];
				ms=1.0*strain_s_d[in_idx]/tao_d[in_idx];

				np=1.0/tao_d[in_idx]*(1-mp);
				ns=1.0/tao_d[in_idx]*(1-ms);
				

				lame1=s_density_d[in_idx]*s_velocity_d[in_idx]*s_velocity_d[in_idx]-2.0*s_density_d[in_idx]*s_velocity1_d[in_idx]*s_velocity1_d[in_idx];
				lame2=s_density_d[in_idx]*s_velocity1_d[in_idx]*s_velocity1_d[in_idx];
				
				p1=tmp_perturb_lame1_d[in_idx]*lame1*1.0;

				p2=tmp_perturb_lame2_d[in_idx]*lame2*1.0;

				p3=tmp_perturb_den_d[in_idx]*s_density_d[in_idx]*1.0;				

				
				//p4=tmp_perturb_vp_d[in_idx]*s_velocity_d[in_idx]*1.0;

				//p5=tmp_perturb_vs_d[in_idx]*s_velocity1_d[in_idx]*1.0;

				//p6=tmp_perturb_density_d[in_idx]*s_density_d[in_idx]*1.0;

				p4=tmp_perturb_vp_d[in_idx]*s_velocity_d[in_idx]*s_density_d[in_idx]*1.0;

				p5=tmp_perturb_vs_d[in_idx]*s_velocity1_d[in_idx]*s_density_d[in_idx]*1.0;

				p6=tmp_perturb_density_d[in_idx]*s_density_d[in_idx]*1.0;

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

					dem_p1_d[in_idx]=(-1.0)*dt_real*p3*vx_t_d[in_idx]/s_density_d[in_idx];
					dem_p2_d[in_idx]=(-1.0)*dt_real*p3*vz_t_d[in_idx]/s_density_d[in_idx];

					dem_p3_d[in_idx]=1.0*dt_real*((p1+2*p2)*mp*vx_x_d[in_idx]+((p1+2*p2)*mp-2*p2*ms)*vz_z_d[in_idx]);
		
					dem_p4_d[in_idx]=1.0*dt_real*((p1+2*p2)*mp*vz_z_d[in_idx]+((p1+2*p2)*mp-2*p2*ms)*vx_x_d[in_idx]);

					dem_p5_d[in_idx]=1.0*dt_real*(p2*ms*(vx_z_d[in_idx]+vz_x_d[in_idx]));


					dem_p6_d[in_idx]=1.0*dt_real*((p1+2*p2)*np*vx_x_d[in_idx]+((p1+2*p2)*np-2*p2*ns)*vz_z_d[in_idx]);

					dem_p7_d[in_idx]=1.0*dt_real*((p1+2*p2)*np*vz_z_d[in_idx]+((p1+2*p2)*np-2*p2*ns)*vx_x_d[in_idx]);

					dem_p8_d[in_idx]=1.0*dt_real*(p2*ns*(vx_z_d[in_idx]+vz_x_d[in_idx]));
				
		}
}

__global__ void cuda_cal_dem_parameter_viscoelastic_media_new(float *dem_p1_d,float *dem_p2_d,float *dem_p_d,float *vx_x_d,float *vx_z_d,float *vz_x_d,float *vz_z_d,float *vx_t_d,float *vz_t_d,float *tmp_perturb_lame1_d,float *tmp_perturb_lame2_d,float *tmp_perturb_den_d,float *tmp_perturb_vp_d,float *tmp_perturb_vs_d,float *tmp_perturb_density_d,int dimx,int dimz,float *s_velocity_d,float *s_velocity1_d,float *s_density_d,float *tao_d,float *strain_p_d,float *strain_s_d,float dt,int inversion_para)
//cuda_cal_dem_parameter_viscoelastic_media_new<<<dimGrid,dimBlock>>>(dem_p1_d,dem_p2_d,dem_p_all_d,vx_x_d,vx_z_d,vz_x_d,vz_z_d,vx_t_d,vz_t_d,tmp_perturb_lame1_d,tmp_perturb_lame2_d,tmp_perturb_den_d,tmp_perturb_vp_d,tmp_perturb_vs_d,tmp_perturb_density_d,nx_append_radius,nz_append_radius,s_velocity_d,s_velocity1_d,s_density_d,tao_d,strain_p_d,strain_s_d,dt,inversion_para);
{
		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;
		
		float dt_real=dt/1000;
		float lame1,lame2;
		float mp=0.0;
		float ms=0.0;
		float np=0.0;
		float ns=0.0;

		float p1,p2,p3;
		float p4,p5,p6;

		if((ix<dimx)&&(iz<dimz))
		{
				dimx=dimx+2*radius2;dimz=dimz+2*radius2;
				ix=ix+radius2;iz=iz+radius2;
				in_idx = ix*dimz+iz;//iz*dimx+ix;
			
				mp=1.0*strain_p_d[in_idx]/tao_d[in_idx];
				ms=1.0*strain_s_d[in_idx]/tao_d[in_idx];

				np=1.0/tao_d[in_idx]*(1-mp);
				ns=1.0/tao_d[in_idx]*(1-ms);
				

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
					//p1=2.0*s_density_d[in_idx]*s_velocity_d[in_idx]*p4-4.0*s_density_d[in_idx]*s_velocity1_d[in_idx]*p5+1.0*(1.0*s_velocity_d[in_idx]*s_velocity_d[in_idx]-2.0*s_velocity1_d[in_idx]*s_velocity1_d[in_idx])*p6;
					p1=2*s_velocity_d[in_idx]*p4-4*s_velocity1_d[in_idx]*p5+(-1.0*s_velocity_d[in_idx]*s_velocity_d[in_idx]+2*s_velocity1_d[in_idx]*s_velocity1_d[in_idx])*p6;
					//p2=2.0*s_density_d[in_idx]*s_velocity1_d[in_idx]*p5+1.0*s_velocity1_d[in_idx]*s_velocity1_d[in_idx]*p6;
					p2=2.0*s_velocity1_d[in_idx]*p5-1.0*s_velocity1_d[in_idx]*s_velocity1_d[in_idx]*p6;

					p3=p6;
				}

					dem_p1_d[in_idx]=(-1.0)*p3*vx_t_d[in_idx];
					dem_p2_d[in_idx]=(-1.0)*p3*vz_t_d[in_idx];

					dem_p_d[2*dimx*dimz+in_idx]=1.0*dt_real*((p1+2*p2)*mp*vx_x_d[in_idx]+((p1+2*p2)*mp-2*p2*ms)*vz_z_d[in_idx]);
		
					dem_p_d[3*dimx*dimz+in_idx]=1.0*dt_real*((p1+2*p2)*mp*vz_z_d[in_idx]+((p1+2*p2)*mp-2*p2*ms)*vx_x_d[in_idx]);

					dem_p_d[4*dimx*dimz+in_idx]=1.0*dt_real*(p2*ms*(vx_z_d[in_idx]+vz_x_d[in_idx]));


					dem_p_d[5*dimx*dimz+in_idx]=1.0*dt_real*((p1+2*p2)*np*vx_x_d[in_idx]+((p1+2*p2)*np-2*p2*ns)*vz_z_d[in_idx]);

					dem_p_d[6*dimx*dimz+in_idx]=1.0*dt_real*((p1+2*p2)*np*vz_z_d[in_idx]+((p1+2*p2)*np-2*p2*ns)*vx_x_d[in_idx]);

					dem_p_d[7*dimx*dimz+in_idx]=1.0*dt_real*(p2*ns*(vx_z_d[in_idx]+vz_x_d[in_idx]));
				
		}
}

__global__ void cuda_cal_multiply(float *tmp_perturb_den_d,float *s_density_d,float *dem_p1_d,int dimx,int dimz)
////////////////cuda_cal_multiply<<<dimGrid,dimBlock>>>(tmp_perturb_den_d,s_density_d,dem_p1_d,nx_append_radius,nz_append_radius);
{
		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		//float m;

		if((ix<dimx)&&(iz<dimz))
		{
				dimx=dimx+2*radius2;dimz=dimz+2*radius2;
				ix=ix+radius2;iz=iz+radius2;
				in_idx = ix*dimz+iz;//iz*dimx+ix;
			
				tmp_perturb_den_d[in_idx]=tmp_perturb_den_d[in_idx]*s_density_d[in_idx];

				dem_p1_d[in_idx]=tmp_perturb_den_d[in_idx];
		}
}

__global__ void cuda_bell_smoothz_new(float *g, float *smg, int rbell, int nx, int nz)
/*< smoothing with gaussian function >*/
{
	int i;
	int ix=threadIdx.x+blockIdx.x*blockDim.x;
	int iz=threadIdx.y+blockIdx.y*blockDim.y;

	int id=iz+ix*nz;

	if(ix<nx && iz<nz)
	{
		float s=0;
		float sum=0;
		for(i=-rbell; i<=rbell; i++) if(iz+i>=0 && iz+i<nz) sum+=expf(-(1.0*i*i)/2.0/rbell);

		for(i=-rbell; i<=rbell; i++) if(iz+i>=0 && iz+i<nz) s+=expf(-(1.0*i*i)/2.0/rbell)/sum*g[id+i];
		smg[id]=s;
	}
}

__global__ void cuda_bell_smoothx_new(float *g, float *smg, int rbell, int nx, int nz)
/*< smoothing with gaussian function >*/
{
	int i;
	int ix=threadIdx.x+blockIdx.x*blockDim.x;
	int iz=threadIdx.y+blockIdx.y*blockDim.y;

	int id=iz+ix*nz;

	if(ix<nx && iz<nz)
	{
		float s=0;
		float sum=0.0;
		for(i=-rbell; i<=rbell; i++) if(ix+i>=0 && ix+i<nx) sum+=expf(-(1.0*i*i)/2.0/rbell);

		for(i=-rbell; i<=rbell; i++) if(ix+i>=0 && ix+i<nx) s+=expf(-(1.0*i*i)/2.0/rbell)/sum*g[id+i*nz];
		smg[id]=s;
	}
}

__global__ void cuda_bell_smooth_2d(float *g, float *smg, int rbell, int nx, int nz)
/*< smoothing with gaussian function >*/
{
	int im,in;
	int ix=threadIdx.x+blockIdx.x*blockDim.x;
	int iz=threadIdx.y+blockIdx.y*blockDim.y;

	int id=iz+ix*nz;

	float distance;

	if(ix<nx && iz<nz)
	{
		float s=0;
		float sum=0.0;

		for(im=-rbell; im<=rbell; im++) 
			for(in=-rbell; in<=rbell; in++) 		
		if(ix+im>=0 && ix+im<nx && iz+in>=0 && iz+in<nz) 
		{
			distance=im*im+in*in;

			sum+=expf(-(1.0*distance)/2/rbell);
		}

		for(im=-rbell; im<=rbell; im++) 
			for(in=-rbell; in<=rbell; in++) 		
		if(ix+im>=0 && ix+im<nx && iz+in>=0 && iz+in<nz) 
		{
			distance=im*im+in*in;

			s+=expf(-(1.0*distance)/2/rbell)/sum*g[id+im*nz+in];;
		}		
		
		smg[id]=s;
	}
}

__global__ void cuda_get_partly_mode_boundary(float *velocity_all_d,float *wf_d,int nx,int nz,int receiver_x_cord,int receiver_interval,int receiver_num,int nx_append_new,int nz_append,int boundary_left,int boundary_up)
{
	
	int ix=threadIdx.x+blockIdx.x*blockDim.x;
	int iz=threadIdx.y+blockIdx.y*blockDim.y;

	int id,id1;

	int nnx;
	nnx=receiver_interval*receiver_num;

	if(ix<nnx&&iz<nz)
	{
		id=ix*nz+iz;

		id1=(ix+boundary_left+receiver_x_cord)*nz_append+iz+boundary_up;

		wf_d[id]=velocity_all_d[id1];	
	}
}

__global__ void cuda_get_partly_mode_boundary_z1_z2(float *velocity_all_d,float *wf_d,int nx,int nz,int receiver_x_cord,int receiver_interval,int receiver_num,int nx_append_new,int nz_append,int boundary_left,int boundary_up,int z1,int z2)
//cuda_get_partly_mode_boundary_z1_z2<<<dimGrid_new,dimBlock>>>(s_velocity_all_d,wf_d,nx,nz,receiver_x_cord[ishot],receiver_interval,receiver_num,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
{
	
	int ix=threadIdx.x+blockIdx.x*blockDim.x;
	int iz=threadIdx.y+blockIdx.y*blockDim.y;

	int id,id1;

	int nnx;
	nnx=receiver_interval*receiver_num;

	if(ix<nnx&&iz<nz)
	{
		id=ix*nz+iz;

		id1=(ix+boundary_left+receiver_x_cord)*nz_append+iz+boundary_up;

		//if(iz>=z1&&iz<=z2)	wf_d[id]=velocity_all_d[id1];
		if(iz>=0&&iz<=z2)	wf_d[id]=velocity_all_d[id1];

		else			wf_d[id]=velocity_all_d[(ix+boundary_left+receiver_x_cord)*nz_append+z2+boundary_up];	
	}
}

__global__ void cuda_get_constant_mode(float *velocity_all_d,float *velocity_d,int nx,int nz)
{
	int ix=threadIdx.x+blockIdx.x*blockDim.x;
	int iz=threadIdx.y+blockIdx.y*blockDim.y;

	int id;

	if(ix<nx&&iz<nz)
	{
		id=ix*nz+iz;

		velocity_d[id]=velocity_all_d[0];	
	}

}

__global__ void cuda_get_partly_mode(float *velocity_all_d,float *wf_d,int nx,int nz,int receiver_x_cord,int receiver_interval,int receiver_num)
{
	
	int ix=threadIdx.x+blockIdx.x*blockDim.x;
	int iz=threadIdx.y+blockIdx.y*blockDim.y;

	int id,id1;

	int nnx;
	nnx=receiver_interval*receiver_num;

	if(ix<nnx&&iz<nz)
	{
		id=ix*nz+iz;

		id1=(ix+receiver_x_cord)*nz+iz;

		wf_d[id]=velocity_all_d[id1];	
	}
}

__global__ void cuda_get_partly_mode_z1_z2(float *velocity_all_d,float *wf_d,int nx,int nz,int receiver_x_cord,int receiver_interval,int receiver_num,int z1,int z2)
//cuda_get_partly_mode_z1_z2<<<dimGrid_new,dimBlock>>>(all_conj_vp_d,wf_d,nx,nz,receiver_x_cord[ishot],receiver_interval,receiver_num,precon_z1,precon_z2);
{
	
	int ix=threadIdx.x+blockIdx.x*blockDim.x;
	int iz=threadIdx.y+blockIdx.y*blockDim.y;

	int id,id1;

	int nnx;
	nnx=receiver_interval*receiver_num;

	if(ix<nnx&&iz<nz)
	{
		id=ix*nz+iz;

		id1=(ix+receiver_x_cord)*nz+iz;
		
		//if(iz>=z1&&iz<=z2)	wf_d[id]=velocity_all_d[id1];

		if(iz>=0&&iz<=z2)	wf_d[id]=velocity_all_d[id1];

		//else			wf_d[id]=velocity_all_d[(ix+receiver_x_cord)*nz+z2];
		else			wf_d[id]=0.0;	
	}
}

__global__ void cuda_sum_new_acqusition(float *all_vresultpp_d,float *vresultpp_d,int nx,int nz,int receiver_x_cord,int receiver_interval,int receiver_num)
{
	int ix=threadIdx.x+blockIdx.x*blockDim.x;
	int iz=threadIdx.y+blockIdx.y*blockDim.y;

	int id,id1;

	int nnx;
	nnx=receiver_interval*receiver_num;
	
	if(ix<nnx&&iz<nz)
	{
		id=ix*nz+iz;

		id1=(ix+receiver_x_cord)*nz+iz;

		all_vresultpp_d[id1]+=vresultpp_d[id];	
	}
}

__global__ void cuda_sum_new_acqusition_illum(float *d_illum_new,float *d_illum,int nx,int nz,int nx_append_new,int nz_append,int boundary_left,int boundary_up,int receiver_x_cord,int receiver_interval,int receiver_num)
{
	int ix=threadIdx.x+blockIdx.x*blockDim.x;
	int iz=threadIdx.y+blockIdx.y*blockDim.y;

	int id,id1;

	int nnx;
	nnx=receiver_interval*receiver_num;
	
	if(ix<nnx&&iz<nz)
	{
		id=(ix+boundary_left)*nz_append+iz+boundary_up;

		id1=(ix+receiver_x_cord+boundary_left)*nz_append+iz+boundary_up;

		d_illum_new[id1]+=d_illum[id];	
	}
}

__global__ void cauda_zero_acqusition_left(float *obs_shot_x_d,int acqusition_left,int receiver_x_cord,int receiver_interval,int receiver_num,int lt)
{
	int ix=threadIdx.x+blockIdx.x*blockDim.x;
	int iz=threadIdx.y+blockIdx.y*blockDim.y;

	int id;

	if(ix<receiver_num&&iz<lt)
	{
		id=ix*lt+iz;

		if(ix>(receiver_num*receiver_interval-acqusition_left))
			obs_shot_x_d[id]=0;

	}
}

__global__ void cauda_zero_acqusition_right(float *obs_shot_x_d,int acqusition_right,int receiver_x_cord,int receiver_interval,int receiver_num,int lt)
{
	int ix=threadIdx.x+blockIdx.x*blockDim.x;
	int iz=threadIdx.y+blockIdx.y*blockDim.y;

	int id;

	if(ix<receiver_num&&iz<lt)
	{
		id=ix*lt+iz;

		if(ix<acqusition_right)
			obs_shot_x_d[id]=0;
	}
}

__global__ void cauda_zero_acqusition_left_and_right(float *obs_shot_x_d,int offset_left,int offset_right,int source_x_cord,int receiver_offset,int receiver_num,int lt)
{
	int ix=threadIdx.x+blockIdx.x*blockDim.x;
	int iz=threadIdx.y+blockIdx.y*blockDim.y;

	int id;

	if(ix<receiver_num&&iz<lt)
	{
		id=ix*lt+iz;

		if(offset_left>receiver_offset)
		{
			if(ix>=0&&ix<(offset_left-receiver_offset))			
				obs_shot_x_d[id]=0;
		}

		if(offset_right>receiver_offset)
		{
			if(ix>=(source_x_cord+receiver_offset)&&ix<receiver_num)			
				obs_shot_x_d[id]=0;
		}		
	}
}

__global__ void cuda_expand_acqusition_left_and_right(float *velocity_d,int offset_left,int offset_right,int source_x_cord,int receiver_offset,int nx_size,int dimx,int dimz,int boundary_left,int boundary_up)
//cuda_expand_acqusition_left_and_right<<<dimGrid_new,dimBlock>>>(velocity_d,offset_left[ishot],offset_right[ishot],source_x_cord[ishot],receiver_offset,nx_size,nx_append,nz_append,boundary_left,boundary_up);
{
	int ix=threadIdx.x+blockIdx.x*blockDim.x;
	int iz=threadIdx.y+blockIdx.y*blockDim.y;

	int id;
	float change;

	if(ix<dimx&&iz<dimz)
	{
		//id=(ix+boundary_left)*dimz+iz+boundary_up;

		id=ix*dimz+iz;

		if(offset_left>receiver_offset)
		{
			change=velocity_d[(offset_left-receiver_offset+boundary_left)*dimz+iz];
		
			if(ix>=0&&ix<(offset_left-receiver_offset+boundary_left))			
				velocity_d[id]=change;
		}

		if(offset_right>receiver_offset)
		{
			change=velocity_d[(source_x_cord+receiver_offset+boundary_left)*dimz+iz];
			if(ix>=(source_x_cord+receiver_offset+boundary_left)&&ix<dimx)			
				velocity_d[id]=change;
		}	
	}
}

__global__ void smooth_acqusition(float *all_grad_density1_d,int nx,int nz,int *offset_left_d,int *offset_right_d,int *source_x_cord_d,int shot_num)
///smooth_acqusition<<<dimGrid_new,dimBlock>>>(all_grad_vp1_d,nx,nz,offset_left_d,offset_right_d,source_x_cord_d,shot_num);
{

	int ix=threadIdx.x+blockIdx.x*blockDim.x;
	int iz=threadIdx.y+blockIdx.y*blockDim.y;

	int id;
	int ishot;

	if(ix<nx&&iz<nz)
	{
		//id=ix*nz+iz;

		for(ishot=0;ishot<shot_num;ishot++)
		{
			//if(source_x_cord_d[ishot]-offset_left_d[ishot]!=0)
			if((source_x_cord_d[ishot]-offset_left_d[ishot])!=0)
			{
				id=(source_x_cord_d[ishot]-offset_left_d[ishot])*nz+iz;
				all_grad_density1_d[id]=(all_grad_density1_d[id+nz]+all_grad_density1_d[id-nz])/2.0;
				//all_grad_density1_d[id]=0;
			}

			//if(source_x_cord_d[ishot]+offset_right_d[ishot]!=nx)
			if((source_x_cord_d[ishot]+offset_right_d[ishot])!=nx)
			{
				id=(source_x_cord_d[ishot]+offset_right_d[ishot])*nz+iz;
				all_grad_density1_d[id]=(all_grad_density1_d[id+nz]+all_grad_density1_d[id-nz])/2.0;
				//all_grad_density1_d[id]=0;
			}
		}
	}
}

__global__ void cuda_attenuation_truncation(float *grad_vp1_d,int nx,int nz,int offset_left,int offset_right,int receiver_offset)
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

__global__ void cauda_zero_and_attenuation_truncation(float *grad_vp1_d,int nx,int nz,int offset_left,int offset_right,int receiver_offset,int offset_attenuation)
//cauda_zero_and_attenuation_truncation<<<dimGrid,dimBlock>>>(grad_den1_d,nx_size,nz,offset_left[ishot],offset_right[ishot],receiver_offset);
{
/////////////////////////attenuation for boundary value in part receiver <receiver_offset
	int ix = blockIdx.x*blockDim.x+threadIdx.x;
	int iz = blockIdx.y*blockDim.y+threadIdx.y;
	int in_idx;

	int distance_left;
	int distance_right;
	
	//int beg,end;

	float m=0.0;

	double change;

	if (ix<nx && iz<nz)
	{
		in_idx=ix*nz+iz;
		
		if(offset_left>receiver_offset)
		{
			if(ix>=0&&ix<(offset_left-receiver_offset))
				grad_vp1_d[in_idx]=0;

			distance_left=int(receiver_offset/offset_attenuation);

			if(distance_left<=2)		distance_left=3;///2017年09月05日 星期二 08时44分25秒  it is important,when I join offset_attenuation

			if(ix>=(offset_left-receiver_offset)&&ix<(offset_left-receiver_offset+distance_left)&&offset_left!=0)
			{
				m=1.0*(offset_left-receiver_offset+distance_left-ix);

				change=pow(cos(pai/2*m/distance_left),2);				

				grad_vp1_d[in_idx]=grad_vp1_d[in_idx]*change;
			}
		}

		if(offset_right>receiver_offset)
		{
			if(ix>=(offset_left+receiver_offset)&&ix<nx)
				grad_vp1_d[in_idx]=0;
			
			distance_right=int(receiver_offset/offset_attenuation);

			if(distance_right<=2)	distance_right=3;///2017年09月05日 星期二 08时44分25秒  it is important,when I join offset_attenuation

			if(ix>=(offset_left+receiver_offset-distance_right)&&ix<(offset_left+receiver_offset)&&offset_right!=0)
			{
				m=1.0*(ix-(offset_left+receiver_offset-distance_right));

				change=pow(cos(pai/2*m/distance_right),2);

				grad_vp1_d[in_idx]=grad_vp1_d[in_idx]*change;
			}
		}	
	
	}
}

__global__ void cauda_zero_and_attenuation_truncation_old(float *grad_vp1_d,int nx,int nz,int offset_left,int offset_right,int receiver_offset)
//cauda_zero_and_attenuation_truncation<<<dimGrid,dimBlock>>>(grad_den1_d,nx_size,nz,offset_left[ishot],offset_right[ishot],receiver_offset);
{
	int ix = blockIdx.x*blockDim.x+threadIdx.x;
	int iz = blockIdx.y*blockDim.y+threadIdx.y;
	int in_idx;

	int distance_left;
	int distance_right;
	
	//int beg,end;

	float m=0.0;

	double change;

	if (ix<nx && iz<nz)
	{
		in_idx=ix*nz+iz;
		
		if(offset_left>receiver_offset)
		{
			if(ix>=0&&ix<(offset_left-receiver_offset))
				grad_vp1_d[in_idx]=0;

			distance_left=int(receiver_offset/3);

			if(ix>=(offset_left-receiver_offset)&&ix<(offset_left-receiver_offset+distance_left)&&offset_left!=0)
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
			
			distance_right=int(receiver_offset/3);

			if(ix>=(offset_left+receiver_offset-distance_right)&&ix<(offset_left+receiver_offset)&&offset_right!=0)
			{
				m=1.0*(ix-(offset_left+receiver_offset-distance_right));

				change=pow(cos(pai/2*m/distance_right),3);

				grad_vp1_d[in_idx]=grad_vp1_d[in_idx]*change;
			}
		}	
	
	}
}
