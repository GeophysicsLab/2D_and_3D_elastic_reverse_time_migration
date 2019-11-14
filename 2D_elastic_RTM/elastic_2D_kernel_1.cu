__constant__ const int BDIMX=32;
__constant__ const int BDIMY=16;
__constant__ const int radius=6;

__global__ void add_source(float *txx_d,float *tzz_d,float *wavelet_d,int source_x_cord,int shot_depth,int it,int boundary_up,int boundary_left,int nz_append)
{
		txx_d[(boundary_left+source_x_cord)*nz_append+(boundary_up+shot_depth)]+=1000000000*wavelet_d[it];
		tzz_d[(boundary_left+source_x_cord)*nz_append+(boundary_up+shot_depth)]+=1000000000*wavelet_d[it];		
}

__global__ void wraddshot(float *wfr_d,float *fr_d,float *shotgather_d,float *shotgather1_d,int it,int lt,int nz_append,int boundary_up,int boundary_left,int receiver_start,int receiver_interval,int receiver_depth,int receiver_num)
//__global__ void wraddshot_new(float *wfr_d,float *fr_d,float *shotgather_d,float *shotgather1_d,int it,int lt,int receiver_num,int receiver_depth,int receiver_interval,int boundary_left,int boundary_up,int nz_append)
{
		int ix=blockIdx.x;

		if(ix<receiver_num)		
		{
			wfr_d[(boundary_left+receiver_start+ix*receiver_interval)*nz_append+boundary_up+receiver_depth]+=shotgather_d[ix*lt+it];
			 fr_d[(boundary_left+receiver_start+ix*receiver_interval)*nz_append+boundary_up+receiver_depth]+=shotgather1_d[ix*lt+it];
		}
}

__global__ void wraddshot_set(float *wfr_d,float *fr_d,float *shotgather_d,float *shotgather1_d,int it,int lt,int nz_append,int boundary_up,int boundary_left,int receiver_start,int receiver_interval,int receiver_depth,int receiver_num)
//__global__ void wraddshot_new(float *wfr_d,float *fr_d,float *shotgather_d,float *shotgather1_d,int it,int lt,int receiver_num,int receiver_depth,int receiver_interval,int boundary_left,int boundary_up,int nz_append)
{
		int ix=blockIdx.x;

		if(ix<receiver_num)		
		{
			wfr_d[(boundary_left+receiver_start+ix*receiver_interval)*nz_append+boundary_up+receiver_depth]=shotgather_d[ix*lt+it];
			 fr_d[(boundary_left+receiver_start+ix*receiver_interval)*nz_append+boundary_up+receiver_depth]=shotgather1_d[ix*lt+it];
		}
}

__global__ void wraddshot_x_z(float *wfr_d,float *shotgather_d,int it,int lt,int nz_append,int boundary_up,int boundary_left,int receiver_x_cord,int receiver_interval,int receiver_z_cord,int receiver_z_interval,int receiver_num,int mark)
//__global__ void wraddshot_new(float *wfr_d,float *fr_d,float *shotgather_d,float *shotgather1_d,int it,int lt,int receiver_num,int receiver_depth,int receiver_interval,int boundary_left,int boundary_up,int nz_append)
{
		int ix=blockIdx.x;

		int id_x,id_z,id;

		if(ix<receiver_num)		
		{
				id_x=boundary_left+receiver_x_cord+ix*receiver_interval;
				id_z=boundary_up+receiver_z_cord+ix*receiver_z_interval;

				id=id_x*nz_append+id_z;

				if(mark==0)	wfr_d[id]=shotgather_d[ix*lt+it];
				else		wfr_d[id]+=shotgather_d[ix*lt+it];
		}	
}

__global__ void wraddshot_x_z_acqusition(float *wfr_d,float *shotgather_d,int it,int lt,int nz_append,int boundary_up,int boundary_left,int receiver_x_cord,int receiver_interval,int receiver_z_cord,int receiver_z_interval,int receiver_num,int mark)
//__global__ void wraddshot_new(float *wfr_d,float *fr_d,float *shotgather_d,float *shotgather1_d,int it,int lt,int receiver_num,int receiver_depth,int receiver_interval,int boundary_left,int boundary_up,int nz_append)
{
		int ix=blockIdx.x;

		int id_x,id_z,id;

		if(ix<receiver_num)		
		{
				id_x=boundary_left+ix*receiver_interval;
				id_z=boundary_up+receiver_z_cord+ix*receiver_z_interval;

				id=id_x*nz_append+id_z;

				if(mark==0)	wfr_d[id]=shotgather_d[ix*lt+it];
				else		wfr_d[id]+=shotgather_d[ix*lt+it];
		}	
}

__global__ void wraddshot_ls(float *wfr_d,float *fr_d,float *shotgather_d,float *shotgather1_d,int it,int lt,int nz_append,int boundary_up,int boundary_left,int receiver_start,int receiver_interval,int receiver_depth,int receiver_num)
//__global__ void wraddshot_new(float *wfr_d,float *fr_d,float *shotgather_d,float *shotgather1_d,int it,int lt,int receiver_num,int receiver_depth,int receiver_interval,int boundary_left,int boundary_up,int nz_append)
{
		int ix=blockIdx.x;

		if(ix<receiver_num)		
		{
			wfr_d[(boundary_left+receiver_start+ix*receiver_interval)*nz_append+boundary_up+receiver_depth]+=-1*shotgather_d[ix*lt+it];
			 fr_d[(boundary_left+receiver_start+ix*receiver_interval)*nz_append+boundary_up+receiver_depth]+=-1*shotgather1_d[ix*lt+it];
		}
}

__global__ void write_shot(float *wf_d,float *f_d,float *shotgather_d,float *shotgather1_d,int it,int lt,int receiver_num,int receiver_depth,int receiver_x_cord,int receiver_interval,int boundary_left,int boundary_up,int nz_append,float dx,float dt,int source_x_cord,float *velocity1_d,float wavelet_half)
{
		int ix=blockIdx.x;

		if(ix<receiver_num)		
		{
				shotgather_d[ix*lt+it]=wf_d[(boundary_left+receiver_x_cord+ix*receiver_interval)*nz_append+boundary_up+receiver_depth];
				shotgather1_d[ix*lt+it]=f_d[(boundary_left+receiver_x_cord+ix*receiver_interval)*nz_append+boundary_up+receiver_depth];
		}	
}

__global__ void write_shot_x_z(float *wf_d,float *shotgather_d,int it,int lt,int receiver_num,int receiver_x_cord,int receiver_interval,int receiver_z_cord,int receiver_z_interval,int boundary_left,int boundary_up,int nz_append)
//write_shot_x_z<<<receiver_num,1>>>(vx2_d,obs_shot_x_d,it-wavelet_half,lt,receiver_num,receiver_x_cord[ishot],receiver_interval,receiver_z_cord[ishot],receiver_z_interval,boundary_left,boundary_up,nz_append);///for vsp 2017年03月14日 星期二 08时46分12秒 
{
		int ix=blockIdx.x;

		int id_x,id_z,id;

		if(ix<receiver_num)		
		{
				id_x=boundary_left+receiver_x_cord+ix*receiver_interval;
				id_z=boundary_up+receiver_z_cord+ix*receiver_z_interval;

				id=id_x*nz_append+id_z;

				shotgather_d[ix*lt+it]=wf_d[id];
		}	
}

__global__ void write_shot_x_z_acqusition(float *wf_d,float *shotgather_d,int it,int lt,int receiver_num,int receiver_x_cord,int receiver_interval,int receiver_z_cord,int receiver_z_interval,int boundary_left,int boundary_up,int nz_append)
//write_shot_x_z<<<receiver_num,1>>>(vx2_d,obs_shot_x_d,it-wavelet_half,lt,receiver_num,receiver_x_cord[ishot],receiver_interval,receiver_z_cord[ishot],receiver_z_interval,boundary_left,boundary_up,nz_append);///for vsp 2017年03月14日 星期二 08时46分12秒 
{
		int ix=blockIdx.x;

		int id_x,id_z,id;

		if(ix<receiver_num)		
		{
				id_x=boundary_left+ix*receiver_interval;
				id_z=boundary_up+receiver_z_cord+ix*receiver_z_interval;

				id=id_x*nz_append+id_z;

				shotgather_d[ix*lt+it]=wf_d[id];
		}	
}


__global__ void wraddshot_new(float *wfr_d,float *fr_d,float *shotgather_d,float *shotgather1_d,int it,int lt,int receiver_num,int receiver_depth,int receiver_interval,int boundary_left,int boundary_up,int nz_append)
//(rwf1_d,rwf1_d,res_shot_d,res_shot_d,it,lt,nz_append,boundary_up,boundary_left,receiver_x_cord[ishot],receiver_interval,receiver_depth,receiver_num);
{
		int ix=blockIdx.x;

		if(ix<receiver_num)		
		{
			wfr_d[(boundary_left+ix*receiver_interval)*nz_append+boundary_up+receiver_depth]=shotgather_d[ix*lt+it];
			 fr_d[(boundary_left+ix*receiver_interval)*nz_append+boundary_up+receiver_depth]=shotgather1_d[ix*lt+it];
		}
}

__global__ void write_shot_new(float *wf_d,float *f_d,float *shotgather_d,float *shotgather1_d,int it,int lt,int receiver_num,int receiver_depth,int receiver_interval,int boundary_left,int boundary_up,int nz_append)
{
		int ix=blockIdx.x;

		if(ix<receiver_num)		
		{
				shotgather_d[ix*lt+it]=wf_d[(boundary_left+ix*receiver_interval)*nz_append+boundary_up+receiver_depth];
				shotgather1_d[ix*lt+it]=f_d[(boundary_left+ix*receiver_interval)*nz_append+boundary_up+receiver_depth];
		}	
}

__global__ void cut_direct(float *shotgather_d,int lt,int source_x_cord,int shot_depth,int receiver_num,int receiver_depth,int receiver_x_cord,int receiver_interval,int boundary_left,int boundary_up,int nz_append,float dx,float dz,float dt,float *velocity_d,int wavelet_half)
{
		int ix=blockIdx.x*blockDim.x+threadIdx.x;

		int mark;

		float dt_real;
		dt_real=dt/1000;
		float distance;
		int time;

		if(ix<receiver_num)
		{
			
			distance=sqrt((receiver_x_cord+ix*receiver_interval-source_x_cord)*(receiver_x_cord+ix*receiver_interval-source_x_cord)*dx*dx*1.0+(shot_depth-receiver_depth)*(shot_depth-receiver_depth)*dz*dz);	
	
			time=distance*1.0/velocity_d[(ix+boundary_left)*nz_append+boundary_up]/dt_real;

			//for(mark=time;mark<time+250;mark++)
			for(mark=0;mark<time+250;mark++)
			//for(mark=0;mark<time+2*wavelet_half;mark++)
			shotgather_d[ix*lt+mark]=0;

			//for(mark=time+200;mark<=time+220;mark++)
			//shotgather_d[ix*lt+mark]=shotgather_d[ix*lt+mark]*float(exp(1.0*(mark-time-220.0)/0.10));
		}
}

__global__ void cut_direct_new1(float *shotgather_d,int lt,int source_x_cord,int shot_depth,int receiver_num,int receiver_depth,int receiver_x_cord,int receiver_interval,int boundary_left,int boundary_up,int nz_append,float dx,float dz,float dt,float *velocity_d,int wavelet_half,int cut_direct_wave)
{
		int ix=blockIdx.x*blockDim.x+threadIdx.x;

		int mark;

		float dt_real;
		dt_real=dt/1000;
		float distance;
		int time;

		if(ix<receiver_num)
		{
			
			distance=sqrt((receiver_x_cord+ix*receiver_interval-source_x_cord)*(receiver_x_cord+ix*receiver_interval-source_x_cord)*dx*dx*1.0+(shot_depth-receiver_depth)*(shot_depth-receiver_depth)*dz*dz);	
	
			time=distance*1.0/velocity_d[(ix+boundary_left)*nz_append+boundary_up]/dt_real;

			//for(mark=time;mark<time+250;mark++)
			//for(mark=0;mark<time+200;mark++)
			//for(mark=0;mark<time+2*wavelet_half;mark++)
			for(mark=0;mark<time+cut_direct_wave;mark++)
			shotgather_d[ix*lt+mark]=0;

			//for(mark=time+200;mark<=time+220;mark++)
			//shotgather_d[ix*lt+mark]=shotgather_d[ix*lt+mark]*float(exp(1.0*(mark-time-220.0)/0.10));
		}
}

__global__ void cut_direct_new(float *shotgather_d,int lt,int source_x_cord,int shot_depth,int receiver_num,int receiver_depth,int receiver_x_cord,int receiver_interval,int boundary_left,int boundary_up,int nz_append,float dx,float dz,float dt,float *velocity_d,int wavelet_half)
{
		int ix=blockIdx.x*blockDim.x+threadIdx.x;

		int mark;

		float dt_real;
		dt_real=dt/1000;
		float distance;
		int time;

		if(ix<receiver_num)
		{
			
			distance=sqrt((receiver_x_cord+ix*receiver_interval-source_x_cord)*(receiver_x_cord+ix*receiver_interval-source_x_cord)*dx*dx*1.0+(shot_depth-receiver_depth)*(shot_depth-receiver_depth)*dz*dz);	
	
			time=distance*1.0/velocity_d[(ix+boundary_left)*nz_append+boundary_up]/dt_real;

			//for(mark=time;mark<time+250;mark++)
			for(mark=0;mark<time+250;mark++)
				shotgather_d[ix*lt+mark]=shotgather_d[ix*lt+mark]*float(exp(-1.0*(time+250-mark)/(time+250)));
		}
}

__global__ void replace(float *c1,float *c2,float *c3,float *c4,float *c5,float *c6,float *c7,float *c8,float *c9,float *c10,int nx_append,int nz_append)
{
		int ix=blockIdx.x*blockDim.x+threadIdx.x;
		int iz=blockIdx.y*blockDim.y+threadIdx.y;

		int in_idx;
		float change1;
		float change2;
		float change3;
		float change4;
		float change5;

		if((ix<nx_append)&&(iz<nz_append))
		{
				ix=ix+radius;iz=iz+radius;
				in_idx=ix*(nz_append+2*radius)+iz;
				change1   =c1[in_idx];
				c1[in_idx]=c2[in_idx];
				c2[in_idx]=    change1;

				change2   =c3[in_idx];
				c3[in_idx]=c4[in_idx];
				c4[in_idx]=    change2;

				change3   =c5[in_idx];
				c5[in_idx]=c6[in_idx];
				c6[in_idx]=    change3;

				change4   =c7[in_idx];
				c7[in_idx]=c8[in_idx];
				c8[in_idx]=    change4;

				change5   =c9[in_idx];
				c9[in_idx]=c10[in_idx];
				c10[in_idx]=    change5;			
		}
}

__global__ void replace_2wf(float *c1,float *c2,int nx_append,int nz_append)
{
		int ix=blockIdx.x*blockDim.x+threadIdx.x;
		int iz=blockIdx.y*blockDim.y+threadIdx.y;

		int in_idx;
		float change1;
		
		if((ix<nx_append)&&(iz<nz_append))
		{
				ix=ix+radius;iz=iz+radius;
				in_idx=ix*(nz_append+2*radius)+iz;
				change1   =c1[in_idx];
				c1[in_idx]=c2[in_idx];
				c2[in_idx]=    change1;			
		}
}


__global__ void fwd_vxp_vzp_vxs_vzs(float *vxp2_d,float *vxp1_d,float *vzp2_d,float *vzp1_d,float *vxs2_d,float *vxs1_d,float *vzs2_d,float *vzs1_d,float *txx1_d,float *tzz1_d,float *txz1_d,float *velocity_d,float *velocity1_d,float *attenuation_d,float dt,float *coe_d,float coe_x,float coe_z,int dimx,int dimz,float *density_d)
//vxp2_d,vxp1_d,vzp2_d,vzp1_d,vxs2_d,vxs1_d,vzs2_d,vzs1_d,txx1_d,tzz1_d,txz1_d,velocity_d,velocity1_d,attenuation_d,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,density_d
{
		__shared__ float s_data1[BDIMY+2*radius][BDIMX+2*radius];
		__shared__ float s_data2[BDIMY+2*radius][BDIMX+2*radius];
		__shared__ float s_data3[BDIMY+2*radius][BDIMX+2*radius];
		
		float dt_real=dt/1000;
		float s_velocity;
		float s_velocity1;
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

		s_data3[tz][tx]=0.0;
		s_data3[threadIdx.y][threadIdx.x]=0.0;
		s_data3[BDIMY+2*radius-1-threadIdx.y][BDIMX+2*radius-1-threadIdx.x]=0.0;
		s_data3[threadIdx.y][BDIMX+2*radius-1-threadIdx.x]=0.0;
		s_data3[BDIMY+2*radius-1-threadIdx.y][threadIdx.x]=0.0;

		if((ix<dimx)&&(iz<dimz))
		{
				dimx=dimx+2*radius;dimz=dimz+2*radius;
				ix=ix+radius;iz=iz+radius;
				in_idx = ix*dimz+iz;//iz*dimx+ix;

				__syncthreads();

				s_data1[tz][tx]=txx1_d[in_idx];
				s_data2[tz][tx]=tzz1_d[in_idx];
				s_data3[tz][tx]=txz1_d[in_idx];

				if(threadIdx.y<radius)
				{
						s_data1[threadIdx.y][tx]=txx1_d[in_idx-radius];//g_input[in_idx-radius*dimx];//up
						s_data1[threadIdx.y+BDIMY+radius][tx]=txx1_d[in_idx+BDIMY];//g_input[in_idx+BDIMY*dimx];//down

						s_data2[threadIdx.y][tx]=tzz1_d[in_idx-radius];//g_input[in_idx-radius*dimx];//up
						s_data2[threadIdx.y+BDIMY+radius][tx]=tzz1_d[in_idx+BDIMY];//g_input[in_idx+BDIMY*dimx];//down

						s_data3[threadIdx.y][tx]=txz1_d[in_idx-radius];//g_input[in_idx-radius*dimx];//up
						s_data3[threadIdx.y+BDIMY+radius][tx]=txz1_d[in_idx+BDIMY];//g_input[in_idx+BDIMY*dimx];//down
				}
				if(threadIdx.x<radius)
				{
						s_data1[tz][threadIdx.x]=txx1_d[in_idx-radius*dimz];//g_input[in_idx-radius];//left
						s_data1[tz][threadIdx.x+BDIMX+radius]=txx1_d[in_idx+BDIMX*dimz];//g_input[in_idx+BDIMX];//right

						s_data2[tz][threadIdx.x]=tzz1_d[in_idx-radius*dimz];//g_input[in_idx-radius];//left
						s_data2[tz][threadIdx.x+BDIMX+radius]=tzz1_d[in_idx+BDIMX*dimz];//g_input[in_idx+BDIMX];//right

						s_data3[tz][threadIdx.x]=txz1_d[in_idx-radius*dimz];//g_input[in_idx-radius];//left
						s_data3[tz][threadIdx.x+BDIMX+radius]=txz1_d[in_idx+BDIMX*dimz];//g_input[in_idx+BDIMX];//right
				}
			
				s_velocity=velocity_d[in_idx]*velocity_d[in_idx];
				s_velocity1=velocity1_d[in_idx]*velocity1_d[in_idx];
				s_attenuation=attenuation_d[in_idx];
				__syncthreads();

//sumx1:the derivation of x direction of txx
				float    sumx1=coe_d[1]*(s_data1[tz][tx]-s_data1[tz][tx-1]);
					sumx1+=coe_d[2]*(s_data1[tz][tx+1]-s_data1[tz][tx-2]);
					sumx1+=coe_d[3]*(s_data1[tz][tx+2]-s_data1[tz][tx-3]);
					sumx1+=coe_d[4]*(s_data1[tz][tx+3]-s_data1[tz][tx-4]);
					sumx1+=coe_d[5]*(s_data1[tz][tx+4]-s_data1[tz][tx-5]);
					sumx1+=coe_d[6]*(s_data1[tz][tx+5]-s_data1[tz][tx-6]);
//sumx2:the derivation of x direction of tzz
				float    sumx2=coe_d[1]*(s_data2[tz][tx]  -s_data2[tz][tx-1]);
					sumx2+=coe_d[2]*(s_data2[tz][tx+1]-s_data2[tz][tx-2]);
					sumx2+=coe_d[3]*(s_data2[tz][tx+2]-s_data2[tz][tx-3]);
					sumx2+=coe_d[4]*(s_data2[tz][tx+3]-s_data2[tz][tx-4]);
					sumx2+=coe_d[5]*(s_data2[tz][tx+4]-s_data2[tz][tx-5]);
					sumx2+=coe_d[6]*(s_data2[tz][tx+5]-s_data2[tz][tx-6]);
//sumx3:the derivation of x direction of txz
				float    sumx3=coe_d[1]*(s_data3[tz][tx+1]-s_data3[tz][tx]);
					sumx3+=coe_d[2]*(s_data3[tz][tx+2]-s_data3[tz][tx-1]);
					sumx3+=coe_d[3]*(s_data3[tz][tx+3]-s_data3[tz][tx-2]);
					sumx3+=coe_d[4]*(s_data3[tz][tx+4]-s_data3[tz][tx-3]);
					sumx3+=coe_d[5]*(s_data3[tz][tx+5]-s_data3[tz][tx-4]);
					sumx3+=coe_d[6]*(s_data3[tz][tx+6]-s_data3[tz][tx-5]);
//sumz1:the derivation of z direction of txx				
				float    sumz1=coe_d[1]*(s_data1[tz+1][tx]-s_data1[tz][tx]);
					sumz1+=coe_d[2]*(s_data1[tz+2][tx]-s_data1[tz-1][tx]);
					sumz1+=coe_d[3]*(s_data1[tz+3][tx]-s_data1[tz-2][tx]);
					sumz1+=coe_d[4]*(s_data1[tz+4][tx]-s_data1[tz-3][tx]);
					sumz1+=coe_d[5]*(s_data1[tz+5][tx]-s_data1[tz-4][tx]);
					sumz1+=coe_d[6]*(s_data1[tz+6][tx]-s_data1[tz-5][tx]);
//sumz2:the derivation of z direction of tzz			
				float    sumz2=coe_d[1]*(s_data2[tz+1][tx]-s_data2[tz][tx]);
					sumz2+=coe_d[2]*(s_data2[tz+2][tx]-s_data2[tz-1][tx]);
					sumz2+=coe_d[3]*(s_data2[tz+3][tx]-s_data2[tz-2][tx]);
					sumz2+=coe_d[4]*(s_data2[tz+4][tx]-s_data2[tz-3][tx]);
					sumz2+=coe_d[5]*(s_data2[tz+5][tx]-s_data2[tz-4][tx]);
					sumz2+=coe_d[6]*(s_data2[tz+6][tx]-s_data2[tz-5][tx]);
//sumz3:the derivation of z direction of txz
				float    sumz3=coe_d[1]*(s_data3[tz][tx]-s_data3[tz-1][tx]);  ////s_data2..... is  a   fault  
					sumz3+=coe_d[2]*(s_data3[tz+1][tx]-s_data3[tz-2][tx]);
					sumz3+=coe_d[3]*(s_data3[tz+2][tx]-s_data3[tz-3][tx]);
					sumz3+=coe_d[4]*(s_data3[tz+3][tx]-s_data3[tz-4][tx]);
					sumz3+=coe_d[5]*(s_data3[tz+4][tx]-s_data3[tz-5][tx]);
					sumz3+=coe_d[6]*(s_data3[tz+5][tx]-s_data3[tz-6][tx]);

					      
				vxp2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*vxp1_d[in_idx]+
(1.0/density_d[in_idx])*(s_velocity/(2*s_velocity-2*s_velocity1))*(sumx1*coe_x+sumx2*coe_x));

				vzp2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*vzp1_d[in_idx]+
(1.0/density_d[in_idx])*(s_velocity/(2*s_velocity-2*s_velocity1))*(sumz1*coe_z+sumz2*coe_z));
				
				vxs2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*vxs1_d[in_idx]+
(1.0/density_d[in_idx])*(sumz3*coe_z-(s_velocity)/(2*s_velocity-2*s_velocity1)*sumx2*coe_x+(s_velocity-2*s_velocity1)/(2*s_velocity-2*s_velocity1)*sumx1*coe_x));

				vzs2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*vzs1_d[in_idx]+
(1.0/density_d[in_idx])*(sumx3*coe_x-(s_velocity)/(2*s_velocity-2*s_velocity1)*sumz1*coe_z+(s_velocity-2*s_velocity1)/(2*s_velocity-2*s_velocity1)*sumz2*coe_z));
		}
}

__global__ void fwd_sum_vx_vz(float *vx2_d,float *vz2_d,float *vxp2_d,float *vzp2_d,float *vxs2_d,float *vzs2_d,int dimx,int dimz)
{
		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;
		
		if(ix<dimx&&iz<dimz)
		{
			ix=ix+radius;
			iz=iz+radius;
			dimx=dimx+2*radius;
			dimz=dimz+2*radius;			
			in_idx=ix*dimz+iz;

			vx2_d[in_idx]=vxs2_d[in_idx]+vxp2_d[in_idx];
			vz2_d[in_idx]=vzs2_d[in_idx]+vzp2_d[in_idx];			
		}

}

__global__ void fwd_txxzzxz(float *txx2,float *txx1,float *tzz2,float *tzz1,float *txz2,float *txz1,float *vx2,float *vz2,float *velocity_d,float *velocity1_d,float *attenuation_d,float dt,float *coe_d,float coe_x,float coe_z,int dimx,int dimz,float *density_d)
//txx2_d,txx1_d,tzz2_d,tzz1_d,txz2_d,txz1_d,vx2_d,vz2_d,velocity_d,velocity1_d,attenuation_d,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius
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

				s_velocity=velocity_d[in_idx]*velocity_d[in_idx];
				s_velocity1=velocity1_d[in_idx]*velocity1_d[in_idx];
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


				txx2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*txx1[in_idx]+
						s_velocity*density_d[in_idx]*sumx*coe_x+(s_velocity-2*s_velocity1)*density_d[in_idx]*sumz*coe_z);//s_velocity  and  s_velocity1

				tzz2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*tzz1[in_idx]+
						(s_velocity-2*s_velocity1)*density_d[in_idx]*sumx*coe_x+s_velocity*density_d[in_idx]*sumz*coe_z);//sumx  and  sumz 
							
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
						((1.0-s_attenuation*dt_real/2.0)*txz1[in_idx]+s_velocity1*density_d[in_idx]*(sumx1*coe_x+sumz1*coe_z));
		}
}

__global__ void fwd_txxzzxz_new(float *vx_x_d,float *vx_z_d,float *vz_x_d,float *vz_z_d,float *txx2,float *txx1,float *tzz2,float *tzz1,float *txz2,float *txz1,float *vx2,float *vz2,float *velocity_d,float *velocity1_d,float *attenuation_d,float dt,float dx,float dz,float *coe_d,float coe_x,float coe_z,int dimx,int dimz,float *density_d)
//<<<dimGrid,dimBlock>>>(vx_x_d,vx_z_d,vz_x_d,vz_z_d,txx2_d,txx1_d,tzz2_d,tzz1_d,txz2_d,txz1_d,vx1_d,vz1_d,s_velocity_d,s_velocity1_d,attenuation_d,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,s_density_d);
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

				s_velocity=velocity_d[in_idx]*velocity_d[in_idx];
				s_velocity1=velocity1_d[in_idx]*velocity1_d[in_idx];
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

				vx_x_d[in_idx]=sumx*1.0/dx;

				vz_z_d[in_idx]=sumz*1.0/dz;


				txx2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*txx1[in_idx]+
						s_velocity*density_d[in_idx]*sumx*coe_x+(s_velocity-2*s_velocity1)*density_d[in_idx]*sumz*coe_z);//s_velocity  and  s_velocity1

				tzz2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*tzz1[in_idx]+
						(s_velocity-2*s_velocity1)*density_d[in_idx]*sumx*coe_x+s_velocity*density_d[in_idx]*sumz*coe_z);//sumx  and  sumz 
							
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

				txz2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*
						((1.0-s_attenuation*dt_real/2.0)*txz1[in_idx]+s_velocity1*density_d[in_idx]*(sumx1*coe_x+sumz1*coe_z));
		}
}


__global__ void rfwd_vxp_vzp_vxs_vzs(float *vxp2_d,float *vxp1_d,float *vzp2_d,float *vzp1_d,float *vxs2_d,float *vxs1_d,float *vzs2_d,float *vzs1_d,float *txx1_d,float *tzz1_d,float *txz1_d,float *velocity_d,float *velocity1_d,float *attenuation_d,float dt,float *coe_d,float coe_x,float coe_z,int dimx,int dimz,float *density_d)
//vxp2_d,vxp1_d,vzp2_d,vzp1_d,vxs2_d,vxs1_d,vzs2_d,vzs1_d,txx1_d,tzz1_d,txz1_d,velocity_d,velocity1_d,attenuation_d,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,density_d
{
		__shared__ float s_data1[BDIMY+2*radius][BDIMX+2*radius];
		__shared__ float s_data2[BDIMY+2*radius][BDIMX+2*radius];
		__shared__ float s_data3[BDIMY+2*radius][BDIMX+2*radius];
		
		float dt_real=dt/1000;
		float s_velocity;
		float s_velocity1;
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

		s_data3[tz][tx]=0.0;
		s_data3[threadIdx.y][threadIdx.x]=0.0;
		s_data3[BDIMY+2*radius-1-threadIdx.y][BDIMX+2*radius-1-threadIdx.x]=0.0;
		s_data3[threadIdx.y][BDIMX+2*radius-1-threadIdx.x]=0.0;
		s_data3[BDIMY+2*radius-1-threadIdx.y][threadIdx.x]=0.0;

		if((ix<dimx)&&(iz<dimz))
		{
				dimx=dimx+2*radius;dimz=dimz+2*radius;
				ix=ix+radius;iz=iz+radius;
				in_idx = ix*dimz+iz;//iz*dimx+ix;

				__syncthreads();

				s_data1[tz][tx]=txx1_d[in_idx];
				s_data2[tz][tx]=tzz1_d[in_idx];
				s_data3[tz][tx]=txz1_d[in_idx];

				if(threadIdx.y<radius)
				{
						s_data1[threadIdx.y][tx]=txx1_d[in_idx-radius];//g_input[in_idx-radius*dimx];//up
						s_data1[threadIdx.y+BDIMY+radius][tx]=txx1_d[in_idx+BDIMY];//g_input[in_idx+BDIMY*dimx];//down

						s_data2[threadIdx.y][tx]=tzz1_d[in_idx-radius];//g_input[in_idx-radius*dimx];//up
						s_data2[threadIdx.y+BDIMY+radius][tx]=tzz1_d[in_idx+BDIMY];//g_input[in_idx+BDIMY*dimx];//down

						s_data3[threadIdx.y][tx]=txz1_d[in_idx-radius];//g_input[in_idx-radius*dimx];//up
						s_data3[threadIdx.y+BDIMY+radius][tx]=txz1_d[in_idx+BDIMY];//g_input[in_idx+BDIMY*dimx];//down
				}
				if(threadIdx.x<radius)
				{
						s_data1[tz][threadIdx.x]=txx1_d[in_idx-radius*dimz];//g_input[in_idx-radius];//left
						s_data1[tz][threadIdx.x+BDIMX+radius]=txx1_d[in_idx+BDIMX*dimz];//g_input[in_idx+BDIMX];//right

						s_data2[tz][threadIdx.x]=tzz1_d[in_idx-radius*dimz];//g_input[in_idx-radius];//left
						s_data2[tz][threadIdx.x+BDIMX+radius]=tzz1_d[in_idx+BDIMX*dimz];//g_input[in_idx+BDIMX];//right

						s_data3[tz][threadIdx.x]=txz1_d[in_idx-radius*dimz];//g_input[in_idx-radius];//left
						s_data3[tz][threadIdx.x+BDIMX+radius]=txz1_d[in_idx+BDIMX*dimz];//g_input[in_idx+BDIMX];//right
				}
			
				s_velocity=velocity_d[in_idx]*velocity_d[in_idx];
				s_velocity1=velocity1_d[in_idx]*velocity1_d[in_idx];
				s_attenuation=attenuation_d[in_idx];
				__syncthreads();

//sumx1:the derivation of x direction of txx
				float    sumx1=coe_d[1]*(s_data1[tz][tx]-s_data1[tz][tx-1]);
					sumx1+=coe_d[2]*(s_data1[tz][tx+1]-s_data1[tz][tx-2]);
					sumx1+=coe_d[3]*(s_data1[tz][tx+2]-s_data1[tz][tx-3]);
					sumx1+=coe_d[4]*(s_data1[tz][tx+3]-s_data1[tz][tx-4]);
					sumx1+=coe_d[5]*(s_data1[tz][tx+4]-s_data1[tz][tx-5]);
					sumx1+=coe_d[6]*(s_data1[tz][tx+5]-s_data1[tz][tx-6]);
//sumx2:the derivation of x direction of tzz
				float    sumx2=coe_d[1]*(s_data2[tz][tx]  -s_data2[tz][tx-1]);
					sumx2+=coe_d[2]*(s_data2[tz][tx+1]-s_data2[tz][tx-2]);
					sumx2+=coe_d[3]*(s_data2[tz][tx+2]-s_data2[tz][tx-3]);
					sumx2+=coe_d[4]*(s_data2[tz][tx+3]-s_data2[tz][tx-4]);
					sumx2+=coe_d[5]*(s_data2[tz][tx+4]-s_data2[tz][tx-5]);
					sumx2+=coe_d[6]*(s_data2[tz][tx+5]-s_data2[tz][tx-6]);
//sumx3:the derivation of x direction of txz
				float    sumx3=coe_d[1]*(s_data3[tz][tx+1]-s_data3[tz][tx]);
					sumx3+=coe_d[2]*(s_data3[tz][tx+2]-s_data3[tz][tx-1]);
					sumx3+=coe_d[3]*(s_data3[tz][tx+3]-s_data3[tz][tx-2]);
					sumx3+=coe_d[4]*(s_data3[tz][tx+4]-s_data3[tz][tx-3]);
					sumx3+=coe_d[5]*(s_data3[tz][tx+5]-s_data3[tz][tx-4]);
					sumx3+=coe_d[6]*(s_data3[tz][tx+6]-s_data3[tz][tx-5]);
//sumz1:the derivation of z direction of txx				
				float    sumz1=coe_d[1]*(s_data1[tz+1][tx]-s_data1[tz][tx]);
					sumz1+=coe_d[2]*(s_data1[tz+2][tx]-s_data1[tz-1][tx]);
					sumz1+=coe_d[3]*(s_data1[tz+3][tx]-s_data1[tz-2][tx]);
					sumz1+=coe_d[4]*(s_data1[tz+4][tx]-s_data1[tz-3][tx]);
					sumz1+=coe_d[5]*(s_data1[tz+5][tx]-s_data1[tz-4][tx]);
					sumz1+=coe_d[6]*(s_data1[tz+6][tx]-s_data1[tz-5][tx]);
//sumz2:the derivation of z direction of tzz			
				float    sumz2=coe_d[1]*(s_data2[tz+1][tx]-s_data2[tz][tx]);
					sumz2+=coe_d[2]*(s_data2[tz+2][tx]-s_data2[tz-1][tx]);
					sumz2+=coe_d[3]*(s_data2[tz+3][tx]-s_data2[tz-2][tx]);
					sumz2+=coe_d[4]*(s_data2[tz+4][tx]-s_data2[tz-3][tx]);
					sumz2+=coe_d[5]*(s_data2[tz+5][tx]-s_data2[tz-4][tx]);
					sumz2+=coe_d[6]*(s_data2[tz+6][tx]-s_data2[tz-5][tx]);
//sumz3:the derivation of z direction of txz
				float    sumz3=coe_d[1]*(s_data3[tz][tx]-s_data3[tz-1][tx]);  ////s_data2..... is  a   fault  
					sumz3+=coe_d[2]*(s_data3[tz+1][tx]-s_data3[tz-2][tx]);
					sumz3+=coe_d[3]*(s_data3[tz+2][tx]-s_data3[tz-3][tx]);
					sumz3+=coe_d[4]*(s_data3[tz+3][tx]-s_data3[tz-4][tx]);
					sumz3+=coe_d[5]*(s_data3[tz+4][tx]-s_data3[tz-5][tx]);
					sumz3+=coe_d[6]*(s_data3[tz+5][tx]-s_data3[tz-6][tx]);

					      
				vxp2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*vxp1_d[in_idx]-
(1.0/density_d[in_idx])*(s_velocity/(2*s_velocity-2*s_velocity1))*(sumx1*coe_x+sumx2*coe_x));

				vzp2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*vzp1_d[in_idx]-
(1.0/density_d[in_idx])*(s_velocity/(2*s_velocity-2*s_velocity1))*(sumz1*coe_z+sumz2*coe_z));
				
				vxs2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*vxs1_d[in_idx]-
(1.0/density_d[in_idx])*(sumz3*coe_z-(s_velocity)/(2*s_velocity-2*s_velocity1)*sumx2*coe_x+(s_velocity-2*s_velocity1)/(2*s_velocity-2*s_velocity1)*sumx1*coe_x));

				vzs2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*vzs1_d[in_idx]-
(1.0/density_d[in_idx])*(sumx3*coe_x-(s_velocity)/(2*s_velocity-2*s_velocity1)*sumz1*coe_z+(s_velocity-2*s_velocity1)/(2*s_velocity-2*s_velocity1)*sumz2*coe_z));
		}
}


__global__ void rfwd_txxzzxz(float *txx2,float *txx1,float *tzz2,float *tzz1,float *txz2,float *txz1,float *vx2,float *vz2,float *velocity_d,float *velocity1_d,float *attenuation_d,float dt,float *coe_d,float coe_x,float coe_z,int dimx,int dimz,float *density_d)
//txx2_d,txx1_d,tzz2_d,tzz1_d,txz2_d,txz1_d,vx2_d,vz2_d,velocity_d,velocity1_d,attenuation_d,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius
{
		__shared__ float s_data1[BDIMY+2*radius][BDIMX+2*radius];
		__shared__ float s_data2[BDIMY+2*radius][BDIMX+2*radius];
		float dt_real=dt/1000;
		float s_velocity;
		float s_velocity1;
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


				txx2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*txx1[in_idx]-
						s_velocity*density_d[in_idx]*sumx*coe_x-(s_velocity-2*s_velocity1)*density_d[in_idx]*sumz*coe_z);//s_velocity  and  s_velocity1

				tzz2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*tzz1[in_idx]-
						(s_velocity-2*s_velocity1)*density_d[in_idx]*sumx*coe_x-s_velocity*density_d[in_idx]*sumz*coe_z);//sumx  and  sumz 
							
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
						((1.0-s_attenuation*dt_real/2.0)*txz1[in_idx]-s_velocity1*density_d[in_idx]*(sumx1*coe_x+sumz1*coe_z));
		}
}

__global__ void rfwd_txxzzxz_new(float *vx_x_d,float *vx_z_d,float *vz_x_d,float *vz_z_d,float *txx2,float *txx1,float *tzz2,float *tzz1,float *txz2,float *txz1,float *vx2,float *vz2,float *velocity_d,float *velocity1_d,float *attenuation_d,float dt,float dx,float dz,float *coe_d,float coe_x,float coe_z,int dimx,int dimz,float *density_d)
//rfwd_txxzzxz_new<<<dimGrid,dimBlock>>>(vx_x_d,vx_z_d,vz_x_d,vz_z_d,txx1_d,txx2_d,tzz1_d,tzz2_d,txz1_d,txz2_d,vx2_d,vz2_d,s_velocity_d,s_velocity1_d,attenuation_d,dt,dx,dz,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,s_density_d);
{
		__shared__ float s_data1[BDIMY+2*radius][BDIMX+2*radius];
		__shared__ float s_data2[BDIMY+2*radius][BDIMX+2*radius];
		float dt_real=dt/1000;
		float s_velocity;
		float s_velocity1;
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


				txx2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*txx1[in_idx]-
						s_velocity*density_d[in_idx]*sumx*coe_x-(s_velocity-2*s_velocity1)*density_d[in_idx]*sumz*coe_z);//s_velocity  and  s_velocity1

				tzz2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*tzz1[in_idx]-
						(s_velocity-2*s_velocity1)*density_d[in_idx]*sumx*coe_x-s_velocity*density_d[in_idx]*sumz*coe_z);//sumx  and  sumz 
							
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
			

				txz2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*
						((1.0-s_attenuation*dt_real/2.0)*txz1[in_idx]-s_velocity1*density_d[in_idx]*(sumx1*coe_x+sumz1*coe_z));
		}
}

__global__ void fwd_vx(float *vx2,float *vx1,float *txx1,float *txz1,float *attenuation_d,float dx,float dz,float dt,float *coe_d,float coe_x,float coe_z,int dimx,int dimz,float *density_d)
//vx2_d,vx1_d,txx1_d,txz1_d,attenuation_d,dx,dz,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius
{
		__shared__ float s_data1[BDIMY+2*radius][BDIMX+2*radius];
		__shared__ float s_data2[BDIMY+2*radius][BDIMX+2*radius];
		float dt_real=dt/1000;
	//	float s_velocity;
		float s_attenuation;

		float density;

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

				s_data1[tz][tx]=txx1[in_idx];
				s_data2[tz][tx]=txz1[in_idx];

				if(threadIdx.y<radius)
				{
						s_data1[threadIdx.y][tx]=txx1[in_idx-radius];//g_input[in_idx-radius*dimx];//up
						s_data1[threadIdx.y+BDIMY+radius][tx]=txx1[in_idx+BDIMY];//g_input[in_idx+BDIMY*dimx];//down

						s_data2[threadIdx.y][tx]=txz1[in_idx-radius];//g_input[in_idx-radius*dimx];//up
						s_data2[threadIdx.y+BDIMY+radius][tx]=txz1[in_idx+BDIMY];//g_input[in_idx+BDIMY*dimx];//down
				}
				if(threadIdx.x<radius)
				{
						s_data1[tz][threadIdx.x]=txx1[in_idx-radius*dimz];//g_input[in_idx-radius];//left
						s_data1[tz][threadIdx.x+BDIMX+radius]=txx1[in_idx+BDIMX*dimz];//g_input[in_idx+BDIMX];//right

						s_data2[tz][threadIdx.x]=txz1[in_idx-radius*dimz];//g_input[in_idx-radius];//left
						s_data2[tz][threadIdx.x+BDIMX+radius]=txz1[in_idx+BDIMX*dimz];//g_input[in_idx+BDIMX];//right
				}

				//s_velocity=velocity_d[in_idx]*velocity_d[in_idx];
		
				density=density_d[in_idx];			
		
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
						((1.0-s_attenuation*dt_real/2.0)*vx1[in_idx]+1.0/density*(sumx*coe_x+sumz*coe_z));
		}
}

__global__ void fwd_vx_new(float* vx_t_d,float *vx2,float *vx1,float *txx1,float *txz1,float *attenuation_d,float dx,float dz,float dt,float *coe_d,float coe_x,float coe_z,int dimx,int dimz,float *density_d)
//vx2_d,vx1_d,txx1_d,txz1_d,attenuation_d,dx,dz,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius
{
		__shared__ float s_data1[BDIMY+2*radius][BDIMX+2*radius];
		__shared__ float s_data2[BDIMY+2*radius][BDIMX+2*radius];
		float dt_real=dt/1000;
	//	float s_velocity;
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

				s_data1[tz][tx]=txx1[in_idx];
				s_data2[tz][tx]=txz1[in_idx];

				if(threadIdx.y<radius)
				{
						s_data1[threadIdx.y][tx]=txx1[in_idx-radius];//g_input[in_idx-radius*dimx];//up
						s_data1[threadIdx.y+BDIMY+radius][tx]=txx1[in_idx+BDIMY];//g_input[in_idx+BDIMY*dimx];//down

						s_data2[threadIdx.y][tx]=txz1[in_idx-radius];//g_input[in_idx-radius*dimx];//up
						s_data2[threadIdx.y+BDIMY+radius][tx]=txz1[in_idx+BDIMY];//g_input[in_idx+BDIMY*dimx];//down
				}
				if(threadIdx.x<radius)
				{
						s_data1[tz][threadIdx.x]=txx1[in_idx-radius*dimz];//g_input[in_idx-radius];//left
						s_data1[tz][threadIdx.x+BDIMX+radius]=txx1[in_idx+BDIMX*dimz];//g_input[in_idx+BDIMX];//right

						s_data2[tz][threadIdx.x]=txz1[in_idx-radius*dimz];//g_input[in_idx-radius];//left
						s_data2[tz][threadIdx.x+BDIMX+radius]=txz1[in_idx+BDIMX*dimz];//g_input[in_idx+BDIMX];//right
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

				vx_t_d[in_idx]=1.0/density_d[in_idx]*(sumx*coe_x+sumz*coe_z)/dt_real;
		}
}

__global__ void fwd_vx_new_new(float *d_illum,float* vx_t_d,float *vx2,float *vx1,float *txx1,float *txz1,float *attenuation_d,float dx,float dz,float dt,float *coe_d,float coe_x,float coe_z,int dimx,int dimz,float *density_d)
//vx2_d,vx1_d,txx1_d,txz1_d,attenuation_d,dx,dz,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius
{
		__shared__ float s_data1[BDIMY+2*radius][BDIMX+2*radius];
		__shared__ float s_data2[BDIMY+2*radius][BDIMX+2*radius];
		float dt_real=dt/1000;
	//	float s_velocity;
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

				s_data1[tz][tx]=txx1[in_idx];
				s_data2[tz][tx]=txz1[in_idx];

				if(threadIdx.y<radius)
				{
						s_data1[threadIdx.y][tx]=txx1[in_idx-radius];//g_input[in_idx-radius*dimx];//up
						s_data1[threadIdx.y+BDIMY+radius][tx]=txx1[in_idx+BDIMY];//g_input[in_idx+BDIMY*dimx];//down

						s_data2[threadIdx.y][tx]=txz1[in_idx-radius];//g_input[in_idx-radius*dimx];//up
						s_data2[threadIdx.y+BDIMY+radius][tx]=txz1[in_idx+BDIMY];//g_input[in_idx+BDIMY*dimx];//down
				}
				if(threadIdx.x<radius)
				{
						s_data1[tz][threadIdx.x]=txx1[in_idx-radius*dimz];//g_input[in_idx-radius];//left
						s_data1[tz][threadIdx.x+BDIMX+radius]=txx1[in_idx+BDIMX*dimz];//g_input[in_idx+BDIMX];//right

						s_data2[tz][threadIdx.x]=txz1[in_idx-radius*dimz];//g_input[in_idx-radius];//left
						s_data2[tz][threadIdx.x+BDIMX+radius]=txz1[in_idx+BDIMX*dimz];//g_input[in_idx+BDIMX];//right
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

				vx_t_d[in_idx]=1.0/density_d[in_idx]*(sumx*coe_x+sumz*coe_z)/dt_real;

				d_illum[in_idx]+=vx2[in_idx]*vx2[in_idx];
		}
}

__global__ void fwd_vz(float *vz2,float *vz1,float *tzz1,float *txz1,float *attenuation_d,float dx,float dz,float dt,float *coe_d,float coe_x,float coe_z,int dimx,int dimz,float *density_d)
{
		__shared__ float s_data1[BDIMY+2*radius][BDIMX+2*radius];
		__shared__ float s_data2[BDIMY+2*radius][BDIMX+2*radius];
		float dt_real=dt/1000;
	//	float s_velocity;
			
		float density;

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

				s_data1[tz][tx]=tzz1[in_idx];
				s_data2[tz][tx]=txz1[in_idx];

				if(threadIdx.y<radius)
				{
						s_data1[threadIdx.y][tx]=tzz1[in_idx-radius];//g_input[in_idx-radius*dimx];//up
						s_data1[threadIdx.y+BDIMY+radius][tx]=tzz1[in_idx+BDIMY];//g_input[in_idx+BDIMY*dimx];//down
						s_data2[threadIdx.y][tx]=txz1[in_idx-radius];//g_input[in_idx-radius*dimx];//up
						s_data2[threadIdx.y+BDIMY+radius][tx]=txz1[in_idx+BDIMY];//g_input[in_idx+BDIMY*dimx];//down
				}
				if(threadIdx.x<radius)
				{
						s_data1[tz][threadIdx.x]=tzz1[in_idx-radius*dimz];//g_input[in_idx-radius];//left
						s_data1[tz][threadIdx.x+BDIMX+radius]=tzz1[in_idx+BDIMX*dimz];//g_input[in_idx+BDIMX];//right
						s_data2[tz][threadIdx.x]=txz1[in_idx-radius*dimz];//g_input[in_idx-radius];//left
						s_data2[tz][threadIdx.x+BDIMX+radius]=txz1[in_idx+BDIMX*dimz];//g_input[in_idx+BDIMX];//right
				}

				density=density_d[in_idx];

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
						((1.0-s_attenuation*dt_real/2.0)*vz1[in_idx]+1.0/density*(sumx*coe_x+sumz*coe_z));
		}
}

__global__ void fwd_vz_new(float *vz_t_d,float *vz2,float *vz1,float *tzz1,float *txz1,float *attenuation_d,float dx,float dz,float dt,float *coe_d,float coe_x,float coe_z,int dimx,int dimz,float *density_d)
{
		__shared__ float s_data1[BDIMY+2*radius][BDIMX+2*radius];
		__shared__ float s_data2[BDIMY+2*radius][BDIMX+2*radius];
		float dt_real=dt/1000;
	//	float s_velocity;
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

				s_data1[tz][tx]=tzz1[in_idx];
				s_data2[tz][tx]=txz1[in_idx];

				if(threadIdx.y<radius)
				{
						s_data1[threadIdx.y][tx]=tzz1[in_idx-radius];//g_input[in_idx-radius*dimx];//up
						s_data1[threadIdx.y+BDIMY+radius][tx]=tzz1[in_idx+BDIMY];//g_input[in_idx+BDIMY*dimx];//down
						s_data2[threadIdx.y][tx]=txz1[in_idx-radius];//g_input[in_idx-radius*dimx];//up
						s_data2[threadIdx.y+BDIMY+radius][tx]=txz1[in_idx+BDIMY];//g_input[in_idx+BDIMY*dimx];//down
				}
				if(threadIdx.x<radius)
				{
						s_data1[tz][threadIdx.x]=tzz1[in_idx-radius*dimz];//g_input[in_idx-radius];//left
						s_data1[tz][threadIdx.x+BDIMX+radius]=tzz1[in_idx+BDIMX*dimz];//g_input[in_idx+BDIMX];//right
						s_data2[tz][threadIdx.x]=txz1[in_idx-radius*dimz];//g_input[in_idx-radius];//left
						s_data2[tz][threadIdx.x+BDIMX+radius]=txz1[in_idx+BDIMX*dimz];//g_input[in_idx+BDIMX];//right
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

				vz_t_d[in_idx]=1.0/density_d[in_idx]*(sumx*coe_x+sumz*coe_z)/dt_real;
		}
}

__global__ void fwd_vz_new_new(float *d_illum,float *vz_t_d,float *vz2,float *vz1,float *tzz1,float *txz1,float *attenuation_d,float dx,float dz,float dt,float *coe_d,float coe_x,float coe_z,int dimx,int dimz,float *density_d)
{
		__shared__ float s_data1[BDIMY+2*radius][BDIMX+2*radius];
		__shared__ float s_data2[BDIMY+2*radius][BDIMX+2*radius];
		float dt_real=dt/1000;
	//	float s_velocity;
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

				s_data1[tz][tx]=tzz1[in_idx];
				s_data2[tz][tx]=txz1[in_idx];

				if(threadIdx.y<radius)
				{
						s_data1[threadIdx.y][tx]=tzz1[in_idx-radius];//g_input[in_idx-radius*dimx];//up
						s_data1[threadIdx.y+BDIMY+radius][tx]=tzz1[in_idx+BDIMY];//g_input[in_idx+BDIMY*dimx];//down
						s_data2[threadIdx.y][tx]=txz1[in_idx-radius];//g_input[in_idx-radius*dimx];//up
						s_data2[threadIdx.y+BDIMY+radius][tx]=txz1[in_idx+BDIMY];//g_input[in_idx+BDIMY*dimx];//down
				}
				if(threadIdx.x<radius)
				{
						s_data1[tz][threadIdx.x]=tzz1[in_idx-radius*dimz];//g_input[in_idx-radius];//left
						s_data1[tz][threadIdx.x+BDIMX+radius]=tzz1[in_idx+BDIMX*dimz];//g_input[in_idx+BDIMX];//right
						s_data2[tz][threadIdx.x]=txz1[in_idx-radius*dimz];//g_input[in_idx-radius];//left
						s_data2[tz][threadIdx.x+BDIMX+radius]=txz1[in_idx+BDIMX*dimz];//g_input[in_idx+BDIMX];//right
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

				vz_t_d[in_idx]=1.0/density_d[in_idx]*(sumx*coe_x+sumz*coe_z)/dt_real;

				d_illum[in_idx]+=vz2[in_idx]*vz2[in_idx];
		}
}

__global__ void fwd_vxp_vzp(float *vxp2_d,float *vxp1_d,float *vzp2_d,float *vzp1_d,float *tp1_d,float coe_x,float coe_z,float dx,float dz,float dt,float *attenuation,float *coe_d,int dimx,int dimz,float *density_d)
//fwd_vxp_vzp<<<dimGrid,dimBlock>>>(rvxp1_d,rvxp2_d,rvzp1_d,rvzp2_d,rtp1_d,coe_x,coe_z,dx,dz,dt,attenuation_d,coe_opt_d,nx_append_radius,nz_append_radius,s_density_d);	
{
		__shared__ float s_data[BDIMY+2*radius][BDIMX+2*radius];

		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		float dt_real=dt/1000;

		float s_attenuation;

		int tx = threadIdx.x+radius;
		int tz = threadIdx.y+radius;

		s_data[tz][tx]=0.0;
		s_data[threadIdx.y][threadIdx.x]=0.0;
		s_data[BDIMY+2*radius-1-threadIdx.y][BDIMX+2*radius-1-threadIdx.x]=0.0;
		s_data[threadIdx.y][BDIMX+2*radius-1-threadIdx.x]=0.0;
		s_data[BDIMY+2*radius-1-threadIdx.y][threadIdx.x]=0.0;

		if(ix<dimx&&iz<dimz)
		{
			ix=ix+radius,iz=iz+radius;
			dimx=dimx+2*radius;dimz=dimz+2*radius;	
			in_idx=ix*dimz+iz;
			__syncthreads();

			s_data[tz][tx]=tp1_d[in_idx];
				
			if(threadIdx.y<radius)
			{
				s_data[threadIdx.y][tx]=tp1_d[in_idx-radius];//g_input[in_idx-radius*dimx];//up
				s_data[threadIdx.y+BDIMY+radius][tx]=tp1_d[in_idx+BDIMY];//g_input[in_idx+BDIMY*dimx];//down
			}
			if(threadIdx.x<radius)
			{
				s_data[tz][threadIdx.x]=tp1_d[in_idx-radius*dimz];//g_input[in_idx-radius];//left
				s_data[tz][threadIdx.x+BDIMX+radius]=tp1_d[in_idx+BDIMX*dimz];//g_input[in_idx+BDIMX];//right
			}
			
			s_attenuation=attenuation[in_idx];
			__syncthreads();


			float    	sumx= coe_d[1]*(s_data[tz][tx]-s_data[tz][tx-1]);
					sumx+=coe_d[2]*(s_data[tz][tx+1]-s_data[tz][tx-2]);
					sumx+=coe_d[3]*(s_data[tz][tx+2]-s_data[tz][tx-3]);
					sumx+=coe_d[4]*(s_data[tz][tx+3]-s_data[tz][tx-4]);
					sumx+=coe_d[5]*(s_data[tz][tx+4]-s_data[tz][tx-5]);
					sumx+=coe_d[6]*(s_data[tz][tx+5]-s_data[tz][tx-6]);

			float    	sumz=coe_d[1]* (s_data[tz+1][tx]-s_data[tz][tx]);
					sumz+=coe_d[2]*(s_data[tz+2][tx]-s_data[tz-1][tx]);
					sumz+=coe_d[3]*(s_data[tz+3][tx]-s_data[tz-2][tx]);
					sumz+=coe_d[4]*(s_data[tz+4][tx]-s_data[tz-3][tx]);
					sumz+=coe_d[5]*(s_data[tz+5][tx]-s_data[tz-4][tx]);
					sumz+=coe_d[6]*(s_data[tz+6][tx]-s_data[tz-5][tx]);


			/*float    	sumx=coe_d[1]*(s_data[tz][tx+1]- s_data[tz][tx]);
					sumx+=coe_d[2]*(s_data[tz][tx+2]-s_data[tz][tx-1]);
					sumx+=coe_d[3]*(s_data[tz][tx+3]-s_data[tz][tx-2]);
					sumx+=coe_d[4]*(s_data[tz][tx+4]-s_data[tz][tx-3]);
					sumx+=coe_d[5]*(s_data[tz][tx+5]-s_data[tz][tx-4]);
					sumx+=coe_d[6]*(s_data[tz][tx+6]-s_data[tz][tx-5]);

			float    	sumz=coe_d[1]*(s_data[tz][tx]-  s_data[tz-1][tx]);
					sumz+=coe_d[2]*(s_data[tz+1][tx]-s_data[tz-2][tx]);
					sumz+=coe_d[3]*(s_data[tz+2][tx]-s_data[tz-3][tx]);
					sumz+=coe_d[4]*(s_data[tz+3][tx]-s_data[tz-4][tx]);
					sumz+=coe_d[5]*(s_data[tz+4][tx]-s_data[tz-5][tx]);
					sumz+=coe_d[6]*(s_data[tz+5][tx]-s_data[tz-6][tx]);*/


			//vxp2_d[in_idx]=vxp1_d[in_idx]+(1.0/density_d[in_idx])*sumx*coe_x;
			vxp2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*vxp1_d[in_idx]+(1.0/density_d[in_idx])*sumx*coe_x);

			vzp2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*vzp1_d[in_idx]+(1.0/density_d[in_idx])*sumz*coe_z);
			//vzp2_d[in_idx]=vzp1_d[in_idx]+(1.0/density_d[in_idx])*sumz*coe_z;
		}
}

__global__ void vp_vs(float *vx2_d,float *vz2_d,float *vxp2_d,float *vzp2_d,float *vxs2_d,float *vzs2_d,int dimx,int dimz)
//vp_vs<<<dimGrid,dimBlock>>>(rvx1_d,rvz1_d,rvxp1_d,rvzp1_d,rvxs1_d,rvzs1_d,nx_append_radius,nz_append_radius);
{
		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;
		
		if(ix<dimx&&iz<dimz)
		{
			ix=ix+radius;
			iz=iz+radius;
			dimx=dimx+2*radius;
			dimz=dimz+2*radius;			
			in_idx=ix*dimz+iz;

			vxs2_d[in_idx]=vx2_d[in_idx]-vxp2_d[in_idx];

			vzs2_d[in_idx]=vz2_d[in_idx]-vzp2_d[in_idx];			
		}
}

__global__ void fwd_txxzzxzpp(float *tp2,float *tp1,float *txx2,float *txx1,float *tzz2,float *tzz1,float *txz2,float *txz1,float *vx2,float *vz2,float *velocity_d,float *velocity1_d,float *attenuation_d,float dt,float *coe_d,float coe_x,float coe_z,int dimx,int dimz,float *density_d)
//fwd_txxzzxzpp<<<dimGrid,dimBlock>>>(tp2_d,tp1_d,txx2_d,txx1_d,tzz2_d,tzz1_d,txz2_d,txz1_d,vx2_d,vz2_d,s_velocity_d,s_velocity1_d,attenuation_d,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,s_density_d);
{
		__shared__ float s_data1[BDIMY+2*radius][BDIMX+2*radius];
		__shared__ float s_data2[BDIMY+2*radius][BDIMX+2*radius];
		float dt_real=dt/1000;
		float s_velocity;
		float s_velocity1;	

		float density;

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

				density=density_d[in_idx];

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
						s_velocity*density*sumx*coe_x+(s_velocity-2*s_velocity1)*density*sumz*coe_z);//s_velocity  and  s_velocity1

				tzz2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*tzz1[in_idx]+
						(s_velocity-2*s_velocity1)*density*sumx*coe_x+s_velocity*density*sumz*coe_z);//sumx  and  sumz 
				
				tp2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*tp1[in_idx]+s_velocity*density*sumx*coe_x+s_velocity*density*sumz*coe_z);	
					
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
						((1.0-s_attenuation*dt_real/2.0)*txz1[in_idx]+s_velocity1*density*(sumx1*coe_x+sumz1*coe_z));
		}
}

__global__ void fwd_txxzzxzpp_new(float *vx_x_d,float *vx_z_d,float *vz_x_d,float *vz_z_d,float *tp2,float *tp1,float *txx2,float *txx1,float *tzz2,float *tzz1,float *txz2,float *txz1,float *vx2,float *vz2,float *velocity_d,float *velocity1_d,float *attenuation_d,float dt,float dx,float dz,float *coe_d,float coe_x,float coe_z,int dimx,int dimz,float *density_d)
//fwd_txxzzxzpp_new<<<dimGrid,dimBlock>>>(vx_x_d,vx_z_d,vz_x_d,vz_z_d,tp2_d,tp1_d,txx2_d,txx1_d,tzz2_d,tzz1_d,txz2_d,txz1_d,vx2_d,vz2_d,s_velocity_d,s_velocity1_d,attenuation_d,dt,dx,dz,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,s_density_d);
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
				
				s_velocity=velocity_d[in_idx]*velocity_d[in_idx];
				s_velocity1=velocity1_d[in_idx]*velocity1_d[in_idx];

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

				vx_x_d[in_idx]=sumx*1.0/dx;

				vz_z_d[in_idx]=sumz*1.0/dz;


				txx2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*txx1[in_idx]+
						s_velocity*density_d[in_idx]*sumx*coe_x+(s_velocity-2*s_velocity1)*density_d[in_idx]*sumz*coe_z);//s_velocity  and  s_velocity1

				tzz2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*tzz1[in_idx]+
						(s_velocity-2*s_velocity1)*density_d[in_idx]*sumx*coe_x+s_velocity*density_d[in_idx]*sumz*coe_z);//sumx  and  sumz 
				
				tp2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*tp1[in_idx]+s_velocity*density_d[in_idx]*sumx*coe_x+s_velocity*density_d[in_idx]*sumz*coe_z);	
					
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

				//vx_z_d[in_idx]=sumx1*1.0/dz;//This is a fault, which leads to the distortion of the graident of the vs

				//vz_x_d[in_idx]=sumz1*1.0/dx;//This is a fault, which leads to the distortion of the graident of the vs

				vx_z_d[in_idx]=sumz1*1.0/dz;/////no 

				vz_x_d[in_idx]=sumx1*1.0/dx;/////no 

				txz2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*
						((1.0-s_attenuation*dt_real/2.0)*txz1[in_idx]+s_velocity1*density_d[in_idx]*(sumx1*coe_x+sumz1*coe_z));
		}
}

__global__ void rfwd_vx(float *vx2,float *vx1,float *txx1,float *txz1,float *attenuation_d,float dx,float dz,float dt,float *coe_d,float coe_x,float coe_z,int dimx,int dimz,int nx,int nz,int boundary_left,int boundary_up,float *density_d)
//vx2_d,vx1_d,txx1_d,txz1_d,attenuation_d,dx,dz,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius
{
		__shared__ float s_data1[BDIMY+2*radius][BDIMX+2*radius];
		__shared__ float s_data2[BDIMY+2*radius][BDIMX+2*radius];
		float dt_real=dt/1000;
	//	float s_velocity;
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

				s_data1[tz][tx]=txx1[in_idx];
				s_data2[tz][tx]=txz1[in_idx];

				if(threadIdx.y<radius)
				{
						s_data1[threadIdx.y][tx]=txx1[in_idx-radius];//g_input[in_idx-radius*dimx];//up
						s_data1[threadIdx.y+BDIMY+radius][tx]=txx1[in_idx+BDIMY];//g_input[in_idx+BDIMY*dimx];//down

						s_data2[threadIdx.y][tx]=txz1[in_idx-radius];//g_input[in_idx-radius*dimx];//up
						s_data2[threadIdx.y+BDIMY+radius][tx]=txz1[in_idx+BDIMY];//g_input[in_idx+BDIMY*dimx];//down
				}
				if(threadIdx.x<radius)
				{
						s_data1[tz][threadIdx.x]=txx1[in_idx-radius*dimz];//g_input[in_idx-radius];//left
						s_data1[tz][threadIdx.x+BDIMX+radius]=txx1[in_idx+BDIMX*dimz];//g_input[in_idx+BDIMX];//right

						s_data2[tz][threadIdx.x]=txz1[in_idx-radius*dimz];//g_input[in_idx-radius];//left
						s_data2[tz][threadIdx.x+BDIMX+radius]=txz1[in_idx+BDIMX*dimz];//g_input[in_idx+BDIMX];//right
				}

				//s_velocity=velocity_d[in_idx]*velocity_d[in_idx];
				s_attenuation=attenuation_d[in_idx];
				__syncthreads();

				float    sumx=coe_d[1]*(s_data1[tz][tx]-  s_data1[tz][tx-1]);
					sumx+=coe_d[2]*(s_data1[tz][tx+1]-s_data1[tz][tx-2]);
					sumx+=coe_d[3]*(s_data1[tz][tx+2]-s_data1[tz][tx-3]);
					sumx+=coe_d[4]*(s_data1[tz][tx+3]-s_data1[tz][tx-4]);
					sumx+=coe_d[5]*(s_data1[tz][tx+4]-s_data1[tz][tx-5]);
					sumx+=coe_d[6]*(s_data1[tz][tx+5]-s_data1[tz][tx-6]);

				float    sumz=coe_d[1]*(s_data2[tz][tx]-  s_data2[tz-1][tx]);
					sumz+=coe_d[2]*(s_data2[tz+1][tx]-s_data2[tz-2][tx]);
					sumz+=coe_d[3]*(s_data2[tz+2][tx]-s_data2[tz-3][tx]);
					sumz+=coe_d[4]*(s_data2[tz+3][tx]-s_data2[tz-4][tx]);
					sumz+=coe_d[5]*(s_data2[tz+4][tx]-s_data2[tz-5][tx]);
					sumz+=coe_d[6]*(s_data2[tz+5][tx]-s_data2[tz-6][tx]);

				vx2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*
						((1.0-s_attenuation*dt_real/2.0)*vx1[in_idx]-1.0/density_d[in_idx]*(sumx*coe_x+sumz*coe_z));
		}
}

__global__ void rfwd_vx_new(float *d_illum,float *d_illum_t,float *vx_t_d,float *vx2,float *vx1,float *txx1,float *txz1,float *attenuation_d,float dx,float dz,float dt,float *coe_d,float coe_x,float coe_z,int dimx,int dimz,int nx,int nz,int boundary_left,int boundary_up,float *density_d)
//vx2_d,vx1_d,txx1_d,txz1_d,attenuation_d,dx,dz,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius
{
		__shared__ float s_data1[BDIMY+2*radius][BDIMX+2*radius];
		__shared__ float s_data2[BDIMY+2*radius][BDIMX+2*radius];
		float dt_real=dt/1000;
	//	float s_velocity;
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

				s_data1[tz][tx]=txx1[in_idx];
				s_data2[tz][tx]=txz1[in_idx];

				if(threadIdx.y<radius)
				{
						s_data1[threadIdx.y][tx]=txx1[in_idx-radius];//g_input[in_idx-radius*dimx];//up
						s_data1[threadIdx.y+BDIMY+radius][tx]=txx1[in_idx+BDIMY];//g_input[in_idx+BDIMY*dimx];//down

						s_data2[threadIdx.y][tx]=txz1[in_idx-radius];//g_input[in_idx-radius*dimx];//up
						s_data2[threadIdx.y+BDIMY+radius][tx]=txz1[in_idx+BDIMY];//g_input[in_idx+BDIMY*dimx];//down
				}
				if(threadIdx.x<radius)
				{
						s_data1[tz][threadIdx.x]=txx1[in_idx-radius*dimz];//g_input[in_idx-radius];//left
						s_data1[tz][threadIdx.x+BDIMX+radius]=txx1[in_idx+BDIMX*dimz];//g_input[in_idx+BDIMX];//right

						s_data2[tz][threadIdx.x]=txz1[in_idx-radius*dimz];//g_input[in_idx-radius];//left
						s_data2[tz][threadIdx.x+BDIMX+radius]=txz1[in_idx+BDIMX*dimz];//g_input[in_idx+BDIMX];//right
				}

				//s_velocity=velocity_d[in_idx]*velocity_d[in_idx];
				s_attenuation=attenuation_d[in_idx];
				__syncthreads();

				float    sumx=coe_d[1]*(s_data1[tz][tx]-  s_data1[tz][tx-1]);
					sumx+=coe_d[2]*(s_data1[tz][tx+1]-s_data1[tz][tx-2]);
					sumx+=coe_d[3]*(s_data1[tz][tx+2]-s_data1[tz][tx-3]);
					sumx+=coe_d[4]*(s_data1[tz][tx+3]-s_data1[tz][tx-4]);
					sumx+=coe_d[5]*(s_data1[tz][tx+4]-s_data1[tz][tx-5]);
					sumx+=coe_d[6]*(s_data1[tz][tx+5]-s_data1[tz][tx-6]);

				float    sumz=coe_d[1]*(s_data2[tz][tx]-  s_data2[tz-1][tx]);
					sumz+=coe_d[2]*(s_data2[tz+1][tx]-s_data2[tz-2][tx]);
					sumz+=coe_d[3]*(s_data2[tz+2][tx]-s_data2[tz-3][tx]);
					sumz+=coe_d[4]*(s_data2[tz+3][tx]-s_data2[tz-4][tx]);
					sumz+=coe_d[5]*(s_data2[tz+4][tx]-s_data2[tz-5][tx]);
					sumz+=coe_d[6]*(s_data2[tz+5][tx]-s_data2[tz-6][tx]);

				vx2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*
						((1.0-s_attenuation*dt_real/2.0)*vx1[in_idx]-1.0/density_d[in_idx]*(sumx*coe_x+sumz*coe_z));

				vx_t_d[in_idx]=1.0/density_d[in_idx]*(sumx*coe_x+sumz*coe_z)/dt_real;

				d_illum_t[in_idx]=d_illum_t[in_idx]+vx_t_d[in_idx]*vx_t_d[in_idx];

				d_illum[in_idx]=d_illum[in_idx]+vx2[in_idx]*vx2[in_idx];
		}
}

__global__ void rfwd_vz(float *vz2,float *vz1,float *tzz1,float *txz1,float *attenuation_d,float dx,float dz,float dt,float *coe_d,float coe_x,float coe_z,int dimx,int dimz,int nx,int nz,int boundary_left,int boundary_up,float *density_d)
{
		__shared__ float s_data1[BDIMY+2*radius][BDIMX+2*radius];
		__shared__ float s_data2[BDIMY+2*radius][BDIMX+2*radius];
		float dt_real=dt/1000;
	//	float s_velocity;
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

				s_data1[tz][tx]=tzz1[in_idx];
				s_data2[tz][tx]=txz1[in_idx];

				if(threadIdx.y<radius)
				{
						s_data1[threadIdx.y][tx]=tzz1[in_idx-radius];//g_input[in_idx-radius*dimx];//up
						s_data1[threadIdx.y+BDIMY+radius][tx]=tzz1[in_idx+BDIMY];//g_input[in_idx+BDIMY*dimx];//down
						s_data2[threadIdx.y][tx]=txz1[in_idx-radius];//g_input[in_idx-radius*dimx];//up
						s_data2[threadIdx.y+BDIMY+radius][tx]=txz1[in_idx+BDIMY];//g_input[in_idx+BDIMY*dimx];//down
				}
				if(threadIdx.x<radius)
				{
						s_data1[tz][threadIdx.x]=tzz1[in_idx-radius*dimz];//g_input[in_idx-radius];//left
						s_data1[tz][threadIdx.x+BDIMX+radius]=tzz1[in_idx+BDIMX*dimz];//g_input[in_idx+BDIMX];//right
						s_data2[tz][threadIdx.x]=txz1[in_idx-radius*dimz];//g_input[in_idx-radius];//left
						s_data2[tz][threadIdx.x+BDIMX+radius]=txz1[in_idx+BDIMX*dimz];//g_input[in_idx+BDIMX];//right
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
						((1.0-s_attenuation*dt_real/2.0)*vz1[in_idx]-1.0/density_d[in_idx]*(sumx*coe_x+sumz*coe_z));
		}
}

__global__ void rfwd_vz_new(float *d_illum,float *d_illum_t,float *vz_t_d,float *vz2,float *vz1,float *tzz1,float *txz1,float *attenuation_d,float dx,float dz,float dt,float *coe_d,float coe_x,float coe_z,int dimx,int dimz,int nx,int nz,int boundary_left,int boundary_up,float *density_d)
{
		__shared__ float s_data1[BDIMY+2*radius][BDIMX+2*radius];
		__shared__ float s_data2[BDIMY+2*radius][BDIMX+2*radius];
		float dt_real=dt/1000;
	//	float s_velocity;
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

				s_data1[tz][tx]=tzz1[in_idx];
				s_data2[tz][tx]=txz1[in_idx];

				if(threadIdx.y<radius)
				{
						s_data1[threadIdx.y][tx]=tzz1[in_idx-radius];//g_input[in_idx-radius*dimx];//up
						s_data1[threadIdx.y+BDIMY+radius][tx]=tzz1[in_idx+BDIMY];//g_input[in_idx+BDIMY*dimx];//down
						s_data2[threadIdx.y][tx]=txz1[in_idx-radius];//g_input[in_idx-radius*dimx];//up
						s_data2[threadIdx.y+BDIMY+radius][tx]=txz1[in_idx+BDIMY];//g_input[in_idx+BDIMY*dimx];//down
				}
				if(threadIdx.x<radius)
				{
						s_data1[tz][threadIdx.x]=tzz1[in_idx-radius*dimz];//g_input[in_idx-radius];//left
						s_data1[tz][threadIdx.x+BDIMX+radius]=tzz1[in_idx+BDIMX*dimz];//g_input[in_idx+BDIMX];//right
						s_data2[tz][threadIdx.x]=txz1[in_idx-radius*dimz];//g_input[in_idx-radius];//left
						s_data2[tz][threadIdx.x+BDIMX+radius]=txz1[in_idx+BDIMX*dimz];//g_input[in_idx+BDIMX];//right
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
						((1.0-s_attenuation*dt_real/2.0)*vz1[in_idx]-1.0/density_d[in_idx]*(sumx*coe_x+sumz*coe_z));

				vz_t_d[in_idx]=1.0/density_d[in_idx]*(sumx*coe_x+sumz*coe_z)/dt_real;

				d_illum_t[in_idx]=d_illum_t[in_idx]+vz_t_d[in_idx]*vz_t_d[in_idx];

				d_illum[in_idx]=d_illum[in_idx]+vz2[in_idx]*vz2[in_idx];
		}
}

__global__ void rfwd_vxp_vzp(float *vxp2_d,float *vxp1_d,float *vzp2_d,float *vzp1_d,float *tp1_d,float coe_x,float coe_z,float dx,float dz,float dt,float *attenuation,float *coe_d,int dimx,int dimz,float *density_d)
{
		__shared__ float s_data[BDIMY+2*radius][BDIMX+2*radius];

		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		float dt_real=dt/1000;

		float s_attenuation;

		int tx = threadIdx.x+radius;
		int tz = threadIdx.y+radius;

		s_data[tz][tx]=0.0;
		s_data[threadIdx.y][threadIdx.x]=0.0;
		s_data[BDIMY+2*radius-1-threadIdx.y][BDIMX+2*radius-1-threadIdx.x]=0.0;
		s_data[threadIdx.y][BDIMX+2*radius-1-threadIdx.x]=0.0;
		s_data[BDIMY+2*radius-1-threadIdx.y][threadIdx.x]=0.0;

		if(ix<dimx&&iz<dimz)
		{
			ix=ix+radius,iz=iz+radius;
			dimx=dimx+2*radius;dimz=dimz+2*radius;	
			in_idx=ix*dimz+iz;
			__syncthreads();

			s_data[tz][tx]=tp1_d[in_idx];
				
			if(threadIdx.y<radius)
			{
				s_data[threadIdx.y][tx]=tp1_d[in_idx-radius];//g_input[in_idx-radius*dimx];//up
				s_data[threadIdx.y+BDIMY+radius][tx]=tp1_d[in_idx+BDIMY];//g_input[in_idx+BDIMY*dimx];//down
			}
			if(threadIdx.x<radius)
			{
				s_data[tz][threadIdx.x]=tp1_d[in_idx-radius*dimz];//g_input[in_idx-radius];//left
				s_data[tz][threadIdx.x+BDIMX+radius]=tp1_d[in_idx+BDIMX*dimz];//g_input[in_idx+BDIMX];//right
			}
			
			s_attenuation=attenuation[in_idx];
			__syncthreads();


			float    	sumx= coe_d[1]*(s_data[tz][tx]-	 s_data[tz][tx-1]);
					sumx+=coe_d[2]*(s_data[tz][tx+1]-s_data[tz][tx-2]);
					sumx+=coe_d[3]*(s_data[tz][tx+2]-s_data[tz][tx-3]);
					sumx+=coe_d[4]*(s_data[tz][tx+3]-s_data[tz][tx-4]);
					sumx+=coe_d[5]*(s_data[tz][tx+4]-s_data[tz][tx-5]);
					sumx+=coe_d[6]*(s_data[tz][tx+5]-s_data[tz][tx-6]);

			float    	sumz=coe_d[1]* (s_data[tz+1][tx]-s_data[tz][tx]);
					sumz+=coe_d[2]*(s_data[tz+2][tx]-s_data[tz-1][tx]);
					sumz+=coe_d[3]*(s_data[tz+3][tx]-s_data[tz-2][tx]);
					sumz+=coe_d[4]*(s_data[tz+4][tx]-s_data[tz-3][tx]);
					sumz+=coe_d[5]*(s_data[tz+5][tx]-s_data[tz-4][tx]);
					sumz+=coe_d[6]*(s_data[tz+6][tx]-s_data[tz-5][tx]);


			/*float    	sumx=coe_d[1]*(s_data[tz][tx+1]- s_data[tz][tx]);
					sumx+=coe_d[2]*(s_data[tz][tx+2]-s_data[tz][tx-1]);
					sumx+=coe_d[3]*(s_data[tz][tx+3]-s_data[tz][tx-2]);
					sumx+=coe_d[4]*(s_data[tz][tx+4]-s_data[tz][tx-3]);
					sumx+=coe_d[5]*(s_data[tz][tx+5]-s_data[tz][tx-4]);
					sumx+=coe_d[6]*(s_data[tz][tx+6]-s_data[tz][tx-5]);

			float    	sumz=coe_d[1]*(s_data[tz][tx]-  s_data[tz-1][tx]);
					sumz+=coe_d[2]*(s_data[tz+1][tx]-s_data[tz-2][tx]);
					sumz+=coe_d[3]*(s_data[tz+2][tx]-s_data[tz-3][tx]);
					sumz+=coe_d[4]*(s_data[tz+3][tx]-s_data[tz-4][tx]);
					sumz+=coe_d[5]*(s_data[tz+4][tx]-s_data[tz-5][tx]);
					sumz+=coe_d[6]*(s_data[tz+5][tx]-s_data[tz-6][tx]);*/


			//vxp2_d[in_idx]=vxp1_d[in_idx]+(1.0/density_d[in_idx])*sumx*coe_x;
			vxp2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*vxp1_d[in_idx]-(1.0/density_d[in_idx])*sumx*coe_x);

			vzp2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*vzp1_d[in_idx]-(1.0/density_d[in_idx])*sumz*coe_z);
			//vzp2_d[in_idx]=vzp1_d[in_idx]+(1.0/density_d[in_idx])*sumz*coe_z;
		}
}

__global__ void rfwd_vxp_vzp_new(float *vxp_t_d,float *vzp_t_d,float *vxs_t_d,float *vzs_t_d,float *vx_t_d,float *vz_t_d,float *vxp2_d,float *vxp1_d,float *vzp2_d,float *vzp1_d,float *tp1_d,float coe_x,float coe_z,float dx,float dz,float dt,float *attenuation,float *coe_d,int dimx,int dimz,float *density_d)
{
		__shared__ float s_data[BDIMY+2*radius][BDIMX+2*radius];

		int ix = blockIdx.x*blockDim.x+threadIdx.x;
		int iz = blockIdx.y*blockDim.y+threadIdx.y;
		int in_idx;

		float dt_real=dt/1000;

		float s_attenuation;

		int tx = threadIdx.x+radius;
		int tz = threadIdx.y+radius;

		s_data[tz][tx]=0.0;
		s_data[threadIdx.y][threadIdx.x]=0.0;
		s_data[BDIMY+2*radius-1-threadIdx.y][BDIMX+2*radius-1-threadIdx.x]=0.0;
		s_data[threadIdx.y][BDIMX+2*radius-1-threadIdx.x]=0.0;
		s_data[BDIMY+2*radius-1-threadIdx.y][threadIdx.x]=0.0;

		if(ix<dimx&&iz<dimz)
		{
			ix=ix+radius,iz=iz+radius;
			dimx=dimx+2*radius;dimz=dimz+2*radius;	
			in_idx=ix*dimz+iz;
			__syncthreads();

			s_data[tz][tx]=tp1_d[in_idx];
				
			if(threadIdx.y<radius)
			{
				s_data[threadIdx.y][tx]=tp1_d[in_idx-radius];//g_input[in_idx-radius*dimx];//up
				s_data[threadIdx.y+BDIMY+radius][tx]=tp1_d[in_idx+BDIMY];//g_input[in_idx+BDIMY*dimx];//down
			}
			if(threadIdx.x<radius)
			{
				s_data[tz][threadIdx.x]=tp1_d[in_idx-radius*dimz];//g_input[in_idx-radius];//left
				s_data[tz][threadIdx.x+BDIMX+radius]=tp1_d[in_idx+BDIMX*dimz];//g_input[in_idx+BDIMX];//right
			}
			
			s_attenuation=attenuation[in_idx];
			__syncthreads();


			float    	sumx= coe_d[1]*(s_data[tz][tx]-	 s_data[tz][tx-1]);
					sumx+=coe_d[2]*(s_data[tz][tx+1]-s_data[tz][tx-2]);
					sumx+=coe_d[3]*(s_data[tz][tx+2]-s_data[tz][tx-3]);
					sumx+=coe_d[4]*(s_data[tz][tx+3]-s_data[tz][tx-4]);
					sumx+=coe_d[5]*(s_data[tz][tx+4]-s_data[tz][tx-5]);
					sumx+=coe_d[6]*(s_data[tz][tx+5]-s_data[tz][tx-6]);

			float    	sumz=coe_d[1]* (s_data[tz+1][tx]-s_data[tz][tx]);
					sumz+=coe_d[2]*(s_data[tz+2][tx]-s_data[tz-1][tx]);
					sumz+=coe_d[3]*(s_data[tz+3][tx]-s_data[tz-2][tx]);
					sumz+=coe_d[4]*(s_data[tz+4][tx]-s_data[tz-3][tx]);
					sumz+=coe_d[5]*(s_data[tz+5][tx]-s_data[tz-4][tx]);
					sumz+=coe_d[6]*(s_data[tz+6][tx]-s_data[tz-5][tx]);


			/*float    	sumx=coe_d[1]*(s_data[tz][tx+1]- s_data[tz][tx]);
					sumx+=coe_d[2]*(s_data[tz][tx+2]-s_data[tz][tx-1]);
					sumx+=coe_d[3]*(s_data[tz][tx+3]-s_data[tz][tx-2]);
					sumx+=coe_d[4]*(s_data[tz][tx+4]-s_data[tz][tx-3]);
					sumx+=coe_d[5]*(s_data[tz][tx+5]-s_data[tz][tx-4]);
					sumx+=coe_d[6]*(s_data[tz][tx+6]-s_data[tz][tx-5]);

			float    	sumz=coe_d[1]*(s_data[tz][tx]-  s_data[tz-1][tx]);
					sumz+=coe_d[2]*(s_data[tz+1][tx]-s_data[tz-2][tx]);
					sumz+=coe_d[3]*(s_data[tz+2][tx]-s_data[tz-3][tx]);
					sumz+=coe_d[4]*(s_data[tz+3][tx]-s_data[tz-4][tx]);
					sumz+=coe_d[5]*(s_data[tz+4][tx]-s_data[tz-5][tx]);
					sumz+=coe_d[6]*(s_data[tz+5][tx]-s_data[tz-6][tx]);*/


			//vxp2_d[in_idx]=vxp1_d[in_idx]+(1.0/density_d[in_idx])*sumx*coe_x;
			vxp2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*vxp1_d[in_idx]-(1.0/density_d[in_idx])*sumx*coe_x);

			vzp2_d[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*vzp1_d[in_idx]-(1.0/density_d[in_idx])*sumz*coe_z);
			//vzp2_d[in_idx]=vzp1_d[in_idx]+(1.0/density_d[in_idx])*sumz*coe_z;

			vxp_t_d[in_idx]=1.0/density_d[in_idx]*sumx*coe_x;

			vzp_t_d[in_idx]=1.0/density_d[in_idx]*sumz*coe_z;

			vxs_t_d[in_idx]=vx_t_d[in_idx]-vxp_t_d[in_idx];

			vzs_t_d[in_idx]=vz_t_d[in_idx]-vzp_t_d[in_idx];
		}
}

__global__ void rfwd_txxzzxzpp(float *tp2,float *tp1,float *txx2,float *txx1,float *tzz2,float *tzz1,float *txz2,float *txz1,float *vx2,float *vz2,float *velocity_d,float *velocity1_d,float *attenuation_d,float dt,float *coe_d,float coe_x,float coe_z,int dimx,int dimz,int nx,int nz,int boundary_left,int boundary_up,float *density_d)
//txx2_d,txx1_d,tzz2_d,tzz1_d,txz2_d,txz1_d,vx2_d,vz2_d,velocity_d,velocity1_d,attenuation_d,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius
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

				s_velocity=velocity_d[in_idx]*velocity_d[in_idx];
				s_velocity1=velocity1_d[in_idx]*velocity1_d[in_idx];
				
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

				float    sumz=coe_d[1]*(s_data2[tz][tx]-s_data2[tz-1][tx]);
					sumz+=coe_d[2]*(s_data2[tz+1][tx]-s_data2[tz-2][tx]);
					sumz+=coe_d[3]*(s_data2[tz+2][tx]-s_data2[tz-3][tx]);
					sumz+=coe_d[4]*(s_data2[tz+3][tx]-s_data2[tz-4][tx]);
					sumz+=coe_d[5]*(s_data2[tz+4][tx]-s_data2[tz-5][tx]);
					sumz+=coe_d[6]*(s_data2[tz+5][tx]-s_data2[tz-6][tx]);


				txx2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*txx1[in_idx]-
						s_velocity*density_d[in_idx]*sumx*coe_x-(s_velocity-2*s_velocity1)*density_d[in_idx]*sumz*coe_z);//s_velocity  and  s_velocity1

				tzz2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*tzz1[in_idx]-
						(s_velocity-2*s_velocity1)*density_d[in_idx]*sumx*coe_x-s_velocity*density_d[in_idx]*sumz*coe_z);//sumx  and  sumz 
					
				tp2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*tp1[in_idx]-s_velocity*density_d[in_idx]*sumx*coe_x-s_velocity*density_d[in_idx]*sumz*coe_z);				
	
				float    sumx1=coe_d[1]*(s_data2[tz][tx]-s_data2[tz][tx-1]);
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
						((1.0-s_attenuation*dt_real/2.0)*txz1[in_idx]-s_velocity1*density_d[in_idx]*(sumx1*coe_x+sumz1*coe_z));
		}
}

__global__ void rfwd_txxzzxzpp_new(float *vx_x_d,float *vx_z_d,float *vz_x_d,float *vz_z_d,float *tp2,float *tp1,float *txx2,float *txx1,float *tzz2,float *tzz1,float *txz2,float *txz1,float *vx2,float *vz2,float *velocity_d,float *velocity1_d,float *attenuation_d,float dt,float dx,float dz,float *coe_d,float coe_x,float coe_z,int dimx,int dimz,int nx,int nz,int boundary_left,int boundary_up,float *density_d)
//rfwd_txxzzxzpp_new<<<dimGrid,dimBlock>>>(vx_x_d,vx_z_d,vz_x_d,vz_z_d,tp1_d,tp2_d,txx1_d,txx2_d,tzz1_d,tzz2_d,txz1_d,txz2_d,vx2_d,vz2_d,s_velocity_d,s_velocity1_d,attenuation_d,dt,dx,dz,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,nx,nz,boundary_left,boundary_up,s_density_d);
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

				s_velocity=velocity_d[in_idx]*velocity_d[in_idx];
				s_velocity1=velocity1_d[in_idx]*velocity1_d[in_idx];
				
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

				float    sumz=coe_d[1]*(s_data2[tz][tx]-s_data2[tz-1][tx]);
					sumz+=coe_d[2]*(s_data2[tz+1][tx]-s_data2[tz-2][tx]);
					sumz+=coe_d[3]*(s_data2[tz+2][tx]-s_data2[tz-3][tx]);
					sumz+=coe_d[4]*(s_data2[tz+3][tx]-s_data2[tz-4][tx]);
					sumz+=coe_d[5]*(s_data2[tz+4][tx]-s_data2[tz-5][tx]);
					sumz+=coe_d[6]*(s_data2[tz+5][tx]-s_data2[tz-6][tx]);

				vx_x_d[in_idx]=sumx*1.0/dx;

				vz_z_d[in_idx]=sumz*1.0/dz;


				txx2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*txx1[in_idx]-
						s_velocity*density_d[in_idx]*sumx*coe_x-(s_velocity-2*s_velocity1)*density_d[in_idx]*sumz*coe_z);//s_velocity  and  s_velocity1

				tzz2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*tzz1[in_idx]-
						(s_velocity-2*s_velocity1)*density_d[in_idx]*sumx*coe_x-s_velocity*density_d[in_idx]*sumz*coe_z);//sumx  and  sumz 
					
				tp2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*((1.0-s_attenuation*dt_real/2.0)*tp1[in_idx]-s_velocity*density_d[in_idx]*sumx*coe_x-s_velocity*density_d[in_idx]*sumz*coe_z);				
	
				float    sumx1=coe_d[1]*(s_data2[tz][tx]-s_data2[tz][tx-1]);
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

				txz2[in_idx]=1.0/(1.0+s_attenuation*dt_real/2.0)*
						((1.0-s_attenuation*dt_real/2.0)*txz1[in_idx]-s_velocity1*density_d[in_idx]*(sumx1*coe_x+sumz1*coe_z));
		}
}

__global__ void save_wfud(float *wf2_d,float *wfu_d,float *wfd_d,float *f2_d,float *fu_d,float *fd_d,int it,int lt,int nz,int nx_append,int nz_append,int boundary_up,int mark)
{
		int ix=blockIdx.x;

		wfu_d[mark*lt*nx_append+ix*lt+it]=wf2_d[ix*nz_append+boundary_up-1-mark];
		 fu_d[mark*lt*nx_append+ix*lt+it]= f2_d[ix*nz_append+boundary_up-1-mark];
		//wfu_d[mark*lt*nx_append+ix*lt+it]=0.0;
		// fu_d[mark*lt*nx_append+ix*lt+it]=0.0;
		
		wfd_d[mark*lt*nx_append+ix*lt+it]=wf2_d[ix*nz_append+boundary_up+nz+mark];
		 fd_d[mark*lt*nx_append+ix*lt+it]= f2_d[ix*nz_append+boundary_up+nz+mark];
}
//
__global__ void save_wflr(float *wf2_d,float *wfl_d,float *wfr_d,float *f2_d,float *fl_d,float *fr_d,int it,int lt,int nx_append,int nz_append,int boundary_left,int boundary_right,int mark)
{
		int iz=blockIdx.x;

		wfl_d[mark*lt*nz_append+iz*lt+it]=wf2_d[(boundary_left-mark-1)*nz_append+iz];
		 fl_d[mark*lt*nz_append+iz*lt+it]= f2_d[(boundary_left-mark-1)*nz_append+iz];

		wfr_d[mark*lt*nz_append+iz*lt+it]=wf2_d[(nx_append-boundary_right+mark)*nz_append+iz];
		 fr_d[mark*lt*nz_append+iz*lt+it]= f2_d[(nx_append-boundary_right+mark)*nz_append+iz];
}	

//
__global__ void set_wfud(float *wf2_d,float *wfu_d,float *wfd_d,float *f2_d,float *fu_d,float *fd_d,int it,int lt,int nz,int nx_append,int nz_append,int boundary_up,int mark)
{
		int ix=blockIdx.x;

		wf2_d[ix*nz_append+boundary_up-1-mark]=wfu_d[mark*lt*nx_append+ix*lt+it];
		 f2_d[ix*nz_append+boundary_up-1-mark]= fu_d[mark*lt*nx_append+ix*lt+it];

		wf2_d[ix*nz_append+boundary_up+nz+mark]=wfd_d[mark*lt*nx_append+ix*lt+it];
		 f2_d[ix*nz_append+boundary_up+nz+mark]= fd_d[mark*lt*nx_append+ix*lt+it];
}
//
__global__ void set_wflr(float *wf2_d,float *wfl_d,float *wfr_d,float *f2_d,float *fl_d,float *fr_d,int it,int lt,int nx_append,int nz_append,int boundary_left,int boundary_right,int mark)
{
		int iz=blockIdx.x;

		wf2_d[(boundary_left-mark-1)*nz_append+iz]=wfl_d[mark*nz_append*lt+iz*lt+it];
		 f2_d[(boundary_left-mark-1)*nz_append+iz]= fl_d[mark*nz_append*lt+iz*lt+it];

		wf2_d[(nx_append-boundary_right+mark)*nz_append+iz]=wfr_d[mark*nz_append*lt+iz*lt+it];
		 f2_d[(nx_append-boundary_right+mark)*nz_append+iz]= fr_d[mark*nz_append*lt+iz*lt+it];
}

__global__ void imaging_down_vector(float *vxp1_d,float *vzp1_d,float *image_down_d,int nx,int nz,int nz_append,int boundary_up,int boundary_left)
{
		int ix=blockIdx.x*blockDim.x+threadIdx.x;
		int iz=blockIdx.y*blockDim.y+threadIdx.y;

		int in_idx;

		if(ix<nx&&iz<nz)
		{
			in_idx=(boundary_left+ix)*nz_append+iz+boundary_up;
			
			image_down_d[ix*nz+iz]=image_down_d[ix*nz+iz]+vxp1_d[in_idx]*vxp1_d[in_idx]+vzp1_d[in_idx]*vzp1_d[in_idx];
		}
}

__global__ void imaging_down_correlation(float *p_d,float *image_down_d,int nx,int nz,int nz_append,int boundary_up,int boundary_left)
{
		int ix=blockIdx.x*blockDim.x+threadIdx.x;
		int iz=blockIdx.y*blockDim.y+threadIdx.y;

		int in_idx;

		if(ix<nx&&iz<nz)
		{
			in_idx=(boundary_left+ix)*nz_append+iz+boundary_up;
			
			image_down_d[ix*nz+iz]=image_down_d[ix*nz+iz]+p_d[in_idx]*p_d[in_idx];
		}
}

__global__ void imaging_vector_correlation(float *px1_d,float *pz1_d,float *rpx1_d,float *rpz1_d,float *imageup_d,int nx,int nz,int nz_append,int boundary_up,int boundary_left)
//imaging_vector<<<dimGrid,dimBlock>>>(vxp1_d,vzp1_d,rvxp2_d,rvzp2_d,rimageup9_d,rimagedown9_d,nx,nz,nz_append,boundary_up,boundary_left);
{
		int ix=blockIdx.x*blockDim.x+threadIdx.x;
		int iz=blockIdx.y*blockDim.y+threadIdx.y;

		int in_idx;

		if(ix<nx&&iz<nz)
		{
			in_idx=(boundary_left+ix)*nz_append+iz+boundary_up;
			imageup_d[ix*nz+iz]=imageup_d[ix*nz+iz]+px1_d[in_idx]*rpx1_d[in_idx]+pz1_d[in_idx]*rpz1_d[in_idx];
		}
}

__global__ void imaging_vector_correlation_new(float *px1_d,float *pz1_d,float *rpx1_d,float *rpz1_d,float *imageup_d,int nx,int nz,int nz_append,int boundary_up,int boundary_left)
//imaging_vector_correlation_new<<<dimGrid,dimBlock>>>(vxp1_d,vzp1_d,rvxp1_d,rvzp1_d,vresultpp_d,nx_size,nz,nz_append,boundary_up,boundary_left);
{
		int ix=blockIdx.x*blockDim.x+threadIdx.x;
		int iz=blockIdx.y*blockDim.y+threadIdx.y;

		int in_idx;

		float change_pz,change_rpz;

		if(ix<nx&&iz<nz)
		{
			in_idx=(boundary_left+ix)*nz_append+iz+boundary_up;

			change_pz=1.0*(pz1_d[in_idx+1]+pz1_d[in_idx-1]+pz1_d[in_idx+nz_append]+pz1_d[in_idx-nz_append])/4.0;

			change_rpz=1.0*(rpz1_d[in_idx+1]+rpz1_d[in_idx-1]+rpz1_d[in_idx+nz_append]+rpz1_d[in_idx-nz_append])/4.0;
			
			imageup_d[ix*nz+iz]=imageup_d[ix*nz+iz]+px1_d[in_idx]*rpx1_d[in_idx]+change_pz*change_rpz;
		}
}

__global__ void imaging_correlation(float *p_d,float *rp_d,float *image_d,int nx,int nz,int nz_append,int boundary_up,int boundary_left)
{
		int ix=blockIdx.x*blockDim.x+threadIdx.x;
		int iz=blockIdx.y*blockDim.y+threadIdx.y;

		int in_idx;

		if(ix<nx&&iz<nz)
		{
			in_idx=(boundary_left+ix)*nz_append+iz+boundary_up;
			
			image_d[ix*nz+iz]=image_d[ix*nz+iz]+p_d[in_idx]*rp_d[in_idx];
		}
}

__global__ void imaging_correlation_source_x_cord(float *p_d,float *rp_d,float *image_d,int nx,int nz,int nz_append,int boundary_up,int boundary_left,int source_x_cord)
{
		int ix=blockIdx.x*blockDim.x+threadIdx.x;
		int iz=blockIdx.y*blockDim.y+threadIdx.y;

		int in_idx;

		if(ix<nx&&iz<nz)
		{
			in_idx=(boundary_left+ix)*nz_append+iz+boundary_up;
			
			if(ix<source_x_cord)		image_d[ix*nz+iz]=image_d[ix*nz+iz]+p_d[in_idx]*rp_d[in_idx];

			else				image_d[ix*nz+iz]=image_d[ix*nz+iz]-p_d[in_idx]*rp_d[in_idx];
			
		}
}

__global__ void imaging_correlation_sign(float *p_d,float *rs_d,float *image_d,float *sign_d,int nx,int nz,int nz_append,int boundary_up,int boundary_left)
{
		int ix=blockIdx.x*blockDim.x+threadIdx.x;
		int iz=blockIdx.y*blockDim.y+threadIdx.y;

		int in_idx;

		if(ix<nx&&iz<nz)
		{
			in_idx=(boundary_left+ix)*nz_append+iz+boundary_up;
			
			if(sign_d[in_idx]>=0)	image_d[ix*nz+iz]=image_d[ix*nz+iz]+p_d[in_idx]*rs_d[in_idx];
			if(sign_d[in_idx]<0)	image_d[ix*nz+iz]=image_d[ix*nz+iz]-1*p_d[in_idx]*rs_d[in_idx];
		}
}

__global__ void imaging_correlation_sign_ps(float *p_d,float *rs_d,float *image_d,float *sign_d,int source_x_cord,int nx,int nz,int nz_append,int boundary_up,int boundary_left)
{
		int ix=blockIdx.x*blockDim.x+threadIdx.x;
		int iz=blockIdx.y*blockDim.y+threadIdx.y;

		int in_idx;

		if(ix<nx&&iz<nz&&iz>0.8*(ix-source_x_cord)&&iz>0.8*(source_x_cord-ix))
		{
			in_idx=(boundary_left+ix)*nz_append+iz+boundary_up;
			
			if(sign_d[in_idx]>=0)	image_d[ix*nz+iz]=image_d[ix*nz+iz]+p_d[in_idx]*rs_d[in_idx];
			if(sign_d[in_idx]<0)	image_d[ix*nz+iz]=image_d[ix*nz+iz]-1*p_d[in_idx]*rs_d[in_idx];
		}
}

__global__ void imaging_vector_correlation_ps(float *px1_d,float *pz1_d,float *rpx1_d,float *rpz1_d,float *imageup_d,int source_x_cord,int nx,int nz,int nz_append,int boundary_up,int boundary_left)
//imaging_vector<<<dimGrid,dimBlock>>>(vxp1_d,vzp1_d,rvxp2_d,rvzp2_d,rimageup9_d,rimagedown9_d,nx,nz,nz_append,boundary_up,boundary_left);
{
		int ix=blockIdx.x*blockDim.x+threadIdx.x;
		int iz=blockIdx.y*blockDim.y+threadIdx.y;

		int in_idx;

		if(ix<nx&&iz<nz&&iz>0.8*(ix-source_x_cord)&&iz>0.8*(source_x_cord-ix))
		{
			in_idx=(boundary_left+ix)*nz_append+iz+boundary_up;
			imageup_d[ix*nz+iz]=imageup_d[ix*nz+iz]+px1_d[in_idx]*rpx1_d[in_idx]+pz1_d[in_idx]*rpz1_d[in_idx];
		}
}

__global__ void imaging_correlation_ps(float *p_d,float *rp_d,float *image_d,int source_x_cord,int nx,int nz,int nz_append,int boundary_up,int boundary_left)
{
		int ix=blockIdx.x*blockDim.x+threadIdx.x;
		int iz=blockIdx.y*blockDim.y+threadIdx.y;

		int in_idx;

		if(ix<nx&&iz<nz&&iz>0.8*(ix-source_x_cord)&&iz>0.8*(source_x_cord-ix))
		{
			in_idx=(boundary_left+ix)*nz_append+iz+boundary_up;
			
			image_d[ix*nz+iz]=image_d[ix*nz+iz]+p_d[in_idx]*rp_d[in_idx];
		}
}

__global__ void imagingadd(float *imageup_d,float *imagedown_d,int nx,int nz,float *max_d,float average)
{
		int ix=blockIdx.x*blockDim.x+threadIdx.x;
		int iz=blockIdx.y*blockDim.y+threadIdx.y;
		
		int in_idx;

		if(ix<nx&&iz<nz)
		{
			in_idx=ix*nz+iz;
			
			//imageup_d[in_idx]=imageup_d[in_idx]/(imagedown_d[in_idx]);
			
			imageup_d[in_idx]=imageup_d[in_idx]/(imagedown_d[in_idx]+average);

			//imageup_d[in_idx]=imageup_d[in_idx]/(0.5*max_d[iz]);

			//imageup_d[ix*nz+iz]=imageup_d[ix*nz+iz]/(imagedown_d[ix*nz+iz]+damping*max_d[iz]);
			//imageup_d[ix*nz+iz]=imageup_d[ix*nz+iz]/(imagedown_d[ix*nz+iz]+0.001);
						
			//imageup_d[ix*nz+iz]=imageup_d[ix*nz+iz]/(imagedown_d[ix*nz+iz]+0.5*max_d[iz]);
					
			//imageup_d[ix*nz+iz]=imageup_d[ix*nz+iz]/(imagedown_d[ix*nz+iz]+damping*global_max);
										
			
		}
}

__global__ void imaging_dot_product(float *p_x_d,float *p_z_d,float *rs_x_d,float *rs_z_d,float *imageup_d,float *imagedown_d,int nx,int nz,int nx_append,int nz_append,int boundary_up,int boundary_left)
{
		int ix=blockIdx.x*blockDim.x+threadIdx.x;
		int iz=blockIdx.y*blockDim.y+threadIdx.y;

		int in_idx;

		if(ix<nx&&iz<nz)
		{
			in_idx=(boundary_left+ix)*nz_append+iz+boundary_up;
			imageup_d[ix*nz+iz]+=p_x_d[in_idx]*rs_x_d[in_idx]+p_z_d[in_idx]*rs_z_d[in_idx];

			imagedown_d[ix*nz+iz]+=p_x_d[in_idx]*p_x_d[in_idx]+p_z_d[in_idx]*p_z_d[in_idx];
		}
}

__global__ void imaging_old(float *wf_d,float *wfr_d,float *imageup_d,float *imagedown_d,int nx,int nz,int nz_append,int boundary_up,int boundary_left)
{
		int ix=blockIdx.x*blockDim.x+threadIdx.x;
		int iz=blockIdx.y*blockDim.y+threadIdx.y;

		if(ix<nx&&iz<nz)
		{
			imageup_d[ix*nz+iz]+=wf_d[(boundary_left+ix)*nz_append+iz+boundary_up]*wfr_d[(boundary_left+ix)*nz_append+iz+boundary_up];

			imagedown_d[ix*nz+iz]+=wf_d[(boundary_left+ix)*nz_append+iz+boundary_up]*wf_d[(boundary_left+ix)*nz_append+iz+boundary_up];
			//imagedown_d[ix*nz+iz]+=wfr_d[(boundary_left+ix)*nz_append+iz+boundary_up]*wfr_d[(boundary_left+ix)*nz_append+iz+boundary_up];
		}
}

