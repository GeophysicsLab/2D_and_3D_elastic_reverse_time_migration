
void output_3d_wavefiled_tao(GPUdevice *mgdevice,int it)
{
		for(int i=0;i<GPU_N;i++)
		{
			checkCudaErrors(cudaSetDevice(gpuid[i]));
			checkCudaErrors(cudaMemcpy(mgdevice[i].wf_h,mgdevice[i].tp1_d,nnx*nny*nnz_device_append*sizeof(float),cudaMemcpyDefault));

			sprintf(filename,"./wavefield/tp-%d-%d",i,it);
			output_file_xyz(filename,mgdevice[i].wf_h,nnx,nny,nnz_device_append);
		}

			checkCudaErrors(cudaDeviceSynchronize());
		
			
						
		for(int i=0;i<GPU_N;i++)
		{
			checkCudaErrors(cudaSetDevice(gpuid[i]));

			seperate_or_togather_vel_att2(mgdevice[i].wf_h,wf_3d,nnx,nny,nnz,nnz_device,i,radius,1);
		}

			checkCudaErrors(cudaDeviceSynchronize());

		sprintf(filename,"./wavefield/tp-%d",it);
		output_file_xyz(filename,wf_3d,nnx,nny,nnz);
		sprintf(filename,"./wavefield/cut-tp-%d",it);
		output_file_xyz_boundary(filename,wf_3d,nx,ny,nz,bl,bf,bu,nnx,nny,nnz);
}

void output_3d_wavefiled_vx(GPUdevice *mgdevice,int it)
{
		for(int i=0;i<GPU_N;i++)
		{
			checkCudaErrors(cudaSetDevice(gpuid[i]));
			checkCudaErrors(cudaMemcpy(mgdevice[i].wf_h,mgdevice[i].vx1_d,nnx*nny*nnz_device_append*sizeof(float),cudaMemcpyDefault));

			sprintf(filename,"./wavefield/vx-%d-%d",i,it);
			output_file_xyz(filename,mgdevice[i].wf_h,nnx,nny,nnz_device_append);
		}

			checkCudaErrors(cudaDeviceSynchronize());
		
			
						
		for(int i=0;i<GPU_N;i++)
		{
			checkCudaErrors(cudaSetDevice(gpuid[i]));

			seperate_or_togather_vel_att2(mgdevice[i].wf_h,wf_3d,nnx,nny,nnz,nnz_device,i,radius,1);
		}

			checkCudaErrors(cudaDeviceSynchronize());

		sprintf(filename,"./wavefield/vx-%d",it);
		output_file_xyz(filename,wf_3d,nnx,nny,nnz);
		sprintf(filename,"./wavefield/cut-vx-%d",it);
		output_file_xyz_boundary(filename,wf_3d,nx,ny,nz,bl,bf,bu,nnx,nny,nnz);
}

void output_3d_wavefiled_vz(GPUdevice *mgdevice,int it)
{
		for(int i=0;i<GPU_N;i++)
		{
			checkCudaErrors(cudaSetDevice(gpuid[i]));
			checkCudaErrors(cudaMemcpy(mgdevice[i].wf_h,mgdevice[i].vz1_d,nnx*nny*nnz_device_append*sizeof(float),cudaMemcpyDefault));

			sprintf(filename,"./wavefield/vz-%d-%d",i,it);
			output_file_xyz(filename,mgdevice[i].wf_h,nnx,nny,nnz_device_append);
		}

			checkCudaErrors(cudaDeviceSynchronize());
		
			
						
		for(int i=0;i<GPU_N;i++)
		{
			checkCudaErrors(cudaSetDevice(gpuid[i]));

			seperate_or_togather_vel_att2(mgdevice[i].wf_h,wf_3d,nnx,nny,nnz,nnz_device,i,radius,1);
		}

			checkCudaErrors(cudaDeviceSynchronize());

		sprintf(filename,"./wavefield/vz-%d",it);
		output_file_xyz(filename,wf_3d,nnx,nny,nnz);
		sprintf(filename,"./wavefield/cut-vz-%d",it);
		output_file_xyz_boundary(filename,wf_3d,nx,ny,nz,bl,bf,bu,nnx,nny,nnz);
}


void output_3d_wavefiled_vzp(GPUdevice *mgdevice,int it)
{
		for(int i=0;i<GPU_N;i++)
		{
			checkCudaErrors(cudaSetDevice(gpuid[i]));
			checkCudaErrors(cudaMemcpy(mgdevice[i].wf_h,mgdevice[i].vzp1_d,nnx*nny*nnz_device_append*sizeof(float),cudaMemcpyDefault));

			sprintf(filename,"./wavefield/vzp-%d-%d",i,it);
			output_file_xyz(filename,mgdevice[i].wf_h,nnx,nny,nnz_device_append);
		}

			checkCudaErrors(cudaDeviceSynchronize());
		
			
						
		for(int i=0;i<GPU_N;i++)
		{
			checkCudaErrors(cudaSetDevice(gpuid[i]));

			seperate_or_togather_vel_att2(mgdevice[i].wf_h,wf_3d,nnx,nny,nnz,nnz_device,i,radius,1);
		}

			checkCudaErrors(cudaDeviceSynchronize());

		sprintf(filename,"./wavefield/vzp-%d",it);
		output_file_xyz(filename,wf_3d,nnx,nny,nnz);
		sprintf(filename,"./wavefield/cut-vzp-%d",it);
		output_file_xyz_boundary(filename,wf_3d,nx,ny,nz,bl,bf,bu,nnx,nny,nnz);
}

void output_3d_wavefiled_vzs(GPUdevice *mgdevice,int it)
{
		for(int i=0;i<GPU_N;i++)
		{
			checkCudaErrors(cudaSetDevice(gpuid[i]));
			checkCudaErrors(cudaMemcpy(mgdevice[i].wf_h,mgdevice[i].vzs1_d,nnx*nny*nnz_device_append*sizeof(float),cudaMemcpyDefault));

			sprintf(filename,"./wavefield/vzs-%d-%d",i,it);
			output_file_xyz(filename,mgdevice[i].wf_h,nnx,nny,nnz_device_append);
		}

			checkCudaErrors(cudaDeviceSynchronize());
		
			
						
		for(int i=0;i<GPU_N;i++)
		{
			checkCudaErrors(cudaSetDevice(gpuid[i]));

			seperate_or_togather_vel_att2(mgdevice[i].wf_h,wf_3d,nnx,nny,nnz,nnz_device,i,radius,1);
		}

			checkCudaErrors(cudaDeviceSynchronize());

		sprintf(filename,"./wavefield/vzs-%d",it);
		output_file_xyz(filename,wf_3d,nnx,nny,nnz);
		sprintf(filename,"./wavefield/cut-vzs-%d",it);
		output_file_xyz_boundary(filename,wf_3d,nx,ny,nz,bl,bf,bu,nnx,nny,nnz);
}



void output_3d_wavefiled_excitation_amp_time(GPUdevice *mgdevice,int isx,int isy,int isz)
{
		for(int i=0;i<GPU_N;i++)
		{
			checkCudaErrors(cudaSetDevice(gpuid[i]));

			checkCudaErrors(cudaMemcpy(mgdevice[i].wf_h,mgdevice[i].ex_time_d,nnx*nny*nnz_device_append*sizeof(float),cudaMemcpyDefault));
			checkCudaErrors(cudaMemcpy(mgdevice[i].wf0_h,mgdevice[i].ex_amp_d,nnx*nny*nnz_device_append*sizeof(float),cudaMemcpyDefault));
			checkCudaErrors(cudaMemcpy(mgdevice[i].wf1_h,mgdevice[i].ex_tp_d,nnx*nny*nnz_device_append*sizeof(float),cudaMemcpyDefault));
			checkCudaErrors(cudaMemcpy(mgdevice[i].wf2_h,mgdevice[i].ex_vxp_d,nnx*nny*nnz_device_append*sizeof(float),cudaMemcpyDefault));
			checkCudaErrors(cudaMemcpy(mgdevice[i].wf3_h,mgdevice[i].ex_vyp_d,nnx*nny*nnz_device_append*sizeof(float),cudaMemcpyDefault));
			checkCudaErrors(cudaMemcpy(mgdevice[i].wf4_h,mgdevice[i].ex_vzp_d,nnx*nny*nnz_device_append*sizeof(float),cudaMemcpyDefault));

			//sprintf(filename,"./wavefield/ex-time-%d-%d",i,it);
			//output_file_xyz(filename,mgdevice[i].wf_h,nnx,nny,nnz_device_append);
		}
			checkCudaErrors(cudaDeviceSynchronize());
		
			
						
		for(int i=0;i<GPU_N;i++)
		{
			checkCudaErrors(cudaSetDevice(gpuid[i]));

			seperate_or_togather_vel_att2(mgdevice[i].wf_h,wf_3d,nnx,nny,nnz,nnz_device,i,radius,1);
		}

			checkCudaErrors(cudaDeviceSynchronize());

		sprintf(filename,"./someoutput/ex-time-%d-%d",sy_real,sx_real);
		output_file_xyz(filename,wf_3d,nnx,nny,nnz);
		sprintf(filename,"./someoutput/cut-ex-time-%d-%d",sy_real,sx_real);
		output_file_xyz_boundary(filename,wf_3d,nx,ny,nz,bl,bf,bu,nnx,nny,nnz);

//////////////////////////////////amp=vx*vx+vy*vy+vz*vz
		for(int i=0;i<GPU_N;i++)
		{
			checkCudaErrors(cudaSetDevice(gpuid[i]));

			seperate_or_togather_vel_att2(mgdevice[i].wf0_h,wf_3d,nnx,nny,nnz,nnz_device,i,radius,1);
		}

			checkCudaErrors(cudaDeviceSynchronize());
		sprintf(filename,"./someoutput/cut-ex-amp-%d-%d",sy_real,sx_real);
		output_file_xyz_boundary(filename,wf_3d,nx,ny,nz,bl,bf,bu,nnx,nny,nnz);

//////////////////////////////////tp
		for(int i=0;i<GPU_N;i++)
		{
			checkCudaErrors(cudaSetDevice(gpuid[i]));

			seperate_or_togather_vel_att2(mgdevice[i].wf1_h,wf_3d,nnx,nny,nnz,nnz_device,i,radius,1);
		}

			checkCudaErrors(cudaDeviceSynchronize());
		sprintf(filename,"./someoutput/cut-ex-tp-%d-%d",sy_real,sx_real);
		output_file_xyz_boundary(filename,wf_3d,nx,ny,nz,bl,bf,bu,nnx,nny,nnz);

//////////////////////////////////vx
		for(int i=0;i<GPU_N;i++)
		{
			checkCudaErrors(cudaSetDevice(gpuid[i]));

			seperate_or_togather_vel_att2(mgdevice[i].wf2_h,wf_3d,nnx,nny,nnz,nnz_device,i,radius,1);
		}

			checkCudaErrors(cudaDeviceSynchronize());
		sprintf(filename,"./someoutput/cut-ex-vxp-%d-%d",sy_real,sx_real);
		output_file_xyz_boundary(filename,wf_3d,nx,ny,nz,bl,bf,bu,nnx,nny,nnz);


//////////////////////////////////vy
		for(int i=0;i<GPU_N;i++)
		{
			checkCudaErrors(cudaSetDevice(gpuid[i]));

			seperate_or_togather_vel_att2(mgdevice[i].wf3_h,wf_3d,nnx,nny,nnz,nnz_device,i,radius,1);
		}

			checkCudaErrors(cudaDeviceSynchronize());
		sprintf(filename,"./someoutput/cut-ex-vyp-%d-%d",sy_real,sx_real);
		output_file_xyz_boundary(filename,wf_3d,nx,ny,nz,bl,bf,bu,nnx,nny,nnz);


//////////////////////////////////vz
		for(int i=0;i<GPU_N;i++)
		{
			checkCudaErrors(cudaSetDevice(gpuid[i]));

			seperate_or_togather_vel_att2(mgdevice[i].wf4_h,wf_3d,nnx,nny,nnz,nnz_device,i,radius,1);
		}

			checkCudaErrors(cudaDeviceSynchronize());
		sprintf(filename,"./someoutput/cut-ex-vzp-%d-%d",sy_real,sx_real);
		output_file_xyz_boundary(filename,wf_3d,nx,ny,nz,bl,bf,bu,nnx,nny,nnz);
}

void output_3d_poyn_p(GPUdevice *mgdevice,int isx,int isy,int isz)
{
		for(int i=0;i<GPU_N;i++)
		{
			checkCudaErrors(cudaSetDevice(gpuid[i]));

			checkCudaErrors(cudaMemcpy(mgdevice[i].wf2_h,mgdevice[i].poyn_px_d,nnx*nny*nnz_device_append*sizeof(float),cudaMemcpyDefault));
			checkCudaErrors(cudaMemcpy(mgdevice[i].wf3_h,mgdevice[i].poyn_py_d,nnx*nny*nnz_device_append*sizeof(float),cudaMemcpyDefault));
			checkCudaErrors(cudaMemcpy(mgdevice[i].wf4_h,mgdevice[i].poyn_pz_d,nnx*nny*nnz_device_append*sizeof(float),cudaMemcpyDefault));

			//sprintf(filename,"./wavefield/ex-time-%d-%d",i,it);
			//output_file_xyz(filename,mgdevice[i].wf_h,nnx,nny,nnz_device_append);
		}
			checkCudaErrors(cudaDeviceSynchronize());
		
			
//////////////////////////////////vx
		for(int i=0;i<GPU_N;i++)
		{
			checkCudaErrors(cudaSetDevice(gpuid[i]));

			seperate_or_togather_vel_att2(mgdevice[i].wf2_h,wf_3d,nnx,nny,nnz,nnz_device,i,radius,1);
		}

			checkCudaErrors(cudaDeviceSynchronize());
		sprintf(filename,"./someoutput/cut-px-%d-%d",sy_real,sx_real);
		output_file_xyz_boundary(filename,wf_3d,nx,ny,nz,bl,bf,bu,nnx,nny,nnz);


//////////////////////////////////vy
		for(int i=0;i<GPU_N;i++)
		{
			checkCudaErrors(cudaSetDevice(gpuid[i]));

			seperate_or_togather_vel_att2(mgdevice[i].wf3_h,wf_3d,nnx,nny,nnz,nnz_device,i,radius,1);
		}

			checkCudaErrors(cudaDeviceSynchronize());
		sprintf(filename,"./someoutput/cut-py-%d-%d",sy_real,sx_real);
		output_file_xyz_boundary(filename,wf_3d,nx,ny,nz,bl,bf,bu,nnx,nny,nnz);


//////////////////////////////////vz
		for(int i=0;i<GPU_N;i++)
		{
			checkCudaErrors(cudaSetDevice(gpuid[i]));

			seperate_or_togather_vel_att2(mgdevice[i].wf4_h,wf_3d,nnx,nny,nnz,nnz_device,i,radius,1);
		}

			checkCudaErrors(cudaDeviceSynchronize());
		sprintf(filename,"./someoutput/cut-pz-%d-%d",sy_real,sx_real);
		output_file_xyz_boundary(filename,wf_3d,nx,ny,nz,bl,bf,bu,nnx,nny,nnz);
}



void output_3d_result(GPUdevice *mgdevice,int isx,int isy,int isz)
{
		if(vsp==0)
		{
			for(int i=0;i<GPU_N;i++)
			{
				checkCudaErrors(cudaSetDevice(gpuid[i]));

				checkCudaErrors(cudaMemcpy(mgdevice[i].wf_h,mgdevice[i].vresult_tp_d,nnx*nny*nnz_device_append*sizeof(float),cudaMemcpyDefault));
				checkCudaErrors(cudaMemcpy(mgdevice[i].wf1_h,mgdevice[i].vresult_pp_d,nnx*nny*nnz_device_append*sizeof(float),cudaMemcpyDefault));
				checkCudaErrors(cudaMemcpy(mgdevice[i].wf2_h,mgdevice[i].vresult_ps_d,nnx*nny*nnz_device_append*sizeof(float),cudaMemcpyDefault));
				
				//sprintf(filename,"./wavefield/ex-time-%d-%d",i,it);
				//output_file_xyz(filename,mgdevice[i].wf_h,nnx,nny,nnz_device_append);
			}
				checkCudaErrors(cudaDeviceSynchronize());
			
				
							
			for(int i=0;i<GPU_N;i++)
			{
				checkCudaErrors(cudaSetDevice(gpuid[i]));

				seperate_or_togather_vel_att2(mgdevice[i].wf_h,wf_3d,nnx,nny,nnz,nnz_device,i,radius,1);
			}

				checkCudaErrors(cudaDeviceSynchronize());

			//sprintf(filename,"./result/result-tp-%d-%d",sy_real,sx_real);
			//output_file_xyz(filename,wf_3d,nnx,nny,nnz);
			sprintf(filename,"./result/result-tp-%d-%d",sy_real,sx_real);
			output_file_xyz_boundary(filename,wf_3d,nx,ny,nz,bl,bf,bu,nnx,nny,nnz);

	//////////////////////////////////tp
			for(int i=0;i<GPU_N;i++)
			{
				checkCudaErrors(cudaSetDevice(gpuid[i]));

				seperate_or_togather_vel_att2(mgdevice[i].wf1_h,wf_3d,nnx,nny,nnz,nnz_device,i,radius,1);
			}

				checkCudaErrors(cudaDeviceSynchronize());
			
			sprintf(filename,"./result/result-pp-%d-%d",sy_real,sx_real);
			output_file_xyz_boundary(filename,wf_3d,nx,ny,nz,bl,bf,bu,nnx,nny,nnz);

	//////////////////////////////////vx
			for(int i=0;i<GPU_N;i++)
			{
				checkCudaErrors(cudaSetDevice(gpuid[i]));

				seperate_or_togather_vel_att2(mgdevice[i].wf2_h,wf_3d,nnx,nny,nnz,nnz_device,i,radius,1);
			}

				checkCudaErrors(cudaDeviceSynchronize());

			sprintf(filename,"./result/result-ps-%d-%d",sy_real,sx_real);
			output_file_xyz_boundary(filename,wf_3d,nx,ny,nz,bl,bf,bu,nnx,nny,nnz);
		}

		else
		{
			for(int i=0;i<GPU_N;i++)
			{
				checkCudaErrors(cudaSetDevice(gpuid[i]));

				checkCudaErrors(cudaMemcpy(mgdevice[i].wf_h,mgdevice[i].vresult_tp_d,nnx*nny*nnz_device_append*sizeof(float),cudaMemcpyDefault));
				checkCudaErrors(cudaMemcpy(mgdevice[i].wf1_h,mgdevice[i].vresult_pp_d,nnx*nny*nnz_device_append*sizeof(float),cudaMemcpyDefault));
				checkCudaErrors(cudaMemcpy(mgdevice[i].wf2_h,mgdevice[i].vresult_ps_d,nnx*nny*nnz_device_append*sizeof(float),cudaMemcpyDefault));
				
				//sprintf(filename,"./wavefield/ex-time-%d-%d",i,it);
				//output_file_xyz(filename,mgdevice[i].wf_h,nnx,nny,nnz_device_append);
			}
				checkCudaErrors(cudaDeviceSynchronize());
			
				
							
			for(int i=0;i<GPU_N;i++)
			{
				checkCudaErrors(cudaSetDevice(gpuid[i]));

				seperate_or_togather_vel_att2(mgdevice[i].wf_h,wf_3d,nnx,nny,nnz,nnz_device,i,radius,1);
			}

				checkCudaErrors(cudaDeviceSynchronize());

			//sprintf(filename,"./result/result-tp-%d",sz_real);
			//output_file_xyz(filename,wf_3d,nnx,nny,nnz);
			sprintf(filename,"./result/result-tp-%d",sz_real);
			output_file_xyz_boundary(filename,wf_3d,nx,ny,nz,bl,bf,bu,nnx,nny,nnz);

	//////////////////////////////////tp
			for(int i=0;i<GPU_N;i++)
			{
				checkCudaErrors(cudaSetDevice(gpuid[i]));

				seperate_or_togather_vel_att2(mgdevice[i].wf1_h,wf_3d,nnx,nny,nnz,nnz_device,i,radius,1);
			}

				checkCudaErrors(cudaDeviceSynchronize());
			
			sprintf(filename,"./result/result-pp-%d",sz_real);
			output_file_xyz_boundary(filename,wf_3d,nx,ny,nz,bl,bf,bu,nnx,nny,nnz);

	//////////////////////////////////vx
			for(int i=0;i<GPU_N;i++)
			{
				checkCudaErrors(cudaSetDevice(gpuid[i]));

				seperate_or_togather_vel_att2(mgdevice[i].wf2_h,wf_3d,nnx,nny,nnz,nnz_device,i,radius,1);
			}

				checkCudaErrors(cudaDeviceSynchronize());

			sprintf(filename,"./result/result-ps-%d",sz_real);
			output_file_xyz_boundary(filename,wf_3d,nx,ny,nz,bl,bf,bu,nnx,nny,nnz);
		}


}

void output_3d_result_compensate(GPUdevice *mgdevice,int isx,int isy,int isz)
{
		if(vsp==0)
		{
			for(int i=0;i<GPU_N;i++)
			{
				checkCudaErrors(cudaSetDevice(gpuid[i]));

				checkCudaErrors(cudaMemcpy(mgdevice[i].wf_h,mgdevice[i].vresult_tp_d,nnx*nny*nnz_device_append*sizeof(float),cudaMemcpyDefault));
				checkCudaErrors(cudaMemcpy(mgdevice[i].wf1_h,mgdevice[i].vresult_pp_d,nnx*nny*nnz_device_append*sizeof(float),cudaMemcpyDefault));
				checkCudaErrors(cudaMemcpy(mgdevice[i].wf2_h,mgdevice[i].vresult_ps_d,nnx*nny*nnz_device_append*sizeof(float),cudaMemcpyDefault));
				
				//sprintf(filename,"./wavefield/ex-time-%d-%d",i,it);
				//output_file_xyz(filename,mgdevice[i].wf_h,nnx,nny,nnz_device_append);
			}
				checkCudaErrors(cudaDeviceSynchronize());
			
				
							
			for(int i=0;i<GPU_N;i++)
			{
				checkCudaErrors(cudaSetDevice(gpuid[i]));

				seperate_or_togather_vel_att2(mgdevice[i].wf_h,wf_3d,nnx,nny,nnz,nnz_device,i,radius,1);
			}

				checkCudaErrors(cudaDeviceSynchronize());

			//sprintf(filename,"./result/result-tp-%d-%d",sy_real,sx_real);
			//output_file_xyz(filename,wf_3d,nnx,nny,nnz);
			sprintf(filename,"./result/compensate-result-tp-%d-%d",sy_real,sx_real);
			output_file_xyz_boundary(filename,wf_3d,nx,ny,nz,bl,bf,bu,nnx,nny,nnz);

//////////////////////////////////pp
			for(int i=0;i<GPU_N;i++)
			{
				checkCudaErrors(cudaSetDevice(gpuid[i]));

				seperate_or_togather_vel_att2(mgdevice[i].wf1_h,wf_3d,nnx,nny,nnz,nnz_device,i,radius,1);
			}

				checkCudaErrors(cudaDeviceSynchronize());
			
			sprintf(filename,"./result/compensate-result-pp-%d-%d",sy_real,sx_real);
			output_file_xyz_boundary(filename,wf_3d,nx,ny,nz,bl,bf,bu,nnx,nny,nnz);

//////////////////////////////////ps
			for(int i=0;i<GPU_N;i++)
			{
				checkCudaErrors(cudaSetDevice(gpuid[i]));

				seperate_or_togather_vel_att2(mgdevice[i].wf2_h,wf_3d,nnx,nny,nnz,nnz_device,i,radius,1);
			}

				checkCudaErrors(cudaDeviceSynchronize());

			sprintf(filename,"./result/compensate-result-ps-%d-%d",sy_real,sx_real);
			output_file_xyz_boundary(filename,wf_3d,nx,ny,nz,bl,bf,bu,nnx,nny,nnz);
		}

		else
		{
			for(int i=0;i<GPU_N;i++)
			{
				checkCudaErrors(cudaSetDevice(gpuid[i]));

				checkCudaErrors(cudaMemcpy(mgdevice[i].wf_h,mgdevice[i].vresult_tp_d,nnx*nny*nnz_device_append*sizeof(float),cudaMemcpyDefault));
				checkCudaErrors(cudaMemcpy(mgdevice[i].wf1_h,mgdevice[i].vresult_pp_d,nnx*nny*nnz_device_append*sizeof(float),cudaMemcpyDefault));
				checkCudaErrors(cudaMemcpy(mgdevice[i].wf2_h,mgdevice[i].vresult_ps_d,nnx*nny*nnz_device_append*sizeof(float),cudaMemcpyDefault));
				
				//sprintf(filename,"./wavefield/ex-time-%d-%d",i,it);
				//output_file_xyz(filename,mgdevice[i].wf_h,nnx,nny,nnz_device_append);
			}
				checkCudaErrors(cudaDeviceSynchronize());
			
				
							
			for(int i=0;i<GPU_N;i++)
			{
				checkCudaErrors(cudaSetDevice(gpuid[i]));

				seperate_or_togather_vel_att2(mgdevice[i].wf_h,wf_3d,nnx,nny,nnz,nnz_device,i,radius,1);
			}

				checkCudaErrors(cudaDeviceSynchronize());

			//sprintf(filename,"./result/result-tp-%d",sz_real);
			//output_file_xyz(filename,wf_3d,nnx,nny,nnz);
			sprintf(filename,"./result/compensate-result-tp-%d",sz_real);
			output_file_xyz_boundary(filename,wf_3d,nx,ny,nz,bl,bf,bu,nnx,nny,nnz);

//////////////////////////////////pp
			for(int i=0;i<GPU_N;i++)
			{
				checkCudaErrors(cudaSetDevice(gpuid[i]));

				seperate_or_togather_vel_att2(mgdevice[i].wf1_h,wf_3d,nnx,nny,nnz,nnz_device,i,radius,1);
			}

				checkCudaErrors(cudaDeviceSynchronize());
			
			sprintf(filename,"./result/compensate-result-pp-%d",sz_real);
			output_file_xyz_boundary(filename,wf_3d,nx,ny,nz,bl,bf,bu,nnx,nny,nnz);

//////////////////////////////////ps
			for(int i=0;i<GPU_N;i++)
			{
				checkCudaErrors(cudaSetDevice(gpuid[i]));

				seperate_or_togather_vel_att2(mgdevice[i].wf2_h,wf_3d,nnx,nny,nnz,nnz_device,i,radius,1);
			}

				checkCudaErrors(cudaDeviceSynchronize());

			sprintf(filename,"./result/compensate-result-ps-%d",sz_real);
			output_file_xyz_boundary(filename,wf_3d,nx,ny,nz,bl,bf,bu,nnx,nny,nnz);
		}


}
