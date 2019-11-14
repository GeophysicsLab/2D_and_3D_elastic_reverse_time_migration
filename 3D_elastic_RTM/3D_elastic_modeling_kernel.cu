
void seperate_or_togather_vel_att2(float *data_1d,float *data_3d,int nxb,int nyb,int nzb,int nzb_aver,int i,int orders,int mark)
{
	int ix,iy,iz;
	//ASSIGN THE BIG 3D ARRAY TO THE LITTLE 1D ARRAY ON HOST FOR EACH GPU
	for(iz=0;iz<nzb_aver;iz++)
	{
		for(iy=0;iy<nyb;iy++)
		{
			for(ix=0;ix<nxb;ix++)
			{
				if(i*nzb_aver+iz<nzb)
				{
					if(mark==0)	data_1d[(iz+orders)*nxb*nyb+iy*nxb+ix]=data_3d[(i*nzb_aver+iz)*nxb*nyb+iy*nxb+ix];

					else		data_3d[(i*nzb_aver+iz)*nxb*nyb+iy*nxb+ix]=data_1d[(iz+orders)*nxb*nyb+iy*nxb+ix];
				}
			}
		}
	}
}

void seperate_vel_att1(GPUdevice *mgdevice)
{
		for(int i=0;i<GPU_N;i++)
		{
				checkCudaErrors(cudaSetDevice(gpuid[i]));

				seperate_or_togather_vel_att2(mgdevice[i].velocity_h,velocity_pml,nnx,nny,nnz,nnz_device,i,radius,0);

				seperate_or_togather_vel_att2(mgdevice[i].velocity1_h,velocity1_pml,nnx,nny,nnz,nnz_device,i,radius,0);

				seperate_or_togather_vel_att2(mgdevice[i].density_h,density_pml,nnx,nny,nnz,nnz_device,i,radius,0);

				seperate_or_togather_vel_att2(mgdevice[i].att_h,att_pml,nnx,nny,nnz,nnz_device,i,radius,0);

			
				//sprintf(filename,"./someoutput/velocity_%d.bin",i);
				//output_3d(filename,mgdevice[i].velocity_h,nnx,nny,nnz_device_append);

				//sprintf(filename,"./someoutput/velocity1_%d.bin",i);
				//output_3d(filename,mgdevice[i].velocity1_h,nnx,nny,nnz_device_append);

				//sprintf(filename,"./someoutput/density1_%d.bin",i);
				//output_3d(filename,mgdevice[i].density1_h,nnx,nny,nnz_device_append);

				//sprintf(filename,"./someoutput/att_%d.bin",i);
				//output_3d(filename,mgdevice[i].att_h,nnx,nny,nnz_device_append);
		}

		cudaDeviceSynchronize();
}

void expand_nnz_residual(float *data_1d,int nnx,int nny,int nnz_device_append,int nnz_residual)
//expand_nnz_residual(mgdevice[GPU_N-1].density_h,nnx,nny,nz_device_append,nnz_residual);
{
	int ix,iy,iz;
	//ASSIGN THE BIG 3D ARRAY TO THE LITTLE 1D ARRAY ON HOST FOR EACH GPU
	for(iz=0;iz<nnz_residual;iz++)
	{
		for(iy=0;iy<nny;iy++)
		{
			for(ix=0;ix<nnx;ix++)
			{
				data_1d[(nnz_device_append-radius-nnz_residual+iz)*nnx*nny+iy*nnx+ix]=data_1d[(nnz_device_append-radius-nnz_residual-1)*nnx*nny+iy*nnx+ix];
			}
		}
	}
}

void elastic_modeling_parameter_cpu_to_gpu(GPUdevice *mgdevice)
{
	
		for(int i=0;i<GPU_N;i++)
		{
				checkCudaErrors(cudaSetDevice(gpuid[i]));

				checkCudaErrors(cudaMemcpy(mgdevice[i].velocity_d,mgdevice[i].velocity_h,nnx*nny*nnz_device_append*sizeof(float),cudaMemcpyDefault));

				checkCudaErrors(cudaMemcpy(mgdevice[i].velocity1_d,mgdevice[i].velocity1_h,nnx*nny*nnz_device_append*sizeof(float),cudaMemcpyDefault));

				checkCudaErrors(cudaMemcpy(mgdevice[i].density_d,mgdevice[i].density_h,nnx*nny*nnz_device_append*sizeof(float),cudaMemcpyDefault));

				checkCudaErrors(cudaMemcpy(mgdevice[i].att_d,mgdevice[i].att_h,nnx*nny*nnz_device_append*sizeof(float),cudaMemcpyDefault));

				checkCudaErrors(cudaMemcpy(mgdevice[i].coe_d,coe_opt,(radius+1)*sizeof(float),cudaMemcpyDefault));

				checkCudaErrors(cudaMemcpy(mgdevice[i].wavelet_d,wavelet,wavelet_length*sizeof(float),cudaMemcpyDefault));
		}

		checkCudaErrors(cudaDeviceSynchronize());

}


void elastic_RTM_parameter_cpu_to_gpu(GPUdevice *mgdevice)
{
	
		for(int i=0;i<GPU_N;i++)
		{
				checkCudaErrors(cudaSetDevice(gpuid[i]));

				checkCudaErrors(cudaMemcpy(mgdevice[i].s_velocity_d,mgdevice[i].velocity_h,nnx*nny*nnz_device_append*sizeof(float),cudaMemcpyDefault));

				checkCudaErrors(cudaMemcpy(mgdevice[i].s_velocity1_d,mgdevice[i].velocity1_h,nnx*nny*nnz_device_append*sizeof(float),cudaMemcpyDefault));

				checkCudaErrors(cudaMemcpy(mgdevice[i].s_density_d,mgdevice[i].density_h,nnx*nny*nnz_device_append*sizeof(float),cudaMemcpyDefault));
		}

		checkCudaErrors(cudaDeviceSynchronize());

}

void get_real_model_parameter()
{
			input_file_xyz_boundary(velocity_name,velocity_pml,nx,ny,nz,bl,bb,bu,nnx,nny,nnz);
			//output_file_xyz("./someoutput/velocity_all1.bin",velocity_pml,nnx,nny,nnz);
			//output_file_xyz_boundary("./someoutput/velocity_cut.bin",velocity_pml,nx,ny,nz,bl,bb,bu,nnx,nny,nnz);

			add_pml_layers_v_h(velocity_pml,nx,ny,nz,bl,bb,bu,nnx,nny,nnz);	
			output_file_xyz("./someoutput/vp_all.bin",velocity_pml,nnx,nny,nnz);
					

			input_file_xyz_boundary(velocity1_name,velocity1_pml,nx,ny,nz,bl,bb,bu,nnx,nny,nnz);	
			add_pml_layers_v_h(velocity1_pml,nx,ny,nz,bl,bb,bu,nnx,nny,nnz);
			output_file_xyz("./someoutput/vs_all.bin",velocity1_pml,nnx,nny,nnz);


			input_file_xyz_boundary(density_name,density_pml,nx,ny,nz,bl,bb,bu,nnx,nny,nnz);	
			add_pml_layers_v_h(density_pml,nx,ny,nz,bl,bb,bu,nnx,nny,nnz);
			output_file_xyz("./someoutput/den_all.bin",density_pml,nnx,nny,nnz);
}

void get_smoothed_model_parameter()
{

////////////////////////read smoothed vp
			if(smooth_time_vp!=0)	
			{	
				openfile=fopen("smooth_3d","wb+");//////cal_shot_*_iter_1  res_shot_*_iter_1
				fprintf(openfile,"#!/bin/sh\n");

				fprintf(openfile,"smooth3d< ./someoutput/vp_all.bin n1=%d n2=%d n3=%d r1=%f r2=%f r3=%f >./someoutput/s_vp.bin \n",nnz,nny,nnx,smooth_time_vp,smooth_time_vp,smooth_time_vp);					
				fclose(openfile);
				system("sh smooth_3d");
				input_file_xyz("./someoutput/s_vp.bin",velocity_pml,nnx,nny,nnz);

				output_file_xyz_boundary("./someoutput/cut_s_vp.bin",velocity_pml,nx,ny,nz,bl,bb,bu,nnx,nny,nnz);
			}



////////////////////////read smoothed vs
			if(smooth_time_vs!=0)	
			{	
				openfile=fopen("smooth_3d","wb+");//////cal_shot_*_iter_1  res_shot_*_iter_1
				fprintf(openfile,"#!/bin/sh\n");

				fprintf(openfile,"smooth3d< ./someoutput/vs_all.bin n1=%d n2=%d n3=%d r1=%f r2=%f r3=%f >./someoutput/s_vs.bin \n",nnz,nny,nnx,smooth_time_vs,smooth_time_vs,smooth_time_vs);					
				fclose(openfile);
				system("sh smooth_3d");
				input_file_xyz("./someoutput/s_vs.bin",velocity1_pml,nnx,nny,nnz);

				output_file_xyz_boundary("./someoutput/cut_s_vs.bin",velocity1_pml,nx,ny,nz,bl,bb,bu,nnx,nny,nnz);
			}


		
////////////////////////read smoothed den			
			if(smooth_time_density!=0)	
			{	
				openfile=fopen("smooth_3d","wb+");//////cal_shot_*_iter_1  res_shot_*_iter_1
				fprintf(openfile,"#!/bin/sh\n");

				fprintf(openfile,"smooth3d< ./someoutput/den_all.bin n1=%d n2=%d n3=%d r1=%f r2=%f r3=%f >./someoutput/s_den.bin \n",nnz,nny,nnx,smooth_time_density,smooth_time_density,smooth_time_density);					
				fclose(openfile);
				system("sh smooth_3d");
				input_file_xyz("./someoutput/s_den.bin",density_pml,nnx,nny,nnz);

				output_file_xyz_boundary("./someoutput/cut_s_den.bin",density_pml,nx,ny,nz,bl,bb,bu,nnx,nny,nnz);
			}

}




void exchange_wavefiled_new(GPUdevice *mgdevice)
{
		for(int i=0;i<GPU_N;i++)
		{
			checkCudaErrors(cudaSetDevice(gpuid[i]));
				
			mgdevice[i].rep=mgdevice[i].vx1_d;	mgdevice[i].vx1_d=mgdevice[i].vx2_d;	mgdevice[i].vx2_d=mgdevice[i].rep;
			mgdevice[i].rep=mgdevice[i].vy1_d;	mgdevice[i].vy1_d=mgdevice[i].vy2_d;	mgdevice[i].vy2_d=mgdevice[i].rep;
			mgdevice[i].rep=mgdevice[i].vz1_d;	mgdevice[i].vz1_d=mgdevice[i].vz2_d;	mgdevice[i].vz2_d=mgdevice[i].rep;
			
			mgdevice[i].rep=mgdevice[i].txx1_d;	mgdevice[i].txx1_d=mgdevice[i].txx2_d;	mgdevice[i].txx2_d=mgdevice[i].rep;
			mgdevice[i].rep=mgdevice[i].tyy1_d;	mgdevice[i].tyy1_d=mgdevice[i].tyy2_d;	mgdevice[i].tyy2_d=mgdevice[i].rep;
			mgdevice[i].rep=mgdevice[i].tzz1_d;	mgdevice[i].tzz1_d=mgdevice[i].tzz2_d;	mgdevice[i].tzz2_d=mgdevice[i].rep;

			mgdevice[i].rep=mgdevice[i].txy1_d;	mgdevice[i].txy1_d=mgdevice[i].txy2_d;	mgdevice[i].txy2_d=mgdevice[i].rep;
			mgdevice[i].rep=mgdevice[i].txz1_d;	mgdevice[i].txz1_d=mgdevice[i].txz2_d;	mgdevice[i].txz2_d=mgdevice[i].rep;
			mgdevice[i].rep=mgdevice[i].tyz1_d;	mgdevice[i].tyz1_d=mgdevice[i].tyz2_d;	mgdevice[i].tyz2_d=mgdevice[i].rep;
			
			
			mgdevice[i].rep=mgdevice[i].tp1_d;	mgdevice[i].tp1_d=mgdevice[i].tp2_d;	mgdevice[i].tp2_d=mgdevice[i].rep;
			mgdevice[i].rep=mgdevice[i].vxp1_d;	mgdevice[i].vxp1_d=mgdevice[i].vxp2_d;	mgdevice[i].vxp2_d=mgdevice[i].rep;
			mgdevice[i].rep=mgdevice[i].vyp1_d;	mgdevice[i].vyp1_d=mgdevice[i].vyp2_d;	mgdevice[i].vyp2_d=mgdevice[i].rep;
			mgdevice[i].rep=mgdevice[i].vzp1_d;	mgdevice[i].vzp1_d=mgdevice[i].vzp2_d;	mgdevice[i].vzp2_d=mgdevice[i].rep;
			mgdevice[i].rep=mgdevice[i].vxs1_d;	mgdevice[i].vxs1_d=mgdevice[i].vxs2_d;	mgdevice[i].vxs2_d=mgdevice[i].rep;
			mgdevice[i].rep=mgdevice[i].vys1_d;	mgdevice[i].vys1_d=mgdevice[i].vys2_d;	mgdevice[i].vys2_d=mgdevice[i].rep;
			mgdevice[i].rep=mgdevice[i].vzs1_d;	mgdevice[i].vzs1_d=mgdevice[i].vzs2_d;	mgdevice[i].vzs2_d=mgdevice[i].rep;
		}

		checkCudaErrors(cudaDeviceSynchronize());

}

__global__ void exchange_wf(float *wf1,float *wf2,int nnx,int nny,int nnz)
{
		//int ix=blockIdx.x*blockDim.x+threadIdx.x;
		//int iy=blockIdx.y*blockDim.y+threadIdx.y;
		//int iz=blockIdx.z*blockDim.z+threadIdx.z;;

		int ix=blockIdx.x*blockDim.x+threadIdx.x;
		int iy=blockIdx.y*blockDim.y+threadIdx.y;
		int iz=blockIdx.z;

		int indx;
		float change;

		if(ix<nnx&&iy<nny&&iz<nnz)
		{
				indx=iz*nnx*nny+iy*nnx+ix;
				change=wf1[indx];
				wf1[indx]=wf2[indx];
				wf2[indx]=change;
		}
}

void exchange_wavefiled_old(GPUdevice *mgdevice)
{
		dim3 dimBlock(32,16);

		dim3 dimGridwf_append((nnx+dimBlock.x-1)/dimBlock.x,(nny+dimBlock.y-1)/dimBlock.y,nnz_device_append);//////单块卡的整个空间

		for(int i=0;i<GPU_N;i++)
		{
			checkCudaErrors(cudaSetDevice(gpuid[i]));
			exchange_wf<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].vx1_d,mgdevice[i].vx2_d,nnx,nny,nnz_device_append);
			exchange_wf<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].vy1_d,mgdevice[i].vy2_d,nnx,nny,nnz_device_append);
			exchange_wf<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].vz1_d,mgdevice[i].vz2_d,nnx,nny,nnz_device_append);

			exchange_wf<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].txx1_d,mgdevice[i].txx2_d,nnx,nny,nnz_device_append);
			exchange_wf<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].tyy1_d,mgdevice[i].tyy2_d,nnx,nny,nnz_device_append);
			exchange_wf<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].tzz1_d,mgdevice[i].tzz2_d,nnx,nny,nnz_device_append);

			exchange_wf<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].txy1_d,mgdevice[i].txy2_d,nnx,nny,nnz_device_append);
			exchange_wf<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].txz1_d,mgdevice[i].txz2_d,nnx,nny,nnz_device_append);
			exchange_wf<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].tyz1_d,mgdevice[i].tyz2_d,nnx,nny,nnz_device_append);

			exchange_wf<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].tp1_d,mgdevice[i].tp2_d,nnx,nny,nnz_device_append);
			exchange_wf<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].vxp1_d,mgdevice[i].vxp2_d,nnx,nny,nnz_device_append);
			exchange_wf<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].vyp1_d,mgdevice[i].vyp2_d,nnx,nny,nnz_device_append);
			exchange_wf<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].vzp1_d,mgdevice[i].vzp2_d,nnx,nny,nnz_device_append);
			exchange_wf<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].vxs1_d,mgdevice[i].vxs2_d,nnx,nny,nnz_device_append);
			exchange_wf<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].vys1_d,mgdevice[i].vys2_d,nnx,nny,nnz_device_append);
			exchange_wf<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].vzs1_d,mgdevice[i].vzs2_d,nnx,nny,nnz_device_append);
		}
			
		checkCudaErrors(cudaDeviceSynchronize());
}



//////////////////////////////////////////////////////in GPU
__global__ void add_source_3D(float *wf_d,float *wavelet_d,int nnx,int nny,int nnz_device_append,int nnz_device,int it,int sx,int sy,int sz,int bl,int bb,int bu)
//add_source_3D<<<1,1,0,mgdevice[choose_ns].stream>>>(mgdevice[choose_ns].txx1_d,mgdevice[choose_ns].wavelet_d,nnx,nny,nnz_device_append,nnz_device,it,sx_real,sy_real,sz_real,bl,bb,bu);
{
		int sx_real=sx+bl;
	
		int sy_real=sy+bb;

		//int choose_ns=(sz+bu)/nnz_device;
		//int sz_real=sz+bu-choose_ns*nnz_device+radius;	

		int sz_real=(sz+bu)%nnz_device+radius;
	
		int id=sz_real*(nnx*nny)+sy_real*nnx+sx_real;

		wf_d[id]+=wavelet_d[it];

		//wf_d[id]+=1.0;///test!!!!
}

__global__ void exchange_device(float *wf1_device1,float *wf2_device1,float *wf1_device2,float *wf2_device2,int nnx,int nny,int nnz)
//exchange_device<<<dimGridwf_append,dimBlock>>>(mgdevice[i].wf1_d,mgdevice[i].wf2_d,mgdevice[i+1].wf1_d,mgdevice[i+1].wf2_d,nnx_device_append,nny,nnz);
{
		int ix=blockIdx.x*blockDim.x+threadIdx.x;
		int iy=blockIdx.y*blockDim.y+threadIdx.y;
		int iz=blockIdx.z;

		if(ix<radius&&iy<nny&&iz<nnz)
		{
				wf1_device1[iz*nnx*nny+iy*nnx+nnx-radius+ix]=wf1_device2[iz*nnx*nny+iy*nnx+radius+ix];
				wf2_device1[iz*nnx*nny+iy*nnx+nnx-radius+ix]=wf2_device2[iz*nnx*nny+iy*nnx+radius+ix];

				wf1_device2[iz*nnx*nny+iy*nnx+ix]=wf1_device1[iz*nnx*nny+iy*nnx+nnx-2*radius+ix];
				wf2_device2[iz*nnx*nny+iy*nnx+ix]=wf2_device1[iz*nnx*nny+iy*nnx+nnx-2*radius+ix];

				//wf1_device2[iz*nnx*nny+iy*nnx+nnx-radius+ix]=wf1_device3[iz*nnx*nny+iy*nnx+radius+ix];
				//wf2_device2[iz*nnx*nny+iy*nnx+nnx-radius+ix]=wf2_device3[iz*nnx*nny+iy*nnx+radius+ix];

				//wf1_device3[iz*nnx*nny+iy*nnx+ix]=wf1_device2[iz*nnx*nny+iy*nnx+nnx-2*radius+ix];
				//wf2_device3[iz*nnx*nny+iy*nnx+ix]=wf2_device2[iz*nnx*nny+iy*nnx+nnx-2*radius+ix];
		}
}

__global__ void exchange_device_new(float *wf1_device1,float *wf2_device1,float *wf1_device2,float *wf2_device2,int nx,int ny,int nz1,int nz2,int orders)
//exchange_device_new<<<dimGrid3D,dimBlock2D,0,plan[i].stream>>>(plan[i-1].wf1_d,plan[i-1].wf2_d,plan[i].wf1_d,plan[i].wf2_d,plan[i].nxb,plan[i].nyb,plan[i-1].nzb,plan[i].nzb,orders);
{
		int tx=blockIdx.x*blockDim.x+threadIdx.x;
		int ty=blockIdx.y*blockDim.y+threadIdx.y;
		int tz=blockIdx.z;

		if(tx<nx&&ty<ny&&tz<orders)
		{

				wf1_device2[tz*nx*ny+ty*nx+tx]=wf1_device1[(nz1-2*orders+tz)*nx*ny+ty*nx+tx];
				wf2_device2[tz*nx*ny+ty*nx+tx]=wf2_device1[(nz1-2*orders+tz)*nx*ny+ty*nx+tx];

				wf1_device1[(nz1-orders+tz)*nx*ny+ty*nx+tx]=wf1_device2[(tz+orders)*nx*ny+ty*nx+tx];
				wf2_device1[(nz1-orders+tz)*nx*ny+ty*nx+tx]=wf2_device2[(tz+orders)*nx*ny+ty*nx+tx];

		}
}

__global__ void exchange_device_nz(float *wf1_device1,float *wf2_device1,float *wf1_device2,float *wf2_device2,int nnx,int nny,int nnz)
//exchange_device_nz<<<dimGridwf_append,dimBlock>>>(mgdevice[i].vx1_d,mgdevice[i].vx2_d,mgdevice[i+1].vx1_d,mgdevice[i+1].vx2_d,nnx,nny,nnz_device_append);
{
		int ix=blockIdx.x*blockDim.x+threadIdx.x;
		int iy=blockIdx.y*blockDim.y+threadIdx.y;
		int iz=blockIdx.z;

		if(ix<nnx&&iy<nny&&iz<radius)
		{
				wf1_device2[iz*nnx*nny+iy*nnx+ix]=wf1_device1[(nnz+iz-2*radius)*nnx*nny+iy*nnx+ix];
				wf2_device2[iz*nnx*nny+iy*nnx+ix]=wf2_device1[(nnz+iz-2*radius)*nnx*nny+iy*nnx+ix];//////important


				wf1_device1[(nnz-radius+iz)*nnx*nny+iy*nnx+ix]=wf1_device2[(iz+radius)*nnx*nny+iy*nnx+ix];
				wf2_device1[(nnz-radius+iz)*nnx*nny+iy*nnx+ix]=wf2_device2[(iz+radius)*nnx*nny+iy*nnx+ix];
		}
}

__global__ void exchange_device_nz_one(float *wf1_device1,float *wf1_device2,int nnx,int nny,int nnz)
//exchange_device_nz<<<dimGridwf_append,dimBlock>>>(mgdevice[i].vx1_d,mgdevice[i].vx2_d,mgdevice[i+1].vx1_d,mgdevice[i+1].vx2_d,nnx,nny,nnz_device_append);
{
		int ix=blockIdx.x*blockDim.x+threadIdx.x;
		int iy=blockIdx.y*blockDim.y+threadIdx.y;
		int iz=blockIdx.z;

		if(ix<nnx&&iy<nny&&iz<radius)
		{
				wf1_device2[iz*nnx*nny+iy*nnx+ix]=wf1_device1[(nnz+iz-2*radius)*nnx*nny+iy*nnx+ix];//////important


				wf1_device1[(nnz-radius+iz)*nnx*nny+iy*nnx+ix]=wf1_device2[(iz+radius)*nnx*nny+iy*nnx+ix];
		}
}

void exchange_device_nz_kernel_txxyyzz(GPUdevice *mgdevice,int mark)
{

			dim3 dimBlock(32,16);

			dim3 dimGridwf_append((nnx+dimBlock.x-1)/dimBlock.x,(nny+dimBlock.y-1)/dimBlock.y,nnz_device_append);//////单块卡的整个空间

			for(int i=0;i<GPU_N-1;i++)
			{
				if(mark==0)
				{
					checkCudaErrors(cudaSetDevice(gpuid[0]));

					exchange_device_nz<<<dimGridwf_append,dimBlock>>>(mgdevice[i].txx1_d,mgdevice[i].txx2_d,mgdevice[i+1].txx1_d,mgdevice[i+1].txx2_d,nnx,nny,nnz_device_append);
					exchange_device_nz<<<dimGridwf_append,dimBlock>>>(mgdevice[i].tyy1_d,mgdevice[i].tyy2_d,mgdevice[i+1].tyy1_d,mgdevice[i+1].tyy2_d,nnx,nny,nnz_device_append);
					exchange_device_nz<<<dimGridwf_append,dimBlock>>>(mgdevice[i].tzz1_d,mgdevice[i].tzz2_d,mgdevice[i+1].tzz1_d,mgdevice[i+1].tzz2_d,nnx,nny,nnz_device_append);
				}


				if(mark==1)
				{
					checkCudaErrors(cudaSetDevice(gpuid[0]));

					exchange_device_nz_one<<<dimGridwf_append,dimBlock>>>(mgdevice[i].txx1_d,mgdevice[i+1].txx1_d,nnx,nny,nnz_device_append);
					exchange_device_nz_one<<<dimGridwf_append,dimBlock>>>(mgdevice[i].tyy1_d,mgdevice[i+1].tyy1_d,nnx,nny,nnz_device_append);
					exchange_device_nz_one<<<dimGridwf_append,dimBlock>>>(mgdevice[i].tzz1_d,mgdevice[i+1].tzz1_d,nnx,nny,nnz_device_append);
				}

				if(mark==2)
				{
					checkCudaErrors(cudaSetDevice(gpuid[i]));

					exchange_device_nz<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].txx1_d,mgdevice[i].txx2_d,mgdevice[i+1].txx1_d,mgdevice[i+1].txx2_d,nnx,nny,nnz_device_append);
					exchange_device_nz<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].tyy1_d,mgdevice[i].tyy2_d,mgdevice[i+1].tyy1_d,mgdevice[i+1].tyy2_d,nnx,nny,nnz_device_append);
					exchange_device_nz<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].tzz1_d,mgdevice[i].tzz2_d,mgdevice[i+1].tzz1_d,mgdevice[i+1].tzz2_d,nnx,nny,nnz_device_append);
				}

				if(mark==3)
				{
					checkCudaErrors(cudaSetDevice(gpuid[i]));

					exchange_device_nz_one<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].txx1_d,mgdevice[i+1].txx1_d,nnx,nny,nnz_device_append);
					exchange_device_nz_one<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].tyy1_d,mgdevice[i+1].tyy1_d,nnx,nny,nnz_device_append);
					exchange_device_nz_one<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].tzz1_d,mgdevice[i+1].tzz1_d,nnx,nny,nnz_device_append);
				}
			}

			checkCudaErrors(cudaDeviceSynchronize());
}

void exchange_device_nz_kernel_tao1(GPUdevice *mgdevice,int mark)
{

			dim3 dimBlock(32,16);

			dim3 dimGridwf_append((nnx+dimBlock.x-1)/dimBlock.x,(nny+dimBlock.y-1)/dimBlock.y,nnz_device_append);//////单块卡的整个空间

			for(int i=0;i<GPU_N-1;i++)
			{
				if(mark==0)
				{
					checkCudaErrors(cudaSetDevice(gpuid[0]));

					exchange_device_nz<<<dimGridwf_append,dimBlock>>>(mgdevice[i].txx1_d,mgdevice[i].txx2_d,mgdevice[i+1].txx1_d,mgdevice[i+1].txx2_d,nnx,nny,nnz_device_append);
					exchange_device_nz<<<dimGridwf_append,dimBlock>>>(mgdevice[i].tyy1_d,mgdevice[i].tyy2_d,mgdevice[i+1].tyy1_d,mgdevice[i+1].tyy2_d,nnx,nny,nnz_device_append);
					exchange_device_nz<<<dimGridwf_append,dimBlock>>>(mgdevice[i].tzz1_d,mgdevice[i].tzz2_d,mgdevice[i+1].tzz1_d,mgdevice[i+1].tzz2_d,nnx,nny,nnz_device_append);
					exchange_device_nz<<<dimGridwf_append,dimBlock>>>(mgdevice[i].txy1_d,mgdevice[i].txy2_d,mgdevice[i+1].txy1_d,mgdevice[i+1].txy2_d,nnx,nny,nnz_device_append);
					exchange_device_nz<<<dimGridwf_append,dimBlock>>>(mgdevice[i].txz1_d,mgdevice[i].txz2_d,mgdevice[i+1].txz1_d,mgdevice[i+1].txz2_d,nnx,nny,nnz_device_append);					
					exchange_device_nz<<<dimGridwf_append,dimBlock>>>(mgdevice[i].tyz1_d,mgdevice[i].tyz2_d,mgdevice[i+1].tyz1_d,mgdevice[i+1].tyz2_d,nnx,nny,nnz_device_append);
				}


				if(mark==1)
				{
					checkCudaErrors(cudaSetDevice(gpuid[0]));

					exchange_device_nz_one<<<dimGridwf_append,dimBlock>>>(mgdevice[i].txx1_d,mgdevice[i+1].txx1_d,nnx,nny,nnz_device_append);
					exchange_device_nz_one<<<dimGridwf_append,dimBlock>>>(mgdevice[i].tyy1_d,mgdevice[i+1].tyy1_d,nnx,nny,nnz_device_append);
					exchange_device_nz_one<<<dimGridwf_append,dimBlock>>>(mgdevice[i].tzz1_d,mgdevice[i+1].tzz1_d,nnx,nny,nnz_device_append);
					exchange_device_nz_one<<<dimGridwf_append,dimBlock>>>(mgdevice[i].txy1_d,mgdevice[i+1].txy1_d,nnx,nny,nnz_device_append);
					exchange_device_nz_one<<<dimGridwf_append,dimBlock>>>(mgdevice[i].txz1_d,mgdevice[i+1].txz1_d,nnx,nny,nnz_device_append);
					exchange_device_nz_one<<<dimGridwf_append,dimBlock>>>(mgdevice[i].tyz1_d,mgdevice[i+1].tyz1_d,nnx,nny,nnz_device_append);
				}

				if(mark==2)
				{
					checkCudaErrors(cudaSetDevice(gpuid[i]));

					exchange_device_nz<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].txx1_d,mgdevice[i].txx2_d,mgdevice[i+1].txx1_d,mgdevice[i+1].txx2_d,nnx,nny,nnz_device_append);
					exchange_device_nz<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].tyy1_d,mgdevice[i].tyy2_d,mgdevice[i+1].tyy1_d,mgdevice[i+1].tyy2_d,nnx,nny,nnz_device_append);
					exchange_device_nz<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].tzz1_d,mgdevice[i].tzz2_d,mgdevice[i+1].tzz1_d,mgdevice[i+1].tzz2_d,nnx,nny,nnz_device_append);
					exchange_device_nz<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].txy1_d,mgdevice[i].txy2_d,mgdevice[i+1].txy1_d,mgdevice[i+1].txy2_d,nnx,nny,nnz_device_append);
					exchange_device_nz<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].txz1_d,mgdevice[i].txz2_d,mgdevice[i+1].txz1_d,mgdevice[i+1].txz2_d,nnx,nny,nnz_device_append);
					exchange_device_nz<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].tyz1_d,mgdevice[i].tyz2_d,mgdevice[i+1].tyz1_d,mgdevice[i+1].tyz2_d,nnx,nny,nnz_device_append);
				}

				if(mark==3)
				{
					checkCudaErrors(cudaSetDevice(gpuid[i]));

					exchange_device_nz_one<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].txx1_d,mgdevice[i+1].txx1_d,nnx,nny,nnz_device_append);
					exchange_device_nz_one<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].tyy1_d,mgdevice[i+1].tyy1_d,nnx,nny,nnz_device_append);
					exchange_device_nz_one<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].tzz1_d,mgdevice[i+1].tzz1_d,nnx,nny,nnz_device_append);
					exchange_device_nz_one<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].txy1_d,mgdevice[i+1].txy1_d,nnx,nny,nnz_device_append);
					exchange_device_nz_one<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].txz1_d,mgdevice[i+1].txz1_d,nnx,nny,nnz_device_append);
					exchange_device_nz_one<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].tyz1_d,mgdevice[i+1].tyz1_d,nnx,nny,nnz_device_append);
				}
			}

			checkCudaErrors(cudaDeviceSynchronize());
}

void exchange_device_nz_kernel_tao2(GPUdevice *mgdevice,int mark)
///exchange_device_nz_kernel_tao(mgdevice);
{

			dim3 dimBlock(32,16);

			dim3 dimGridwf_append((nnx+dimBlock.x-1)/dimBlock.x,(nny+dimBlock.y-1)/dimBlock.y,nnz_device_append);//////单块卡的整个空间

			for(int i=0;i<GPU_N-1;i++)
			{
				if(mark==0)
				{
					checkCudaErrors(cudaSetDevice(gpuid[0]));

					exchange_device_nz<<<dimGridwf_append,dimBlock>>>(mgdevice[i].txx1_d,mgdevice[i].txx2_d,mgdevice[i+1].txx1_d,mgdevice[i+1].txx2_d,nnx,nny,nnz_device_append);
					exchange_device_nz<<<dimGridwf_append,dimBlock>>>(mgdevice[i].tyy1_d,mgdevice[i].tyy2_d,mgdevice[i+1].tyy1_d,mgdevice[i+1].tyy2_d,nnx,nny,nnz_device_append);
					exchange_device_nz<<<dimGridwf_append,dimBlock>>>(mgdevice[i].tzz1_d,mgdevice[i].tzz2_d,mgdevice[i+1].tzz1_d,mgdevice[i+1].tzz2_d,nnx,nny,nnz_device_append);
					exchange_device_nz<<<dimGridwf_append,dimBlock>>>(mgdevice[i].txy1_d,mgdevice[i].txy2_d,mgdevice[i+1].txy1_d,mgdevice[i+1].txy2_d,nnx,nny,nnz_device_append);
					exchange_device_nz<<<dimGridwf_append,dimBlock>>>(mgdevice[i].txz1_d,mgdevice[i].txz2_d,mgdevice[i+1].txz1_d,mgdevice[i+1].txz2_d,nnx,nny,nnz_device_append);					
					exchange_device_nz<<<dimGridwf_append,dimBlock>>>(mgdevice[i].tyz1_d,mgdevice[i].tyz2_d,mgdevice[i+1].tyz1_d,mgdevice[i+1].tyz2_d,nnx,nny,nnz_device_append);
				}


				if(mark==1)
				{
					checkCudaErrors(cudaSetDevice(gpuid[0]));

					exchange_device_nz_one<<<dimGridwf_append,dimBlock>>>(mgdevice[i].txx2_d,mgdevice[i+1].txx2_d,nnx,nny,nnz_device_append);
					exchange_device_nz_one<<<dimGridwf_append,dimBlock>>>(mgdevice[i].tyy2_d,mgdevice[i+1].tyy2_d,nnx,nny,nnz_device_append);
					exchange_device_nz_one<<<dimGridwf_append,dimBlock>>>(mgdevice[i].tzz2_d,mgdevice[i+1].tzz2_d,nnx,nny,nnz_device_append);
					exchange_device_nz_one<<<dimGridwf_append,dimBlock>>>(mgdevice[i].txy2_d,mgdevice[i+1].txy2_d,nnx,nny,nnz_device_append);
					exchange_device_nz_one<<<dimGridwf_append,dimBlock>>>(mgdevice[i].txz2_d,mgdevice[i+1].txz2_d,nnx,nny,nnz_device_append);
					exchange_device_nz_one<<<dimGridwf_append,dimBlock>>>(mgdevice[i].tyz2_d,mgdevice[i+1].tyz2_d,nnx,nny,nnz_device_append);
				}

				if(mark==2)
				{
					checkCudaErrors(cudaSetDevice(gpuid[i]));

					exchange_device_nz<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].txx1_d,mgdevice[i].txx2_d,mgdevice[i+1].txx1_d,mgdevice[i+1].txx2_d,nnx,nny,nnz_device_append);
					exchange_device_nz<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].tyy1_d,mgdevice[i].tyy2_d,mgdevice[i+1].tyy1_d,mgdevice[i+1].tyy2_d,nnx,nny,nnz_device_append);
					exchange_device_nz<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].tzz1_d,mgdevice[i].tzz2_d,mgdevice[i+1].tzz1_d,mgdevice[i+1].tzz2_d,nnx,nny,nnz_device_append);
					exchange_device_nz<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].txy1_d,mgdevice[i].txy2_d,mgdevice[i+1].txy1_d,mgdevice[i+1].txy2_d,nnx,nny,nnz_device_append);
					exchange_device_nz<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].txz1_d,mgdevice[i].txz2_d,mgdevice[i+1].txz1_d,mgdevice[i+1].txz2_d,nnx,nny,nnz_device_append);
					exchange_device_nz<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].tyz1_d,mgdevice[i].tyz2_d,mgdevice[i+1].tyz1_d,mgdevice[i+1].tyz2_d,nnx,nny,nnz_device_append);
				}

				if(mark==3)
				{
					checkCudaErrors(cudaSetDevice(gpuid[i]));

					exchange_device_nz_one<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].txx2_d,mgdevice[i+1].txx2_d,nnx,nny,nnz_device_append);
					exchange_device_nz_one<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].tyy2_d,mgdevice[i+1].tyy2_d,nnx,nny,nnz_device_append);
					exchange_device_nz_one<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].tzz2_d,mgdevice[i+1].tzz2_d,nnx,nny,nnz_device_append);
					exchange_device_nz_one<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].txy2_d,mgdevice[i+1].txy2_d,nnx,nny,nnz_device_append);
					exchange_device_nz_one<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].txz2_d,mgdevice[i+1].txz2_d,nnx,nny,nnz_device_append);
					exchange_device_nz_one<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].tyz2_d,mgdevice[i+1].tyz2_d,nnx,nny,nnz_device_append);
				}
			}

			checkCudaErrors(cudaDeviceSynchronize());
}

void exchange_device_nz_kernel_vx_vy_vz1(GPUdevice *mgdevice,int mark)
///exchange_device_nz_kernel(mgdevice);
{

			dim3 dimBlock(32,16);

			dim3 dimGridwf_append((nnx+dimBlock.x-1)/dimBlock.x,(nny+dimBlock.y-1)/dimBlock.y,nnz_device_append);//////单块卡的整个空间

			for(int i=0;i<GPU_N-1;i++)
			{
				if(mark==0)
				{
					checkCudaErrors(cudaSetDevice(gpuid[0]));

					exchange_device_nz<<<dimGridwf_append,dimBlock>>>(mgdevice[i].vx1_d,mgdevice[i].vx2_d,mgdevice[i+1].vx1_d,mgdevice[i+1].vx2_d,nnx,nny,nnz_device_append);
					exchange_device_nz<<<dimGridwf_append,dimBlock>>>(mgdevice[i].vy1_d,mgdevice[i].vy2_d,mgdevice[i+1].vy1_d,mgdevice[i+1].vy2_d,nnx,nny,nnz_device_append);
					exchange_device_nz<<<dimGridwf_append,dimBlock>>>(mgdevice[i].vz1_d,mgdevice[i].vz2_d,mgdevice[i+1].vz1_d,mgdevice[i+1].vz2_d,nnx,nny,nnz_device_append);
				}

				if(mark==1)
				{
					checkCudaErrors(cudaSetDevice(gpuid[0]));

					exchange_device_nz_one<<<dimGridwf_append,dimBlock>>>(mgdevice[i].vx1_d,mgdevice[i+1].vx1_d,nnx,nny,nnz_device_append);
					exchange_device_nz_one<<<dimGridwf_append,dimBlock>>>(mgdevice[i].vy1_d,mgdevice[i+1].vy1_d,nnx,nny,nnz_device_append);
					exchange_device_nz_one<<<dimGridwf_append,dimBlock>>>(mgdevice[i].vz1_d,mgdevice[i+1].vz1_d,nnx,nny,nnz_device_append);
				}

				if(mark==2)
				{				
					checkCudaErrors(cudaSetDevice(gpuid[i]));

					exchange_device_nz<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].vx1_d,mgdevice[i].vx2_d,mgdevice[i+1].vx1_d,mgdevice[i+1].vx2_d,nnx,nny,nnz_device_append);

					exchange_device_nz<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].vy1_d,mgdevice[i].vy2_d,mgdevice[i+1].vy1_d,mgdevice[i+1].vy2_d,nnx,nny,nnz_device_append);

					exchange_device_nz<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].vz1_d,mgdevice[i].vz2_d,mgdevice[i+1].vz1_d,mgdevice[i+1].vz2_d,nnx,nny,nnz_device_append);
				}

				if(mark==3)
				{	
					checkCudaErrors(cudaSetDevice(gpuid[i]));

					exchange_device_nz_one<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].vx1_d,mgdevice[i+1].vx1_d,nnx,nny,nnz_device_append);

					exchange_device_nz_one<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].vy1_d,mgdevice[i+1].vy1_d,nnx,nny,nnz_device_append);

					exchange_device_nz_one<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].vz1_d,mgdevice[i+1].vz1_d,nnx,nny,nnz_device_append);
				}

			}

			checkCudaErrors(cudaDeviceSynchronize());
}

void exchange_device_nz_kernel_vx_vy_vz2(GPUdevice *mgdevice,int mark)
///exchange_device_nz_kernel(mgdevice);
{

			dim3 dimBlock(32,16);

			dim3 dimGridwf_append((nnx+dimBlock.x-1)/dimBlock.x,(nny+dimBlock.y-1)/dimBlock.y,nnz_device_append);//////单块卡的整个空间

			for(int i=0;i<GPU_N-1;i++)
			{
				if(mark==0)
				{
					checkCudaErrors(cudaSetDevice(gpuid[0]));

					exchange_device_nz<<<dimGridwf_append,dimBlock>>>(mgdevice[i].vx1_d,mgdevice[i].vx2_d,mgdevice[i+1].vx1_d,mgdevice[i+1].vx2_d,nnx,nny,nnz_device_append);
					exchange_device_nz<<<dimGridwf_append,dimBlock>>>(mgdevice[i].vy1_d,mgdevice[i].vy2_d,mgdevice[i+1].vy1_d,mgdevice[i+1].vy2_d,nnx,nny,nnz_device_append);
					exchange_device_nz<<<dimGridwf_append,dimBlock>>>(mgdevice[i].vz1_d,mgdevice[i].vz2_d,mgdevice[i+1].vz1_d,mgdevice[i+1].vz2_d,nnx,nny,nnz_device_append);
				}

				if(mark==1)
				{
					checkCudaErrors(cudaSetDevice(gpuid[0]));

					exchange_device_nz_one<<<dimGridwf_append,dimBlock>>>(mgdevice[i].vx2_d,mgdevice[i+1].vx2_d,nnx,nny,nnz_device_append);
					exchange_device_nz_one<<<dimGridwf_append,dimBlock>>>(mgdevice[i].vy2_d,mgdevice[i+1].vy2_d,nnx,nny,nnz_device_append);
					exchange_device_nz_one<<<dimGridwf_append,dimBlock>>>(mgdevice[i].vz2_d,mgdevice[i+1].vz2_d,nnx,nny,nnz_device_append);
				}

				if(mark==2)
				{				
					checkCudaErrors(cudaSetDevice(gpuid[i]));

					exchange_device_nz<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].vx1_d,mgdevice[i].vx2_d,mgdevice[i+1].vx1_d,mgdevice[i+1].vx2_d,nnx,nny,nnz_device_append);

					exchange_device_nz<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].vy1_d,mgdevice[i].vy2_d,mgdevice[i+1].vy1_d,mgdevice[i+1].vy2_d,nnx,nny,nnz_device_append);

					exchange_device_nz<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].vz1_d,mgdevice[i].vz2_d,mgdevice[i+1].vz1_d,mgdevice[i+1].vz2_d,nnx,nny,nnz_device_append);
				}

				if(mark==3)
				{	
					checkCudaErrors(cudaSetDevice(gpuid[i]));

					exchange_device_nz_one<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].vx2_d,mgdevice[i+1].vx2_d,nnx,nny,nnz_device_append);

					exchange_device_nz_one<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].vy2_d,mgdevice[i+1].vy2_d,nnx,nny,nnz_device_append);

					exchange_device_nz_one<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].vz2_d,mgdevice[i+1].vz2_d,nnx,nny,nnz_device_append);
				}

			}

			checkCudaErrors(cudaDeviceSynchronize());
}



void exchange_device_nz_kernel_taop1(GPUdevice *mgdevice,int mark)
///exchange_device_nz_kernel(mgdevice);
{

			dim3 dimBlock(32,16);

			dim3 dimGridwf_append((nnx+dimBlock.x-1)/dimBlock.x,(nny+dimBlock.y-1)/dimBlock.y,nnz_device_append);//////单块卡的整个空间

			for(int i=0;i<GPU_N-1;i++)
			{
				if(mark==0)
				{
					checkCudaErrors(cudaSetDevice(gpuid[0]));

					exchange_device_nz<<<dimGridwf_append,dimBlock>>>(mgdevice[i].tp1_d,mgdevice[i].tp2_d,mgdevice[i+1].tp1_d,mgdevice[i+1].tp2_d,nnx,nny,nnz_device_append);
				}

				if(mark==1)
				{
					checkCudaErrors(cudaSetDevice(gpuid[0]));

					exchange_device_nz_one<<<dimGridwf_append,dimBlock>>>(mgdevice[i].tp1_d,mgdevice[i+1].tp1_d,nnx,nny,nnz_device_append);
				}

				if(mark==2)
				{				
					checkCudaErrors(cudaSetDevice(gpuid[i]));

					exchange_device_nz<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].tp1_d,mgdevice[i].tp2_d,mgdevice[i+1].tp1_d,mgdevice[i+1].tp2_d,nnx,nny,nnz_device_append);
				}

				if(mark==3)
				{	
					checkCudaErrors(cudaSetDevice(gpuid[i]));

					exchange_device_nz_one<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].tp1_d,mgdevice[i+1].tp1_d,nnx,nny,nnz_device_append);
				}

			}

			checkCudaErrors(cudaDeviceSynchronize());
}


void exchange_device_nz_kernel_taop2(GPUdevice *mgdevice,int mark)
///exchange_device_nz_kernel(mgdevice);
{

			dim3 dimBlock(32,16);

			dim3 dimGridwf_append((nnx+dimBlock.x-1)/dimBlock.x,(nny+dimBlock.y-1)/dimBlock.y,nnz_device_append);//////单块卡的整个空间

			for(int i=0;i<GPU_N-1;i++)
			{
				if(mark==0)
				{
					checkCudaErrors(cudaSetDevice(gpuid[0]));

					exchange_device_nz<<<dimGridwf_append,dimBlock>>>(mgdevice[i].tp1_d,mgdevice[i].tp2_d,mgdevice[i+1].tp1_d,mgdevice[i+1].tp2_d,nnx,nny,nnz_device_append);
				}

				if(mark==1)
				{
					checkCudaErrors(cudaSetDevice(gpuid[0]));

					exchange_device_nz_one<<<dimGridwf_append,dimBlock>>>(mgdevice[i].tp2_d,mgdevice[i+1].tp2_d,nnx,nny,nnz_device_append);
				}

				if(mark==2)
				{				
					checkCudaErrors(cudaSetDevice(gpuid[i]));

					exchange_device_nz<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].tp1_d,mgdevice[i].tp2_d,mgdevice[i+1].tp1_d,mgdevice[i+1].tp2_d,nnx,nny,nnz_device_append);
				}

				if(mark==3)
				{	
					checkCudaErrors(cudaSetDevice(gpuid[i]));

					exchange_device_nz_one<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].tp2_d,mgdevice[i+1].tp2_d,nnx,nny,nnz_device_append);
				}

			}

			checkCudaErrors(cudaDeviceSynchronize());
}













void test_exchange_device(GPUdevice *mgdevice)
{
	for(int i=0;i<GPU_N;i++)
		{
				checkCudaErrors(cudaSetDevice(gpuid[i]));
				
				for(int ix=0;ix<nnx;ix++)
					for(int iy=0;iy<nny;iy++)
						for(int iz=0;iz<nnz_device_append;iz++)
							mgdevice[i].wf_h[iz*nnx*nny+iy*nnx+ix]=1.0*i+1;

				checkCudaErrors(cudaMemcpy(mgdevice[i].vx1_d,mgdevice[i].wf_h,nnx*nny*nnz_device_append*sizeof(float),cudaMemcpyDefault));
				checkCudaErrors(cudaMemcpy(mgdevice[i].txx1_d,mgdevice[i].wf_h,nnx*nny*nnz_device_append*sizeof(float),cudaMemcpyDefault));
								
		}

		cudaDeviceSynchronize();
}



void transfer_gpu_to_cpu_multicomponent_seismic(GPUdevice *mgdevice,int it,int mark)
{
		
		if(mark==0)
		{
			checkCudaErrors(cudaSetDevice(gpuid[choose_re]));

			checkCudaErrors(cudaMemcpy(mgdevice[choose_re].obs_shot_x_h,mgdevice[choose_re].obs_shot_x_d,receiver_num_x*receiver_num_y*sizeof(float),cudaMemcpyDefault));
			checkCudaErrors(cudaMemcpy(mgdevice[choose_re].obs_shot_y_h,mgdevice[choose_re].obs_shot_y_d,receiver_num_x*receiver_num_y*sizeof(float),cudaMemcpyDefault));
			checkCudaErrors(cudaMemcpy(mgdevice[choose_re].obs_shot_z_h,mgdevice[choose_re].obs_shot_z_d,receiver_num_x*receiver_num_y*sizeof(float),cudaMemcpyDefault));

			cudaDeviceSynchronize();

			checkCudaErrors(cudaSetDevice(gpuid[choose_re]));


			for(int ix=0;ix<receiver_num_x;ix++)
				for(int iy=0;iy<receiver_num_y;iy++)
					{
						obs_shot_x_all[it][iy][ix]=mgdevice[choose_re].obs_shot_x_h[iy*receiver_num_x+ix];
						obs_shot_y_all[it][iy][ix]=mgdevice[choose_re].obs_shot_y_h[iy*receiver_num_x+ix];
						obs_shot_z_all[it][iy][ix]=mgdevice[choose_re].obs_shot_z_h[iy*receiver_num_x+ix];
					}

			cudaDeviceSynchronize();
		}

		else
		{
			checkCudaErrors(cudaSetDevice(gpuid[choose_re]));

			for(int ix=0;ix<receiver_num_x;ix++)
				for(int iy=0;iy<receiver_num_y;iy++)
					{
						mgdevice[choose_re].obs_shot_x_h[iy*receiver_num_x+ix]=obs_shot_x_all[it][iy][ix];
						mgdevice[choose_re].obs_shot_y_h[iy*receiver_num_x+ix]=obs_shot_y_all[it][iy][ix];
						mgdevice[choose_re].obs_shot_z_h[iy*receiver_num_x+ix]=obs_shot_z_all[it][iy][ix];
					}
			cudaDeviceSynchronize();

			checkCudaErrors(cudaSetDevice(gpuid[choose_re]));
		
			checkCudaErrors(cudaMemcpy(mgdevice[choose_re].obs_shot_x_d,mgdevice[choose_re].obs_shot_x_h,receiver_num_x*receiver_num_y*sizeof(float),cudaMemcpyDefault));
			checkCudaErrors(cudaMemcpy(mgdevice[choose_re].obs_shot_y_d,mgdevice[choose_re].obs_shot_y_h,receiver_num_x*receiver_num_y*sizeof(float),cudaMemcpyDefault));
			checkCudaErrors(cudaMemcpy(mgdevice[choose_re].obs_shot_z_d,mgdevice[choose_re].obs_shot_z_h,receiver_num_x*receiver_num_y*sizeof(float),cudaMemcpyDefault));

			cudaDeviceSynchronize();

		}
}

void output_or_input_multicomponent_seismic(int mark)
{
		//system("mkdir shotgather");
		if(mark==0)
		{
			sprintf(filename,"./shotgather/obs_x_shot_%d_%d",sy_real*int(dy),sx_real*int(dx));
			write_file_3d(obs_shot_x_all,receiver_num_x,receiver_num_y,lt,filename);

			sprintf(filename,"./shotgather/obs_y_shot_%d_%d",sy_real*int(dy),sx_real*int(dx));
			write_file_3d(obs_shot_y_all,receiver_num_x,receiver_num_y,lt,filename);

			sprintf(filename,"./shotgather/obs_z_shot_%d_%d",sy_real*int(dy),sx_real*int(dx));
			write_file_3d(obs_shot_z_all,receiver_num_x,receiver_num_y,lt,filename);
		}

		else
		{
			sprintf(filename,infile_shot_name);
			sprintf(filename1,"_x_shot_%d_%d",sy_real*int(dy),sx_real*int(dx));
			strcat(filename,filename1);
			fread_file_3d(obs_shot_x_all,receiver_num_x,receiver_num_y,lt,filename);


			sprintf(filename,infile_shot_name);
			sprintf(filename1,"_y_shot_%d_%d",sy_real*int(dy),sx_real*int(dx));
			strcat(filename,filename1);
			fread_file_3d(obs_shot_y_all,receiver_num_x,receiver_num_y,lt,filename);


			sprintf(filename,infile_shot_name);
			sprintf(filename1,"_z_shot_%d_%d",sy_real*int(dy),sx_real*int(dx));
			strcat(filename,filename1);
			fread_file_3d(obs_shot_z_all,receiver_num_x,receiver_num_y,lt,filename);
		}
}

void output_or_input_multicomponent_seismic_vsp(int mark)
{
		//system("mkdir shotgather");
		if(mark==0)
		{
			sprintf(filename,"./shotgather/obs_x_shot_%d",sz_real*int(dz));
			write_file_3d(obs_shot_x_all,receiver_num_x,receiver_num_y,lt,filename);

			sprintf(filename,"./shotgather/obs_y_shot_%d",sz_real*int(dz));
			write_file_3d(obs_shot_y_all,receiver_num_x,receiver_num_y,lt,filename);

			sprintf(filename,"./shotgather/obs_z_shot_%d",sz_real*int(dz));
			write_file_3d(obs_shot_z_all,receiver_num_x,receiver_num_y,lt,filename);
		}

		if(mark==3)
		{
			sprintf(filename,infile_shot_name);
			sprintf(filename1,"_x_shot_%d",sz_real*int(dz));
			strcat(filename,filename1);
			fread_file_3d(obs_shot_x_all,receiver_num_x,receiver_num_y,lt,filename);


			sprintf(filename,infile_shot_name);
			sprintf(filename1,"_y_shot_%d",sz_real*int(dz));
			strcat(filename,filename1);
			fread_file_3d(obs_shot_y_all,receiver_num_x,receiver_num_y,lt,filename);


			sprintf(filename,infile_shot_name);
			sprintf(filename1,"_z_shot_%d",sz_real*int(dz));
			strcat(filename,filename1);
			fread_file_3d(obs_shot_z_all,receiver_num_x,receiver_num_y,lt,filename);
		}

		if(mark==1)
		{
			set_zero_3d(obs_shot_x_all,receiver_num_x,receiver_num_y,lt);
			set_zero_3d(obs_shot_y_all,receiver_num_x,receiver_num_y,lt);
			set_zero_3d(obs_shot_z_all,receiver_num_x,receiver_num_y,lt);


			sprintf(filename,infile_shot_name);
			sprintf(filename1,"_%d",sz_real*int(dz));
			strcat(filename,filename1);
			fread_file_3d(obs_shot_z_all,receiver_num_x,receiver_num_y,lt,filename);
		}
}
//sprintf(filename1,"/nvresultppx-shot_%d",ishot);
//sprintf(filename,infile_shot_name);
//strcat(filename,filename1);
//write_file_1d(nvresultppx,nx*nz,filename);























//////////////2018年01月11日 星期四 20时32分14秒 3D elastic modeling
__global__ void fwd_vx_3D(float *vx2_d,float *vx1_d,float *txx1_d,float *txy1_d,float *txz1_d,float *velocity_d,float *velocity1_d,float *density_d,float *att_d,float *coe_d,int nx_pml,int ny_pml,int nz_pml,float dt,float coe_x,float coe_y,float coe_z)
//fwd_vx_3D<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].vx2_d,mgdevice[i].vx1_d,mgdevice[i].txx1_d,mgdevice[i].txy1_d,mgdevice[i].txz1_d,mgdevice[i].velocity_d,mgdevice[i].velocity1_d,mgdevice[i].density_d,mgdevice[i].att_d,mgdevice[i].coe_d,nnx,nny,nnz_device_append,dt,coe_x,coe_y,coe_z);
{
		float dt_real;
		dt_real=dt/1000;

		float s_att;
		//float vp,vs;
		float den;

		int tx=blockIdx.x*blockDim.x+threadIdx.x;
		int ty=blockIdx.y*blockDim.y+threadIdx.y;
		int tz=blockIdx.z;

		int idx;

		if((tx<nx_pml-2*radius)&&(ty<ny_pml-2*radius)&&(tz<nz_pml-2*radius))
		{
				tx=tx+radius;ty=ty+radius;tz=tz+radius;

				idx=tz*nx_pml*ny_pml+ty*nx_pml+tx;

					
				s_att=att_d[idx];
				//vp=velocity_d[idx];
				//vs=velocity1_d[idx];
				den=density_d[idx];


				float txx_x=			coe_d[1]*coe_x*(txx1_d[idx+1]-txx1_d[idx]);
								txx_x+=coe_d[2]*coe_x*(txx1_d[idx+2]-txx1_d[idx-1]);
								txx_x+=coe_d[3]*coe_x*(txx1_d[idx+3]-txx1_d[idx-2]);
								txx_x+=coe_d[4]*coe_x*(txx1_d[idx+4]-txx1_d[idx-3]);
								txx_x+=coe_d[5]*coe_x*(txx1_d[idx+5]-txx1_d[idx-4]);
								txx_x+=coe_d[6]*coe_x*(txx1_d[idx+6]-txx1_d[idx-5]);

				float txy_y=			coe_d[1]*coe_y*(txy1_d[idx]-txy1_d[idx-1*nx_pml]);
								txy_y+=coe_d[2]*coe_y*(txy1_d[idx+1*nx_pml]-txy1_d[idx-2*nx_pml]);
								txy_y+=coe_d[3]*coe_y*(txy1_d[idx+2*nx_pml]-txy1_d[idx-3*nx_pml]);
								txy_y+=coe_d[4]*coe_y*(txy1_d[idx+3*nx_pml]-txy1_d[idx-4*nx_pml]);
								txy_y+=coe_d[5]*coe_y*(txy1_d[idx+4*nx_pml]-txy1_d[idx-5*nx_pml]);
								txy_y+=coe_d[6]*coe_y*(txy1_d[idx+5*nx_pml]-txy1_d[idx-6*nx_pml]);

				float txz_z=			coe_d[1]*coe_z*(txz1_d[idx]-txz1_d[idx-1*nx_pml*ny_pml]);
								txz_z+=coe_d[2]*coe_z*(txz1_d[idx+1*nx_pml*ny_pml]-txz1_d[idx-2*nx_pml*ny_pml]);
								txz_z+=coe_d[3]*coe_z*(txz1_d[idx+2*nx_pml*ny_pml]-txz1_d[idx-3*nx_pml*ny_pml]);
							 	txz_z+=coe_d[4]*coe_z*(txz1_d[idx+3*nx_pml*ny_pml]-txz1_d[idx-4*nx_pml*ny_pml]);
							 	txz_z+=coe_d[5]*coe_z*(txz1_d[idx+4*nx_pml*ny_pml]-txz1_d[idx-5*nx_pml*ny_pml]);
							 	txz_z+=coe_d[6]*coe_z*(txz1_d[idx+5*nx_pml*ny_pml]-txz1_d[idx-6*nx_pml*ny_pml]);

				//if(den!=0)	vx2_d[idx]=1.0/(1.0+s_att*dt_real/2.0)*((1.0-s_att*dt_real/2.0)*vx1_d[idx]+1.0/den*(txx_x+txy_y+txz_z));

						vx2_d[idx]=1.0/(1.0+s_att*dt_real/2.0)*((1.0-s_att*dt_real/2.0)*vx1_d[idx]+1.0/den*(txx_x+txy_y+txz_z));

		}
}

__global__ void fwd_vy_3D(float *vy2_d,float *vy1_d,float *txy1_d,float *tyy1_d,float *tyz1_d,float *velocity_d,float *velocity1_d,float *density_d,float *att_d,float *coe_d,int nx_pml,int ny_pml,int nz_pml,float dt,float coe_x,float coe_y,float coe_z)
//fwd_vy_3D<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].vy2_d,mgdevice[i].vy1_d,mgdevice[i].txy1_d,mgdevice[i].tyy1_d,mgdevice[i].tyz1_d,mgdevice[i].velocity_d,mgdevice[i].velocity1_d,mgdevice[i].density_d,mgdevice[i].att_d,mgdevice[i].coe_d,nnx,nny,nnz_device_append,dt,coe_x,coe_y,coe_z);
{
		float dt_real;
		dt_real=dt/1000;

		float s_att;
		//float vp,vs;
		float den;

		int tx=blockIdx.x*blockDim.x+threadIdx.x;
		int ty=blockIdx.y*blockDim.y+threadIdx.y;
		int tz=blockIdx.z;

		int idx;

		if((tx<nx_pml-2*radius)&&(ty<ny_pml-2*radius)&&(tz<nz_pml-2*radius))
		{
				tx=tx+radius;ty=ty+radius;tz=tz+radius;

				idx=tz*nx_pml*ny_pml+ty*nx_pml+tx;

					
				s_att=att_d[idx];
				//vp=velocity_d[idx];
				//vs=velocity1_d[idx];
				den=density_d[idx];


				float txy_x=			coe_d[1]*coe_x*(txy1_d[idx]-txy1_d[idx-1]);
								txy_x+=coe_d[2]*coe_x*(txy1_d[idx+1]-txy1_d[idx-2]);
								txy_x+=coe_d[3]*coe_x*(txy1_d[idx+2]-txy1_d[idx-3]);
								txy_x+=coe_d[4]*coe_x*(txy1_d[idx+3]-txy1_d[idx-4]);
								txy_x+=coe_d[5]*coe_x*(txy1_d[idx+4]-txy1_d[idx-5]);
								txy_x+=coe_d[6]*coe_x*(txy1_d[idx+5]-txy1_d[idx-6]);

				float tyy_y=			coe_d[1]*coe_y*(tyy1_d[idx+1*nx_pml]-tyy1_d[idx]);
								tyy_y+=coe_d[2]*coe_y*(tyy1_d[idx+2*nx_pml]-tyy1_d[idx-1*nx_pml]);
								tyy_y+=coe_d[3]*coe_y*(tyy1_d[idx+3*nx_pml]-tyy1_d[idx-2*nx_pml]);
								tyy_y+=coe_d[4]*coe_y*(tyy1_d[idx+4*nx_pml]-tyy1_d[idx-3*nx_pml]);
								tyy_y+=coe_d[5]*coe_y*(tyy1_d[idx+5*nx_pml]-tyy1_d[idx-4*nx_pml]);
								tyy_y+=coe_d[6]*coe_y*(tyy1_d[idx+6*nx_pml]-tyy1_d[idx-5*nx_pml]);

				float tyz_z=			coe_d[1]*coe_z*(tyz1_d[idx]-tyz1_d[idx-1*nx_pml*ny_pml]);
								tyz_z+=coe_d[2]*coe_z*(tyz1_d[idx+1*nx_pml*ny_pml]-tyz1_d[idx-2*nx_pml*ny_pml]);
								tyz_z+=coe_d[3]*coe_z*(tyz1_d[idx+2*nx_pml*ny_pml]-tyz1_d[idx-3*nx_pml*ny_pml]);
							 	tyz_z+=coe_d[4]*coe_z*(tyz1_d[idx+3*nx_pml*ny_pml]-tyz1_d[idx-4*nx_pml*ny_pml]);
							 	tyz_z+=coe_d[5]*coe_z*(tyz1_d[idx+4*nx_pml*ny_pml]-tyz1_d[idx-5*nx_pml*ny_pml]);
							 	tyz_z+=coe_d[6]*coe_z*(tyz1_d[idx+5*nx_pml*ny_pml]-tyz1_d[idx-6*nx_pml*ny_pml]);

				//if(den!=0)	vy2_d[idx]=1.0/(1.0+s_att*dt_real/2.0)*((1.0-s_att*dt_real/2.0)*vy1_d[idx]+1.0/den*(txy_x+tyy_y+tyz_z));
			
						vy2_d[idx]=1.0/(1.0+s_att*dt_real/2.0)*((1.0-s_att*dt_real/2.0)*vy1_d[idx]+1.0/den*(txy_x+tyy_y+tyz_z));

		}
}

__global__ void fwd_vz_3D(float *vz2_d,float *vz1_d,float *txz1_d,float *tyz1_d,float *tzz1_d,float *velocity_d,float *velocity1_d,float *density_d,float *att_d,float *coe_d,int nx_pml,int ny_pml,int nz_pml,float dt,float coe_x,float coe_y,float coe_z)
//fwd_vz_3D<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].vz2_d,mgdevice[i].vz1_d,mgdevice[i].txz1_d,mgdevice[i].tyz1_d,mgdevice[i].tzz1_d,mgdevice[i].velocity_d,mgdevice[i].velocity1_d,mgdevice[i].density_d,mgdevice[i].att_d,mgdevice[i].coe_d,nnx,nny,nnz_device_append,dt,coe_x,coe_y,coe_z);
{
		float dt_real;
		dt_real=dt/1000;

		float s_att;
		//float vp,vs;
		float den;

		int tx=blockIdx.x*blockDim.x+threadIdx.x;
		int ty=blockIdx.y*blockDim.y+threadIdx.y;
		int tz=blockIdx.z;

		int idx;

		if((tx<nx_pml-2*radius)&&(ty<ny_pml-2*radius)&&(tz<nz_pml-2*radius))
		{
				tx=tx+radius;ty=ty+radius;tz=tz+radius;

				idx=tz*nx_pml*ny_pml+ty*nx_pml+tx;

					
				s_att=att_d[idx];
				//vp=velocity_d[idx];
				//vs=velocity1_d[idx];
				den=density_d[idx];


				float txz_x=			coe_d[1]*coe_x*(txz1_d[idx]-txz1_d[idx-1]);
								txz_x+=coe_d[2]*coe_x*(txz1_d[idx+1]-txz1_d[idx-2]);
								txz_x+=coe_d[3]*coe_x*(txz1_d[idx+2]-txz1_d[idx-3]);
								txz_x+=coe_d[4]*coe_x*(txz1_d[idx+3]-txz1_d[idx-4]);
								txz_x+=coe_d[5]*coe_x*(txz1_d[idx+4]-txz1_d[idx-5]);
								txz_x+=coe_d[6]*coe_x*(txz1_d[idx+5]-txz1_d[idx-6]);

				float tyz_y=			coe_d[1]*coe_y*(tyz1_d[idx]-tyz1_d[idx-1*nx_pml]);
								tyz_y+=coe_d[2]*coe_y*(tyz1_d[idx+1*nx_pml]-tyz1_d[idx-2*nx_pml]);
								tyz_y+=coe_d[3]*coe_y*(tyz1_d[idx+2*nx_pml]-tyz1_d[idx-3*nx_pml]);
								tyz_y+=coe_d[4]*coe_y*(tyz1_d[idx+3*nx_pml]-tyz1_d[idx-4*nx_pml]);
								tyz_y+=coe_d[5]*coe_y*(tyz1_d[idx+4*nx_pml]-tyz1_d[idx-5*nx_pml]);
								tyz_y+=coe_d[6]*coe_y*(tyz1_d[idx+5*nx_pml]-tyz1_d[idx-6*nx_pml]);

				float tzz_z=			coe_d[1]*coe_z*(tzz1_d[idx+1*nx_pml*ny_pml]-tzz1_d[idx]);
								tzz_z+=coe_d[2]*coe_z*(tzz1_d[idx+2*nx_pml*ny_pml]-tzz1_d[idx-1*nx_pml*ny_pml]);
								tzz_z+=coe_d[3]*coe_z*(tzz1_d[idx+3*nx_pml*ny_pml]-tzz1_d[idx-2*nx_pml*ny_pml]);
							 	tzz_z+=coe_d[4]*coe_z*(tzz1_d[idx+4*nx_pml*ny_pml]-tzz1_d[idx-3*nx_pml*ny_pml]);
							 	tzz_z+=coe_d[5]*coe_z*(tzz1_d[idx+5*nx_pml*ny_pml]-tzz1_d[idx-4*nx_pml*ny_pml]);
							 	tzz_z+=coe_d[6]*coe_z*(tzz1_d[idx+6*nx_pml*ny_pml]-tzz1_d[idx-5*nx_pml*ny_pml]);

				//if(den!=0)	vz2_d[idx]=1.0/(1.0+s_att*dt_real/2.0)*((1.0-s_att*dt_real/2.0)*vz1_d[idx]+1.0/den*(txz_x+tyz_y+tzz_z));

						vz2_d[idx]=1.0/(1.0+s_att*dt_real/2.0)*((1.0-s_att*dt_real/2.0)*vz1_d[idx]+1.0/den*(txz_x+tyz_y+tzz_z));

		}
}


__global__ void fwd_txxzzxzpp_3D(float *tp2_d,float *tp1_d,float *txx2_d,float *txx1_d,float *tyy2_d,float *tyy1_d,float *tzz2_d,float *tzz1_d,float *txy2_d,float *txy1_d,float *txz2_d,float *txz1_d,float *tyz2_d,float *tyz1_d,float *vx2_d,float *vy2_d,float *vz2_d,float *velocity_d,float *velocity1_d,float *density_d,float *att_d,float *coe_d,int nx_pml,int ny_pml,int nz_pml,float dt,float coe_x,float coe_y,float coe_z)
//fwd_txxzzxzpp_3D<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].tp2_d,mgdevice[i].tp1_d,mgdevice[i].txx2_d,mgdevice[i].txx1_d,mgdevice[i].tyy2_d,mgdevice[i].tyy1_d,mgdevice[i].tzz2_d,mgdevice[i].tzz1_d,mgdevice[i].txy2_d,mgdevice[i].txy1_d,mgdevice[i].txz2_d,mgdevice[i].txz1_d,mgdevice[i].tyz2_d,mgdevice[i].tyz1_d,mgdevice[i].vx2_d,mgdevice[i].vy2_d,mgdevice[i].vz2_d,mgdevice[i].velocity_d,mgdevice[i].velocity1_d,mgdevice[i].density_d,mgdevice[i].att_d,mgdevice[i].coe_d,nnx,nny,nnz_device_append,dt,coe_x,coe_y,coe_z);
{
		float dt_real;
		dt_real=dt/1000;

		float s_att;
		float vp,vs;
		float den;

		int tx=blockIdx.x*blockDim.x+threadIdx.x;
		int ty=blockIdx.y*blockDim.y+threadIdx.y;
		int tz=blockIdx.z;

		int idx;

		if((tx<nx_pml-2*radius)&&(ty<ny_pml-2*radius)&&(tz<nz_pml-2*radius))
		{
				tx=tx+radius;ty=ty+radius;tz=tz+radius;

				idx=tz*nx_pml*ny_pml+ty*nx_pml+tx;

					
				s_att=att_d[idx];
				vp=velocity_d[idx];
				vs=velocity1_d[idx];
				den=density_d[idx];


				float vx_x=			coe_d[1]*coe_x*(vx2_d[idx]-vx2_d[idx-1]);
								vx_x+=coe_d[2]*coe_x*(vx2_d[idx+1]-vx2_d[idx-2]);
								vx_x+=coe_d[3]*coe_x*(vx2_d[idx+2]-vx2_d[idx-3]);
								vx_x+=coe_d[4]*coe_x*(vx2_d[idx+3]-vx2_d[idx-4]);
								vx_x+=coe_d[5]*coe_x*(vx2_d[idx+4]-vx2_d[idx-5]);
								vx_x+=coe_d[6]*coe_x*(vx2_d[idx+5]-vx2_d[idx-6]);

				float vy_y=			coe_d[1]*coe_y*(vy2_d[idx]-vy2_d[idx-1*nx_pml]);
								vy_y+=coe_d[2]*coe_y*(vy2_d[idx+1*nx_pml]-vy2_d[idx-2*nx_pml]);
								vy_y+=coe_d[3]*coe_y*(vy2_d[idx+2*nx_pml]-vy2_d[idx-3*nx_pml]);
								vy_y+=coe_d[4]*coe_y*(vy2_d[idx+3*nx_pml]-vy2_d[idx-4*nx_pml]);
								vy_y+=coe_d[5]*coe_y*(vy2_d[idx+4*nx_pml]-vy2_d[idx-5*nx_pml]);
								vy_y+=coe_d[6]*coe_y*(vy2_d[idx+5*nx_pml]-vy2_d[idx-6*nx_pml]);

				float vz_z=			coe_d[1]*coe_z*(vz2_d[idx]-vz2_d[idx-1*nx_pml*ny_pml]);
								vz_z+=coe_d[2]*coe_z*(vz2_d[idx+1*nx_pml*ny_pml]-vz2_d[idx-2*nx_pml*ny_pml]);
								vz_z+=coe_d[3]*coe_z*(vz2_d[idx+2*nx_pml*ny_pml]-vz2_d[idx-3*nx_pml*ny_pml]);
							 	vz_z+=coe_d[4]*coe_z*(vz2_d[idx+3*nx_pml*ny_pml]-vz2_d[idx-4*nx_pml*ny_pml]);
							 	vz_z+=coe_d[5]*coe_z*(vz2_d[idx+4*nx_pml*ny_pml]-vz2_d[idx-5*nx_pml*ny_pml]);
							 	vz_z+=coe_d[6]*coe_z*(vz2_d[idx+5*nx_pml*ny_pml]-vz2_d[idx-6*nx_pml*ny_pml]);

				txx2_d[idx]=1.0/(1.0+s_att*dt_real/2.0)*((1.0-s_att*dt_real/2.0)*txx1_d[idx]+
									vp*vp*den*(vx_x+vy_y+vz_z)-2.0*vs*vs*den*(vy_y+vz_z));

				
				tyy2_d[idx]=1.0/(1.0+s_att*dt_real/2.0)*((1.0-s_att*dt_real/2.0)*tyy1_d[idx]+
									vp*vp*den*(vx_x+vy_y+vz_z)-2.0*vs*vs*den*(vx_x+vz_z));
				 

				tzz2_d[idx]=1.0/(1.0+s_att*dt_real/2.0)*((1.0-s_att*dt_real/2.0)*tzz1_d[idx]+
									vp*vp*den*(vx_x+vy_y+vz_z)-2.0*vs*vs*den*(vx_x+vy_y));


				tp2_d[idx]=1.0/(1.0+s_att*dt_real/2.0)*((1.0-s_att*dt_real/2.0)*tp1_d[idx]+vp*vp*den*(vx_x+vy_y+vz_z));


				float vx_y=			coe_d[1]*coe_y*(vx2_d[idx+1*nx_pml]-vx2_d[idx]);
								vx_y+=coe_d[2]*coe_y*(vx2_d[idx+2*nx_pml]-vx2_d[idx-1*nx_pml]);
								vx_y+=coe_d[3]*coe_y*(vx2_d[idx+3*nx_pml]-vx2_d[idx-2*nx_pml]);
								vx_y+=coe_d[4]*coe_y*(vx2_d[idx+4*nx_pml]-vx2_d[idx-3*nx_pml]);
								vx_y+=coe_d[5]*coe_y*(vx2_d[idx+5*nx_pml]-vx2_d[idx-4*nx_pml]);
								vx_y+=coe_d[6]*coe_y*(vx2_d[idx+6*nx_pml]-vx2_d[idx-5*nx_pml]);

				float vx_z=			coe_d[1]*coe_z*(vx2_d[idx+1*nx_pml*ny_pml]-vx2_d[idx]);
								vx_z+=coe_d[2]*coe_z*(vx2_d[idx+2*nx_pml*ny_pml]-vx2_d[idx-1*nx_pml*ny_pml]);
								vx_z+=coe_d[3]*coe_z*(vx2_d[idx+3*nx_pml*ny_pml]-vx2_d[idx-2*nx_pml*ny_pml]);
							 	vx_z+=coe_d[4]*coe_z*(vx2_d[idx+4*nx_pml*ny_pml]-vx2_d[idx-3*nx_pml*ny_pml]);
							 	vx_z+=coe_d[5]*coe_z*(vx2_d[idx+5*nx_pml*ny_pml]-vx2_d[idx-4*nx_pml*ny_pml]);
							 	vx_z+=coe_d[6]*coe_z*(vx2_d[idx+6*nx_pml*ny_pml]-vx2_d[idx-5*nx_pml*ny_pml]);;

				float vy_x=			coe_d[1]*coe_x*(vy2_d[idx+1]-vy2_d[idx]);
								vy_x+=coe_d[2]*coe_x*(vy2_d[idx+2]-vy2_d[idx-1]);
								vy_x+=coe_d[3]*coe_x*(vy2_d[idx+3]-vy2_d[idx-2]);
								vy_x+=coe_d[4]*coe_x*(vy2_d[idx+4]-vy2_d[idx-3]);
								vy_x+=coe_d[5]*coe_x*(vy2_d[idx+5]-vy2_d[idx-4]);
								vy_x+=coe_d[6]*coe_x*(vy2_d[idx+6]-vy2_d[idx-5]);

				float vy_z=			coe_d[1]*coe_z*(vy2_d[idx+1*nx_pml*ny_pml]-vy2_d[idx]);
								vy_z+=coe_d[2]*coe_z*(vy2_d[idx+2*nx_pml*ny_pml]-vy2_d[idx-1*nx_pml*ny_pml]);
								vy_z+=coe_d[3]*coe_z*(vy2_d[idx+3*nx_pml*ny_pml]-vy2_d[idx-2*nx_pml*ny_pml]);
							 	vy_z+=coe_d[4]*coe_z*(vy2_d[idx+4*nx_pml*ny_pml]-vy2_d[idx-3*nx_pml*ny_pml]);
							 	vy_z+=coe_d[5]*coe_z*(vy2_d[idx+5*nx_pml*ny_pml]-vy2_d[idx-4*nx_pml*ny_pml]);
							 	vy_z+=coe_d[6]*coe_z*(vy2_d[idx+6*nx_pml*ny_pml]-vy2_d[idx-5*nx_pml*ny_pml]);


				float vz_x=			coe_d[1]*coe_x*(vz2_d[idx+1]-vz2_d[idx]);
								vz_x+=coe_d[2]*coe_x*(vz2_d[idx+2]-vz2_d[idx-1]);
								vz_x+=coe_d[3]*coe_x*(vz2_d[idx+3]-vz2_d[idx-2]);
								vz_x+=coe_d[4]*coe_x*(vz2_d[idx+4]-vz2_d[idx-3]);
								vz_x+=coe_d[5]*coe_x*(vz2_d[idx+5]-vz2_d[idx-4]);
								vz_x+=coe_d[6]*coe_x*(vz2_d[idx+6]-vz2_d[idx-5]);

				float vz_y=			coe_d[1]*coe_y*(vz2_d[idx+1*nx_pml]-vz2_d[idx]);
								vz_y+=coe_d[2]*coe_y*(vz2_d[idx+2*nx_pml]-vz2_d[idx-1*nx_pml]);
								vz_y+=coe_d[3]*coe_y*(vz2_d[idx+3*nx_pml]-vz2_d[idx-2*nx_pml]);
								vz_y+=coe_d[4]*coe_y*(vz2_d[idx+4*nx_pml]-vz2_d[idx-3*nx_pml]);
								vz_y+=coe_d[5]*coe_y*(vz2_d[idx+5*nx_pml]-vz2_d[idx-4*nx_pml]);
								vz_y+=coe_d[6]*coe_y*(vz2_d[idx+6*nx_pml]-vz2_d[idx-5*nx_pml]);


					txy2_d[idx]=1.0/(1.0+s_att*dt_real/2.0)*((1.0-s_att*dt_real/2.0)*txy1_d[idx]+vs*vs*den*(vx_y+vy_x));

					txz2_d[idx]=1.0/(1.0+s_att*dt_real/2.0)*((1.0-s_att*dt_real/2.0)*txz1_d[idx]+vs*vs*den*(vx_z+vz_x));
										
					tyz2_d[idx]=1.0/(1.0+s_att*dt_real/2.0)*((1.0-s_att*dt_real/2.0)*tyz1_d[idx]+vs*vs*den*(vy_z+vz_y));	

		}
}
__global__ void fwd_vxp_vzp_3D(float *vxp2_d,float *vxp1_d,float *vyp2_d,float *vyp1_d,float *vzp2_d,float *vzp1_d,float *tp2_d,float *vxs2_d,float *vys2_d,float *vzs2_d,float *vx2_d,float *vy2_d,float *vz2_d,float *velocity_d,float *velocity1_d,float *density_d,float *att_d,float *coe_d,int nx_pml,int ny_pml,int nz_pml,float dt,float coe_x,float coe_y,float coe_z)
//fwd_vxp_vzp_3D<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].vxp2_d,mgdevice[i].vxp1_d,mgdevice[i].vyp2_d,mgdevice[i].vyp1_d,mgdevice[i].vzp2_d,mgdevice[i].vzp1_d,mgdevice[i].tp2_d,mgdevice[i].vxs2_d,mgdevice[i].vys2_d,mgdevice[i].vzs2_d,mgdevice[i].vx2_d,mgdevice[i].vy2_d,mgdevice[i].vz2_d,mgdevice[i].velocity_d,mgdevice[i].velocity1_d,mgdevice[i].density_d,mgdevice[i].att_d,mgdevice[i].coe_d,nnx,nny,nnz_device_append,dt,coe_x,coe_y,coe_z);	
{
		float dt_real;
		dt_real=dt/1000;

		float s_att;
		//float vp,vs;
		float den;

		int tx=blockIdx.x*blockDim.x+threadIdx.x;
		int ty=blockIdx.y*blockDim.y+threadIdx.y;
		int tz=blockIdx.z;

		int idx;

		if((tx<nx_pml-2*radius)&&(ty<ny_pml-2*radius)&&(tz<nz_pml-2*radius))
		{
				tx=tx+radius;ty=ty+radius;tz=tz+radius;

				idx=tz*nx_pml*ny_pml+ty*nx_pml+tx;

					
				s_att=att_d[idx];
				//vp=velocity_d[idx];
				//vs=velocity1_d[idx];
				den=density_d[idx];


				float tp_x=			coe_d[1]*coe_x*(tp2_d[idx+1]-tp2_d[idx]);
								tp_x+=coe_d[2]*coe_x*(tp2_d[idx+2]-tp2_d[idx-1]);
								tp_x+=coe_d[3]*coe_x*(tp2_d[idx+3]-tp2_d[idx-2]);
								tp_x+=coe_d[4]*coe_x*(tp2_d[idx+4]-tp2_d[idx-3]);
								tp_x+=coe_d[5]*coe_x*(tp2_d[idx+5]-tp2_d[idx-4]);
								tp_x+=coe_d[6]*coe_x*(tp2_d[idx+6]-tp2_d[idx-5]);

				float tp_y=			coe_d[1]*coe_y*(tp2_d[idx+1*nx_pml]-tp2_d[idx]);
								tp_y+=coe_d[2]*coe_y*(tp2_d[idx+2*nx_pml]-tp2_d[idx-1*nx_pml]);
								tp_y+=coe_d[3]*coe_y*(tp2_d[idx+3*nx_pml]-tp2_d[idx-2*nx_pml]);
								tp_y+=coe_d[4]*coe_y*(tp2_d[idx+4*nx_pml]-tp2_d[idx-3*nx_pml]);
								tp_y+=coe_d[5]*coe_y*(tp2_d[idx+5*nx_pml]-tp2_d[idx-4*nx_pml]);
								tp_y+=coe_d[6]*coe_y*(tp2_d[idx+6*nx_pml]-tp2_d[idx-5*nx_pml]);

				float tp_z=			coe_d[1]*coe_z*(tp2_d[idx+1*nx_pml*ny_pml]-tp2_d[idx]);
								tp_z+=coe_d[2]*coe_z*(tp2_d[idx+2*nx_pml*ny_pml]-tp2_d[idx-1*nx_pml*ny_pml]);
								tp_z+=coe_d[3]*coe_z*(tp2_d[idx+3*nx_pml*ny_pml]-tp2_d[idx-2*nx_pml*ny_pml]);
							 	tp_z+=coe_d[4]*coe_z*(tp2_d[idx+4*nx_pml*ny_pml]-tp2_d[idx-3*nx_pml*ny_pml]);
							 	tp_z+=coe_d[5]*coe_z*(tp2_d[idx+5*nx_pml*ny_pml]-tp2_d[idx-4*nx_pml*ny_pml]);
							 	tp_z+=coe_d[6]*coe_z*(tp2_d[idx+6*nx_pml*ny_pml]-tp2_d[idx-5*nx_pml*ny_pml]);


				vxp2_d[idx]=1.0/(1.0+s_att*dt_real/2.0)*((1.0-s_att*dt_real/2.0)*vxp1_d[idx]+1.0/den*tp_x);

				vyp2_d[idx]=1.0/(1.0+s_att*dt_real/2.0)*((1.0-s_att*dt_real/2.0)*vyp1_d[idx]+1.0/den*tp_y);

				vzp2_d[idx]=1.0/(1.0+s_att*dt_real/2.0)*((1.0-s_att*dt_real/2.0)*vzp1_d[idx]+1.0/den*tp_z);

				vxs2_d[idx]=vx2_d[idx]-vxp2_d[idx];

				vys2_d[idx]=vy2_d[idx]-vyp2_d[idx];

				vzs2_d[idx]=vz2_d[idx]-vzp2_d[idx];

		}
}	
__global__ void vp_vs_3D(float *vx2_d,float *vy2_d,float *vz2_d,float *vxp2_d,float *vyp2_d,float *vzp2_d,float *vxs2_d,float *vys2_d,float *vzs2_d,int nx_pml,int ny_pml,int nz_pml)
{
		int tx=blockIdx.x*blockDim.x+threadIdx.x;
		int ty=blockIdx.y*blockDim.y+threadIdx.y;
		int tz=blockIdx.z;

		int idx;

		if((tx<nx_pml-2*radius)&&(ty<ny_pml-2*radius)&&(tz<nz_pml-2*radius))
		{
				tx=tx+radius;ty=ty+radius;tz=tz+radius;

				idx=tz*nx_pml*ny_pml+ty*nx_pml+tx;

				vxs2_d[idx]=vx2_d[idx]-vxp2_d[idx];
				vys2_d[idx]=vy2_d[idx]-vyp2_d[idx];
				vzs2_d[idx]=vz2_d[idx]-vzp2_d[idx];
		}

}


__global__ void write_or_add_shot_3D_surface(float *obs_shot_x_d,float *vx2_d,int nnx,int nny,int nnz_device_append,int nnz_device,int bl,int bb,int bu,int receiver_start_x,int receiver_num_x,int receiver_interval_x,int receiver_start_y,int receiver_num_y,int receiver_interval_y,int receiver_start_z,int receiver_num_z,int receiver_interval_z,int mark)
//write_or_add_shot_3D_surface(obs_shot_x_d,vx2_d,nnx,nny,nnz_device_append,nnz_device,bl,bb,bu,receiver_start_x,receiver_num_x,receiver_interval_x,receiver_start_y,receiver_num_y,receiver_interval_y,receiver_start_z,receiver_num_z,receiver_interval_z,0);
{
		int ix=blockIdx.x*blockDim.x+threadIdx.x;
		int iy=blockIdx.y*blockDim.y+threadIdx.y;

		int id,id1;

		int remain=(receiver_start_z+bu)%nnz_device+radius;

		if(ix<receiver_num_x&&iy<receiver_num_y)
		{
			id=iy*receiver_num_x+ix;

			id1=remain*nnx*nny+(receiver_start_y+iy*receiver_interval_y+bb)*nnx+(receiver_start_x+ix*receiver_interval_x+bl);

			if(mark==0)	obs_shot_x_d[id]=vx2_d[id1];

			if(mark==1)	vx2_d[id1]+=obs_shot_x_d[id];
		}
}

__global__ void write_or_add_shot_3D_surface_three(float *obs_shot_x_d,float *obs_shot_y_d,float *obs_shot_z_d,float *vx2_d,float *vy2_d,float *vz2_d,int nnx,int nny,int nnz_device_append,int nnz_device,int bl,int bb,int bu,int receiver_start_x,int receiver_num_x,int receiver_interval_x,int receiver_start_y,int receiver_num_y,int receiver_interval_y,int receiver_start_z,int receiver_num_z,int receiver_interval_z,int mark)
//write_or_add_shot_3D_surface_three<<<dimGrid_rec_lt_x_y,dimBlock,0,mgdevice[choose_re].stream>>>(mgdevice[choose_re].obs_shot_x_d,mgdevice[choose_re].obs_shot_y_d,mgdevice[choose_re].obs_shot_z_d,mgdevice[choose_re].vx2_d,mgdevice[choose_re].vy2_d,mgdevice[choose_re].vz2_d,nnx,nny,nnz_device_append,nnz_device,bl,bb,bu,receiver_start_x,receiver_num_x,receiver_interval_x,receiver_start_y,receiver_num_y,receiver_interval_y,receiver_start_z,receiver_num_z,receiver_interval_z,0);	
{
		int ix=blockIdx.x*blockDim.x+threadIdx.x;
		int iy=blockIdx.y*blockDim.y+threadIdx.y;

		int id,id1;

		int remain=(receiver_start_z+bu)%nnz_device+radius;

		if(ix<receiver_num_x&&iy<receiver_num_y)
		{
			id=iy*receiver_num_x+ix;

			id1=remain*nnx*nny+(receiver_start_y+iy*receiver_interval_y+bb)*nnx+(receiver_start_x+ix*receiver_interval_x+bl);

			if(mark==0)	
			{
				obs_shot_x_d[id]=vx2_d[id1];
				obs_shot_y_d[id]=vy2_d[id1];
				obs_shot_z_d[id]=vz2_d[id1];
			}

			if(mark==1)	
			{
				vx2_d[id1]=obs_shot_x_d[id];
				vy2_d[id1]=obs_shot_y_d[id];
				vz2_d[id1]=obs_shot_z_d[id];
			}

			if(mark==2)	
			{
				vx2_d[id1]+=obs_shot_x_d[id];
				vy2_d[id1]+=obs_shot_y_d[id];
				vz2_d[id1]+=obs_shot_z_d[id];
			}
		}
}


__global__ void cut_direct_shot_3D_surface_three(float *obs_shot_x_d,float *obs_shot_y_d,float *obs_shot_z_d,float *velocity_d,int nnx,int nny,int nnz_device,int bl,int bb,int bu,int receiver_start_x,int receiver_num_x,int receiver_interval_x,int receiver_start_y,int receiver_num_y,int receiver_interval_y,int receiver_start_z,int receiver_num_z,int receiver_interval_z,int sx,int sy,int sz,int it,int wavelet_length,float dx,float dy,float dz,float dt)
//cut_direct_shot_3D_surface<<<dimGrid_rec_lt_x_y,dimBlock,0,mgdevice[choose_ns].stream>>>(mgdevice[choose_re].obs_shot_x_d,mgdevice[choose_re].velocity_d,nnx,nny,nnz_device,bl,bb,bu,receiver_start_x,receiver_num_x,receiver_interval_x,receiver_start_y,receiver_num_y,receiver_interval_y,receiver_start_z,receiver_num_z,receiver_interval_z,sx_real,sy_real,sz_real,it,wavelet_length,dx,dy,dz,dt);
{

		int ix=blockIdx.x*blockDim.x+threadIdx.x;
		int iy=blockIdx.y*blockDim.y+threadIdx.y;

		int id,id1;

		int sx_real=sx+bl;
	
		int sy_real=sy+bb;

		int sz_real=(sz+bu)%nnz_device+radius;

		int rx=receiver_start_x+ix*receiver_interval_x;
		int ry=receiver_start_y+iy*receiver_interval_y;
		int rz=receiver_start_z;

		float distance;

		float d_x=(sx-rx)*dx;
		float d_y=(sy-ry)*dy;
		float d_z=(sz-rz)*dz;
	
		
		distance=sqrt(d_x*d_x*1.0+d_y*d_y*1.0+d_z*d_z*1.0);

		int time;

		if(ix<receiver_num_x&&iy<receiver_num_y)
		{
			id=iy*receiver_num_x+ix;

			id1=sz_real*(nnx*nny)+sy_real*nnx+sx_real;

			time=(distance/velocity_d[id1])*1000/dt;

			if(it<time+wavelet_length+50)	
			{
				obs_shot_x_d[id]=0.0;
				obs_shot_y_d[id]=0.0;
				obs_shot_z_d[id]=0.0;
			}
		}
}

__global__ void cut_direct_shot_3D_surface_three_new(float *obs_shot_x_d,float *obs_shot_y_d,float *obs_shot_z_d,float *velocity_d,int nnx,int nny,int nnz_device,int bl,int bb,int bu,int receiver_start_x,int receiver_num_x,int receiver_interval_x,int receiver_start_y,int receiver_num_y,int receiver_interval_y,int receiver_start_z,int receiver_num_z,int receiver_interval_z,int sx,int sy,int sz,int it,int wavelet_length,float dx,float dy,float dz,float dt,int cut_direct_wave)
//cut_direct_shot_3D_surface<<<dimGrid_rec_lt_x_y,dimBlock,0,mgdevice[choose_ns].stream>>>(mgdevice[choose_re].obs_shot_x_d,mgdevice[choose_re].velocity_d,nnx,nny,nnz_device,bl,bb,bu,receiver_start_x,receiver_num_x,receiver_interval_x,receiver_start_y,receiver_num_y,receiver_interval_y,receiver_start_z,receiver_num_z,receiver_interval_z,sx_real,sy_real,sz_real,it,wavelet_length,dx,dy,dz,dt);
{

		int ix=blockIdx.x*blockDim.x+threadIdx.x;
		int iy=blockIdx.y*blockDim.y+threadIdx.y;

		int id,id1;

		int sx_real=sx+bl;
	
		int sy_real=sy+bb;

		int sz_real=(sz+bu)%nnz_device+radius;

		int rx=receiver_start_x+ix*receiver_interval_x;
		int ry=receiver_start_y+iy*receiver_interval_y;
		int rz=receiver_start_z;

		float distance;

		float d_x=(sx-rx)*dx;
		float d_y=(sy-ry)*dy;
		float d_z=(sz-rz)*dz;
	
		
		distance=sqrt(d_x*d_x*1.0+d_y*d_y*1.0+d_z*d_z*1.0);

		int time;

		if(ix<receiver_num_x&&iy<receiver_num_y)
		{
			id=iy*receiver_num_x+ix;

			id1=sz_real*(nnx*nny)+sy_real*nnx+sx_real;

			time=(distance/velocity_d[id1])*1000/dt;

			if(it<time+cut_direct_wave)	
			{
				obs_shot_x_d[id]=0.0;
				obs_shot_y_d[id]=0.0;
				obs_shot_z_d[id]=0.0;
			}
		}
}
















/////////////////////////For Elastic RTM2018年01月17日 星期三 16时29分25秒
__global__ void cuda_cal_excitation_amp_time(float *ex_time_d,float *ex_tp_d,float *tp2_d,float *ex_vxp_d,float *vxp2_d,float *ex_vyp_d,float *vyp2_d,float *ex_vzp_d,float *vzp2_d,int nx_pml,int ny_pml,int nz_pml,int it)
//cuda_cal_excitation_amp_time<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].ex_time_d,mgdevice[i].ex_tp_d,mgdevice[i].tp2_d,mgdevice[i].ex_vxp_d,mgdevice[i].vxp2_d,mgdevice[i].ex_vyp_d,mgdevice[i].vyp2_d,mgdevice[i].ex_vzp_d,mgdevice[i].vzp2_d,nnx,nny,nnz_device_append,it);
{
		int tx=blockIdx.x*blockDim.x+threadIdx.x;
		int ty=blockIdx.y*blockDim.y+threadIdx.y;
		int tz=blockIdx.z;

		int idx;

		if((tx<nx_pml-2*radius)&&(ty<ny_pml-2*radius)&&(tz<nz_pml-2*radius))
		{
				tx=tx+radius;ty=ty+radius;tz=tz+radius;

				idx=tz*nx_pml*ny_pml+ty*nx_pml+tx;

				if(fabs(tp2_d[idx])>fabs(ex_tp_d[idx]))
				{
					ex_time_d[idx]=it;

					ex_tp_d[idx]=tp2_d[idx];

					ex_vxp_d[idx]=vxp2_d[idx];

					ex_vyp_d[idx]=vyp2_d[idx];

					ex_vzp_d[idx]=vzp2_d[idx];
				}
		}
}


__global__ void cuda_cal_excitation_amp_time_new(float *ex_time_d,float *ex_amp_d,float *ex_tp_d,float *tp2_d,float *ex_vxp_d,float *vxp2_d,float *ex_vyp_d,float *vyp2_d,float *ex_vzp_d,float *vzp2_d,int nx_pml,int ny_pml,int nz_pml,int it)
//cuda_cal_excitation_amp_time_new<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].ex_time_d,mgdevice[i].ex_amp_d,mgdevice[i].ex_tp_d,mgdevice[i].tp2_d,mgdevice[i].ex_vxp_d,mgdevice[i].vxp2_d,mgdevice[i].ex_vyp_d,mgdevice[i].vyp2_d,mgdevice[i].ex_vzp_d,mgdevice[i].vzp2_d,nnx,nny,nnz_device_append,it);
{
		int tx=blockIdx.x*blockDim.x+threadIdx.x;
		int ty=blockIdx.y*blockDim.y+threadIdx.y;
		int tz=blockIdx.z;

		int idx;
		float change;

		if((tx<nx_pml-2*radius)&&(ty<ny_pml-2*radius)&&(tz<nz_pml-2*radius))
		{
				tx=tx+radius;ty=ty+radius;tz=tz+radius;

				idx=tz*nx_pml*ny_pml+ty*nx_pml+tx;
			
				change=sqrt(vxp2_d[idx]*vxp2_d[idx]+vyp2_d[idx]*vyp2_d[idx]+vzp2_d[idx]*vzp2_d[idx]);

				if(ex_amp_d[idx]<change)
				{
					ex_time_d[idx]=it;
					
					ex_amp_d[idx]=change;

					ex_tp_d[idx]=tp2_d[idx];

					ex_vxp_d[idx]=vxp2_d[idx];

					ex_vyp_d[idx]=vyp2_d[idx];

					ex_vzp_d[idx]=vzp2_d[idx];
				}
		}
}

__global__ void cuda_cal_source_poyn_3D(float *poyn_px_d,float *poyn_py_d,float *poyn_pz_d,float *ex_time_d,float *vxp2_d,float *vyp2_d,float *vzp2_d,float *tp2_d,int nx_pml,int ny_pml,int nz_pml,int it)
//cuda_cal_source_poyn_3D<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].poyn_px_d,mgdevice[i].poyn_py_d,mgdevice[i].poyn_pz_d,mgdevice[i].ex_time_d,mgdevice[i].vxp2_d,mgdevice[i].vyp2_d,mgdevice[i].vzp2_d,mgdevice[i].tp2_d,nnx,nny,nnz_device_append,it);
{
		int tx=blockIdx.x*blockDim.x+threadIdx.x;
		int ty=blockIdx.y*blockDim.y+threadIdx.y;
		int tz=blockIdx.z;

		int idx;

		if((tx<nx_pml-2*radius)&&(ty<ny_pml-2*radius)&&(tz<nz_pml-2*radius))
		{
				tx=tx+radius;ty=ty+radius;tz=tz+radius;

				idx=tz*nx_pml*ny_pml+ty*nx_pml+tx;
			
				if(ex_time_d[idx]==it)
				{
					poyn_px_d[idx]=-1.0*vxp2_d[idx]*tp2_d[idx];

					poyn_py_d[idx]=-1.0*vyp2_d[idx]*tp2_d[idx];

					poyn_pz_d[idx]=-1.0*vzp2_d[idx]*tp2_d[idx];
				}
		}
}
__global__ void cuda_cal_receiver_poyn_3D(float *poyn_px_d,float *poyn_py_d,float *poyn_pz_d,float *poyn_sx_d,float *poyn_sy_d,float *poyn_sz_d,float *ex_time_d,float *vxp2_d,float *vyp2_d,float *vzp2_d,float *vxs2_d,float *vys2_d,float *vzs2_d,float *tp2_d,float *txx2_d,float *tyy2_d,float *tzz2_d,float *txy2_d,float *txz2_d,float *tyz2_d,int nx_pml,int ny_pml,int nz_pml,int it)
//cuda_cal_receiver_poyn_3D<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].poyn_rpx_d,mgdevice[i].poyn_rpy_d,mgdevice[i].poyn_rpz_d,mgdevice[i].poyn_rsx_d,mgdevice[i].poyn_rsy_d,mgdevice[i].poyn_rsz_d,mgdevice[i].ex_time_d,mgdevice[i].vxp2_d,mgdevice[i].vyp2_d,mgdevice[i].vzp2_d,mgdevice[i].vxs2_d,mgdevice[i].vys2_d,mgdevice[i].vzs2_d,mgdevice[i].tp2_d,mgdevice[i].txx2_d,mgdevice[i].tyy2_d,mgdevice[i].tzz2_d,mgdevice[i].txy2_d,mgdevice[i].txz2_d,mgdevice[i].tyz2_d,nnx,nny,nnz_device_append,it);
{
		int tx=blockIdx.x*blockDim.x+threadIdx.x;
		int ty=blockIdx.y*blockDim.y+threadIdx.y;
		int tz=blockIdx.z;

		int idx;

		if((tx<nx_pml-2*radius)&&(ty<ny_pml-2*radius)&&(tz<nz_pml-2*radius))
		{
				tx=tx+radius;ty=ty+radius;tz=tz+radius;

				idx=tz*nx_pml*ny_pml+ty*nx_pml+tx;
			
				if(ex_time_d[idx]==it)
				{
					poyn_px_d[idx]=-1.0*vxp2_d[idx]*tp2_d[idx];

					poyn_py_d[idx]=-1.0*vyp2_d[idx]*tp2_d[idx];

					poyn_pz_d[idx]=-1.0*vzp2_d[idx]*tp2_d[idx];


					poyn_sx_d[idx]=-1.0*((txx2_d[idx]-tp2_d[idx])*vxs2_d[idx]+txy2_d[idx]*vys2_d[idx]+txz2_d[idx]*vzs2_d[idx]);

					poyn_sy_d[idx]=-1.0*(txy2_d[idx]*vxs2_d[idx]+(tyy2_d[idx]-tp2_d[idx])*vys2_d[idx]+tyz2_d[idx]*vzs2_d[idx]);

					poyn_sz_d[idx]=-1.0*(txz2_d[idx]*vxs2_d[idx]+tyz2_d[idx]*vys2_d[idx]+(tzz2_d[idx]-tp2_d[idx])*vzs2_d[idx]);
				}
		}
}

__global__ void cuda_cal_angle_3D(float *angle_pp_d,float *angle_ps_d,float *poyn_px_d,float *poyn_py_d,float *poyn_pz_d,float *poyn_rpx_d,float *poyn_rpy_d,float *poyn_rpz_d,float *poyn_rsx_d,float *poyn_rsy_d,float *poyn_rsz_d,int nx_pml,int ny_pml,int nz_pml)
//cuda_cal_angle_3D<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].angle_pp_d,mgdevice[i].angle_ps_d,mgdevice[i].poyn_px_d,mgdevice[i].poyn_py_d,mgdevice[i].poyn_pz_d,mgdevice[i].poyn_rpx_d,mgdevice[i].poyn_rpy_d,mgdevice[i].poyn_rpz_d,mgdevice[i].poyn_rsx_d,mgdevice[i].poyn_rsy_d,mgdevice[i].poyn_rsz_d,nnx,nny,nnz_device_append);
{
		int tx=blockIdx.x*blockDim.x+threadIdx.x;
		int ty=blockIdx.y*blockDim.y+threadIdx.y;
		int tz=blockIdx.z;

		float magntiude1;
		float magntiude2;
		float magntiude3;

		float radian_pp;
		float radian_ps;

		int idx;

		if((tx<nx_pml-2*radius)&&(ty<ny_pml-2*radius)&&(tz<nz_pml-2*radius))
		{
				tx=tx+radius;ty=ty+radius;tz=tz+radius;

				idx=tz*nx_pml*ny_pml+ty*nx_pml+tx;

				magntiude1=float(sqrt(1.0*poyn_px_d[idx]*poyn_px_d[idx]+1.0*poyn_py_d[idx]*poyn_py_d[idx]+1.0*poyn_py_d[idx]*poyn_py_d[idx]));
				
				magntiude2=float(sqrt(1.0*poyn_rpx_d[idx]*poyn_rpx_d[idx]+1.0*poyn_rpy_d[idx]*poyn_rpy_d[idx]+1.0*poyn_rpy_d[idx]*poyn_rpy_d[idx]));

				magntiude3=float(sqrt(1.0*poyn_rsx_d[idx]*poyn_rsx_d[idx]+1.0*poyn_rsy_d[idx]*poyn_rsy_d[idx]+1.0*poyn_rsy_d[idx]*poyn_rsy_d[idx]));
				
				if(magntiude1!=0&&magntiude2!=0)
						radian_pp=1.0*(poyn_px_d[idx]*poyn_rpx_d[idx]+poyn_py_d[idx]*poyn_rpy_d[idx]+poyn_pz_d[idx]*poyn_rpz_d[idx])/magntiude1/magntiude2*1.0;

				if(magntiude1!=0&&magntiude3!=0)
						radian_ps=1.0*(poyn_px_d[idx]*poyn_rsx_d[idx]+poyn_py_d[idx]*poyn_rsy_d[idx]+poyn_pz_d[idx]*poyn_rsz_d[idx])/magntiude1/magntiude3*1.0;

				angle_pp_d[idx]=acos(1.0*radian_pp);
				angle_ps_d[idx]=acos(1.0*radian_ps);
		}
}

__global__ void imaging_correlation_ex(float *ex_time_d,float *ex_tp_d,float *tp2_d,float *vresult_tp_d,int nx_pml,int ny_pml,int nz_pml,int it,float max,int precon_z1)
//imaging_correlation_ex<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].ex_time_d,mgdevice[i].ex_tp_d,mgdevice[i].tp2_d,mgdevice[i].vresult_tp_d,nnx,nny,nnz_device_append,it);
{
		int tx=blockIdx.x*blockDim.x+threadIdx.x;
		int ty=blockIdx.y*blockDim.y+threadIdx.y;
		int tz=blockIdx.z;

		int idx;

		if((tx<nx_pml-2*radius)&&(ty<ny_pml-2*radius)&&(tz<nz_pml-2*radius))
		{
				tx=tx+radius;ty=ty+radius;tz=tz+radius;

				idx=tz*nx_pml*ny_pml+ty*nx_pml+tx;

				if(it==ex_time_d[idx])
				{
					//vresult_tp_d[idx]=tp2_d[idx]*1.0/(ex_tp_d[idx]+0.00001*max);
					
					vresult_tp_d[idx]=tp2_d[idx]*1.0/(ex_tp_d[idx]);
				}
		}
}

__global__ void imaging_inner_product_ex(float *ex_time_d,float *ex_vxp_d,float *ex_vyp_d,float *ex_vzp_d,float *vxp2_d,float *vyp2_d,float *vzp2_d,float *vresult_pp_d,int nx_pml,int ny_pml,int nz_pml,int it,float max,int precon_z1)
//imaging_inner_product_ex<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].ex_time_d,mgdevice[i].ex_vxp_d,mgdevice[i].ex_vyp_d,mgdevice[i].ex_vzp_d,mgdevice[i].vxp2_d,mgdevice[i].vyp2_d,mgdevice[i].vzp2_d,mgdevice[i].vresult_pp_d,nnx,nny,nnz_device_append,it);
{
		int tx=blockIdx.x*blockDim.x+threadIdx.x;
		int ty=blockIdx.y*blockDim.y+threadIdx.y;
		int tz=blockIdx.z;

		int idx;

		if((tx<nx_pml-2*radius)&&(ty<ny_pml-2*radius)&&(tz<nz_pml-2*radius))
		{
				tx=tx+radius;ty=ty+radius;tz=tz+radius;

				idx=tz*nx_pml*ny_pml+ty*nx_pml+tx;

				if(it==ex_time_d[idx])
				{
					float molecular=ex_vxp_d[idx]*vxp2_d[idx]+ex_vyp_d[idx]*vyp2_d[idx]+ex_vzp_d[idx]*vzp2_d[idx];

					float denominator=ex_vxp_d[idx]*ex_vxp_d[idx]+ex_vyp_d[idx]*ex_vyp_d[idx]+ex_vzp_d[idx]*ex_vzp_d[idx];

					//vresult_pp_d[idx]=molecular*1.0/(denominator+0.00001*max*max);

					vresult_pp_d[idx]=molecular*1.0/(denominator);
				}
		}
}

__global__ void imaging_compensate_dependent_angle(float *vresult_pp_d,float *vresult_ps_d,float *angle_pp_d,float *angle_ps_d,int nx_pml,int ny_pml,int nz_pml)
{
		int tx=blockIdx.x*blockDim.x+threadIdx.x;
		int ty=blockIdx.y*blockDim.y+threadIdx.y;
		int tz=blockIdx.z;

		int idx;

		if((tx<nx_pml-2*radius)&&(ty<ny_pml-2*radius)&&(tz<nz_pml-2*radius))
		{
				tx=tx+radius;ty=ty+radius;tz=tz+radius;

				idx=tz*nx_pml*ny_pml+ty*nx_pml+tx;


				if(fabs(cos(1.0*angle_pp_d[idx]))>0.001)
				{
					vresult_pp_d[idx]=1.0*vresult_pp_d[idx]/cos(1.0*angle_pp_d[idx]);
				}

				else 	vresult_pp_d[idx]=1.0*vresult_pp_d[idx]/(cos(1.0*angle_pp_d[idx])+0.001);
		
				if(fabs(sin(1.0*angle_ps_d[idx]))<0.001)
				{
					vresult_ps_d[idx]=1.0*vresult_ps_d[idx]/sin(1.0*angle_ps_d[idx]);
				}

				else 	vresult_ps_d[idx]=1.0*vresult_ps_d[idx]/(sin(1.0*angle_ps_d[idx])+0.001);
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
