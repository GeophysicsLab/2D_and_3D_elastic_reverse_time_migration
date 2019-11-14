



///////////////////////////////////////////////////////////////////////**********************************************///////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////**********************************************//////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////**********************************************///////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////**********************************************//////////////////////////////////////////////////////
void forward_together_using_real_model(GPUdevice *mgdevice)
{


		dim3 dimBlock(32,16);

		dim3 dimGridwf_append((nnx+dimBlock.x-1)/dimBlock.x,(nny+dimBlock.y-1)/dimBlock.y,nnz_device_append);//////单块卡的整个空间
		
		dim3 dimGridwf_radius((nnx_radius+dimBlock.x-1)/dimBlock.x,(nny_radius+dimBlock.y-1)/dimBlock.y,nnz_radius);///单块卡的整个空间减去半径
		
		dim3 dimGrid_rec_lt_x_y((receiver_num_x+dimBlock.x-1)/dimBlock.x,(receiver_num_y+dimBlock.y-1)/dimBlock.y);

		dim3 dimGrid_rec_lt_x_z((receiver_num_x+dimBlock.x-1)/dimBlock.x,(receiver_num_z+dimBlock.y-1)/dimBlock.y);///seismic process and receive



					for(it=0;it<lt;it++)
						{
							if(fmod(it+1.0,1000.0)==1)	
							{
								warn("forward for modeling,isx=%d,isy=%d,isz=%d,it=%d",isx+1,isy+1,isz+1,it);
							}

							if(it<wavelet_length)
							{
								checkCudaErrors(cudaSetDevice(gpuid[choose_ns]));

								add_source_3D<<<1,1,0,mgdevice[choose_ns].stream>>>(mgdevice[choose_ns].txx1_d,mgdevice[choose_ns].wavelet_d,nnx,nny,nnz_device_append,nnz_device,it,sx_real,sy_real,sz_real,bl,bb,bu);
								add_source_3D<<<1,1,0,mgdevice[choose_ns].stream>>>(mgdevice[choose_ns].tyy1_d,mgdevice[choose_ns].wavelet_d,nnx,nny,nnz_device_append,nnz_device,it,sx_real,sy_real,sz_real,bl,bb,bu);
								add_source_3D<<<1,1,0,mgdevice[choose_ns].stream>>>(mgdevice[choose_ns].tzz1_d,mgdevice[choose_ns].wavelet_d,nnx,nny,nnz_device_append,nnz_device,it,sx_real,sy_real,sz_real,bl,bb,bu);
							}

							checkCudaErrors(cudaDeviceSynchronize());
							//if(fmod(it+1.0,1000.0)==1)	warn("add_source_3D is passing");

							exchange_device_nz_kernel_txxyyzz(mgdevice,exchange_device_bool);//////////////gpu_i exchange  gpu_i+1;///
							
							checkCudaErrors(cudaDeviceSynchronize());
							//if(fmod(it+1.0,1000.0)==1)	warn("exchange_device_nz_kernel_txxyyzz is passing");


							if((fmod(it+1.0,wavefield_interval)==0)&&join_wavefield!=0)
							//if((it==500)&&join_wavefield!=0)
							{
								system("mkdir wavefield");
								output_3d_wavefiled_tao(mgdevice,it+1);
								checkCudaErrors(cudaDeviceSynchronize());

								output_3d_wavefiled_vx(mgdevice,it+1);
								checkCudaErrors(cudaDeviceSynchronize());

								output_3d_wavefiled_vz(mgdevice,it+1);
								checkCudaErrors(cudaDeviceSynchronize());

								output_3d_wavefiled_vzp(mgdevice,it+1);
								checkCudaErrors(cudaDeviceSynchronize());

								output_3d_wavefiled_vzs(mgdevice,it+1);
								checkCudaErrors(cudaDeviceSynchronize());

								if(((it+1.0)/wavefield_interval)==(lt/wavefield_interval))
								{
									system("rm -r wavefield1");
									system("mv wavefield wavefield1");
								}
							}
					
							checkCudaErrors(cudaDeviceSynchronize());
							//if(fmod(it+1.0,1000.0)==1)	warn("output_3d_wavefiled is passing");

							for(int i=0;i<GPU_N;i++)
							{
								checkCudaErrors(cudaSetDevice(gpuid[i]));

								fwd_txxzzxzpp_3D<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].tp2_d,mgdevice[i].tp1_d,mgdevice[i].txx2_d,mgdevice[i].txx1_d,mgdevice[i].tyy2_d,mgdevice[i].tyy1_d,mgdevice[i].tzz2_d,mgdevice[i].tzz1_d,mgdevice[i].txy2_d,mgdevice[i].txy1_d,mgdevice[i].txz2_d,mgdevice[i].txz1_d,mgdevice[i].tyz2_d,mgdevice[i].tyz1_d,mgdevice[i].vx1_d,mgdevice[i].vy1_d,mgdevice[i].vz1_d,mgdevice[i].velocity_d,mgdevice[i].velocity1_d,mgdevice[i].density_d,mgdevice[i].att_d,mgdevice[i].coe_d,nnx,nny,nnz_device_append,dt,coe_x,coe_y,coe_z);							
							}
					
							checkCudaErrors(cudaDeviceSynchronize());
							//if(fmod(it+1.0,1000.0)==1)	warn("fwd_txxzzxzpp_3D is passing");

							exchange_device_nz_kernel_tao2(mgdevice,exchange_device_bool);//////////////gpu_i exchange  gpu_i+1;///		

							exchange_device_nz_kernel_taop2(mgdevice,exchange_device_bool);//////////////gpu_i exchange  gpu_i+1;///
							
							checkCudaErrors(cudaDeviceSynchronize());
							//if(fmod(it+1.0,1000.0)==1)	warn("exchange_device_nz_kernel_tao2 taop2 is passing");

							for(int i=0;i<GPU_N;i++)
							{
								cudaSetDevice(gpuid[i]);

								fwd_vx_3D<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].vx2_d,mgdevice[i].vx1_d,mgdevice[i].txx2_d,mgdevice[i].txy2_d,mgdevice[i].txz2_d,mgdevice[i].velocity_d,mgdevice[i].velocity1_d,mgdevice[i].density_d,mgdevice[i].att_d,mgdevice[i].coe_d,nnx,nny,nnz_device_append,dt,coe_x,coe_y,coe_z);

								fwd_vy_3D<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].vy2_d,mgdevice[i].vy1_d,mgdevice[i].txy2_d,mgdevice[i].tyy2_d,mgdevice[i].tyz2_d,mgdevice[i].velocity_d,mgdevice[i].velocity1_d,mgdevice[i].density_d,mgdevice[i].att_d,mgdevice[i].coe_d,nnx,nny,nnz_device_append,dt,coe_x,coe_y,coe_z);

								fwd_vz_3D<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].vz2_d,mgdevice[i].vz1_d,mgdevice[i].txz2_d,mgdevice[i].tyz2_d,mgdevice[i].tzz2_d,mgdevice[i].velocity_d,mgdevice[i].velocity1_d,mgdevice[i].density_d,mgdevice[i].att_d,mgdevice[i].coe_d,nnx,nny,nnz_device_append,dt,coe_x,coe_y,coe_z);
							}

							checkCudaErrors(cudaDeviceSynchronize());
							//if(fmod(it+1.0,1000.0)==1)	warn("fwd_vx_3D fwd_vy_3D fwd_vz_3D is passing");
					
							exchange_device_nz_kernel_vx_vy_vz2(mgdevice,exchange_device_bool);//////////////gpu_i exchange  gpu_i+1;///
							checkCudaErrors(cudaDeviceSynchronize());
							//if(fmod(it+1.0,1000.0)==1)	warn("exchange_device_nz_kernel_vx_vy_vz2 is passing");

							for(int i=0;i<GPU_N;i++)
							{
								checkCudaErrors(cudaSetDevice(gpuid[i]));

								fwd_vxp_vzp_3D<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].vxp2_d,mgdevice[i].vxp1_d,mgdevice[i].vyp2_d,mgdevice[i].vyp1_d,mgdevice[i].vzp2_d,mgdevice[i].vzp1_d,mgdevice[i].tp2_d,mgdevice[i].vxs2_d,mgdevice[i].vys2_d,mgdevice[i].vzs2_d,mgdevice[i].vx2_d,mgdevice[i].vy2_d,mgdevice[i].vz2_d,mgdevice[i].velocity_d,mgdevice[i].velocity1_d,mgdevice[i].density_d,mgdevice[i].att_d,mgdevice[i].coe_d,nnx,nny,nnz_device_append,dt,coe_x,coe_y,coe_z);	
							}

							checkCudaErrors(cudaDeviceSynchronize());
							//if(fmod(it+1.0,1000.0)==1)	warn("fwd_vxp_vzp_3D is passing");
																		
							{
								checkCudaErrors(cudaSetDevice(gpuid[choose_re]));

								write_or_add_shot_3D_surface_three<<<dimGrid_rec_lt_x_y,dimBlock,0,mgdevice[choose_re].stream>>>(mgdevice[choose_re].obs_shot_x_d,mgdevice[choose_re].obs_shot_y_d,mgdevice[choose_re].obs_shot_z_d,mgdevice[choose_re].vx2_d,mgdevice[choose_re].vy2_d,mgdevice[choose_re].vz2_d,nnx,nny,nnz_device_append,nnz_device,bl,bb,bu,receiver_start_x,receiver_num_x,receiver_interval_x,receiver_start_y,receiver_num_y,receiver_interval_y,receiver_start_z,receiver_num_z,receiver_interval_z,0);

								checkCudaErrors(cudaDeviceSynchronize());
								//if(fmod(it+1.0,1000.0)==1)	warn("write_or_add_shot_3D_surface_three is passing");
	
								if(vsp==0)
								{
									//cut_direct_shot_3D_surface_three<<<dimGrid_rec_lt_x_y,dimBlock,0,mgdevice[choose_re].stream>>>(mgdevice[choose_re].obs_shot_x_d,mgdevice[choose_re].obs_shot_y_d,mgdevice[choose_re].obs_shot_z_d,mgdevice[choose_re].velocity_d,nnx,nny,nnz_device,bl,bb,bu,receiver_start_x,receiver_num_x,receiver_interval_x,receiver_start_y,receiver_num_y,receiver_interval_y,receiver_start_z,receiver_num_z,receiver_interval_z,sx_real,sy_real,sz_real,it,wavelet_length,dx,dy,dz,dt);
									cut_direct_shot_3D_surface_three_new<<<dimGrid_rec_lt_x_y,dimBlock,0,mgdevice[choose_re].stream>>>(mgdevice[choose_re].obs_shot_x_d,mgdevice[choose_re].obs_shot_y_d,mgdevice[choose_re].obs_shot_z_d,mgdevice[choose_re].velocity_d,nnx,nny,nnz_device,bl,bb,bu,receiver_start_x,receiver_num_x,receiver_interval_x,receiver_start_y,receiver_num_y,receiver_interval_y,receiver_start_z,receiver_num_z,receiver_interval_z,sx_real,sy_real,sz_real,it,wavelet_length,dx,dy,dz,dt,cut_direct_wave);												
								}
								checkCudaErrors(cudaDeviceSynchronize());
								//if(fmod(it+1.0,1000.0)==1)	warn("cut_direct_shot_3D_surface is passing");
															
								transfer_gpu_to_cpu_multicomponent_seismic(mgdevice,it,0);

								checkCudaErrors(cudaDeviceSynchronize());
								//if(fmod(it+1.0,1000.0)==1)	warn("transfer_gpu_to_cpu_multicomponent_seismic is passing");	
							}

							checkCudaErrors(cudaDeviceSynchronize());
							//if(fmod(it+1.0,1000.0)==1)	warn("write_shot is passing");

							exchange_wavefiled_new(mgdevice);//////////////////change wavefield_new
							//exchange_wavefiled_old(mgdevice);//////////////////change wavefield_old	

							checkCudaErrors(cudaDeviceSynchronize());		
							//if(fmod(it+1.0,1000.0)==1)	warn("exchange_wavefiled is passing");
						}
}

///////////////////////////////////////////////////////////////////////**********************************************///////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////**********************************************//////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////**********************************************///////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////**********************************************//////////////////////////////////////////////////////
void forward_together_using_smoothed_model(GPUdevice *mgdevice)
{
		dim3 dimBlock(32,16);

		dim3 dimGridwf_append((nnx+dimBlock.x-1)/dimBlock.x,(nny+dimBlock.y-1)/dimBlock.y,nnz_device_append);//////单块卡的整个空间
		
		dim3 dimGridwf_radius((nnx_radius+dimBlock.x-1)/dimBlock.x,(nny_radius+dimBlock.y-1)/dimBlock.y,nnz_radius);///单块卡的整个空间减去半径
		
		dim3 dimGrid_rec_lt_x_y((receiver_num_x+dimBlock.x-1)/dimBlock.x,(receiver_num_y+dimBlock.y-1)/dimBlock.y);

		dim3 dimGrid_rec_lt_x_z((receiver_num_x+dimBlock.x-1)/dimBlock.x,(receiver_num_z+dimBlock.y-1)/dimBlock.y);///seismic process and receive

					for(it=0;it<2*lt/3;it++)
						{
							if(fmod(it+1.0,1000.0)==1)	
							{
								warn("forward for Elastic RTM,isx=%d,isy=%d,isz=%d,it=%d",isx+1,isy+1,isz+1,it);
							}

							if(it<wavelet_length)
							{
								checkCudaErrors(cudaSetDevice(gpuid[choose_ns]));

								add_source_3D<<<1,1,0,mgdevice[choose_ns].stream>>>(mgdevice[choose_ns].txx1_d,mgdevice[choose_ns].wavelet_d,nnx,nny,nnz_device_append,nnz_device,it,sx_real,sy_real,sz_real,bl,bb,bu);
								add_source_3D<<<1,1,0,mgdevice[choose_ns].stream>>>(mgdevice[choose_ns].tyy1_d,mgdevice[choose_ns].wavelet_d,nnx,nny,nnz_device_append,nnz_device,it,sx_real,sy_real,sz_real,bl,bb,bu);
								add_source_3D<<<1,1,0,mgdevice[choose_ns].stream>>>(mgdevice[choose_ns].tzz1_d,mgdevice[choose_ns].wavelet_d,nnx,nny,nnz_device_append,nnz_device,it,sx_real,sy_real,sz_real,bl,bb,bu);
							}

							checkCudaErrors(cudaDeviceSynchronize());
							//if(fmod(it+1.0,1000.0)==1)	warn("add_source_3D is passing");

							exchange_device_nz_kernel_txxyyzz(mgdevice,exchange_device_bool);//////////////gpu_i exchange  gpu_i+1;///
							
							checkCudaErrors(cudaDeviceSynchronize());
							//if(fmod(it+1.0,1000.0)==1)	warn("exchange_device_nz_kernel_txxyyzz is passing");


							if((fmod(it+1.0,wavefield_interval)==0)&&join_wavefield!=0)
							//if((it==500)&&join_wavefield!=0)
							{
								system("mkdir wavefield");
								output_3d_wavefiled_tao(mgdevice,it+1);
								checkCudaErrors(cudaDeviceSynchronize());

								output_3d_wavefiled_vx(mgdevice,it+1);
								checkCudaErrors(cudaDeviceSynchronize());

								output_3d_wavefiled_vz(mgdevice,it+1);
								checkCudaErrors(cudaDeviceSynchronize());

								output_3d_wavefiled_vzp(mgdevice,it+1);
								checkCudaErrors(cudaDeviceSynchronize());

								output_3d_wavefiled_vzs(mgdevice,it+1);
								checkCudaErrors(cudaDeviceSynchronize());

								if(((it+1.0)/wavefield_interval)==((2*lt/3)/wavefield_interval))
								{
									system("rm -r wavefield2");
									system("mv wavefield wavefield2");
								}
							}
					
							checkCudaErrors(cudaDeviceSynchronize());
							//if(fmod(it+1.0,1000.0)==1)	warn("output_3d_wavefiled is passing");

							for(int i=0;i<GPU_N;i++)
							{
								checkCudaErrors(cudaSetDevice(gpuid[i]));

								fwd_txxzzxzpp_3D<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].tp2_d,mgdevice[i].tp1_d,mgdevice[i].txx2_d,mgdevice[i].txx1_d,mgdevice[i].tyy2_d,mgdevice[i].tyy1_d,mgdevice[i].tzz2_d,mgdevice[i].tzz1_d,mgdevice[i].txy2_d,mgdevice[i].txy1_d,mgdevice[i].txz2_d,mgdevice[i].txz1_d,mgdevice[i].tyz2_d,mgdevice[i].tyz1_d,mgdevice[i].vx1_d,mgdevice[i].vy1_d,mgdevice[i].vz1_d,mgdevice[i].s_velocity_d,mgdevice[i].s_velocity1_d,mgdevice[i].s_density_d,mgdevice[i].att_d,mgdevice[i].coe_d,nnx,nny,nnz_device_append,dt,coe_x,coe_y,coe_z);							
							}
					
							checkCudaErrors(cudaDeviceSynchronize());
							//if(fmod(it+1.0,1000.0)==1)	warn("fwd_txxzzxzpp_3D is passing");

							exchange_device_nz_kernel_tao2(mgdevice,exchange_device_bool);//////////////gpu_i exchange  gpu_i+1;///		

							exchange_device_nz_kernel_taop2(mgdevice,exchange_device_bool);//////////////gpu_i exchange  gpu_i+1;///
							
							checkCudaErrors(cudaDeviceSynchronize());
							//if(fmod(it+1.0,1000.0)==1)	warn("exchange_device_nz_kernel_tao2 taop2 is passing");

							for(int i=0;i<GPU_N;i++)
							{
								cudaSetDevice(gpuid[i]);

								fwd_vx_3D<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].vx2_d,mgdevice[i].vx1_d,mgdevice[i].txx2_d,mgdevice[i].txy2_d,mgdevice[i].txz2_d,mgdevice[i].s_velocity_d,mgdevice[i].s_velocity1_d,mgdevice[i].s_density_d,mgdevice[i].att_d,mgdevice[i].coe_d,nnx,nny,nnz_device_append,dt,coe_x,coe_y,coe_z);

								fwd_vy_3D<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].vy2_d,mgdevice[i].vy1_d,mgdevice[i].txy2_d,mgdevice[i].tyy2_d,mgdevice[i].tyz2_d,mgdevice[i].s_velocity_d,mgdevice[i].s_velocity1_d,mgdevice[i].s_density_d,mgdevice[i].att_d,mgdevice[i].coe_d,nnx,nny,nnz_device_append,dt,coe_x,coe_y,coe_z);

								fwd_vz_3D<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].vz2_d,mgdevice[i].vz1_d,mgdevice[i].txz2_d,mgdevice[i].tyz2_d,mgdevice[i].tzz2_d,mgdevice[i].s_velocity_d,mgdevice[i].s_velocity1_d,mgdevice[i].s_density_d,mgdevice[i].att_d,mgdevice[i].coe_d,nnx,nny,nnz_device_append,dt,coe_x,coe_y,coe_z);
							}

							checkCudaErrors(cudaDeviceSynchronize());
							//if(fmod(it+1.0,1000.0)==1)	warn("fwd_vx_3D fwd_vy_3D fwd_vz_3D is passing");
					
							exchange_device_nz_kernel_vx_vy_vz2(mgdevice,exchange_device_bool);//////////////gpu_i exchange  gpu_i+1;///
							//if(fmod(it+1.0,1000.0)==1)	warn("exchange_device_nz_kernel_vx_vy_vz2 is passing");

							for(int i=0;i<GPU_N;i++)
							{
								checkCudaErrors(cudaSetDevice(gpuid[i]));

								fwd_vxp_vzp_3D<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].vxp2_d,mgdevice[i].vxp1_d,mgdevice[i].vyp2_d,mgdevice[i].vyp1_d,mgdevice[i].vzp2_d,mgdevice[i].vzp1_d,mgdevice[i].tp2_d,mgdevice[i].vxs2_d,mgdevice[i].vys2_d,mgdevice[i].vzs2_d,mgdevice[i].vx2_d,mgdevice[i].vy2_d,mgdevice[i].vz2_d,mgdevice[i].s_velocity_d,mgdevice[i].s_velocity1_d,mgdevice[i].s_density_d,mgdevice[i].att_d,mgdevice[i].coe_d,nnx,nny,nnz_device_append,dt,coe_x,coe_y,coe_z);	
							}

							checkCudaErrors(cudaDeviceSynchronize());
							//if(fmod(it+1.0,1000.0)==1)	warn("fwd_vxp_vzp_3D is passing");							

							exchange_wavefiled_new(mgdevice);//////////////////change wavefield_new
							//exchange_wavefiled_old(mgdevice);//////////////////change wavefield_old	

							checkCudaErrors(cudaDeviceSynchronize());
							//if(fmod(it+1.0,1000.0)==1)	warn("exchange_wavefiled_new is passing");

							for(int i=0;i<GPU_N;i++)
							{
								checkCudaErrors(cudaSetDevice(gpuid[i]));

								//cuda_cal_excitation_amp_time<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].ex_time_d,mgdevice[i].ex_tp_d,mgdevice[i].tp2_d,mgdevice[i].ex_vxp_d,mgdevice[i].vxp2_d,mgdevice[i].ex_vyp_d,mgdevice[i].vyp2_d,mgdevice[i].ex_vzp_d,mgdevice[i].vzp2_d,nnx,nny,nnz_device_append,it);
								cuda_cal_excitation_amp_time_new<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].ex_time_d,mgdevice[i].ex_amp_d,mgdevice[i].ex_tp_d,mgdevice[i].tp2_d,mgdevice[i].ex_vxp_d,mgdevice[i].vxp2_d,mgdevice[i].ex_vyp_d,mgdevice[i].vyp2_d,mgdevice[i].ex_vzp_d,mgdevice[i].vzp2_d,nnx,nny,nnz_device_append,it);
								//cuda_cal_source_poyn_3D<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].poyn_px_d,mgdevice[i].poyn_py_d,mgdevice[i].poyn_pz_d,mgdevice[i].ex_time_d,mgdevice[i].vxp2_d,mgdevice[i].vyp2_d,mgdevice[i].vzp2_d,mgdevice[i].tp2_d,nnx,nny,nnz_device_append,it);
							}

							checkCudaErrors(cudaDeviceSynchronize());
							//if(fmod(it+1.0,1000.0)==1)	warn("cuda_cal_excitation_amp_time_new is passing");
						
						}
}

///////////////////////////////////////////////////////////////////////**********************************************///////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////**********************************************//////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////**********************************************///////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////**********************************************//////////////////////////////////////////////////////
void backward_together_using_smoothed_model(GPUdevice *mgdevice)
{

		dim3 dimBlock(32,16);

		dim3 dimGridwf_append((nnx+dimBlock.x-1)/dimBlock.x,(nny+dimBlock.y-1)/dimBlock.y,nnz_device_append);//////单块卡的整个空间
		
		dim3 dimGridwf_radius((nnx_radius+dimBlock.x-1)/dimBlock.x,(nny_radius+dimBlock.y-1)/dimBlock.y,nnz_radius);///单块卡的整个空间减去半径
		
		dim3 dimGrid_rec_lt_x_y((receiver_num_x+dimBlock.x-1)/dimBlock.x,(receiver_num_y+dimBlock.y-1)/dimBlock.y);

		dim3 dimGrid_rec_lt_x_z((receiver_num_x+dimBlock.x-1)/dimBlock.x,(receiver_num_z+dimBlock.y-1)/dimBlock.y);///seismic process and receive


						for(it=lt-1;it>=0;it--)
						{
							if(fmod(it+1.0,1000.0)==1)	
							{
								warn("backward for Elastic RTM,isx=%d,isy=%d,isz=%d,it=%d",isx+1,isy+1,isz+1,it);
							}
							
							{
								transfer_gpu_to_cpu_multicomponent_seismic(mgdevice,it,1);

								checkCudaErrors(cudaDeviceSynchronize());
								//if(fmod(it+1.0,1000.0)==1)	warn("transfer_gpu_to_cpu_multicomponent_seismic is passing");			
							}


							{
								checkCudaErrors(cudaSetDevice(gpuid[choose_re]));

								write_or_add_shot_3D_surface_three<<<dimGrid_rec_lt_x_y,dimBlock,0,mgdevice[choose_re].stream>>>(mgdevice[choose_re].obs_shot_x_d,mgdevice[choose_re].obs_shot_y_d,mgdevice[choose_re].obs_shot_z_d,mgdevice[choose_re].vx1_d,mgdevice[choose_re].vy1_d,mgdevice[choose_re].vz1_d,nnx,nny,nnz_device_append,nnz_device,bl,bb,bu,receiver_start_x,receiver_num_x,receiver_interval_x,receiver_start_y,receiver_num_y,receiver_interval_y,receiver_start_z,receiver_num_z,receiver_interval_z,add_receiver_bool);	

								checkCudaErrors(cudaDeviceSynchronize());
								//if(fmod(it+1.0,1000.0)==1)	warn("write_or_add_shot_3D_surface_three is passing");
							}
								
							{
								exchange_device_nz_kernel_vx_vy_vz1(mgdevice,exchange_device_bool);//////////////gpu_i exchange  gpu_i+1;///

								checkCudaErrors(cudaDeviceSynchronize());
								//if(fmod(it+1.0,1000.0)==1)	warn("exchange_device_nz_kernel_vx_vy_vz1 is passing");
							}

							if((fmod(it+1.0,wavefield_interval)==0)&&join_wavefield!=0)
							//if((it==500)&&join_wavefield!=0)
							{
								system("mkdir wavefield");
								output_3d_wavefiled_tao(mgdevice,it+1);
								checkCudaErrors(cudaDeviceSynchronize());

								output_3d_wavefiled_vx(mgdevice,it+1);
								checkCudaErrors(cudaDeviceSynchronize());

								output_3d_wavefiled_vz(mgdevice,it+1);
								checkCudaErrors(cudaDeviceSynchronize());

								output_3d_wavefiled_vzp(mgdevice,it+1);
								checkCudaErrors(cudaDeviceSynchronize());

								output_3d_wavefiled_vzs(mgdevice,it+1);
								checkCudaErrors(cudaDeviceSynchronize());

								if(((it+1.0)/wavefield_interval)==(lt/wavefield_interval))
								{
									system("rm -r wavefield3");
									system("mv wavefield wavefield3");
								}
							}
					
								checkCudaErrors(cudaDeviceSynchronize());
								//if(fmod(it+1.0,1000.0)==1)	warn("output_3d_wavefiled is passing");

							for(int i=0;i<GPU_N;i++)
							{
								checkCudaErrors(cudaSetDevice(gpuid[i]));

								fwd_txxzzxzpp_3D<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].tp2_d,mgdevice[i].tp1_d,mgdevice[i].txx2_d,mgdevice[i].txx1_d,mgdevice[i].tyy2_d,mgdevice[i].tyy1_d,mgdevice[i].tzz2_d,mgdevice[i].tzz1_d,mgdevice[i].txy2_d,mgdevice[i].txy1_d,mgdevice[i].txz2_d,mgdevice[i].txz1_d,mgdevice[i].tyz2_d,mgdevice[i].tyz1_d,mgdevice[i].vx1_d,mgdevice[i].vy1_d,mgdevice[i].vz1_d,mgdevice[i].s_velocity_d,mgdevice[i].s_velocity1_d,mgdevice[i].s_density_d,mgdevice[i].att_d,mgdevice[i].coe_d,nnx,nny,nnz_device_append,dt,coe_x,coe_y,coe_z);							
							}
					
								checkCudaErrors(cudaDeviceSynchronize());
								//if(fmod(it+1.0,1000.0)==1)	warn("fwd_txxzzxzpp_3D is passing");

							{
								exchange_device_nz_kernel_tao2(mgdevice,exchange_device_bool);//////////////gpu_i exchange  gpu_i+1;///

								exchange_device_nz_kernel_taop2(mgdevice,exchange_device_bool);//////////////gpu_i exchange  gpu_i+1;///
								
								checkCudaErrors(cudaDeviceSynchronize());
								//if(fmod(it+1.0,1000.0)==1)	warn("exchange_device_nz_kernel_tao2 taop2 is passing");
							}

							for(int i=0;i<GPU_N;i++)
							{
								cudaSetDevice(gpuid[i]);

								fwd_vx_3D<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].vx2_d,mgdevice[i].vx1_d,mgdevice[i].txx2_d,mgdevice[i].txy2_d,mgdevice[i].txz2_d,mgdevice[i].s_velocity_d,mgdevice[i].s_velocity1_d,mgdevice[i].s_density_d,mgdevice[i].att_d,mgdevice[i].coe_d,nnx,nny,nnz_device_append,dt,coe_x,coe_y,coe_z);

								fwd_vy_3D<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].vy2_d,mgdevice[i].vy1_d,mgdevice[i].txy2_d,mgdevice[i].tyy2_d,mgdevice[i].tyz2_d,mgdevice[i].s_velocity_d,mgdevice[i].s_velocity1_d,mgdevice[i].s_density_d,mgdevice[i].att_d,mgdevice[i].coe_d,nnx,nny,nnz_device_append,dt,coe_x,coe_y,coe_z);

								fwd_vz_3D<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].vz2_d,mgdevice[i].vz1_d,mgdevice[i].txz2_d,mgdevice[i].tyz2_d,mgdevice[i].tzz2_d,mgdevice[i].s_velocity_d,mgdevice[i].s_velocity1_d,mgdevice[i].s_density_d,mgdevice[i].att_d,mgdevice[i].coe_d,nnx,nny,nnz_device_append,dt,coe_x,coe_y,coe_z);
							}

								checkCudaErrors(cudaDeviceSynchronize());
								//if(fmod(it+1.0,1000.0)==1)	warn("fwd_vx_3D fwd_vy_3D fwd_vz_3D is passing");
					
							{
								exchange_device_nz_kernel_vx_vy_vz2(mgdevice,exchange_device_bool);//////////////gpu_i exchange  gpu_i+1;///

								checkCudaErrors(cudaDeviceSynchronize());
								//if(fmod(it+1.0,1000.0)==1)	warn("exchange_device_nz_kernel_vx_vy_vz2 is passing");
							}

							for(int i=0;i<GPU_N;i++)
							{
								checkCudaErrors(cudaSetDevice(gpuid[i]));

								fwd_vxp_vzp_3D<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].vxp2_d,mgdevice[i].vxp1_d,mgdevice[i].vyp2_d,mgdevice[i].vyp1_d,mgdevice[i].vzp2_d,mgdevice[i].vzp1_d,mgdevice[i].tp2_d,mgdevice[i].vxs2_d,mgdevice[i].vys2_d,mgdevice[i].vzs2_d,mgdevice[i].vx2_d,mgdevice[i].vy2_d,mgdevice[i].vz2_d,mgdevice[i].s_velocity_d,mgdevice[i].s_velocity1_d,mgdevice[i].s_density_d,mgdevice[i].att_d,mgdevice[i].coe_d,nnx,nny,nnz_device_append,dt,coe_x,coe_y,coe_z);	
							}

								checkCudaErrors(cudaDeviceSynchronize());
								//if(fmod(it+1.0,1000.0)==1)	warn("fwd_vxp_vzp_3D is passing");							

							{
								exchange_wavefiled_new(mgdevice);//////////////////change wavefield_new
								//exchange_wavefiled_old(mgdevice);//////////////////change wavefield_old	

								checkCudaErrors(cudaDeviceSynchronize());
								//if(fmod(it+1.0,1000.0)==1)	warn("exchange_wavefiled_new is passing");	
							}

							for(int i=0;i<GPU_N;i++)
							{
								checkCudaErrors(cudaSetDevice(gpuid[i]));

								//cuda_cal_receiver_poyn_3D<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].poyn_rpx_d,mgdevice[i].poyn_rpy_d,mgdevice[i].poyn_rpz_d,mgdevice[i].poyn_rsx_d,mgdevice[i].poyn_rsy_d,mgdevice[i].poyn_rsz_d,mgdevice[i].ex_time_d,mgdevice[i].vxp2_d,mgdevice[i].vyp2_d,mgdevice[i].vzp2_d,mgdevice[i].vxs2_d,mgdevice[i].vys2_d,mgdevice[i].vzs2_d,mgdevice[i].tp2_d,mgdevice[i].txx2_d,mgdevice[i].tyy2_d,mgdevice[i].tzz2_d,mgdevice[i].txy2_d,mgdevice[i].txz2_d,mgdevice[i].tyz2_d,nnx,nny,nnz_device_append,it);
							}	
					
								checkCudaErrors(cudaDeviceSynchronize());
								//if(fmod(it+1.0,1000.0)==1)	warn("cuda_cal_receiver_poyn_3D is passing");

							for(int i=0;i<GPU_N;i++)
							{
								checkCudaErrors(cudaSetDevice(gpuid[i]));

								imaging_correlation_ex<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].ex_time_d,mgdevice[i].ex_tp_d,mgdevice[i].tp2_d,mgdevice[i].vresult_tp_d,nnx,nny,nnz_device_append,it,amp_max,precon_z1);

								imaging_inner_product_ex<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].ex_time_d,mgdevice[i].ex_vxp_d,mgdevice[i].ex_vyp_d,mgdevice[i].ex_vzp_d,mgdevice[i].vxp2_d,mgdevice[i].vyp2_d,mgdevice[i].vzp2_d,mgdevice[i].vresult_pp_d,nnx,nny,nnz_device_append,it,tp_max,precon_z1);

								imaging_inner_product_ex<<<dimGridwf_append,dimBlock,0,mgdevice[i].stream>>>(mgdevice[i].ex_time_d,mgdevice[i].ex_vxp_d,mgdevice[i].ex_vyp_d,mgdevice[i].ex_vzp_d,mgdevice[i].vxs2_d,mgdevice[i].vys2_d,mgdevice[i].vzs2_d,mgdevice[i].vresult_ps_d,nnx,nny,nnz_device_append,it,tp_max,precon_z1);
							}

								checkCudaErrors(cudaDeviceSynchronize());
								//if(fmod(it+1.0,1000.0)==1)	warn("imaging_inner_product_ex is passing");

						}
}
