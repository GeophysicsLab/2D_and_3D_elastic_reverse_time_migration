#include <time.h>
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <malloc.h>
#include <math.h>
#include "cublas.h"
// includes, project
#include <cufft.h>
//#include <cutil_inline.h>
//#include <shrQATest.h>
#include "su.h"
#include "segy.h"
#include "Complex.h"
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include "zzzzz"
#include "elastic_2D_kernel_1.cu"
#include "elastic_2D_kernel_2.cu"
#include "elastic_2D_kernel_3.cu"
#include "elastic_adjoint_equation.cu"
#include "viscoelastic_equation.cu"

#define radius 6
#define pai 3.1415926
#define Block_Size  512	/* vector computation blocklength */
#define scale 9

/*********************** self documentation ******************************/
char *sdoc[] = {
"                                                                        ",
" this is a program to model elastic zhengyan by FD ",
" this was created by zhange in Daqing in 2015-09-06 ",
" Prestack visco/elastic LSRTM in 2017-8-31 ",
" Prestack           Prestack               Prestack"
"                                                                        ",
NULL};
/**************** end self doc *******************************************/
segy tr;

//static time_t t1,t2;
//----------------------------- main -------------------------------------------
int main(int argc, char **argv)
{
		//requestdoc(1);
		initargs(argc,argv);

		int iter_start,niter,iter,join_vs,join_den,join_shot,precon,precon_z1,precon_z2,laplace,laplace_compensate,rbell,inversion_para,inversion_den;
		int nx,nz,nx_append,nx_append_new,nz_append,nxnz,nxanza,nx_size_nz,nxa_new_nza,lt_rec;
		float dx,dz;
		int lt;
		float dt;
		float freq;
		int wavelet_length,wavelet_half;
		int shot_num,shot_start,shot_interval,shot_depth;
		int receiver_num,receiver_start,receiver_interval,receiver_depth,receiver_offset,receiver_mark;
		int vsp,shot_z_interval,receiver_z_interval,decomposition;/////////////////for vsp.............2017年03月14日 星期二 08时32分25秒
		int vsp_2,receiver_start_2,receiver_interval_2,receiver_depth_2,receiver_z_interval_2,receiver_num_2;//for vsp2...2017年03月17日 星期二 08时32分25秒
		int vsp_precon;//for vsp2...2017年03月17日 星期二 08时32分25秒  
		int boundary_up,boundary_down,boundary_left,boundary_right;
		float coe_attenuation,*cal_max,*obs_max,array_max;
		char filename[100],filename1[100];

		float smooth_time_vp,smooth_time_vs,smooth_time_density,smooth_time_qp,smooth_time_qs;////velocity smooth time
		int length_vp,length_vs,length_density,length_qp,length_qs;////velocity smooth time2017年08月15日 星期二 09时57分09秒 

		int cuda_code;
		float *rep;
		FILE *logfile=NULL,*rm_f_file=NULL;float mstimer=0.0,totaltime=0.0;
		logfile=fopen("log.txt","ab");//remember to free log file

		int ittt_beg=0,ittt_end=0;
		int join_wavefield,RTM_only;

		float gpumem=0.0,gpumem_residual=0.0,change=0.0;/////for check 2017年07月27日 星期四 10时11分40秒 
		int check_number,check_interval,check_residual,variable_number;/////for check 2017年07月27日 星期四 10时11分40秒 
		if(!getparint("variable_number",&variable_number))		err("variable_number is not set!");/////for check 2017年07月27日 星期四 10时11分40秒		
		
		int migration_type,modeling_type,accumulation,correlation_misfit,amplitude_error,amplitude_error_number,cut_direct_wave,shot_scale,offset_attenuation;///2017年07月27日 星期四 20时17分11秒 
		if(!getparint("migration_type",&migration_type))				err("migration_type is not set!");////QQQQQQQ2017年07月30日 星期日 09时23分52秒 
		if(!getparint("modeling_type",&modeling_type))				err("modeling_type is not set!");
		if(!getparint("accumulation",&accumulation))				err("accumulation is not set!");/////for add virtual source
		if(!getparint("correlation_misfit",&correlation_misfit))			err("correlation_misfit is not set!");/////for correlation_misfit 
		if(!getparint("amplitude_error",&amplitude_error))				err("amplitude_error is not set!");/////for correlation_misfit 
		if(!getparint("amplitude_error_number",&amplitude_error_number))		err("amplitude_error_number is not set!");/////for correlation_misfit 
		if(!getparint("shot_scale",&shot_scale))					err("shot_scale is not set!");/////for correlation_misfit 
		if(!getparint("cut_direct_wave",&cut_direct_wave))				err("cut_direct_wave is not set!");/////for correlation_misfit

		if(!getparint("offset_attenuation",&offset_attenuation))			err("offset_attenuation is not set!");/////for correlation_misfit 
		//migration_type=1  denote: viscoelastic migration 
		//migration_type=0  denote: elastic migration
		//modeling_type=1  denote:  viscoelastic modeling 
		//modeling_type=0  denote:  elastic modeling 
		if(!getparint("join_wavefield",&join_wavefield))		err("join_wavefield is not set!"); 
		if(!getparint("RTM_only",&RTM_only))			err("RTM_only is not set!"); 

		if(!getparint("cuda_code",&cuda_code))			err("cuda_code is not set!");
		cudaSetDevice(cuda_code);

		if(!getparint("inversion_para",&inversion_para))		err("inversion_para is not set!");//////////   inversion      parameter
		if(inversion_para==0||inversion_para==1)			warn("inversion parameter is lame coefficient\n");
		if(inversion_para==2)					warn("inversion parameter is velocity\n");
		if(inversion_para==3)					warn("inversion parameter is impedance\n");

		if(!getparint("inversion_den",&inversion_den))		err("inversion_den is not set!");//////////   inversion      parameter
		if(inversion_den==0)						warn("inversion parameter have not density\n");
		if(inversion_den==1)						warn("inversion parameter have density\n");
		/////////
		if(!getparint("precon",&precon))				err("precon is not set!");
		if(!getparint("precon_z1",&precon_z1))			err("precon_z1 is not set!");
		if(!getparint("precon_z2",&precon_z2))			err("precon_z2 is not set!");

		if(!getparint("laplace",&laplace))				err("laplace is not set!");
		if(!getparint("laplace_compensate",&laplace_compensate))				err("laplace_compensate is not set!");

		if(!getparint("rbell",&rbell))				rbell=2;
		if(!getparint("iter_start",&iter_start))			err("iter_start is not set!");///////////for program died ,restart
		if(!getparint("niter",&niter))				err("niter is not set!");

		if(!getparint("join_vs",&join_vs))				err("join_vs is not set!");////join_vs=1 denote input s wave velocity(velocity1)
		if(!getparint("join_den",&join_den))			err("join_den is not set!");////join_den=1 denote input density
		if(!getparint("join_shot",&join_shot))			err("join_shot is not set!");////join_shot=1 denote obs_x_shot and obs_z_shot has gotten

		if(!getparint("nx",&nx))					err("nx is not set!");
		if(!getparint("nz",&nz))					err("nz is not set!");
		if(!getparfloat("dx",&dx))					err("dx is not set!");
		if(!getparfloat("dz",&dz))					err("dz is not set!");

		if(!getparint("lt",&lt))					err("lt is not set!");
		if(!getparfloat("dt",&dt))					err("dt is not set!");
		if(!getparfloat("freq",&freq))				err("freq is not set!");

		if(!getparint("shot_num",&shot_num))			err("shot_num is not set!");
		if(!getparint("shot_start",&shot_start))			err("shot_start is not set!");
		if(!getparint("shot_interval",&shot_interval))		err("shot_interval is not set!");
		if(!getparint("shot_depth",&shot_depth))			err("shot_depth is not set!");

		if(!getparint("receiver_num",&receiver_num))		err("receiver_num is not set!");
		if(!getparint("receiver_start",&receiver_start))		err("receiver_start is not set!");
		if(!getparint("receiver_interval",&receiver_interval))	err("receiver_interval is not set!");
		if(!getparint("receiver_depth",&receiver_depth))		err("receiver_depth is not set!");
		if(!getparint("receiver_offset",&receiver_offset))		err("receiver_offset is not set!");
		if(!getparint("receiver_mark",&receiver_mark))		err("receiver_mark is not set!");
/////////for vsp???/////////////////for vsp.............2017年03月14日 星期二 08时32分25秒
		if(!getparint("decomposition",&decomposition))			err("decomposition is not set!"); 
		if(!getparint("vsp",&vsp))						err("vsp is not set!");
		if(!getparint("receiver_z_interval",&receiver_z_interval))	err("receiver_z_interval is not set!");
		if(!getparint("shot_z_interval",&shot_z_interval))			err("shot_z_interval is not set!");

///////////////////////for vsp2222222222222222222
		if(!getparint("vsp_2",&vsp_2))						err("vsp_2 is not set!");
		if(vsp_2!=0)
		{
			if(!getparint("receiver_start_2",&receiver_start_2))		err("receiver_start_2 is not set!");
			if(!getparint("receiver_interval_2",&receiver_interval_2))	err("receiver_interval_2 is not set!");
			if(!getparint("receiver_depth_2",&receiver_depth_2))		err("receiver_depth_2 is not set!");
			if(!getparint("receiver_z_interval_2",&receiver_z_interval_2))	err("receiver_z_interval_2 is not set!");		
			if(!getparint("receiver_num_2",&receiver_num_2))			err("receiver_num_2 is not set!");
		}
		if(!getparint("vsp_precon",&vsp_precon))					err("vsp_precon is not set!");
/////////for vsp???/////////////////for vsp.............2017年03月14日 星期二 08时32分25秒 
		if(!getparint("boundary_up",&boundary_up))				err("boundary_up is not set!");
		if(!getparint("boundary_down",&boundary_down))			err("boundary_down is not set!");
		if(!getparint("boundary_left",&boundary_left))			err("boundary_left is not set!");
		if(!getparint("boundary_right",&boundary_right))			err("boundary_right is not set!");
		if(!getparfloat("coe_attenuation",&coe_attenuation))		err("coe_attenuation is not set!");

		//if(!getparint("smooth_time",&smooth_time))			err("smooth_time is not set!");

		if(!getparfloat("smooth_time_vp",&smooth_time_vp))			err("smooth_time_vp is not set!");///velocity smooth 2017年08月15日 星期二 09时57分09秒 

		if(!getparfloat("smooth_time_vs",&smooth_time_vs))			err("smooth_time_vs is not set!");///velocity smooth 2017年08月15日 星期二 09时57分09秒

		if(!getparfloat("smooth_time_density",&smooth_time_density))	err("smooth_time_density is not set!");///velocity smooth 2017年08月15日 星期二 09时57分09秒

		if(!getparfloat("smooth_time_qp",&smooth_time_qp))			err("smooth_time_qp is not set!");////QQQQQQQ2017年07月30日 星期日 09时23分52秒 

		if(!getparfloat("smooth_time_qs",&smooth_time_qs))			err("smooth_time_qs is not set!");
		
		char *velocity_name;
		if(!getparstring("velocity",&velocity_name))			err("can not read velocity model!");

		char *velocity1_name;
		if(!getparstring("velocity1",&velocity1_name))			err("can not read velocity1 model!");

		char *density_name;
		if(!getparstring("density",&density_name))				err("can not read density model!");

		char *qp_name;
		if(!getparstring("qp_model",&qp_name))				err("can not read qp_model!");////////QQQQ2017年07月27日 星期四 19时47分30秒

		char *qs_name;
		if(!getparstring("qs_model",&qs_name))				err("can not read qs_model!");////////QQQQQ2017年07月27日 星期四 19时47分30秒
		
		//////////2017年03月07日 星期二 21时46分03秒 
		char *s_velocity_name;
		if(smooth_time_vp==0)
		{		
			if(!getparstring("s_velocity",&s_velocity_name))		err("can not read s_velocity model!");

		}///velocity smooth 2017年08月15日 星期二 09时57分09秒

		char *s_velocity1_name;
		if(smooth_time_vs==0)
		{
			if(!getparstring("s_velocity1",&s_velocity1_name))		err("can not read s_velocity1 model!");
		}///velocity smooth 2017年08月15日 星期二 09时57分09秒

		char *s_density_name;
		if(smooth_time_density==0)
		{	
			if(!getparstring("s_density",&s_density_name))		err("can not read s_density model!");
		}///velocity smooth 2017年08月15日 星期二 09时57分09秒

		char *s_qp_name;
		if(smooth_time_qp==0)
		{		
			if(!getparstring("s_qp_model",&s_qp_name))			err("can not read s_qp_name model!");
		}///velocity smooth 2017年08月15日 星期二 09时57分09秒

		char *s_qs_name;
		if(smooth_time_qs==0)
		{
			if(!getparstring("s_qs_model",&s_qs_name))			err("can not read s_qs_name model!");
		}///velocity smooth 2017年08月15日 星期二 09时57分09秒

		if(join_vs==0)					fprintf(logfile,"vs has not been joined\n");
		if(join_vs==1)					fprintf(logfile,"vs has been joined\n");
		if(join_den==0)					fprintf(logfile,"density has not been joined\n");
		if(join_den==1)					fprintf(logfile,"density has been joined\n");
		if(inversion_para==0||inversion_para==1)		fprintf(logfile,"inversion parameter is lame coefficient\n");
		if(inversion_para==2)				fprintf(logfile,"inversion parameter is velocity\n");
		if(inversion_para==3)				fprintf(logfile,"inversion parameter is impedance\n");
		if(inversion_den==0)					fprintf(logfile,"inversion has not density\n");
		if(inversion_den==1)					fprintf(logfile,"inversion has density\n");

		/* creat timing variables on device */
		cudaEvent_t start, stop;
  		cudaEventCreate(&start);	
		cudaEventCreate(&stop);			
//////////2017年03月07日 星期二 21时46分03秒 
		char *outfile_name;
		if(!getparstring("outfile_name",&outfile_name))    	err("can not read outfile_name!");
//////////////We compile this program, if there is no this file, a disadvantage that segmentation fault is not easy to find. But we can read this file and output "warn". Anthoer solution is opening this file in there and close this.

//////////////////////////////////////////////////////input 

		float *wavelet,*wavelet_integral;		
		wavelet=make_ricker_new(freq,dt/1000.0,&wavelet_length);
		wavelet_integral=make_ricker_new(freq,dt/1000.0,&wavelet_length);
		wavelet_half=wavelet_length/2;
		/*wavelet=alloc1float(200);
		wavelet_integral=alloc1float(200);
		set_zero_1d(wavelet,200);
		make_ricker_initial(wavelet,freq,dt,200);
		make_ricker_initial(wavelet_integral,freq,dt,200);
		wavelet_length=200;
		wavelet_half=wavelet_length/2;*/
		warn("Ricker wavelet is set   wavelet_length=%d,wavelet_half=%d\n",wavelet_length,wavelet_half);

		float *coe_opt;
		coe_opt=alloc1float(radius+1);
		make_coe_optimized_new(coe_opt);
		float *coe_opt1;
		coe_opt1=alloc1float(radius+1);
		make_coe_optimized1_new(coe_opt1);
		float  coe_x;
		coe_x=dt/(1000.0*dx);
		float  coe_z;
		coe_z=dt/(1000.0*dz);
		
		if(laplace_compensate!=0)/////////////////////////twice integral
		{
			intergrating_seismic(wavelet_integral,wavelet_length,1);
			intergrating_seismic(wavelet_integral,wavelet_length,1);
			//derivation(wavelet,wavelet_length,coe_opt);
		}

/////////////////////////////smoother
		float *tmp;
		tmp=alloc1float(nx*nz);
		fread_file_1d(tmp,nx,nz,velocity_name);
		//array_max=cpu_caculate_max(velocity,nx,nz);warn("velocity_max=%f\n",array_max); 	 fprintf(logfile,"velocity_max=%f\n",array_max);

		array_max=caculate_average_new(tmp,nx,nz);warn("velocity_average=%f\n",array_max); fprintf(logfile,"velocity_average=%f\n",array_max);
	
		int landa;///velocity smooth 2017年08月15日 星期二 09时57分09秒
		landa=int(array_max/2.0/(freq*1.0)/dx);warn("landa=%d\n",landa);     fprintf(logfile,"landa=%d\n",landa);

		//landa=int(2000.0/2.0*wavelet_length*0.001/dx);warn("landa=%d\n",landa);   fprintf(logfile,"landa=%d\n",landa);

		length_vp=int(landa*smooth_time_vp);
		length_vs=int(landa*smooth_time_vs);
		length_density=int(landa*smooth_time_density);
		length_qp=int(landa*smooth_time_qp);
		length_qs=int(landa*smooth_time_qs);

		landa=max(max(max(max(length_vp,length_vs),length_density),length_qp),length_qs);

		warn("max=%d,length_vp=%d,length_vs=%d,length_density=%d,length_qp=%d,length_qs=%d",
			landa,length_vp,length_vs,length_density,length_qp,length_qs);

		fprintf(logfile,"max=%d,length_vp=%d,length_vs=%d,length_density=%d,length_qp=%d,length_qs=%d\n",
			landa,length_vp,length_vs,length_density,length_qp,length_qs);
	
		if(landa>boundary_up)	boundary_up=landa;	
		if(landa>boundary_down)	boundary_down=landa;
		if(landa>boundary_left)	boundary_left=landa;
		if(landa>boundary_right)	boundary_right=landa;

		fclose(logfile);
		logfile=fopen("log.txt","ab");//remember to free log file	
/////////////////////////////smoother

/////////////////////new  acquisition way 2017年08月16日 星期三 20时31分29秒 
		int *source_x_cord;
		source_x_cord=alloc1int(shot_num);
		for(int is=0;is<shot_num;is++)
				source_x_cord[is]=shot_start+is*shot_interval;

		int *receiver_x_cord;
		receiver_x_cord=alloc1int(shot_num);
		if(0==receiver_offset)
		{
			for(int is=0;is<shot_num;is++)
				receiver_x_cord[is]=receiver_start;
		}
		else
		{
			for(int is=0;is<shot_num;is++)
			{
				receiver_x_cord[is]=source_x_cord[is]-receiver_offset;

				if(receiver_x_cord[is]<0)	receiver_x_cord[is]=0;/////////////////////new  acquisition way 2017年08月16日 星期三 20时31分29秒

				if(receiver_x_cord[is]+receiver_interval*receiver_num>=nx) receiver_x_cord[is]=nx-receiver_interval*receiver_num;//new  acquisition way
			} 
		}

		/*int *acqusition_left,*acqusition_right;
		acqusition_left=alloc1int(shot_num);memset(acqusition_left,0,shot_num*sizeof(int));
		acqusition_right=alloc1int(shot_num);memset(acqusition_right,0,shot_num*sizeof(int));
		if(0!=receiver_offset)
		{
			for(int is=0;is<shot_num;is++)
			{
				if(source_x_cord[is]-receiver_offset<0)	
					acqusition_left[is]=-1*(source_x_cord[is]-receiver_offset);

				//if(acqusition_left[is]!=0)	warn("acqusition_left[%d]=%d\n",is,acqusition_left[is]);

				if(source_x_cord[is]-receiver_offset+receiver_interval*receiver_num>=nx)	
					acqusition_right[is]=source_x_cord[is]-receiver_offset-receiver_x_cord[is];

				//if(acqusition_right[is]!=0)	warn("acqusition_right[%d]=%d\n",is,acqusition_right[is]);	
			}
		}*/

		int *offset_left,*offset_right;
		//int *offset_left_d,*offset_right_d,*source_x_cord_d;
		offset_left=alloc1int(shot_num);				memset(offset_left,0,shot_num*sizeof(int));
		offset_right=alloc1int(shot_num);				memset(offset_right,0,shot_num*sizeof(int));
		//cudaMalloc(&offset_left_d,shot_num*sizeof(int));		cudaMemcpy(offset_left_d,offset_left,shot_num*sizeof(int),cudaMemcpyHostToDevice);
		//cudaMalloc(&offset_right_d,shot_num*sizeof(int));	cudaMemcpy(offset_right_d,offset_right,shot_num*sizeof(int),cudaMemcpyHostToDevice);
		//cudaMalloc(&source_x_cord_d,shot_num*sizeof(int));	cudaMemcpy(source_x_cord_d,source_x_cord,shot_num*sizeof(int),cudaMemcpyHostToDevice);
		if(0!=receiver_offset)
		{
			for(int is=0;is<shot_num;is++)
			{
				offset_left[is]=source_x_cord[is]-receiver_x_cord[is];

				offset_right[is]=receiver_x_cord[is]+receiver_num*receiver_interval-source_x_cord[is];	
			
				//if((is%20)==0)	warn("offset_left[%d]=%d,offset_right[%d]=%d\n",is,offset_left[is],is,offset_right[is]);
			}
		}
		//cudaMalloc(&offset_left_d,shot_num*sizeof(int));		cudaMemcpy(offset_left_d,offset_left,shot_num*sizeof(int),cudaMemcpyHostToDevice);
		//cudaMalloc(&offset_right_d,shot_num*sizeof(int));	cudaMemcpy(offset_right_d,offset_right,shot_num*sizeof(int),cudaMemcpyHostToDevice);
		//cudaMalloc(&source_x_cord_d,shot_num*sizeof(int));	cudaMemcpy(source_x_cord_d,source_x_cord,shot_num*sizeof(int),cudaMemcpyHostToDevice);
/////////////////////new  acquisition way 2017年08月16日 星期三 20时31分29秒  
/////////////////////new  acquisition way 2017年08月16日 星期三 20时31分29秒  
		int *imaging_start,*imaging_size,*imaging_end,nx_size;
		imaging_start=alloc1int(shot_num);
		imaging_size=alloc1int(shot_num);
		imaging_end=alloc1int(shot_num);
		for(int is=0;is<shot_num;is++)
		{
			imaging_start[is]=receiver_x_cord[is];
			imaging_end[is]=receiver_x_cord[is]+receiver_interval*receiver_num;
			imaging_size[is]=imaging_end[is]-imaging_start[is];
			nx_size=imaging_size[0];
		}
			
		if(nx_size==nx)
		{
			nx_append=nx+boundary_left+boundary_right;
			nz_append=nz+boundary_up+boundary_down;
		}
		if(nx_size!=nx)
		{
			nx_append=nx_size+boundary_left+boundary_right;
			nz_append=nz+boundary_up+boundary_down;
		}			
/////////////////////new  acquisition way 2017年08月16日 星期三 20时31分29秒  

///////////////////////for vsp.............2017年03月14日 星期二 08时32分25秒 
		int *source_z_cord;
		source_z_cord=alloc1int(shot_num);
		for(int is=0;is<shot_num;is++)
				source_z_cord[is]=shot_depth+is*shot_z_interval;

		int *receiver_z_cord;
		receiver_z_cord=alloc1int(shot_num);
		for(int is=0;is<shot_num;is++)
				receiver_z_cord[is]=receiver_depth;
///////////////////////for vsp.............2017年03月14日 星期二 08时32分25秒

///////////////////////for vsp2222222222222222222
		//if(vsp_2!=0)
		//{
			int *receiver_x_cord_2;
			receiver_x_cord_2=alloc1int(shot_num);	
				for(int is=0;is<shot_num;is++)
					receiver_x_cord_2[is]=receiver_start_2;
			int *receiver_z_cord_2;
			receiver_z_cord_2=alloc1int(shot_num);
				for(int is=0;is<shot_num;is++)
					receiver_z_cord_2[is]=receiver_depth_2;
		//}
///////////////////////for vsp2222222222222222222
		
		nx_append_new=nx+boundary_left+boundary_right;
		nxnz=nx*nz;////////////////////////////////////////all mode
		nxa_new_nza=nx_append_new*nz_append;//////////all model +boundary			

		nx_size_nz=nx_size*nz;////////////////////////////////calculated mode
		nxanza=nx_append*nz_append;/////////////////////calculated mode+boundary

		lt_rec=lt*receiver_num;		

		warn("nxnz=%d,nx=%d,nz=%d,dx=%f,dz=%f",nxnz,nx,nz,dx,dz);
		warn("lt=%d,dt=%f,freq=%f",lt,dt,freq);
		warn("shot_num=%d,shot_start=%d",shot_num,shot_start);
		warn("shot_interval=%d,shot_depth=%d",shot_interval,shot_depth);
		warn("receiver_num=%d,receiver_start=%d",receiver_num,receiver_start);
		warn("receiver_interval=%d,receiver_depth=%d,receiver_offset=%d",receiver_interval,receiver_depth,receiver_offset);
		warn("coe_attenuation=%f",coe_attenuation);
		warn("nx_append=%d,nz_append=%d",nx_append,nz_append);
		
		
		float *shotgather,*shotgather1,*wf_shot;
		shotgather=alloc1float(lt_rec);		memset((void *) (shotgather), 0, lt_rec * sizeof (float));
		shotgather1=alloc1float(lt_rec);		memset((void *) (shotgather1), 0, lt_rec * sizeof (float));
		wf_shot=alloc1float(lt_rec);			memset((void *) (wf_shot), 0, lt_rec * sizeof (float));

		float *wf_append,*wf;
		wf_append=alloc1float(nxanza);		memset((void *) (wf_append), 0, nxanza * sizeof (float));

		wf=alloc1float(nx_size_nz);			memset((void *) (wf), 0, nx_size_nz * sizeof (float));

		float *wf_append_new,*wf_nxnz;
		wf_append_new=alloc1float(nxa_new_nza);	memset((void *) (wf_append_new), 0, nxa_new_nza * sizeof (float));
								//memset((void *) (wf_append), 0, nxa_new_nza * sizeof (float));
		wf_nxnz=alloc1float(nxnz);			memset((void *) (wf_nxnz), 0, nxnz * sizeof (float));

//////////////////parameters
		float *attenuation;		
		
		attenuation=alloc1float(nxanza);		memset((void *) (attenuation), 0, nxanza * sizeof (float));
		make_attenuation_new(attenuation,nx_size,nz,boundary_up,boundary_down,boundary_left,boundary_right,coe_attenuation);
///////////////parameters
//////////////fread_file_1d all  parameters
		float *velocity_all,*velocity1_all,*density_all,*qp_all,*qs_all;
		velocity_all=alloc1float(nxa_new_nza);	
		velocity1_all=alloc1float(nxa_new_nza);
		density_all=alloc1float(nxa_new_nza);
		qp_all=alloc1float(nxa_new_nza);
		qs_all=alloc1float(nxa_new_nza);
///////////////////for vp
					read_velocity_new(velocity_all,nx,nz,boundary_up,boundary_down,boundary_left,boundary_right,velocity_name);//read_real_vp
	
///////////////////for vs
		if(join_vs==1)	read_velocity_new(velocity1_all,nx,nz,boundary_up,boundary_down,boundary_left,boundary_right,velocity1_name);//read_real_vs

		else			vp_set_vs(velocity_all,velocity1_all,nx_append_new,nz_append);//set_real_vs by the  relation with vp

///////////////////for density
		if(join_den==1)	read_velocity_new(density_all,nx,nz,boundary_up,boundary_down,boundary_left,boundary_right,density_name);//read_real_density

		else			vp_set_density(velocity_all,density_all,nx_append_new,nz_append);//set_real_density by the  relation with vp

///////////////////for qp
		if(modeling_type!=0||migration_type!=0)
				{ 
					read_velocity_new(qp_all,nx,nz,boundary_up,boundary_down,boundary_left,boundary_right,qp_name);	
///////////////////for qs 
					read_velocity_new(qs_all,nx,nz,boundary_up,boundary_down,boundary_left,boundary_right,qs_name);
				}
////output nxanza size:
		write_file_1d(wavelet,wavelet_length,"./someoutput/wavelet2.bin");
		write_file_1d(wavelet_integral,wavelet_length,"./someoutput/wavelet3.bin");

		write_file_1d(attenuation,nxanza,"./someoutput/att.bin");

		write_file_1d(velocity_all,nxa_new_nza,"./someoutput/vp.bin");
	
		write_file_1d(velocity1_all,nxa_new_nza,"./someoutput/vs.bin");		

		write_file_1d(density_all,nxa_new_nza,"./someoutput/density.bin");		

		write_file_1d(qp_all,nxa_new_nza,"./someoutput/qp.bin");

		write_file_1d(qs_all,nxa_new_nza,"./someoutput/qs.bin");
////output nxanza size:
	
////output nxnz size:
		exchange(velocity_all,wf_nxnz,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up);
		write_file_1d(wf_nxnz,nxnz,"./someoutput/cut-vp.bin");
		
		exchange(velocity1_all,wf_nxnz,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up);
		write_file_1d(wf_nxnz,nxnz,"./someoutput/cut-vs.bin");
			
		exchange(density_all,wf_nxnz,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up);
		write_file_1d(wf_nxnz,nxnz,"./someoutput/cut-density.bin");

		exchange(qp_all,wf_nxnz,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up);
		write_file_1d(wf_nxnz,nxnz,"./someoutput/cut-qp.bin");
			
		exchange(qs_all,wf_nxnz,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up);
		write_file_1d(wf_nxnz,nxnz,"./someoutput/cut-qs.bin");

		warn("Real_Velocity and Attenuation has been read!");
////output nxnz size:

		int nx_append_radius=nx_append-2*radius;//chu qu liangbian jie shu  
		int nz_append_radius=nz_append-2*radius;		

		dim3 dimBlock(32,16);

		dim3 dimGrid((nx_append+dimBlock.x-1)/dimBlock.x,(nz_append+dimBlock.y-1)/dimBlock.y);////cal

		dim3 dimGrid_new((nx_append_new+dimBlock.x-1)/dimBlock.x,(nz_append+dimBlock.y-1)/dimBlock.y);///all
		
		dim3 dimGrid_3nx_nz((3*nx+dimBlock.x-1)/dimBlock.x,(nz+dimBlock.y-1)/dimBlock.y);///for conjugated method

		dim3 dimGrid_3nx_size_nz((3*nx_size+dimBlock.x-1)/dimBlock.x,(nz+dimBlock.y-1)/dimBlock.y);///for conjugated method

		dim3 dimGrid_nx_nz((nx+dimBlock.x-1)/dimBlock.x,(nz+dimBlock.y-1)/dimBlock.y);

		//dim3 trans_dimGrid((nz_append+dimBlock.y-1)/dimBlock.y,(nx_append+dimBlock.x-1)/dimBlock.x);//smooth

		dim3 dimGrid_lt((receiver_num+dimBlock.x-1)/dimBlock.x,(lt+dimBlock.y-1)/dimBlock.y);///some operation on obs/cal seismic data

		int numofblock=((nx_append_radius+dimBlock.x-1)/dimBlock.x)*((nz_append_radius+dimBlock.y-1)/dimBlock.y);
		warn("num of block in x direction=%d",(nx_append_radius+dimBlock.x-1)/dimBlock.x);
		warn("num of block in z direction=%d",(nz_append_radius+dimBlock.y-1)/dimBlock.y);
		warn("num of block in total direction=%d",numofblock);
		
//////////////////////////////////////////correct velocity and smooth velocity2017年08月17日 星期四 08时40分12秒 
		float *velocity_all_d,*velocity1_all_d,*density_all_d,*qp_all_d,*qs_all_d;
		gpumem += (nxa_new_nza*5)*sizeof(float)/1024.0/1024.0;/////for check 2017年07月27日 星期四 10时11分40秒
		cudaMalloc(&velocity_all_d,nxa_new_nza*sizeof(float));	cudaMemcpy(velocity_all_d,velocity_all,nxa_new_nza*sizeof(float),cudaMemcpyHostToDevice);
		cudaMalloc(&velocity1_all_d,nxa_new_nza*sizeof(float));	cudaMemcpy(velocity1_all_d,velocity1_all,nxa_new_nza*sizeof(float),cudaMemcpyHostToDevice);
		cudaMalloc(&density_all_d,nxa_new_nza*sizeof(float));	cudaMemcpy(density_all_d,density_all,nxa_new_nza*sizeof(float),cudaMemcpyHostToDevice);
		cudaMalloc(&qp_all_d,nxa_new_nza*sizeof(float));		cudaMemcpy(qp_all_d,qp_all,nxa_new_nza*sizeof(float),cudaMemcpyHostToDevice);
		cudaMalloc(&qs_all_d,nxa_new_nza*sizeof(float));		cudaMemcpy(qs_all_d,qs_all,nxa_new_nza*sizeof(float),cudaMemcpyHostToDevice);

		float *s_velocity_all_d,*s_velocity1_all_d,*s_density_all_d,*s_qp_all_d,*s_qs_all_d;
		gpumem += (nxa_new_nza*5)*sizeof(float)/1024.0/1024.0;/////for check 2017年07月27日 星期四 10时11分40秒
		cudaMalloc(&s_velocity_all_d,nxa_new_nza*sizeof(float));	cudaMemcpy(s_velocity_all_d,velocity_all,nxa_new_nza*sizeof(float),cudaMemcpyHostToDevice);
		cudaMalloc(&s_velocity1_all_d,nxa_new_nza*sizeof(float));	cudaMemcpy(s_velocity1_all_d,velocity1_all,nxa_new_nza*sizeof(float),cudaMemcpyHostToDevice);
		cudaMalloc(&s_density_all_d,nxa_new_nza*sizeof(float));	cudaMemcpy(s_density_all_d,density_all,nxa_new_nza*sizeof(float),cudaMemcpyHostToDevice);
		cudaMalloc(&s_qp_all_d,nxa_new_nza*sizeof(float));		cudaMemcpy(s_qp_all_d,qp_all,nxa_new_nza*sizeof(float),cudaMemcpyHostToDevice);
		cudaMalloc(&s_qs_all_d,nxa_new_nza*sizeof(float));		cudaMemcpy(s_qs_all_d,qs_all,nxa_new_nza*sizeof(float),cudaMemcpyHostToDevice);

		float *wf_d,*wf_append_d,*wf_nxnz_d,*wf_append_new_d;
		gpumem += (nxa_new_nza*2)*sizeof(float)/1024.0/1024.0;/////for check 2017年07月27日 星期四 10时11分40秒
		cudaMalloc(&wf_d,nx_size_nz*sizeof(float));		cudaMemset(wf_d,0,nx_size_nz*sizeof(float));
		cudaMalloc(&wf_append_d,nxanza*sizeof(float));		cudaMemset(wf_append_d,0,nxanza*sizeof(float));

		//cudaMalloc(&wf_nxnz_d,nx_size_nz*sizeof(float));		cudaMemset(wf_nxnz_d,0,nxnz*sizeof(float));
		cudaMalloc(&wf_nxnz_d,nxnz*sizeof(float));			cudaMemset(wf_nxnz_d,0,nxnz*sizeof(float));
		cudaMalloc(&wf_append_new_d,nxa_new_nza*sizeof(float));	cudaMemset(wf_append_new_d,0,nxa_new_nza*sizeof(float));

			////////////////////////////////QQQQQQQQQQQQQQQ2017年07月27日 星期四 10时11分40秒 
			if(length_vp!=0)///////////vp
			{
				//cuda_bell_smoothx_new<<< dimGrid_new,dimBlock>>>(s_velocity_all_d,wf_append_new_d,length_vp,nx_append_new,nz_append);
				//cuda_bell_smoothz_new<<< dimGrid_new,dimBlock>>>(wf_append_new_d,s_velocity_all_d,length_vp,nx_append_new,nz_append);

				cuda_bell_smooth_2d<<< dimGrid_new,dimBlock>>>(s_velocity_all_d,wf_append_new_d,length_vp,nx_append_new,nz_append);	
				cudaMemcpy(s_velocity_all_d,wf_append_new_d,nxa_new_nza*sizeof(float),cudaMemcpyDeviceToDevice);
			}
				cuda_cal_window<<<dimGrid_new,dimBlock>>>(s_velocity_all_d,wf_nxnz_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up);
				cuda_cal_expand<<<dimGrid_new,dimBlock>>>(s_velocity_all_d,wf_nxnz_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up);

			if(length_vs!=0)///////////vs
			{
				//cuda_bell_smoothx_new<<< dimGrid_new,dimBlock>>>(s_velocity1_all_d,wf_append_new_d,length_vs,nx_append_new,nz_append);
				//cuda_bell_smoothz_new<<< dimGrid_new,dimBlock>>>(wf_append_new_d,s_velocity1_all_d,length_vs,nx_append_new,nz_append);

				cuda_bell_smooth_2d<<< dimGrid_new,dimBlock>>>(s_velocity1_all_d,wf_append_new_d,length_vs,nx_append_new,nz_append);	
				cudaMemcpy(s_velocity1_all_d,wf_append_new_d,nxa_new_nza*sizeof(float),cudaMemcpyDeviceToDevice);	
			}
				cuda_cal_window<<<dimGrid_new,dimBlock>>>(s_velocity1_all_d,wf_nxnz_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up);
				cuda_cal_expand<<<dimGrid_new,dimBlock>>>(s_velocity1_all_d,wf_nxnz_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up);

			if(length_density!=0)///////////density
			{
				//cuda_bell_smoothx_new<<< dimGrid_new,dimBlock>>>(s_density_all_d,wf_append_new_d,length_density,nx_append_new,nz_append);
				//cuda_bell_smoothz_new<<< dimGrid_new,dimBlock>>>(wf_append_new_d,s_density_all_d,length_density,nx_append_new,nz_append);

				cuda_bell_smooth_2d<<< dimGrid_new,dimBlock>>>(s_density_all_d,wf_append_new_d,length_density,nx_append_new,nz_append);	
				cudaMemcpy(s_density_all_d,wf_append_new_d,nxa_new_nza*sizeof(float),cudaMemcpyDeviceToDevice);	
			}
				cuda_cal_window<<<dimGrid_new,dimBlock>>>(s_density_all_d,wf_nxnz_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up);
				cuda_cal_expand<<<dimGrid_new,dimBlock>>>(s_density_all_d,wf_nxnz_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up);

			if(length_qp!=0)///////////qp
			{
				//cuda_bell_smoothx_new<<< dimGrid_new,dimBlock>>>(s_qp_all_d,wf_append_new_d,length_qp,nx_append_new,nz_append);
				//cuda_bell_smoothz_new<<< dimGrid_new,dimBlock>>>(wf_append_new_d,s_qp_all_d,length_qp,nx_append_new,nz_append);

				cuda_bell_smooth_2d<<< dimGrid_new,dimBlock>>>(s_qp_all_d,wf_append_new_d,length_qp,nx_append_new,nz_append);	
				cudaMemcpy(s_qp_all_d,wf_append_new_d,nxa_new_nza*sizeof(float),cudaMemcpyDeviceToDevice);		
			}
				cuda_cal_window<<<dimGrid_new,dimBlock>>>(s_qp_all_d,wf_nxnz_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up);
				cuda_cal_expand<<<dimGrid_new,dimBlock>>>(s_qp_all_d,wf_nxnz_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up);

			if(length_qs!=0)///////////qs
			{
				//cuda_bell_smoothx_new<<< dimGrid_new,dimBlock>>>(s_qs_all_d,wf_append_new_d,length_qs,nx_append_new,nz_append);
				//cuda_bell_smoothz_new<<< dimGrid_new,dimBlock>>>(wf_append_new_d,s_qs_all_d,length_qs,nx_append_new,nz_append);

				cuda_bell_smooth_2d<<< dimGrid_new,dimBlock>>>(s_qs_all_d,wf_append_new_d,length_qs,nx_append_new,nz_append);	
				cudaMemcpy(s_qs_all_d,wf_append_new_d,nxa_new_nza*sizeof(float),cudaMemcpyDeviceToDevice);	
			}
				cuda_cal_window<<<dimGrid_new,dimBlock>>>(s_qs_all_d,wf_nxnz_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up);
				cuda_cal_expand<<<dimGrid_new,dimBlock>>>(s_qs_all_d,wf_nxnz_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up);

				cudaMemcpy(wf_append_new,s_velocity_all_d,nxa_new_nza*sizeof(float),cudaMemcpyDeviceToHost);
				write_file_1d(wf_append_new,nxa_new_nza,"./someoutput/vp-s.bin");
				exchange(wf_append_new,wf_nxnz,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up);
				write_file_1d(wf_nxnz,nxnz,"./someoutput/cut-vp-s.bin");

				cudaMemcpy(wf_append_new,s_velocity1_all_d,nxa_new_nza*sizeof(float),cudaMemcpyDeviceToHost);
				write_file_1d(wf_append_new,nxa_new_nza,"./someoutput/vs-s.bin");
				exchange(wf_append_new,wf_nxnz,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up);
				write_file_1d(wf_nxnz,nxnz,"./someoutput/cut-vs-s.bin");

				cudaMemcpy(wf_append_new,s_density_all_d,nxa_new_nza*sizeof(float),cudaMemcpyDeviceToHost);
				write_file_1d(wf_append_new,nxa_new_nza,"./someoutput/density-s.bin");
				exchange(wf_append_new,wf_nxnz,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up);
				write_file_1d(wf_nxnz,nxnz,"./someoutput/cut-density-s.bin");

				cudaMemcpy(wf_append_new,s_qp_all_d,nxa_new_nza*sizeof(float),cudaMemcpyDeviceToHost);
				write_file_1d(wf_append_new,nxa_new_nza,"./someoutput/qp-s.bin");
				exchange(wf_append_new,wf_nxnz,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up);
				write_file_1d(wf_nxnz,nxnz,"./someoutput/cut-qp-s.bin");

				cudaMemcpy(wf_append_new,s_qs_all_d,nxa_new_nza*sizeof(float),cudaMemcpyDeviceToHost);
				write_file_1d(wf_append_new,nxa_new_nza,"./someoutput/qs-s.bin");
				exchange(wf_append_new,wf_nxnz,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up);
				write_file_1d(wf_nxnz,nxnz,"./someoutput/cut-qs-s.bin");

				warn("smooth velocity has been gotten!");
//////////////////////////////////////////correct velocity and smooth velocity2017年08月17日 星期四 08时40分12秒	
		
		float *qp_d,*qs_d,*s_qp_d,*s_qs_d,*tao_d,*strain_p_d,*strain_s_d,*modul_p_d,*modul_s_d;///////////////QQQQQQQQQQQQQQQ2017年07月27日 星期四 10时11分40秒 
		float *mem_p1_d,*mem_p2_d,*mem_xx1_d,*mem_xx2_d,*mem_zz1_d,*mem_zz2_d,*mem_xz1_d,*mem_xz2_d;
		float *rmem_p1_d,*rmem_p2_d,*rmem_xx1_d,*rmem_xx2_d,*rmem_zz1_d,*rmem_zz2_d,*rmem_xz1_d,*rmem_xz2_d;
		cudaMalloc(&qp_d,nxanza*sizeof(float));	//cudaMemcpy(qp_d,qp,nxanza*sizeof(float),cudaMemcpyHostToDevice);
		cudaMalloc(&qs_d,nxanza*sizeof(float));	//cudaMemcpy(qs_d,qs,nxanza*sizeof(float),cudaMemcpyHostToDevice);
		cudaMalloc(&s_qp_d,nxanza*sizeof(float));	//cudaMemcpy(s_qp_d,s_qp,nxanza*sizeof(float),cudaMemcpyHostToDevice);
		cudaMalloc(&s_qs_d,nxanza*sizeof(float));	//cudaMemcpy(s_qs_d,s_qs,nxanza*sizeof(float),cudaMemcpyHostToDevice);
		cudaMalloc(&tao_d,nxanza*sizeof(float));
		cudaMalloc(&strain_p_d,nxanza*sizeof(float));
		cudaMalloc(&strain_s_d,nxanza*sizeof(float));
		cudaMalloc(&modul_p_d,nxanza*sizeof(float));
		cudaMalloc(&modul_s_d,nxanza*sizeof(float));///////////////QQQQQQQQQQQQQQQ2017年07月27日 星期四 10时11分40秒 

		cudaMalloc(&mem_p1_d,nxanza*sizeof(float));
		cudaMalloc(&mem_p2_d,nxanza*sizeof(float));		
		cudaMalloc(&mem_xx1_d,nxanza*sizeof(float));
		cudaMalloc(&mem_xx2_d,nxanza*sizeof(float));
		cudaMalloc(&mem_zz1_d,nxanza*sizeof(float));
		cudaMalloc(&mem_zz2_d,nxanza*sizeof(float));
		cudaMalloc(&mem_xz1_d,nxanza*sizeof(float));
		cudaMalloc(&mem_xz2_d,nxanza*sizeof(float));

		cudaMalloc(&rmem_p1_d,nxanza*sizeof(float));
		cudaMalloc(&rmem_p2_d,nxanza*sizeof(float));		
		cudaMalloc(&rmem_xx1_d,nxanza*sizeof(float));
		cudaMalloc(&rmem_xx2_d,nxanza*sizeof(float));
		cudaMalloc(&rmem_zz1_d,nxanza*sizeof(float));
		cudaMalloc(&rmem_zz2_d,nxanza*sizeof(float));
		cudaMalloc(&rmem_xz1_d,nxanza*sizeof(float));
		cudaMalloc(&rmem_xz2_d,nxanza*sizeof(float));///////////////QQQQQQQQQQQQQQQ2017年07月27日 星期四 10时11分40秒 

		gpumem += (nxanza*23)*sizeof(float)/1024.0/1024.0;/////for check 2017年07月27日 星期四 10时11分40秒

////////////////////vsp 3.17
		//if(vsp_2!=0)
		//{
			float *obs_shot_x_d_2,*obs_shot_z_d_2,*cal_shot_x_d_2,*cal_shot_z_d_2,*res_shot_x_d_2,*res_shot_z_d_2;
			cudaMalloc(&obs_shot_x_d_2,lt_rec*sizeof(float));		cudaMemset(obs_shot_x_d_2,0,lt_rec*sizeof(float));
			cudaMalloc(&obs_shot_z_d_2,lt_rec*sizeof(float));		cudaMemset(obs_shot_z_d_2,0,lt_rec*sizeof(float));
			cudaMalloc(&cal_shot_x_d_2,lt_rec*sizeof(float));		cudaMemset(cal_shot_x_d_2,0,lt_rec*sizeof(float));
			cudaMalloc(&cal_shot_z_d_2,lt_rec*sizeof(float));		cudaMemset(cal_shot_z_d_2,0,lt_rec*sizeof(float));
			cudaMalloc(&res_shot_x_d_2,lt_rec*sizeof(float));		cudaMemset(res_shot_x_d_2,0,lt_rec*sizeof(float));
			cudaMalloc(&res_shot_z_d_2,lt_rec*sizeof(float));		cudaMemset(res_shot_z_d_2,0,lt_rec*sizeof(float));
		//}
////////////////////vsp 3.17
		float *wavelet_d,*coe_opt_d,*coe_opt1_d,*s_velocity_d,*s_velocity1_d,*velocity_d,*velocity1_d,*attenuation_d,*density_d,*s_density_d;
		float *vx1_d,*vz1_d,*txx1_d,*tzz1_d,*txz1_d,*vx2_d,*vz2_d,*txx2_d,*tzz2_d,*txz2_d;
		float *rvx1_d,*rvz1_d,*rtxx1_d,*rtzz1_d,*rtxz1_d,*rvx2_d,*rvz2_d,*rtxx2_d,*rtzz2_d,*rtxz2_d;
		float *vx_t_d,*vz_t_d;///////vx of the direvation of time  vz of the direvation of time
		float *obs_shot_x_d,*obs_shot_z_d,*cal_shot_x_d,*cal_shot_z_d,*res_shot_x_d,*res_shot_z_d,*obs_shot_all_d,*cal_shot_all_d,*res_shot_all_d;
		float *cal_shot_x1_d,*cal_shot_z1_d,*obs_shot_x1_d,*obs_shot_z1_d,*res_shot_x1_d,*res_shot_z1_d,*res_shot_x2_d,*res_shot_z2_d;

		float *tmp_shot_x_d,*tmp_shot_z_d;
		float *adj_shot_x_d,*adj_shot_z_d;
		float *correlation_parameter_d;
		float *obj_parameter_d;
		//////////////tmp_shot_x_d:the sum of cal_shot in previous iteration for cross-correlation misfunction 2017年08月25日 星期五 09时28分54秒 

		cudaMalloc(&cal_max,1*sizeof(float));
		cudaMalloc(&obs_max,1*sizeof(float));
		cudaMalloc(&wavelet_d,wavelet_length*sizeof(float));
		cudaMalloc(&coe_opt_d,(radius+1)*sizeof(float));
		cudaMalloc(&coe_opt1_d,(radius+1)*sizeof(float));
		cudaMalloc(&velocity_d,nxanza*sizeof(float));
		cudaMalloc(&velocity1_d,nxanza*sizeof(float));
		cudaMalloc(&s_velocity_d,nxanza*sizeof(float));
		cudaMalloc(&s_velocity1_d,nxanza*sizeof(float));
		cudaMalloc(&attenuation_d,nxanza*sizeof(float));
		cudaMalloc(&density_d,nxanza*sizeof(float));
		cudaMalloc(&s_density_d,nxanza*sizeof(float));
		cudaMalloc(&vx1_d,nxanza*sizeof(float));
		cudaMalloc(&vz1_d,nxanza*sizeof(float));
		cudaMalloc(&txx1_d,nxanza*sizeof(float));
		cudaMalloc(&tzz1_d,nxanza*sizeof(float));
		cudaMalloc(&txz1_d,nxanza*sizeof(float));
		cudaMalloc(&vx2_d,nxanza*sizeof(float));
		cudaMalloc(&vz2_d,nxanza*sizeof(float));
		cudaMalloc(&txx2_d,nxanza*sizeof(float));
		cudaMalloc(&tzz2_d,nxanza*sizeof(float));
		cudaMalloc(&txz2_d,nxanza*sizeof(float));
		cudaMalloc(&rvx1_d,nxanza*sizeof(float));
		cudaMalloc(&rvz1_d,nxanza*sizeof(float));
		cudaMalloc(&rtxx1_d,nxanza*sizeof(float));
		cudaMalloc(&rtzz1_d,nxanza*sizeof(float));
		cudaMalloc(&rtxz1_d,nxanza*sizeof(float));
		cudaMalloc(&rvx2_d,nxanza*sizeof(float));
		cudaMalloc(&rvz2_d,nxanza*sizeof(float));
		cudaMalloc(&rtxx2_d,nxanza*sizeof(float));
		cudaMalloc(&rtzz2_d,nxanza*sizeof(float));
		cudaMalloc(&rtxz2_d,nxanza*sizeof(float));
		cudaMalloc(&vx_t_d,nxanza*sizeof(float));
		cudaMalloc(&vz_t_d,nxanza*sizeof(float));
		/*this is a falut smilar with free space happen error */
		cudaMalloc(&obs_shot_x_d,lt_rec*sizeof(float));			cudaMemset(obs_shot_x_d,0,lt_rec*sizeof(float));
		cudaMalloc(&obs_shot_z_d,lt_rec*sizeof(float));			cudaMemset(obs_shot_z_d,0,lt_rec*sizeof(float));
		cudaMalloc(&cal_shot_x_d,lt_rec*sizeof(float));			cudaMemset(cal_shot_x_d,0,lt_rec*sizeof(float));
		cudaMalloc(&cal_shot_z_d,lt_rec*sizeof(float));			cudaMemset(cal_shot_z_d,0,lt_rec*sizeof(float));
		cudaMalloc(&res_shot_x_d,lt_rec*sizeof(float));			cudaMemset(res_shot_x_d,0,lt_rec*sizeof(float));
		cudaMalloc(&res_shot_z_d,lt_rec*sizeof(float));			cudaMemset(res_shot_z_d,0,lt_rec*sizeof(float));
		cudaMalloc(&res_shot_x1_d,lt_rec*sizeof(float));			cudaMemset(res_shot_x1_d,0,lt_rec*sizeof(float));
		cudaMalloc(&res_shot_z1_d,lt_rec*sizeof(float));			cudaMemset(res_shot_z1_d,0,lt_rec*sizeof(float));
		cudaMalloc(&res_shot_x2_d,lt_rec*sizeof(float));			cudaMemset(res_shot_x2_d,0,lt_rec*sizeof(float));
		cudaMalloc(&res_shot_z2_d,lt_rec*sizeof(float));			cudaMemset(res_shot_z2_d,0,lt_rec*sizeof(float));
		cudaMalloc(&obs_shot_all_d,lt_rec*sizeof(float));			cudaMemset(obs_shot_all_d,0,lt_rec*sizeof(float));
		cudaMalloc(&cal_shot_all_d,lt_rec*sizeof(float));			cudaMemset(cal_shot_all_d,0,lt_rec*sizeof(float));
		cudaMalloc(&res_shot_all_d,lt_rec*sizeof(float));			cudaMemset(res_shot_all_d,0,lt_rec*sizeof(float));
		cudaMalloc(&cal_shot_x1_d,lt_rec*sizeof(float));			cudaMemset(cal_shot_x1_d,0,lt_rec*sizeof(float));
		cudaMalloc(&cal_shot_z1_d,lt_rec*sizeof(float));			cudaMemset(cal_shot_z1_d,0,lt_rec*sizeof(float));
		cudaMalloc(&obs_shot_x1_d,lt_rec*sizeof(float));			cudaMemset(obs_shot_x1_d,0,lt_rec*sizeof(float));
		cudaMalloc(&obs_shot_z1_d,lt_rec*sizeof(float));			cudaMemset(obs_shot_z1_d,0,lt_rec*sizeof(float));

		cudaMalloc(&tmp_shot_x_d,lt_rec*sizeof(float));			cudaMemset(tmp_shot_x_d,0,lt_rec*sizeof(float));
		cudaMalloc(&tmp_shot_z_d,lt_rec*sizeof(float));			cudaMemset(tmp_shot_z_d,0,lt_rec*sizeof(float));
		cudaMalloc(&adj_shot_x_d,lt_rec*sizeof(float));			cudaMemset(adj_shot_x_d,0,lt_rec*sizeof(float));
		cudaMalloc(&adj_shot_z_d,lt_rec*sizeof(float));			cudaMemset(adj_shot_z_d,0,lt_rec*sizeof(float));
		cudaMalloc(&correlation_parameter_d,10*sizeof(float));		cudaMemset(correlation_parameter_d,0,10*sizeof(float));
		cudaMalloc(&obj_parameter_d,3*sizeof(float));			cudaMemset(obj_parameter_d,0,3*sizeof(float));//for cross-correlation misfunction
		
		cudaMemcpy(wavelet_d,wavelet,wavelet_length*sizeof(float),cudaMemcpyHostToDevice);
		cudaMemcpy(coe_opt_d,coe_opt,(radius+1)*sizeof(float),cudaMemcpyHostToDevice);
		cudaMemcpy(coe_opt1_d,coe_opt1,(radius+1)*sizeof(float),cudaMemcpyHostToDevice);
		//cudaMemcpy(velocity_d,velocity,nxanza*sizeof(float),cudaMemcpyHostToDevice);
		//cudaMemcpy(s_velocity_d,s_velocity,nxanza*sizeof(float),cudaMemcpyHostToDevice);	
		//cudaMemcpy(density_d,density,nxanza*sizeof(float),cudaMemcpyHostToDevice);
		//cudaMemcpy(s_density_d,s_density,nxanza*sizeof(float),cudaMemcpyHostToDevice);
		//cudaMemcpy(velocity1_d,velocity1,nxanza*sizeof(float),cudaMemcpyHostToDevice);
		//cudaMemcpy(s_velocity1_d,s_velocity1,nxanza*sizeof(float),cudaMemcpyHostToDevice);
		cudaMemcpy(attenuation_d,attenuation,nxanza*sizeof(float),cudaMemcpyHostToDevice);


		gpumem += (nxanza*48)*sizeof(float)/1024.0/1024.0;/////for check 2017年07月27日 星期四 10时11分40秒
		gpumem += (lt_rec*4)*sizeof(float)/1024.0/1024.0;/////for check 2017年07月27日 星期四 10时11分40秒

		/////////////////when we used viscoelastic modeling or migration, we don't perform wavefield construction. why???
		/*float *vxu_d,*vxd_d,*vxl_d,*vxr_d,*vzu_d,*vzd_d,*vzl_d,*vzr_d;
		cudaMalloc(&vxu_d,radius*nx_append*lt*sizeof(float));cudaMemset(vxu_d,0,radius*nx_append*lt*sizeof(float));
		cudaMalloc(&vxd_d,radius*nx_append*lt*sizeof(float));cudaMemset(vxd_d,0,radius*nx_append*lt*sizeof(float));
		cudaMalloc(&vzu_d,radius*nx_append*lt*sizeof(float));cudaMemset(vzu_d,0,radius*nx_append*lt*sizeof(float));
		cudaMalloc(&vzd_d,radius*nx_append*lt*sizeof(float));cudaMemset(vzd_d,0,radius*nx_append*lt*sizeof(float));
		cudaMalloc(&vxl_d,radius*nz_append*lt*sizeof(float));cudaMemset(vxl_d,0,radius*nz_append*lt*sizeof(float));
		cudaMalloc(&vxr_d,radius*nz_append*lt*sizeof(float));cudaMemset(vxr_d,0,radius*nz_append*lt*sizeof(float));
		cudaMalloc(&vzl_d,radius*nz_append*lt*sizeof(float));cudaMemset(vzl_d,0,radius*nz_append*lt*sizeof(float));
		cudaMalloc(&vzr_d,radius*nz_append*lt*sizeof(float));cudaMemset(vzr_d,0,radius*nz_append*lt*sizeof(float));*/ 

////derivation for vx or vz
		float *vx_x_d,*vz_z_d,*vx_z_d,*vz_x_d;
		cudaMalloc(&vx_x_d,nxanza*sizeof(float));cudaMemset(vx_x_d,0,nxanza*sizeof(float));
		cudaMalloc(&vz_z_d,nxanza*sizeof(float));cudaMemset(vz_z_d,0,nxanza*sizeof(float));
		cudaMalloc(&vx_z_d,nxanza*sizeof(float));cudaMemset(vx_z_d,0,nxanza*sizeof(float));
		cudaMalloc(&vz_x_d,nxanza*sizeof(float));cudaMemset(vz_x_d,0,nxanza*sizeof(float));

////gradient or conjugate direction
///////////////2017年03月14日 星期二 20时21分13秒 波场分离的LSRTM
		float *tp2_d,*tp1_d,*vxp2_d,*vxp1_d,*vzp2_d,*vzp1_d,*vxs2_d,*vxs1_d,*vzs2_d,*vzs1_d;
		cudaMalloc(&tp2_d,nxanza*sizeof(float));
		cudaMalloc(&tp1_d,nxanza*sizeof(float));
		cudaMalloc(&vxp2_d,nxanza*sizeof(float));
		cudaMalloc(&vxp1_d,nxanza*sizeof(float));
		cudaMalloc(&vzp2_d,nxanza*sizeof(float));
		cudaMalloc(&vzp1_d,nxanza*sizeof(float));
		cudaMalloc(&vxs2_d,nxanza*sizeof(float));
		cudaMalloc(&vxs1_d,nxanza*sizeof(float));
		cudaMalloc(&vzs2_d,nxanza*sizeof(float));
		cudaMalloc(&vzs1_d,nxanza*sizeof(float));
			
		float *rtp2_d,*rtp1_d,*rvxp2_d,*rvxp1_d,*rvzp2_d,*rvzp1_d,*rvxs2_d,*rvxs1_d,*rvzs2_d,*rvzs1_d;
		cudaMalloc(&rtp2_d,nxanza*sizeof(float));
		cudaMalloc(&rtp1_d,nxanza*sizeof(float));
		cudaMalloc(&rvxp2_d,nxanza*sizeof(float));
		cudaMalloc(&rvxp1_d,nxanza*sizeof(float));
		cudaMalloc(&rvzp2_d,nxanza*sizeof(float));
		cudaMalloc(&rvzp1_d,nxanza*sizeof(float));
		cudaMalloc(&rvxs2_d,nxanza*sizeof(float));
		cudaMalloc(&rvxs1_d,nxanza*sizeof(float));
		cudaMalloc(&rvzs2_d,nxanza*sizeof(float));
		cudaMalloc(&rvzs1_d,nxanza*sizeof(float));

///////////////2017年03月14日 星期二 20时21分13秒 波场分离的LSRTM
		float *rvxp_integral_d,*rvzp_integral_d,*rvxs_integral_d,*rvzs_integral_d;
		cudaMalloc(&rvxp_integral_d,nxanza*sizeof(float));cudaMemset(rvxp_integral_d,0,nxanza*sizeof(float));
		cudaMalloc(&rvzp_integral_d,nxanza*sizeof(float));cudaMemset(rvzp_integral_d,0,nxanza*sizeof(float));
		cudaMalloc(&rvxs_integral_d,nxanza*sizeof(float));cudaMemset(rvxs_integral_d,0,nxanza*sizeof(float));
		cudaMalloc(&rvzs_integral_d,nxanza*sizeof(float));cudaMemset(rvzs_integral_d,0,nxanza*sizeof(float));

		float *rvxp_x_d,*rvzp_z_d,*rvxp_z_d,*rvzp_x_d;
		cudaMalloc(&rvxp_x_d,nxanza*sizeof(float));cudaMemset(rvxp_x_d,0,nxanza*sizeof(float));
		cudaMalloc(&rvzp_z_d,nxanza*sizeof(float));cudaMemset(rvzp_z_d,0,nxanza*sizeof(float));
		cudaMalloc(&rvxp_z_d,nxanza*sizeof(float));cudaMemset(rvxp_z_d,0,nxanza*sizeof(float));
		cudaMalloc(&rvzp_x_d,nxanza*sizeof(float));cudaMemset(rvzp_x_d,0,nxanza*sizeof(float));

		float *rvxs_x_d,*rvzs_z_d,*rvxs_z_d,*rvzs_x_d;
		cudaMalloc(&rvxs_x_d,nxanza*sizeof(float));cudaMemset(rvxs_x_d,0,nxanza*sizeof(float));
		cudaMalloc(&rvzs_z_d,nxanza*sizeof(float));cudaMemset(rvzs_z_d,0,nxanza*sizeof(float));
		cudaMalloc(&rvxs_z_d,nxanza*sizeof(float));cudaMemset(rvxs_z_d,0,nxanza*sizeof(float));
		cudaMalloc(&rvzs_x_d,nxanza*sizeof(float));cudaMemset(rvzs_x_d,0,nxanza*sizeof(float));

		float *vxp_x_d,*vzp_z_d,*vxp_z_d,*vzp_x_d;
		cudaMalloc(&vxp_x_d,nxanza*sizeof(float));cudaMemset(vxp_x_d,0,nxanza*sizeof(float));
		cudaMalloc(&vzp_z_d,nxanza*sizeof(float));cudaMemset(vzp_z_d,0,nxanza*sizeof(float));
		cudaMalloc(&vxp_z_d,nxanza*sizeof(float));cudaMemset(vxp_z_d,0,nxanza*sizeof(float));
		cudaMalloc(&vzp_x_d,nxanza*sizeof(float));cudaMemset(vzp_x_d,0,nxanza*sizeof(float));

		float *vxs_x_d,*vzs_z_d,*vxs_z_d,*vzs_x_d;
		cudaMalloc(&vxs_x_d,nxanza*sizeof(float));cudaMemset(vxs_x_d,0,nxanza*sizeof(float));
		cudaMalloc(&vzs_z_d,nxanza*sizeof(float));cudaMemset(vzs_z_d,0,nxanza*sizeof(float));
		cudaMalloc(&vxs_z_d,nxanza*sizeof(float));cudaMemset(vxs_z_d,0,nxanza*sizeof(float));
		cudaMalloc(&vzs_x_d,nxanza*sizeof(float));cudaMemset(vzs_x_d,0,nxanza*sizeof(float));

		float *vxp_t_d,*vzp_t_d,*vxs_t_d,*vzs_t_d;
		cudaMalloc(&vxp_t_d,nxanza*sizeof(float));cudaMemset(vxp_t_d,0,nxanza*sizeof(float));
		cudaMalloc(&vzp_t_d,nxanza*sizeof(float));cudaMemset(vzp_t_d,0,nxanza*sizeof(float));
		cudaMalloc(&vxs_t_d,nxanza*sizeof(float));cudaMemset(vxs_t_d,0,nxanza*sizeof(float));
		cudaMalloc(&vzs_t_d,nxanza*sizeof(float));cudaMemset(vzs_t_d,0,nxanza*sizeof(float));

		float *p_d,*s_d,*rp_d,*rs_d;
		cudaMalloc(&p_d,nxanza*sizeof(float));		cudaMemset(p_d,0,nxanza*sizeof(float));
		cudaMalloc(&s_d,nxanza*sizeof(float));		cudaMemset(s_d,0,nxanza*sizeof(float));
		cudaMalloc(&rp_d,nxanza*sizeof(float));		cudaMemset(rp_d,0,nxanza*sizeof(float));
		cudaMalloc(&rs_d,nxanza*sizeof(float));		cudaMemset(rs_d,0,nxanza*sizeof(float));
///////////////2017年03月14日 星期二 20时21分13秒 波场分离的LSRTM

////objective value
		float *obj_d;
		cudaMalloc(&obj_d,3*sizeof(float));			cudaMemset(obj_d,0,3*sizeof(float));///obj_d[0]:a=0  obj_d[1]:a=0.7  obj_d[2]:a=1.3 otherwise a=2/3

		float *obj_h,*obj_niter_h,*obj_niter_h1;
		obj_h=alloc1float(3);memset(obj_h,0,3*sizeof(float));///obj_d[0]:a=0  obj_d[1]:a=0.7  obj_d[2]:a=1.3 otherwise a=2/3
		obj_niter_h=alloc1float(niter);			memset(obj_niter_h,0,niter*sizeof(float));
		obj_niter_h1=alloc1float(niter);			memset(obj_niter_h1,0,niter*sizeof(float));
		float obj_exchange=0.0;//,obj_first=0;
////objective value

		
/////conjugate parameter
		float *beta_d,*alpha_d,*beta_step_d;	
		cudaMalloc(&beta_d,3*sizeof(float));		cudaMemset(beta_d,0,3*sizeof(float));/////make gradient direction transform to conjuagte direction
		cudaMalloc(&alpha_d,1*sizeof(float));		cudaMemset(alpha_d,0,1*sizeof(float));///////a step length of Hybrid conjugate direction 
		cudaMalloc(&beta_step_d,3*sizeof(float));		cudaMemset(beta_step_d,0,3*sizeof(float));/////assign different step for vp/vs/density

		float *d_alpha1,*d_alpha2;/////for beta_setp /* compute the numerator and the denominator of alpha: equations 5 and 12 */
		cudaMalloc(&d_alpha1,lt_rec*sizeof(float));		cudaMemset(d_alpha1,0,lt_rec*sizeof(float));
		cudaMalloc(&d_alpha2,lt_rec*sizeof(float));		cudaMemset(d_alpha2,0,lt_rec*sizeof(float));
		
		float *beta_h,*alpha_h,*beta_step_h;
		beta_h=alloc1float(3);					memset(beta_h,0,3*sizeof(float));
		alpha_h=alloc1float(1);					memset(alpha_h,0,1*sizeof(float));
		beta_step_h=alloc1float(3);					memset(beta_step_h,0,3*sizeof(float));
		
		float *epsil_d,*epsil_h;
		cudaMalloc(&epsil_d,4*sizeof(float));			cudaMemset(epsil_d,0,4*sizeof(float));	
		epsil_h=alloc1float(4);					memset(epsil_h,0,4*sizeof(float));
//////four small perturbation for vp or vx or density///epsil_d[0]:vp epsil_d[1]:vs  epsil_d[2]:density  epsil_d[3]:all


///////////////////////new acqusition way  2017年08月17日 星期四 10时06分44秒 
///////////final output lame parameterization
		float *perturb_lame1_d,*perturb_lame2_d,*perturb_den_d;
		cudaMalloc(&perturb_lame1_d,nxnz*sizeof(float));		cudaMemset(perturb_lame1_d,0,nxnz*sizeof(float));/////final output lame parameterization
		cudaMalloc(&perturb_lame2_d,nxnz*sizeof(float));		cudaMemset(perturb_lame2_d,0,nxnz*sizeof(float));/////final output lame parameterization
		cudaMalloc(&perturb_den_d,nxnz*sizeof(float));		cudaMemset(perturb_den_d,0,nxnz*sizeof(float));/////final output lame parameterization

///////////final output velocity or impedance parameterization
		float *perturb_vp_d,*perturb_vs_d,*perturb_density_d;
		cudaMalloc(&perturb_vp_d,nxnz*sizeof(float));		cudaMemset(perturb_vp_d,0,nxnz*sizeof(float));////final output velocity or impedance parameterization
		cudaMalloc(&perturb_vs_d,nxnz*sizeof(float));		cudaMemset(perturb_vs_d,0,nxnz*sizeof(float));////final output velocity or impedance parameterization
		cudaMalloc(&perturb_density_d,nxnz*sizeof(float));		cudaMemset(perturb_density_d,0,nxnz*sizeof(float));////final output velocity or impedance paramete
///////////final output


////////all gradient or conjugate direction
		float *all_grad_den_pp_d,*all_grad_lame1_pp_d,*all_grad_lame2_pp_d;		
		float *all_grad_den_ps_d,*all_grad_lame1_ps_d,*all_grad_lame2_ps_d;
		float *all_grad_den_sp_d,*all_grad_lame1_sp_d,*all_grad_lame2_sp_d;
		float *all_grad_den_ss_d,*all_grad_lame1_ss_d,*all_grad_lame2_ss_d;

		cudaMalloc(&all_grad_den_pp_d,nxnz*sizeof(float));		cudaMemset(all_grad_den_pp_d,0,nxnz*sizeof(float));
		cudaMalloc(&all_grad_lame1_pp_d,nxnz*sizeof(float));	cudaMemset(all_grad_lame1_pp_d,0,nxnz*sizeof(float));
		cudaMalloc(&all_grad_lame2_pp_d,nxnz*sizeof(float));	cudaMemset(all_grad_lame2_pp_d,0,nxnz*sizeof(float));
		cudaMalloc(&all_grad_den_ps_d,nxnz*sizeof(float));		cudaMemset(all_grad_den_ps_d,0,nxnz*sizeof(float));
		cudaMalloc(&all_grad_lame1_ps_d,nxnz*sizeof(float));	cudaMemset(all_grad_lame1_ps_d,0,nxnz*sizeof(float));
		cudaMalloc(&all_grad_lame2_ps_d,nxnz*sizeof(float));	cudaMemset(all_grad_lame2_ps_d,0,nxnz*sizeof(float));
		cudaMalloc(&all_grad_den_sp_d,nxnz*sizeof(float));		cudaMemset(all_grad_den_sp_d,0,nxnz*sizeof(float));
		cudaMalloc(&all_grad_lame1_sp_d,nxnz*sizeof(float));	cudaMemset(all_grad_lame1_sp_d,0,nxnz*sizeof(float));
		cudaMalloc(&all_grad_lame2_sp_d,nxnz*sizeof(float));	cudaMemset(all_grad_lame2_sp_d,0,nxnz*sizeof(float));
		cudaMalloc(&all_grad_den_ss_d,nxnz*sizeof(float));		cudaMemset(all_grad_den_ss_d,0,nxnz*sizeof(float));
		cudaMalloc(&all_grad_lame1_ss_d,nxnz*sizeof(float));	cudaMemset(all_grad_lame1_ss_d,0,nxnz*sizeof(float));
		cudaMalloc(&all_grad_lame2_ss_d,nxnz*sizeof(float));	cudaMemset(all_grad_lame2_ss_d,0,nxnz*sizeof(float));
////////////////when 波场分离的LSRTM 使用的时候
		//float *all_grad_den_d,*all_grad_lame1_d,*all_grad_lame2_d;
		float *all_grad_den1_d,*all_grad_lame11_d,*all_grad_lame22_d;
		float *all_conj_den_d,*all_conj_lame1_d,*all_conj_lame2_d;

		//cudaMalloc(&all_grad_den_d,nxnz*sizeof(float));		cudaMemset(all_grad_den_d,0,nxnz*sizeof(float));
		//cudaMalloc(&all_grad_lame1_d,nxnz*sizeof(float));	cudaMemset(all_grad_lame1_d,0,nxnz*sizeof(float));
		//cudaMalloc(&all_grad_lame2_d,nxnz*sizeof(float));	cudaMemset(all_grad_lame2_d,0,nxnz*sizeof(float));/////////the previous step

		cudaMalloc(&all_grad_den1_d,nxnz*sizeof(float));		cudaMemset(all_grad_den1_d,0,nxnz*sizeof(float));
		cudaMalloc(&all_grad_lame11_d,nxnz*sizeof(float));		cudaMemset(all_grad_lame11_d,0,nxnz*sizeof(float));
		cudaMalloc(&all_grad_lame22_d,nxnz*sizeof(float));		cudaMemset(all_grad_lame22_d,0,nxnz*sizeof(float));/////////the current step

		cudaMalloc(&all_conj_den_d,nxnz*sizeof(float));		cudaMemset(all_conj_den_d,0,nxnz*sizeof(float));
		cudaMalloc(&all_conj_lame1_d,nxnz*sizeof(float));		cudaMemset(all_conj_lame1_d,0,nxnz*sizeof(float));
		cudaMalloc(&all_conj_lame2_d,nxnz*sizeof(float));		cudaMemset(all_conj_lame2_d,0,nxnz*sizeof(float));/////////the current conjugate step

		//float *all_grad_density_d,*all_grad_vp_d ,*all_grad_vs_d;
		float *all_grad_density1_d,*all_grad_vp1_d,*all_grad_vs1_d;
		float *all_conj_density_d,*all_conj_vp_d,*all_conj_vs_d;

		float *all_hydrid_conj_d,*all_hydrid_grad1_d,*all_hydrid_grad2_d;

		//cudaMalloc(&all_grad_density_d,nxnz*sizeof(float));	cudaMemset(all_grad_density_d,0,nxnz*sizeof(float));
		//cudaMalloc(&all_grad_vp_d,nxnz*sizeof(float));		cudaMemset(all_grad_vp_d,0,nxnz*sizeof(float));
		//cudaMalloc(&all_grad_vs_d,nxnz*sizeof(float));		cudaMemset(all_grad_vs_d,0,nxnz*sizeof(float));/////////the previous step

		cudaMalloc(&all_grad_density1_d,nxnz*sizeof(float));	cudaMemset(all_grad_density1_d,0,nxnz*sizeof(float));
		cudaMalloc(&all_grad_vp1_d,nxnz*sizeof(float));		cudaMemset(all_grad_vp1_d,0,nxnz*sizeof(float));
		cudaMalloc(&all_grad_vs1_d,nxnz*sizeof(float));		cudaMemset(all_grad_vs1_d,0,nxnz*sizeof(float));/////////the current step

		cudaMalloc(&all_conj_vp_d,nxnz*sizeof(float));		cudaMemset(all_conj_vp_d,0,nxnz*sizeof(float));	
		cudaMalloc(&all_conj_vs_d,nxnz*sizeof(float));		cudaMemset(all_conj_vs_d,0,nxnz*sizeof(float));	
		cudaMalloc(&all_conj_density_d,nxnz*sizeof(float));	cudaMemset(all_conj_density_d,0,nxnz*sizeof(float));/////////the current conjugate step

		cudaMalloc(&all_hydrid_conj_d,3*nxnz*sizeof(float));	cudaMemset(all_hydrid_conj_d,0,3*nxnz*sizeof(float));
		cudaMalloc(&all_hydrid_grad1_d,3*nxnz*sizeof(float));	cudaMemset(all_hydrid_grad1_d,0,3*nxnz*sizeof(float));
		cudaMalloc(&all_hydrid_grad2_d,3*nxnz*sizeof(float));	cudaMemset(all_hydrid_grad2_d,0,3*nxnz*sizeof(float));
////////all gradient or conjugate direction


////////migration result
		float *all_vresultpp_d,*all_vresultps_d,*all_vresultsp_d,*all_vresultss_d; 
		float *all_vresultppx_d,*all_vresultpsx_d,*all_vresultspx_d,*all_vresultssx_d,*all_vresultppz_d,*all_vresultpsz_d,*all_vresultspz_d,*all_vresultssz_d;
		cudaMalloc(&all_vresultpp_d,nxnz*sizeof(float));	cudaMemset(all_vresultpp_d,0,nxnz*sizeof(float));
		cudaMalloc(&all_vresultps_d,nxnz*sizeof(float));	cudaMemset(all_vresultps_d,0,nxnz*sizeof(float));
		cudaMalloc(&all_vresultsp_d,nxnz*sizeof(float));	cudaMemset(all_vresultsp_d,0,nxnz*sizeof(float));
		cudaMalloc(&all_vresultss_d,nxnz*sizeof(float));	cudaMemset(all_vresultss_d,0,nxnz*sizeof(float));

		cudaMalloc(&all_vresultppx_d,nxnz*sizeof(float));	cudaMemset(all_vresultppx_d,0,nxnz*sizeof(float));
		cudaMalloc(&all_vresultpsx_d,nxnz*sizeof(float));	cudaMemset(all_vresultpsx_d,0,nxnz*sizeof(float));
		cudaMalloc(&all_vresultspx_d,nxnz*sizeof(float));	cudaMemset(all_vresultspx_d,0,nxnz*sizeof(float));
		cudaMalloc(&all_vresultssx_d,nxnz*sizeof(float));	cudaMemset(all_vresultssx_d,0,nxnz*sizeof(float));

		cudaMalloc(&all_vresultppz_d,nxnz*sizeof(float));	cudaMemset(all_vresultppz_d,0,nxnz*sizeof(float));
		cudaMalloc(&all_vresultpsz_d,nxnz*sizeof(float));	cudaMemset(all_vresultpsz_d,0,nxnz*sizeof(float));
		cudaMalloc(&all_vresultspz_d,nxnz*sizeof(float));	cudaMemset(all_vresultspz_d,0,nxnz*sizeof(float));
		cudaMalloc(&all_vresultssz_d,nxnz*sizeof(float));	cudaMemset(all_vresultssz_d,0,nxnz*sizeof(float));

		float *all_resultpp_d,*all_resultps_d,*all_resultps1_d,*all_resultps2_d,*all_resultsp_d,*all_resultsp1_d,*all_resultsp2_d,*all_resultss_d;		
		cudaMalloc(&all_resultpp_d,nxnz*sizeof(float));	cudaMemset(all_resultpp_d,0,nxnz*sizeof(float));

		cudaMalloc(&all_resultps_d,nxnz*sizeof(float));	cudaMemset(all_resultps_d,0,nxnz*sizeof(float));
		cudaMalloc(&all_resultps1_d,nxnz*sizeof(float));	cudaMemset(all_resultps1_d,0,nxnz*sizeof(float));
		cudaMalloc(&all_resultps2_d,nxnz*sizeof(float));	cudaMemset(all_resultps2_d,0,nxnz*sizeof(float));

		cudaMalloc(&all_resultsp_d,nxnz*sizeof(float));	cudaMemset(all_resultsp_d,0,nxnz*sizeof(float));
		cudaMalloc(&all_resultsp1_d,nxnz*sizeof(float));	cudaMemset(all_resultsp1_d,0,nxnz*sizeof(float));
		cudaMalloc(&all_resultsp2_d,nxnz*sizeof(float));	cudaMemset(all_resultsp2_d,0,nxnz*sizeof(float));

		cudaMalloc(&all_resultss_d,nxnz*sizeof(float));	cudaMemset(all_resultss_d,0,nxnz*sizeof(float));
	
		float *all_result_tp_d;
		cudaMalloc(&all_result_tp_d,nxnz*sizeof(float));	cudaMemset(all_result_tp_d,0,nxnz*sizeof(float));

		gpumem += (nxnz*51)*sizeof(float)/1024.0/1024.0;/////for check 2017年07月27日 星期四 10时11分40秒

////////migration result
///////////2017年03月18日 星期六 21时07分00秒Traditional   ERTM 
		float *resultpp_d,*resultps_d,*resultps1_d,*resultps2_d,*resultsp_d,*resultsp1_d,*resultsp2_d,*resultss_d;
		cudaMalloc(&resultpp_d,nx_size_nz*sizeof(float));		cudaMemset(resultpp_d,0,nx_size_nz*sizeof(float));

		cudaMalloc(&resultps_d,nx_size_nz*sizeof(float));		cudaMemset(resultps_d,0,nx_size_nz*sizeof(float));
		cudaMalloc(&resultps1_d,nx_size_nz*sizeof(float));		cudaMemset(resultps1_d,0,nx_size_nz*sizeof(float));
		cudaMalloc(&resultps2_d,nx_size_nz*sizeof(float));		cudaMemset(resultps2_d,0,nx_size_nz*sizeof(float));

		cudaMalloc(&resultsp_d,nx_size_nz*sizeof(float));		cudaMemset(resultsp_d,0,nx_size_nz*sizeof(float));
		cudaMalloc(&resultsp1_d,nx_size_nz*sizeof(float));		cudaMemset(resultsp1_d,0,nx_size_nz*sizeof(float));
		cudaMalloc(&resultsp2_d,nx_size_nz*sizeof(float));		cudaMemset(resultsp2_d,0,nx_size_nz*sizeof(float));

		cudaMalloc(&resultss_d,nx_size_nz*sizeof(float));		cudaMemset(resultss_d,0,nx_size_nz*sizeof(float));
	
		float *result_tp_d;
		cudaMalloc(&result_tp_d,nx_size_nz*sizeof(float));		cudaMemset(result_tp_d,0,nx_size_nz*sizeof(float));

///////////2017年03月18日 星期六 21时07分00秒Traditional   ERTM
		float *vresultpp_d,*vresultps_d,*vresultsp_d,*vresultss_d; 
		float *vresultppx_d,*vresultpsx_d,*vresultspx_d,*vresultssx_d,*vresultppz_d,*vresultpsz_d,*vresultspz_d,*vresultssz_d;
		cudaMalloc(&vresultpp_d,nx_size_nz*sizeof(float));		cudaMemset(vresultpp_d,0,nx_size_nz*sizeof(float));
		cudaMalloc(&vresultps_d,nx_size_nz*sizeof(float));		cudaMemset(vresultps_d,0,nx_size_nz*sizeof(float));
		cudaMalloc(&vresultsp_d,nx_size_nz*sizeof(float));		cudaMemset(vresultsp_d,0,nx_size_nz*sizeof(float));
		cudaMalloc(&vresultss_d,nx_size_nz*sizeof(float));		cudaMemset(vresultss_d,0,nx_size_nz*sizeof(float));

		cudaMalloc(&vresultppx_d,nx_size_nz*sizeof(float));	cudaMemset(vresultppx_d,0,nx_size_nz*sizeof(float));
		cudaMalloc(&vresultpsx_d,nx_size_nz*sizeof(float));	cudaMemset(vresultpsx_d,0,nx_size_nz*sizeof(float));
		cudaMalloc(&vresultspx_d,nx_size_nz*sizeof(float));	cudaMemset(vresultspx_d,0,nx_size_nz*sizeof(float));
		cudaMalloc(&vresultssx_d,nx_size_nz*sizeof(float));	cudaMemset(vresultssx_d,0,nx_size_nz*sizeof(float));

		cudaMalloc(&vresultppz_d,nx_size_nz*sizeof(float));	cudaMemset(vresultppz_d,0,nx_size_nz*sizeof(float));
		cudaMalloc(&vresultpsz_d,nx_size_nz*sizeof(float));	cudaMemset(vresultpsz_d,0,nx_size_nz*sizeof(float));
		cudaMalloc(&vresultspz_d,nx_size_nz*sizeof(float));	cudaMemset(vresultspz_d,0,nx_size_nz*sizeof(float));
		cudaMalloc(&vresultssz_d,nx_size_nz*sizeof(float));	cudaMemset(vresultssz_d,0,nx_size_nz*sizeof(float));
///////////2017年03月18日 星期六 21时07分00秒Traditional   ERTM 

//////////////////////////////source_illumination or excitation amplitude imaging condition  2018年01月24日 星期三 20时33分21秒 
		float *down_vpp_x_d,*down_vpp_z_d,*down_vss_x_d,*down_vss_z_d;
		float *down_tp_d,*down_vpp_d,*down_vss_d,*down_pp_d,*down_ss_d,*down_xx_d,*down_zz_d; 
		cudaMalloc(&down_vpp_x_d,nx_size_nz*sizeof(float));		cudaMemset(down_vpp_x_d,0,nx_size_nz*sizeof(float));
		cudaMalloc(&down_vpp_z_d,nx_size_nz*sizeof(float));		cudaMemset(down_vpp_z_d,0,nx_size_nz*sizeof(float));
		cudaMalloc(&down_vss_x_d,nx_size_nz*sizeof(float));		cudaMemset(down_vss_x_d,0,nx_size_nz*sizeof(float));
		cudaMalloc(&down_vss_z_d,nx_size_nz*sizeof(float));		cudaMemset(down_vss_z_d,0,nx_size_nz*sizeof(float));

		cudaMalloc(&down_tp_d,nx_size_nz*sizeof(float));			cudaMemset(down_tp_d,0,nx_size_nz*sizeof(float));

		cudaMalloc(&down_vpp_d,nx_size_nz*sizeof(float));			cudaMemset(down_vpp_d,0,nx_size_nz*sizeof(float));
		cudaMalloc(&down_vss_d,nx_size_nz*sizeof(float));			cudaMemset(down_vss_d,0,nx_size_nz*sizeof(float));

		cudaMalloc(&down_pp_d,nx_size_nz*sizeof(float));			cudaMemset(down_pp_d,0,nx_size_nz*sizeof(float));
		cudaMalloc(&down_ss_d,nx_size_nz*sizeof(float));			cudaMemset(down_ss_d,0,nx_size_nz*sizeof(float));

		cudaMalloc(&down_xx_d,nx_size_nz*sizeof(float));			cudaMemset(down_xx_d,0,nx_size_nz*sizeof(float));
		cudaMalloc(&down_zz_d,nx_size_nz*sizeof(float));			cudaMemset(down_zz_d,0,nx_size_nz*sizeof(float));
//////excitation:   	related function in kernel_3  excitation amplitude imaging condition
		float *ex_vresultpp_d,*ex_vresultps_d,*ex_result_tp_d,*ex_result_tp_old_d,*resultxx_d,*resultzz_d;
		float *com_ex_vresultpp_d,*com_ex_vresultps_d;
		cudaMalloc(&com_ex_vresultpp_d,nx_size_nz*sizeof(float));		cudaMemset(com_ex_vresultpp_d,0,nx_size_nz*sizeof(float));
		cudaMalloc(&com_ex_vresultps_d,nx_size_nz*sizeof(float));		cudaMemset(com_ex_vresultps_d,0,nx_size_nz*sizeof(float));
		cudaMalloc(&ex_vresultpp_d,nx_size_nz*sizeof(float));		cudaMemset(ex_vresultpp_d,0,nx_size_nz*sizeof(float));
		cudaMalloc(&ex_vresultps_d,nx_size_nz*sizeof(float));		cudaMemset(ex_vresultps_d,0,nx_size_nz*sizeof(float));
		cudaMalloc(&ex_result_tp_d,nx_size_nz*sizeof(float));		cudaMemset(ex_result_tp_d,0,nx_size_nz*sizeof(float));
		cudaMalloc(&ex_result_tp_old_d,nx_size_nz*sizeof(float));		cudaMemset(ex_result_tp_old_d,0,nx_size_nz*sizeof(float));
		cudaMalloc(&resultxx_d,nx_size_nz*sizeof(float));			cudaMemset(resultxx_d,0,nx_size_nz*sizeof(float));
		cudaMalloc(&resultzz_d,nx_size_nz*sizeof(float));			cudaMemset(resultzz_d,0,nx_size_nz*sizeof(float));

		float *all_ex_vresultpp_d,*all_ex_vresultps_d,*all_ex_result_tp_d,*all_ex_result_tp_old_d,*all_resultxx_d,*all_resultzz_d;
		float *all_com_ex_vresultpp_d,*all_com_ex_vresultps_d;
		cudaMalloc(&all_com_ex_vresultpp_d,nxnz*sizeof(float));		cudaMemset(all_com_ex_vresultpp_d,0,nxnz*sizeof(float));
		cudaMalloc(&all_com_ex_vresultps_d,nxnz*sizeof(float));		cudaMemset(all_com_ex_vresultps_d,0,nxnz*sizeof(float));
		cudaMalloc(&all_ex_vresultpp_d,nxnz*sizeof(float));		cudaMemset(all_ex_vresultpp_d,0,nxnz*sizeof(float));
		cudaMalloc(&all_ex_vresultps_d,nxnz*sizeof(float));		cudaMemset(all_ex_vresultps_d,0,nxnz*sizeof(float)); 
		cudaMalloc(&all_ex_result_tp_d,nxnz*sizeof(float));		cudaMemset(all_ex_result_tp_d,0,nxnz*sizeof(float));
		cudaMalloc(&all_ex_result_tp_old_d,nxnz*sizeof(float));		cudaMemset(all_ex_result_tp_old_d,0,nxnz*sizeof(float));
		cudaMalloc(&all_resultxx_d,nxnz*sizeof(float));			cudaMemset(all_resultxx_d,0,nxnz*sizeof(float)); 
		cudaMalloc(&all_resultzz_d,nxnz*sizeof(float));			cudaMemset(all_resultzz_d,0,nxnz*sizeof(float));  

		float *ex_amp_d,*ex_amp_x_d,*ex_amp_z_d,*ex_amp_tp_old_d,*ex_time_d;
		cudaMalloc(&ex_amp_d,nxanza*sizeof(float));			cudaMemset(ex_amp_d,0,nxanza*sizeof(float));
		cudaMalloc(&ex_amp_x_d,nxanza*sizeof(float));			cudaMemset(ex_amp_x_d,0,nxanza*sizeof(float));
		cudaMalloc(&ex_amp_z_d,nxanza*sizeof(float));			cudaMemset(ex_amp_z_d,0,nxanza*sizeof(float));
		cudaMalloc(&ex_amp_tp_old_d,nxanza*sizeof(float));			cudaMemset(ex_amp_tp_old_d,0,nxanza*sizeof(float));
		cudaMalloc(&ex_time_d,nxanza*sizeof(float));			cudaMemset(ex_time_d,0,nxanza*sizeof(float));
////////////////////////////2018年05月21日 星期一 10时22分36秒 
		float *ex_tp_time_d,*ex_amp_tp_d;
		cudaMalloc(&ex_amp_tp_d,nxanza*sizeof(float));			cudaMemset(ex_amp_tp_d,0,nxanza*sizeof(float));
		cudaMalloc(&ex_tp_time_d,nxanza*sizeof(float));			cudaMemset(ex_tp_time_d,0,nxanza*sizeof(float));

		float *ex_angle_pp_d,*ex_angle_rpp_d,*ex_angle_rps_d;
		cudaMalloc(&ex_angle_pp_d,nxanza*sizeof(float));			cudaMemset(ex_angle_pp_d,0,nxanza*sizeof(float));
		cudaMalloc(&ex_angle_rpp_d,nxanza*sizeof(float));			cudaMemset(ex_angle_rpp_d,0,nxanza*sizeof(float));
		cudaMalloc(&ex_angle_rps_d,nxanza*sizeof(float));			cudaMemset(ex_angle_rps_d,0,nxanza*sizeof(float));
		float *ex_angle_pp1_d,*ex_angle_rpp1_d,*ex_angle_rps1_d;
		cudaMalloc(&ex_angle_pp1_d,nxanza*sizeof(float));			cudaMemset(ex_angle_pp1_d,0,nxanza*sizeof(float));
		cudaMalloc(&ex_angle_rpp1_d,nxanza*sizeof(float));			cudaMemset(ex_angle_rpp1_d,0,nxanza*sizeof(float));
		cudaMalloc(&ex_angle_rps1_d,nxanza*sizeof(float));			cudaMemset(ex_angle_rps1_d,0,nxanza*sizeof(float));
		float *para_max_d,p_printf;
		cudaMalloc(&para_max_d,20*sizeof(float));				cudaMemset(para_max_d,0,20*sizeof(float));

		float *ex_open_pp_d,*ex_open_ps_d,*ex_open_pp1_d,*ex_open_ps1_d;
		cudaMalloc(&ex_open_pp_d,nxanza*sizeof(float));			cudaMemset(ex_open_pp_d,0,nxanza*sizeof(float));
		cudaMalloc(&ex_open_ps_d,nxanza*sizeof(float));			cudaMemset(ex_open_ps_d,0,nxanza*sizeof(float));
		cudaMalloc(&ex_open_pp1_d,nxanza*sizeof(float));			cudaMemset(ex_open_pp1_d,0,nxanza*sizeof(float));
		cudaMalloc(&ex_open_ps1_d,nxanza*sizeof(float));			cudaMemset(ex_open_ps1_d,0,nxanza*sizeof(float));
		float *ex_com_pp_sign_d,*ex_com_ps_sign_d;
		cudaMalloc(&ex_com_pp_sign_d,nxanza*sizeof(float));			cudaMemset(ex_com_pp_sign_d,0,nxanza*sizeof(float));
		cudaMalloc(&ex_com_ps_sign_d,nxanza*sizeof(float));			cudaMemset(ex_com_ps_sign_d,0,nxanza*sizeof(float));
//////////////////////////////

///////////////////////new acqusition way  2017年08月17日 星期四 10时06分44秒 
////lame1:langda,lame2:u
/////////////////Ren and Liu 2016 in geophysics  Xu and Mcmechan 2014 in geophysics
		float *grad_den_pp_d,*grad_lame1_pp_d,*grad_lame2_pp_d;		
		float *grad_den_ps_d,*grad_lame1_ps_d,*grad_lame2_ps_d;
		float *grad_den_sp_d,*grad_lame1_sp_d,*grad_lame2_sp_d;
		float *grad_den_ss_d,*grad_lame1_ss_d,*grad_lame2_ss_d;

		cudaMalloc(&grad_den_pp_d,nx_size_nz*sizeof(float));	cudaMemset(grad_den_pp_d,0,nx_size_nz*sizeof(float));
		cudaMalloc(&grad_lame1_pp_d,nx_size_nz*sizeof(float));	cudaMemset(grad_lame1_pp_d,0,nx_size_nz*sizeof(float));
		cudaMalloc(&grad_lame2_pp_d,nx_size_nz*sizeof(float));	cudaMemset(grad_lame2_pp_d,0,nx_size_nz*sizeof(float));
		cudaMalloc(&grad_den_ps_d,nx_size_nz*sizeof(float));	cudaMemset(grad_den_ps_d,0,nx_size_nz*sizeof(float));
		cudaMalloc(&grad_lame1_ps_d,nx_size_nz*sizeof(float));	cudaMemset(grad_lame1_ps_d,0,nx_size_nz*sizeof(float));
		cudaMalloc(&grad_lame2_ps_d,nx_size_nz*sizeof(float));	cudaMemset(grad_lame2_ps_d,0,nx_size_nz*sizeof(float));
		cudaMalloc(&grad_den_sp_d,nx_size_nz*sizeof(float));	cudaMemset(grad_den_sp_d,0,nx_size_nz*sizeof(float));
		cudaMalloc(&grad_lame1_sp_d,nx_size_nz*sizeof(float));	cudaMemset(grad_lame1_sp_d,0,nx_size_nz*sizeof(float));
		cudaMalloc(&grad_lame2_sp_d,nx_size_nz*sizeof(float));	cudaMemset(grad_lame2_sp_d,0,nx_size_nz*sizeof(float));
		cudaMalloc(&grad_den_ss_d,nx_size_nz*sizeof(float));	cudaMemset(grad_den_ss_d,0,nx_size_nz*sizeof(float));
		cudaMalloc(&grad_lame1_ss_d,nx_size_nz*sizeof(float));	cudaMemset(grad_lame1_ss_d,0,nx_size_nz*sizeof(float));
		cudaMalloc(&grad_lame2_ss_d,nx_size_nz*sizeof(float));	cudaMemset(grad_lame2_ss_d,0,nx_size_nz*sizeof(float));

		float *grad_den_d,*grad_lame1_d,*grad_lame2_d;
		float *grad_den1_d,*grad_lame11_d,*grad_lame22_d;
	
		cudaMalloc(&grad_den_d,nx_size_nz*sizeof(float));		cudaMemset(grad_den_d,0,nx_size_nz*sizeof(float));
		cudaMalloc(&grad_lame1_d,nx_size_nz*sizeof(float));	cudaMemset(grad_lame1_d,0,nx_size_nz*sizeof(float));
		cudaMalloc(&grad_lame2_d,nx_size_nz*sizeof(float));	cudaMemset(grad_lame2_d,0,nx_size_nz*sizeof(float));/////////the previous step

		cudaMalloc(&grad_den1_d,nx_size_nz*sizeof(float));		cudaMemset(grad_den1_d,0,nx_size_nz*sizeof(float));
		cudaMalloc(&grad_lame11_d,nx_size_nz*sizeof(float));	cudaMemset(grad_lame11_d,0,nx_size_nz*sizeof(float));
		cudaMalloc(&grad_lame22_d,nx_size_nz*sizeof(float));	cudaMemset(grad_lame22_d,0,nx_size_nz*sizeof(float));/////////the current step	

		
		float *grad_vp1_d,*grad_vs1_d,*grad_density1_d;
		cudaMalloc(&grad_vp1_d,nx_size_nz*sizeof(float));		cudaMemset(grad_vp1_d,0,nx_size_nz*sizeof(float));
		cudaMalloc(&grad_vs1_d,nx_size_nz*sizeof(float));		cudaMemset(grad_vs1_d,0,nx_size_nz*sizeof(float));
		cudaMalloc(&grad_density1_d,nx_size_nz*sizeof(float));	cudaMemset(grad_density1_d,0,nx_size_nz*sizeof(float));/////////the current step		
////lame1:langda,lame2:u		

////gradient or conjugate direction for lame1 and lame2 den	
		float *tmp_perturb_lame1_d,*tmp_perturb_lame2_d,*tmp_perturb_den_d;
		cudaMalloc(&tmp_perturb_lame1_d,nxanza*sizeof(float));cudaMemset(tmp_perturb_lame1_d,0,nxanza*sizeof(float));/////the tmp perturb result
		cudaMalloc(&tmp_perturb_lame2_d,nxanza*sizeof(float));cudaMemset(tmp_perturb_lame2_d,0,nxanza*sizeof(float));/////the tmp perturb result
		cudaMalloc(&tmp_perturb_den_d,nxanza*sizeof(float));cudaMemset(tmp_perturb_den_d,0,nxanza*sizeof(float));/////the tmp perturb result
	
		float *tmp_perturb_vp_d,*tmp_perturb_vs_d,*tmp_perturb_density_d;
		cudaMalloc(&tmp_perturb_vp_d,nxanza*sizeof(float));cudaMemset(tmp_perturb_vp_d,0,nxanza*sizeof(float));/////the tmp perturb result
		cudaMalloc(&tmp_perturb_vs_d,nxanza*sizeof(float));cudaMemset(tmp_perturb_vs_d,0,nxanza*sizeof(float));/////the tmp  perturb result
		cudaMalloc(&tmp_perturb_density_d,nxanza*sizeof(float));cudaMemset(tmp_perturb_density_d,0,nxanza*sizeof(float));/////the tmp perturb result
////gradient or conjugate direction for vp and vs density	

/////////for nomarlized
		float *d_illum,*d_illum_new,*r_d_illum;
		cudaMalloc(&d_illum,nxanza*sizeof(float));			cudaMemset(d_illum,0,nxanza*sizeof(float));

		cudaMalloc(&r_d_illum,nxanza*sizeof(float));		cudaMemset(r_d_illum,0,nxanza*sizeof(float));

		cudaMalloc(&d_illum_new,nxa_new_nza*sizeof(float));	cudaMemset(d_illum_new,0,nxa_new_nza*sizeof(float));		

//////////2016年11月20日 星期日 05时55分17秒 and optimize at 2017年01月03日 星期二 10时08分13秒 
		float *dem_p1_d,*dem_p2_d,*dem_p3_d,*dem_p4_d,*dem_p5_d,*dem_p6_d,*dem_p7_d,*dem_p8_d;
		cudaMalloc(&dem_p1_d,nxanza*sizeof(float));cudaMemset(dem_p1_d,0,nxanza*sizeof(float));
		cudaMalloc(&dem_p2_d,nxanza*sizeof(float));cudaMemset(dem_p2_d,0,nxanza*sizeof(float));
		cudaMalloc(&dem_p3_d,nxanza*sizeof(float));cudaMemset(dem_p3_d,0,nxanza*sizeof(float));
		cudaMalloc(&dem_p4_d,nxanza*sizeof(float));cudaMemset(dem_p4_d,0,nxanza*sizeof(float));
		cudaMalloc(&dem_p5_d,nxanza*sizeof(float));cudaMemset(dem_p5_d,0,nxanza*sizeof(float));
		cudaMalloc(&dem_p6_d,nxanza*sizeof(float));cudaMemset(dem_p6_d,0,nxanza*sizeof(float));
		cudaMalloc(&dem_p7_d,nxanza*sizeof(float));cudaMemset(dem_p7_d,0,nxanza*sizeof(float));
		cudaMalloc(&dem_p8_d,nxanza*sizeof(float));cudaMemset(dem_p8_d,0,nxanza*sizeof(float));

		float *dem_p_all_d;
		cudaMalloc(&dem_p_all_d,8*nxanza*sizeof(float));cudaMemset(dem_p_all_d,0,8*nxanza*sizeof(float));

//filter_signy_d  this is for Du or Li method for polarity correction
		/*float *sign_d,*sign1_d;float *signx_d,*signy_d,*signz_d,*filter_signx_d,*filter_signy_d,*filter_signz_d;
		cudaMalloc(&sign_d,nxanza*sizeof(float));cudaMemset(sign_d,0,nxanza*sizeof(float));
		cudaMalloc(&sign1_d,nxanza*sizeof(float));cudaMemset(sign1_d,0,nxanza*sizeof(float));
		cudaMalloc(&signx_d,nxanza*sizeof(float));cudaMemset(signx_d,0,nxanza*sizeof(float));
		cudaMalloc(&signy_d,nxanza*sizeof(float));cudaMemset(signy_d,0,nxanza*sizeof(float));
		cudaMalloc(&signz_d,nxanza*sizeof(float));cudaMemset(signz_d,0,nxanza*sizeof(float));
		cudaMalloc(&filter_signx_d,nxanza*sizeof(float));cudaMemset(filter_signx_d,0,nxanza*sizeof(float));
		cudaMalloc(&filter_signy_d,nxanza*sizeof(float));cudaMemset(filter_signy_d,0,nxanza*sizeof(float));
		cudaMalloc(&filter_signz_d,nxanza*sizeof(float));cudaMemset(filter_signz_d,0,nxanza*sizeof(float));*/
//define poynting vector for x or z component
		/*float *poyn_x_d,*poyn_z_d,*poyn_rx_d,*poyn_rz_d;
		cudaMalloc(&poyn_x_d, nxanza*sizeof(float));
		cudaMalloc(&poyn_z_d, nxanza*sizeof(float));
		cudaMalloc(&poyn_rx_d,nxanza*sizeof(float));
		cudaMalloc(&poyn_rz_d,nxanza*sizeof(float));*/
		float *poyn_px_d,*poyn_pz_d,*poyn_sx_d,*poyn_sz_d;
		cudaMalloc(&poyn_px_d, nxanza*sizeof(float));		cudaMemset(poyn_px_d,0,nxanza*sizeof(float));
		cudaMalloc(&poyn_pz_d, nxanza*sizeof(float));		cudaMemset(poyn_pz_d,0,nxanza*sizeof(float));
		cudaMalloc(&poyn_sx_d,nxanza*sizeof(float));		cudaMemset(poyn_sx_d,0,nxanza*sizeof(float));
		cudaMalloc(&poyn_sz_d,nxanza*sizeof(float));		cudaMemset(poyn_sz_d,0,nxanza*sizeof(float));

		float *poyn_rpx_d,*poyn_rpz_d,*poyn_rsx_d,*poyn_rsz_d;
		cudaMalloc(&poyn_rpx_d, nxanza*sizeof(float));		cudaMemset(poyn_rpx_d,0,nxanza*sizeof(float));
		cudaMalloc(&poyn_rpz_d, nxanza*sizeof(float));		cudaMemset(poyn_rpz_d,0,nxanza*sizeof(float));
		cudaMalloc(&poyn_rsx_d,nxanza*sizeof(float));		cudaMemset(poyn_rsx_d,0,nxanza*sizeof(float));
		cudaMalloc(&poyn_rsz_d,nxanza*sizeof(float));		cudaMemset(poyn_rsz_d,0,nxanza*sizeof(float));

		float *direction_px_d,*direction_pz_d,*direction_sx_d,*direction_sz_d;
		cudaMalloc(&direction_px_d, nxanza*sizeof(float));		cudaMemset(direction_px_d,0,nxanza*sizeof(float));
		cudaMalloc(&direction_pz_d, nxanza*sizeof(float));		cudaMemset(direction_pz_d,0,nxanza*sizeof(float));
		cudaMalloc(&direction_sx_d,nxanza*sizeof(float));		cudaMemset(direction_sx_d,0,nxanza*sizeof(float));
		cudaMalloc(&direction_sz_d,nxanza*sizeof(float));		cudaMemset(direction_sz_d,0,nxanza*sizeof(float));

		float *direction_rpx_d,*direction_rpz_d,*direction_rsx_d,*direction_rsz_d;
		cudaMalloc(&direction_rpx_d, nxanza*sizeof(float));	cudaMemset(direction_rpx_d,0,nxanza*sizeof(float));
		cudaMalloc(&direction_rpz_d, nxanza*sizeof(float));	cudaMemset(direction_rpz_d,0,nxanza*sizeof(float));
		cudaMalloc(&direction_rsx_d,nxanza*sizeof(float));		cudaMemset(direction_rsx_d,0,nxanza*sizeof(float));
		cudaMalloc(&direction_rsz_d,nxanza*sizeof(float));		cudaMemset(direction_rsz_d,0,nxanza*sizeof(float));
///////////2017年03月12日 星期日 10时51分29秒Traditional   ERTM 

		gpumem += (nxanza*90)*sizeof(float)/1024.0/1024.0;/////for check 2017年07月27日 星期四 10时11分40秒

/////for check 2017年07月27日 星期四 10时11分40秒
		gpumem += (nxanza*65)*sizeof(float)/1024.0/1024.0;/////for check 2017年07月27日 星期四 10时11分40秒
		
		//check_number,check_interval,check_residual;

		gpumem_residual=2700-gpumem;

		change=1.0*variable_number*lt*nx_size*nz*4.0/1024.0/1024.0;
		warn("GPU memory cost: %f (MB).",gpumem);
		warn("gpumem_residual memory cost: %f (MB).",gpumem_residual);
		warn("all memory cost: %f (MB).",change);

		if(change<=gpumem_residual)
		{
			check_number=1;
			check_interval=lt;
			check_residual=0;
		}

		if(change>gpumem_residual)
		{
			check_number=int(1.0*change/gpumem_residual+1);

			check_interval=int(1.0*lt/check_number);

			check_residual=int(lt-1.0*check_number*check_interval);
		}

		float *save_vx_x_d,*save_vx_z_d,*save_vx_t_d,*save_vz_x_d,*save_vz_z_d,*save_vz_t_d;
		float *save_vxp_d,*save_vxs_d,*save_vzp_d,*save_vzs_d;
		float *save_tp_d,*save_p_d,*save_s_d;		

		cudaMalloc(&save_vx_x_d,check_interval*nx_size_nz*sizeof(float));	cudaMemset(save_vx_x_d,0,check_interval*nx_size_nz*sizeof(float));
		cudaMalloc(&save_vx_z_d,check_interval*nx_size_nz*sizeof(float));	cudaMemset(save_vx_z_d,0,check_interval*nx_size_nz*sizeof(float));
		cudaMalloc(&save_vx_t_d,check_interval*nx_size_nz*sizeof(float));	cudaMemset(save_vx_t_d,0,check_interval*nx_size_nz*sizeof(float));
		cudaMalloc(&save_vz_x_d,check_interval*nx_size_nz*sizeof(float));	cudaMemset(save_vz_x_d,0,check_interval*nx_size_nz*sizeof(float));
		cudaMalloc(&save_vz_z_d,check_interval*nx_size_nz*sizeof(float));	cudaMemset(save_vz_z_d,0,check_interval*nx_size_nz*sizeof(float));
		cudaMalloc(&save_vz_t_d,check_interval*nx_size_nz*sizeof(float));	cudaMemset(save_vz_t_d,0,check_interval*nx_size_nz*sizeof(float));

		cudaMalloc(&save_vxp_d,check_interval*nx_size_nz*sizeof(float));		cudaMemset(save_vxp_d,0,check_interval*nx_size_nz*sizeof(float));
		cudaMalloc(&save_vxs_d,check_interval*nx_size_nz*sizeof(float));		cudaMemset(save_vxs_d,0,check_interval*nx_size_nz*sizeof(float));
		cudaMalloc(&save_vzp_d,check_interval*nx_size_nz*sizeof(float));		cudaMemset(save_vzp_d,0,check_interval*nx_size_nz*sizeof(float));
		cudaMalloc(&save_vzs_d,check_interval*nx_size_nz*sizeof(float));		cudaMemset(save_vzs_d,0,check_interval*nx_size_nz*sizeof(float));
	
		cudaMalloc(&save_tp_d,check_interval*nx_size_nz*sizeof(float));		cudaMemset(save_tp_d,0,check_interval*nx_size_nz*sizeof(float));
		cudaMalloc(&save_p_d,check_interval*nx_size_nz*sizeof(float));		cudaMemset(save_p_d,0,check_interval*nx_size_nz*sizeof(float));
		cudaMalloc(&save_s_d,check_interval*nx_size_nz*sizeof(float));		cudaMemset(save_s_d,0,check_interval*nx_size_nz*sizeof(float));

		float *save_h;
		save_h=alloc1float(check_interval*nx_size_nz);				memset(save_h,0,check_interval*nx_size_nz*sizeof(float));				

		warn("check_number=%d,check_interval=%d,check_residual=%d\n.",check_number,check_interval,check_residual);

		float *packaging_d;
		cudaMalloc(&packaging_d,20*sizeof(float));		cudaMemset(packaging_d,0,20*sizeof(float));
/////for check 2017年07月27日 星期四 10时11分40秒

//////////////////////////To bring in large amplitude errors/////for correlation_misfit
		float *error_random,*error_random_d;/////for correlation_misfit
		error_random=alloc1float(receiver_num);				memset(error_random,0,receiver_num*sizeof(float));
		cudaMalloc(&error_random_d,receiver_num*sizeof(float));		cudaMemset(error_random_d,0,receiver_num*sizeof(float));
		
		cal_cpu_error_random(error_random,amplitude_error_number,receiver_num,receiver_interval);

		cudaMemcpy(error_random_d,error_random,receiver_num*sizeof(float),cudaMemcpyHostToDevice);

		write_file_1d(error_random,receiver_num,"./someoutput/error_random");
//////////////////To bring in source strength errors/////for correlation_misfit
		float *shot_scale_h,*shot_scale_d;
		shot_scale_h=alloc1float(shot_num);				memset(shot_scale_h,0,shot_num*sizeof(float));
		cudaMalloc(&shot_scale_d,shot_num*sizeof(float));		cudaMemset(shot_scale_d,0,shot_num*sizeof(float));

		if(shot_scale!=0)	cal_cpu_shot_scale(shot_scale_h,shot_num,shot_scale);

		cudaMemcpy(shot_scale_d,shot_scale_h,shot_num*sizeof(float),cudaMemcpyHostToDevice);

		write_file_1d(shot_scale_h,shot_num,"./someoutput/shot_scale");		
//////////////////////////To bring in large amplitude errors/////for correlation_misfit
			
		warn("******Start to Calculate******");
		warn("nx=%d,nz=%d",nx,nz);
		warn("boundary_up=%d,boundary_down=%d,boundary_left=%d,boundary_right=%d",boundary_up,boundary_down,boundary_left,boundary_right);
		warn("nx_append=%d,nz_append=%d",nx_append,nz_append);
		//clock_t start,finish;
		//double duration;
		//time(&t1);
		//start = clock();

		if(join_wavefield==1)	
		{

			system("mkdir wavefield1");
			system("mkdir wavefield1/0");	
			system("mkdir wavefield1/1");
			system("mkdir wavefield1/2");
			system("mkdir wavefield1/3");
			system("mkdir wavefield1/4");
			system("mkdir wavefield1/5");
		}

		cuda_packaging<<<10,1>>>(packaging_d,dx,dz,dt,coe_x,coe_z,nx_append_radius,nz_append_radius,coe_opt_d);
		/////////////////////////////////QQQQQQQQQQQQQQQ2017年07月27日 星期四 10时11分40秒 

		cudaEventRecord(start);/* record starting time */
		int ishot=0,mark;
		wavelet_half=0;
		while(ishot<shot_num&&join_shot==0)//////join_shot=0 denote that there is no obs shots, we need simulate obs shots
		{					
				if(fmod((ishot*1.0),40.0)==0)		
				{
					//if(fmod((it+1.0)-wavelet_half,1000.0)==0) warn("shot=%d,step=forward 1,it=%d",ishot+1,(it+1)-wavelet_half);
					warn("ishot=%d",ishot+1);
					warn("shot cord (x:%d z:%d)",source_x_cord[ishot],shot_depth);
					warn("receiver cord [start (x:%d z:%d ) interval:%d number:%d]",receiver_x_cord[ishot],receiver_depth,receiver_interval,receiver_num);
				}

				if(cut_direct_wave==0||cut_direct_wave==1)
				{
					/////////////////////////////////////get constant mode;				
					cuda_get_constant_mode<<<dimGrid,dimBlock>>>(velocity_all_d,velocity_d,nx_append,nz_append);
					cuda_get_constant_mode<<<dimGrid,dimBlock>>>(velocity1_all_d,velocity1_d,nx_append,nz_append);
					cuda_get_constant_mode<<<dimGrid,dimBlock>>>(density_all_d,density_d,nx_append,nz_append);
					cuda_get_constant_mode<<<dimGrid,dimBlock>>>(qp_all_d,qp_d,nx_append,nz_append);
					cuda_get_constant_mode<<<dimGrid,dimBlock>>>(qs_all_d,qs_d,nx_append,nz_append);
///////////////QQQQQQQQQQQQQQQ2017年07月27日 星期四 10时11分40秒 
					cuda_cal_viscoelastic<<< dimGrid,dimBlock>>>(modul_p_d,modul_s_d,qp_d,qs_d,tao_d,strain_p_d,strain_s_d,freq,velocity_d,velocity1_d,density_d,nx_append,nz_append);
					
					memset((void *)(wf_append),0,nxanza*sizeof(float));
					cudaMemset(vx1_d,0,nxanza*sizeof(float));
					cudaMemset(vz1_d,0,nxanza*sizeof(float));
					cudaMemset(txx1_d,0,nxanza*sizeof(float));
					cudaMemset(tzz1_d,0,nxanza*sizeof(float));
					cudaMemset(txz1_d,0,nxanza*sizeof(float));
					cudaMemset(vx2_d,0,nxanza*sizeof(float));
					cudaMemset(vz2_d,0,nxanza*sizeof(float));
					cudaMemset(txx2_d,0,nxanza*sizeof(float));
					cudaMemset(tzz2_d,0,nxanza*sizeof(float));
					cudaMemset(txz2_d,0,nxanza*sizeof(float));

					cudaMemset(tp2_d,0,nxanza*sizeof(float));
					cudaMemset(tp1_d,0,nxanza*sizeof(float));
					cudaMemset(vxp2_d,0,nxanza*sizeof(float));
					cudaMemset(vxp1_d,0,nxanza*sizeof(float));
					cudaMemset(vzp2_d,0,nxanza*sizeof(float));
					cudaMemset(vzp1_d,0,nxanza*sizeof(float));
					cudaMemset(vxs2_d,0,nxanza*sizeof(float));
					cudaMemset(vxs1_d,0,nxanza*sizeof(float));
					cudaMemset(vzs2_d,0,nxanza*sizeof(float));
					cudaMemset(vzs1_d,0,nxanza*sizeof(float));

					cudaMemset(mem_p1_d,0,nxanza*sizeof(float));
					cudaMemset(mem_xx1_d,0,nxanza*sizeof(float));
					cudaMemset(mem_zz1_d,0,nxanza*sizeof(float));
					cudaMemset(mem_xz1_d,0,nxanza*sizeof(float));
					cudaMemset(mem_p2_d,0,nxanza*sizeof(float));
					cudaMemset(mem_xx2_d,0,nxanza*sizeof(float));
					cudaMemset(mem_zz2_d,0,nxanza*sizeof(float));
					cudaMemset(mem_xz2_d,0,nxanza*sizeof(float));///////////////QQQQQQQQQQQQQQQ2017年07月27日 星期四 10时11分40秒		
				
	    			for(int it=0;it<lt+wavelet_half;it++)
					{
							//if(fmod((it+1.0)-wavelet_half,1000.0)==0) warn("shot=%d,step=forward 1,it=%d",ishot+1,(it+1)-wavelet_half);
							
							if(it<wavelet_length)
							{
								//add_source<<<1,1>>>(txx1_d,tzz1_d,wavelet_d,source_x_cord[ishot],shot_depth,it,boundary_up,boundary_left,nz_append);
								//add_source<<<1,1>>>(txx1_d,tzz1_d,wavelet_d,source_x_cord[ishot],source_z_cord[ishot],it,boundary_up,boundary_left,nz_append);////for vsp or surface 2017年03月14日 星期二 08时41分20秒 

								add_source<<<1,1>>>(txx1_d,tzz1_d,wavelet_d,source_x_cord[ishot]-receiver_x_cord[ishot],source_z_cord[ishot],it,boundary_up,boundary_left,nz_append);////for new acqusition way 2017年08月17日 星期四 09时10分03秒 
							}

							fwd_vx<<<dimGrid,dimBlock>>>(vx2_d,vx1_d,txx1_d,txz1_d,attenuation_d,dx,dz,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,density_d);

							fwd_vz<<<dimGrid,dimBlock>>>(vz2_d,vz1_d,tzz1_d,txz1_d,attenuation_d,dx,dz,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,density_d);

							//fwd_vxp_vzp<<<dimGrid,dimBlock>>>(vxp2_d,vxp1_d,vzp2_d,vzp1_d,tp1_d,coe_x,coe_z,dx,dz,dt,attenuation_d,coe_opt_d,nx_append_radius,nz_append_radius,density_d);

							//vp_vs<<<dimGrid,dimBlock>>>(vx2_d,vz2_d,vxp2_d,vzp2_d,vxs2_d,vzs2_d,nx_append_radius,nz_append_radius);

							if(modeling_type==0)	fwd_txxzzxzpp<<<dimGrid,dimBlock>>>(tp2_d,tp1_d,txx2_d,txx1_d,tzz2_d,tzz1_d,txz2_d,txz1_d,vx2_d,vz2_d,velocity_d,velocity1_d,attenuation_d,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,density_d);

							//else	fwd_txxzzxzpp_viscoelastic_and_memory<<<dimGrid,dimBlock>>>(tp2_d,tp1_d,txx2_d,txx1_d,tzz2_d,tzz1_d,txz2_d,txz1_d,vx2_d,vz2_d,modul_p_d,modul_s_d,attenuation_d,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,density_d,mem_p2_d,mem_p1_d,mem_xx2_d,mem_xx1_d,mem_zz2_d,mem_zz1_d,mem_xz2_d,mem_xz1_d,velocity_d,velocity1_d,tao_d,strain_p_d,strain_s_d);

							//fwd_memory<<<dimGrid,dimBlock>>>(mem_p2_d,mem_p1_d,mem_xx2_d,mem_xx1_d,mem_zz2_d,mem_zz1_d,mem_xz2_d,mem_xz1_d,vx2_d,vz2_d,attenuation_d,dx,dz,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,density_d,velocity_d,velocity1_d,tao_d,strain_p_d,strain_s_d);

							//fwd_txxzzxzpp_viscoelastic<<<dimGrid,dimBlock>>>(tp2_d,tp1_d,txx2_d,txx1_d,tzz2_d,tzz1_d,txz2_d,txz1_d,vx2_d,vz2_d,modul_p_d,modul_s_d,attenuation_d,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,density_d,mem_p1_d,mem_xx1_d,mem_zz1_d,mem_xz1_d);

							else	fwd_txxzzxzpp_viscoelastic_and_memory_3parameterization<<<dimGrid,dimBlock>>>(tp2_d,tp1_d,txx2_d,txx1_d,tzz2_d,tzz1_d,txz2_d,txz1_d,vx2_d,vz2_d,modul_p_d,modul_s_d,attenuation_d,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,density_d,mem_p2_d,mem_p1_d,mem_xx2_d,mem_xx1_d,mem_zz2_d,mem_zz1_d,mem_xz2_d,mem_xz1_d,velocity_d,velocity1_d,tao_d,strain_p_d,strain_s_d);

							//decom<<<dimGrid,dimBlock>>>(vx2_d,vz2_d,p_d,s_d,coe_opt_d,nx_append,nz_append,dx,dz);

							//decom_new<<<dimGrid,dimBlock>>>(vx2_d,vz2_d,p_d,s_d,velocity_d,velocity1_d,coe_opt_d,nx_append_radius,nz_append_radius,dx,dz);

							if(it>=wavelet_half&&it<(lt+wavelet_half))
							{
								if(receiver_offset==0)
								{
									write_shot_x_z<<<receiver_num,1>>>(vx2_d,cal_shot_x_d,it-wavelet_half,lt,receiver_num,receiver_x_cord[ishot],receiver_interval,receiver_z_cord[ishot],receiver_z_interval,boundary_left,boundary_up,nz_append);///for vsp 2017年03月14日 星期二 08时46分12秒 
									write_shot_x_z<<<receiver_num,1>>>(vz2_d,cal_shot_z_d,it-wavelet_half,lt,receiver_num,receiver_x_cord[ishot],receiver_interval,receiver_z_cord[ishot],receiver_z_interval,boundary_left,boundary_up,nz_append);///for vsp 2017年03月14日 星期二 08时46分12秒 
								}

								else
								{
									write_shot_x_z_acqusition<<<receiver_num,1>>>(vx2_d,cal_shot_x_d,it-wavelet_half,lt,receiver_num,receiver_x_cord[ishot],receiver_interval,receiver_z_cord[ishot],receiver_z_interval,boundary_left,boundary_up,nz_append);///for vsp 2017年03月14日 星期二 08时46分12秒 
									write_shot_x_z_acqusition<<<receiver_num,1>>>(vz2_d,cal_shot_z_d,it-wavelet_half,lt,receiver_num,receiver_x_cord[ishot],receiver_interval,receiver_z_cord[ishot],receiver_z_interval,boundary_left,boundary_up,nz_append);///for vsp 2017年03月14日 星期二 08时46分12秒 
								}

							}
						
							rep=vx1_d;vx1_d=vx2_d;vx2_d=rep;
							rep=vz1_d;vz1_d=vz2_d;vz2_d=rep;
							rep=txx1_d;txx1_d=txx2_d;txx2_d=rep;
							rep=tzz1_d;tzz1_d=tzz2_d;tzz2_d=rep;
							rep=txz1_d;txz1_d=txz2_d;txz2_d=rep;

							rep=tp1_d;tp1_d=tp2_d;tp2_d=rep;
							rep=vxp1_d;vxp1_d=vxp2_d;vxp2_d=rep;
							rep=vzp1_d;vzp1_d=vzp2_d;vzp2_d=rep;
							rep=vxs1_d;vxs1_d=vxs2_d;vxs2_d=rep;
							rep=vzs1_d;vzs1_d=vzs2_d;vzs2_d=rep;

							rep=mem_p1_d;mem_p1_d=mem_p2_d;mem_p2_d=rep;
							rep=mem_xx1_d;mem_xx1_d=mem_xx2_d;mem_xx2_d=rep;
							rep=mem_zz1_d;mem_zz1_d=mem_zz2_d;mem_zz2_d=rep;
							rep=mem_xz1_d;mem_xz1_d=mem_xz2_d;mem_xz2_d=rep;
					}

						if(ishot%40==0)
						{
							cudaMemcpy(shotgather,cal_shot_x_d,lt_rec*sizeof(float),cudaMemcpyDeviceToHost);
							sprintf(filename,"./someoutput/bin/direct1_obs_shot_x_%d",ishot+1);
							write_file_1d(shotgather,lt_rec,filename);
							cudaMemcpy(shotgather,cal_shot_z_d,lt_rec*sizeof(float),cudaMemcpyDeviceToHost);
							sprintf(filename,"./someoutput/bin/direct1_obs_shot_z_%d",ishot+1);
							write_file_1d(shotgather,lt_rec,filename);
						}
				}				
					
/////////////////////////////////////get correct vp;				
				cuda_get_partly_mode_boundary<<<dimGrid_new,dimBlock>>>(velocity_all_d,wf_d,nx,nz,receiver_x_cord[ishot],receiver_interval,receiver_num,nx_append_new,nz_append,boundary_left,boundary_up);
				cuda_cal_expand<<<dimGrid,dimBlock>>>(velocity_d,wf_d,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);

				cudaMemcpy(wf,wf_d,nx_size_nz*sizeof(float),cudaMemcpyDeviceToHost);
				sprintf(filename,"./someoutput/cut-vp-%d.bin",ishot+1);
				write_file_1d(wf,nx_size_nz,filename);

				cudaMemcpy(wf_append,velocity_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
				sprintf(filename,"./someoutput/vp-%d.bin",ishot+1);
				write_file_1d(wf_append,nxanza,filename);
/////////////////////////////////////get correct vs;				
				cuda_get_partly_mode_boundary<<<dimGrid_new,dimBlock>>>(velocity1_all_d,wf_d,nx,nz,receiver_x_cord[ishot],receiver_interval,receiver_num,nx_append_new,nz_append,boundary_left,boundary_up);
				cuda_cal_expand<<<dimGrid,dimBlock>>>(velocity1_d,wf_d,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);

/////////////////////////////////////get correct density;				
				cuda_get_partly_mode_boundary<<<dimGrid_new,dimBlock>>>(density_all_d,wf_d,nx,nz,receiver_x_cord[ishot],receiver_interval,receiver_num,nx_append_new,nz_append,boundary_left,boundary_up);
				cuda_cal_expand<<<dimGrid,dimBlock>>>(density_d,wf_d,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);

/////////////////////////////////////get correct qp;				
				cuda_get_partly_mode_boundary<<<dimGrid_new,dimBlock>>>(qp_all_d,wf_d,nx,nz,receiver_x_cord[ishot],receiver_interval,receiver_num,nx_append_new,nz_append,boundary_left,boundary_up);
				cuda_cal_expand<<<dimGrid,dimBlock>>>(qp_d,wf_d,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);

/////////////////////////////////////get correct qs;				
				cuda_get_partly_mode_boundary<<<dimGrid_new,dimBlock>>>(qs_all_d,wf_d,nx,nz,receiver_x_cord[ishot],receiver_interval,receiver_num,nx_append_new,nz_append,boundary_left,boundary_up);
				cuda_cal_expand<<<dimGrid,dimBlock>>>(qs_d,wf_d,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);

				/*if((receiver_offset!=0)||(offset_left[ishot]>receiver_offset)||(offset_right[ishot]>receiver_offset))
				{								
					cuda_expand_acqusition_left_and_right<<<dimGrid_new,dimBlock>>>(velocity_d,offset_left[ishot],offset_right[ishot],source_x_cord[ishot],receiver_offset,nx_size,nx_append,nz_append,boundary_left,boundary_up);
					cudaMemcpy(wf_append,velocity_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
					sprintf(filename,"./someoutput/vp-new-%d.bin",ishot+1);
					write_file_1d(wf_append,nxanza,filename);

					cuda_expand_acqusition_left_and_right<<<dimGrid_new,dimBlock>>>(velocity1_d,offset_left[ishot],offset_right[ishot],source_x_cord[ishot],receiver_offset,nx_size,nx_append,nz_append,boundary_left,boundary_up);

					cuda_expand_acqusition_left_and_right<<<dimGrid_new,dimBlock>>>(density_d,offset_left[ishot],offset_right[ishot],source_x_cord[ishot],receiver_offset,nx_size,nx_append,nz_append,boundary_left,boundary_up);

					cuda_expand_acqusition_left_and_right<<<dimGrid_new,dimBlock>>>(qp_d,offset_left[ishot],offset_right[ishot],source_x_cord[ishot],receiver_offset,nx_size,nx_append,nz_append,boundary_left,boundary_up);

					cuda_expand_acqusition_left_and_right<<<dimGrid_new,dimBlock>>>(qs_d,offset_left[ishot],offset_right[ishot],source_x_cord[ishot],receiver_offset,nx_size,nx_append,nz_append,boundary_left,boundary_up);						
				}*/
///////////////QQQQQQQQQQQQQQQ2017年07月27日 星期四 10时11分40秒 
				cuda_cal_viscoelastic<<< dimGrid,dimBlock>>>(modul_p_d,modul_s_d,qp_d,qs_d,tao_d,strain_p_d,strain_s_d,freq,velocity_d,velocity1_d,density_d,nx_append,nz_append);	
				
				memset((void *)(wf_append),0,nxanza*sizeof(float));
				cudaMemset(vx1_d,0,nxanza*sizeof(float));
				cudaMemset(vz1_d,0,nxanza*sizeof(float));
				cudaMemset(txx1_d,0,nxanza*sizeof(float));
				cudaMemset(tzz1_d,0,nxanza*sizeof(float));
				cudaMemset(txz1_d,0,nxanza*sizeof(float));
				cudaMemset(vx2_d,0,nxanza*sizeof(float));
				cudaMemset(vz2_d,0,nxanza*sizeof(float));
				cudaMemset(txx2_d,0,nxanza*sizeof(float));
				cudaMemset(tzz2_d,0,nxanza*sizeof(float));
				cudaMemset(txz2_d,0,nxanza*sizeof(float));

				cudaMemset(tp2_d,0,nxanza*sizeof(float));
				cudaMemset(tp1_d,0,nxanza*sizeof(float));
				cudaMemset(vxp2_d,0,nxanza*sizeof(float));
				cudaMemset(vxp1_d,0,nxanza*sizeof(float));
				cudaMemset(vzp2_d,0,nxanza*sizeof(float));
				cudaMemset(vzp1_d,0,nxanza*sizeof(float));
				cudaMemset(vxs2_d,0,nxanza*sizeof(float));
				cudaMemset(vxs1_d,0,nxanza*sizeof(float));
				cudaMemset(vzs2_d,0,nxanza*sizeof(float));
				cudaMemset(vzs1_d,0,nxanza*sizeof(float));

				cudaMemset(mem_p1_d,0,nxanza*sizeof(float));
				cudaMemset(mem_xx1_d,0,nxanza*sizeof(float));
				cudaMemset(mem_zz1_d,0,nxanza*sizeof(float));
				cudaMemset(mem_xz1_d,0,nxanza*sizeof(float));
				cudaMemset(mem_p2_d,0,nxanza*sizeof(float));
				cudaMemset(mem_xx2_d,0,nxanza*sizeof(float));
				cudaMemset(mem_zz2_d,0,nxanza*sizeof(float));
				cudaMemset(mem_xz2_d,0,nxanza*sizeof(float));///////////////QQQQQQQQQQQQQQQ2017年07月27日 星期四 10时11分40秒		
			
    			for(int it=0;it<lt+wavelet_half;it++)
				{
						//if(fmod((it+1.0)-wavelet_half,1000.0)==0) warn("shot=%d,step=forward 1,it=%d",ishot+1,(it+1)-wavelet_half);
						
						if(it<wavelet_length)
						{
							//add_source<<<1,1>>>(txx1_d,tzz1_d,wavelet_d,source_x_cord[ishot],shot_depth,it,boundary_up,boundary_left,nz_append);
							//add_source<<<1,1>>>(txx1_d,tzz1_d,wavelet_d,source_x_cord[ishot],source_z_cord[ishot],it,boundary_up,boundary_left,nz_append);////for vsp or surface 2017年03月14日 星期二 08时41分20秒 

							add_source<<<1,1>>>(txx1_d,tzz1_d,wavelet_d,source_x_cord[ishot]-receiver_x_cord[ishot],source_z_cord[ishot],it,boundary_up,boundary_left,nz_append);////for new acqusition way 2017年08月17日 星期四 09时10分03秒 
						}

						fwd_vx<<<dimGrid,dimBlock>>>(vx2_d,vx1_d,txx1_d,txz1_d,attenuation_d,dx,dz,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,density_d);

						fwd_vz<<<dimGrid,dimBlock>>>(vz2_d,vz1_d,tzz1_d,txz1_d,attenuation_d,dx,dz,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,density_d);

						//fwd_vxp_vzp<<<dimGrid,dimBlock>>>(vxp2_d,vxp1_d,vzp2_d,vzp1_d,tp1_d,coe_x,coe_z,dx,dz,dt,attenuation_d,coe_opt_d,nx_append_radius,nz_append_radius,density_d);

						//vp_vs<<<dimGrid,dimBlock>>>(vx2_d,vz2_d,vxp2_d,vzp2_d,vxs2_d,vzs2_d,nx_append_radius,nz_append_radius);

						if(modeling_type==0)	fwd_txxzzxzpp<<<dimGrid,dimBlock>>>(tp2_d,tp1_d,txx2_d,txx1_d,tzz2_d,tzz1_d,txz2_d,txz1_d,vx2_d,vz2_d,velocity_d,velocity1_d,attenuation_d,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,density_d);

						//else	fwd_txxzzxzpp_viscoelastic_and_memory<<<dimGrid,dimBlock>>>(tp2_d,tp1_d,txx2_d,txx1_d,tzz2_d,tzz1_d,txz2_d,txz1_d,vx2_d,vz2_d,modul_p_d,modul_s_d,attenuation_d,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,density_d,mem_p2_d,mem_p1_d,mem_xx2_d,mem_xx1_d,mem_zz2_d,mem_zz1_d,mem_xz2_d,mem_xz1_d,velocity_d,velocity1_d,tao_d,strain_p_d,strain_s_d);

						//fwd_memory<<<dimGrid,dimBlock>>>(mem_p2_d,mem_p1_d,mem_xx2_d,mem_xx1_d,mem_zz2_d,mem_zz1_d,mem_xz2_d,mem_xz1_d,vx2_d,vz2_d,attenuation_d,dx,dz,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,density_d,velocity_d,velocity1_d,tao_d,strain_p_d,strain_s_d);

						//fwd_txxzzxzpp_viscoelastic<<<dimGrid,dimBlock>>>(tp2_d,tp1_d,txx2_d,txx1_d,tzz2_d,tzz1_d,txz2_d,txz1_d,vx2_d,vz2_d,modul_p_d,modul_s_d,attenuation_d,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,density_d,mem_p1_d,mem_xx1_d,mem_zz1_d,mem_xz1_d);

						else	fwd_txxzzxzpp_viscoelastic_and_memory_3parameterization<<<dimGrid,dimBlock>>>(tp2_d,tp1_d,txx2_d,txx1_d,tzz2_d,tzz1_d,txz2_d,txz1_d,vx2_d,vz2_d,modul_p_d,modul_s_d,attenuation_d,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,density_d,mem_p2_d,mem_p1_d,mem_xx2_d,mem_xx1_d,mem_zz2_d,mem_zz1_d,mem_xz2_d,mem_xz1_d,velocity_d,velocity1_d,tao_d,strain_p_d,strain_s_d);

						//sum_poynting<<<dimGrid,dimBlock>>>(poyn_px_d,poyn_pz_d,poyn_sx_d,poyn_sz_d,vxp2_d,vzp2_d,vxs2_d,vzs2_d,txx2_d,tzz2_d,txz2_d,tp2_d,nx_append_radius,nz_append_radius);
							
						//decom<<<dimGrid,dimBlock>>>(vx2_d,vz2_d,p_d,s_d,coe_opt_d,nx_append,nz_append,dx,dz);

						//decom_new<<<dimGrid,dimBlock>>>(vx2_d,vz2_d,p_d,s_d,velocity_d,velocity1_d,coe_opt_d,nx_append_radius,nz_append_radius,dx,dz);
						if(0==(it-wavelet_half)%100&&join_wavefield==1&&iter==0)
						{
							cudaMemcpy(wf_append,vx2_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
							sprintf(filename,"./wavefield1/0/vx-%d-shot_%d",ishot+1,it-wavelet_half);
							write_file_1d(wf_append,nxanza,filename);
							exchange(wf_append,wf,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
							write_file_1d(wf,nx_size_nz,filename);
										
							cudaMemcpy(wf_append,vxp2_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
							sprintf(filename,"./wavefield1/0/vxp-%d-shot_%d",ishot+1,it-wavelet_half);
							write_file_1d(wf_append,nxanza,filename);
							exchange(wf_append,wf,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
							write_file_1d(wf,nx_size_nz,filename);
						}

						if(it>=wavelet_half&&it<(lt+wavelet_half))
						{
							//write_shot<<<receiver_num,1>>>(vx2_d,vz2_d,obs_shot_x_d,obs_shot_z_d,it-wavelet_half,lt,receiver_num,receiver_depth,receiver_x_cord[ishot],receiver_interval,boundary_left,boundary_up,nz_append,dx,dt,source_x_cord[ishot],velocity_d,wavelet_half);
							if(receiver_offset==0)
							{
								write_shot_x_z<<<receiver_num,1>>>(vx2_d,obs_shot_x_d,it-wavelet_half,lt,receiver_num,receiver_x_cord[ishot],receiver_interval,receiver_z_cord[ishot],receiver_z_interval,boundary_left,boundary_up,nz_append);///for vsp 2017年03月14日 星期二 08时46分12秒 
								write_shot_x_z<<<receiver_num,1>>>(vz2_d,obs_shot_z_d,it-wavelet_half,lt,receiver_num,receiver_x_cord[ishot],receiver_interval,receiver_z_cord[ishot],receiver_z_interval,boundary_left,boundary_up,nz_append);///for vsp 2017年03月14日 星期二 08时46分12秒 
							}
							else
							{
								write_shot_x_z_acqusition<<<receiver_num,1>>>(vx2_d,obs_shot_x_d,it-wavelet_half,lt,receiver_num,receiver_x_cord[ishot],receiver_interval,receiver_z_cord[ishot],receiver_z_interval,boundary_left,boundary_up,nz_append);///for vsp 2017年03月14日 星期二 08时46分12秒 
								write_shot_x_z_acqusition<<<receiver_num,1>>>(vz2_d,obs_shot_z_d,it-wavelet_half,lt,receiver_num,receiver_x_cord[ishot],receiver_interval,receiver_z_cord[ishot],receiver_z_interval,boundary_left,boundary_up,nz_append);///for vsp 2017年03月14日 星期二 08时46分12秒 
							}
//////////////vsp_2
							if(vsp_2!=0)
							{
								write_shot_x_z<<<receiver_num,1>>>(vx2_d,obs_shot_x_d_2,it-wavelet_half,lt,receiver_num_2,receiver_x_cord_2[ishot],receiver_interval_2,receiver_z_cord_2[ishot],receiver_z_interval_2,boundary_left,boundary_up,nz_append);///for vsp 2017年03月17日 星期二 08时46分12秒 
								write_shot_x_z<<<receiver_num,1>>>(vz2_d,obs_shot_z_d_2,it-wavelet_half,lt,receiver_num_2,receiver_x_cord_2[ishot],receiver_interval_2,receiver_z_cord_2[ishot],receiver_z_interval_2,boundary_left,boundary_up,nz_append);///for vsp 2017年03月17日 星期二 08时46分12秒 
							}
						}
					
						rep=vx1_d;vx1_d=vx2_d;vx2_d=rep;
						rep=vz1_d;vz1_d=vz2_d;vz2_d=rep;
						rep=txx1_d;txx1_d=txx2_d;txx2_d=rep;
						rep=tzz1_d;tzz1_d=tzz2_d;tzz2_d=rep;
						rep=txz1_d;txz1_d=txz2_d;txz2_d=rep;

						rep=tp1_d;tp1_d=tp2_d;tp2_d=rep;
						rep=vxp1_d;vxp1_d=vxp2_d;vxp2_d=rep;
						rep=vzp1_d;vzp1_d=vzp2_d;vzp2_d=rep;
						rep=vxs1_d;vxs1_d=vxs2_d;vxs2_d=rep;
						rep=vzs1_d;vzs1_d=vzs2_d;vzs2_d=rep;
										
						rep=mem_p1_d;mem_p1_d=mem_p2_d;mem_p2_d=rep;
						rep=mem_xx1_d;mem_xx1_d=mem_xx2_d;mem_xx2_d=rep;
						rep=mem_zz1_d;mem_zz1_d=mem_zz2_d;mem_zz2_d=rep;
						rep=mem_xz1_d;mem_xz1_d=mem_xz2_d;mem_xz2_d=rep;
				}
					
						if(ishot%40==0)
						{
							cudaMemcpy(shotgather,obs_shot_x_d,lt_rec*sizeof(float),cudaMemcpyDeviceToHost);
							sprintf(filename,"./someoutput/bin/direct_obs_shot_x_%d",ishot+1);
							write_file_1d(shotgather,lt_rec,filename);
							cudaMemcpy(shotgather,obs_shot_z_d,lt_rec*sizeof(float),cudaMemcpyDeviceToHost);
							sprintf(filename,"./someoutput/bin/direct_obs_shot_z_%d",ishot+1);
							write_file_1d(shotgather,lt_rec,filename);
						}

//cut direct wave
						if(cut_direct_wave==0||cut_direct_wave==1)
						{
							cal_sub_a_b_to_c<<<dimGrid_lt,dimBlock>>>(obs_shot_x_d,cal_shot_x_d,obs_shot_x_d,receiver_num,lt);

							cal_sub_a_b_to_c<<<dimGrid_lt,dimBlock>>>(obs_shot_z_d,cal_shot_z_d,obs_shot_z_d,receiver_num,lt);

							if(receiver_offset!=0)
							{							
								cauda_zero_acqusition_left_and_right<<<dimGrid_lt,dimBlock>>>(obs_shot_x_d,offset_left[ishot],offset_right[ishot],source_x_cord[ishot],receiver_offset,receiver_num,lt);
								cauda_zero_acqusition_left_and_right<<<dimGrid_lt,dimBlock>>>(obs_shot_z_d,offset_left[ishot],offset_right[ishot],source_x_cord[ishot],receiver_offset,receiver_num,lt);							
							}

							cudaMemcpy(shotgather,obs_shot_x_d,lt_rec*sizeof(float),cudaMemcpyDeviceToHost);
							sprintf(filename,"./someoutput/bin/obs_shot_x_%d",ishot+1);
							write_file_1d(shotgather,lt_rec,filename);
							cudaMemcpy(shotgather,obs_shot_z_d,lt_rec*sizeof(float),cudaMemcpyDeviceToHost);
							sprintf(filename,"./someoutput/bin/obs_shot_z_%d",ishot+1);
							write_file_1d(shotgather,lt_rec,filename);
						}

						else///for vsp 2017年03月14日 星期二 08时55分03秒 
						{
							cut_direct_new1<<<dimGrid,dimBlock>>>(obs_shot_x_d,lt,source_x_cord[ishot],shot_depth,receiver_num,receiver_depth,receiver_x_cord[ishot],receiver_interval,boundary_left,boundary_up,nz_append,dx,dz,dt,velocity_d,wavelet_half,cut_direct_wave);
							cut_direct_new1<<<dimGrid,dimBlock>>>(obs_shot_z_d,lt,source_x_cord[ishot],shot_depth,receiver_num,receiver_depth,receiver_x_cord[ishot],receiver_interval,boundary_left,boundary_up,nz_append,dx,dz,dt,velocity_d,wavelet_half,cut_direct_wave);

							if(receiver_offset!=0)
							{							
								cauda_zero_acqusition_left_and_right<<<dimGrid_lt,dimBlock>>>(obs_shot_x_d,offset_left[ishot],offset_right[ishot],source_x_cord[ishot],receiver_offset,receiver_num,lt);
								cauda_zero_acqusition_left_and_right<<<dimGrid_lt,dimBlock>>>(obs_shot_z_d,offset_left[ishot],offset_right[ishot],source_x_cord[ishot],receiver_offset,receiver_num,lt);							
							}
							cudaMemcpy(shotgather,obs_shot_x_d,lt_rec*sizeof(float),cudaMemcpyDeviceToHost);
							sprintf(filename,"./someoutput/bin/obs_shot_x_%d",ishot+1);
							write_file_1d(shotgather,lt_rec,filename);
							cudaMemcpy(shotgather,obs_shot_z_d,lt_rec*sizeof(float),cudaMemcpyDeviceToHost);
							sprintf(filename,"./someoutput/bin/obs_shot_z_%d",ishot+1);
							write_file_1d(shotgather,lt_rec,filename);
						}		

						if(amplitude_error!=0)//////////////////////////To bring in large amplitude errors/////for correlation_misfit
						{
							cuda_mul_error_random<<<dimGrid_lt,dimBlock>>>(obs_shot_x_d,error_random_d,receiver_interval,receiver_num,lt);
							cuda_mul_error_random<<<dimGrid_lt,dimBlock>>>(obs_shot_z_d,error_random_d,receiver_interval,receiver_num,lt);

							cudaMemcpy(shotgather,obs_shot_x_d,lt_rec*sizeof(float),cudaMemcpyDeviceToHost);
							sprintf(filename,"./someoutput/bin/obs_shot_x_%d",ishot+1);
							write_file_1d(shotgather,lt_rec,filename);
							cudaMemcpy(shotgather,obs_shot_z_d,lt_rec*sizeof(float),cudaMemcpyDeviceToHost);
							sprintf(filename,"./someoutput/bin/obs_shot_z_%d",ishot+1);
							write_file_1d(shotgather,lt_rec,filename);
						}

						if(shot_scale!=0)//////////////////////////To bring in large amplitude errors/////for correlation_misfit
						{
							//cuda_mul_shot_scale<<<dimGrid_lt,dimBlock>>>(obs_shot_x_d,ishot,shot_num,shot_scale,receiver_num,lt);
							//cuda_mul_shot_scale<<<dimGrid_lt,dimBlock>>>(obs_shot_z_d,ishot,shot_num,shot_scale,receiver_num,lt);

							cuda_mul_shot_scale_new<<<dimGrid_lt,dimBlock>>>(obs_shot_x_d,ishot,shot_num,shot_scale_d,receiver_num,lt);
							cuda_mul_shot_scale_new<<<dimGrid_lt,dimBlock>>>(obs_shot_z_d,ishot,shot_num,shot_scale_d,receiver_num,lt);

							cudaMemcpy(shotgather,obs_shot_x_d,lt_rec*sizeof(float),cudaMemcpyDeviceToHost);
							sprintf(filename,"./someoutput/bin/obs_shot_x_%d",ishot+1);
							write_file_1d(shotgather,lt_rec,filename);
							cudaMemcpy(shotgather,obs_shot_z_d,lt_rec*sizeof(float),cudaMemcpyDeviceToHost);
							sprintf(filename,"./someoutput/bin/obs_shot_z_%d",ishot+1);
							write_file_1d(shotgather,lt_rec,filename);
						}
//////////////////////////////////////////////////output:vsp2:::::::::
						if(vsp_2!=0)
						{
							cudaMemcpy(shotgather,obs_shot_x_d_2,lt_rec*sizeof(float),cudaMemcpyDeviceToHost);
							sprintf(filename,"./someoutput/bin/obs_shot_x_%d_2",ishot+1);
							write_file_1d(shotgather,lt_rec,filename);
							cudaMemcpy(shotgather,obs_shot_z_d_2,lt_rec*sizeof(float),cudaMemcpyDeviceToHost);
							sprintf(filename,"./someoutput/bin/obs_shot_z_%d_2",ishot+1);
							write_file_1d(shotgather,lt_rec,filename);
						}				
	
				ishot++;	
		}
		warn("modeling seismic is over");
		cudaEventRecord(stop);/* record ending time */
  		cudaEventSynchronize(stop);
  		cudaEventElapsedTime(&mstimer, start, stop);
		totaltime+=mstimer*1e-3;

		warn("modeling finished: %f (s)\n", mstimer*1e-3);////////to current step has cost times
		fprintf(logfile,"modeling finished: %f (s)\n\n", mstimer*1e-3);////////the current step  cost times
		fclose(logfile);////important


		
		warn("iterative inversion begin");	
		//for(iter=0;iter<niter;iter++)
		for(iter=iter_start;iter<niter;iter++)
		{
			if(iter==1)
			{
				cudaMemcpy(wf_append_new,d_illum_new,nxa_new_nza*sizeof(float),cudaMemcpyDeviceToHost);
				write_file_1d(wf_append_new,nxa_new_nza,"./check_file/d_illum");
			}
			
			if(iter_start!=0)
			{
				fread_file_1d(wf_append_new,nx_append_new,nz_append,"./check_file/d_illum");
				cudaMemcpy(d_illum_new,wf_append_new,nxa_new_nza*sizeof(float),cudaMemcpyHostToDevice);
			}
///////////////////2017年03月31日 星期五 08时20分42秒 
			if(iter>=5)
			{
				if(iter==5)	system("mkdir ./someoutput/save");

				rm_f_file=fopen("rm-f-shot","wb");//////cal_shot_*_iter_1  res_shot_*_iter_1
				fprintf(rm_f_file,"#!/bin/sh\n");

				fprintf(rm_f_file,"cp -r 	./someoutput/bin/res_shot_*_%d_iter_%d   ./someoutput/save\n",int((shot_num+1)/2.0),iter-2);
				fprintf(rm_f_file,"cp -r 	./someoutput/bin/cal_shot_*_%d_iter_%d   ./someoutput/save\n",int((shot_num+1)/2.0),iter-2);
				fprintf(rm_f_file,"cp -r 	./someoutput/bin/tmp_shot_*_%d_iter_%d   ./someoutput/save\n",int((shot_num+1)/2.0),iter-2);
				fprintf(rm_f_file,"cp -r 	./someoutput/bin/adj_shot_*_%d_iter_%d   ./someoutput/save\n",int((shot_num+1)/2.0),iter-2);
				//fprintf(rm_f_file,"cp -r 	./someoutput/bin/adj1_shot_*_%d_iter_%d   ./someoutput/save\n",int((shot_num+1)/2.0),iter-2);
				

				fprintf(rm_f_file,"rm -f 	./someoutput/bin/res_shot_*_iter_%d\n",iter-2);////////rm -f res
				fprintf(rm_f_file,"rm -f     ./someoutput/bin/cal_shot_*_iter_%d\n",iter-2);////////rm -f cal
				fprintf(rm_f_file,"rm -f     ./someoutput/bin/tmp_shot_*_iter_%d\n",iter-2);////////rm -f tmp
				fprintf(rm_f_file,"rm -f     ./someoutput/bin/adj_shot_*_iter_%d\n",iter-2);////////rm -f tmp
				//fprintf(rm_f_file,"rm -f     ./someoutput/bin/adj1_shot_*_iter_%d\n",iter-2);////////rm -f tmp

				fclose(rm_f_file);////important

				system("sh rm-f-shot");
			}
///////////////////2017年03月31日 星期五 08时20分42秒
			
			//warn("1\n");

			logfile=fopen("log.txt","ab");//remember to free log file			

			cudaEventRecord(start);/* record starting time */

			if(laplace_compensate!=0)
			{
				cudaMemcpy(wavelet_d,wavelet_integral,wavelet_length*sizeof(float),cudaMemcpyHostToDevice);
			}/////////////////////////tiwce integral

//////
			cudaMemset(obs_shot_x_d,0,lt_rec*sizeof(float));
			cudaMemset(obs_shot_z_d,0,lt_rec*sizeof(float));
			cudaMemset(cal_shot_x_d,0,lt_rec*sizeof(float));
			cudaMemset(cal_shot_z_d,0,lt_rec*sizeof(float));
			cudaMemset(res_shot_x_d,0,lt_rec*sizeof(float));
			cudaMemset(res_shot_z_d,0,lt_rec*sizeof(float));
			cudaMemset(tmp_shot_x_d,0,lt_rec*sizeof(float));
			cudaMemset(tmp_shot_z_d,0,lt_rec*sizeof(float));				
			cudaMemset(adj_shot_x_d,0,lt_rec*sizeof(float));
			cudaMemset(adj_shot_z_d,0,lt_rec*sizeof(float));

			cudaMemset(obs_shot_x_d_2,0,lt_rec*sizeof(float));
			cudaMemset(obs_shot_z_d_2,0,lt_rec*sizeof(float));
			cudaMemset(cal_shot_x_d_2,0,lt_rec*sizeof(float));
			cudaMemset(cal_shot_z_d_2,0,lt_rec*sizeof(float));
			cudaMemset(res_shot_x_d_2,0,lt_rec*sizeof(float));
			cudaMemset(res_shot_z_d_2,0,lt_rec*sizeof(float));
//////
////setz zero 
			if(decomposition!=0)
			{
				cudaMemset(all_grad_den_pp_d,0,nxnz*sizeof(float));
				cudaMemset(all_grad_lame1_pp_d,0,nxnz*sizeof(float));
				cudaMemset(all_grad_lame2_pp_d,0,nxnz*sizeof(float));
				cudaMemset(all_grad_den_ps_d,0,nxnz*sizeof(float));
				cudaMemset(all_grad_lame1_ps_d,0,nxnz*sizeof(float));
				cudaMemset(all_grad_lame2_ps_d,0,nxnz*sizeof(float));
				cudaMemset(all_grad_den_sp_d,0,nxnz*sizeof(float));
				cudaMemset(all_grad_lame1_sp_d,0,nxnz*sizeof(float));
				cudaMemset(all_grad_lame2_sp_d,0,nxnz*sizeof(float));
				cudaMemset(all_grad_den_ss_d,0,nxnz*sizeof(float));
				cudaMemset(all_grad_lame1_ss_d,0,nxnz*sizeof(float));
				cudaMemset(all_grad_lame2_ss_d,0,nxnz*sizeof(float));
			}

			cudaMemset(all_grad_lame11_d,0,nxnz*sizeof(float));
			cudaMemset(all_grad_lame22_d,0,nxnz*sizeof(float));
			cudaMemset(all_grad_den1_d,0,nxnz*sizeof(float));

			cudaMemset(all_grad_vp1_d,0,nxnz*sizeof(float));
			cudaMemset(all_grad_vs1_d,0,nxnz*sizeof(float));
			cudaMemset(all_grad_density1_d,0,nxnz*sizeof(float));
///exchange gradient
			cudaMemcpy(all_hydrid_grad1_d,all_hydrid_grad2_d,3*nxnz*sizeof(float), cudaMemcpyDeviceToDevice);
			cudaMemset(all_hydrid_grad2_d,0,3*nxnz*sizeof(float));//////hybrid conjugated gradient method for exchange
///exchange gradient

///set_zero for objetive vaule
			cudaMemset(obj_parameter_d,0,3*sizeof(float));
			cudaMemset(obj_d,0,3*sizeof(float));
			memset(obj_h,0,3*sizeof(float));
///set_zero for objetive vaule

			ishot=0;
			while(ishot<shot_num)
			{		
				if(fmod((ishot*1.0),40.0)==0)		
				{
					//if(fmod((it+1.0)-wavelet_half,1000.0)==0) warn("shot=%d,step=forward 1,it=%d",ishot+1,(it+1)-wavelet_half);
					warn("ishot=%d",ishot+1);
					warn("shot cord (x:%d z:%d)",source_x_cord[ishot],shot_depth);
					warn("receiver cord [start (x:%d z:%d ) interval:%d number:%d]",receiver_x_cord[ishot],receiver_depth,receiver_interval,receiver_num);
					warn("imaging scope(start:%d size:%d end:%d)",imaging_start[ishot],imaging_size[ishot],imaging_end[ishot]);
				}

/////////////////////////////////////get smooth vp;				
				cuda_get_partly_mode_boundary<<<dimGrid_new,dimBlock>>>(s_velocity_all_d,wf_d,nx,nz,receiver_x_cord[ishot],receiver_interval,receiver_num,nx_append_new,nz_append,boundary_left,boundary_up);
				cuda_cal_expand<<<dimGrid,dimBlock>>>(s_velocity_d,wf_d,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);

				cudaMemcpy(wf,wf_d,nx_size_nz*sizeof(float),cudaMemcpyDeviceToHost);
				sprintf(filename,"./someoutput/cut-vp-s-%d.bin",ishot+1);
				write_file_1d(wf,nx_size_nz,filename);

				cudaMemcpy(wf_append,s_velocity_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
				sprintf(filename,"./someoutput/vp-s-%d.bin",ishot+1);
				write_file_1d(wf_append,nxanza,filename);

/////////////////////////////////////get smooth vs;				
				cuda_get_partly_mode_boundary<<<dimGrid_new,dimBlock>>>(s_velocity1_all_d,wf_d,nx,nz,receiver_x_cord[ishot],receiver_interval,receiver_num,nx_append_new,nz_append,boundary_left,boundary_up);
				cuda_cal_expand<<<dimGrid,dimBlock>>>(s_velocity1_d,wf_d,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);

/////////////////////////////////////get smooth density;				
				cuda_get_partly_mode_boundary<<<dimGrid_new,dimBlock>>>(s_density_all_d,wf_d,nx,nz,receiver_x_cord[ishot],receiver_interval,receiver_num,nx_append_new,nz_append,boundary_left,boundary_up);
				cuda_cal_expand<<<dimGrid,dimBlock>>>(s_density_d,wf_d,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);

/////////////////////////////////////get smooth qp;				
				cuda_get_partly_mode_boundary<<<dimGrid_new,dimBlock>>>(s_qp_all_d,wf_d,nx,nz,receiver_x_cord[ishot],receiver_interval,receiver_num,nx_append_new,nz_append,boundary_left,boundary_up);
				cuda_cal_expand<<<dimGrid,dimBlock>>>(s_qp_d,wf_d,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);

/////////////////////////////////////get smooth qs;				
				cuda_get_partly_mode_boundary<<<dimGrid_new,dimBlock>>>(s_qs_all_d,wf_d,nx,nz,receiver_x_cord[ishot],receiver_interval,receiver_num,nx_append_new,nz_append,boundary_left,boundary_up);
				cuda_cal_expand<<<dimGrid,dimBlock>>>(s_qs_d,wf_d,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);

				/*if((receiver_offset!=0)||(offset_left[ishot]>receiver_offset)||(offset_right[ishot]>receiver_offset))
				{								
					cuda_expand_acqusition_left_and_right<<<dimGrid_new,dimBlock>>>(s_velocity_d,offset_left[ishot],offset_right[ishot],source_x_cord[ishot],receiver_offset,nx_size,nx_append,nz_append,boundary_left,boundary_up);

					cuda_expand_acqusition_left_and_right<<<dimGrid_new,dimBlock>>>(s_velocity1_d,offset_left[ishot],offset_right[ishot],source_x_cord[ishot],receiver_offset,nx_size,nx_append,nz_append,boundary_left,boundary_up);

					cuda_expand_acqusition_left_and_right<<<dimGrid_new,dimBlock>>>(s_density_d,offset_left[ishot],offset_right[ishot],source_x_cord[ishot],receiver_offset,nx_size,nx_append,nz_append,boundary_left,boundary_up);

					cuda_expand_acqusition_left_and_right<<<dimGrid_new,dimBlock>>>(s_qp_d,offset_left[ishot],offset_right[ishot],source_x_cord[ishot],receiver_offset,nx_size,nx_append,nz_append,boundary_left,boundary_up);

					cuda_expand_acqusition_left_and_right<<<dimGrid_new,dimBlock>>>(s_qs_d,offset_left[ishot],offset_right[ishot],source_x_cord[ishot],receiver_offset,nx_size,nx_append,nz_append,boundary_left,boundary_up);						
				}*/

/////////////////////////////////QQQQQQQQQQQQQQQ2017年07月27日 星期四 10时11分40秒 
				cuda_cal_viscoelastic<<< dimGrid,dimBlock>>>(modul_p_d,modul_s_d,s_qp_d,s_qs_d,tao_d,strain_p_d,strain_s_d,freq,s_velocity_d,s_velocity1_d,s_density_d,nx_append,nz_append);


				cudaMemset(correlation_parameter_d,0,10*sizeof(float));/////////correlation misfit function  it is important for calculating adjiont source
				//////???????????????????????????????????
				cudaMemset(obs_shot_x_d,0,lt_rec*sizeof(float));
				cudaMemset(obs_shot_z_d,0,lt_rec*sizeof(float));
				cudaMemset(cal_shot_x_d,0,lt_rec*sizeof(float));
				cudaMemset(cal_shot_z_d,0,lt_rec*sizeof(float));
				cudaMemset(res_shot_x_d,0,lt_rec*sizeof(float));
				cudaMemset(res_shot_z_d,0,lt_rec*sizeof(float));
				cudaMemset(tmp_shot_x_d,0,lt_rec*sizeof(float));
				cudaMemset(tmp_shot_z_d,0,lt_rec*sizeof(float));
				cudaMemset(adj_shot_x_d,0,lt_rec*sizeof(float));
				cudaMemset(adj_shot_z_d,0,lt_rec*sizeof(float));

				cudaMemset(obs_shot_x_d_2,0,lt_rec*sizeof(float));
				cudaMemset(obs_shot_z_d_2,0,lt_rec*sizeof(float));
				cudaMemset(cal_shot_x_d_2,0,lt_rec*sizeof(float));
				cudaMemset(cal_shot_z_d_2,0,lt_rec*sizeof(float));
				cudaMemset(res_shot_x_d_2,0,lt_rec*sizeof(float));
				cudaMemset(res_shot_z_d_2,0,lt_rec*sizeof(float));
				//////???????????????????????????????????
			
				if(iter==0)
				{
					//////fread obs shot	
					sprintf(filename,"./someoutput/bin/obs_shot_x_%d",ishot+1);
					fread_file_1d(shotgather,receiver_num,lt,filename);
					cudaMemcpy(obs_shot_x_d,shotgather,lt_rec*sizeof(float),cudaMemcpyHostToDevice);

					sprintf(filename,"./someoutput/bin/obs_shot_z_%d",ishot+1);
					fread_file_1d(shotgather,receiver_num,lt,filename);
					cudaMemcpy(obs_shot_z_d,shotgather,lt_rec*sizeof(float),cudaMemcpyHostToDevice);
					//////fread obs shot

					////////for sn!=0 data
					if(receiver_offset!=0)
					{					
						cauda_zero_acqusition_left_and_right<<<dimGrid_lt,dimBlock>>>(obs_shot_x_d,offset_left[ishot],offset_right[ishot],source_x_cord[ishot],receiver_offset,receiver_num,lt);
						cauda_zero_acqusition_left_and_right<<<dimGrid_lt,dimBlock>>>(obs_shot_z_d,offset_left[ishot],offset_right[ishot],source_x_cord[ishot],receiver_offset,receiver_num,lt);	
					}

					//cudaMemcpy(res_shot_x_d,obs_shot_x_d,lt_rec*sizeof(float),cudaMemcpyDeviceToDevice);
					//cudaMemcpy(res_shot_z_d,obs_shot_z_d,lt_rec*sizeof(float),cudaMemcpyDeviceToDevice);///2016年10月27日 星期四 05时07分14秒 
					cuda_cal_residuals_new<<<dimGrid_lt,dimBlock>>>(res_shot_x_d,obs_shot_x_d,cal_shot_x_d,receiver_num,lt);
					cuda_cal_residuals_new<<<dimGrid_lt,dimBlock>>>(res_shot_z_d,obs_shot_z_d,cal_shot_z_d,receiver_num,lt);

					/////cuda_adj_shot
					cudaMemcpy(adj_shot_x_d,res_shot_x_d,lt_rec*sizeof(float),cudaMemcpyDeviceToDevice);
					cudaMemcpy(adj_shot_z_d,res_shot_z_d,lt_rec*sizeof(float),cudaMemcpyDeviceToDevice);

					/////////output first residuals	
					cudaMemcpy(shotgather,res_shot_x_d,lt_rec*sizeof(float),cudaMemcpyDeviceToHost);
					sprintf(filename,"./someoutput/bin/res_shot_x_%d_iter_%d",ishot+1,iter+1);
					write_file_1d(shotgather,lt_rec,filename);

					cudaMemcpy(shotgather,res_shot_z_d,lt_rec*sizeof(float),cudaMemcpyDeviceToHost);
					sprintf(filename,"./someoutput/bin/res_shot_z_%d_iter_%d",ishot+1,iter+1);
					write_file_1d(shotgather,lt_rec,filename);
					/////////output first residuals


					/////////output tmp cal	
					cudaMemcpy(shotgather,tmp_shot_x_d,lt_rec*sizeof(float),cudaMemcpyDeviceToHost);
					sprintf(filename,"./someoutput/bin/tmp_shot_x_%d_iter_%d",ishot+1,iter+1);
					write_file_1d(shotgather,lt_rec,filename);

					cudaMemcpy(shotgather,tmp_shot_z_d,lt_rec*sizeof(float),cudaMemcpyDeviceToHost);
					sprintf(filename,"./someoutput/bin/tmp_shot_z_%d_iter_%d",ishot+1,iter+1);
					write_file_1d(shotgather,lt_rec,filename);
					/////////output tmp cal
				}

				if(correlation_misfit==0)
				{
					if(iter>0)
					{	
						//////fread iter+1 res_shot			
						sprintf(filename,"./someoutput/bin/res_shot_x_%d_iter_%d",ishot+1,iter+1);
						fread_file_1d(shotgather,receiver_num,lt,filename);
						cudaMemcpy(res_shot_x_d,shotgather,lt_rec*sizeof(float),cudaMemcpyHostToDevice);

						sprintf(filename,"./someoutput/bin/res_shot_z_%d_iter_%d",ishot+1,iter+1);
						fread_file_1d(shotgather,receiver_num,lt,filename);
						cudaMemcpy(res_shot_z_d,shotgather,lt_rec*sizeof(float),cudaMemcpyHostToDevice);
					
						/////cuda_adj_shot
						cudaMemcpy(adj_shot_x_d,res_shot_x_d,lt_rec*sizeof(float),cudaMemcpyDeviceToDevice);
						cudaMemcpy(adj_shot_z_d,res_shot_z_d,lt_rec*sizeof(float),cudaMemcpyDeviceToDevice);
					}
////////calculate objective value
					cuda_cal_objective<<<1, Block_Size>>>(&obj_d[0], res_shot_x_d, lt_rec);
					cudaMemcpy(&obj_exchange,&obj_d[0],1*sizeof(float),cudaMemcpyDeviceToHost);
					obj_h[0]+=obj_exchange;

					cuda_cal_objective<<<1, Block_Size>>>(&obj_d[0], res_shot_z_d, lt_rec);
					cudaMemcpy(&obj_exchange,&obj_d[0],1*sizeof(float),cudaMemcpyDeviceToHost);
					obj_h[0]+=obj_exchange;
////////calculate objective value
				}

				else
				{
					if(iter>0)//correlation misfit function
					{	
						//////fread obs shot	
						sprintf(filename,"./someoutput/bin/obs_shot_x_%d",ishot+1);
						fread_file_1d(shotgather,receiver_num,lt,filename);
						cudaMemcpy(obs_shot_x_d,shotgather,lt_rec*sizeof(float),cudaMemcpyHostToDevice);

						sprintf(filename,"./someoutput/bin/obs_shot_z_%d",ishot+1);
						fread_file_1d(shotgather,receiver_num,lt,filename);
						cudaMemcpy(obs_shot_z_d,shotgather,lt_rec*sizeof(float),cudaMemcpyHostToDevice);
						//////fread obs shot

						////////for sn!=0 data
						if(receiver_offset!=0)
						{
							cauda_zero_acqusition_left_and_right<<<dimGrid_lt,dimBlock>>>(obs_shot_x_d,offset_left[ishot],offset_right[ishot],source_x_cord[ishot],receiver_offset,receiver_num,lt);
							cauda_zero_acqusition_left_and_right<<<dimGrid_lt,dimBlock>>>(obs_shot_z_d,offset_left[ishot],offset_right[ishot],source_x_cord[ishot],receiver_offset,receiver_num,lt);
						}

						//////fread iter+1 tmp_shot			
						sprintf(filename,"./someoutput/bin/tmp_shot_x_%d_iter_%d",ishot+1,iter+1);
						fread_file_1d(shotgather,receiver_num,lt,filename);
						cudaMemcpy(tmp_shot_x_d,shotgather,lt_rec*sizeof(float),cudaMemcpyHostToDevice);

						sprintf(filename,"./someoutput/bin/tmp_shot_z_%d_iter_%d",ishot+1,iter+1);
						fread_file_1d(shotgather,receiver_num,lt,filename);
						cudaMemcpy(tmp_shot_z_d,shotgather,lt_rec*sizeof(float),cudaMemcpyHostToDevice);
					
						cuda_dot_sum<<<1,Block_Size>>>(tmp_shot_x_d,tmp_shot_x_d,lt_rec,&correlation_parameter_d[0]);
						cuda_dot_sum<<<1,Block_Size>>>(tmp_shot_z_d,tmp_shot_z_d,lt_rec,&correlation_parameter_d[0]);
						

						cuda_dot_sum<<<1,Block_Size>>>(obs_shot_x_d,obs_shot_x_d,lt_rec,&correlation_parameter_d[1]);
						cuda_dot_sum<<<1,Block_Size>>>(obs_shot_z_d,obs_shot_z_d,lt_rec,&correlation_parameter_d[1]);


						cuda_dot_sum<<<1,Block_Size>>>(tmp_shot_x_d,obs_shot_x_d,lt_rec,&correlation_parameter_d[2]);
						cuda_dot_sum<<<1,Block_Size>>>(tmp_shot_z_d,obs_shot_z_d,lt_rec,&correlation_parameter_d[2]);


						cuda_adj_shot<<<dimGrid_lt,dimBlock>>>(adj_shot_x_d,tmp_shot_x_d,obs_shot_x_d,receiver_num,lt,correlation_parameter_d);
						cuda_adj_shot<<<dimGrid_lt,dimBlock>>>(adj_shot_z_d,tmp_shot_z_d,obs_shot_z_d,receiver_num,lt,correlation_parameter_d);

						cuda_dot_sum<<<1,Block_Size>>>(tmp_shot_x_d,tmp_shot_x_d,lt_rec,&obj_parameter_d[0]);
						cuda_dot_sum<<<1,Block_Size>>>(tmp_shot_z_d,tmp_shot_z_d,lt_rec,&obj_parameter_d[0]);
						

						cuda_dot_sum<<<1,Block_Size>>>(obs_shot_x_d,obs_shot_x_d,lt_rec,&obj_parameter_d[1]);
						cuda_dot_sum<<<1,Block_Size>>>(obs_shot_z_d,obs_shot_z_d,lt_rec,&obj_parameter_d[1]);


						cuda_dot_sum<<<1,Block_Size>>>(tmp_shot_x_d,obs_shot_x_d,lt_rec,&obj_parameter_d[2]);
						cuda_dot_sum<<<1,Block_Size>>>(tmp_shot_z_d,obs_shot_z_d,lt_rec,&obj_parameter_d[2]);

						if(ishot==shot_num-1)//correlation misfit function
						{	
							cuda_cal_correlation_objective<<<1,1>>>(&obj_d[0],obj_parameter_d);
							cudaMemcpy(&obj_exchange,&obj_d[0],1*sizeof(float),cudaMemcpyDeviceToHost);
							obj_h[0]=obj_exchange;
						}
					}
				}
					/////////output adj_shot	
					cudaMemcpy(shotgather,adj_shot_x_d,lt_rec*sizeof(float),cudaMemcpyDeviceToHost);
					sprintf(filename,"./someoutput/bin/adj_shot_x_%d_iter_%d",ishot+1,iter+1);
					write_file_1d(shotgather,lt_rec,filename);

					cudaMemcpy(shotgather,adj_shot_z_d,lt_rec*sizeof(float),cudaMemcpyDeviceToHost);
					sprintf(filename,"./someoutput/bin/adj_shot_z_%d_iter_%d",ishot+1,iter+1);
					write_file_1d(shotgather,lt_rec,filename);

					/*if(receiver_offset!=0)////////////////try to weigthed adjoint source
					{
						cut_direct<<<dimGrid,dimBlock>>>(adj_shot_x_d,lt,source_x_cord[ishot],shot_depth,receiver_num,receiver_depth,receiver_x_cord[ishot],receiver_interval,boundary_left,boundary_up,nz_append,dx,dz,dt,velocity_d,wavelet_half);
						cut_direct<<<dimGrid,dimBlock>>>(adj_shot_z_d,lt,source_x_cord[ishot],shot_depth,receiver_num,receiver_depth,receiver_x_cord[ishot],receiver_interval,boundary_left,boundary_up,nz_append,dx,dz,dt,velocity_d,wavelet_half);

						cuda_attenuation_adj<<<dimGrid_lt,dimBlock>>>(adj_shot_x_d,receiver_num,lt,offset_left[ishot],offset_right[ishot],receiver_offset);
						cuda_attenuation_adj<<<dimGrid_lt,dimBlock>>>(adj_shot_z_d,receiver_num,lt,offset_left[ishot],offset_right[ishot],receiver_offset);
						/////////output adj_shot	
						cudaMemcpy(shotgather,adj_shot_x_d,lt_rec*sizeof(float),cudaMemcpyDeviceToHost);
						sprintf(filename,"./someoutput/bin/adj1_shot_x_%d_iter_%d",ishot+1,iter+1);
						write_file_1d(shotgather,lt_rec,filename);

						cudaMemcpy(shotgather,adj_shot_z_d,lt_rec*sizeof(float),cudaMemcpyDeviceToHost);
						sprintf(filename,"./someoutput/bin/adj1_shot_z_%d_iter_%d",ishot+1,iter+1);
						write_file_1d(shotgather,lt_rec,filename);
					}*/

				//set zero
				cudaMemset(vx1_d,0,nxanza*sizeof(float));
				cudaMemset(vz1_d,0,nxanza*sizeof(float));
				cudaMemset(txx1_d,0,nxanza*sizeof(float));
				cudaMemset(tzz1_d,0,nxanza*sizeof(float));
				cudaMemset(txz1_d,0,nxanza*sizeof(float));

				cudaMemset(vx2_d,0,nxanza*sizeof(float));
				cudaMemset(vz2_d,0,nxanza*sizeof(float));
				cudaMemset(txx2_d,0,nxanza*sizeof(float));
				cudaMemset(tzz2_d,0,nxanza*sizeof(float));
				cudaMemset(txz2_d,0,nxanza*sizeof(float));

				cudaMemset(mem_p1_d,0,nxanza*sizeof(float));
				cudaMemset(mem_xx1_d,0,nxanza*sizeof(float));
				cudaMemset(mem_zz1_d,0,nxanza*sizeof(float));
				cudaMemset(mem_xz1_d,0,nxanza*sizeof(float));

				cudaMemset(mem_p2_d,0,nxanza*sizeof(float));
				cudaMemset(mem_xx2_d,0,nxanza*sizeof(float));
				cudaMemset(mem_zz2_d,0,nxanza*sizeof(float));
				cudaMemset(mem_xz2_d,0,nxanza*sizeof(float));

				cudaMemset(rvx1_d,0,nxanza*sizeof(float));
				cudaMemset(rvz1_d,0,nxanza*sizeof(float));
				cudaMemset(rtxx1_d,0,nxanza*sizeof(float));
				cudaMemset(rtzz1_d,0,nxanza*sizeof(float));
				cudaMemset(rtxz1_d,0,nxanza*sizeof(float));

				cudaMemset(rvx2_d,0,nxanza*sizeof(float));
				cudaMemset(rvz2_d,0,nxanza*sizeof(float));
				cudaMemset(rtxx2_d,0,nxanza*sizeof(float));
				cudaMemset(rtzz2_d,0,nxanza*sizeof(float));
				cudaMemset(rtxz2_d,0,nxanza*sizeof(float));

				cudaMemset(rmem_p1_d,0,nxanza*sizeof(float));
				cudaMemset(rmem_xx1_d,0,nxanza*sizeof(float));
				cudaMemset(rmem_zz1_d,0,nxanza*sizeof(float));
				cudaMemset(rmem_xz1_d,0,nxanza*sizeof(float));

				cudaMemset(rmem_p2_d,0,nxanza*sizeof(float));
				cudaMemset(rmem_xx2_d,0,nxanza*sizeof(float));
				cudaMemset(rmem_zz2_d,0,nxanza*sizeof(float));
				cudaMemset(rmem_xz2_d,0,nxanza*sizeof(float));

				if(iter==0||decomposition!=0)
				{
					cudaMemset(tp2_d,0,nxanza*sizeof(float));
					cudaMemset(tp1_d,0,nxanza*sizeof(float));
					cudaMemset(vxp2_d,0,nxanza*sizeof(float));
					cudaMemset(vxp1_d,0,nxanza*sizeof(float));
					cudaMemset(vzp2_d,0,nxanza*sizeof(float));
					cudaMemset(vzp1_d,0,nxanza*sizeof(float));
					cudaMemset(vxs2_d,0,nxanza*sizeof(float));
					cudaMemset(vxs1_d,0,nxanza*sizeof(float));
					cudaMemset(vzs2_d,0,nxanza*sizeof(float));
					cudaMemset(vzs1_d,0,nxanza*sizeof(float));

					cudaMemset(rtp2_d,0,nxanza*sizeof(float));
					cudaMemset(rtp1_d,0,nxanza*sizeof(float));
					cudaMemset(rvxp2_d,0,nxanza*sizeof(float));
					cudaMemset(rvxp1_d,0,nxanza*sizeof(float));
					cudaMemset(rvzp2_d,0,nxanza*sizeof(float));
					cudaMemset(rvzp1_d,0,nxanza*sizeof(float));
					cudaMemset(rvxs2_d,0,nxanza*sizeof(float));
					cudaMemset(rvxs1_d,0,nxanza*sizeof(float));
					cudaMemset(rvzs2_d,0,nxanza*sizeof(float));
					cudaMemset(rvzs1_d,0,nxanza*sizeof(float));

					cudaMemset(vx_x_d,0,nxanza*sizeof(float));
					cudaMemset(vx_z_d,0,nxanza*sizeof(float));
					cudaMemset(vz_x_d,0,nxanza*sizeof(float));
					cudaMemset(vz_z_d,0,nxanza*sizeof(float));
					cudaMemset(vx_t_d,0,nxanza*sizeof(float));
					cudaMemset(vz_t_d,0,nxanza*sizeof(float));
				
					cudaMemset(rvxp_integral_d,0,nxanza*sizeof(float));
					cudaMemset(rvzp_integral_d,0,nxanza*sizeof(float));
					cudaMemset(rvxs_integral_d,0,nxanza*sizeof(float));
					cudaMemset(rvzs_integral_d,0,nxanza*sizeof(float));

					cudaMemset(grad_den_pp_d,0,nx_size_nz*sizeof(float));
					cudaMemset(grad_lame1_pp_d,0,nx_size_nz*sizeof(float));
					cudaMemset(grad_lame2_pp_d,0,nx_size_nz*sizeof(float));

					cudaMemset(grad_den_ps_d,0,nx_size_nz*sizeof(float));
					cudaMemset(grad_lame1_ps_d,0,nx_size_nz*sizeof(float));
					cudaMemset(grad_lame2_ps_d,0,nx_size_nz*sizeof(float));

					cudaMemset(grad_den_sp_d,0,nx_size_nz*sizeof(float));
					cudaMemset(grad_lame1_sp_d,0,nx_size_nz*sizeof(float));
					cudaMemset(grad_lame2_sp_d,0,nx_size_nz*sizeof(float));

					cudaMemset(grad_den_ss_d,0,nx_size_nz*sizeof(float));
					cudaMemset(grad_lame1_ss_d,0,nx_size_nz*sizeof(float));
					cudaMemset(grad_lame2_ss_d,0,nx_size_nz*sizeof(float));
				}

				if(iter==0)/////////////for migration
				{	
					cudaMemset(resultpp_d,0,nx_size_nz*sizeof(float));
					cudaMemset(resultps_d,0,nx_size_nz*sizeof(float));
					cudaMemset(resultps1_d,0,nx_size_nz*sizeof(float));
					cudaMemset(resultps2_d,0,nx_size_nz*sizeof(float));
					cudaMemset(resultsp_d,0,nx_size_nz*sizeof(float));
					cudaMemset(resultsp1_d,0,nx_size_nz*sizeof(float));
					cudaMemset(resultsp2_d,0,nx_size_nz*sizeof(float));
					cudaMemset(resultss_d,0,nx_size_nz*sizeof(float));
		
					cudaMemset(result_tp_d,0,nx_size_nz*sizeof(float));		
					cudaMemset(vresultpp_d,0,nx_size_nz*sizeof(float));
					cudaMemset(vresultps_d,0,nx_size_nz*sizeof(float));
					cudaMemset(vresultsp_d,0,nx_size_nz*sizeof(float));
					cudaMemset(vresultss_d,0,nx_size_nz*sizeof(float));

					cudaMemset(vresultppx_d,0,nx_size_nz*sizeof(float));
					cudaMemset(vresultpsx_d,0,nx_size_nz*sizeof(float));
					cudaMemset(vresultspx_d,0,nx_size_nz*sizeof(float));
					cudaMemset(vresultssx_d,0,nx_size_nz*sizeof(float));

					cudaMemset(vresultppz_d,0,nx_size_nz*sizeof(float));
					cudaMemset(vresultpsz_d,0,nx_size_nz*sizeof(float));
					cudaMemset(vresultspz_d,0,nx_size_nz*sizeof(float));
					cudaMemset(vresultssz_d,0,nx_size_nz*sizeof(float));


					cudaMemset(down_vpp_x_d,0,nx_size_nz*sizeof(float));
					cudaMemset(down_vpp_z_d,0,nx_size_nz*sizeof(float));
					cudaMemset(down_vss_x_d,0,nx_size_nz*sizeof(float));
					cudaMemset(down_vss_z_d,0,nx_size_nz*sizeof(float));

					cudaMemset(down_tp_d,0,nx_size_nz*sizeof(float));

					cudaMemset(down_vpp_d,0,nx_size_nz*sizeof(float));
					cudaMemset(down_vss_d,0,nx_size_nz*sizeof(float));

					cudaMemset(down_pp_d,0,nx_size_nz*sizeof(float));
					cudaMemset(down_ss_d,0,nx_size_nz*sizeof(float));
				
					cudaMemset(down_xx_d,0,nx_size_nz*sizeof(float));
					cudaMemset(down_zz_d,0,nx_size_nz*sizeof(float));
//////excitation:   	related function in kernel_3	
					cudaMemset(com_ex_vresultpp_d,0,nx_size_nz*sizeof(float));
					cudaMemset(com_ex_vresultps_d,0,nx_size_nz*sizeof(float));
					cudaMemset(ex_vresultpp_d,0,nx_size_nz*sizeof(float));
					cudaMemset(ex_vresultps_d,0,nx_size_nz*sizeof(float));
					cudaMemset(ex_result_tp_d,0,nx_size_nz*sizeof(float));
					cudaMemset(ex_result_tp_old_d,0,nx_size_nz*sizeof(float));
					cudaMemset(resultxx_d,0,nx_size_nz*sizeof(float));
					cudaMemset(resultzz_d,0,nx_size_nz*sizeof(float));

					cudaMemset(ex_amp_d,0,nxanza*sizeof(float));
					cudaMemset(ex_time_d,0,nxanza*sizeof(float));
					cudaMemset(ex_amp_x_d,0,nxanza*sizeof(float));
					cudaMemset(ex_amp_z_d,0,nxanza*sizeof(float));
					cudaMemset(ex_amp_tp_old_d,0,nxanza*sizeof(float));

					cudaMemset(ex_tp_time_d,0,nxanza*sizeof(float));
					cudaMemset(ex_amp_tp_d,0,nxanza*sizeof(float));
			
					cudaMemset(para_max_d,0,20*sizeof(float));

					cudaMemset(ex_angle_pp_d,0,nxanza*sizeof(float));
					cudaMemset(ex_angle_rpp_d,0,nxanza*sizeof(float));
					cudaMemset(ex_angle_rps_d,0,nxanza*sizeof(float));

					cudaMemset(ex_angle_pp1_d,0,nxanza*sizeof(float));
					cudaMemset(ex_angle_rpp1_d,0,nxanza*sizeof(float));
					cudaMemset(ex_angle_rps1_d,0,nxanza*sizeof(float));

					cudaMemset(ex_open_pp_d,0,nxanza*sizeof(float));
					cudaMemset(ex_open_ps_d,0,nxanza*sizeof(float));
					cudaMemset(ex_open_pp1_d,0,nxanza*sizeof(float));
					cudaMemset(ex_open_ps1_d,0,nxanza*sizeof(float));

					cudaMemset(ex_com_pp_sign_d,0,nxanza*sizeof(float));
					cudaMemset(ex_com_ps_sign_d,0,nxanza*sizeof(float));
				}

////setz zero for gradient
				cudaMemset(grad_lame11_d,0,nx_size_nz*sizeof(float));
				cudaMemset(grad_lame22_d,0,nx_size_nz*sizeof(float));
				cudaMemset(grad_den1_d,0,nx_size_nz*sizeof(float));
//////setz zero for gradient

///////setz zero for gradient
				cudaMemset(grad_vp1_d,0,nx_size_nz*sizeof(float));
				cudaMemset(grad_vs1_d,0,nx_size_nz*sizeof(float));
				cudaMemset(grad_density1_d,0,nx_size_nz*sizeof(float));/////////it is noted that set zero for calculating gradient
///////setz zero for gradient
				cudaMemset(d_illum,0,nxanza*sizeof(float));///////////////for  every shot  and variation////so we must set zero

			if(iter==0)
			{
				for(int it=0;it<lt+wavelet_half;it++)
					{
						//if(fmod((it+1.0)-wavelet_half,1000.0)==0) warn("shot=%d,step=forward 2,it=%d",ishot+1,(it+1)-wavelet_half);

						if(it<wavelet_length)
						{
							//add_source<<<1,1>>>(txx1_d,tzz1_d,wavelet_d,source_x_cord[ishot],shot_depth,it,boundary_up,boundary_left,nz_append);
							//add_source<<<1,1>>>(txx1_d,tzz1_d,wavelet_d,source_x_cord[ishot],source_z_cord[ishot],it,boundary_up,boundary_left,nz_append);////for vsp or surface 2017年03月14日 星期二 08时41分20秒
							add_source<<<1,1>>>(txx1_d,tzz1_d,wavelet_d,source_x_cord[ishot]-receiver_x_cord[ishot],source_z_cord[ishot],it,boundary_up,boundary_left,nz_append);////for new acqusition way 2017年08月17日 星期四 09时10分03秒
						}

						fwd_vx_new_new<<<dimGrid,dimBlock>>>(d_illum,vx_t_d,vx2_d,vx1_d,txx1_d,txz1_d,attenuation_d,dx,dz,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,s_density_d);

						fwd_vz_new_new<<<dimGrid,dimBlock>>>(d_illum,vz_t_d,vz2_d,vz1_d,tzz1_d,txz1_d,attenuation_d,dx,dz,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,s_density_d);

						if(decomposition!=0||iter==0)
						{
							fwd_vxp_vzp<<<dimGrid,dimBlock>>>(vxp2_d,vxp1_d,vzp2_d,vzp1_d,tp1_d,coe_x,coe_z,dx,dz,dt,attenuation_d,coe_opt_d,nx_append_radius,nz_append_radius,s_density_d);

							vp_vs<<<dimGrid,dimBlock>>>(vx2_d,vz2_d,vxp2_d,vzp2_d,vxs2_d,vzs2_d,nx_append_radius,nz_append_radius);

							//decom<<<dimGrid,dimBlock>>>(vx2_d,vz2_d,p_d,s_d,coe_opt_d,nx_append,nz_append,dx,dz);
							//decom_new<<<dimGrid,dimBlock>>>(vx2_d,vz2_d,p_d,s_d,s_velocity_d,s_velocity1_d,coe_opt_d,nx_append_radius,nz_append_radius,dx,dz);
							sum_poynting<<<dimGrid,dimBlock>>>(poyn_px_d,poyn_pz_d,poyn_sx_d,poyn_sz_d,vxp2_d,vzp2_d,vxs2_d,vzs2_d,txx2_d,tzz2_d,txz2_d,tp2_d,nx_append_radius,nz_append_radius);

							cal_direction_2D_elastic<<<dimGrid,dimBlock>>>(direction_px_d,direction_pz_d,direction_sx_d,direction_sz_d,vxp2_d,vzp2_d,vxs2_d,vzs2_d,txx2_d,tzz2_d,txz2_d,tp2_d,nx_append_radius,nz_append_radius);
						}

						if(migration_type==0)	fwd_txxzzxzpp<<<dimGrid,dimBlock>>>(tp2_d,tp1_d,txx2_d,txx1_d,tzz2_d,tzz1_d,txz2_d,txz1_d,vx2_d,vz2_d,s_velocity_d,s_velocity1_d,attenuation_d,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,s_density_d);	
						
						//if(migration_type==0)	fwd_txxzzxzpp_new<<<dimGrid,dimBlock>>>(vx_x_d,vx_z_d,vz_x_d,vz_z_d,tp2_d,tp1_d,txx2_d,txx1_d,tzz2_d,tzz1_d,txz2_d,txz1_d,vx2_d,vz2_d,s_velocity_d,s_velocity1_d,attenuation_d,dt,dx,dz,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,s_density_d);

						//else	fwd_txxzzxzpp_viscoelastic_and_memory<<<dimGrid,dimBlock>>>(tp2_d,tp1_d,txx2_d,txx1_d,tzz2_d,tzz1_d,txz2_d,txz1_d,vx2_d,vz2_d,modul_p_d,modul_s_d,attenuation_d,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,s_density_d,mem_p2_d,mem_p1_d,mem_xx2_d,mem_xx1_d,mem_zz2_d,mem_zz1_d,mem_xz2_d,mem_xz1_d,s_velocity_d,s_velocity1_d,tao_d,strain_p_d,strain_s_d);

						else	fwd_txxzzxzpp_viscoelastic_and_memory_3parameterization<<<dimGrid,dimBlock>>>(tp2_d,tp1_d,txx2_d,txx1_d,tzz2_d,tzz1_d,txz2_d,txz1_d,vx2_d,vz2_d,modul_p_d,modul_s_d,attenuation_d,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,s_density_d,mem_p2_d,mem_p1_d,mem_xx2_d,mem_xx1_d,mem_zz2_d,mem_zz1_d,mem_xz2_d,mem_xz1_d,s_velocity_d,s_velocity1_d,tao_d,strain_p_d,strain_s_d);

/////////////////////////////////////////////////////////////////////////////excitation imaging condition
						if(it>=wavelet_half)
						{
///1111
							//caculate_ex_amp_time<<<dimGrid,dimBlock>>>(p_down_d,ex_amp_d,ex_time_d,it-wavelet_half,nx_append_radius,nz_append_radius);
							//caculate_ex_amp_time_new<<<dimGrid,dimBlock>>>(vxp_down_d,vzp_down_d,ex_amp_d,ex_time_d,it-wavelet_half,nx_append_radius,nz_append_radius);
							//caculate_ex_x_z<<<dimGrid,dimBlock>>>(ex_amp_x_d,ex_amp_z_d,vxp_down_d,vzp_down_d,ex_time_d,it-wavelet_half,nx_append_radius,nz_append_radius);				
							//caculate_ex_angle<<<dimGrid,dimBlock>>>(ex_angle_d,angle_pp1_d,ex_time_d,it-wavelet_half,nx_append_radius,nz_append_radius);	
///2222			
							//caculate_ex_amp_time<<<dimGrid,dimBlock>>>(p_d,ex_amp_d,ex_time_d,it-wavelet_half,nx_append_radius,nz_append_radius);
							caculate_ex_amp_time_new<<<dimGrid,dimBlock>>>(vxp1_d,vzp1_d,ex_amp_d,ex_time_d,it-wavelet_half,nx_append_radius,nz_append_radius);
							caculate_ex_tp_time_new<<<dimGrid,dimBlock>>>(tp1_d,ex_amp_tp_d,ex_tp_time_d,it-wavelet_half,nx_append_radius,nz_append_radius);
							//caculate_ex_x_z_new<<<dimGrid,dimBlock>>>(ex_amp_x_d,ex_amp_z_d,vxp1_d,vzp1_d,ex_time_d,it-wavelet_half,nx_append_radius,nz_append_radius);
							caculate_ex_x_z<<<dimGrid,dimBlock>>>(ex_amp_x_d,ex_amp_z_d,vxp1_d,vzp1_d,ex_time_d,it-wavelet_half,nx_append_radius,nz_append_radius);
							caculate_ex_x_z<<<dimGrid,dimBlock>>>(ex_amp_tp_old_d,ex_amp_tp_old_d,tp1_d,tp1_d,ex_time_d,it-wavelet_half,nx_append_radius,nz_append_radius);
								
							caculate_ex_angle_pp_only_RTM<<<dimGrid,dimBlock>>>(ex_angle_pp_d,poyn_px_d,poyn_pz_d,ex_time_d,it-wavelet_half,nx_append_radius,nz_append_radius);

							caculate_ex_angle_pp_only_RTM<<<dimGrid,dimBlock>>>(ex_angle_pp1_d,direction_px_d,direction_pz_d,ex_time_d,it-wavelet_half,nx_append_radius,nz_append_radius);
							//caculate_ex_angle<<<dimGrid,dimBlock>>>(ex_angle_d,angle_pp_d,ex_time_d,it-wavelet_half,nx_append_radius,nz_append_radius);
							//caculate_ex_angle_new<<<dimGrid,dimBlock>>>(ex_angle1_d,angle_pp1_d,normal_angle_d,poyn_px_d,poyn_pz_d,ex_time_d,it-wavelet_half,nx_append_radius,nz_append_radius);
						}					
							
							/////for check 2017年07月27日 星期四 10时11分40秒
							if(0==(it-wavelet_half)%check_interval)
							{
								/////for check 2017年07月27日 星期四 10时11分40秒
								////////velocity
								cudaMemcpy(wf_append,vx1_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
								sprintf(filename,"./check_file/velocity/vx-%d-shot_%d",ishot+1,it-wavelet_half);
								write_file_1d(wf_append,nxanza,filename);

								cudaMemcpy(wf_append,vz1_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
								sprintf(filename,"./check_file/velocity/vz-%d-shot_%d",ishot+1,it-wavelet_half);
								write_file_1d(wf_append,nxanza,filename);

								cudaMemcpy(wf_append,vxp1_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
								sprintf(filename,"./check_file/velocity/vxp-%d-shot_%d",ishot+1,it-wavelet_half);
								write_file_1d(wf_append,nxanza,filename);

								cudaMemcpy(wf_append,vzp1_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
								sprintf(filename,"./check_file/velocity/vzp-%d-shot_%d",ishot+1,it-wavelet_half);
								write_file_1d(wf_append,nxanza,filename);

								/*cudaMemcpy(wf_append,vxs1_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
								sprintf(filename,"./check_file/velocity/vxs-%d-shot_%d",ishot+1,it-wavelet_half);
								write_file_1d(wf_append,nxanza,filename);

								cudaMemcpy(wf_append,vzs1_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
								sprintf(filename,"./check_file/velocity/vzs-%d-shot_%d",ishot+1,it-wavelet_half);
								write_file_1d(wf_append,nxanza,filename);*/

								////////stress
								cudaMemcpy(wf_append,txx1_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
								sprintf(filename,"./check_file/stress/txx-%d-shot_%d",ishot+1,it-wavelet_half);
								write_file_1d(wf_append,nxanza,filename);

								cudaMemcpy(wf_append,txz1_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
								sprintf(filename,"./check_file/stress/txz-%d-shot_%d",ishot+1,it-wavelet_half);
								write_file_1d(wf_append,nxanza,filename);

								cudaMemcpy(wf_append,tzz1_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
								sprintf(filename,"./check_file/stress/tzz-%d-shot_%d",ishot+1,it-wavelet_half);
								write_file_1d(wf_append,nxanza,filename);

								cudaMemcpy(wf_append,tp1_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
								sprintf(filename,"./check_file/stress/tp-%d-shot_%d",ishot+1,it-wavelet_half);
								write_file_1d(wf_append,nxanza,filename);

								////////memory
								cudaMemcpy(wf_append,mem_p1_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
								sprintf(filename,"./check_file/memory/mem_p-%d-shot_%d",ishot+1,it-wavelet_half);
								write_file_1d(wf_append,nxanza,filename);

								cudaMemcpy(wf_append,mem_xx1_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
								sprintf(filename,"./check_file/memory/mem_xx-%d-shot_%d",ishot+1,it-wavelet_half);
								write_file_1d(wf_append,nxanza,filename);

								cudaMemcpy(wf_append,mem_zz1_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
								sprintf(filename,"./check_file/memory/mem_zz-%d-shot_%d",ishot+1,it-wavelet_half);
								write_file_1d(wf_append,nxanza,filename);

								cudaMemcpy(wf_append,mem_xz1_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
								sprintf(filename,"./check_file/memory/mem_xz-%d-shot_%d",ishot+1,it-wavelet_half);
								write_file_1d(wf_append,nxanza,filename);								
							}
						
							if(0==(it-wavelet_half)%100&&join_wavefield==1&&iter==0)
							{	
								cudaMemcpy(wf_append,vx2_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
								sprintf(filename,"./wavefield1/1/vx-%d-shot_%d",ishot+1,it-wavelet_half);
								write_file_1d(wf_append,nxanza,filename);
								exchange(wf_append,wf,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
								write_file_1d(wf,nx_size_nz,filename);
									
								cudaMemcpy(wf_append,vz2_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
								sprintf(filename,"./wavefield1/1/vz-%d-shot_%d",ishot+1,it-wavelet_half);
								write_file_1d(wf_append,nxanza,filename);
								exchange(wf_append,wf,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
								write_file_1d(wf,nx_size_nz,filename);

								cudaMemcpy(wf_append,vxp2_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
								sprintf(filename,"./wavefield1/1/vxp-%d-shot_%d",ishot+1,it-wavelet_half);
								write_file_1d(wf_append,nxanza,filename);
								exchange(wf_append,wf,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
								write_file_1d(wf,nx_size_nz,filename);

								cudaMemcpy(wf_append,vzp2_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
								sprintf(filename,"./wavefield1/1/vzp-%d-shot_%d",ishot+1,it-wavelet_half);
								write_file_1d(wf_append,nxanza,filename);
								exchange(wf_append,wf,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
								write_file_1d(wf,nx_size_nz,filename);
						
								cudaMemcpy(wf_append,vxs2_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
								sprintf(filename,"./wavefield1/1/vxs-%d-shot_%d",ishot+1,it-wavelet_half);
								write_file_1d(wf_append,nxanza,filename);
								exchange(wf_append,wf,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
								write_file_1d(wf,nx_size_nz,filename);
				
								cudaMemcpy(wf_append,vzs2_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
								sprintf(filename,"./wavefield1/1/vzs-%d-shot_%d",ishot+1,it-wavelet_half);
								write_file_1d(wf_append,nxanza,filename);
								exchange(wf_append,wf,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
								write_file_1d(wf,nx_size_nz,filename);
							}
						
							rep=vx1_d;vx1_d=vx2_d;vx2_d=rep;
							rep=vz1_d;vz1_d=vz2_d;vz2_d=rep;
							rep=txx1_d;txx1_d=txx2_d;txx2_d=rep;
							rep=tzz1_d;tzz1_d=tzz2_d;tzz2_d=rep;
							rep=txz1_d;txz1_d=txz2_d;txz2_d=rep;

							rep=tp1_d;tp1_d=tp2_d;tp2_d=rep;
							rep=vxp1_d;vxp1_d=vxp2_d;vxp2_d=rep;
							rep=vzp1_d;vzp1_d=vzp2_d;vzp2_d=rep;
							rep=vxs1_d;vxs1_d=vxs2_d;vxs2_d=rep;
							rep=vzs1_d;vzs1_d=vzs2_d;vzs2_d=rep;

							rep=mem_p1_d;mem_p1_d=mem_p2_d;mem_p2_d=rep;
							rep=mem_xx1_d;mem_xx1_d=mem_xx2_d;mem_xx2_d=rep;
							rep=mem_zz1_d;mem_zz1_d=mem_zz2_d;mem_zz2_d=rep;
							rep=mem_xz1_d;mem_xz1_d=mem_xz2_d;mem_xz2_d=rep;///////////////QQQQQQQQQQQQQQQ2017年07月27日 星期四 10时11分40秒
					}
			}
						cuda_cal_max<<<1,Block_Size>>>(&para_max_d[2],ex_amp_tp_old_d,nxanza);
						cuda_cal_max<<<1,Block_Size>>>(&para_max_d[0],ex_amp_tp_d,nxanza);
						cuda_cal_max<<<1,Block_Size>>>(&para_max_d[1],ex_amp_d,nxanza);

						cudaMemcpy(&p_printf,&para_max_d[2],sizeof(float),cudaMemcpyDeviceToHost);
						warn("tp_max_old=%f\n",p_printf);

						cudaMemcpy(&p_printf,&para_max_d[0],sizeof(float),cudaMemcpyDeviceToHost);
						warn("tp_max=%f\n",p_printf);

						cudaMemcpy(&p_printf,&para_max_d[1],sizeof(float),cudaMemcpyDeviceToHost);
						warn("amp_max=%f\n",p_printf);
						
						if(ishot%10==0)//outup excitation amp or time
						{	
////////amp
							cudaMemcpy(wf_append,ex_amp_d,nx_append*nz_append*sizeof(float),cudaMemcpyDeviceToHost);
							//sprintf(filename,"./someoutput/ex-amp");
							//write_file_1d(wf_append,nx_append*nz_append,filename);
							sprintf(filename,"./someoutput/cut-ex-amp_%d",ishot+1);
							exchange(wf_append,wf,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
							write_file_1d(wf,nx_size*nz,filename);
////////time
							cudaMemcpy(wf_append,ex_tp_time_d,nx_append*nz_append*sizeof(float),cudaMemcpyDeviceToHost);
							sprintf(filename,"./someoutput/cut-ex-tp-time_%d",ishot+1);
							exchange(wf_append,wf,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
							write_file_1d(wf,nx_size*nz,filename);

////////time
							cudaMemcpy(wf_append,ex_time_d,nx_append*nz_append*sizeof(float),cudaMemcpyDeviceToHost);
							sprintf(filename,"./someoutput/cut-ex-time_%d",ishot+1);
							exchange(wf_append,wf,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
							write_file_1d(wf,nx_size*nz,filename);
////vxp
							cudaMemcpy(wf_append,ex_amp_x_d,nx_append*nz_append*sizeof(float),cudaMemcpyDeviceToHost);
							sprintf(filename,"./someoutput/cut-ex-amp-vxp_%d",ishot+1);
							exchange(wf_append,wf,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
							write_file_1d(wf,nx_size*nz,filename);
///vzp
							cudaMemcpy(wf_append,ex_amp_z_d,nx_append*nz_append*sizeof(float),cudaMemcpyDeviceToHost);
							sprintf(filename,"./someoutput/cut-ex-amp-vzp_%d",ishot+1);
							exchange(wf_append,wf,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
							write_file_1d(wf,nx_size*nz,filename);
///tp_new
							cudaMemcpy(wf_append,ex_amp_tp_d,nx_append*nz_append*sizeof(float),cudaMemcpyDeviceToHost);
							sprintf(filename,"./someoutput/cut-ex-amp-tp_%d",ishot+1);
							exchange(wf_append,wf,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
							write_file_1d(wf,nx_size*nz,filename);
///tp_old
							cudaMemcpy(wf_append,ex_amp_tp_old_d,nx_append*nz_append*sizeof(float),cudaMemcpyDeviceToHost);
							sprintf(filename,"./someoutput/cut-ex-amp-tp_old_%d",ishot+1);
							exchange(wf_append,wf,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
							write_file_1d(wf,nx_size*nz,filename);

							cudaMemcpy(wf_append,ex_angle_pp_d,nx_append*nz_append*sizeof(float),cudaMemcpyDeviceToHost);
							sprintf(filename,"./someoutput/cut-ex-angle-pp_%d",ishot+1);
							exchange(wf_append,wf,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
							write_file_1d(wf,nx_size*nz,filename);

							cudaMemcpy(wf_append,ex_angle_pp1_d,nx_append*nz_append*sizeof(float),cudaMemcpyDeviceToHost);
							sprintf(filename,"./someoutput/cut-ex-angle-pp1_%d",ishot+1);
							exchange(wf_append,wf,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
							write_file_1d(wf,nx_size*nz,filename);
						}
						warn("forwrd_modeling_for_ERTM_is_over");
							
			for(int it=lt-1;it>=0;it--)
				{
					//if(fmod(it*1.0,1000)==0) warn("shot=%d,step=back,it=%d",ishot+1,it);

					if((0==fmod(it*1.0,check_interval)||(it==lt-1))&&it!=0)
					{
						cudaMemset(vx1_d,0,nxanza*sizeof(float));
						cudaMemset(vz1_d,0,nxanza*sizeof(float));
						cudaMemset(txx1_d,0,nxanza*sizeof(float));
						cudaMemset(tzz1_d,0,nxanza*sizeof(float));
						cudaMemset(txz1_d,0,nxanza*sizeof(float));
						cudaMemset(vx2_d,0,nxanza*sizeof(float));
						cudaMemset(vz2_d,0,nxanza*sizeof(float));
						cudaMemset(txx2_d,0,nxanza*sizeof(float));
						cudaMemset(tzz2_d,0,nxanza*sizeof(float));
						cudaMemset(txz2_d,0,nxanza*sizeof(float));

						cudaMemset(tp2_d,0,nxanza*sizeof(float));
						cudaMemset(tp1_d,0,nxanza*sizeof(float));
						cudaMemset(vxp2_d,0,nxanza*sizeof(float));
						cudaMemset(vxp1_d,0,nxanza*sizeof(float));
						cudaMemset(vzp2_d,0,nxanza*sizeof(float));
						cudaMemset(vzp1_d,0,nxanza*sizeof(float));
						cudaMemset(vxs2_d,0,nxanza*sizeof(float));
						cudaMemset(vxs1_d,0,nxanza*sizeof(float));
						cudaMemset(vzs2_d,0,nxanza*sizeof(float));
						cudaMemset(vzs1_d,0,nxanza*sizeof(float));

						cudaMemset(mem_p1_d,0,nxanza*sizeof(float));
						cudaMemset(mem_xx1_d,0,nxanza*sizeof(float));
						cudaMemset(mem_zz1_d,0,nxanza*sizeof(float));
						cudaMemset(mem_xz1_d,0,nxanza*sizeof(float));
						cudaMemset(mem_p2_d,0,nxanza*sizeof(float));
						cudaMemset(mem_xx2_d,0,nxanza*sizeof(float));
						cudaMemset(mem_zz2_d,0,nxanza*sizeof(float));
						cudaMemset(mem_xz2_d,0,nxanza*sizeof(float));

						if(0==fmod(it*1.0,check_interval))	
						{
							ittt_beg=it-check_interval;

							ittt_end=it;
						}

						else
						{
							if(check_residual==0)	
							{
								ittt_beg=it-check_interval+1;
								ittt_end=lt-1;
							}							
									
							else
							{				
								ittt_beg=it-check_residual+1;
								ittt_end=lt-1;
							}
						}
						
						//warn("ittt_beg=%d\n",ittt_beg);					
						//warn("ittt_end=%d\n",ittt_end);

						if(ittt_beg!=0)
						{
							///////velocity
							sprintf(filename,"./check_file/velocity/vx-%d-shot_%d",ishot+1,ittt_beg);
							fread_file_1d(wf_append,nx_append,nz_append,filename);
							cudaMemcpy(vx1_d,wf_append,nxanza*sizeof(float),cudaMemcpyHostToDevice);
						
							sprintf(filename,"./check_file/velocity/vz-%d-shot_%d",ishot+1,ittt_beg);
							fread_file_1d(wf_append,nx_append,nz_append,filename);
							cudaMemcpy(vz1_d,wf_append,nxanza*sizeof(float),cudaMemcpyHostToDevice);

							sprintf(filename,"./check_file/velocity/vxp-%d-shot_%d",ishot+1,ittt_beg);
							fread_file_1d(wf_append,nx_append,nz_append,filename);
							cudaMemcpy(vxp1_d,wf_append,nxanza*sizeof(float),cudaMemcpyHostToDevice);

							sprintf(filename,"./check_file/velocity/vzp-%d-shot_%d",ishot+1,ittt_beg);
							fread_file_1d(wf_append,nx_append,nz_append,filename);
							cudaMemcpy(vzp1_d,wf_append,nxanza*sizeof(float),cudaMemcpyHostToDevice);

							/*sprintf(filename,"./check_file/velocity/vxs-%d-shot_%d",ishot+1,ittt_beg);
							fread_file_1d(wf_append,nx_append,nz_append,filename);
							cudaMemcpy(vxs1_d,wf_append,nxanza*sizeof(float),cudaMemcpyHostToDevice);

							sprintf(filename,"./check_file/velocity/vzs-%d-shot_%d",ishot+1,ittt_beg);
							fread_file_1d(wf_append,nx_append,nz_append,filename);
							cudaMemcpy(vzs1_d,wf_append,nxanza*sizeof(float),cudaMemcpyHostToDevice);*/

							///////stress
							sprintf(filename,"./check_file/stress/txx-%d-shot_%d",ishot+1,ittt_beg);
							fread_file_1d(wf_append,nx_append,nz_append,filename);
							cudaMemcpy(txx1_d,wf_append,nxanza*sizeof(float),cudaMemcpyHostToDevice);

							sprintf(filename,"./check_file/stress/txz-%d-shot_%d",ishot+1,ittt_beg);
							fread_file_1d(wf_append,nx_append,nz_append,filename);
							cudaMemcpy(txz1_d,wf_append,nxanza*sizeof(float),cudaMemcpyHostToDevice);

							sprintf(filename,"./check_file/stress/tzz-%d-shot_%d",ishot+1,ittt_beg);
							fread_file_1d(wf_append,nx_append,nz_append,filename);
							cudaMemcpy(tzz1_d,wf_append,nxanza*sizeof(float),cudaMemcpyHostToDevice);

							sprintf(filename,"./check_file/stress/tp-%d-shot_%d",ishot+1,ittt_beg);
							fread_file_1d(wf_append,nx_append,nz_append,filename);
							cudaMemcpy(tp1_d,wf_append,nxanza*sizeof(float),cudaMemcpyHostToDevice);

							///////memory
							sprintf(filename,"./check_file/memory/mem_p-%d-shot_%d",ishot+1,ittt_beg);
							fread_file_1d(wf_append,nx_append,nz_append,filename);
							cudaMemcpy(mem_p1_d,wf_append,nxanza*sizeof(float),cudaMemcpyHostToDevice);

							sprintf(filename,"./check_file/memory/mem_xx-%d-shot_%d",ishot+1,ittt_beg);
							fread_file_1d(wf_append,nx_append,nz_append,filename);
							cudaMemcpy(mem_xx1_d,wf_append,nxanza*sizeof(float),cudaMemcpyHostToDevice);

							sprintf(filename,"./check_file/memory/mem_zz-%d-shot_%d",ishot+1,ittt_beg);
							fread_file_1d(wf_append,nx_append,nz_append,filename);
							cudaMemcpy(mem_zz1_d,wf_append,nxanza*sizeof(float),cudaMemcpyHostToDevice);

							sprintf(filename,"./check_file/memory/mem_xz-%d-shot_%d",ishot+1,ittt_beg);
							fread_file_1d(wf_append,nx_append,nz_append,filename);
							cudaMemcpy(mem_xz1_d,wf_append,nxanza*sizeof(float),cudaMemcpyHostToDevice);
						}

						//for(int ittt=ittt_beg;ittt<ittt_end;ittt++)
						for(int ittt=ittt_beg+1;ittt<ittt_end+1;ittt++)
						{
							//if(fmod((ittt+1.0),check_interval/2)==0) warn("shot=%d,step=forward,it=%d",ishot+1,(ittt+1));
/////////////////////////////////////////////recalculate and save wavefied						
							if(ittt<wavelet_length&&ittt!=ittt_beg+1)
							{
								add_source<<<1,1>>>(txx1_d,tzz1_d,wavelet_d,source_x_cord[ishot]-receiver_x_cord[ishot],shot_depth,ittt,boundary_up,boundary_left,nz_append);
							}						

							fwd_vx_new<<<dimGrid,dimBlock>>>(vx_t_d,vx2_d,vx1_d,txx1_d,txz1_d,attenuation_d,dx,dz,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,s_density_d);

							fwd_vz_new<<<dimGrid,dimBlock>>>(vz_t_d,vz2_d,vz1_d,tzz1_d,txz1_d,attenuation_d,dx,dz,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,s_density_d);
							if(decomposition!=0||iter==0)
							{
								fwd_vxp_vzp<<<dimGrid,dimBlock>>>(vxp2_d,vxp1_d,vzp2_d,vzp1_d,tp1_d,coe_x,coe_z,dx,dz,dt,attenuation_d,coe_opt_d,nx_append_radius,nz_append_radius,s_density_d);

								vp_vs<<<dimGrid,dimBlock>>>(vx2_d,vz2_d,vxp2_d,vzp2_d,vxs2_d,vzs2_d,nx_append_radius,nz_append_radius);

								decom<<<dimGrid,dimBlock>>>(vx2_d,vz2_d,p_d,s_d,coe_opt_d,nx_append,nz_append,dx,dz);

								//decom_new<<<dimGrid,dimBlock>>>(vx2_d,vz2_d,p_d,s_d,s_velocity_d,s_velocity1_d,coe_opt_d,nx_append_radius,nz_append_radius,dx,dz);
							}

							if(migration_type==0)	fwd_txxzzxzpp_new<<<dimGrid,dimBlock>>>(vx_x_d,vx_z_d,vz_x_d,vz_z_d,tp2_d,tp1_d,txx2_d,txx1_d,tzz2_d,tzz1_d,txz2_d,txz1_d,vx2_d,vz2_d,s_velocity_d,s_velocity1_d,attenuation_d,dt,dx,dz,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,s_density_d);

							//else	fwd_txxzzxzpp_viscoelastic_and_memory<<<dimGrid,dimBlock>>>(tp2_d,tp1_d,txx2_d,txx1_d,tzz2_d,tzz1_d,txz2_d,txz1_d,vx2_d,vz2_d,modul_p_d,modul_s_d,attenuation_d,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,s_density_d,mem_p2_d,mem_p1_d,mem_xx2_d,mem_xx1_d,mem_zz2_d,mem_zz1_d,mem_xz2_d,mem_xz1_d,s_velocity_d,s_velocity1_d,tao_d,strain_p_d,strain_s_d);

							else	fwd_txxzzxzpp_viscoelastic_and_memory_3parameterization_new<<<dimGrid,dimBlock>>>(vx_x_d,vx_z_d,vz_x_d,vz_z_d,tp2_d,tp1_d,txx2_d,txx1_d,tzz2_d,tzz1_d,txz2_d,txz1_d,vx2_d,vz2_d,modul_p_d,modul_s_d,attenuation_d,s_density_d,mem_p2_d,mem_p1_d,mem_xx2_d,mem_xx1_d,mem_zz2_d,mem_zz1_d,mem_xz2_d,mem_xz1_d,s_velocity_d,s_velocity1_d,tao_d,strain_p_d,strain_s_d,packaging_d);
/////////////////////////////////////////////recalculate and save wavefied						
							mark=int(fmod(ittt*1.0,check_interval*1.0));

							save_and_set_wavefiled<<<dimGrid,dimBlock>>>(vx_x_d,&save_vx_x_d[mark*nx_size_nz],nx_size,nz,nx_append,nz_append,boundary_left,boundary_up,0);
							save_and_set_wavefiled<<<dimGrid,dimBlock>>>(vx_z_d,&save_vx_z_d[mark*nx_size_nz],nx_size,nz,nx_append,nz_append,boundary_left,boundary_up,0);
							save_and_set_wavefiled<<<dimGrid,dimBlock>>>(vx_t_d,&save_vx_t_d[mark*nx_size_nz],nx_size,nz,nx_append,nz_append,boundary_left,boundary_up,0);
							save_and_set_wavefiled<<<dimGrid,dimBlock>>>(vz_x_d,&save_vz_x_d[mark*nx_size_nz],nx_size,nz,nx_append,nz_append,boundary_left,boundary_up,0);
							save_and_set_wavefiled<<<dimGrid,dimBlock>>>(vz_z_d,&save_vz_z_d[mark*nx_size_nz],nx_size,nz,nx_append,nz_append,boundary_left,boundary_up,0);
							save_and_set_wavefiled<<<dimGrid,dimBlock>>>(vz_t_d,&save_vz_t_d[mark*nx_size_nz],nx_size,nz,nx_append,nz_append,boundary_left,boundary_up,0);
							if(iter==0||decomposition!=0)
							{
							save_and_set_wavefiled<<<dimGrid,dimBlock>>>(vxp2_d,&save_vxp_d[mark*nx_size_nz],nx_size,nz,nx_append,nz_append,boundary_left,boundary_up,0);
							save_and_set_wavefiled<<<dimGrid,dimBlock>>>(vzp2_d,&save_vzp_d[mark*nx_size_nz],nx_size,nz,nx_append,nz_append,boundary_left,boundary_up,0);
							save_and_set_wavefiled<<<dimGrid,dimBlock>>>(vxs2_d,&save_vxs_d[mark*nx_size_nz],nx_size,nz,nx_append,nz_append,boundary_left,boundary_up,0);
							save_and_set_wavefiled<<<dimGrid,dimBlock>>>(vzs2_d,&save_vzs_d[mark*nx_size_nz],nx_size,nz,nx_append,nz_append,boundary_left,boundary_up,0);
							save_and_set_wavefiled<<<dimGrid,dimBlock>>>(tp2_d,&save_tp_d[mark*nx_size_nz],nx_size,nz,nx_append,nz_append,boundary_left,boundary_up,0);
							save_and_set_wavefiled<<<dimGrid,dimBlock>>>(p_d,&save_p_d[mark*nx_size_nz],nx_size,nz,nx_append,nz_append,boundary_left,boundary_up,0);
							save_and_set_wavefiled<<<dimGrid,dimBlock>>>(s_d,&save_s_d[mark*nx_size_nz],nx_size,nz,nx_append,nz_append,boundary_left,boundary_up,0);
							}
/////////////////////////////////////////////recalculate and save wavefied
						
							if(0==ittt%100&&join_wavefield==1&&iter==0)
							{
								cudaMemcpy(wf_append,vx_z_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
								sprintf(filename,"./wavefield1/2/vx-z-%d-shot_%d",ishot+1,ittt);
								write_file_1d(wf_append,nxanza,filename);
								//exchange(wf_append,wf,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
								//write_file_1d(wf,nx_size_nz,filename);

								cudaMemcpy(wf_append,vz_x_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
								sprintf(filename,"./wavefield1/2/vz-x-%d-shot_%d",ishot+1,ittt);
								write_file_1d(wf_append,nxanza,filename);
								//exchange(wf_append,wf,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
								//write_file_1d(wf,nx_size_nz,filename);

								cudaMemcpy(wf_append,vx2_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
								sprintf(filename,"./wavefield1/2/vx-%d-shot_%d",ishot+1,ittt);
								write_file_1d(wf_append,nxanza,filename);
								exchange(wf_append,wf,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
								write_file_1d(wf,nx_size_nz,filename);

								sprintf(filename,"./wavefield1/1/vx-%d-shot_%d",ishot+1,ittt);
								fread_file_1d(wf_append,nx_append,nz_append,filename);
								cudaMemcpy(wf_append_d,wf_append,nxanza*sizeof(float),cudaMemcpyHostToDevice);
								cal_sub_a_b_to_c<<<dimGrid,dimBlock>>>(vx2_d,wf_append_d,wf_append_d,nx_append,nz_append);
								cudaMemcpy(wf_append,wf_append_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
								sprintf(filename,"./wavefield1/2/vx-difference-%d-shot_%d",ishot+1,ittt);
								write_file_1d(wf_append,nxanza,filename);

								sprintf(filename,"./wavefield1/1/vz-%d-shot_%d",ishot+1,ittt);
								fread_file_1d(wf_append,nx_append,nz_append,filename);
								cudaMemcpy(wf_append_d,wf_append,nxanza*sizeof(float),cudaMemcpyHostToDevice);
								cal_sub_a_b_to_c<<<dimGrid,dimBlock>>>(vz2_d,wf_append_d,wf_append_d,nx_append,nz_append);
								cudaMemcpy(wf_append,wf_append_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
								sprintf(filename,"./wavefield1/2/vz-difference-%d-shot_%d",ishot+1,ittt);
								write_file_1d(wf_append,nxanza,filename);
									
								cudaMemcpy(wf_append,vxp2_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
								sprintf(filename,"./wavefield1/2/vxp-%d-shot_%d",ishot+1,ittt);
								write_file_1d(wf_append,nxanza,filename);
								exchange(wf_append,wf,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
								write_file_1d(wf,nx_size_nz,filename);

								cudaMemcpy(wf_append,vzp2_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
								sprintf(filename,"./wavefield1/2/vzp-%d-shot_%d",ishot+1,ittt);
								write_file_1d(wf_append,nxanza,filename);
								exchange(wf_append,wf,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
								write_file_1d(wf,nx_size_nz,filename);
									
								cudaMemcpy(wf_append,vxs2_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
								sprintf(filename,"./wavefield1/2/vxs-%d-shot_%d",ishot+1,ittt);
								write_file_1d(wf_append,nxanza,filename);
								exchange(wf_append,wf,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
								write_file_1d(wf,nx_size_nz,filename);
	
								cudaMemcpy(wf_append,vzs2_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
								sprintf(filename,"./wavefield1/2/vzs-%d-shot_%d",ishot+1,ittt);
								write_file_1d(wf_append,nxanza,filename);
								exchange(wf_append,wf,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
								write_file_1d(wf,nx_size_nz,filename);

								sprintf(filename,"./wavefield1/1/vxp-%d-shot_%d",ishot+1,ittt);
								fread_file_1d(wf_append,nx_append,nz_append,filename);
								cudaMemcpy(wf_append_d,wf_append,nxanza*sizeof(float),cudaMemcpyHostToDevice);
								cal_sub_a_b_to_c<<<dimGrid,dimBlock>>>(vxp2_d,wf_append_d,wf_append_d,nx_append,nz_append);
								cudaMemcpy(wf_append,wf_append_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
								sprintf(filename,"./wavefield1/2/vxp-difference-%d-shot_%d",ishot+1,ittt);
								write_file_1d(wf_append,nxanza,filename);

								sprintf(filename,"./wavefield1/1/vzs-%d-shot_%d",ishot+1,ittt);
								fread_file_1d(wf_append,nx_append,nz_append,filename);
								cudaMemcpy(wf_append_d,wf_append,nxanza*sizeof(float),cudaMemcpyHostToDevice);
								cal_sub_a_b_to_c<<<dimGrid,dimBlock>>>(vzs2_d,wf_append_d,wf_append_d,nx_append,nz_append);
								cudaMemcpy(wf_append,wf_append_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
								sprintf(filename,"./wavefield1/2/vzs-difference-%d-shot_%d",ishot+1,ittt);
								write_file_1d(wf_append,nxanza,filename);
							}

								rep=vx1_d;vx1_d=vx2_d;vx2_d=rep;
								rep=vz1_d;vz1_d=vz2_d;vz2_d=rep;
								rep=txx1_d;txx1_d=txx2_d;txx2_d=rep;
								rep=tzz1_d;tzz1_d=tzz2_d;tzz2_d=rep;
								rep=txz1_d;txz1_d=txz2_d;txz2_d=rep;

								rep=tp1_d;tp1_d=tp2_d;tp2_d=rep;
								rep=vxp1_d;vxp1_d=vxp2_d;vxp2_d=rep;
								rep=vzp1_d;vzp1_d=vzp2_d;vzp2_d=rep;
								rep=vxs1_d;vxs1_d=vxs2_d;vxs2_d=rep;
								rep=vzs1_d;vzs1_d=vzs2_d;vzs2_d=rep;

								rep=mem_p1_d;mem_p1_d=mem_p2_d;mem_p2_d=rep;
								rep=mem_xx1_d;mem_xx1_d=mem_xx2_d;mem_xx2_d=rep;
								rep=mem_zz1_d;mem_zz1_d=mem_zz2_d;mem_zz2_d=rep;
								rep=mem_xz1_d;mem_xz1_d=mem_xz2_d;mem_xz2_d=rep;
						}

								/*if(join_wavefield==1&&iter==0)
								{
									cudaMemcpy(save_h,save_vx_x_d,check_interval*nx_size_nz*sizeof(float),cudaMemcpyDeviceToHost);
									sprintf(filename,"./wavefield1/save-vx-x-%d-shot_%d",ishot+1,it);
									write_file_1d(save_h,check_interval*nx_size_nz,filename);
								}*/
					}
					
/////////////////////////////////////////////recover/set wavefied						
							mark=int(fmod(it*1.0,check_interval*1.0));

							save_and_set_wavefiled<<<dimGrid,dimBlock>>>(vx_x_d,&save_vx_x_d[mark*nx_size_nz],nx_size,nz,nx_append,nz_append,boundary_left,boundary_up,1);
							save_and_set_wavefiled<<<dimGrid,dimBlock>>>(vx_z_d,&save_vx_z_d[mark*nx_size_nz],nx_size,nz,nx_append,nz_append,boundary_left,boundary_up,1);
							save_and_set_wavefiled<<<dimGrid,dimBlock>>>(vx_t_d,&save_vx_t_d[mark*nx_size_nz],nx_size,nz,nx_append,nz_append,boundary_left,boundary_up,1);
							save_and_set_wavefiled<<<dimGrid,dimBlock>>>(vz_x_d,&save_vz_x_d[mark*nx_size_nz],nx_size,nz,nx_append,nz_append,boundary_left,boundary_up,1);
							save_and_set_wavefiled<<<dimGrid,dimBlock>>>(vz_z_d,&save_vz_z_d[mark*nx_size_nz],nx_size,nz,nx_append,nz_append,boundary_left,boundary_up,1);
							save_and_set_wavefiled<<<dimGrid,dimBlock>>>(vz_t_d,&save_vz_t_d[mark*nx_size_nz],nx_size,nz,nx_append,nz_append,boundary_left,boundary_up,1);
							if(iter==0||decomposition!=0)
							{
							save_and_set_wavefiled<<<dimGrid,dimBlock>>>(vxp2_d,&save_vxp_d[mark*nx_size_nz],nx_size,nz,nx_append,nz_append,boundary_left,boundary_up,1);
							save_and_set_wavefiled<<<dimGrid,dimBlock>>>(vzp2_d,&save_vzp_d[mark*nx_size_nz],nx_size,nz,nx_append,nz_append,boundary_left,boundary_up,1);
							save_and_set_wavefiled<<<dimGrid,dimBlock>>>(vxs2_d,&save_vxs_d[mark*nx_size_nz],nx_size,nz,nx_append,nz_append,boundary_left,boundary_up,1);
							save_and_set_wavefiled<<<dimGrid,dimBlock>>>(vzs2_d,&save_vzs_d[mark*nx_size_nz],nx_size,nz,nx_append,nz_append,boundary_left,boundary_up,1);
							save_and_set_wavefiled<<<dimGrid,dimBlock>>>(tp2_d,&save_tp_d[mark*nx_size_nz],nx_size,nz,nx_append,nz_append,boundary_left,boundary_up,1);
							save_and_set_wavefiled<<<dimGrid,dimBlock>>>(p_d,&save_p_d[mark*nx_size_nz],nx_size,nz,nx_append,nz_append,boundary_left,boundary_up,1);
							save_and_set_wavefiled<<<dimGrid,dimBlock>>>(s_d,&save_s_d[mark*nx_size_nz],nx_size,nz,nx_append,nz_append,boundary_left,boundary_up,1);
							}
////////////////////////////////////////////recover/set wavefied
						
						rep=vx1_d;vx1_d=vx2_d;vx2_d=rep;
						rep=vz1_d;vz1_d=vz2_d;vz2_d=rep;
						rep=txx1_d;txx1_d=txx2_d;txx2_d=rep;
						rep=tzz1_d;tzz1_d=tzz2_d;tzz2_d=rep;
						rep=txz1_d;txz1_d=txz2_d;txz2_d=rep;

						rep=tp1_d;tp1_d=tp2_d;tp2_d=rep;
						rep=vxp1_d;vxp1_d=vxp2_d;vxp2_d=rep;
						rep=vzp1_d;vzp1_d=vzp2_d;vzp2_d=rep;
						rep=vxs1_d;vxs1_d=vxs2_d;vxs2_d=rep;
						rep=vzs1_d;vzs1_d=vzs2_d;vzs2_d=rep;

						rep=mem_p1_d;mem_p1_d=mem_p2_d;mem_p2_d=rep;
						rep=mem_xx1_d;mem_xx1_d=mem_xx2_d;mem_xx2_d=rep;
						rep=mem_zz1_d;mem_zz1_d=mem_zz2_d;mem_zz2_d=rep;
						rep=mem_xz1_d;mem_xz1_d=mem_xz2_d;mem_xz2_d=rep;///////////////QQQQQQQQQQQQQQQ2017年07月27日 星期四 10时11分40秒
///2016年10月08日 星期六 10时03分28秒 伴随状态反传
												
							//wraddshot<<<receiver_num,1>>>(rvx2_d,rvz1_d,res_shot_x_d,res_shot_z_d,it,lt,nz_append,boundary_up,boundary_left,receiver_x_cord[ishot],receiver_interval,receiver_depth,receiver_num);
							//wraddshot_set<<<receiver_num,1>>>(rvx2_d,rvz1_d,res_shot_x_d,res_shot_z_d,it,lt,nz_append,boundary_up,boundary_left,receiver_x_cord[ishot],receiver_interval,receiver_depth,receiver_num);
						if(receiver_offset==0)
						{
							wraddshot_x_z<<<receiver_num,1>>>(rvx2_d,adj_shot_x_d,it,lt,nz_append,boundary_up,boundary_left,receiver_x_cord[ishot],receiver_interval,receiver_z_cord[ishot],receiver_z_interval,receiver_num,accumulation);//for vsp 2017年03月14日 星期二 09时02分11秒 
							wraddshot_x_z<<<receiver_num,1>>>(rvz2_d,adj_shot_z_d,it,lt,nz_append,boundary_up,boundary_left,receiver_x_cord[ishot],receiver_interval,receiver_z_cord[ishot],receiver_z_interval,receiver_num,accumulation);//for vsp 2017年03月14日 星期二 09时02分11秒 
						}
						else//correlation
						{
							wraddshot_x_z_acqusition<<<receiver_num,1>>>(rvx2_d,adj_shot_x_d,it,lt,nz_append,boundary_up,boundary_left,receiver_x_cord[ishot],receiver_interval,receiver_z_cord[ishot],receiver_z_interval,receiver_num,accumulation);//for vsp 2017年03月14日 星期二 09时02分11秒 
							wraddshot_x_z_acqusition<<<receiver_num,1>>>(rvz2_d,adj_shot_z_d,it,lt,nz_append,boundary_up,boundary_left,receiver_x_cord[ishot],receiver_interval,receiver_z_cord[ishot],receiver_z_interval,receiver_num,accumulation);//for vsp 2017年03月14日 星期二 09时02分11秒 
						}
					

						if(RTM_only==0)
						{
							if(migration_type==0)
							{	
								///2016年10月08日 星期六 10时03分28秒 伴随状态反传	
								//receiver wavefield reverse propagation   (vetor)						
								/*wraddshot<<<receiver_num,1>>>(rvx2_d,rvz2_d,res_shot_x_d,res_shot_z_d,it,lt,nz_append,boundary_up,boundary_left,receiver_x_cord[ishot],receiver_interval,receiver_depth,receiver_num);
										
								fwd_txxzzxz<<<dimGrid,dimBlock>>>(rtxx1_d,rtxx2_d,rtzz1_d,rtzz2_d,rtxz1_d,rtxz2_d,rvx2_d,rvz2_d,s_velocity_d,s_velocity1_d,attenuation_d,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,s_density_d);		
								fwd_vx<<<dimGrid,dimBlock>>>(rvx1_d,rvx2_d,rtxx1_d,rtxz1_d,attenuation_d,dx,dz,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,s_density_d);						
								fwd_vz<<<dimGrid,dimBlock>>>(rvz1_d,rvz2_d,rtzz1_d,rtxz1_d,attenuation_d,dx,dz,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,s_density_d);*/
								adjoint_fwd_txxzzxzpp<<<dimGrid,dimBlock>>>(rtp1_d,rtp2_d,rtxx1_d,rtxx2_d,rtzz1_d,rtzz2_d,rtxz1_d,rtxz2_d,rvx2_d,rvz2_d,s_velocity_d,s_velocity1_d,attenuation_d,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,s_density_d);

								adjoint_fwd_vx<<<dimGrid,dimBlock>>>(rvx1_d,rvx2_d,rtxx1_d,rtxz1_d,rtzz1_d,attenuation_d,dx,dz,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,s_velocity_d,s_velocity1_d,s_density_d);

								adjoint_fwd_vz<<<dimGrid,dimBlock>>>(rvz1_d,rvz2_d,rtxx1_d,rtxz1_d,rtzz1_d,attenuation_d,dx,dz,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,s_velocity_d,s_velocity1_d,s_density_d);

								if(decomposition!=0||iter==0)
								{
									fwd_vxp_vzp<<<dimGrid,dimBlock>>>(rvxp1_d,rvxp2_d,rvzp1_d,rvzp2_d,rtp1_d,coe_x,coe_z,dx,dz,dt,attenuation_d,coe_opt_d,nx_append_radius,nz_append_radius,s_density_d);////////////////////////why??????????????Wang's method works!!!!

									vp_vs<<<dimGrid,dimBlock>>>(rvx1_d,rvz1_d,rvxp1_d,rvzp1_d,rvxs1_d,rvzs1_d,nx_append_radius,nz_append_radius);

									decom<<<dimGrid,dimBlock>>>(rvx1_d,rvz1_d,rp_d,rs_d,coe_opt_d,nx_append,nz_append,dx,dz);

									//decom_new<<<dimGrid,dimBlock>>>(rvx1_d,rvz1_d,rp_d,rs_d,s_velocity_d,s_velocity1_d,coe_opt_d,nx_append_radius,nz_append_radius,dx,dz);///////////////divergence and curl operator  and amplitude correction in Li's method(2016)

									//poynting<<<dimGrid,dimBlock>>>(rtxx1_d,rtxz1_d,rtzz1_d,rvx1_d,rvz1_d,poyn_rz_d,poyn_rx_d,nx_append_radius,nz_append_radius);						
									//sum_poynting<<<dimGrid,dimBlock>>>(poyn_rpx_d,poyn_rpz_d,poyn_rsx_d,poyn_rsz_d,rvxp1_d,rvzp1_d,rvxs1_d,rvzs1_d,rtxx1_d,rtzz1_d,rtxz1_d,rtp1_d,nx_append_radius,nz_append_radius);
								}		
							}
					
							if(migration_type==1)
							{
								/////////////////////////////////adjoint equation in viscoelastic media  for viscoelastic LSRTM
								adjoint_fwd_txxzzxzpp<<<dimGrid,dimBlock>>>(rtp1_d,rtp2_d,rtxx1_d,rtxx2_d,rtzz1_d,rtzz2_d,rtxz1_d,rtxz2_d,rvx2_d,rvz2_d,s_velocity_d,s_velocity1_d,attenuation_d,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,s_density_d);
							
								adjoint_fwd_memory<<<dimGrid,dimBlock>>>(rmem_p1_d,rmem_p2_d,rmem_xx1_d,rmem_xx2_d,rmem_zz1_d,rmem_zz2_d,rmem_xz1_d,rmem_xz2_d,rtp1_d,rtxx1_d,rtzz1_d,rtxz1_d,attenuation_d,dx,dz,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,s_density_d,s_velocity_d,s_velocity1_d,tao_d,strain_p_d,strain_s_d);
								adjoint_fwd_vx_viscoelastic<<<dimGrid,dimBlock>>>(rvx1_d,rvx2_d,rtxx1_d,rtxz1_d,rtzz1_d,attenuation_d,dx,dz,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,modul_p_d,modul_s_d,s_density_d,rmem_p1_d,rmem_xx1_d,rmem_zz1_d,rmem_xz1_d,s_velocity_d,s_velocity1_d,tao_d,strain_p_d,strain_s_d);
								adjoint_fwd_vz_viscoelastic<<<dimGrid,dimBlock>>>(rvz1_d,rvz2_d,rtxx1_d,rtxz1_d,rtzz1_d,attenuation_d,dx,dz,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,modul_p_d,modul_s_d,s_density_d,rmem_p1_d,rmem_xx1_d,rmem_zz1_d,rmem_xz1_d,s_velocity_d,s_velocity1_d,tao_d,strain_p_d,strain_s_d);
								if(decomposition!=0||iter==0)
								{
									fwd_vxp_vzp<<<dimGrid,dimBlock>>>(rvxp1_d,rvxp2_d,rvzp1_d,rvzp2_d,rtp1_d,coe_x,coe_z,dx,dz,dt,attenuation_d,coe_opt_d,nx_append_radius,nz_append_radius,s_density_d);	
										
									vp_vs<<<dimGrid,dimBlock>>>(rvx1_d,rvz1_d,rvxp1_d,rvzp1_d,rvxs1_d,rvzs1_d,nx_append_radius,nz_append_radius);

									decom<<<dimGrid,dimBlock>>>(rvx1_d,rvz1_d,rp_d,rs_d,coe_opt_d,nx_append,nz_append,dx,dz);

									//decom_new<<<dimGrid,dimBlock>>>(rvx1_d,rvz1_d,rp_d,rs_d,s_velocity_d,s_velocity1_d,coe_opt_d,nx_append_radius,nz_append_radius,dx,dz);///////////////divergence and curl operator  and amplitude correction in Li's method(2016)

									//poynting<<<dimGrid,dimBlock>>>(rtxx1_d,rtxz1_d,rtzz1_d,rvx1_d,rvz1_d,poyn_rz_d,poyn_rx_d,nx_append_radius,nz_append_radius);						
									//sum_poynting<<<dimGrid,dimBlock>>>(poyn_rpx_d,poyn_rpz_d,poyn_rsx_d,poyn_rsz_d,rvxp1_d,rvzp1_d,rvxs1_d,rvzs1_d,rtxx1_d,rtzz1_d,rtxz1_d,rtp1_d,nx_append_radius,nz_append_radius)
								}

								/////////////////////////////////receiver propgagation equation in viscoelastic media  for viscoelastic RTM
								/*receiver_fwd_txxzzxzpp_viscoelastic_and_memory_3parameterization<<<dimGrid,dimBlock>>>(rtp1_d,rtp2_d,rtxx1_d,rtxx2_d,rtzz1_d,rtzz2_d,rtxz1_d,rtxz2_d,rvx2_d,rvz2_d,modul_p_d,modul_s_d,attenuation_d,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,s_density_d,rmem_p1_d,rmem_p2_d,rmem_xx1_d,rmem_xx2_d,rmem_zz1_d,rmem_zz2_d,rmem_xz1_d,rmem_xz2_d,s_velocity_d,s_velocity1_d,tao_d,strain_p_d,strain_s_d);
			
								fwd_vx<<<dimGrid,dimBlock>>>(rvx1_d,rvx2_d,rtxx1_d,rtxz1_d,attenuation_d,dx,dz,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,s_density_d);
							
								fwd_vz<<<dimGrid,dimBlock>>>(rvz1_d,rvz2_d,rtzz1_d,rtxz1_d,attenuation_d,dx,dz,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,s_density_d);
											
								fwd_vxp_vzp<<<dimGrid,dimBlock>>>(rvxp1_d,rvxp2_d,rvzp1_d,rvzp2_d,rtp1_d,coe_x,coe_z,dx,dz,dt,attenuation_d,coe_opt_d,nx_append_radius,nz_append_radius,s_density_d);	
										
								vp_vs<<<dimGrid,dimBlock>>>(rvx1_d,rvz1_d,rvxp1_d,rvzp1_d,rvxs1_d,rvzs1_d,nx_append_radius,nz_append_radius);

								decom<<<dimGrid,dimBlock>>>(rvx1_d,rvz1_d,rp_d,rs_d,coe_opt_d,nx_append,nz_append,dx,dz);

								//decom_new<<<dimGrid,dimBlock>>>(rvx1_d,rvz1_d,rp_d,rs_d,s_velocity_d,s_velocity1_d,coe_opt_d,nx_append_radius,nz_append_radius,dx,dz);///////////////divergence and curl operator  and amplitude correction in Li's method(2016)

								//poynting<<<dimGrid,dimBlock>>>(rtxx1_d,rtxz1_d,rtzz1_d,rvx1_d,rvz1_d,poyn_rz_d,poyn_rx_d,nx_append_radius,nz_append_radius);*/
							}
						}
						
						else
						{
								fwd_txxzzxzpp<<<dimGrid,dimBlock>>>(rtp1_d,rtp2_d,rtxx1_d,rtxx2_d,rtzz1_d,rtzz2_d,rtxz1_d,rtxz2_d,rvx2_d,rvz2_d,s_velocity_d,s_velocity1_d,attenuation_d,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,s_density_d);
		
								fwd_vx<<<dimGrid,dimBlock>>>(rvx1_d,rvx2_d,rtxx1_d,rtxz1_d,attenuation_d,dx,dz,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,s_density_d);						
								fwd_vz<<<dimGrid,dimBlock>>>(rvz1_d,rvz2_d,rtzz1_d,rtxz1_d,attenuation_d,dx,dz,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,s_density_d);

								fwd_vxp_vzp<<<dimGrid,dimBlock>>>(rvxp1_d,rvxp2_d,rvzp1_d,rvzp2_d,rtp1_d,coe_x,coe_z,dx,dz,dt,attenuation_d,coe_opt_d,nx_append_radius,nz_append_radius,s_density_d);	
										
								vp_vs<<<dimGrid,dimBlock>>>(rvx1_d,rvz1_d,rvxp1_d,rvzp1_d,rvxs1_d,rvzs1_d,nx_append_radius,nz_append_radius);

								decom<<<dimGrid,dimBlock>>>(rvx1_d,rvz1_d,rp_d,rs_d,coe_opt_d,nx_append,nz_append,dx,dz);

								//decom_new<<<dimGrid,dimBlock>>>(rvx1_d,rvz1_d,rp_d,rs_d,s_velocity_d,s_velocity1_d,coe_opt_d,nx_append_radius,nz_append_radius,dx,dz);///////////////divergence and curl operator  and amplitude correction in Li's method(2016)

								//poynting<<<dimGrid,dimBlock>>>(rtxx1_d,rtxz1_d,rtzz1_d,rvx1_d,rvz1_d,poyn_rz_d,poyn_rx_d,nx_append_radius,nz_append_radius);						
								sum_poynting<<<dimGrid,dimBlock>>>(poyn_rpx_d,poyn_rpz_d,poyn_rsx_d,poyn_rsz_d,rvxp1_d,rvzp1_d,rvxs1_d,rvzs1_d,rtxx1_d,rtzz1_d,rtxz1_d,rtp1_d,nx_append_radius,nz_append_radius);

								cal_direction_2D_elastic<<<dimGrid,dimBlock>>>(direction_rpx_d,direction_rpz_d,direction_rsx_d,direction_rsz_d,rvxp1_d,rvzp1_d,rvxs1_d,rvzs1_d,rtxx1_d,rtzz1_d,rtxz1_d,rtp1_d,nx_append_radius,nz_append_radius);
						}					
		
							if(0==(it)%100&&join_wavefield==1&&iter==0)
							{
								cudaMemcpy(wf_append,rvx2_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
								sprintf(filename,"./wavefield1/3/vx-%d-shot_%d",ishot+1,it);
								write_file_1d(wf_append,nxanza,filename);
								exchange(wf_append,wf,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
								write_file_1d(wf,nx_size_nz,filename);
																		
								cudaMemcpy(wf_append,rvxp2_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
								sprintf(filename,"./wavefield1/3/vxp-%d-shot_%d",ishot+1,it);
								write_file_1d(wf_append,nxanza,filename);
								exchange(wf_append,wf,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
								write_file_1d(wf,nx_size_nz,filename);
										
								cudaMemcpy(wf_append,rvxs2_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
								sprintf(filename,"./wavefield1/3/vxs-%d-shot_%d",ishot+1,it);
								write_file_1d(wf_append,nxanza,filename);
								exchange(wf_append,wf,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
								write_file_1d(wf,nx_size_nz,filename);

								cudaMemcpy(wf_append,rvz2_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
								sprintf(filename,"./wavefield1/3/vz-%d-shot_%d",ishot+1,it);
								write_file_1d(wf_append,nxanza,filename);
								exchange(wf_append,wf,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
								write_file_1d(wf,nx_size_nz,filename);
																		
								cudaMemcpy(wf_append,rvzp2_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
								sprintf(filename,"./wavefield1/3/vzp-%d-shot_%d",ishot+1,it);
								write_file_1d(wf_append,nxanza,filename);
								exchange(wf_append,wf,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
								write_file_1d(wf,nx_size_nz,filename);
										
								cudaMemcpy(wf_append,rvzs2_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
								sprintf(filename,"./wavefield1/3/vzs-%d-shot_%d",ishot+1,it);
								write_file_1d(wf_append,nxanza,filename);
								exchange(wf_append,wf,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
								write_file_1d(wf,nx_size_nz,filename);

								cudaMemcpy(wf_append,poyn_rpz_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
								sprintf(filename,"./wavefield1/3/pz-%d-shot_%d",ishot+1,it);
								write_file_1d(wf_append,nxanza,filename);
								exchange(wf_append,wf,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
								write_file_1d(wf,nx_size_nz,filename);

								cudaMemcpy(wf_append,poyn_rsz_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
								sprintf(filename,"./wavefield1/3/sz-%d-shot_%d",ishot+1,it);
								write_file_1d(wf_append,nxanza,filename);
								exchange(wf_append,wf,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
								write_file_1d(wf,nx_size_nz,filename);


								cudaMemcpy(wf_append,poyn_rpx_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
								sprintf(filename,"./wavefield1/3/px-%d-shot_%d",ishot+1,it);
								write_file_1d(wf_append,nxanza,filename);
								exchange(wf_append,wf,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
								write_file_1d(wf,nx_size_nz,filename);

								cudaMemcpy(wf_append,poyn_rsx_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
								sprintf(filename,"./wavefield1/3/sx-%d-shot_%d",ishot+1,it);
								write_file_1d(wf_append,nxanza,filename);
								exchange(wf_append,wf,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
								write_file_1d(wf,nx_size_nz,filename);
							}

							rep=rvx1_d;rvx1_d=rvx2_d;rvx2_d=rep;
							rep=rvz1_d;rvz1_d=rvz2_d;rvz2_d=rep;
							rep=rtxx1_d;rtxx1_d=rtxx2_d;rtxx2_d=rep;
							rep=rtzz1_d;rtzz1_d=rtzz2_d;rtzz2_d=rep;
							rep=rtxz1_d;rtxz1_d=rtxz2_d;rtxz2_d=rep;

							rep=rtp1_d;rtp1_d=rtp2_d;rtp2_d=rep;
							rep=rvxp1_d;rvxp1_d=rvxp2_d;rvxp2_d=rep;
							rep=rvzp1_d;rvzp1_d=rvzp2_d;rvzp2_d=rep;
							rep=rvxs1_d;rvxs1_d=rvxs2_d;rvxs2_d=rep;
							rep=rvzs1_d;rvzs1_d=rvzs2_d;rvzs2_d=rep;/////fast...........................................

							rep=rmem_p1_d;rmem_p1_d=rmem_p2_d;rmem_p2_d=rep;
							rep=rmem_xx1_d;rmem_xx1_d=rmem_xx2_d;rmem_xx2_d=rep;
							rep=rmem_zz1_d;rmem_zz1_d=rmem_zz2_d;rmem_zz2_d=rep;
							rep=rmem_xz1_d;rmem_xz1_d=rmem_xz2_d;rmem_xz2_d=rep;


						if(iter==0)
						{
							imaging_correlation<<<dimGrid,dimBlock>>>(tp1_d,tp1_d,down_tp_d,nx_size,nz,nz_append,boundary_up,boundary_left);
			
							imaging_correlation<<<dimGrid,dimBlock>>>(p_d,p_d,down_pp_d,nx_size,nz,nz_append,boundary_up,boundary_left);
							imaging_correlation<<<dimGrid,dimBlock>>>(p_d,s_d,down_ss_d,nx_size,nz,nz_append,boundary_up,boundary_left);

							imaging_correlation<<<dimGrid,dimBlock>>>(vxp1_d,vxp1_d,down_vpp_x_d,nx_size,nz,nz_append,boundary_up,boundary_left);
							imaging_correlation<<<dimGrid,dimBlock>>>(vzp1_d,vzp1_d,down_vpp_z_d,nx_size,nz,nz_append,boundary_up,boundary_left);

							imaging_correlation<<<dimGrid,dimBlock>>>(vxs1_d,vxs1_d,down_vss_x_d,nx_size,nz,nz_append,boundary_up,boundary_left);
							imaging_correlation<<<dimGrid,dimBlock>>>(vzs1_d,vzs1_d,down_vss_z_d,nx_size,nz,nz_append,boundary_up,boundary_left);
							imaging_vector_correlation<<<dimGrid,dimBlock>>>(vxp1_d,vzp1_d,vxp1_d,vzp1_d,down_vpp_d,nx_size,nz,nz_append,boundary_up,boundary_left);
							imaging_vector_correlation<<<dimGrid,dimBlock>>>(vxs1_d,vzs1_d,vxs1_d,vzs1_d,down_vss_d,nx_size,nz,nz_append,boundary_up,boundary_left);
			
							//imaging_vector_correlation_new<<<dimGrid,dimBlock>>>(vxp1_d,vzp1_d,vxp1_d,vzp1_d,down_vpp_d,nx_size,nz,nz_append,boundary_up,boundary_left);
							//imaging_vector_correlation_new<<<dimGrid,dimBlock>>>(vxs1_d,vzs1_d,vxs1_d,vzs1_d,down_vss_d,nx_size,nz,nz_append,boundary_up,boundary_left);
///////////////////////////////////////////////////////////excitation imaing condition
///////////////////////////////////////////////////////one method to calculate reflected angle
							caculate_ex_angle_rp_only_RTM<<<dimGrid,dimBlock>>>(ex_angle_rpp_d,poyn_rpx_d,poyn_rpz_d,ex_time_d,it,nx_append_radius,nz_append_radius);
							caculate_ex_angle_rp_only_RTM<<<dimGrid,dimBlock>>>(ex_angle_rps_d,poyn_rsx_d,poyn_rsz_d,ex_time_d,it,nx_append_radius,nz_append_radius);
///////////////////////////////////////////////////////one method to calculate reflected angle
							caculate_ex_angle_rp_only_RTM<<<dimGrid,dimBlock>>>(ex_angle_rpp1_d,direction_rpx_d,direction_rpz_d,ex_time_d,it,nx_append_radius,nz_append_radius);
							caculate_ex_angle_rp_only_RTM<<<dimGrid,dimBlock>>>(ex_angle_rps1_d,direction_rsx_d,direction_rsz_d,ex_time_d,it,nx_append_radius,nz_append_radius);

							imaging_correlation_ex_2D<<<dimGrid,dimBlock>>>(ex_result_tp_d,ex_amp_tp_d,ex_tp_time_d,rtp2_d,nx_size,nz,nz_append,boundary_up,boundary_left,&para_max_d[0],it);

							imaging_correlation_ex_2D<<<dimGrid,dimBlock>>>(ex_result_tp_old_d,ex_amp_tp_old_d,ex_time_d,rtp2_d,nx_size,nz,nz_append,boundary_up,boundary_left,&para_max_d[0],it);


							imaging_inner_product_ex_2D<<<dimGrid,dimBlock>>>(ex_vresultpp_d,ex_amp_d,ex_amp_x_d,ex_amp_z_d,ex_time_d,rvxp2_d,rvzp2_d,nx_size,nz,nz_append,boundary_up,boundary_left,&para_max_d[1],it);
							imaging_inner_product_ex_2D<<<dimGrid,dimBlock>>>(ex_vresultps_d,ex_amp_d,ex_amp_x_d,ex_amp_z_d,ex_time_d,rvxs2_d,rvzs2_d,nx_size,nz,nz_append,boundary_up,boundary_left,&para_max_d[1],it);

							//imaging_inner_product_ex_2D_new<<<dimGrid,dimBlock>>>(ex_vresultpp_d,ex_amp_d,ex_amp_x_d,ex_amp_z_d,ex_time_d,rvxp2_d,rvzp2_d,nx_size,nz,nz_append,boundary_up,boundary_left,&para_max_d[1],it);
							//imaging_inner_product_ex_2D_new<<<dimGrid,dimBlock>>>(ex_vresultps_d,ex_amp_d,ex_amp_x_d,ex_amp_z_d,ex_time_d,rvxs2_d,rvzs2_d,nx_size,nz,nz_append,boundary_up,boundary_left,&para_max_d[1],it);

							//imaging_pp_compensate_dependent_angle_2D<<<dimGrid,dimBlock>>>(ex_angle_pp1_d,ex_angle_rpp1_d,com_ex_vresultpp_d,ex_vresultpp_d,ex_time_d,nx_size,nz,nz_append,boundary_up,boundary_left,it);

							//imaging_ps_compensate_dependent_angle_2D<<<dimGrid,dimBlock>>>(ex_angle_pp1_d,ex_angle_rps1_d,com_ex_vresultps_d,ex_vresultps_d,ex_time_d,nx_size,nz,nz_append,boundary_up,boundary_left,it);
							
							caculate_ex_open_pp_ps<<<dimGrid,dimBlock>>>(ex_open_pp_d,ex_angle_pp_d,ex_angle_rpp_d,nx_size,nz,nx_append,nz_append,boundary_up,boundary_left,it,ex_time_d);
							caculate_ex_open_pp_ps<<<dimGrid,dimBlock>>>(ex_open_ps_d,ex_angle_pp_d,ex_angle_rps_d,nx_size,nz,nx_append,nz_append,boundary_up,boundary_left,it,ex_time_d);

							caculate_ex_open_pp_ps<<<dimGrid,dimBlock>>>(ex_open_pp1_d,ex_angle_pp1_d,ex_angle_rpp1_d,nx_size,nz,nx_append,nz_append,boundary_up,boundary_left,it,ex_time_d);
							caculate_ex_open_pp_ps<<<dimGrid,dimBlock>>>(ex_open_ps1_d,ex_angle_pp1_d,ex_angle_rps1_d,nx_size,nz,nx_append,nz_append,boundary_up,boundary_left,it,ex_time_d);

//////////////////////////////////////////////////////////////xx or zz
							imaging_correlation_for_xxzz<<<dimGrid,dimBlock>>>(vxp1_d,vxs1_d,vxp1_d,vxs1_d,down_xx_d,nx_size,nz,nz_append,boundary_up,boundary_left);
							imaging_correlation_for_xxzz<<<dimGrid,dimBlock>>>(vzp1_d,vzs1_d,vzp1_d,vzs1_d,down_zz_d,nx_size,nz,nz_append,boundary_up,boundary_left);
							imaging_correlation_for_xxzz<<<dimGrid,dimBlock>>>(vxp1_d,vxs1_d,rvxp1_d,rvxs1_d,resultxx_d,nx_size,nz,nz_append,boundary_up,boundary_left);
							imaging_correlation_for_xxzz<<<dimGrid,dimBlock>>>(vzp1_d,vzs1_d,rvzp1_d,rvzs1_d,resultzz_d,nx_size,nz,nz_append,boundary_up,boundary_left);							
/////////////based on Li/Du' method 2016/2012  correction
							//ps8				
							/*set_sign_basedon_polarization_ps<<<dimGrid,dimBlock>>>(vxp1_d,vzp1_d,rvxs1_d,rvzs1_d,signx_d,signy_d,signz_d,nx_append_radius,nz_append_radius);					
							filter_sign_new_share<<<dimGrid,dimBlock>>>(signx_d,filter_signx_d,nx_append_radius,nz_append_radius,scale);
							filter_sign_new_share<<<dimGrid,dimBlock>>>(signy_d,filter_signy_d,nx_append_radius,nz_append_radius,scale);
							filter_sign_new_share<<<dimGrid,dimBlock>>>(signz_d,filter_signz_d,nx_append_radius,nz_append_radius,scale);
							compare_sign<<<dimGrid,dimBlock>>>(filter_signx_d,filter_signy_d,filter_signz_d,sign_d,nx_append_radius,nz_append_radius);
							imaging_correlation_sign_ps<<<dimGrid,dimBlock>>>(p_d,rs_d,resultps2_d,sign_d,source_x_cord[ishot],nx_size,nz,nz_append,boundary_up,boundary_left);
//sp8
							set_sign_basedon_polarization_sp<<<dimGrid,dimBlock>>>(vxs1_d,vzs1_d,rvxp1_d,rvzp1_d,signx_d,signy_d,signz_d,nx_append_radius,nz_append_radius);					
							filter_sign_new_share<<<dimGrid,dimBlock>>>(signx_d,filter_signx_d,nx_append_radius,nz_append_radius,scale);
							filter_sign_new_share<<<dimGrid,dimBlock>>>(signy_d,filter_signy_d,nx_append_radius,nz_append_radius,scale);
							filter_sign_new_share<<<dimGrid,dimBlock>>>(signz_d,filter_signz_d,nx_append_radius,nz_append_radius,scale);
							compare_sign<<<dimGrid,dimBlock>>>(filter_signx_d,filter_signy_d,filter_signz_d,sign_d,nx_append_radius,nz_append_radius);
							imaging_correlation_sign<<<dimGrid,dimBlock>>>(s_d,rp_d,resultsp2_d,sign_d,nx_size,nz,nz_append,boundary_up,boundary_left);*/
//ADCIGS					
							/*set_sign_forps<<<dimGrid,dimBlock>>>(poyn_x_d,poyn_z_d,poyn_rx_d,poyn_rz_d,signx_d,signy_d,signz_d,nx_append_radius,nz_append_radius);				
							filter_sign_new_share<<<dimGrid,dimBlock>>>(signx_d,filter_signx_d,nx_append_radius,nz_append_radius,scale);
							filter_sign_new_share<<<dimGrid,dimBlock>>>(signy_d,filter_signy_d,nx_append_radius,nz_append_radius,scale);
							filter_sign_new_share<<<dimGrid,dimBlock>>>(signz_d,filter_signz_d,nx_append_radius,nz_append_radius,scale);
							compare_sign<<<dimGrid,dimBlock>>>(filter_signx_d,filter_signy_d,filter_signz_d,sign_d,nx_append_radius,nz_append_radius);
							imaging_correlation_sign<<<dimGrid,dimBlock>>>(p_d,rs_d,resultps2_d,sign_d,nx_size,nz,nz_append,boundary_up,boundary_left);
							//imaging_correlation_sign_ps<<<dimGrid,dimBlock>>>(p_d,rs_d,resultps2_d,sign_d,source_x_cord[ishot],nx_size,nz,nz_append,boundary_up,boundary_left);
							imaging_correlation_sign<<<dimGrid,dimBlock>>>(s_d,rp_d,resultsp2_d,sign_d,nx_size,nz,nz_append,boundary_up,boundary_left);
							//imaging_correlation_sign_ps<<<dimGrid,dimBlock>>>(s_d,rp_d,resultsp2_d,sign_d,source_x_cord[ishot],nx_size,nz,nz_append,boundary_up,boundary_left);*/
/////////////based on Li/Du' method 2016/2012  correction
							//pp ps ps1
							imaging_correlation<<<dimGrid,dimBlock>>>(p_d,rp_d,resultpp_d,nx_size,nz,nz_append,boundary_up,boundary_left);
							imaging_correlation<<<dimGrid,dimBlock>>>(p_d,rs_d,resultps_d,nx_size,nz,nz_append,boundary_up,boundary_left);
							imaging_correlation<<<dimGrid,dimBlock>>>(s_d,rp_d,resultsp_d,nx_size,nz,nz_append,boundary_up,boundary_left);
							imaging_correlation<<<dimGrid,dimBlock>>>(s_d,rs_d,resultss_d,nx_size,nz,nz_append,boundary_up,boundary_left);
							
							imaging_correlation_source_x_cord<<<dimGrid,dimBlock>>>(p_d,rs_d,resultps1_d,nx_size,nz,nz_append,boundary_up,boundary_left,source_x_cord[ishot]-receiver_x_cord[ishot]);
							imaging_correlation_source_x_cord<<<dimGrid,dimBlock>>>(s_d,rp_d,resultsp1_d,nx_size,nz,nz_append,boundary_up,boundary_left,source_x_cord[ishot]-receiver_x_cord[ishot]);
							/////tp*tp
							imaging_correlation<<<dimGrid,dimBlock>>>(tp1_d,rtp1_d,result_tp_d,nx_size,nz,nz_append,boundary_up,boundary_left);
							/////vppx vppz
							imaging_correlation<<<dimGrid,dimBlock>>>(vxp1_d,rvxp1_d,vresultppx_d,nx_size,nz,nz_append,boundary_up,boundary_left);
							imaging_correlation<<<dimGrid,dimBlock>>>(vzp1_d,rvzp1_d,vresultppz_d,nx_size,nz,nz_append,boundary_up,boundary_left);
							/////vpsx vpsz
							imaging_correlation<<<dimGrid,dimBlock>>>(vxp1_d,rvxs1_d,vresultpsx_d,nx_size,nz,nz_append,boundary_up,boundary_left);
							imaging_correlation<<<dimGrid,dimBlock>>>(vzp1_d,rvzs1_d,vresultpsz_d,nx_size,nz,nz_append,boundary_up,boundary_left);
							/////vspx vspz
							imaging_correlation<<<dimGrid,dimBlock>>>(vxs1_d,rvxp1_d,vresultspx_d,nx_size,nz,nz_append,boundary_up,boundary_left);
							imaging_correlation<<<dimGrid,dimBlock>>>(vzs1_d,rvzp1_d,vresultspz_d,nx_size,nz,nz_append,boundary_up,boundary_left);
							/////vssx vssz
							imaging_correlation<<<dimGrid,dimBlock>>>(vxs1_d,rvxs1_d,vresultssx_d,nx_size,nz,nz_append,boundary_up,boundary_left);
							imaging_correlation<<<dimGrid,dimBlock>>>(vzs1_d,rvzs1_d,vresultssz_d,nx_size,nz,nz_append,boundary_up,boundary_left);
							/////vpp vps vsp vss
							imaging_vector_correlation<<<dimGrid,dimBlock>>>(vxp1_d,vzp1_d,rvxp1_d,rvzp1_d,vresultpp_d,nx_size,nz,nz_append,boundary_up,boundary_left);
							imaging_vector_correlation<<<dimGrid,dimBlock>>>(vxp1_d,vzp1_d,rvxs1_d,rvzs1_d,vresultps_d,nx_size,nz,nz_append,boundary_up,boundary_left);
							imaging_vector_correlation<<<dimGrid,dimBlock>>>(vxs1_d,vzs1_d,rvxp1_d,rvzp1_d,vresultsp_d,nx_size,nz,nz_append,boundary_up,boundary_left);
							imaging_vector_correlation<<<dimGrid,dimBlock>>>(vxs1_d,vzs1_d,rvxs1_d,rvzs1_d,vresultss_d,nx_size,nz,nz_append,boundary_up,boundary_left); 
							//imaging_vector_correlation_new<<<dimGrid,dimBlock>>>(vxp1_d,vzp1_d,rvxp1_d,rvzp1_d,vresultpp_d,nx_size,nz,nz_append,boundary_up,boundary_left);
							//imaging_vector_correlation_new<<<dimGrid,dimBlock>>>(vxp1_d,vzp1_d,rvxs1_d,rvzs1_d,vresultps_d,nx_size,nz,nz_append,boundary_up,boundary_left);
							//imaging_vector_correlation_new<<<dimGrid,dimBlock>>>(vxs1_d,vzs1_d,rvxp1_d,rvzp1_d,vresultsp_d,nx_size,nz,nz_append,boundary_up,boundary_left);
							//imaging_vector_correlation_new<<<dimGrid,dimBlock>>>(vxs1_d,vzs1_d,rvxs1_d,rvzs1_d,vresultss_d,nx_size,nz,nz_append,boundary_up,boundary_left);
						}

						if(decomposition==0)
						{
							if(migration_type==0)
							{					
////cal_gradient_for_lame1
								//cal_gradient_for_lame1_mul<<<dimGrid,dimBlock>>>(grad_lame11_d,rtxx2_d,rtzz2_d,vx_x_d,vz_z_d,boundary_left,boundary_up,nx_size,nz,nx_append,nz_append,s_velocity_d,s_velocity1_d,s_density_d);
	////cal_gradient_for_lame2
								//cal_gradient_for_lame2_mul<<<dimGrid,dimBlock>>>(grad_lame22_d,rtxx2_d,rtxz2_d,rtzz2_d,vx_x_d,vz_z_d,vx_z_d,vz_x_d,boundary_left,boundary_up,nx_size,nz,nx_append,nz_append,s_velocity_d,s_velocity1_d,s_density_d);
////cal_gradient_for_density							
								//cal_gradient_for_den_mul<<<dimGrid,dimBlock>>>(grad_den1_d,vx_t_d,vz_t_d,rvx2_d,rvz2_d,dt,boundary_left,boundary_up,nx_size,nz,nx_append,nz_append,s_velocity_d,s_velocity1_d,s_density_d);

								cal_gradient_in_elastic_media<<<dimGrid,dimBlock>>>(grad_lame11_d,grad_lame22_d,grad_den1_d,vx_t_d,vz_t_d,vx_x_d,vz_z_d,vx_z_d,vz_x_d,rvx2_d,rvz2_d,rtxx2_d,rtxz2_d,rtzz2_d,boundary_left,boundary_up,nx_size,nz,nx_append,nz_append,s_velocity_d,s_velocity1_d,s_density_d);

								//cal_gradient_in_elastic_media_new<<<dimGrid,dimBlock>>>(grad_lame11_d,grad_lame22_d,grad_den1_d,vx_t_d,vz_t_d,vx_x_d,vz_z_d,vx_z_d,vz_x_d,rvx2_d,rvz2_d,rtxx2_d,rtxz2_d,rtzz2_d,boundary_left,boundary_up,nx_size,nz,nx_append,nz_append,s_velocity_d,s_velocity1_d,s_density_d);
							}

							if(migration_type==1)
							{
								cal_gradient_in_viscoelastic_media<<<dimGrid,dimBlock>>>(grad_lame11_d,grad_lame22_d,grad_den1_d,vx_t_d,vz_t_d,vx_x_d,vz_z_d,vx_z_d,vz_x_d,rvx2_d,rvz2_d,rtxx2_d,rtxz2_d,rtzz2_d,rmem_xx2_d,rmem_xz2_d,rmem_zz2_d,boundary_left,boundary_up,nx_size,nz,nx_append,nz_append,s_velocity_d,s_velocity1_d,s_density_d,tao_d,strain_p_d,strain_s_d);
								//cal_gradient_in_viscoelastic_media_new<<<dimGrid,dimBlock>>>(grad_lame11_d,grad_lame22_d,grad_den1_d,vx_t_d,vz_t_d,vx_x_d,vz_z_d,vx_z_d,vz_x_d,rvx2_d,rvz2_d,rtxx2_d,rtxz2_d,rtzz2_d,rmem_xx2_d,rmem_xz2_d,rmem_zz2_d,boundary_left,boundary_up,nx_size,nz,nx_append,nz_append,s_velocity_d,s_velocity1_d,s_density_d,tao_d,strain_p_d,strain_s_d);
							}
						}

						else
						{
////////////////////////////////////////////Ren 2016
							/*cal_derivation_x<<<dimGrid,dimBlock>>>(vx1_d,vx_x_d,coe_opt_d,dx,dz,nx_append_radius,nz_append_radius,0);
							cal_derivation_z<<<dimGrid,dimBlock>>>(vx1_d,vx_z_d,coe_opt_d,dx,dz,nx_append_radius,nz_append_radius,1);
							cal_derivation_z<<<dimGrid,dimBlock>>>(vz1_d,vz_z_d,coe_opt_d,dx,dz,nx_append_radius,nz_append_radius,0);
							cal_derivation_x<<<dimGrid,dimBlock>>>(vz1_d,vz_x_d,coe_opt_d,dx,dz,nx_append_radius,nz_append_radius,1);*/

///////////////////////////////forward vxp vzp for x or z direction derivation vxp vzp for x or z direction derivation
							cal_derivation_x<<<dimGrid,dimBlock>>>(vxp1_d,vxp_x_d,coe_opt_d,dx,dz,nx_append_radius,nz_append_radius,0);

							cal_derivation_z<<<dimGrid,dimBlock>>>(vxp1_d,vxp_z_d,coe_opt_d,dx,dz,nx_append_radius,nz_append_radius,1);
	
							cal_derivation_z<<<dimGrid,dimBlock>>>(vzp1_d,vzp_z_d,coe_opt_d,dx,dz,nx_append_radius,nz_append_radius,0);

							cal_derivation_x<<<dimGrid,dimBlock>>>(vzp1_d,vzp_x_d,coe_opt_d,dx,dz,nx_append_radius,nz_append_radius,1);

///////////////////////////////forward vxs vzs for x or z direction derivation			
							cal_derivation_x<<<dimGrid,dimBlock>>>(vxs1_d,vxs_x_d,coe_opt_d,dx,dz,nx_append_radius,nz_append_radius,0);

							cal_derivation_z<<<dimGrid,dimBlock>>>(vxs1_d,vxs_z_d,coe_opt_d,dx,dz,nx_append_radius,nz_append_radius,1);
	
							cal_derivation_z<<<dimGrid,dimBlock>>>(vzs1_d,vzs_z_d,coe_opt_d,dx,dz,nx_append_radius,nz_append_radius,0);

							cal_derivation_x<<<dimGrid,dimBlock>>>(vzs1_d,vzs_x_d,coe_opt_d,dx,dz,nx_append_radius,nz_append_radius,1);

///////////////////////////////back vxs vzs integral   for x or z direction derivation	
							sum_integral<<<dimGrid,dimBlock>>>(rvxp_integral_d,rvxp2_d,nx_append_radius,nz_append_radius);
	
							sum_integral<<<dimGrid,dimBlock>>>(rvzp_integral_d,rvzp2_d,nx_append_radius,nz_append_radius);

							sum_integral<<<dimGrid,dimBlock>>>(rvxs_integral_d,rvxs2_d,nx_append_radius,nz_append_radius);

							sum_integral<<<dimGrid,dimBlock>>>(rvzs_integral_d,rvzs2_d,nx_append_radius,nz_append_radius);

///////////////////////////////back vxs vzs integral   for x or z direction derivation	
							cal_derivation_x<<<dimGrid,dimBlock>>>(rvxp_integral_d,rvxp_x_d,coe_opt_d,dx,dz,nx_append_radius,nz_append_radius,0);

							cal_derivation_z<<<dimGrid,dimBlock>>>(rvxp_integral_d,rvxp_z_d,coe_opt_d,dx,dz,nx_append_radius,nz_append_radius,1);
	
							cal_derivation_z<<<dimGrid,dimBlock>>>(rvzp_integral_d,rvzp_z_d,coe_opt_d,dx,dz,nx_append_radius,nz_append_radius,0);

							cal_derivation_x<<<dimGrid,dimBlock>>>(rvzp_integral_d,rvzp_x_d,coe_opt_d,dx,dz,nx_append_radius,nz_append_radius,1);

///////////////////////////////back vxs vzs integral   for x or z direction derivation	
							cal_derivation_x<<<dimGrid,dimBlock>>>(rvxs_integral_d,rvxs_x_d,coe_opt_d,dx,dz,nx_append_radius,nz_append_radius,0);

							cal_derivation_z<<<dimGrid,dimBlock>>>(rvxs_integral_d,rvxs_z_d,coe_opt_d,dx,dz,nx_append_radius,nz_append_radius,1);
	
							cal_derivation_z<<<dimGrid,dimBlock>>>(rvzs_integral_d,rvzs_z_d,coe_opt_d,dx,dz,nx_append_radius,nz_append_radius,0);

							cal_derivation_x<<<dimGrid,dimBlock>>>(rvzs_integral_d,rvzs_x_d,coe_opt_d,dx,dz,nx_append_radius,nz_append_radius,1);

							/*if(0==(it)%100&&join_wavefield==1&&iter==0)
							{
								cudaMemcpy(wf_append,rvxp_integral_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
								sprintf(filename,"./wavefield/3/vxp_integral-%d-shot_%d",ishot+1,it);
								write_file_1d(wf_append,nxanza,filename);
								exchange(wf_append,wf,nx,nz,nx_append,nz_append,boundary_left,boundary_up);
								write_file_1d(wf,nx_size_nz,filename);
							}*/
////////////////////////////////////////////Ren 2016
////cal_gradient_for_density
								if(inversion_den!=0)
								{
									cal_gradient_for_den_mul<<<dimGrid,dimBlock>>>(grad_den_pp_d,vxp_t_d,vzp_t_d,rvxp2_d,rvzp2_d,dt,boundary_left,boundary_up,nx_size,nz,nx_append,nz_append,s_velocity_d,s_velocity1_d,s_density_d);
									cal_gradient_for_den_mul<<<dimGrid,dimBlock>>>(grad_den_ps_d,vxp_t_d,vzp_t_d,rvxs2_d,rvzs2_d,dt,boundary_left,boundary_up,nx_size,nz,nx_append,nz_append,s_velocity_d,s_velocity1_d,s_density_d);
									cal_gradient_for_den_mul<<<dimGrid,dimBlock>>>(grad_den_sp_d,vxs_t_d,vzs_t_d,rvxp2_d,rvzp2_d,dt,boundary_left,boundary_up,nx_size,nz,nx_append,nz_append,s_velocity_d,s_velocity1_d,s_density_d);
									cal_gradient_for_den_mul<<<dimGrid,dimBlock>>>(grad_den_ss_d,vxs_t_d,vzs_t_d,rvxs2_d,rvzs2_d,dt,boundary_left,boundary_up,nx_size,nz,nx_append,nz_append,s_velocity_d,s_velocity1_d,s_density_d);
								}
////cal_gradient_for_lame1
									//cal_gradient_for_lame1_mul<<<dimGrid,dimBlock>>>(grad_lame11_d,rtxx2_d,rtzz2_d,vx_x_d,vz_z_d,boundary_left,boundary_up,nx_size,nz,nx_append,nz_append,s_velocity_d,s_velocity1_d,s_density_d);
									cal_gradient_for_lame1_mul<<<dimGrid,dimBlock>>>(grad_lame1_pp_d,rvxp_x_d,rvzp_z_d,vxp_x_d,vzp_z_d,boundary_left,boundary_up,nx_size,nz,nx_append,nz_append,s_velocity_d,s_velocity1_d,s_density_d);

									cal_gradient_for_lame1_mul<<<dimGrid,dimBlock>>>(grad_lame1_sp_d,rvxp_x_d,rvzp_z_d,vxs_x_d,vzs_z_d,boundary_left,boundary_up,nx_size,nz,nx_append,nz_append,s_velocity_d,s_velocity1_d,s_density_d);

									cal_gradient_for_lame1_mul<<<dimGrid,dimBlock>>>(grad_lame1_ps_d,rvxs_x_d,rvzs_z_d,vxp_x_d,vzp_z_d,boundary_left,boundary_up,nx_size,nz,nx_append,nz_append,s_velocity_d,s_velocity1_d,s_density_d);

									cal_gradient_for_lame1_mul<<<dimGrid,dimBlock>>>(grad_lame1_ss_d,rvxs_x_d,rvzs_z_d,vxs_x_d,vzs_z_d,boundary_left,boundary_up,nx_size,nz,nx_append,nz_append,s_velocity_d,s_velocity1_d,s_density_d);

////cal_gradient_for_lame2
									cal_sum_a_b_to_c<<<dimGrid,dimBlock>>>(rvxp_z_d,rvzp_x_d,wf_append_d,nx_append,nz_append);
									//cal_gradient_for_lame2_mul<<<dimGrid,dimBlock>>>(grad_lame22_d,rtxx2_d,rtxz2_d,rtzz2_d,vx_x_d,vz_z_d,vx_z_d,vz_x_d,boundary_left,boundary_up,nx_size,nz,nx_append,nz_append,s_velocity_d,s_velocity1_d,s_density_d);
									cal_gradient_for_lame2_mul<<<dimGrid,dimBlock>>>(grad_lame2_pp_d,rvxp_x_d,wf_append_d,rvzp_z_d,vxp_x_d,vzp_z_d,vxp_z_d,vzp_x_d,boundary_left,boundary_up,nx_size,nz,nx_append,nz_append,s_velocity_d,s_velocity1_d,s_density_d);

									cal_gradient_for_lame2_mul<<<dimGrid,dimBlock>>>(grad_lame2_sp_d,rvxp_x_d,wf_append_d,rvzp_z_d,vxs_x_d,vzs_z_d,vxs_z_d,vzs_x_d,boundary_left,boundary_up,nx_size,nz,nx_append,nz_append,s_velocity_d,s_velocity1_d,s_density_d);


									cal_sum_a_b_to_c<<<dimGrid,dimBlock>>>(rvxs_z_d,rvzs_x_d,wf_append_d,nx_append,nz_append);
									//cal_gradient_for_lame2_mul<<<dimGrid,dimBlock>>>(grad_lame22_d,rtxx2_d,rtxz2_d,rtzz2_d,vx_x_d,vz_z_d,vx_z_d,vz_x_d,boundary_left,boundary_up,nx_size,nz,nx_append,nz_append,s_velocity_d,s_velocity1_d,s_density_d);
									cal_gradient_for_lame2_mul<<<dimGrid,dimBlock>>>(grad_lame2_ps_d,rvxs_x_d,wf_append_d,rvzs_z_d,vxp_x_d,vzp_z_d,vxp_z_d,vzp_x_d,boundary_left,boundary_up,nx_size,nz,nx_append,nz_append,s_velocity_d,s_velocity1_d,s_density_d);

									cal_gradient_for_lame2_mul<<<dimGrid,dimBlock>>>(grad_lame2_ss_d,rvxs_x_d,wf_append_d,rvzs_z_d,vxs_x_d,vzs_z_d,vxs_z_d,vzs_x_d,boundary_left,boundary_up,nx_size,nz,nx_append,nz_append,s_velocity_d,s_velocity1_d,s_density_d);
						}
				}
//////////////////////////////////illumination
							if(iter==0)
							{	
//////////////////////////////////////////////////////////////2018年05月24日 星期四 20时18分34秒  compensate PP or  PS  new
								cuda_ex_com_pp_ps_sign<<<dimGrid,dimBlock>>>(ex_com_pp_sign_d,ex_open_pp1_d,nx_size,nz,nx_append,nz_append,0);
								cuda_ex_com_pp_ps_sign<<<dimGrid,dimBlock>>>(ex_com_ps_sign_d,ex_open_ps1_d,nx_size,nz,nx_append,nz_append,1);
								if(ishot==0)
								{
									sprintf(filename,"./someoutput/cut-ex-pp-sign_%d",ishot+1);
									cudaMemcpy(wf_append,ex_com_pp_sign_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
									exchange(wf_append,wf,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
									write_file_1d(wf,nx_size*nz,filename);

									sprintf(filename,"./someoutput/cut-ex-ps-sign_%d",ishot+1);
									cudaMemcpy(wf_append,ex_com_ps_sign_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
									exchange(wf_append,wf,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
									write_file_1d(wf,nx_size*nz,filename);
								}

/////////////////////////////////////////////////////////////////smooth pp sign								
								cuda_bell_smooth_2d<<<dimGrid_new,dimBlock>>>(ex_com_pp_sign_d,wf_append_d,20,nx_append,nz_append);	
								cudaMemcpy(ex_com_pp_sign_d,wf_append_d,nxanza*sizeof(float),cudaMemcpyDeviceToDevice);
/////////////////////////////////////////////////////////////////smooth ps sign	
								cuda_bell_smooth_2d<<<dimGrid_new,dimBlock>>>(ex_com_ps_sign_d,wf_append_d,20,nx_append,nz_append);	
								cudaMemcpy(ex_com_ps_sign_d,wf_append_d,nxanza*sizeof(float),cudaMemcpyDeviceToDevice);

								if(ishot==0)
								{
									sprintf(filename,"./someoutput/cut-ex-pp-sign-smooth_%d",ishot+1);
									cudaMemcpy(wf_append,ex_com_pp_sign_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
									exchange(wf_append,wf,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
									write_file_1d(wf,nx_size*nz,filename);

									sprintf(filename,"./someoutput/cut-ex-ps-sign-smooth_%d",ishot+1);
									cudaMemcpy(wf_append,ex_com_ps_sign_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
									exchange(wf_append,wf,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
									write_file_1d(wf,nx_size*nz,filename);
								}

								imaging_pp_compensate_dependent_angle_2D_new<<<dimGrid,dimBlock>>>(ex_open_pp1_d,ex_com_pp_sign_d,com_ex_vresultpp_d,ex_vresultpp_d,ex_time_d,nx_size,nz,nz_append,boundary_up,boundary_left);

								imaging_ps_compensate_dependent_angle_2D_new<<<dimGrid,dimBlock>>>(ex_open_ps1_d,ex_com_ps_sign_d,com_ex_vresultps_d,ex_vresultps_d,ex_time_d,nx_size,nz,nz_append,boundary_up,boundary_left);

//////////////////////////////////////////////////////////////2018年05月24日 星期四 20时18分34秒  compensate PP or  PS  new
					
								cuda_scale_gradient_acqusition_only_RTM<<<dimGrid,dimBlock>>>(result_tp_d,down_tp_d,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up,precon,precon_z1,precon_z2,1.0,offset_left[ishot],offset_right[ishot],receiver_offset,offset_attenuation);

								cuda_scale_gradient_acqusition_only_RTM<<<dimGrid,dimBlock>>>(resultpp_d,down_pp_d,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up,precon,precon_z1,precon_z2,1.0,offset_left[ishot],offset_right[ishot],receiver_offset,offset_attenuation);
								cuda_scale_gradient_acqusition_only_RTM<<<dimGrid,dimBlock>>>(resultps_d,down_pp_d,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up,precon,precon_z1,precon_z2,1.0,offset_left[ishot],offset_right[ishot],receiver_offset,offset_attenuation);
								cuda_scale_gradient_acqusition_only_RTM<<<dimGrid,dimBlock>>>(resultsp_d,down_ss_d,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up,precon,precon_z1,precon_z2,1.0,offset_left[ishot],offset_right[ishot],receiver_offset,offset_attenuation);
								cuda_scale_gradient_acqusition_only_RTM<<<dimGrid,dimBlock>>>(resultss_d,down_ss_d,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up,precon,precon_z1,precon_z2,1.0,offset_left[ishot],offset_right[ishot],receiver_offset,offset_attenuation);

								cuda_scale_gradient_acqusition_only_RTM<<<dimGrid,dimBlock>>>(resultps1_d,down_pp_d,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up,precon,precon_z1,precon_z2,1.0,offset_left[ishot],offset_right[ishot],receiver_offset,offset_attenuation);
								cuda_scale_gradient_acqusition_only_RTM<<<dimGrid,dimBlock>>>(resultps2_d,down_pp_d,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up,precon,precon_z1,precon_z2,1.0,offset_left[ishot],offset_right[ishot],receiver_offset,offset_attenuation);
								cuda_scale_gradient_acqusition_only_RTM<<<dimGrid,dimBlock>>>(resultsp1_d,down_ss_d,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up,precon,precon_z1,precon_z2,1.0,offset_left[ishot],offset_right[ishot],receiver_offset,offset_attenuation);
								cuda_scale_gradient_acqusition_only_RTM<<<dimGrid,dimBlock>>>(resultsp2_d,down_ss_d,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up,precon,precon_z1,precon_z2,1.0,offset_left[ishot],offset_right[ishot],receiver_offset,offset_attenuation);

								cuda_scale_gradient_acqusition_only_RTM<<<dimGrid,dimBlock>>>(vresultpp_d,down_vpp_d,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up,precon,precon_z1,precon_z2,1.0,offset_left[ishot],offset_right[ishot],receiver_offset,offset_attenuation);
								cuda_scale_gradient_acqusition_only_RTM<<<dimGrid,dimBlock>>>(vresultps_d,down_vpp_d,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up,precon,precon_z1,precon_z2,1.0,offset_left[ishot],offset_right[ishot],receiver_offset,offset_attenuation);
								cuda_scale_gradient_acqusition_only_RTM<<<dimGrid,dimBlock>>>(vresultsp_d,down_vss_d,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up,precon,precon_z1,precon_z2,1.0,offset_left[ishot],offset_right[ishot],receiver_offset,offset_attenuation);
								cuda_scale_gradient_acqusition_only_RTM<<<dimGrid,dimBlock>>>(vresultss_d,down_vss_d,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up,precon,precon_z1,precon_z2,1.0,offset_left[ishot],offset_right[ishot],receiver_offset,offset_attenuation);

								cuda_scale_gradient_acqusition_only_RTM<<<dimGrid,dimBlock>>>(vresultppx_d,down_vpp_x_d,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up,precon,precon_z1,precon_z2,1.0,offset_left[ishot],offset_right[ishot],receiver_offset,offset_attenuation);
								cuda_scale_gradient_acqusition_only_RTM<<<dimGrid,dimBlock>>>(vresultpsx_d,down_vpp_x_d,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up,precon,precon_z1,precon_z2,1.0,offset_left[ishot],offset_right[ishot],receiver_offset,offset_attenuation);
								cuda_scale_gradient_acqusition_only_RTM<<<dimGrid,dimBlock>>>(vresultspx_d,down_vss_x_d,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up,precon,precon_z1,precon_z2,1.0,offset_left[ishot],offset_right[ishot],receiver_offset,offset_attenuation);
								cuda_scale_gradient_acqusition_only_RTM<<<dimGrid,dimBlock>>>(vresultssx_d,down_vss_x_d,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up,precon,precon_z1,precon_z2,1.0,offset_left[ishot],offset_right[ishot],receiver_offset,offset_attenuation);

								cuda_scale_gradient_acqusition_only_RTM<<<dimGrid,dimBlock>>>(vresultppz_d,down_vpp_z_d,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up,precon,precon_z1,precon_z2,1.0,offset_left[ishot],offset_right[ishot],receiver_offset,offset_attenuation);
								cuda_scale_gradient_acqusition_only_RTM<<<dimGrid,dimBlock>>>(vresultpsz_d,down_vpp_z_d,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up,precon,precon_z1,precon_z2,1.0,offset_left[ishot],offset_right[ishot],receiver_offset,offset_attenuation);
								cuda_scale_gradient_acqusition_only_RTM<<<dimGrid,dimBlock>>>(vresultspz_d,down_vss_z_d,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up,precon,precon_z1,precon_z2,1.0,offset_left[ishot],offset_right[ishot],receiver_offset,offset_attenuation);
								cuda_scale_gradient_acqusition_only_RTM<<<dimGrid,dimBlock>>>(vresultssz_d,down_vss_z_d,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up,precon,precon_z1,precon_z2,1.0,offset_left[ishot],offset_right[ishot],receiver_offset,offset_attenuation);
//////////////////////////////////////////////////////////////////////excitation amplitude imaging condition  attenuation only
								cuda_scale_gradient_acqusition_only_RTM_ex_amp<<<dimGrid,dimBlock>>>(ex_vresultpp_d,down_vpp_d,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up,precon,precon_z1,precon_z2,1.0,offset_left[ishot],offset_right[ishot],receiver_offset,offset_attenuation);
								cuda_scale_gradient_acqusition_only_RTM_ex_amp<<<dimGrid,dimBlock>>>(ex_vresultps_d,down_vpp_d,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up,precon,precon_z1,precon_z2,1.0,offset_left[ishot],offset_right[ishot],receiver_offset,offset_attenuation);
								cuda_scale_gradient_acqusition_only_RTM_ex_amp<<<dimGrid,dimBlock>>>(ex_result_tp_d,down_vpp_d,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up,precon,precon_z1,precon_z2,1.0,offset_left[ishot],offset_right[ishot],receiver_offset,offset_attenuation);
								cuda_scale_gradient_acqusition_only_RTM_ex_amp<<<dimGrid,dimBlock>>>(ex_result_tp_old_d,down_vpp_d,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up,precon,precon_z1,precon_z2,1.0,offset_left[ishot],offset_right[ishot],receiver_offset,offset_attenuation);

								cuda_scale_gradient_acqusition_only_RTM_ex_amp<<<dimGrid,dimBlock>>>(com_ex_vresultpp_d,down_vpp_d,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up,precon,precon_z1,precon_z2,1.0,offset_left[ishot],offset_right[ishot],receiver_offset,offset_attenuation);
								cuda_scale_gradient_acqusition_only_RTM_ex_amp<<<dimGrid,dimBlock>>>(com_ex_vresultps_d,down_vpp_d,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up,precon,precon_z1,precon_z2,1.0,offset_left[ishot],offset_right[ishot],receiver_offset,offset_attenuation);
////////////////////////////////////////////////////////////////////////////////xx or zz
								cuda_scale_gradient_acqusition_only_RTM<<<dimGrid,dimBlock>>>(resultxx_d,down_xx_d,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up,precon,precon_z1,precon_z2,1.0,offset_left[ishot],offset_right[ishot],receiver_offset,offset_attenuation);
								cuda_scale_gradient_acqusition_only_RTM<<<dimGrid,dimBlock>>>(resultzz_d,down_zz_d,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up,precon,precon_z1,precon_z2,1.0,offset_left[ishot],offset_right[ishot],receiver_offset,offset_attenuation);
//////////////////RTM
//////////////////////////////////////////////////////////////////////excitation amplitude imaging condition
								cuda_sum_new_acqusition<<<dimGrid_new,dimBlock>>>(all_ex_result_tp_d,ex_result_tp_d,nx,nz,receiver_x_cord[ishot],receiver_interval,receiver_num);
								cuda_sum_new_acqusition<<<dimGrid_new,dimBlock>>>(all_ex_result_tp_old_d,ex_result_tp_old_d,nx,nz,receiver_x_cord[ishot],receiver_interval,receiver_num);
								cuda_sum_new_acqusition<<<dimGrid_new,dimBlock>>>(all_ex_vresultpp_d,ex_vresultpp_d,nx,nz,receiver_x_cord[ishot],receiver_interval,receiver_num);
								cuda_sum_new_acqusition<<<dimGrid_new,dimBlock>>>(all_ex_vresultps_d,ex_vresultps_d,nx,nz,receiver_x_cord[ishot],receiver_interval,receiver_num);

								cuda_sum_new_acqusition<<<dimGrid_new,dimBlock>>>(all_com_ex_vresultpp_d,com_ex_vresultpp_d,nx,nz,receiver_x_cord[ishot],receiver_interval,receiver_num);
								cuda_sum_new_acqusition<<<dimGrid_new,dimBlock>>>(all_com_ex_vresultps_d,com_ex_vresultps_d,nx,nz,receiver_x_cord[ishot],receiver_interval,receiver_num);
////////////////////////////////////////////////////////////////////////////////xx or zz
								cuda_sum_new_acqusition<<<dimGrid_new,dimBlock>>>(all_resultxx_d,resultxx_d,nx,nz,receiver_x_cord[ishot],receiver_interval,receiver_num);
								cuda_sum_new_acqusition<<<dimGrid_new,dimBlock>>>(all_resultzz_d,resultzz_d,nx,nz,receiver_x_cord[ishot],receiver_interval,receiver_num);	
///////////////////////////////////inner prodcut
								cuda_sum_new_acqusition<<<dimGrid_new,dimBlock>>>(all_result_tp_d,result_tp_d,nx,nz,receiver_x_cord[ishot],receiver_interval,receiver_num);
								cuda_sum_new_acqusition<<<dimGrid_new,dimBlock>>>(all_vresultpp_d,vresultpp_d,nx,nz,receiver_x_cord[ishot],receiver_interval,receiver_num);
								cuda_sum_new_acqusition<<<dimGrid_new,dimBlock>>>(all_vresultps_d,vresultps_d,nx,nz,receiver_x_cord[ishot],receiver_interval,receiver_num);
								cuda_sum_new_acqusition<<<dimGrid_new,dimBlock>>>(all_vresultsp_d,vresultsp_d,nx,nz,receiver_x_cord[ishot],receiver_interval,receiver_num);
								cuda_sum_new_acqusition<<<dimGrid_new,dimBlock>>>(all_vresultss_d,vresultss_d,nx,nz,receiver_x_cord[ishot],receiver_interval,receiver_num);
		///////////////////////////////////inner prodcut xxxxxxxxxxxxxx
								cuda_sum_new_acqusition<<<dimGrid_new,dimBlock>>>(all_vresultppx_d,vresultppx_d,nx,nz,receiver_x_cord[ishot],receiver_interval,receiver_num);
								cuda_sum_new_acqusition<<<dimGrid_new,dimBlock>>>(all_vresultpsx_d,vresultpsx_d,nx,nz,receiver_x_cord[ishot],receiver_interval,receiver_num);
								cuda_sum_new_acqusition<<<dimGrid_new,dimBlock>>>(all_vresultspx_d,vresultspx_d,nx,nz,receiver_x_cord[ishot],receiver_interval,receiver_num);
								cuda_sum_new_acqusition<<<dimGrid_new,dimBlock>>>(all_vresultssx_d,vresultssx_d,nx,nz,receiver_x_cord[ishot],receiver_interval,receiver_num);
		///////////////////////////////////inner prodcut zzzzzzzzzzzzzzz
								cuda_sum_new_acqusition<<<dimGrid_new,dimBlock>>>(all_vresultppz_d,vresultppz_d,nx,nz,receiver_x_cord[ishot],receiver_interval,receiver_num);
								cuda_sum_new_acqusition<<<dimGrid_new,dimBlock>>>(all_vresultpsz_d,vresultpsz_d,nx,nz,receiver_x_cord[ishot],receiver_interval,receiver_num);
								cuda_sum_new_acqusition<<<dimGrid_new,dimBlock>>>(all_vresultspz_d,vresultspz_d,nx,nz,receiver_x_cord[ishot],receiver_interval,receiver_num);
								cuda_sum_new_acqusition<<<dimGrid_new,dimBlock>>>(all_vresultssz_d,vresultssz_d,nx,nz,receiver_x_cord[ishot],receiver_interval,receiver_num);
		///////////////////////////////////conventional migration 
								cuda_sum_new_acqusition<<<dimGrid_new,dimBlock>>>(all_resultpp_d,resultpp_d,nx,nz,receiver_x_cord[ishot],receiver_interval,receiver_num);
								cuda_sum_new_acqusition<<<dimGrid_new,dimBlock>>>(all_resultps_d,resultps_d,nx,nz,receiver_x_cord[ishot],receiver_interval,receiver_num);
								cuda_sum_new_acqusition<<<dimGrid_new,dimBlock>>>(all_resultsp_d,resultsp_d,nx,nz,receiver_x_cord[ishot],receiver_interval,receiver_num);
								cuda_sum_new_acqusition<<<dimGrid_new,dimBlock>>>(all_resultss_d,resultss_d,nx,nz,receiver_x_cord[ishot],receiver_interval,receiver_num);
		///////////////////////////////////corrected ps or sp imaging conditions
								cuda_sum_new_acqusition<<<dimGrid_new,dimBlock>>>(all_resultps1_d,resultps1_d,nx,nz,receiver_x_cord[ishot],receiver_interval,receiver_num);
								cuda_sum_new_acqusition<<<dimGrid_new,dimBlock>>>(all_resultsp1_d,resultsp1_d,nx,nz,receiver_x_cord[ishot],receiver_interval,receiver_num);
								cuda_sum_new_acqusition<<<dimGrid_new,dimBlock>>>(all_resultps2_d,resultps2_d,nx,nz,receiver_x_cord[ishot],receiver_interval,receiver_num);
								cuda_sum_new_acqusition<<<dimGrid_new,dimBlock>>>(all_resultsp2_d,resultsp2_d,nx,nz,receiver_x_cord[ishot],receiver_interval,receiver_num);
//////////////////RTM
							}
							
							if(ishot%10==0)
							{
								cudaMemcpy(wf,grad_lame11_d,nx_size_nz*sizeof(float),cudaMemcpyDeviceToHost);
								sprintf(filename1,"./result/RTM/grad_lame11_%d_iter_%d",ishot+1,iter+1);
								write_file_1d(wf,nx_size_nz,filename1);

								cudaMemcpy(wf,grad_lame22_d,nx_size_nz*sizeof(float),cudaMemcpyDeviceToHost);
								sprintf(filename1,"./result/RTM/grad_lame22_%d_iter_%d",ishot+1,iter+1);
								write_file_1d(wf,nx_size_nz,filename1);

								
								sprintf(filename,"./check_file/down_pp_%d",ishot+1);
								cudaMemcpy(wf_append,down_pp_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_append,nxanza,filename);

								sprintf(filename,"./check_file/down_ss_%d",ishot+1);
								cudaMemcpy(wf_append,down_ss_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_append,nxanza,filename);

								sprintf(filename,"./check_file/down_vpp_%d",ishot+1);
								cudaMemcpy(wf_append,down_vpp_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_append,nxanza,filename);

								sprintf(filename,"./check_file/down_vss_%d",ishot+1);
								cudaMemcpy(wf_append,down_vss_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_append,nxanza,filename);

								sprintf(filename,"./check_file/down_xx_%d",ishot+1);
								cudaMemcpy(wf_append,down_xx_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_append,nxanza,filename);

								sprintf(filename,"./check_file/down_zz_%d",ishot+1);
								cudaMemcpy(wf_append,down_zz_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_append,nxanza,filename);

								sprintf(filename,"./someoutput/cut-ex-open-pp_%d",ishot+1);
								cudaMemcpy(wf_append,ex_open_pp_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
								exchange(wf_append,wf,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
								write_file_1d(wf,nx_size*nz,filename);

								sprintf(filename,"./someoutput/cut-ex-open-pp1_%d",ishot+1);
								cudaMemcpy(wf_append,ex_open_pp1_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
								exchange(wf_append,wf,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
								write_file_1d(wf,nx_size*nz,filename);

								sprintf(filename,"./someoutput/cut-ex-open-ps_%d",ishot+1);
								cudaMemcpy(wf_append,ex_open_ps_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
								exchange(wf_append,wf,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
								write_file_1d(wf,nx_size*nz,filename);

								sprintf(filename,"./someoutput/cut-ex-open-ps1_%d",ishot+1);
								cudaMemcpy(wf_append,ex_open_ps1_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
								exchange(wf_append,wf,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
								write_file_1d(wf,nx_size*nz,filename);


								sprintf(filename,"./someoutput/cut-ex-angle-rpp_%d",ishot+1);
								cudaMemcpy(wf_append,ex_angle_rpp_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
								exchange(wf_append,wf,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
								write_file_1d(wf,nx_size*nz,filename);

								sprintf(filename,"./someoutput/cut-ex-angle-rps_%d",ishot+1);
								cudaMemcpy(wf_append,ex_angle_rps_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
								exchange(wf_append,wf,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
								write_file_1d(wf,nx_size*nz,filename);


								sprintf(filename,"./someoutput/cut-ex-angle-rpp1_%d",ishot+1);
								cudaMemcpy(wf_append,ex_angle_rpp1_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
								exchange(wf_append,wf,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
								write_file_1d(wf,nx_size*nz,filename);

								sprintf(filename,"./someoutput/cut-ex-angle-rps1_%d",ishot+1);
								cudaMemcpy(wf_append,ex_angle_rps1_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
								exchange(wf_append,wf,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
								write_file_1d(wf,nx_size*nz,filename);
							}

					if(precon!=0)
					{	
						if(iter==0)
						{
							sprintf(filename,"./check_file/d_illum_%d",ishot+1);
							cudaMemcpy(wf_append,d_illum,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
							write_file_1d(wf_append,nxanza,filename);
						}
			
						if(iter>0)
						{
							sprintf(filename,"./check_file/d_illum_%d",ishot+1);
							fread_file_1d(wf_append,nx_append,nz_append,filename);
							cudaMemcpy(d_illum,wf_append,nxanza*sizeof(float),cudaMemcpyHostToDevice);
						}
			
							cuda_scale_gradient_acqusition_new<<<dimGrid,dimBlock>>>(grad_lame11_d,d_illum,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up,precon,precon_z1,precon_z2,0.0000001,offset_left[ishot],offset_right[ishot],receiver_offset,offset_attenuation);
							cuda_scale_gradient_acqusition_new<<<dimGrid,dimBlock>>>(grad_lame22_d,d_illum,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up,precon,precon_z1,precon_z2,0.0000001,offset_left[ishot],offset_right[ishot],receiver_offset,offset_attenuation);
							cuda_scale_gradient_acqusition_new<<<dimGrid,dimBlock>>>(grad_den1_d,d_illum,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up,precon,precon_z1,precon_z2,0.0000001,offset_left[ishot],offset_right[ishot],receiver_offset,offset_attenuation);

						/*if(iter==0)
						{
							cuda_sum_new_acqusition_illum<<<dimGrid_new,dimBlock>>>(d_illum_new,d_illum,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,receiver_x_cord[ishot],receiver_interval,receiver_num);
						}*/
					}
						/*if(receiver_offset!=0&&offset_attenuation!=0)
						{
							cauda_zero_and_attenuation_truncation<<<dimGrid,dimBlock>>>(grad_den1_d,nx_size,nz,offset_left[ishot],offset_right[ishot],receiver_offset,offset_attenuation);
							cauda_zero_and_attenuation_truncation<<<dimGrid,dimBlock>>>(grad_lame11_d,nx_size,nz,offset_left[ishot],offset_right[ishot],receiver_offset,offset_attenuation);
							cauda_zero_and_attenuation_truncation<<<dimGrid,dimBlock>>>(grad_lame22_d,nx_size,nz,offset_left[ishot],offset_right[ishot],receiver_offset,offset_attenuation);
						}*/
////////////////////////////new acqusition//////////////////////gradient
						cuda_sum_new_acqusition<<<dimGrid_new,dimBlock>>>(all_grad_den1_d,grad_den1_d,nx,nz,receiver_x_cord[ishot],receiver_interval,receiver_num);
						cuda_sum_new_acqusition<<<dimGrid_new,dimBlock>>>(all_grad_lame11_d,grad_lame11_d,nx,nz,receiver_x_cord[ishot],receiver_interval,receiver_num);
						cuda_sum_new_acqusition<<<dimGrid_new,dimBlock>>>(all_grad_lame22_d,grad_lame22_d,nx,nz,receiver_x_cord[ishot],receiver_interval,receiver_num);
//////////////////////gradient////////////////////////////new acqusition
						////////////////////The output is used to check artifacts in surface 
						ishot++;
			}

						if(inversion_den==0)
						{
							cudaMemset(all_grad_den1_d,0,nxnz*sizeof(float));
						}

						if(iter==0)/////////////2017年03月12日 星期日 11时04分38秒    elastic RTM for PP and PS reflection
						{
								///////////////////////////////////////////////////cuda_attenuation_after_lap_new2lace
//////////////////////////////////////////////////excitation imaging condition	
								cuda_attenuation_after_lap_new2<<<dimGrid_new,dimBlock>>>(all_ex_result_tp_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
								cuda_attenuation_after_lap_new2<<<dimGrid_new,dimBlock>>>(all_ex_result_tp_old_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
								cuda_attenuation_after_lap_new2<<<dimGrid_new,dimBlock>>>(all_ex_vresultpp_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
								cuda_attenuation_after_lap_new2<<<dimGrid_new,dimBlock>>>(all_ex_vresultps_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);

								cuda_attenuation_after_lap_new2<<<dimGrid_new,dimBlock>>>(all_com_ex_vresultpp_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
								cuda_attenuation_after_lap_new2<<<dimGrid_new,dimBlock>>>(all_com_ex_vresultps_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
////////////////////////////////////////////////////////////////////////////////xx or zz
								cuda_attenuation_after_lap_new2<<<dimGrid_new,dimBlock>>>(all_resultxx_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
								cuda_attenuation_after_lap_new2<<<dimGrid_new,dimBlock>>>(all_resultzz_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
						
///////////////////////////////////////////////////vresult
								cuda_attenuation_after_lap_new2<<<dimGrid_new,dimBlock>>>(all_result_tp_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
								cuda_attenuation_after_lap_new2<<<dimGrid_new,dimBlock>>>(all_vresultpp_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
								cuda_attenuation_after_lap_new2<<<dimGrid_new,dimBlock>>>(all_vresultps_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
								cuda_attenuation_after_lap_new2<<<dimGrid_new,dimBlock>>>(all_vresultsp_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
								cuda_attenuation_after_lap_new2<<<dimGrid_new,dimBlock>>>(all_vresultss_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
								cuda_attenuation_after_lap_new2<<<dimGrid_new,dimBlock>>>(all_vresultppx_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
								cuda_attenuation_after_lap_new2<<<dimGrid_new,dimBlock>>>(all_vresultpsx_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
								cuda_attenuation_after_lap_new2<<<dimGrid_new,dimBlock>>>(all_vresultspx_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
								cuda_attenuation_after_lap_new2<<<dimGrid_new,dimBlock>>>(all_vresultssx_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
								cuda_attenuation_after_lap_new2<<<dimGrid_new,dimBlock>>>(all_vresultppz_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
								cuda_attenuation_after_lap_new2<<<dimGrid_new,dimBlock>>>(all_vresultpsz_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
								cuda_attenuation_after_lap_new2<<<dimGrid_new,dimBlock>>>(all_vresultspz_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
								cuda_attenuation_after_lap_new2<<<dimGrid_new,dimBlock>>>(all_vresultssz_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
///////////////////////////////////////////////////result
								cuda_attenuation_after_lap_new2<<<dimGrid_new,dimBlock>>>(all_resultpp_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
								cuda_attenuation_after_lap_new2<<<dimGrid_new,dimBlock>>>(all_resultps_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
								cuda_attenuation_after_lap_new2<<<dimGrid_new,dimBlock>>>(all_resultsp_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
								cuda_attenuation_after_lap_new2<<<dimGrid_new,dimBlock>>>(all_resultss_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
///////////////////////////////////////////////////correction
								cuda_attenuation_after_lap_new2<<<dimGrid_new,dimBlock>>>(all_resultps1_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
								cuda_attenuation_after_lap_new2<<<dimGrid_new,dimBlock>>>(all_resultsp1_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
								cuda_attenuation_after_lap_new2<<<dimGrid_new,dimBlock>>>(all_resultps2_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
								cuda_attenuation_after_lap_new2<<<dimGrid_new,dimBlock>>>(all_resultsp2_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);

								sprintf(filename1,"./result/RTM/initial-com-ex-vresultpp");
								cudaMemcpy(wf_nxnz,all_com_ex_vresultpp_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);

								sprintf(filename1,"./result/RTM/initial-com-ex-vresultps");
								cudaMemcpy(wf_nxnz,all_com_ex_vresultps_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);

								sprintf(filename1,"./result/RTM/initial-ex-vresultpp");
								cudaMemcpy(wf_nxnz,all_ex_vresultpp_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);

								sprintf(filename1,"./result/RTM/initial-ex-vresultps");
								cudaMemcpy(wf_nxnz,all_ex_vresultps_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);

								sprintf(filename1,"./result/RTM/initial-ex-result-tp");
								cudaMemcpy(wf_nxnz,all_ex_result_tp_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);

								sprintf(filename1,"./result/RTM/initial-ex-result-tp-old");
								cudaMemcpy(wf_nxnz,all_ex_result_tp_old_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);

								sprintf(filename1,"./result/RTM/initial-resultxx");
								cudaMemcpy(wf_nxnz,all_resultxx_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);

								sprintf(filename1,"./result/RTM/initial-resultzz");
								cudaMemcpy(wf_nxnz,all_resultzz_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);

								sprintf(filename1,"./result/RTM/initial-result-tp");
								cudaMemcpy(wf_nxnz,all_result_tp_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);								

								sprintf(filename1,"./result/RTM/initial-resultpp");
								cudaMemcpy(wf_nxnz,all_resultpp_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);

								sprintf(filename1,"./result/RTM/initial-resultps",iter+1);
								cudaMemcpy(wf_nxnz,all_resultps_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);
						
								sprintf(filename1,"./result/RTM/initial-resultps1",iter+1);
								cudaMemcpy(wf_nxnz,all_resultps1_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);

								sprintf(filename1,"./result/RTM/initial-resultps2",iter+1);
								cudaMemcpy(wf_nxnz,all_resultps2_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);

								sprintf(filename1,"./result/RTM/initial-resultsp1",iter+1);
								cudaMemcpy(wf_nxnz,all_resultsp1_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);

								sprintf(filename1,"./result/RTM/initial-resultsp2",iter+1);
								cudaMemcpy(wf_nxnz,all_resultsp2_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);

								sprintf(filename1,"./result/RTM/initial-resultss",iter+1);
								cudaMemcpy(wf_nxnz,all_resultss_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);

								sprintf(filename1,"./result/RTM/initial-vresultpp");
								cudaMemcpy(wf_nxnz,all_vresultpp_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);

								sprintf(filename1,"./result/RTM/initial-vresultps");
								cudaMemcpy(wf_nxnz,all_vresultps_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);

								sprintf(filename1,"./result/RTM/initial-vresultsp");
								cudaMemcpy(wf_nxnz,all_vresultsp_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);

								sprintf(filename1,"./result/RTM/initial-vresultss");
								cudaMemcpy(wf_nxnz,all_vresultss_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);

								sprintf(filename1,"./result/RTM/initial-vresultppx");
								cudaMemcpy(wf_nxnz,all_vresultppx_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);

								sprintf(filename1,"./result/RTM/initial-vresultpsx");
								cudaMemcpy(wf_nxnz,all_vresultpsx_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);

								sprintf(filename1,"./result/RTM/initial-vresultppz");
								cudaMemcpy(wf_nxnz,all_vresultppz_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);

								sprintf(filename1,"./result/RTM/initial-vresultpsz");
								cudaMemcpy(wf_nxnz,all_vresultpsz_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);

							//if(laplace==1)////RTM for laplace  is different LSRTM
							{
//////////////////////////////////////////////////////////excitation imaging condition
//////////////////////////////////////////////////////////ex_tp
								cudaMemcpy(wf_nxnz_d,all_ex_result_tp_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);
								cuda_laplace<<<dimGrid_new,dimBlock>>>(all_ex_result_tp_d,wf_nxnz_d,s_velocity_all_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,dx,dz,1,laplace);
								cuda_lap<<<dimGrid_new,dimBlock>>>(wf_nxnz_d,nx,nz,laplace);
								cudaMemcpy(all_ex_result_tp_d,wf_nxnz_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);
//////////////////////////////////////////////////////////ex_tp_old
								cudaMemcpy(wf_nxnz_d,all_ex_result_tp_old_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);
								cuda_laplace<<<dimGrid_new,dimBlock>>>(all_ex_result_tp_old_d,wf_nxnz_d,s_velocity_all_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,dx,dz,1,laplace);
								cuda_lap<<<dimGrid_new,dimBlock>>>(wf_nxnz_d,nx,nz,laplace);
								cudaMemcpy(all_ex_result_tp_old_d,wf_nxnz_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);

//////////////////////////////////////////////////////////ex_pp
								cudaMemcpy(wf_nxnz_d,all_ex_vresultpp_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);
								cuda_laplace<<<dimGrid_new,dimBlock>>>(all_ex_vresultpp_d,wf_nxnz_d,s_velocity_all_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,dx,dz,1,laplace);
								cuda_lap<<<dimGrid_new,dimBlock>>>(wf_nxnz_d,nx,nz,laplace);
								cudaMemcpy(all_ex_vresultpp_d,wf_nxnz_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);

//////////////////////////////////////////////////////////ex_ps						
								cudaMemcpy(wf_nxnz_d,all_ex_vresultps_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);
								cuda_laplace<<<dimGrid_new,dimBlock>>>(all_ex_vresultps_d,wf_nxnz_d,s_velocity_all_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,dx,dz,1,laplace);
								cuda_lap<<<dimGrid_new,dimBlock>>>(wf_nxnz_d,nx,nz,laplace);
								cudaMemcpy(all_ex_vresultps_d,wf_nxnz_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);

//////////////////////////////////////////////////////////ex_pp
								cudaMemcpy(wf_nxnz_d,all_com_ex_vresultpp_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);
								cuda_laplace<<<dimGrid_new,dimBlock>>>(all_com_ex_vresultpp_d,wf_nxnz_d,s_velocity_all_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,dx,dz,1,laplace);
								cuda_lap<<<dimGrid_new,dimBlock>>>(wf_nxnz_d,nx,nz,laplace);
								cudaMemcpy(all_com_ex_vresultpp_d,wf_nxnz_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);

//////////////////////////////////////////////////////////ex_ps						
								cudaMemcpy(wf_nxnz_d,all_com_ex_vresultps_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);
								cuda_laplace<<<dimGrid_new,dimBlock>>>(all_com_ex_vresultps_d,wf_nxnz_d,s_velocity_all_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,dx,dz,1,laplace);
								cuda_lap<<<dimGrid_new,dimBlock>>>(wf_nxnz_d,nx,nz,laplace);
								cudaMemcpy(all_com_ex_vresultps_d,wf_nxnz_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);
//////////////////////////////////////////////////////////xx
								cudaMemcpy(wf_nxnz_d,all_resultxx_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);
								cuda_laplace<<<dimGrid_new,dimBlock>>>(all_resultxx_d,wf_nxnz_d,s_velocity_all_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,dx,dz,1,laplace);
								cuda_lap<<<dimGrid_new,dimBlock>>>(wf_nxnz_d,nx,nz,laplace);
								cudaMemcpy(all_resultxx_d,wf_nxnz_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);
//////////////////////////////////////////////////////////zz
								cudaMemcpy(wf_nxnz_d,all_resultzz_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);
								cuda_laplace<<<dimGrid_new,dimBlock>>>(all_resultzz_d,wf_nxnz_d,s_velocity_all_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,dx,dz,1,laplace);
								cuda_lap<<<dimGrid_new,dimBlock>>>(wf_nxnz_d,nx,nz,laplace);
								cudaMemcpy(all_resultzz_d,wf_nxnz_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);
//////////////////////////////////////////////////////////tp
								cudaMemcpy(wf_nxnz_d,all_result_tp_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);
								cuda_laplace<<<dimGrid_new,dimBlock>>>(all_result_tp_d,wf_nxnz_d,s_velocity_all_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,dx,dz,1,laplace);
								cuda_lap<<<dimGrid_new,dimBlock>>>(wf_nxnz_d,nx,nz,laplace);
								cudaMemcpy(all_result_tp_d,wf_nxnz_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);

//////////////////////////////////////////////////////////pp
								cudaMemcpy(wf_nxnz_d,all_resultpp_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);
								cuda_laplace<<<dimGrid_new,dimBlock>>>(all_resultpp_d,wf_nxnz_d,s_velocity_all_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,dx,dz,1,laplace);
								cuda_lap<<<dimGrid_new,dimBlock>>>(wf_nxnz_d,nx,nz,laplace);
								cudaMemcpy(all_resultpp_d,wf_nxnz_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);

//////////////////////////////////////////////////////////ps						
								cudaMemcpy(wf_nxnz_d,all_resultps_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);
								cuda_laplace<<<dimGrid_new,dimBlock>>>(all_resultps_d,wf_nxnz_d,s_velocity_all_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,dx,dz,1,laplace);
								cuda_lap<<<dimGrid_new,dimBlock>>>(wf_nxnz_d,nx,nz,laplace);
								cudaMemcpy(all_resultps_d,wf_nxnz_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);

//////////////////////////////////////////////////////////sp						
								cudaMemcpy(wf_nxnz_d,all_resultsp_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);
								cuda_laplace<<<dimGrid_new,dimBlock>>>(all_resultsp_d,wf_nxnz_d,s_velocity_all_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,dx,dz,1,laplace);
								cuda_lap<<<dimGrid_new,dimBlock>>>(wf_nxnz_d,nx,nz,laplace);
								cudaMemcpy(all_resultsp_d,wf_nxnz_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);

//////////////////////////////////////////////////////////ss						
								cudaMemcpy(wf_nxnz_d,all_resultss_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);
								cuda_laplace<<<dimGrid_new,dimBlock>>>(all_resultss_d,wf_nxnz_d,s_velocity_all_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,dx,dz,1,laplace);
								cuda_lap<<<dimGrid_new,dimBlock>>>(wf_nxnz_d,nx,nz,laplace);
								cudaMemcpy(all_resultss_d,wf_nxnz_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);

//////////////////////////////////////////////////////////ps1								
								cudaMemcpy(wf_nxnz_d,all_resultps1_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);
								cuda_laplace<<<dimGrid_new,dimBlock>>>(all_resultps1_d,wf_nxnz_d,s_velocity_all_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,dx,dz,1,laplace);
								cuda_lap<<<dimGrid_new,dimBlock>>>(wf_nxnz_d,nx,nz,laplace);
								cudaMemcpy(all_resultps1_d,wf_nxnz_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);
//////////////////////////////////////////////////////////ps2
								cudaMemcpy(wf_nxnz_d,all_resultps2_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);
								cuda_laplace<<<dimGrid_new,dimBlock>>>(all_resultps2_d,wf_nxnz_d,s_velocity_all_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,dx,dz,1,laplace);
								cuda_lap<<<dimGrid_new,dimBlock>>>(wf_nxnz_d,nx,nz,laplace);
								cudaMemcpy(all_resultps2_d,wf_nxnz_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);
//////////////////////////////////////////////////////////sp1								
								cudaMemcpy(wf_nxnz_d,all_resultsp1_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);
								cuda_laplace<<<dimGrid_new,dimBlock>>>(all_resultsp1_d,wf_nxnz_d,s_velocity_all_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,dx,dz,1,laplace);
								cuda_lap<<<dimGrid_new,dimBlock>>>(wf_nxnz_d,nx,nz,laplace);
								cudaMemcpy(all_resultsp1_d,wf_nxnz_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);
//////////////////////////////////////////////////////////sp2								
								cudaMemcpy(wf_nxnz_d,all_resultsp2_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);
								cuda_laplace<<<dimGrid_new,dimBlock>>>(all_resultsp2_d,wf_nxnz_d,s_velocity_all_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,dx,dz,1,laplace);
								cuda_lap<<<dimGrid_new,dimBlock>>>(wf_nxnz_d,nx,nz,laplace);
								cudaMemcpy(all_resultsp2_d,wf_nxnz_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);

//////////////////////////////////////////////////////////vpp
								cudaMemcpy(wf_nxnz_d,all_vresultpp_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);
								cuda_laplace<<<dimGrid_new,dimBlock>>>(all_vresultpp_d,wf_nxnz_d,s_velocity_all_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,dx,dz,1,laplace);
								cuda_lap<<<dimGrid_new,dimBlock>>>(wf_nxnz_d,nx,nz,laplace);
								cudaMemcpy(all_vresultpp_d,wf_nxnz_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);
//////////////////////////////////////////////////////////vps						
								cudaMemcpy(wf_nxnz_d,all_vresultps_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);
								cuda_laplace<<<dimGrid_new,dimBlock>>>(all_vresultps_d,wf_nxnz_d,s_velocity_all_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,dx,dz,1,laplace);
								cuda_lap<<<dimGrid_new,dimBlock>>>(wf_nxnz_d,nx,nz,laplace);
								cudaMemcpy(all_vresultps_d,wf_nxnz_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);
//////////////////////////////////////////////////////////vsp								
								cudaMemcpy(wf_nxnz_d,all_vresultsp_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);
								cuda_laplace<<<dimGrid_new,dimBlock>>>(all_vresultsp_d,wf_nxnz_d,s_velocity_all_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,dx,dz,1,laplace);
								cuda_lap<<<dimGrid_new,dimBlock>>>(wf_nxnz_d,nx,nz,laplace);
								cudaMemcpy(all_vresultsp_d,wf_nxnz_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);
//////////////////////////////////////////////////////////vss
								cudaMemcpy(wf_nxnz_d,all_vresultss_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);
								cuda_laplace<<<dimGrid_new,dimBlock>>>(all_vresultss_d,wf_nxnz_d,s_velocity_all_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,dx,dz,1,laplace);
								cuda_lap<<<dimGrid_new,dimBlock>>>(wf_nxnz_d,nx,nz,laplace);
								cudaMemcpy(all_vresultss_d,wf_nxnz_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);

//////////////////////////////////////////////////////////vppx
								cudaMemcpy(wf_nxnz_d,all_vresultppx_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);
								cuda_laplace<<<dimGrid_new,dimBlock>>>(all_vresultppx_d,wf_nxnz_d,s_velocity_all_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,dx,dz,1,laplace);
								cuda_lap<<<dimGrid_new,dimBlock>>>(wf_nxnz_d,nx,nz,laplace);
								cudaMemcpy(all_vresultppx_d,wf_nxnz_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);
//////////////////////////////////////////////////////////vpsx						
								cudaMemcpy(wf_nxnz_d,all_vresultpsx_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);
								cuda_laplace<<<dimGrid_new,dimBlock>>>(all_vresultpsx_d,wf_nxnz_d,s_velocity_all_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,dx,dz,1,laplace);
								cuda_lap<<<dimGrid_new,dimBlock>>>(wf_nxnz_d,nx,nz,laplace);
								cudaMemcpy(all_vresultpsx_d,wf_nxnz_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);
//////////////////////////////////////////////////////////vspx								
								cudaMemcpy(wf_nxnz_d,all_vresultspx_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);
								cuda_laplace<<<dimGrid_new,dimBlock>>>(all_vresultspx_d,wf_nxnz_d,s_velocity_all_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,dx,dz,1,laplace);
								cuda_lap<<<dimGrid_new,dimBlock>>>(wf_nxnz_d,nx,nz,laplace);
								cudaMemcpy(all_vresultspx_d,wf_nxnz_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);
//////////////////////////////////////////////////////////vssx
								cudaMemcpy(wf_nxnz_d,all_vresultssx_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);
								cuda_laplace<<<dimGrid_new,dimBlock>>>(all_vresultssx_d,wf_nxnz_d,s_velocity_all_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,dx,dz,1,laplace);
								cuda_lap<<<dimGrid_new,dimBlock>>>(wf_nxnz_d,nx,nz,laplace);
								cudaMemcpy(all_vresultssx_d,wf_nxnz_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);

//////////////////////////////////////////////////////////vppz
								cudaMemcpy(wf_nxnz_d,all_vresultppz_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);
								cuda_laplace<<<dimGrid_new,dimBlock>>>(all_vresultppz_d,wf_nxnz_d,s_velocity_all_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,dx,dz,1,laplace);
								cuda_lap<<<dimGrid_new,dimBlock>>>(wf_nxnz_d,nx,nz,laplace);
								cudaMemcpy(all_vresultppz_d,wf_nxnz_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);
//////////////////////////////////////////////////////////vpsz						
								cudaMemcpy(wf_nxnz_d,all_vresultpsz_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);
								cuda_laplace<<<dimGrid_new,dimBlock>>>(all_vresultpsz_d,wf_nxnz_d,s_velocity_all_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,dx,dz,1,laplace);
								cuda_lap<<<dimGrid_new,dimBlock>>>(wf_nxnz_d,nx,nz,laplace);
								cudaMemcpy(all_vresultpsz_d,wf_nxnz_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);
//////////////////////////////////////////////////////////vspz								
								cudaMemcpy(wf_nxnz_d,all_vresultspz_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);
								cuda_laplace<<<dimGrid_new,dimBlock>>>(all_vresultspz_d,wf_nxnz_d,s_velocity_all_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,dx,dz,1,laplace);
								cuda_lap<<<dimGrid_new,dimBlock>>>(wf_nxnz_d,nx,nz,laplace);
								cudaMemcpy(all_vresultspz_d,wf_nxnz_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);
//////////////////////////////////////////////////////////vssz
								cudaMemcpy(wf_nxnz_d,all_vresultssz_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);
								cuda_laplace<<<dimGrid_new,dimBlock>>>(all_vresultssz_d,wf_nxnz_d,s_velocity_all_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,dx,dz,1,laplace);
								cuda_lap<<<dimGrid_new,dimBlock>>>(wf_nxnz_d,nx,nz,laplace);
								cudaMemcpy(all_vresultssz_d,wf_nxnz_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);
							}

///////////////////////////////////////////////////cuda_attenuation_after_lap_new2lace
//////////////////////////////////////////////////excitation imaging condition	
								cuda_attenuation_after_lap_new2<<<dimGrid_new,dimBlock>>>(all_ex_result_tp_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
								cuda_attenuation_after_lap_new2<<<dimGrid_new,dimBlock>>>(all_ex_result_tp_old_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
								cuda_attenuation_after_lap_new2<<<dimGrid_new,dimBlock>>>(all_ex_vresultpp_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
								cuda_attenuation_after_lap_new2<<<dimGrid_new,dimBlock>>>(all_ex_vresultps_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
								cuda_attenuation_after_lap_new2<<<dimGrid_new,dimBlock>>>(all_resultxx_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
								cuda_attenuation_after_lap_new2<<<dimGrid_new,dimBlock>>>(all_resultzz_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
						
///////////////////////////////////////////////////vresult
								cuda_attenuation_after_lap_new2<<<dimGrid_new,dimBlock>>>(all_result_tp_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
								cuda_attenuation_after_lap_new2<<<dimGrid_new,dimBlock>>>(all_vresultpp_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
								cuda_attenuation_after_lap_new2<<<dimGrid_new,dimBlock>>>(all_vresultps_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
								cuda_attenuation_after_lap_new2<<<dimGrid_new,dimBlock>>>(all_vresultsp_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
								cuda_attenuation_after_lap_new2<<<dimGrid_new,dimBlock>>>(all_vresultss_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
								cuda_attenuation_after_lap_new2<<<dimGrid_new,dimBlock>>>(all_vresultppx_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
								cuda_attenuation_after_lap_new2<<<dimGrid_new,dimBlock>>>(all_vresultpsx_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
								cuda_attenuation_after_lap_new2<<<dimGrid_new,dimBlock>>>(all_vresultspx_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
								cuda_attenuation_after_lap_new2<<<dimGrid_new,dimBlock>>>(all_vresultssx_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
								cuda_attenuation_after_lap_new2<<<dimGrid_new,dimBlock>>>(all_vresultppz_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
								cuda_attenuation_after_lap_new2<<<dimGrid_new,dimBlock>>>(all_vresultpsz_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
								cuda_attenuation_after_lap_new2<<<dimGrid_new,dimBlock>>>(all_vresultspz_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
								cuda_attenuation_after_lap_new2<<<dimGrid_new,dimBlock>>>(all_vresultssz_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
///////////////////////////////////////////////////result
								cuda_attenuation_after_lap_new2<<<dimGrid_new,dimBlock>>>(all_resultpp_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
								cuda_attenuation_after_lap_new2<<<dimGrid_new,dimBlock>>>(all_resultps_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
								cuda_attenuation_after_lap_new2<<<dimGrid_new,dimBlock>>>(all_resultsp_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
								cuda_attenuation_after_lap_new2<<<dimGrid_new,dimBlock>>>(all_resultss_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
///////////////////////////////////////////////////correction
								cuda_attenuation_after_lap_new2<<<dimGrid_new,dimBlock>>>(all_resultps1_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
								cuda_attenuation_after_lap_new2<<<dimGrid_new,dimBlock>>>(all_resultsp1_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
								cuda_attenuation_after_lap_new2<<<dimGrid_new,dimBlock>>>(all_resultps2_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
								cuda_attenuation_after_lap_new2<<<dimGrid_new,dimBlock>>>(all_resultsp2_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);					

/////////////////////excitation amplitude imaging condition
/////////////////////////////////////////ex_tp
								sprintf(filename1,"./result/RTM/ex-result-tp-lap");
								cudaMemcpy(wf_nxnz,all_ex_result_tp_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);
							
								sprintf(filename1,"./result/RTM/ex-result-tp-old-lap");
								cudaMemcpy(wf_nxnz,all_ex_result_tp_old_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);

								sprintf(filename1,"./result/RTM/ex-vresultpp-lap");
								cudaMemcpy(wf_nxnz,all_ex_vresultpp_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);

								sprintf(filename1,"./result/RTM/ex-vresultps-lap");
								cudaMemcpy(wf_nxnz,all_ex_vresultps_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);

								sprintf(filename1,"./result/RTM/com-ex-vresultpp-lap");
								cudaMemcpy(wf_nxnz,all_com_ex_vresultpp_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);

								sprintf(filename1,"./result/RTM/com-ex-vresultps-lap");
								cudaMemcpy(wf_nxnz,all_com_ex_vresultps_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);
///////////////////////////////////////////xx
								sprintf(filename1,"./result/RTM/resultxx-lap");
								cudaMemcpy(wf_nxnz,all_resultxx_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);
///////////////////////////////////////////zz
								sprintf(filename1,"./result/RTM/resultzz-lap");
								cudaMemcpy(wf_nxnz,all_resultzz_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);
/////////////////////////////////////////tp					
								sprintf(filename1,"./result/RTM/result-tp-lap");
								cudaMemcpy(wf_nxnz,all_result_tp_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);										
////////////////tradtional method
								sprintf(filename1,"./result/RTM/resultpp-lap");
								cudaMemcpy(wf_nxnz,all_resultpp_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);

								sprintf(filename1,"./result/RTM/resultps-lap",iter+1);
								cudaMemcpy(wf_nxnz,all_resultps_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);

								sprintf(filename1,"./result/RTM/resultsp-lap",iter+1);
								cudaMemcpy(wf_nxnz,all_resultsp_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);

								sprintf(filename1,"./result/RTM/resultss-lap",iter+1);
								cudaMemcpy(wf_nxnz,all_resultss_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);

								sprintf(filename1,"./result/RTM/resultps1-lap",iter+1);
								cudaMemcpy(wf_nxnz,all_resultps1_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);

								sprintf(filename1,"./result/RTM/resultps2-lap",iter+1);
								cudaMemcpy(wf_nxnz,all_resultps2_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);

								sprintf(filename1,"./result/RTM/resultsp1-lap",iter+1);
								cudaMemcpy(wf_nxnz,all_resultsp1_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);

								sprintf(filename1,"./result/RTM/resultsp2-lap",iter+1);
								cudaMemcpy(wf_nxnz,all_resultsp2_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);
////////////////inner method
								sprintf(filename1,"./result/RTM/vresultpp-lap");
								cudaMemcpy(wf_nxnz,all_vresultpp_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);

								sprintf(filename1,"./result/RTM/vresultps-lap",iter+1);
								cudaMemcpy(wf_nxnz,all_vresultps_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);

								sprintf(filename1,"./result/RTM/vresultsp-lap",iter+1);
								cudaMemcpy(wf_nxnz,all_vresultsp_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);

								sprintf(filename1,"./result/RTM/vresultss-lap",iter+1);
								cudaMemcpy(wf_nxnz,all_vresultss_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);
////////////////correlation method
								sprintf(filename1,"./result/RTM/vresultppx-lap");
								cudaMemcpy(wf_nxnz,all_vresultppx_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);

								sprintf(filename1,"./result/RTM/vresultpsx-lap",iter+1);
								cudaMemcpy(wf_nxnz,all_vresultpsx_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);

								sprintf(filename1,"./result/RTM/vresultspx-lap",iter+1);
								cudaMemcpy(wf_nxnz,all_vresultspx_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);

								sprintf(filename1,"./result/RTM/vresultssx-lap",iter+1);
								cudaMemcpy(wf_nxnz,all_vresultssx_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);
////////////////correlation method
								sprintf(filename1,"./result/RTM/vresultppz-lap");
								cudaMemcpy(wf_nxnz,all_vresultppz_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);

								sprintf(filename1,"./result/RTM/vresultpsz-lap",iter+1);
								cudaMemcpy(wf_nxnz,all_vresultpsz_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);

								sprintf(filename1,"./result/RTM/vresultspz-lap",iter+1);
								cudaMemcpy(wf_nxnz,all_vresultspz_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);

								sprintf(filename1,"./result/RTM/vresultssz-lap",iter+1);
								cudaMemcpy(wf_nxnz,all_vresultssz_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);		
						}

///////////////////////////all  mode:PP PS SP and SS
						if(decomposition!=0)
						{
///////////////////////////one  mode:PP 
								sprintf(filename1,"./result/gradient/grad_lame1_pp_iter_%d",iter+1);
								cudaMemcpy(wf_nxnz,all_grad_lame1_pp_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);

								sprintf(filename1,"./result/gradient/grad_lame2_pp_iter_%d",iter+1);
								cudaMemcpy(wf_nxnz,all_grad_lame2_pp_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);

								sprintf(filename1,"./result/gradient/grad_den_pp_iter_%d",iter+1);
								cudaMemcpy(wf_nxnz,all_grad_den_pp_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);
///////////////////////////one  mode:SS 
								sprintf(filename1,"./result/gradient/grad_lame1_ss_iter_%d",iter+1);
								cudaMemcpy(wf_nxnz,all_grad_lame1_ss_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);

								sprintf(filename1,"./result/gradient/grad_lame2_ss_iter_%d",iter+1);
								cudaMemcpy(wf_nxnz,all_grad_lame2_ss_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);

								sprintf(filename1,"./result/gradient/grad_den_ss_iter_%d",iter+1);
								cudaMemcpy(wf_nxnz,all_grad_den_ss_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);
///////////////////////////one  mode:PS
								sprintf(filename1,"./result/gradient/grad_lame1_ps_iter_%d",iter+1);
								cudaMemcpy(wf_nxnz,all_grad_lame1_ps_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);

								sprintf(filename1,"./result/gradient/grad_lame2_ps_iter_%d",iter+1);
								cudaMemcpy(wf_nxnz,all_grad_lame2_ps_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);

								sprintf(filename1,"./result/gradient/grad_den_ps_iter_%d",iter+1);
								cudaMemcpy(wf_nxnz,all_grad_den_ps_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);
//////////////////////////one  mode:SP
								sprintf(filename1,"./result/gradient/grad_lame1_sp_iter_%d",iter+1);
								cudaMemcpy(wf_nxnz,all_grad_lame1_sp_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);

								sprintf(filename1,"./result/gradient/grad_lame2_sp_iter_%d",iter+1);
								cudaMemcpy(wf_nxnz,all_grad_lame2_sp_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);

								sprintf(filename1,"./result/gradient/grad_den_sp_iter_%d",iter+1);
								cudaMemcpy(wf_nxnz,all_grad_den_sp_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);
						}

////////////////////////////////////output gradient				
							sprintf(filename1,"./result/gradient/grad_lame1_iter_%d",iter+1);
							cudaMemcpy(wf_nxnz,all_grad_lame11_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
							write_file_1d(wf_nxnz,nxnz,filename1);

							sprintf(filename1,"./result/gradient/grad_lame2_iter_%d",iter+1);
							cudaMemcpy(wf_nxnz,all_grad_lame22_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
							write_file_1d(wf_nxnz,nxnz,filename1);

							sprintf(filename1,"./result/gradient/grad_den_iter_%d",iter+1);
							cudaMemcpy(wf_nxnz,all_grad_den1_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
							write_file_1d(wf_nxnz,nxnz,filename1);

/////////////////////////precondition scale
						/* compute the gradient of FWI by scaling, precondition incorporated here: equations 9 and 10 */
						if(inversion_para==0||inversion_para==1)
						{

							if(laplace==1)
							{
								cudaMemcpy(wf_nxnz_d,all_grad_lame11_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);
								cuda_laplace<<<dimGrid_new,dimBlock>>>(all_grad_lame11_d,wf_nxnz_d,s_velocity_all_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,dx,dz,1,laplace);
								cuda_lap<<<dimGrid_new,dimBlock>>>(wf_nxnz_d,nx,nz,laplace);
								cudaMemcpy(all_grad_lame11_d,wf_nxnz_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);
//////////////////////////////////////////////////////////////////////////////////////////////						
								cudaMemcpy(wf_nxnz_d,all_grad_lame22_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);
								cuda_laplace<<<dimGrid_new,dimBlock>>>(all_grad_lame22_d,wf_nxnz_d,s_velocity_all_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,dx,dz,1,laplace);
								cuda_lap<<<dimGrid_new,dimBlock>>>(wf_nxnz_d,nx,nz,laplace);
								cudaMemcpy(all_grad_lame22_d,wf_nxnz_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);
//////////////////////////////////////////////////////////////////////////////////////////////							
								cudaMemcpy(wf_nxnz_d,all_grad_den1_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);
								cuda_laplace<<<dimGrid_new,dimBlock>>>(all_grad_den1_d,wf_nxnz_d,s_velocity_all_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,dx,dz,1,laplace);
								cuda_lap<<<dimGrid_new,dimBlock>>>(wf_nxnz_d,nx,nz,laplace);
								cudaMemcpy(all_grad_den1_d,wf_nxnz_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);
							}
///////////////////////////////////////////////////grad_lame
								cuda_attenuation_after_lap_new2<<<dimGrid_new,dimBlock>>>(all_grad_lame11_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
								cuda_attenuation_after_lap_new2<<<dimGrid_new,dimBlock>>>(all_grad_lame22_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
								cuda_attenuation_after_lap_new2<<<dimGrid_new,dimBlock>>>(all_grad_den1_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
								sprintf(filename1,"./result/gradient/grad_lame111_iter_%d",iter+1);
								cudaMemcpy(wf_nxnz,all_grad_lame11_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);

								sprintf(filename1,"./result/gradient/grad_lame222_iter_%d",iter+1);
								cudaMemcpy(wf_nxnz,all_grad_lame22_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);

								sprintf(filename1,"./result/gradient/grad_den11_iter_%d",iter+1);
								cudaMemcpy(wf_nxnz,all_grad_den1_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);							
						}

////////////////////////////////////////////invert lame gradient to velocity  gradient and density gradient
						if(inversion_para==2)
						{			
							if(decomposition==0)
							{
								//invert_lame_to_velocity_para_new<<<dimGrid_new,dimBlock>>>(all_grad_vp1_d,all_grad_vs1_d,all_grad_density1_d,all_grad_lame11_d,all_grad_lame22_d,all_grad_den1_d,s_velocity_all_d,s_velocity1_all_d,s_density_all_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up);

								invert_lame_to_velocity_para<<<dimGrid_new,dimBlock>>>(all_grad_vp1_d,all_grad_vs1_d,all_grad_density1_d,all_grad_lame11_d,all_grad_lame22_d,all_grad_den1_d,s_velocity_all_d,s_velocity1_all_d,s_density_all_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up);
							}

							if(decomposition!=0)
							{
								invert_lame_to_velocity_vp<<<dimGrid_new,dimBlock>>>(all_grad_vp1_d,all_grad_lame1_pp_d,all_grad_lame2_pp_d,all_grad_den_pp_d,s_velocity_all_d,s_velocity1_all_d,s_density_all_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up);

								invert_lame_to_velocity_vs<<<dimGrid_new,dimBlock>>>(all_grad_vs1_d,all_grad_lame1_ps_d,all_grad_lame2_ps_d,all_grad_den_ps_d,s_velocity_all_d,s_velocity1_all_d,s_density_all_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up);

								invert_lame_to_velocity_density<<<dimGrid_new,dimBlock>>>(all_grad_density1_d,all_grad_lame11_d,all_grad_lame22_d,all_grad_den1_d,s_velocity_all_d,s_velocity1_all_d,s_density_all_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up);
							}

								sprintf(filename1,"./result/gradient/grad_vp_iter_%d",iter+1);
								cudaMemcpy(wf_nxnz,all_grad_vp1_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);

								sprintf(filename1,"./result/gradient/grad_vs_iter_%d",iter+1);
								cudaMemcpy(wf_nxnz,all_grad_vs1_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);

								sprintf(filename1,"./result/gradient/grad_density_iter_%d",iter+1);
								cudaMemcpy(wf_nxnz,all_grad_density1_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);
						}

						if(inversion_para==3)
						{									
							if(decomposition==0)
							{
								//invert_lame_to_impedance_para_new<<<dimGrid_new,dimBlock>>>(all_grad_vp1_d,all_grad_vs1_d,all_grad_density1_d,all_grad_lame11_d,all_grad_lame22_d,all_grad_den1_d,s_velocity_all_d,s_velocity1_all_d,s_density_all_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up);

								invert_lame_to_impedance_para<<<dimGrid_new,dimBlock>>>(all_grad_vp1_d,all_grad_vs1_d,all_grad_density1_d,all_grad_lame11_d,all_grad_lame22_d,all_grad_den1_d,s_velocity_all_d,s_velocity1_all_d,s_density_all_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up);
							}

							if(decomposition!=0)
							{
								invert_lame_to_impedance_vp<<<dimGrid_new,dimBlock>>>(all_grad_vp1_d,all_grad_lame1_pp_d,all_grad_lame2_pp_d,all_grad_den_pp_d,s_velocity_all_d,s_velocity1_all_d,s_density_all_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up);

								invert_lame_to_impedance_vs<<<dimGrid_new,dimBlock>>>(all_grad_vs1_d,all_grad_lame1_ps_d,all_grad_lame2_ps_d,all_grad_den_ps_d,s_velocity_all_d,s_velocity1_all_d,s_density_all_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up);

								invert_lame_to_impedance_density<<<dimGrid_new,dimBlock>>>(all_grad_density1_d,all_grad_lame11_d,all_grad_lame22_d,all_grad_den1_d,s_velocity_all_d,s_velocity1_all_d,s_density_all_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up);
							}

								sprintf(filename1,"./result/gradient/grad_vp_iter_%d",iter+1);
								cudaMemcpy(wf_nxnz,all_grad_vp1_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);

								sprintf(filename1,"./result/gradient/grad_vs_iter_%d",iter+1);
								cudaMemcpy(wf_nxnz,all_grad_vs1_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);

								sprintf(filename1,"./result/gradient/grad_density_iter_%d",iter+1);
								cudaMemcpy(wf_nxnz,all_grad_density1_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);			
						}			
////////////////////////////////////////////invert lame gradient to velocity gradient and density gradient
	
						if(inversion_para==2||inversion_para==3)
						{
							//////////////////////////////////////////////vsp_precondition
							if(vsp_precon==1)
							{
								cuda_scale_gradient_new<<<dimGrid_new,dimBlock>>>(all_grad_vp1_d,r_d_illum,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,vsp_precon);
								cuda_scale_gradient_new<<<dimGrid_new,dimBlock>>>(all_grad_vs1_d,r_d_illum,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,vsp_precon);
								cuda_scale_gradient_new<<<dimGrid_new,dimBlock>>>(all_grad_density1_d,r_d_illum,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,vsp_precon);			
							}
	
							if(laplace==1)
							{
								cudaMemcpy(wf_nxnz_d,all_grad_vp1_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);
								cuda_laplace<<<dimGrid_new,dimBlock>>>(all_grad_vp1_d,wf_nxnz_d,s_velocity_all_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,dx,dz,1,laplace);
								cuda_lap<<<dimGrid_new,dimBlock>>>(wf_nxnz_d,nx,nz,laplace);
								cudaMemcpy(all_grad_vp1_d,wf_nxnz_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);
//////////////////////////////////////////////////////////////////////////////////////////////							
								cudaMemcpy(wf_nxnz_d,all_grad_vs1_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);
								cuda_laplace<<<dimGrid_new,dimBlock>>>(all_grad_vs1_d,wf_nxnz_d,s_velocity_all_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,dx,dz,1,laplace);
								cuda_lap<<<dimGrid_new,dimBlock>>>(wf_nxnz_d,nx,nz,laplace);
								cudaMemcpy(all_grad_vs1_d,wf_nxnz_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);
//////////////////////////////////////////////////////////////////////////////////////////////							
								cudaMemcpy(wf_nxnz_d,all_grad_density1_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);
								cuda_laplace<<<dimGrid_new,dimBlock>>>(all_grad_density1_d,wf_nxnz_d,s_velocity_all_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,dx,dz,1,laplace);
								cuda_lap<<<dimGrid_new,dimBlock>>>(wf_nxnz_d,nx,nz,laplace);
								cudaMemcpy(all_grad_density1_d,wf_nxnz_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);
							}
///////////////////////////////////////////////////grad_vp vs density
								cuda_attenuation_after_lap_new2<<<dimGrid_new,dimBlock>>>(all_grad_vp1_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
								cuda_attenuation_after_lap_new2<<<dimGrid_new,dimBlock>>>(all_grad_vs1_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
								cuda_attenuation_after_lap_new2<<<dimGrid_new,dimBlock>>>(all_grad_density1_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);

								sprintf(filename1,"./result/gradient/grad_vp2_iter_%d",iter+1);
								cudaMemcpy(wf_nxnz,all_grad_vp1_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);

								sprintf(filename1,"./result/gradient/grad_vs2_iter_%d",iter+1);
								cudaMemcpy(wf_nxnz,all_grad_vs1_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);

								sprintf(filename1,"./result/gradient/grad_density2_iter_%d",iter+1);
								cudaMemcpy(wf_nxnz,all_grad_density1_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);

								/*smooth_acqusition<<<dimGrid_new,dimBlock>>>(all_grad_vp1_d,nx,nz,offset_left_d,offset_right_d,source_x_cord_d,shot_num);
								smooth_acqusition<<<dimGrid_new,dimBlock>>>(all_grad_vs1_d,nx,nz,offset_left_d,offset_right_d,source_x_cord_d,shot_num);
								smooth_acqusition<<<dimGrid_new,dimBlock>>>(all_grad_density1_d,nx,nz,offset_left_d,offset_right_d,source_x_cord_d,shot_num);
								sprintf(filename1,"./result/gradient/grad_vp3_iter_%d",iter+1);
								cudaMemcpy(wf_nxnz,all_grad_vp1_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);

								sprintf(filename1,"./result/gradient/grad_vs3_iter_%d",iter+1);
								cudaMemcpy(wf_nxnz,all_grad_vs1_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);

								sprintf(filename1,"./result/gradient/grad_density3_iter_%d",iter+1);
								cudaMemcpy(wf_nxnz,all_grad_density1_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
								write_file_1d(wf_nxnz,nxnz,filename1);*/								
						}


////////////////////YPL's conjugate method
						if(inversion_para==0||inversion_para==1)//////////iter_start!=0:restart
						{
							////new conjugate method
							calcualte_hydrid_grad<<<dimGrid_new,dimBlock>>>(all_hydrid_grad2_d,all_grad_lame11_d,nx,nz,0);

							calcualte_hydrid_grad<<<dimGrid_new,dimBlock>>>(all_hydrid_grad2_d,all_grad_lame22_d,nx,nz,1);

							if(inversion_den!=0)
							{
								calcualte_hydrid_grad<<<dimGrid_new,dimBlock>>>(all_hydrid_grad2_d,all_grad_den1_d,nx,nz,2);
							}

							if(iter>0&&iter_start!=iter)//////////iter_start!=0:restart
							{								
								cuda_cal_beta_new<<<1, Block_Size>>>(beta_d,all_hydrid_grad1_d,all_hydrid_grad2_d,all_hydrid_conj_d,3*nxnz,0);
							}

							cuda_cal_conjgrad_new<<<dimGrid_3nx_nz,dimBlock>>>(all_hydrid_grad2_d,all_hydrid_conj_d,beta_d,3*nx,nz,0);
							cuda_cal_conjgrad_new<<<dimGrid_new,dimBlock>>>(all_grad_lame11_d,all_conj_lame1_d,beta_d,nx,nz,0);
							cuda_cal_conjgrad_new<<<dimGrid_new,dimBlock>>>(all_grad_lame22_d,all_conj_lame2_d,beta_d,nx,nz,0);
							cuda_cal_conjgrad_new<<<dimGrid_new,dimBlock>>>(all_grad_den1_d,all_conj_den_d,beta_d,nx,nz,0);

							sprintf(filename1,"./result/gradient/conj_lame1_iter_%d",iter+1);
							cudaMemcpy(wf_nxnz,all_conj_lame1_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
							write_file_1d(wf_nxnz,nxnz,filename1);

							sprintf(filename1,"./result/gradient/conj_lame2_iter_%d",iter+1);
							cudaMemcpy(wf_nxnz,all_conj_lame2_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
							write_file_1d(wf_nxnz,nxnz,filename1);

							sprintf(filename1,"./result/gradient/conj_den_iter_%d",iter+1);
							cudaMemcpy(wf_nxnz,all_conj_den_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
							write_file_1d(wf_nxnz,nxnz,filename1);
						}
					
						if(inversion_para==2||inversion_para==3)//////////iter_start!=0:restart 
						{	
							////new conjugate method
							calcualte_hydrid_grad<<<dimGrid_new,dimBlock>>>(all_hydrid_grad2_d,all_grad_vp1_d,nx,nz,0);

							calcualte_hydrid_grad<<<dimGrid_new,dimBlock>>>(all_hydrid_grad2_d,all_grad_vs1_d,nx,nz,1);

							if(inversion_den!=0)
							{
								calcualte_hydrid_grad<<<dimGrid_new,dimBlock>>>(all_hydrid_grad2_d,all_grad_density1_d,nx,nz,2);
							}
	
							if (iter>0&&iter_start!=iter)  
							{
								cuda_cal_beta_new<<<1, Block_Size>>>(beta_d,all_hydrid_grad1_d,all_hydrid_grad2_d,all_hydrid_conj_d,3*nxnz,0);
							}
						
							cuda_cal_conjgrad_new<<<dimGrid_3nx_nz,dimBlock>>>(all_hydrid_grad2_d,all_hydrid_conj_d,beta_d,3*nx,nz,0);
							cuda_cal_conjgrad_new<<<dimGrid_new,dimBlock>>>(all_grad_vp1_d,all_conj_vp_d,beta_d,nx,nz,0);
							cuda_cal_conjgrad_new<<<dimGrid_new,dimBlock>>>(all_grad_vs1_d,all_conj_vs_d,beta_d,nx,nz,0);
							cuda_cal_conjgrad_new<<<dimGrid_new,dimBlock>>>(all_grad_density1_d,all_conj_density_d,beta_d,nx,nz,0);

							sprintf(filename1,"./result/gradient/conj_vp_iter_%d",iter+1);
							cudaMemcpy(wf_nxnz,all_conj_vp_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
							write_file_1d(wf_nxnz,nxnz,filename1);

							sprintf(filename1,"./result/gradient/conj_vs_iter_%d",iter+1);
							cudaMemcpy(wf_nxnz,all_conj_vs_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
							write_file_1d(wf_nxnz,nxnz,filename1);

							sprintf(filename1,"./result/gradient/conj_density_iter_%d",iter+1);
							cudaMemcpy(wf_nxnz,all_conj_density_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
							write_file_1d(wf_nxnz,nxnz,filename1);	
						}
					
////////////////////////////////////////////2017年09月11日 星期一 12时05分48秒   it is important for set zero
						if(inversion_den==0)
						{
							cudaMemset(all_conj_den_d,0,nxnz*sizeof(float));
							cudaMemset(all_conj_density_d,0,nxnz*sizeof(float));
							warn("density parameter is no inversion\n");
						}
////////////////////YPL's conjugate method

//直接拿扰动结果正演得到地震记录，对于正演为线性（如：Born线性正演），可以这样计算！！！，但是非线性必须给定扰动（如正常的正演模拟）（给定微小扰动，计算lame1的最优步长：），意义是一样！！！/////////this process refer to Claerbout	YPL in 2015
				if(laplace_compensate!=0)/////////////////////////real wavelet
				{
					cudaMemcpy(wavelet_d,wavelet,wavelet_length*sizeof(float),cudaMemcpyHostToDevice);
				}
				ishot=0;
				cudaMemset(d_alpha1, 0, lt_rec*sizeof(float));
				cudaMemset(d_alpha2, 0, lt_rec*sizeof(float));
				cudaMemset(correlation_parameter_d,0,10*sizeof(float));///////important
				while(ishot<shot_num)
				{
					if(cut_direct_wave==1)
					{
/////////////////////////////////////one_born_modeling
/////////////////////////////////////get smooth vp;				
						cuda_get_partly_mode_boundary_z1_z2<<<dimGrid_new,dimBlock>>>(s_velocity_all_d,wf_d,nx,nz,receiver_x_cord[ishot],receiver_interval,receiver_num,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
						cuda_cal_expand<<<dimGrid,dimBlock>>>(s_velocity_d,wf_d,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);

						/*cudaMemcpy(wf,wf_d,nx_size_nz*sizeof(float),cudaMemcpyDeviceToHost);
						sprintf(filename,"./someoutput/cut1-vp-%d.bin",ishot+1);
						write_file_1d(wf,nx_size_nz,filename);

						cudaMemcpy(wf_append,s_velocity_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
						sprintf(filename,"./someoutput/vp1-%d.bin",ishot+1);
						write_file_1d(wf_append,nxanza,filename);*/

/////////////////////////////////////get smooth vs;				
						cuda_get_partly_mode_boundary_z1_z2<<<dimGrid_new,dimBlock>>>(s_velocity1_all_d,wf_d,nx,nz,receiver_x_cord[ishot],receiver_interval,receiver_num,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
						cuda_cal_expand<<<dimGrid,dimBlock>>>(s_velocity1_d,wf_d,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);

/////////////////////////////////////get smooth density;				
						cuda_get_partly_mode_boundary_z1_z2<<<dimGrid_new,dimBlock>>>(s_density_all_d,wf_d,nx,nz,receiver_x_cord[ishot],receiver_interval,receiver_num,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
						cuda_cal_expand<<<dimGrid,dimBlock>>>(s_density_d,wf_d,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);

/////////////////////////////////////get smooth qp;				
						cuda_get_partly_mode_boundary_z1_z2<<<dimGrid_new,dimBlock>>>(s_qp_all_d,wf_d,nx,nz,receiver_x_cord[ishot],receiver_interval,receiver_num,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
						cuda_cal_expand<<<dimGrid,dimBlock>>>(s_qp_d,wf_d,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);

/////////////////////////////////////get smooth qs;				
						cuda_get_partly_mode_boundary_z1_z2<<<dimGrid_new,dimBlock>>>(s_qs_all_d,wf_d,nx,nz,receiver_x_cord[ishot],receiver_interval,receiver_num,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
						cuda_cal_expand<<<dimGrid,dimBlock>>>(s_qs_d,wf_d,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);

						/*if((receiver_offset!=0)||(offset_left[ishot]>receiver_offset)||(offset_right[ishot]>receiver_offset))
						{								
							cuda_expand_acqusition_left_and_right<<<dimGrid_new,dimBlock>>>(s_velocity_d,offset_left[ishot],offset_right[ishot],source_x_cord[ishot],receiver_offset,nx_size,nx_append,nz_append,boundary_left,boundary_up);

							cuda_expand_acqusition_left_and_right<<<dimGrid_new,dimBlock>>>(s_velocity1_d,offset_left[ishot],offset_right[ishot],source_x_cord[ishot],receiver_offset,nx_size,nx_append,nz_append,boundary_left,boundary_up);

							cuda_expand_acqusition_left_and_right<<<dimGrid_new,dimBlock>>>(s_density_d,offset_left[ishot],offset_right[ishot],source_x_cord[ishot],receiver_offset,nx_size,nx_append,nz_append,boundary_left,boundary_up);

							cuda_expand_acqusition_left_and_right<<<dimGrid_new,dimBlock>>>(s_qp_d,offset_left[ishot],offset_right[ishot],source_x_cord[ishot],receiver_offset,nx_size,nx_append,nz_append,boundary_left,boundary_up);

							cuda_expand_acqusition_left_and_right<<<dimGrid_new,dimBlock>>>(s_qs_d,offset_left[ishot],offset_right[ishot],source_x_cord[ishot],receiver_offset,nx_size,nx_append,nz_append,boundary_left,boundary_up);						
						}*/

/////////////////////////////////QQQQQQQQQQQQQQQ2017年07月27日 星期四 10时11分40秒 
						cuda_cal_viscoelastic<<< dimGrid,dimBlock>>>(modul_p_d,modul_s_d,s_qp_d,s_qs_d,tao_d,strain_p_d,strain_s_d,freq,s_velocity_d,s_velocity1_d,s_density_d,nx_append,nz_append);
						
						if(inversion_para==0||inversion_para==1)
						{
/////////////////////////////////////get perturbed lame1;
							cuda_get_partly_mode_z1_z2<<<dimGrid_new,dimBlock>>>(all_conj_lame1_d,wf_d,nx,nz,receiver_x_cord[ishot],receiver_interval,receiver_num,precon_z1,precon_z2);
							cuda_cal_expand<<<dimGrid,dimBlock>>>(tmp_perturb_lame1_d,wf_d,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
/////////////////////////////////////get perturbed lame2;
							cuda_get_partly_mode_z1_z2<<<dimGrid_new,dimBlock>>>(all_conj_lame2_d,wf_d,nx,nz,receiver_x_cord[ishot],receiver_interval,receiver_num,precon_z1,precon_z2);
							cuda_cal_expand<<<dimGrid,dimBlock>>>(tmp_perturb_lame2_d,wf_d,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
/////////////////////////////////////get perturbed lame2;
							cuda_get_partly_mode_z1_z2<<<dimGrid_new,dimBlock>>>(all_conj_den_d,wf_d,nx,nz,receiver_x_cord[ishot],receiver_interval,receiver_num,precon_z1,precon_z2);
							cuda_cal_expand<<<dimGrid,dimBlock>>>(tmp_perturb_den_d,wf_d,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
							/*if((receiver_offset!=0)||(offset_left[ishot]>receiver_offset)||(offset_right[ishot]>receiver_offset))
							{								
								cuda_expand_acqusition_left_and_right<<<dimGrid_new,dimBlock>>>(tmp_perturb_lame1_d,offset_left[ishot],offset_right[ishot],source_x_cord[ishot],receiver_offset,nx_size,nx_append,nz_append,boundary_left,boundary_up);

								cuda_expand_acqusition_left_and_right<<<dimGrid_new,dimBlock>>>(tmp_perturb_lame2_d,offset_left[ishot],offset_right[ishot],source_x_cord[ishot],receiver_offset,nx_size,nx_append,nz_append,boundary_left,boundary_up);

								cuda_expand_acqusition_left_and_right<<<dimGrid_new,dimBlock>>>(tmp_perturb_den_d,offset_left[ishot],offset_right[ishot],source_x_cord[ishot],receiver_offset,nx_size,nx_append,nz_append,boundary_left,boundary_up);
							}*/
						}

						if(inversion_para==2||inversion_para==3)
						{
/////////////////////////////////////get perturbed lame1;
							cuda_get_partly_mode_z1_z2<<<dimGrid_new,dimBlock>>>(all_conj_vp_d,wf_d,nx,nz,receiver_x_cord[ishot],receiver_interval,receiver_num,precon_z1,precon_z2);
							cuda_cal_expand<<<dimGrid,dimBlock>>>(tmp_perturb_vp_d,wf_d,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
							/*cudaMemcpy(wf,wf_d,nx_size_nz*sizeof(float),cudaMemcpyDeviceToHost);
							sprintf(filename,"./someoutput/cut-perturb-vp-%d.bin",ishot+1);
							write_file_1d(wf,nx_size_nz,filename);

							cudaMemcpy(wf_append,tmp_perturb_vp_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
							sprintf(filename,"./someoutput/perturb-vp-%d.bin",ishot+1);
							write_file_1d(wf_append,nxanza,filename);*/
/////////////////////////////////////get perturbed lame2;
							cuda_get_partly_mode_z1_z2<<<dimGrid_new,dimBlock>>>(all_conj_vs_d,wf_d,nx,nz,receiver_x_cord[ishot],receiver_interval,receiver_num,precon_z1,precon_z2);
							cuda_cal_expand<<<dimGrid,dimBlock>>>(tmp_perturb_vs_d,wf_d,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
/////////////////////////////////////get perturbed lame2;
							cuda_get_partly_mode_z1_z2<<<dimGrid_new,dimBlock>>>(all_conj_density_d,wf_d,nx,nz,receiver_x_cord[ishot],receiver_interval,receiver_num,precon_z1,precon_z2);
							cuda_cal_expand<<<dimGrid,dimBlock>>>(tmp_perturb_density_d,wf_d,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);

							/*if((receiver_offset!=0)||(offset_left[ishot]>receiver_offset)||(offset_right[ishot]>receiver_offset))
							{								
								cuda_expand_acqusition_left_and_right<<<dimGrid_new,dimBlock>>>(tmp_perturb_vp_d,offset_left[ishot],offset_right[ishot],source_x_cord[ishot],receiver_offset,nx_size,nx_append,nz_append,boundary_left,boundary_up);

								cuda_expand_acqusition_left_and_right<<<dimGrid_new,dimBlock>>>(tmp_perturb_vs_d,offset_left[ishot],offset_right[ishot],source_x_cord[ishot],receiver_offset,nx_size,nx_append,nz_append,boundary_left,boundary_up);

								cuda_expand_acqusition_left_and_right<<<dimGrid_new,dimBlock>>>(tmp_perturb_density_d,offset_left[ishot],offset_right[ishot],source_x_cord[ishot],receiver_offset,nx_size,nx_append,nz_append,boundary_left,boundary_up);
							}*/
						}

						memset((void *)(wf_append),0,nxanza*sizeof(float));

						cudaMemset(vx1_d,0,nxanza*sizeof(float));
						cudaMemset(vz1_d,0,nxanza*sizeof(float));
						cudaMemset(txx1_d,0,nxanza*sizeof(float));
						cudaMemset(tzz1_d,0,nxanza*sizeof(float));
						cudaMemset(txz1_d,0,nxanza*sizeof(float));

						cudaMemset(vx2_d,0,nxanza*sizeof(float));
						cudaMemset(vz2_d,0,nxanza*sizeof(float));
						cudaMemset(txx2_d,0,nxanza*sizeof(float));
						cudaMemset(tzz2_d,0,nxanza*sizeof(float));
						cudaMemset(txz2_d,0,nxanza*sizeof(float));
		
						cudaMemset(tp2_d,0,nxanza*sizeof(float));
						cudaMemset(tp1_d,0,nxanza*sizeof(float));
						cudaMemset(vxp2_d,0,nxanza*sizeof(float));
						cudaMemset(vxp1_d,0,nxanza*sizeof(float));
						cudaMemset(vzp2_d,0,nxanza*sizeof(float));
						cudaMemset(vzp1_d,0,nxanza*sizeof(float));
						cudaMemset(vxs2_d,0,nxanza*sizeof(float));
						cudaMemset(vxs1_d,0,nxanza*sizeof(float));
						cudaMemset(vzs2_d,0,nxanza*sizeof(float));
						cudaMemset(vzs1_d,0,nxanza*sizeof(float));

						cudaMemset(mem_p1_d,0,nxanza*sizeof(float));
						cudaMemset(mem_xx1_d,0,nxanza*sizeof(float));
						cudaMemset(mem_zz1_d,0,nxanza*sizeof(float));
						cudaMemset(mem_xz1_d,0,nxanza*sizeof(float));
						cudaMemset(mem_p2_d,0,nxanza*sizeof(float));
						cudaMemset(mem_xx2_d,0,nxanza*sizeof(float));
						cudaMemset(mem_zz2_d,0,nxanza*sizeof(float));
						cudaMemset(mem_xz2_d,0,nxanza*sizeof(float));

						cudaMemset(rvx1_d,0,nxanza*sizeof(float));
						cudaMemset(rvz1_d,0,nxanza*sizeof(float));
						cudaMemset(rtxx1_d,0,nxanza*sizeof(float));
						cudaMemset(rtzz1_d,0,nxanza*sizeof(float));
						cudaMemset(rtxz1_d,0,nxanza*sizeof(float));

						cudaMemset(rvx2_d,0,nxanza*sizeof(float));
						cudaMemset(rvz2_d,0,nxanza*sizeof(float));
						cudaMemset(rtxx2_d,0,nxanza*sizeof(float));
						cudaMemset(rtzz2_d,0,nxanza*sizeof(float));
						cudaMemset(rtxz2_d,0,nxanza*sizeof(float));

						cudaMemset(rtp2_d,0,nxanza*sizeof(float));
						cudaMemset(rtp1_d,0,nxanza*sizeof(float));
						cudaMemset(rvxp2_d,0,nxanza*sizeof(float));
						cudaMemset(rvxp1_d,0,nxanza*sizeof(float));
						cudaMemset(rvzp2_d,0,nxanza*sizeof(float));
						cudaMemset(rvzp1_d,0,nxanza*sizeof(float));
						cudaMemset(rvxs2_d,0,nxanza*sizeof(float));
						cudaMemset(rvxs1_d,0,nxanza*sizeof(float));
						cudaMemset(rvzs2_d,0,nxanza*sizeof(float));
						cudaMemset(rvzs1_d,0,nxanza*sizeof(float));

						cudaMemset(rmem_p1_d,0,nxanza*sizeof(float));
						cudaMemset(rmem_xx1_d,0,nxanza*sizeof(float));
						cudaMemset(rmem_zz1_d,0,nxanza*sizeof(float));
						cudaMemset(rmem_xz1_d,0,nxanza*sizeof(float));
						cudaMemset(rmem_p2_d,0,nxanza*sizeof(float));
						cudaMemset(rmem_xx2_d,0,nxanza*sizeof(float));
						cudaMemset(rmem_zz2_d,0,nxanza*sizeof(float));
						cudaMemset(rmem_xz2_d,0,nxanza*sizeof(float));

						cudaMemset(vx_x_d,0,nxanza*sizeof(float));
						cudaMemset(vx_z_d,0,nxanza*sizeof(float));
						cudaMemset(vz_x_d,0,nxanza*sizeof(float));
						cudaMemset(vz_z_d,0,nxanza*sizeof(float));

						cudaMemset(vx_t_d,0,nxanza*sizeof(float));
						cudaMemset(vz_t_d,0,nxanza*sizeof(float));

				for(int it=0;it<lt+wavelet_half;it++)
						{
							//if(fmod((it+1.0)-wavelet_half,1000.0)==0) warn("shot=%d,step=forward 2,it=%d",ishot+1,(it+1)-wavelet_half);
							if(it<wavelet_length)
							{
								//add_source<<<1,1>>>(txx1_d,tzz1_d,wavelet_d,source_x_cord[ishot],shot_depth,it,boundary_up,boundary_left,nz_append);
								//add_source<<<1,1>>>(txx1_d,tzz1_d,wavelet_d,source_x_cord[ishot],source_z_cord[ishot],it,boundary_up,boundary_left,nz_append);///for vsp 2017年03月14日 星期二 08时55分59秒 
								add_source<<<1,1>>>(txx1_d,tzz1_d,wavelet_d,source_x_cord[ishot]-receiver_x_cord[ishot],source_z_cord[ishot],it,boundary_up,boundary_left,nz_append);///for vsp 2017年03月14日 星期二 08时55分59秒 
							}	
								fwd_vx_new<<<dimGrid,dimBlock>>>(vx_t_d,vx2_d,vx1_d,txx1_d,txz1_d,attenuation_d,dx,dz,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,s_density_d);

								fwd_vz_new<<<dimGrid,dimBlock>>>(vz_t_d,vz2_d,vz1_d,tzz1_d,txz1_d,attenuation_d,dx,dz,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,s_density_d);	

								if(migration_type==0)	fwd_txxzzxzpp_new<<<dimGrid,dimBlock>>>(vx_x_d,vx_z_d,vz_x_d,vz_z_d,tp2_d,tp1_d,txx2_d,txx1_d,tzz2_d,tzz1_d,txz2_d,txz1_d,vx2_d,vz2_d,s_velocity_d,s_velocity1_d,attenuation_d,dt,dx,dz,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,s_density_d);

								else	fwd_txxzzxzpp_viscoelastic_and_memory_3parameterization_new<<<dimGrid,dimBlock>>>(vx_x_d,vx_z_d,vz_x_d,vz_z_d,tp2_d,tp1_d,txx2_d,txx1_d,tzz2_d,tzz1_d,txz2_d,txz1_d,vx2_d,vz2_d,modul_p_d,modul_s_d,attenuation_d,s_density_d,mem_p2_d,mem_p1_d,mem_xx2_d,mem_xx1_d,mem_zz2_d,mem_zz1_d,mem_xz2_d,mem_xz1_d,s_velocity_d,s_velocity1_d,tao_d,strain_p_d,strain_s_d,packaging_d);

								if(0==(it)%100&&join_wavefield==1&&iter==0)
								{
									cudaMemcpy(wf_append,vx2_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
									sprintf(filename,"./wavefield1/4/vx-%d-shot_%d",ishot+1,it);
									write_file_1d(wf_append,nxanza,filename);
									//exchange(wf_append,wf,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
									//write_file_1d(wf,nx_size_nz,filename);
											
									cudaMemcpy(wf_append,vz2_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
									sprintf(filename,"./wavefield1/4/vz-%d-shot_%d",ishot+1,it);
									write_file_1d(wf_append,nxanza,filename);
									//exchange(wf_append,wf,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
									//write_file_1d(wf,nx_size_nz,filename);

									cudaMemcpy(wf_append,vz_z_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
									sprintf(filename,"./wavefield1/4/vz-z-%d-shot_%d",ishot+1,it);
									write_file_1d(wf_append,nxanza,filename);
									//exchange(wf_append,wf,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
									//write_file_1d(wf,nx_size_nz,filename);

									cudaMemcpy(wf_append,vz_x_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
									sprintf(filename,"./wavefield1/4/vz-x-%d-shot_%d",ishot+1,it);
									write_file_1d(wf_append,nxanza,filename);
									//exchange(wf_append,wf,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
									//write_file_1d(wf,nx_size_nz,filename);

									cudaMemcpy(wf_append,vz_t_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
									sprintf(filename,"./wavefield1/4/vz-t-%d-shot_%d",ishot+1,it);
									write_file_1d(wf_append,nxanza,filename);
									//exchange(wf_append,wf,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
									//write_file_1d(wf,nx_size_nz,filename);
								}

							rep=vx1_d;vx1_d=vx2_d;vx2_d=rep;
							rep=vz1_d;vz1_d=vz2_d;vz2_d=rep;
							rep=txx1_d;txx1_d=txx2_d;txx2_d=rep;
							rep=tzz1_d;tzz1_d=tzz2_d;tzz2_d=rep;
							rep=txz1_d;txz1_d=txz2_d;txz2_d=rep;

							rep=tp1_d;tp1_d=tp2_d;tp2_d=rep;
							rep=vxp1_d;vxp1_d=vxp2_d;vxp2_d=rep;
							rep=vzp1_d;vzp1_d=vzp2_d;vzp2_d=rep;
							rep=vxs1_d;vxs1_d=vxs2_d;vxs2_d=rep;
							rep=vzs1_d;vzs1_d=vzs2_d;vzs2_d=rep;

							rep=mem_p1_d;mem_p1_d=mem_p2_d;mem_p2_d=rep;
							rep=mem_xx1_d;mem_xx1_d=mem_xx2_d;mem_xx2_d=rep;
							rep=mem_zz1_d;mem_zz1_d=mem_zz2_d;mem_zz2_d=rep;
							rep=mem_xz1_d;mem_xz1_d=mem_xz2_d;mem_xz2_d=rep;
	///////////////////////demigration to calculate cal_shots!!!!!!!!!!
							if(migration_type==0)
							{	
								cuda_cal_dem_parameter_elastic_media<<<dimGrid,dimBlock>>>(dem_p1_d,dem_p2_d,dem_p3_d,dem_p4_d,dem_p5_d,vx_x_d,vx_z_d,vz_x_d,vz_z_d,vx_t_d,vz_t_d,tmp_perturb_lame1_d,tmp_perturb_lame2_d,tmp_perturb_den_d,tmp_perturb_vp_d,tmp_perturb_vs_d,tmp_perturb_density_d,nx_append_radius,nz_append_radius,s_velocity_d,s_velocity1_d,s_density_d,inversion_para);

								demig_fwd_txxzzxz_mul<<<dimGrid,dimBlock>>>(rtxx2_d,rtxx1_d,rtzz2_d,rtzz1_d,rtxz2_d,rtxz1_d,rvx1_d,rvz1_d,s_velocity_d,s_velocity1_d,attenuation_d,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,s_density_d,dem_p1_d,dem_p2_d,dem_p3_d,dem_p4_d,dem_p5_d);

								demig_fwd_vx_mul<<<dimGrid,dimBlock>>>(rvx2_d,rvx1_d,rtxx2_d,rtxz2_d,attenuation_d,dx,dz,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,s_density_d,dem_p1_d,dem_p2_d,dem_p3_d,dem_p4_d,dem_p5_d);

								demig_fwd_vz_mul<<<dimGrid,dimBlock>>>(rvz2_d,rvz1_d,rtzz2_d,rtxz2_d,attenuation_d,dx,dz,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,s_density_d,dem_p1_d,dem_p2_d,dem_p3_d,dem_p4_d,dem_p5_d);			
							}

							if(migration_type==1)
							{
								cuda_cal_dem_parameter_viscoelastic_media_new<<<dimGrid,dimBlock>>>(dem_p1_d,dem_p2_d,dem_p_all_d,vx_x_d,vx_z_d,vz_x_d,vz_z_d,vx_t_d,vz_t_d,tmp_perturb_lame1_d,tmp_perturb_lame2_d,tmp_perturb_den_d,tmp_perturb_vp_d,tmp_perturb_vs_d,tmp_perturb_density_d,nx_append_radius,nz_append_radius,s_velocity_d,s_velocity1_d,s_density_d,tao_d,strain_p_d,strain_s_d,dt,inversion_para);

								demig_fwd_txxzzxzpp_viscoelastic_and_memory_3parameterization<<<dimGrid,dimBlock>>>(rtp2_d,rtp1_d,rtxx2_d,rtxx1_d,rtzz2_d,rtzz1_d,rtxz2_d,rtxz1_d,rvx1_d,rvz1_d,modul_p_d,modul_s_d,attenuation_d,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,s_density_d,rmem_p2_d,rmem_p1_d,rmem_xx2_d,rmem_xx1_d,rmem_zz2_d,rmem_zz1_d,rmem_xz2_d,rmem_xz1_d,s_velocity_d,s_velocity1_d,tao_d,strain_p_d,strain_s_d,dem_p_all_d);

								demig_fwd_vx_mul<<<dimGrid,dimBlock>>>(rvx2_d,rvx1_d,rtxx2_d,rtxz2_d,attenuation_d,dx,dz,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,s_density_d,dem_p1_d,dem_p2_d,dem_p3_d,dem_p4_d,dem_p5_d);

								demig_fwd_vz_mul<<<dimGrid,dimBlock>>>(rvz2_d,rvz1_d,rtzz2_d,rtxz2_d,attenuation_d,dx,dz,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,s_density_d,dem_p1_d,dem_p2_d,dem_p3_d,dem_p4_d,dem_p5_d);
							}

								if(0==(it)%100&&join_wavefield==1&&iter==0)
								{
									cudaMemcpy(wf_append,rvx2_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
									sprintf(filename,"./wavefield1/5/vx-%d-shot_%d",ishot+1,it);
									write_file_1d(wf_append,nxanza,filename);
									//exchange(wf_append,wf,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
									//write_file_1d(wf,nx_size_nz,filename);
											
									cudaMemcpy(wf_append,rvz2_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
									sprintf(filename,"./wavefield1/5/vz-%d-shot_%d",ishot+1,it);
									write_file_1d(wf_append,nxanza,filename);
									//exchange(wf_append,wf,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
									//write_file_1d(wf,nx_size_nz,filename);

									cudaMemcpy(wf_append,dem_p3_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
									sprintf(filename,"./wavefield1/5/dem-p3-%d-shot_%d",ishot+1,it);
									write_file_1d(wf_append,nxanza,filename);
									//exchange(wf_append,wf,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
									//write_file_1d(wf,nx_size_nz,filename);
								}

							if(it>=wavelet_half&&it<(lt+wavelet_half))
							{
									//write_shot<<<receiver_num,1>>>(rvx2_d,rvz2_d,cal_shot_x_d,cal_shot_z_d,it-wavelet_half,lt,receiver_num,receiver_depth,receiver_x_cord[ishot],receiver_interval,boundary_left,boundary_up,nz_append,dx,dt,source_x_cord[ishot],s_velocity_d,wavelet_half);
								if(receiver_offset==0)
								{
									write_shot_x_z<<<receiver_num,1>>>(rvx2_d,cal_shot_x1_d,it-wavelet_half,lt,receiver_num,receiver_x_cord[ishot],receiver_interval,receiver_z_cord[ishot],receiver_z_interval,boundary_left,boundary_up,nz_append);///for vsp 2017年03月14日 星期二 08时46分12秒 
									write_shot_x_z<<<receiver_num,1>>>(rvz2_d,cal_shot_z1_d,it-wavelet_half,lt,receiver_num,receiver_x_cord[ishot],receiver_interval,receiver_z_cord[ishot],receiver_z_interval,boundary_left,boundary_up,nz_append);///for vsp 2017年03月14日 星期二 08时46分12秒
								}
								else
								{
									write_shot_x_z_acqusition<<<receiver_num,1>>>(rvx2_d,cal_shot_x1_d,it-wavelet_half,lt,receiver_num,receiver_x_cord[ishot],receiver_interval,receiver_z_cord[ishot],receiver_z_interval,boundary_left,boundary_up,nz_append);///for vsp 2017年03月14日 星期二 08时46分12秒 
									write_shot_x_z_acqusition<<<receiver_num,1>>>(rvz2_d,cal_shot_z1_d,it-wavelet_half,lt,receiver_num,receiver_x_cord[ishot],receiver_interval,receiver_z_cord[ishot],receiver_z_interval,boundary_left,boundary_up,nz_append);///for vsp 2017年03月14日 星期二 08时46分12秒
								}
							}

							rep=rvx1_d;rvx1_d=rvx2_d;rvx2_d=rep;
							rep=rvz1_d;rvz1_d=rvz2_d;rvz2_d=rep;
							rep=rtxx1_d;rtxx1_d=rtxx2_d;rtxx2_d=rep;
							rep=rtzz1_d;rtzz1_d=rtzz2_d;rtzz2_d=rep;
							rep=rtxz1_d;rtxz1_d=rtxz2_d;rtxz2_d=rep;

							rep=rtp1_d;rtp1_d=rtp2_d;rtp2_d=rep;
							rep=rvxp1_d;rvxp1_d=rvxp2_d;rvxp2_d=rep;
							rep=rvzp1_d;rvzp1_d=rvzp2_d;rvzp2_d=rep;
							rep=rvxs1_d;rvxs1_d=rvxs2_d;rvxs2_d=rep;
							rep=rvzs1_d;rvzs1_d=rvzs2_d;rvzs2_d=rep;/////fast...........................................

							rep=rmem_p1_d;rmem_p1_d=rmem_p2_d;rmem_p2_d=rep;
							rep=rmem_xx1_d;rmem_xx1_d=rmem_xx2_d;rmem_xx2_d=rep;
							rep=rmem_zz1_d;rmem_zz1_d=rmem_zz2_d;rmem_zz2_d=rep;
							rep=rmem_xz1_d;rmem_xz1_d=rmem_xz2_d;rmem_xz2_d=rep;
						}

							if(ishot%20==0)
							{		
								/////////output cal shots
								cudaMemcpy(shotgather,cal_shot_x1_d,lt_rec*sizeof(float),cudaMemcpyDeviceToHost);
								sprintf(filename,"./someoutput/bin/direct1_cal_shot_x_%d_iter_%d",ishot+1,iter+1);
								write_file_1d(shotgather,lt_rec,filename);
								cudaMemcpy(shotgather,cal_shot_z1_d,lt_rec*sizeof(float),cudaMemcpyDeviceToHost);
								sprintf(filename,"./someoutput/bin/direct1_cal_shot_z_%d_iter_%d",ishot+1,iter+1);
								write_file_1d(shotgather,lt_rec,filename);
								/////////output cal shots
							}
					}

/////////////////////////////////////another_born_modeling
/////////////////////////////////////get smooth vp;				
					cuda_get_partly_mode_boundary<<<dimGrid_new,dimBlock>>>(s_velocity_all_d,wf_d,nx,nz,receiver_x_cord[ishot],receiver_interval,receiver_num,nx_append_new,nz_append,boundary_left,boundary_up);
					cuda_cal_expand<<<dimGrid,dimBlock>>>(s_velocity_d,wf_d,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);

/////////////////////////////////////get smooth vs;				
					cuda_get_partly_mode_boundary<<<dimGrid_new,dimBlock>>>(s_velocity1_all_d,wf_d,nx,nz,receiver_x_cord[ishot],receiver_interval,receiver_num,nx_append_new,nz_append,boundary_left,boundary_up);
					cuda_cal_expand<<<dimGrid,dimBlock>>>(s_velocity1_d,wf_d,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);

/////////////////////////////////////get smooth density;				
					cuda_get_partly_mode_boundary<<<dimGrid_new,dimBlock>>>(s_density_all_d,wf_d,nx,nz,receiver_x_cord[ishot],receiver_interval,receiver_num,nx_append_new,nz_append,boundary_left,boundary_up);
					cuda_cal_expand<<<dimGrid,dimBlock>>>(s_density_d,wf_d,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);

/////////////////////////////////////get smooth qp;				
					cuda_get_partly_mode_boundary<<<dimGrid_new,dimBlock>>>(s_qp_all_d,wf_d,nx,nz,receiver_x_cord[ishot],receiver_interval,receiver_num,nx_append_new,nz_append,boundary_left,boundary_up);
					cuda_cal_expand<<<dimGrid,dimBlock>>>(s_qp_d,wf_d,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);

/////////////////////////////////////get smooth qs;				
					cuda_get_partly_mode_boundary<<<dimGrid_new,dimBlock>>>(s_qs_all_d,wf_d,nx,nz,receiver_x_cord[ishot],receiver_interval,receiver_num,nx_append_new,nz_append,boundary_left,boundary_up);
					cuda_cal_expand<<<dimGrid,dimBlock>>>(s_qs_d,wf_d,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);

					/*if((receiver_offset!=0)||(offset_left[ishot]>receiver_offset)||(offset_right[ishot]>receiver_offset))
					{								
						cuda_expand_acqusition_left_and_right<<<dimGrid_new,dimBlock>>>(s_velocity_d,offset_left[ishot],offset_right[ishot],source_x_cord[ishot],receiver_offset,nx_size,nx_append,nz_append,boundary_left,boundary_up);

						cuda_expand_acqusition_left_and_right<<<dimGrid_new,dimBlock>>>(s_velocity1_d,offset_left[ishot],offset_right[ishot],source_x_cord[ishot],receiver_offset,nx_size,nx_append,nz_append,boundary_left,boundary_up);

						cuda_expand_acqusition_left_and_right<<<dimGrid_new,dimBlock>>>(s_density_d,offset_left[ishot],offset_right[ishot],source_x_cord[ishot],receiver_offset,nx_size,nx_append,nz_append,boundary_left,boundary_up);

						cuda_expand_acqusition_left_and_right<<<dimGrid_new,dimBlock>>>(s_qp_d,offset_left[ishot],offset_right[ishot],source_x_cord[ishot],receiver_offset,nx_size,nx_append,nz_append,boundary_left,boundary_up);

						cuda_expand_acqusition_left_and_right<<<dimGrid_new,dimBlock>>>(s_qs_d,offset_left[ishot],offset_right[ishot],source_x_cord[ishot],receiver_offset,nx_size,nx_append,nz_append,boundary_left,boundary_up);						
					}*/

/////////////////////////////////QQQQQQQQQQQQQQQ2017年07月27日 星期四 10时11分40秒 
					cuda_cal_viscoelastic<<< dimGrid,dimBlock>>>(modul_p_d,modul_s_d,s_qp_d,s_qs_d,tao_d,strain_p_d,strain_s_d,freq,s_velocity_d,s_velocity1_d,s_density_d,nx_append,nz_append);
					
					if(inversion_para==0||inversion_para==1)
					{
/////////////////////////////////////get perturbed lame1;
						cuda_get_partly_mode<<<dimGrid_new,dimBlock>>>(all_conj_lame1_d,wf_d,nx,nz,receiver_x_cord[ishot],receiver_interval,receiver_num);
						cuda_cal_expand<<<dimGrid,dimBlock>>>(tmp_perturb_lame1_d,wf_d,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
/////////////////////////////////////get perturbed lame2;
						cuda_get_partly_mode<<<dimGrid_new,dimBlock>>>(all_conj_lame2_d,wf_d,nx,nz,receiver_x_cord[ishot],receiver_interval,receiver_num);
						cuda_cal_expand<<<dimGrid,dimBlock>>>(tmp_perturb_lame2_d,wf_d,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
/////////////////////////////////////get perturbed lame2;
						cuda_get_partly_mode<<<dimGrid_new,dimBlock>>>(all_conj_den_d,wf_d,nx,nz,receiver_x_cord[ishot],receiver_interval,receiver_num);
						cuda_cal_expand<<<dimGrid,dimBlock>>>(tmp_perturb_den_d,wf_d,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);

						/*if((receiver_offset!=0)||(offset_left[ishot]>receiver_offset)||(offset_right[ishot]>receiver_offset))
						{								
							cuda_expand_acqusition_left_and_right<<<dimGrid_new,dimBlock>>>(tmp_perturb_lame1_d,offset_left[ishot],offset_right[ishot],source_x_cord[ishot],receiver_offset,nx_size,nx_append,nz_append,boundary_left,boundary_up);

							cuda_expand_acqusition_left_and_right<<<dimGrid_new,dimBlock>>>(tmp_perturb_lame2_d,offset_left[ishot],offset_right[ishot],source_x_cord[ishot],receiver_offset,nx_size,nx_append,nz_append,boundary_left,boundary_up);

							cuda_expand_acqusition_left_and_right<<<dimGrid_new,dimBlock>>>(tmp_perturb_den_d,offset_left[ishot],offset_right[ishot],source_x_cord[ishot],receiver_offset,nx_size,nx_append,nz_append,boundary_left,boundary_up);
						}*/
					}

					if(inversion_para==2||inversion_para==3)
					{
/////////////////////////////////////get perturbed lame1;
						cuda_get_partly_mode<<<dimGrid_new,dimBlock>>>(all_conj_vp_d,wf_d,nx,nz,receiver_x_cord[ishot],receiver_interval,receiver_num);
						cuda_cal_expand<<<dimGrid,dimBlock>>>(tmp_perturb_vp_d,wf_d,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
/////////////////////////////////////get perturbed lame2;
						cuda_get_partly_mode<<<dimGrid_new,dimBlock>>>(all_conj_vs_d,wf_d,nx,nz,receiver_x_cord[ishot],receiver_interval,receiver_num);
						cuda_cal_expand<<<dimGrid,dimBlock>>>(tmp_perturb_vs_d,wf_d,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
/////////////////////////////////////get perturbed lame2;
						cuda_get_partly_mode<<<dimGrid_new,dimBlock>>>(all_conj_density_d,wf_d,nx,nz,receiver_x_cord[ishot],receiver_interval,receiver_num);
						cuda_cal_expand<<<dimGrid,dimBlock>>>(tmp_perturb_density_d,wf_d,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);

						/*if((receiver_offset!=0)||(offset_left[ishot]>receiver_offset)||(offset_right[ishot]>receiver_offset))
						{								
							cuda_expand_acqusition_left_and_right<<<dimGrid_new,dimBlock>>>(tmp_perturb_vp_d,offset_left[ishot],offset_right[ishot],source_x_cord[ishot],receiver_offset,nx_size,nx_append,nz_append,boundary_left,boundary_up);

							cuda_expand_acqusition_left_and_right<<<dimGrid_new,dimBlock>>>(tmp_perturb_vs_d,offset_left[ishot],offset_right[ishot],source_x_cord[ishot],receiver_offset,nx_size,nx_append,nz_append,boundary_left,boundary_up);

							cuda_expand_acqusition_left_and_right<<<dimGrid_new,dimBlock>>>(tmp_perturb_density_d,offset_left[ishot],offset_right[ishot],source_x_cord[ishot],receiver_offset,nx_size,nx_append,nz_append,boundary_left,boundary_up);
						}*/
					}

					if(correlation_misfit==0)
					{
						////read real residual
						sprintf(filename,"./someoutput/bin/res_shot_x_%d_iter_%d",ishot+1,iter+1);
						fread_file_1d(shotgather,receiver_num,lt,filename);
						cudaMemcpy(res_shot_x_d,shotgather,lt_rec*sizeof(float),cudaMemcpyHostToDevice);

						sprintf(filename,"./someoutput/bin/res_shot_z_%d_iter_%d",ishot+1,iter+1);
						fread_file_1d(shotgather,receiver_num,lt,filename);
						cudaMemcpy(res_shot_z_d,shotgather,lt_rec*sizeof(float),cudaMemcpyHostToDevice);
						////read real residual
					}
					
					else
					{	
						///////////////////it is noted that  the first iteration is conventional LSRTM
						if(iter==0)
						{
							////read real residual
							sprintf(filename,"./someoutput/bin/res_shot_x_%d_iter_%d",ishot+1,iter+1);
							fread_file_1d(shotgather,receiver_num,lt,filename);
							cudaMemcpy(res_shot_x_d,shotgather,lt_rec*sizeof(float),cudaMemcpyHostToDevice);

							sprintf(filename,"./someoutput/bin/res_shot_z_%d_iter_%d",ishot+1,iter+1);
							fread_file_1d(shotgather,receiver_num,lt,filename);
							cudaMemcpy(res_shot_z_d,shotgather,lt_rec*sizeof(float),cudaMemcpyHostToDevice);
							////read real residual
						}
						
						if(iter>0)
						{
							//////fread obs shot	
							sprintf(filename,"./someoutput/bin/obs_shot_x_%d",ishot+1);
							fread_file_1d(shotgather,receiver_num,lt,filename);
							cudaMemcpy(obs_shot_x_d,shotgather,lt_rec*sizeof(float),cudaMemcpyHostToDevice);

							sprintf(filename,"./someoutput/bin/obs_shot_z_%d",ishot+1);
							fread_file_1d(shotgather,receiver_num,lt,filename);
							cudaMemcpy(obs_shot_z_d,shotgather,lt_rec*sizeof(float),cudaMemcpyHostToDevice);
							//////fread obs shot
								
							////////for sn!=0 data
							if(receiver_offset!=0)
							{
								cauda_zero_acqusition_left_and_right<<<dimGrid_lt,dimBlock>>>(obs_shot_x_d,offset_left[ishot],offset_right[ishot],source_x_cord[ishot],receiver_offset,receiver_num,lt);
								cauda_zero_acqusition_left_and_right<<<dimGrid_lt,dimBlock>>>(obs_shot_z_d,offset_left[ishot],offset_right[ishot],source_x_cord[ishot],receiver_offset,receiver_num,lt);
							}

							//////fread iter+1 tmp_shot			
							sprintf(filename,"./someoutput/bin/tmp_shot_x_%d_iter_%d",ishot+1,iter+1);
							fread_file_1d(shotgather,receiver_num,lt,filename);
							cudaMemcpy(tmp_shot_x_d,shotgather,lt_rec*sizeof(float),cudaMemcpyHostToDevice);

							sprintf(filename,"./someoutput/bin/tmp_shot_z_%d_iter_%d",ishot+1,iter+1);
							fread_file_1d(shotgather,receiver_num,lt,filename);
							cudaMemcpy(tmp_shot_z_d,shotgather,lt_rec*sizeof(float),cudaMemcpyHostToDevice);
						}				
					}

					if(vsp_2!=0)
					{
						////read real residual
						sprintf(filename,"./someoutput/bin/res_shot_x_%d_iter_%d_2",ishot+1,iter+1);
						fread_file_1d(shotgather,receiver_num,lt,filename);
						cudaMemcpy(res_shot_x_d_2,shotgather,lt_rec*sizeof(float),cudaMemcpyHostToDevice);

						sprintf(filename,"./someoutput/bin/res_shot_z_%d_iter_%d_2",ishot+1,iter+1);
						fread_file_1d(shotgather,receiver_num,lt,filename);
						cudaMemcpy(res_shot_z_d_2,shotgather,lt_rec*sizeof(float),cudaMemcpyHostToDevice);
						////read real residual
					}

					memset((void *)(wf_append),0,nxanza*sizeof(float));

					cudaMemset(vx1_d,0,nxanza*sizeof(float));
					cudaMemset(vz1_d,0,nxanza*sizeof(float));
					cudaMemset(txx1_d,0,nxanza*sizeof(float));
					cudaMemset(tzz1_d,0,nxanza*sizeof(float));
					cudaMemset(txz1_d,0,nxanza*sizeof(float));

					cudaMemset(vx2_d,0,nxanza*sizeof(float));
					cudaMemset(vz2_d,0,nxanza*sizeof(float));
					cudaMemset(txx2_d,0,nxanza*sizeof(float));
					cudaMemset(tzz2_d,0,nxanza*sizeof(float));
					cudaMemset(txz2_d,0,nxanza*sizeof(float));
	
					cudaMemset(tp2_d,0,nxanza*sizeof(float));
					cudaMemset(tp1_d,0,nxanza*sizeof(float));
					cudaMemset(vxp2_d,0,nxanza*sizeof(float));
					cudaMemset(vxp1_d,0,nxanza*sizeof(float));
					cudaMemset(vzp2_d,0,nxanza*sizeof(float));
					cudaMemset(vzp1_d,0,nxanza*sizeof(float));
					cudaMemset(vxs2_d,0,nxanza*sizeof(float));
					cudaMemset(vxs1_d,0,nxanza*sizeof(float));
					cudaMemset(vzs2_d,0,nxanza*sizeof(float));
					cudaMemset(vzs1_d,0,nxanza*sizeof(float));

					cudaMemset(mem_p1_d,0,nxanza*sizeof(float));
					cudaMemset(mem_xx1_d,0,nxanza*sizeof(float));
					cudaMemset(mem_zz1_d,0,nxanza*sizeof(float));
					cudaMemset(mem_xz1_d,0,nxanza*sizeof(float));
					cudaMemset(mem_p2_d,0,nxanza*sizeof(float));
					cudaMemset(mem_xx2_d,0,nxanza*sizeof(float));
					cudaMemset(mem_zz2_d,0,nxanza*sizeof(float));
					cudaMemset(mem_xz2_d,0,nxanza*sizeof(float));

					cudaMemset(rvx1_d,0,nxanza*sizeof(float));
					cudaMemset(rvz1_d,0,nxanza*sizeof(float));
					cudaMemset(rtxx1_d,0,nxanza*sizeof(float));
					cudaMemset(rtzz1_d,0,nxanza*sizeof(float));
					cudaMemset(rtxz1_d,0,nxanza*sizeof(float));

					cudaMemset(rvx2_d,0,nxanza*sizeof(float));
					cudaMemset(rvz2_d,0,nxanza*sizeof(float));
					cudaMemset(rtxx2_d,0,nxanza*sizeof(float));
					cudaMemset(rtzz2_d,0,nxanza*sizeof(float));
					cudaMemset(rtxz2_d,0,nxanza*sizeof(float));

					cudaMemset(rtp2_d,0,nxanza*sizeof(float));
					cudaMemset(rtp1_d,0,nxanza*sizeof(float));
					cudaMemset(rvxp2_d,0,nxanza*sizeof(float));
					cudaMemset(rvxp1_d,0,nxanza*sizeof(float));
					cudaMemset(rvzp2_d,0,nxanza*sizeof(float));
					cudaMemset(rvzp1_d,0,nxanza*sizeof(float));
					cudaMemset(rvxs2_d,0,nxanza*sizeof(float));
					cudaMemset(rvxs1_d,0,nxanza*sizeof(float));
					cudaMemset(rvzs2_d,0,nxanza*sizeof(float));
					cudaMemset(rvzs1_d,0,nxanza*sizeof(float));

					cudaMemset(rmem_p1_d,0,nxanza*sizeof(float));
					cudaMemset(rmem_xx1_d,0,nxanza*sizeof(float));
					cudaMemset(rmem_zz1_d,0,nxanza*sizeof(float));
					cudaMemset(rmem_xz1_d,0,nxanza*sizeof(float));
					cudaMemset(rmem_p2_d,0,nxanza*sizeof(float));
					cudaMemset(rmem_xx2_d,0,nxanza*sizeof(float));
					cudaMemset(rmem_zz2_d,0,nxanza*sizeof(float));
					cudaMemset(rmem_xz2_d,0,nxanza*sizeof(float));

					cudaMemset(vx_x_d,0,nxanza*sizeof(float));
					cudaMemset(vx_z_d,0,nxanza*sizeof(float));
					cudaMemset(vz_x_d,0,nxanza*sizeof(float));
					cudaMemset(vz_z_d,0,nxanza*sizeof(float));

					cudaMemset(vx_t_d,0,nxanza*sizeof(float));
					cudaMemset(vz_t_d,0,nxanza*sizeof(float));

			for(int it=0;it<lt+wavelet_half;it++)
					{
						//if(fmod((it+1.0)-wavelet_half,1000.0)==0) warn("shot=%d,step=forward 2,it=%d",ishot+1,(it+1)-wavelet_half);
						if(it<wavelet_length)
						{
							//add_source<<<1,1>>>(txx1_d,tzz1_d,wavelet_d,source_x_cord[ishot],shot_depth,it,boundary_up,boundary_left,nz_append);
							//add_source<<<1,1>>>(txx1_d,tzz1_d,wavelet_d,source_x_cord[ishot],source_z_cord[ishot],it,boundary_up,boundary_left,nz_append);///for vsp 2017年03月14日 星期二 08时55分59秒 
							add_source<<<1,1>>>(txx1_d,tzz1_d,wavelet_d,source_x_cord[ishot]-receiver_x_cord[ishot],source_z_cord[ishot],it,boundary_up,boundary_left,nz_append);///for vsp 2017年03月14日 星期二 08时55分59秒 
						}	
							fwd_vx_new<<<dimGrid,dimBlock>>>(vx_t_d,vx2_d,vx1_d,txx1_d,txz1_d,attenuation_d,dx,dz,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,s_density_d);

							fwd_vz_new<<<dimGrid,dimBlock>>>(vz_t_d,vz2_d,vz1_d,tzz1_d,txz1_d,attenuation_d,dx,dz,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,s_density_d);	

							if(migration_type==0)	fwd_txxzzxzpp_new<<<dimGrid,dimBlock>>>(vx_x_d,vx_z_d,vz_x_d,vz_z_d,tp2_d,tp1_d,txx2_d,txx1_d,tzz2_d,tzz1_d,txz2_d,txz1_d,vx2_d,vz2_d,s_velocity_d,s_velocity1_d,attenuation_d,dt,dx,dz,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,s_density_d);

							else	fwd_txxzzxzpp_viscoelastic_and_memory_3parameterization_new<<<dimGrid,dimBlock>>>(vx_x_d,vx_z_d,vz_x_d,vz_z_d,tp2_d,tp1_d,txx2_d,txx1_d,tzz2_d,tzz1_d,txz2_d,txz1_d,vx2_d,vz2_d,modul_p_d,modul_s_d,attenuation_d,s_density_d,mem_p2_d,mem_p1_d,mem_xx2_d,mem_xx1_d,mem_zz2_d,mem_zz1_d,mem_xz2_d,mem_xz1_d,s_velocity_d,s_velocity1_d,tao_d,strain_p_d,strain_s_d,packaging_d);

							if(0==(it)%100&&join_wavefield==1&&iter==0)
							{
								cudaMemcpy(wf_append,vx2_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
								sprintf(filename,"./wavefield1/4/vx-%d-shot_%d",ishot+1,it);
								write_file_1d(wf_append,nxanza,filename);
								//exchange(wf_append,wf,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
								//write_file_1d(wf,nx_size_nz,filename);
										
								cudaMemcpy(wf_append,vz2_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
								sprintf(filename,"./wavefield1/4/vz-%d-shot_%d",ishot+1,it);
								write_file_1d(wf_append,nxanza,filename);
								//exchange(wf_append,wf,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
								//write_file_1d(wf,nx_size_nz,filename);

								cudaMemcpy(wf_append,vz_z_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
								sprintf(filename,"./wavefield1/4/vz-z-%d-shot_%d",ishot+1,it);
								write_file_1d(wf_append,nxanza,filename);
								//exchange(wf_append,wf,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
								//write_file_1d(wf,nx_size_nz,filename);

								cudaMemcpy(wf_append,vz_x_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
								sprintf(filename,"./wavefield1/4/vz-x-%d-shot_%d",ishot+1,it);
								write_file_1d(wf_append,nxanza,filename);
								//exchange(wf_append,wf,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
								//write_file_1d(wf,nx_size_nz,filename);

								cudaMemcpy(wf_append,vz_t_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
								sprintf(filename,"./wavefield1/4/vz-t-%d-shot_%d",ishot+1,it);
								write_file_1d(wf_append,nxanza,filename);
								//exchange(wf_append,wf,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
								//write_file_1d(wf,nx_size_nz,filename);
							}

						rep=vx1_d;vx1_d=vx2_d;vx2_d=rep;
						rep=vz1_d;vz1_d=vz2_d;vz2_d=rep;
						rep=txx1_d;txx1_d=txx2_d;txx2_d=rep;
						rep=tzz1_d;tzz1_d=tzz2_d;tzz2_d=rep;
						rep=txz1_d;txz1_d=txz2_d;txz2_d=rep;

						rep=tp1_d;tp1_d=tp2_d;tp2_d=rep;
						rep=vxp1_d;vxp1_d=vxp2_d;vxp2_d=rep;
						rep=vzp1_d;vzp1_d=vzp2_d;vzp2_d=rep;
						rep=vxs1_d;vxs1_d=vxs2_d;vxs2_d=rep;
						rep=vzs1_d;vzs1_d=vzs2_d;vzs2_d=rep;

						rep=mem_p1_d;mem_p1_d=mem_p2_d;mem_p2_d=rep;
						rep=mem_xx1_d;mem_xx1_d=mem_xx2_d;mem_xx2_d=rep;
						rep=mem_zz1_d;mem_zz1_d=mem_zz2_d;mem_zz2_d=rep;
						rep=mem_xz1_d;mem_xz1_d=mem_xz2_d;mem_xz2_d=rep;
///////////////////////demigration to calculate cal_shots!!!!!!!!!!
						if(migration_type==0)
						{
							/*if(inversion_para==0)
							{
							cuda_cal_dem_parameter<<<dimGrid,dimBlock>>>(dem_p1_d,dem_p2_d,dem_p3_d,dem_p4_d,dem_p5_d,vx_x_d,vx_z_d,vz_x_d,vz_z_d,vx_t_d,vz_t_d,tmp_perturb_lame1_d,tmp_perturb_lame2_d,tmp_perturb_den_d,nx_append_radius,nz_append_radius);
							}
					
							if(inversion_para==1)
							{
							cuda_cal_dem_parameter_lame<<<dimGrid,dimBlock>>>(dem_p1_d,dem_p2_d,dem_p3_d,dem_p4_d,dem_p5_d,vx_x_d,vx_z_d,vz_x_d,vz_z_d,vx_t_d,vz_t_d,tmp_perturb_lame1_d,tmp_perturb_lame2_d,tmp_perturb_den_d,nx_append_radius,nz_append_radius,s_velocity_d,s_velocity1_d,s_density_d);
							}

							if(inversion_para==2)
							{
							cuda_cal_dem_parameter_velocity<<<dimGrid,dimBlock>>>(dem_p1_d,dem_p2_d,dem_p3_d,dem_p4_d,dem_p5_d,vx_x_d,vx_z_d,vz_x_d,vz_z_d,vx_t_d,vz_t_d,tmp_perturb_vp_d,tmp_perturb_vs_d,tmp_perturb_density_d,nx_append_radius,nz_append_radius,s_velocity_d,s_velocity1_d,s_density_d);
							}

							if(inversion_para==3)
							{
							cuda_cal_dem_parameter_impedance<<<dimGrid,dimBlock>>>(dem_p1_d,dem_p2_d,dem_p3_d,dem_p4_d,dem_p5_d,vx_x_d,vx_z_d,vz_x_d,vz_z_d,vx_t_d,vz_t_d,tmp_perturb_vp_d,tmp_perturb_vs_d,tmp_perturb_density_d,nx_append_radius,nz_append_radius,s_velocity_d,s_velocity1_d,s_density_d);
							}*/

							cuda_cal_dem_parameter_elastic_media<<<dimGrid,dimBlock>>>(dem_p1_d,dem_p2_d,dem_p3_d,dem_p4_d,dem_p5_d,vx_x_d,vx_z_d,vz_x_d,vz_z_d,vx_t_d,vz_t_d,tmp_perturb_lame1_d,tmp_perturb_lame2_d,tmp_perturb_den_d,tmp_perturb_vp_d,tmp_perturb_vs_d,tmp_perturb_density_d,nx_append_radius,nz_append_radius,s_velocity_d,s_velocity1_d,s_density_d,inversion_para);

							demig_fwd_txxzzxz_mul<<<dimGrid,dimBlock>>>(rtxx2_d,rtxx1_d,rtzz2_d,rtzz1_d,rtxz2_d,rtxz1_d,rvx1_d,rvz1_d,s_velocity_d,s_velocity1_d,attenuation_d,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,s_density_d,dem_p1_d,dem_p2_d,dem_p3_d,dem_p4_d,dem_p5_d);

							demig_fwd_vx_mul<<<dimGrid,dimBlock>>>(rvx2_d,rvx1_d,rtxx2_d,rtxz2_d,attenuation_d,dx,dz,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,s_density_d,dem_p1_d,dem_p2_d,dem_p3_d,dem_p4_d,dem_p5_d);

							demig_fwd_vz_mul<<<dimGrid,dimBlock>>>(rvz2_d,rvz1_d,rtzz2_d,rtxz2_d,attenuation_d,dx,dz,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,s_density_d,dem_p1_d,dem_p2_d,dem_p3_d,dem_p4_d,dem_p5_d);

							/*demig_fwd_vx_mul<<<dimGrid,dimBlock>>>(rvx2_d,rvx1_d,rtxx1_d,rtxz1_d,attenuation_d,dx,dz,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,s_density_d,dem_p1_d,dem_p2_d,dem_p3_d,dem_p4_d,dem_p5_d);

							demig_fwd_vz_mul<<<dimGrid,dimBlock>>>(rvz2_d,rvz1_d,rtzz1_d,rtxz1_d,attenuation_d,dx,dz,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,s_density_d,dem_p1_d,dem_p2_d,dem_p3_d,dem_p4_d,dem_p5_d);

							demig_fwd_txxzzxz_mul<<<dimGrid,dimBlock>>>(rtxx2_d,rtxx1_d,rtzz2_d,rtzz1_d,rtxz2_d,rtxz1_d,rvx2_d,rvz2_d,s_velocity_d,s_velocity1_d,attenuation_d,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,s_density_d,dem_p1_d,dem_p2_d,dem_p3_d,dem_p4_d,dem_p5_d);*/

							/*cal_sum_a_b_to_c<<<dimGrid,dimBlock>>>(dem_p1_d,rvx1_d,rvx1_d,nx_append,nz_append);

							cal_sum_a_b_to_c<<<dimGrid,dimBlock>>>(dem_p2_d,rvz1_d,rvz1_d,nx_append,nz_append);

							cal_sum_a_b_to_c<<<dimGrid,dimBlock>>>(dem_p3_d,rtxx1_d,rtxx1_d,nx_append,nz_append);

							cal_sum_a_b_to_c<<<dimGrid,dimBlock>>>(dem_p4_d,rtzz1_d,rtzz1_d,nx_append,nz_append);

							cal_sum_a_b_to_c<<<dimGrid,dimBlock>>>(dem_p5_d,rtxz1_d,rtxz1_d,nx_append,nz_append);

							fwd_txxzzxz<<<dimGrid,dimBlock>>>(rtxx2_d,rtxx1_d,rtzz2_d,rtzz1_d,rtxz2_d,rtxz1_d,rvx1_d,rvz1_d,s_velocity_d,s_velocity1_d,attenuation_d,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,s_density_d);

							fwd_vx<<<dimGrid,dimBlock>>>(rvx2_d,rvx1_d,rtxx2_d,rtxz2_d,attenuation_d,dx,dz,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,s_density_d);
							fwd_vz<<<dimGrid,dimBlock>>>(rvz2_d,rvz1_d,rtzz2_d,rtxz2_d,attenuation_d,dx,dz,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,s_density_d);*/
						}

						if(migration_type==1)
						{
							/*cuda_cal_dem_parameter_viscoelastic_media<<<dimGrid,dimBlock>>>(dem_p1_d,dem_p2_d,dem_p3_d,dem_p4_d,dem_p5_d,dem_p6_d,dem_p7_d,dem_p8_d,vx_x_d,vx_z_d,vz_x_d,vz_z_d,vx_t_d,vz_t_d,tmp_perturb_lame1_d,tmp_perturb_lame2_d,tmp_perturb_den_d,tmp_perturb_vp_d,tmp_perturb_vs_d,tmp_perturb_density_d,nx_append_radius,nz_append_radius,s_velocity_d,s_velocity1_d,s_density_d,tao_d,strain_p_d,strain_s_d,dt,inversion_para);
							//cuda_cal_multiply<<<dimGrid,dimBlock>>>(tmp_perturb_den_d,s_density_d,dem_p1_d,nx_append_radius,nz_append_radius);

							cal_sum_a_b_to_c<<<dimGrid,dimBlock>>>(dem_p1_d,rvx1_d,rvx1_d,nx_append,nz_append);

							cal_sum_a_b_to_c<<<dimGrid,dimBlock>>>(dem_p2_d,rvz1_d,rvz1_d,nx_append,nz_append);

							cal_sum_a_b_to_c<<<dimGrid,dimBlock>>>(dem_p3_d,rtxx1_d,rtxx1_d,nx_append,nz_append);

							cal_sum_a_b_to_c<<<dimGrid,dimBlock>>>(dem_p4_d,rtzz1_d,rtzz1_d,nx_append,nz_append);

							cal_sum_a_b_to_c<<<dimGrid,dimBlock>>>(dem_p5_d,rtxz1_d,rtxz1_d,nx_append,nz_append);

							cal_sum_a_b_to_c<<<dimGrid,dimBlock>>>(dem_p6_d,rmem_xx1_d,rmem_xx1_d,nx_append,nz_append);

							cal_sum_a_b_to_c<<<dimGrid,dimBlock>>>(dem_p7_d,rmem_zz1_d,rmem_zz1_d,nx_append,nz_append);

							cal_sum_a_b_to_c<<<dimGrid,dimBlock>>>(dem_p8_d,rmem_xz1_d,rmem_xz1_d,nx_append,nz_append);
							fwd_vx<<<dimGrid,dimBlock>>>(rvx2_d,rvx1_d,rtxx1_d,rtxz1_d,attenuation_d,dx,dz,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,density_d);
							fwd_vz<<<dimGrid,dimBlock>>>(rvz2_d,rvz1_d,rtzz1_d,rtxz1_d,attenuation_d,dx,dz,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,density_d);
							fwd_txxzzxzpp_viscoelastic_and_memory_3parameterization<<<dimGrid,dimBlock>>>(rtp2_d,rtp1_d,rtxx2_d,rtxx1_d,rtzz2_d,rtzz1_d,rtxz2_d,rtxz1_d,rvx2_d,rvz2_d,modul_p_d,modul_s_d,attenuation_d,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,s_density_d,rmem_p2_d,rmem_p1_d,rmem_xx2_d,rmem_xx1_d,rmem_zz2_d,rmem_zz1_d,rmem_xz2_d,rmem_xz1_d,s_velocity_d,s_velocity1_d,tao_d,strain_p_d,strain_s_d);*/

							cuda_cal_dem_parameter_viscoelastic_media_new<<<dimGrid,dimBlock>>>(dem_p1_d,dem_p2_d,dem_p_all_d,vx_x_d,vx_z_d,vz_x_d,vz_z_d,vx_t_d,vz_t_d,tmp_perturb_lame1_d,tmp_perturb_lame2_d,tmp_perturb_den_d,tmp_perturb_vp_d,tmp_perturb_vs_d,tmp_perturb_density_d,nx_append_radius,nz_append_radius,s_velocity_d,s_velocity1_d,s_density_d,tao_d,strain_p_d,strain_s_d,dt,inversion_para);

							demig_fwd_txxzzxzpp_viscoelastic_and_memory_3parameterization<<<dimGrid,dimBlock>>>(rtp2_d,rtp1_d,rtxx2_d,rtxx1_d,rtzz2_d,rtzz1_d,rtxz2_d,rtxz1_d,rvx1_d,rvz1_d,modul_p_d,modul_s_d,attenuation_d,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,s_density_d,rmem_p2_d,rmem_p1_d,rmem_xx2_d,rmem_xx1_d,rmem_zz2_d,rmem_zz1_d,rmem_xz2_d,rmem_xz1_d,s_velocity_d,s_velocity1_d,tao_d,strain_p_d,strain_s_d,dem_p_all_d);

							demig_fwd_vx_mul<<<dimGrid,dimBlock>>>(rvx2_d,rvx1_d,rtxx2_d,rtxz2_d,attenuation_d,dx,dz,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,s_density_d,dem_p1_d,dem_p2_d,dem_p3_d,dem_p4_d,dem_p5_d);

							demig_fwd_vz_mul<<<dimGrid,dimBlock>>>(rvz2_d,rvz1_d,rtzz2_d,rtxz2_d,attenuation_d,dx,dz,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,s_density_d,dem_p1_d,dem_p2_d,dem_p3_d,dem_p4_d,dem_p5_d);

							/*demig_fwd_vx_mul<<<dimGrid,dimBlock>>>(rvx2_d,rvx1_d,rtxx1_d,rtxz1_d,attenuation_d,dx,dz,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,s_density_d,dem_p1_d,dem_p2_d,dem_p3_d,dem_p4_d,dem_p5_d);

							demig_fwd_vz_mul<<<dimGrid,dimBlock>>>(rvz2_d,rvz1_d,rtzz1_d,rtxz1_d,attenuation_d,dx,dz,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,s_density_d,dem_p1_d,dem_p2_d,dem_p3_d,dem_p4_d,dem_p5_d);

							demig_fwd_txxzzxzpp_viscoelastic_and_memory_3parameterization<<<dimGrid,dimBlock>>>(rtp2_d,rtp1_d,rtxx2_d,rtxx1_d,rtzz2_d,rtzz1_d,rtxz2_d,rtxz1_d,rvx2_d,rvz2_d,modul_p_d,modul_s_d,attenuation_d,dt,coe_opt_d,coe_x,coe_z,nx_append_radius,nz_append_radius,s_density_d,rmem_p2_d,rmem_p1_d,rmem_xx2_d,rmem_xx1_d,rmem_zz2_d,rmem_zz1_d,rmem_xz2_d,rmem_xz1_d,s_velocity_d,s_velocity1_d,tao_d,strain_p_d,strain_s_d,dem_p_all_d);*/
						}

							if(0==(it)%100&&join_wavefield==1&&iter==0)
							{
								cudaMemcpy(wf_append,rvx2_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
								sprintf(filename,"./wavefield1/5/vx-%d-shot_%d",ishot+1,it);
								write_file_1d(wf_append,nxanza,filename);
								//exchange(wf_append,wf,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
								//write_file_1d(wf,nx_size_nz,filename);
										
								cudaMemcpy(wf_append,rvz2_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
								sprintf(filename,"./wavefield1/5/vz-%d-shot_%d",ishot+1,it);
								write_file_1d(wf_append,nxanza,filename);
								//exchange(wf_append,wf,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
								//write_file_1d(wf,nx_size_nz,filename);

								cudaMemcpy(wf_append,dem_p1_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
								sprintf(filename,"./wavefield1/5/dem-p1-%d-shot_%d",ishot+1,it);
								write_file_1d(wf_append,nxanza,filename);
								//exchange(wf_append,wf,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
								//write_file_1d(wf,nx_size_nz,filename);

								cudaMemcpy(wf_append,dem_p2_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
								sprintf(filename,"./wavefield1/5/dem-p2-%d-shot_%d",ishot+1,it);
								write_file_1d(wf_append,nxanza,filename);
								//exchange(wf_append,wf,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
								//write_file_1d(wf,nx_size_nz,filename);

								cudaMemcpy(wf_append,dem_p3_d,nxanza*sizeof(float),cudaMemcpyDeviceToHost);
								sprintf(filename,"./wavefield1/5/dem-p3-%d-shot_%d",ishot+1,it);
								write_file_1d(wf_append,nxanza,filename);
								//exchange(wf_append,wf,nx_size,nz,nx_append,nz_append,boundary_left,boundary_up);
								//write_file_1d(wf,nx_size_nz,filename);
							}

						if(it>=wavelet_half&&it<(lt+wavelet_half))
						{
								//write_shot<<<receiver_num,1>>>(rvx2_d,rvz2_d,cal_shot_x_d,cal_shot_z_d,it-wavelet_half,lt,receiver_num,receiver_depth,receiver_x_cord[ishot],receiver_interval,boundary_left,boundary_up,nz_append,dx,dt,source_x_cord[ishot],s_velocity_d,wavelet_half);
							if(receiver_offset==0)
							{
								write_shot_x_z<<<receiver_num,1>>>(rvx2_d,cal_shot_x_d,it-wavelet_half,lt,receiver_num,receiver_x_cord[ishot],receiver_interval,receiver_z_cord[ishot],receiver_z_interval,boundary_left,boundary_up,nz_append);///for vsp 2017年03月14日 星期二 08时46分12秒 
								write_shot_x_z<<<receiver_num,1>>>(rvz2_d,cal_shot_z_d,it-wavelet_half,lt,receiver_num,receiver_x_cord[ishot],receiver_interval,receiver_z_cord[ishot],receiver_z_interval,boundary_left,boundary_up,nz_append);///for vsp 2017年03月14日 星期二 08时46分12秒
							}
							else
							{
								write_shot_x_z_acqusition<<<receiver_num,1>>>(rvx2_d,cal_shot_x_d,it-wavelet_half,lt,receiver_num,receiver_x_cord[ishot],receiver_interval,receiver_z_cord[ishot],receiver_z_interval,boundary_left,boundary_up,nz_append);///for vsp 2017年03月14日 星期二 08时46分12秒 
								write_shot_x_z_acqusition<<<receiver_num,1>>>(rvz2_d,cal_shot_z_d,it-wavelet_half,lt,receiver_num,receiver_x_cord[ishot],receiver_interval,receiver_z_cord[ishot],receiver_z_interval,boundary_left,boundary_up,nz_append);///for vsp 2017年03月14日 星期二 08时46分12秒
							}
						}

						rep=rvx1_d;rvx1_d=rvx2_d;rvx2_d=rep;
						rep=rvz1_d;rvz1_d=rvz2_d;rvz2_d=rep;
						rep=rtxx1_d;rtxx1_d=rtxx2_d;rtxx2_d=rep;
						rep=rtzz1_d;rtzz1_d=rtzz2_d;rtzz2_d=rep;
						rep=rtxz1_d;rtxz1_d=rtxz2_d;rtxz2_d=rep;

						rep=rtp1_d;rtp1_d=rtp2_d;rtp2_d=rep;
						rep=rvxp1_d;rvxp1_d=rvxp2_d;rvxp2_d=rep;
						rep=rvzp1_d;rvzp1_d=rvzp2_d;rvzp2_d=rep;
						rep=rvxs1_d;rvxs1_d=rvxs2_d;rvxs2_d=rep;
						rep=rvzs1_d;rvzs1_d=rvzs2_d;rvzs2_d=rep;/////fast...........................................

						rep=rmem_p1_d;rmem_p1_d=rmem_p2_d;rmem_p2_d=rep;
						rep=rmem_xx1_d;rmem_xx1_d=rmem_xx2_d;rmem_xx2_d=rep;
						rep=rmem_zz1_d;rmem_zz1_d=rmem_zz2_d;rmem_zz2_d=rep;
						rep=rmem_xz1_d;rmem_xz1_d=rmem_xz2_d;rmem_xz2_d=rep;
					}

						if(ishot%20==0)
						{
							cudaMemcpy(shotgather,cal_shot_x_d,lt_rec*sizeof(float),cudaMemcpyDeviceToHost);
							sprintf(filename,"./someoutput/bin/direct_cal_shot_x_%d_iter_%d",ishot+1,iter+1);
							write_file_1d(shotgather,lt_rec,filename);
							cudaMemcpy(shotgather,cal_shot_z_d,lt_rec*sizeof(float),cudaMemcpyDeviceToHost);
							sprintf(filename,"./someoutput/bin/direct_cal_shot_z_%d_iter_%d",ishot+1,iter+1);
							write_file_1d(shotgather,lt_rec,filename);
						}

						if(cut_direct_wave==0)
						{
							if(receiver_offset!=0)
							{
								cauda_zero_acqusition_left_and_right<<<dimGrid_lt,dimBlock>>>(cal_shot_x_d,offset_left[ishot],offset_right[ishot],source_x_cord[ishot],receiver_offset,receiver_num,lt);
								cauda_zero_acqusition_left_and_right<<<dimGrid_lt,dimBlock>>>(cal_shot_z_d,offset_left[ishot],offset_right[ishot],source_x_cord[ishot],receiver_offset,receiver_num,lt);
							}

							/////////output cal shots
							cudaMemcpy(shotgather,cal_shot_x_d,lt_rec*sizeof(float),cudaMemcpyDeviceToHost);
							sprintf(filename,"./someoutput/bin/cal_shot_x_%d_iter_%d",ishot+1,iter+1);
							write_file_1d(shotgather,lt_rec,filename);
							cudaMemcpy(shotgather,cal_shot_z_d,lt_rec*sizeof(float),cudaMemcpyDeviceToHost);
							sprintf(filename,"./someoutput/bin/cal_shot_z_%d_iter_%d",ishot+1,iter+1);
							write_file_1d(shotgather,lt_rec,filename);
							/////////output cal shots
						}

						if(cut_direct_wave==1)
						{
							cal_sub_a_b_to_c<<<dimGrid_lt,dimBlock>>>(cal_shot_x_d,cal_shot_x1_d,cal_shot_x_d,receiver_num,lt);

							cal_sub_a_b_to_c<<<dimGrid_lt,dimBlock>>>(cal_shot_z_d,cal_shot_z1_d,cal_shot_z_d,receiver_num,lt);

							if(receiver_offset!=0)
							{
								cauda_zero_acqusition_left_and_right<<<dimGrid_lt,dimBlock>>>(cal_shot_x_d,offset_left[ishot],offset_right[ishot],source_x_cord[ishot],receiver_offset,receiver_num,lt);
								cauda_zero_acqusition_left_and_right<<<dimGrid_lt,dimBlock>>>(cal_shot_z_d,offset_left[ishot],offset_right[ishot],source_x_cord[ishot],receiver_offset,receiver_num,lt);
							}

							/////////output cal shots
							cudaMemcpy(shotgather,cal_shot_x_d,lt_rec*sizeof(float),cudaMemcpyDeviceToHost);
							sprintf(filename,"./someoutput/bin/cal_shot_x_%d_iter_%d",ishot+1,iter+1);
							write_file_1d(shotgather,lt_rec,filename);
							cudaMemcpy(shotgather,cal_shot_z_d,lt_rec*sizeof(float),cudaMemcpyDeviceToHost);
							sprintf(filename,"./someoutput/bin/cal_shot_z_%d_iter_%d",ishot+1,iter+1);
							write_file_1d(shotgather,lt_rec,filename);
							/////////output cal shots
						}
		
						if(cut_direct_wave!=0&&cut_direct_wave!=1)///for vsp 2017年03月14日 星期二 08时55分03秒 
						{
							cut_direct_new1<<<dimGrid,dimBlock>>>(cal_shot_x_d,lt,source_x_cord[ishot],shot_depth,receiver_num,receiver_depth,receiver_x_cord[ishot],receiver_interval,boundary_left,boundary_up,nz_append,dx,dz,dt,velocity_d,wavelet_half,cut_direct_wave);
							cut_direct_new1<<<dimGrid,dimBlock>>>(cal_shot_z_d,lt,source_x_cord[ishot],shot_depth,receiver_num,receiver_depth,receiver_x_cord[ishot],receiver_interval,boundary_left,boundary_up,nz_append,dx,dz,dt,velocity_d,wavelet_half,cut_direct_wave);

							if(receiver_offset!=0)
							{
								cauda_zero_acqusition_left_and_right<<<dimGrid_lt,dimBlock>>>(cal_shot_x_d,offset_left[ishot],offset_right[ishot],source_x_cord[ishot],receiver_offset,receiver_num,lt);
								cauda_zero_acqusition_left_and_right<<<dimGrid_lt,dimBlock>>>(cal_shot_z_d,offset_left[ishot],offset_right[ishot],source_x_cord[ishot],receiver_offset,receiver_num,lt);
							}

							/////////output cal shots
							cudaMemcpy(shotgather,cal_shot_x_d,lt_rec*sizeof(float),cudaMemcpyDeviceToHost);
							sprintf(filename,"./someoutput/bin/cal_shot_x_%d_iter_%d",ishot+1,iter+1);
							write_file_1d(shotgather,lt_rec,filename);
							cudaMemcpy(shotgather,cal_shot_z_d,lt_rec*sizeof(float),cudaMemcpyDeviceToHost);
							sprintf(filename,"./someoutput/bin/cal_shot_z_%d_iter_%d",ishot+1,iter+1);
							write_file_1d(shotgather,lt_rec,filename);
							/////////output cal shots
						}

						if(correlation_misfit==0)//correlation
						{
							cuda_sum_alpha12_new_for_lsrtm<<<dimGrid_lt,dimBlock>>>(d_alpha1,d_alpha2,cal_shot_x_d,obs_shot_x_d,res_shot_x_d,receiver_num,lt);
							cuda_sum_alpha12_new_for_lsrtm<<<dimGrid_lt,dimBlock>>>(d_alpha1,d_alpha2,cal_shot_z_d,obs_shot_z_d,res_shot_z_d,receiver_num,lt);
						}

						else///correlation
						{
							if(iter==0)///////////////////it is noted that  the first iteration is conventional LSRTM
							{
								cuda_sum_alpha12_new_for_lsrtm<<<dimGrid_lt,dimBlock>>>(d_alpha1,d_alpha2,cal_shot_x_d,obs_shot_x_d,res_shot_x_d,receiver_num,lt);
								cuda_sum_alpha12_new_for_lsrtm<<<dimGrid_lt,dimBlock>>>(d_alpha1,d_alpha2,cal_shot_z_d,obs_shot_z_d,res_shot_z_d,receiver_num,lt);
							}				
							
							if(iter>0)
							{
								cuda_dot_sum<<<1,Block_Size>>>(tmp_shot_x_d,tmp_shot_x_d,lt_rec,&correlation_parameter_d[0]);
								cuda_dot_sum<<<1,Block_Size>>>(tmp_shot_z_d,tmp_shot_z_d,lt_rec,&correlation_parameter_d[0]);//tmp*tmp

								cuda_dot_sum<<<1,Block_Size>>>(obs_shot_x_d,obs_shot_x_d,lt_rec,&correlation_parameter_d[1]);
								cuda_dot_sum<<<1,Block_Size>>>(obs_shot_z_d,obs_shot_z_d,lt_rec,&correlation_parameter_d[1]);//obs*obs

								cuda_dot_sum<<<1,Block_Size>>>(cal_shot_x_d,cal_shot_x_d,lt_rec,&correlation_parameter_d[2]);
								cuda_dot_sum<<<1,Block_Size>>>(cal_shot_z_d,cal_shot_z_d,lt_rec,&correlation_parameter_d[2]);//cal*cal

								cuda_dot_sum<<<1,Block_Size>>>(tmp_shot_x_d,obs_shot_x_d,lt_rec,&correlation_parameter_d[3]);
								cuda_dot_sum<<<1,Block_Size>>>(tmp_shot_z_d,obs_shot_z_d,lt_rec,&correlation_parameter_d[3]);//tmp*obs	

								cuda_dot_sum<<<1,Block_Size>>>(tmp_shot_x_d,cal_shot_x_d,lt_rec,&correlation_parameter_d[4]);
								cuda_dot_sum<<<1,Block_Size>>>(tmp_shot_z_d,cal_shot_z_d,lt_rec,&correlation_parameter_d[4]);//tmp*cal

								cuda_dot_sum<<<1,Block_Size>>>(cal_shot_x_d,obs_shot_x_d,lt_rec,&correlation_parameter_d[5]);
								cuda_dot_sum<<<1,Block_Size>>>(cal_shot_z_d,obs_shot_z_d,lt_rec,&correlation_parameter_d[5]);//cal*obs
							}
						}

						ishot++;
				}

						if(correlation_misfit==0)
						{
							cuda_cal_alpha_new_for_lsrtm<<<1, Block_Size>>>(beta_step_d,d_alpha1,d_alpha2,epsil_d,lt_rec,0);
						}

						else
						{
							///////////////////it is noted that  the first iteration is conventional LSRTM
							if(iter==0)
							{
								cuda_cal_alpha_new_for_lsrtm<<<1, Block_Size>>>(beta_step_d,d_alpha1,d_alpha2,epsil_d,lt_rec,0);
							}
							
							if(iter>0)////correlation
							{
								cuda_cal_alpha_new_for_correlation_lsrtm<<<1,1>>>(beta_step_d,correlation_parameter_d,0);
							}
						}
					
						/////for ELSRTM 
//////update res_shots:res_shot_x res_shot_x1
///////////////////////////////* update the res_shots according to previous res_shots, gradient/conjugate gradient  and estimated stepsize */
						ishot=0;
						while(ishot<shot_num)
						{	
							/////////read residuals	
							sprintf(filename,"./someoutput/bin/res_shot_x_%d_iter_%d",ishot+1,iter+1);
							fread_file_1d(shotgather,receiver_num,lt,filename);
							cudaMemcpy(res_shot_x_d,shotgather,lt_rec*sizeof(float),cudaMemcpyHostToDevice);

							sprintf(filename,"./someoutput/bin/res_shot_z_%d_iter_%d",ishot+1,iter+1);
							fread_file_1d(shotgather,receiver_num,lt,filename);
							cudaMemcpy(res_shot_z_d,shotgather,lt_rec*sizeof(float),cudaMemcpyHostToDevice);
			
						
							/////////read tmp shot
							sprintf(filename,"./someoutput/bin/tmp_shot_x_%d_iter_%d",ishot+1,iter+1);
							fread_file_1d(shotgather,receiver_num,lt,filename);
							cudaMemcpy(tmp_shot_x_d,shotgather,lt_rec*sizeof(float),cudaMemcpyHostToDevice);

							sprintf(filename,"./someoutput/bin/tmp_shot_z_%d_iter_%d",ishot+1,iter+1);
							fread_file_1d(shotgather,receiver_num,lt,filename);
							cudaMemcpy(tmp_shot_z_d,shotgather,lt_rec*sizeof(float),cudaMemcpyHostToDevice);


							/////////read cal shot
							sprintf(filename,"./someoutput/bin/cal_shot_x_%d_iter_%d",ishot+1,iter+1);
							fread_file_1d(shotgather,receiver_num,lt,filename);
							cudaMemcpy(cal_shot_x_d,shotgather,lt_rec*sizeof(float),cudaMemcpyHostToDevice);

							sprintf(filename,"./someoutput/bin/cal_shot_z_%d_iter_%d",ishot+1,iter+1);
							fread_file_1d(shotgather,receiver_num,lt,filename);
							cudaMemcpy(cal_shot_z_d,shotgather,lt_rec*sizeof(float),cudaMemcpyHostToDevice);



							cuda_update_shots<<<dimGrid_lt,dimBlock>>>(res_shot_x_d,cal_shot_x_d,beta_step_d,receiver_num,lt,0);
							cuda_update_shots<<<dimGrid_lt,dimBlock>>>(res_shot_z_d,cal_shot_z_d,beta_step_d,receiver_num,lt,0);

							/////////output residuals
							cudaMemcpy(shotgather,res_shot_x_d,lt_rec*sizeof(float),cudaMemcpyDeviceToHost);
							sprintf(filename,"./someoutput/bin/res_shot_x_%d_iter_%d",ishot+1,iter+2);
							write_file_1d(shotgather,lt_rec,filename);

							cudaMemcpy(shotgather,res_shot_z_d,lt_rec*sizeof(float),cudaMemcpyDeviceToHost);
							sprintf(filename,"./someoutput/bin/res_shot_z_%d_iter_%d",ishot+1,iter+2);
							write_file_1d(shotgather,lt_rec,filename);



							cuda_update_tmp_shots<<<dimGrid_lt,dimBlock>>>(tmp_shot_x_d,cal_shot_x_d,beta_step_d,receiver_num,lt,0);
							cuda_update_tmp_shots<<<dimGrid_lt,dimBlock>>>(tmp_shot_z_d,cal_shot_z_d,beta_step_d,receiver_num,lt,0);
							/////////for cross-correlation misfunction 2017年08月25日 星期五 09时28分54秒 
							/////////output tmp_cal
							cudaMemcpy(shotgather,tmp_shot_x_d,lt_rec*sizeof(float),cudaMemcpyDeviceToHost);
							sprintf(filename,"./someoutput/bin/tmp_shot_x_%d_iter_%d",ishot+1,iter+2);
							write_file_1d(shotgather,lt_rec,filename);

							cudaMemcpy(shotgather,tmp_shot_z_d,lt_rec*sizeof(float),cudaMemcpyDeviceToHost);
							sprintf(filename,"./someoutput/bin/tmp_shot_z_%d_iter_%d",ishot+1,iter+2);
							write_file_1d(shotgather,lt_rec,filename);////

							if(vsp_2!=0)
							{
								/////////read residuals	
								sprintf(filename,"./someoutput/bin/res_shot_x_%d_iter_%d_2",ishot+1,iter+1);
								fread_file_1d(shotgather,receiver_num,lt,filename);
								cudaMemcpy(res_shot_x_d_2,shotgather,lt_rec*sizeof(float),cudaMemcpyHostToDevice);

								sprintf(filename,"./someoutput/bin/res_shot_z_%d_iter_%d_2",ishot+1,iter+1);
								fread_file_1d(shotgather,receiver_num,lt,filename);
								cudaMemcpy(res_shot_z_d_2,shotgather,lt_rec*sizeof(float),cudaMemcpyHostToDevice);

								/////////read cal shot
								sprintf(filename,"./someoutput/bin/cal_shot_x_%d_iter_%d_2",ishot+1,iter+1);
								fread_file_1d(shotgather,receiver_num,lt,filename);
								cudaMemcpy(cal_shot_x_d_2,shotgather,lt_rec*sizeof(float),cudaMemcpyHostToDevice);

								sprintf(filename,"./someoutput/bin/cal_shot_z_%d_iter_%d_2",ishot+1,iter+1);
								fread_file_1d(shotgather,receiver_num,lt,filename);
								cudaMemcpy(cal_shot_z_d_2,shotgather,lt_rec*sizeof(float),cudaMemcpyHostToDevice);

								cuda_update_shots<<<dimGrid_lt,dimBlock>>>(res_shot_x_d_2,cal_shot_x_d_2,beta_step_d,receiver_num,lt,0);
								cuda_update_shots<<<dimGrid_lt,dimBlock>>>(res_shot_z_d_2,cal_shot_z_d_2,beta_step_d,receiver_num,lt,0);

								/////////output residuals
								cudaMemcpy(shotgather,res_shot_x_d_2,lt_rec*sizeof(float),cudaMemcpyDeviceToHost);
								sprintf(filename,"./someoutput/bin/res_shot_x_%d_iter_%d_2",ishot+1,iter+2);
								write_file_1d(shotgather,lt_rec,filename);

								cudaMemcpy(shotgather,res_shot_z_d_2,lt_rec*sizeof(float),cudaMemcpyDeviceToHost);
								sprintf(filename,"./someoutput/bin/res_shot_z_%d_iter_%d_2",ishot+1,iter+2);
								write_file_1d(shotgather,lt_rec,filename);
							}			
							ishot++;
						}
//////////////////////////iter_start!=0  restart 因为程序中断，需要重新开始！！！！！！！！！！！！！！！！！
						if(iter_start!=0)///////////read current result to restart
						{
							sprintf(filename,"./result/obj_niter-%d",iter);
							fread_file_1d(obj_niter_h,1,niter,filename);

							sprintf(filename,"./result/obj_niter1-%d",iter);
							fread_file_1d(obj_niter_h1,1,niter,filename);

							if(inversion_para==1||inversion_para==0)
							{
								sprintf(filename,"./result/result-lame1-%d",iter);
								fread_file_1d(wf_nxnz,nx,nz,filename);	
								cudaMemcpy(perturb_lame1_d,wf_nxnz,nxnz*sizeof(float),cudaMemcpyHostToDevice);

								sprintf(filename,"./result/result-lame2-%d",iter);
								fread_file_1d(wf_nxnz,nx,nz,filename);	
								cudaMemcpy(perturb_lame2_d,wf_nxnz,nxnz*sizeof(float),cudaMemcpyHostToDevice);

								sprintf(filename,"./result/result-den-%d",iter);
								fread_file_1d(wf_nxnz,nx,nz,filename);	
								cudaMemcpy(perturb_den_d,wf_nxnz,nxnz*sizeof(float),cudaMemcpyHostToDevice);
							}
						
							if(inversion_para==2||inversion_para==3)
							{
								sprintf(filename,"./result/result-vp-%d",iter);
								fread_file_1d(wf_nxnz,nx,nz,filename);	
								cudaMemcpy(perturb_vp_d,wf_nxnz,nxnz*sizeof(float),cudaMemcpyHostToDevice);

								sprintf(filename,"./result/result-vs-%d",iter);
								fread_file_1d(wf_nxnz,nx,nz,filename);	
								cudaMemcpy(perturb_vs_d,wf_nxnz,nxnz*sizeof(float),cudaMemcpyHostToDevice);

								sprintf(filename,"./result/result-density-%d",iter);
								fread_file_1d(wf_nxnz,nx,nz,filename);	
								cudaMemcpy(perturb_density_d,wf_nxnz,nxnz*sizeof(float),cudaMemcpyHostToDevice);
							}
						}
//////////////////////////iter_start!=0  restart 因为程序中断，需要重新开始！！！！！！！！！！！！！！！！！

///////////////////////////////* update the image model according to previous image model, gradient/conjugate gradient  and estimated stepsize */
					if(inversion_para==1||inversion_para==0)
					{
						/////for ELSRTM
						/*cuda_attenuation_after_lap_new<<<dimGrid_new,dimBlock>>>(all_conj_lame1_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
						cuda_attenuation_after_lap_new<<<dimGrid_new,dimBlock>>>(all_conj_lame2_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
						cuda_attenuation_after_lap_new<<<dimGrid_new,dimBlock>>>(all_conj_den_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);


						cuda_update_shots<<<dimGrid_new,dimBlock>>>(perturb_lame1_d,all_conj_lame1_d,beta_step_d,nx,nz,0);
//////update lame2
						cuda_update_shots<<<dimGrid_new,dimBlock>>>(perturb_lame2_d,all_conj_lame2_d,beta_step_d,nx,nz,0);		
//////update den
						cuda_update_shots<<<dimGrid_new,dimBlock>>>(perturb_den_d,all_conj_den_d,beta_step_d,nx,nz,0);*/
//////update lame1						
						cuda_update_shots_new<<<dimGrid_new,dimBlock>>>(perturb_lame1_d,all_conj_lame1_d,beta_step_d,nx,nz,0,precon_z2);
//////update lame2
						cuda_update_shots_new<<<dimGrid_new,dimBlock>>>(perturb_lame2_d,all_conj_lame2_d,beta_step_d,nx,nz,0,precon_z2);		
//////update den
						cuda_update_shots_new<<<dimGrid_new,dimBlock>>>(perturb_den_d,all_conj_den_d,beta_step_d,nx,nz,0,precon_z2);					
						/////for ELSRTM 
					}
		
					if(inversion_para==2||inversion_para==3)
					{
						/////for ELSRTM
						/*cuda_attenuation_after_lap_new<<<dimGrid_new,dimBlock>>>(all_conj_vp_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
						cuda_attenuation_after_lap_new<<<dimGrid_new,dimBlock>>>(all_conj_vs_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
						cuda_attenuation_after_lap_new<<<dimGrid_new,dimBlock>>>(all_conj_density_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);


						cuda_update_shots<<<dimGrid_new,dimBlock>>>(perturb_vp_d,all_conj_vp_d,beta_step_d,nx,nz,0);
//////update velocity1
						cuda_update_shots<<<dimGrid_new,dimBlock>>>(perturb_vs_d,all_conj_vs_d,beta_step_d,nx,nz,0);
//////update density
						cuda_update_shots<<<dimGrid_new,dimBlock>>>(perturb_density_d,all_conj_density_d,beta_step_d,nx,nz,0);*/
//////update velocity						
						cuda_update_shots_new<<<dimGrid_new,dimBlock>>>(perturb_vp_d,all_conj_vp_d,beta_step_d,nx,nz,0,precon_z2);
//////update velocity1
						cuda_update_shots_new<<<dimGrid_new,dimBlock>>>(perturb_vs_d,all_conj_vs_d,beta_step_d,nx,nz,0,precon_z2);
//////update density
						cuda_update_shots_new<<<dimGrid_new,dimBlock>>>(perturb_density_d,all_conj_density_d,beta_step_d,nx,nz,0,precon_z2);
						/////for ELSRTM 
					}
			
///////////////////////////////////////one iteration is over
					cudaMemcpy(beta_h,beta_d,3*sizeof(float),cudaMemcpyDeviceToHost);
					cudaMemcpy(beta_step_h,beta_step_d,3*sizeof(float),cudaMemcpyDeviceToHost);
					cudaMemcpy(epsil_h,epsil_d,4*sizeof(float),cudaMemcpyDeviceToHost);
					//cudaMemcpy(beta_h,beta_d,3*sizeof(float),cudaMemcpyDeviceToHost);

						////time consume
						warn("iterative times is =%d",iter+1);
						fprintf(logfile,"iterative times=%d\n",iter+1);

					if(correlation_misfit==0)
					{
						if(iter==0) 	
						{
							obj_niter_h1[iter]=obj_h[0];

							obj_niter_h[iter]=1.0;					
						}

						else		
						{
							obj_niter_h1[iter]=obj_h[0];

							obj_niter_h[iter]=obj_niter_h1[iter]/obj_niter_h1[0];
						}

						/* output important information at each FWI iteration */
						//warn("obj=%f  beta=%f  epsil=%f  alpha=%f",obj, beta, epsil, alpha);

						////normolized objection vaule
						warn("normlaized_obj=%f",obj_niter_h[iter]*100);
						fprintf(logfile,"normlaized_obj=%f\n",obj_niter_h[iter]);

						////objection vaule
						warn("obj=%f",obj_niter_h1[iter]);
						fprintf(logfile,"obj=%f\n",obj_niter_h1[iter]);
					}

					else
					{
						obj_niter_h[iter]=obj_h[0];

						////normolized objection vaule
						warn("normlaized_obj=%f",obj_niter_h[iter]*100);
						fprintf(logfile,"normlaized_obj=%f\n",obj_niter_h[iter]);

						////objection vaule
						warn("obj=%f",obj_niter_h[iter]);
						fprintf(logfile,"obj=%f\n",obj_niter_h[iter]);
					}


					////conjugated method and overall'step
					warn("beta_vp=%f  beta_vs=%f  beta_density=%f ", beta_h[0],beta_h[1],beta_h[2]);
					warn("overall'step=%f",beta_step_h[0]);
					
					fprintf(logfile,"overall'step=%f beta_vp=%f  beta_vs=%f  beta_density=%f\n",beta_step_h[0],beta_h[0],beta_h[1],beta_h[2]);

					cudaEventRecord(stop);/* record ending time */
  					cudaEventSynchronize(stop);
  					cudaEventElapsedTime(&mstimer, start, stop);
					totaltime+=mstimer*1e-3;

					warn("Programe is done, total time cost: %f (s)", totaltime);////////to current step has cost times
					fprintf(logfile,"iteration %d finished: %f (s)\n\n",iter+1, mstimer*1e-3);////////the current step  cost times

					//warn("epsil_vp=%f  epsil_step_vs=%f  epsil_step_density=%f ",epsil_h[0],epsil_h[1],epsil_h[2]);
					//warn("iteration %d finished: %f (s)",iter+1, mstimer*1e-3);
					//fprintf(logfile,"obj=%f  beta=%f  epsil=%f  alpha=%f\n",obj, beta, epsil, alpha);
					//fprintf(logfile,"iteration %d finished: %f (s)\n\n",iter+1, mstimer*1e-3);

////////////////output update lame coefficient
					if(inversion_para==1||inversion_para==0)
					{
						cudaMemcpy(wf_nxnz,perturb_lame1_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
						sprintf(filename,"./result/result-lame1-%d",iter+1);
						write_file_1d(wf_nxnz,nxnz,filename);
	
						cudaMemcpy(wf_nxnz,perturb_lame2_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
						sprintf(filename,"./result/result-lame2-%d",iter+1);
						write_file_1d(wf_nxnz,nxnz,filename);

						cudaMemcpy(wf_nxnz,perturb_den_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
						sprintf(filename,"./result/result-den-%d",iter+1);
						write_file_1d(wf_nxnz,nxnz,filename);
					}
////////////////output update velocity
					if(inversion_para==2||inversion_para==3)
					{
						cudaMemcpy(wf_nxnz,perturb_vp_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
						sprintf(filename,"./result/result-vp-%d",iter+1);
						write_file_1d(wf_nxnz,nxnz,filename);
	
						cudaMemcpy(wf_nxnz,perturb_vs_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
						sprintf(filename,"./result/result-vs-%d",iter+1);
						write_file_1d(wf_nxnz,nxnz,filename);

						cudaMemcpy(wf_nxnz,perturb_density_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
						sprintf(filename,"./result/result-density-%d",iter+1);
						write_file_1d(wf_nxnz,nxnz,filename);
					}

					sprintf(filename,"./result/obj_niter-%d",iter+1);
					write_file_1d(obj_niter_h,niter,filename);

					sprintf(filename,"./result/obj_niter1-%d",iter+1);
					write_file_1d(obj_niter_h1,niter,filename);

					fclose(logfile);////important

////////////////output update velocity and lame coefficient after laplace operator
					if(laplace==0)
					{
						if(inversion_para==2||inversion_para==3)
						{
							//////////1111111
							cudaMemcpy(wf_nxnz_d,perturb_vp_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);
							cuda_laplace<<<dimGrid_new,dimBlock>>>(perturb_vp_d,wf_nxnz_d,s_velocity_all_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,dx,dz,1,laplace);
							cuda_lap<<<dimGrid,dimBlock>>>(wf_nxnz_d,nx,nz,laplace);
							cuda_attenuation_after_lap_new2<<<dimGrid_new,dimBlock>>>(wf_nxnz_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
							cudaMemcpy(wf_nxnz,wf_nxnz_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
							sprintf(filename,"./result/result-vp1-%d",iter+1);
							write_file_1d(wf_nxnz,nxnz,filename);

							/////////2222222222
							cudaMemcpy(wf_nxnz_d,perturb_vs_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);
							cuda_laplace<<<dimGrid_new,dimBlock>>>(perturb_vs_d,wf_nxnz_d,s_velocity_all_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,dx,dz,1,laplace);
							cuda_lap<<<dimGrid_new,dimBlock>>>(wf_nxnz_d,nx,nz,laplace);
							cuda_attenuation_after_lap_new2<<<dimGrid_new,dimBlock>>>(wf_nxnz_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
							cudaMemcpy(wf_nxnz,wf_nxnz_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
							sprintf(filename,"./result/result-vs1-%d",iter+1);
							write_file_1d(wf_nxnz,nxnz,filename);

							//////////333333333
							cudaMemcpy(wf_nxnz_d,perturb_density_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);
							cuda_laplace<<<dimGrid_new,dimBlock>>>(perturb_density_d,wf_nxnz_d,s_velocity_all_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,dx,dz,1,laplace);
							cuda_lap<<<dimGrid_new,dimBlock>>>(wf_nxnz_d,nx,nz,laplace);
							cuda_attenuation_after_lap_new2<<<dimGrid_new,dimBlock>>>(wf_nxnz_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
							cudaMemcpy(wf_nxnz,wf_nxnz_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
							sprintf(filename,"./result/result-density1-%d",iter+1);
							write_file_1d(wf_nxnz,nxnz,filename);
						}

						if(inversion_para==0||inversion_para==1)
						{
							//////////1111111
							cudaMemcpy(wf_nxnz_d,perturb_lame1_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);
							cuda_laplace<<<dimGrid_new,dimBlock>>>(perturb_lame1_d,wf_nxnz_d,s_velocity_all_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,dx,dz,1,laplace);
							cuda_lap<<<dimGrid_new,dimBlock>>>(wf_nxnz_d,nx,nz,laplace);
							cuda_attenuation_after_lap_new2<<<dimGrid_new,dimBlock>>>(wf_nxnz_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
							cudaMemcpy(wf_nxnz,wf_nxnz_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
							sprintf(filename,"./result/result-lame11-%d",iter+1);
							write_file_1d(wf_nxnz,nxnz,filename);

							/////////2222222222
							cudaMemcpy(wf_nxnz_d,perturb_lame2_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);
							cuda_laplace<<<dimGrid_new,dimBlock>>>(perturb_lame2_d,wf_nxnz_d,s_velocity_all_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,dx,dz,1,laplace);
							cuda_lap<<<dimGrid_new,dimBlock>>>(wf_nxnz_d,nx,nz,laplace);
							cuda_attenuation_after_lap_new2<<<dimGrid_new,dimBlock>>>(wf_nxnz_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
							cudaMemcpy(wf_nxnz,wf_nxnz_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
							sprintf(filename,"./result/result-lame22-%d",iter+1);
							write_file_1d(wf_nxnz,nxnz,filename);

							//////////333333333
							cudaMemcpy(wf_nxnz_d,perturb_den_d,nxnz*sizeof(float),cudaMemcpyDeviceToDevice);
							cuda_laplace<<<dimGrid_new,dimBlock>>>(perturb_den_d,wf_nxnz_d,s_velocity_all_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,dx,dz,1,laplace);
							cuda_lap<<<dimGrid_new,dimBlock>>>(wf_nxnz_d,nx,nz,laplace);
							cuda_attenuation_after_lap_new2<<<dimGrid_new,dimBlock>>>(wf_nxnz_d,nx,nz,nx_append_new,nz_append,boundary_left,boundary_up,precon_z1,precon_z2);
							cudaMemcpy(wf_nxnz,wf_nxnz_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
							sprintf(filename,"./result/result-den1-%d",iter+1);
							write_file_1d(wf_nxnz,nxnz,filename);
						}
					}		
	}	
					write_file_1d(obj_niter_h,niter,"./result/obj_niter");

					write_file_1d(obj_niter_h1,niter,"./result/obj_niter1");

////////////////output update lame coefficient
					if(inversion_para==1||inversion_para==0)
					{
						cudaMemcpy(wf_nxnz,perturb_lame1_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
						sprintf(filename,"./result/result-lame1");
						write_file_1d(wf_nxnz,nxnz,filename);
	
						cudaMemcpy(wf_nxnz,perturb_lame2_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
						sprintf(filename,"./result/result-lame2");
						write_file_1d(wf_nxnz,nxnz,filename);

						cudaMemcpy(wf_nxnz,perturb_den_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
						sprintf(filename,"./result/result-den");
						write_file_1d(wf_nxnz,nxnz,filename);
					}					
////////////////output update velocity
					if(inversion_para==2||inversion_para==3)
					{
						cudaMemcpy(wf_nxnz,perturb_vp_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
						sprintf(filename,"./result/result-vp");
						write_file_1d(wf_nxnz,nxnz,filename);
	
						cudaMemcpy(wf_nxnz,perturb_vs_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
						sprintf(filename,"./result/result-vs");
						write_file_1d(wf_nxnz,nxnz,filename);

						cudaMemcpy(wf_nxnz,perturb_density_d,nxnz*sizeof(float),cudaMemcpyDeviceToHost);
						sprintf(filename,"./result/result-density");
						write_file_1d(wf,nxnz,filename);
					}
			
		//finish = clock();
		//time(&t2);
		//warn("time is = %f\n",difftime(t2,t1));
		//duration = (double)(finish - start)/CLOCKS_PER_SEC;
		//warn( "CUDA duration time is =%f seconds\n", duration );
		//warn("get the shot gather");
		logfile=fopen("log.txt","ab");//remember to free log file
		fprintf(logfile,"Programe is done, total time cost: %f (s)\n", totaltime);
		fclose(logfile);////important

		/* destroy timing varibles */
		cudaEventDestroy(start);
		cudaEventDestroy(stop);

		

		cudaFree(s_density_d);
		cudaFree(s_velocity_d);
		cudaFree(s_velocity1_d);

		cudaFree(obs_shot_x_d);
		cudaFree(obs_shot_z_d);
		cudaFree(cal_shot_x_d);
		cudaFree(cal_shot_z_d);
		cudaFree(res_shot_x_d);
		cudaFree(res_shot_z_d);

		cudaFree(attenuation_d);

		cudaFree(density_d);
		cudaFree(velocity_d);
		cudaFree(velocity1_d);
		
		cudaFree(coe_opt_d);
		cudaFree(coe_opt1_d);
		cudaFree(wavelet_d);

		cudaFree(vx1_d);
		cudaFree(vz1_d);
		cudaFree(txx1_d);
		cudaFree(tzz1_d);
		cudaFree(txz1_d);
		cudaFree(vx2_d);
		cudaFree(vz2_d);
		cudaFree(txx2_d);
		cudaFree(tzz2_d);
		cudaFree(txz2_d);

		/*cudaFree(vxu_d);
		cudaFree(vxd_d);
		cudaFree(vxr_d);
		cudaFree(vxl_d);
		cudaFree(vzu_d);
		cudaFree(vzd_d);
		cudaFree(vzr_d);
		cudaFree(vzl_d);*/

		cudaFree(rvx1_d);
		cudaFree(rvz1_d);
		cudaFree(rtxx1_d);
		cudaFree(rtzz1_d);
		cudaFree(rtxz1_d);
		cudaFree(rvx2_d);
		cudaFree(rvz2_d);
		cudaFree(rtxx2_d);
		cudaFree(rtzz2_d);
		cudaFree(rtxz2_d);
	
		//warn("free1\n");
	
		free1int(receiver_x_cord);
		
		//warn("free2\n");		

		free1int(source_x_cord);

		//warn("free3\n");

		free1float(shotgather);
		free1float(shotgather1);
		//warn("free4\n");

		free1float(wf_append);
		//warn("free5\n");

		free1float(wf);
		//warn("free6\n");

		free1float(attenuation);	
		//warn("free7\n");

		/*free1float(velocity);
		free1float(velocity1);
		free1float(density);

		free1float(s_density);
		free1float(s_velocity);
		free1float(s_velocity1);*/

		free1float(coe_opt);
		free1float(wavelet);

		warn("free_over\n");
		warn("***end***\n");
} 
