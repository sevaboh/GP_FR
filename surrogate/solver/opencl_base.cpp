/* Author:Vsevolod Bohaienko */
/*        3D kernel visualization module */
/* high level classes: basic opencl operation */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <string>
#include "opencl_class.h"
#define SWARN(str) printf("%s\n",str)
#define SERROR_INT(str,i) {printf("%s %d\n",str,i);exit(0);}
#define SERROR(str) {printf("%s\n",str);exit(0);}
int opencl_debug=0;
using namespace std;

OpenCL_buffer::OpenCL_buffer(cl_context *ct,cl_mem_flags flags,size_t sz,void * mem)
	{
	   cl_int err;
	   buffer = clCreateBuffer(ct[0],flags,size=sz,mem,&err);
	   if (err!=CL_SUCCESS) SERROR_INT("OpenCL failed to create buffer",err);
		if (opencl_debug)
	    {
		char str[1024];
		sprintf(str,"OpenCL buffer created: size %d, flags %d",(int)sz,(int)flags);
		SWARN(str);
	    }
	}
void OpenCL_buffer::realloc(cl_context *ct,cl_mem_flags flags,size_t sz,void * mem)
	{
	   cl_int err;
	   clReleaseMemObject(buffer); 
	   buffer = clCreateBuffer(ct[0],flags,size=sz,mem,&err);
	   if (err!=CL_SUCCESS) SERROR_INT("OpenCL failed to create buffer",err);
		if (opencl_debug)
	    {
		char str[1024];
		sprintf(str,"OpenCL buffer recreated: size %d, flags %d",(int)sz,(int)flags);
		SWARN(str);
	    }
	}
OpenCL_buffer::~OpenCL_buffer()
	{
	   clReleaseMemObject(buffer); 
	}
//////////////////////////////////////////////////////////
/////////////// hash func  //////////////////////////////
/*
  Name  : CRC-32
  Poly  : 0x04C11DB7    x^32 + x^26 + x^23 + x^22 + x^16 + x^12 + x^11 
                       + x^10 + x^8 + x^7 + x^5 + x^4 + x^2 + x + 1
  Init  : 0xFFFFFFFF
  Revert: true
  XorOut: 0xFFFFFFFF
  Check : 0xCBF43926 ("123456789")
  MaxLen: 268 435 455 байт (2 147 483 647 бит) - обнаружение
   одинарных, двойных, пакетных и всех нечетных ошибок
*/
unsigned int Crc32(const unsigned char *buf, size_t len)
{
    unsigned int crc_table[256];
    unsigned int crc; int i, j;
 
    for (i = 0; i < 256; i++)
    {
        crc = i;
        for (j = 0; j < 8; j++)
            crc = crc & 1 ? (crc >> 1) ^ 0xEDB88320UL : crc >> 1;
 
        crc_table[i] = crc;
    };
 
    crc = 0xFFFFFFFFUL;
 
    while (len--) 
        crc = crc_table[(crc ^ *buf++) & 0xFF] ^ (crc >> 8);
 
    return crc ^ 0xFFFFFFFFUL;
}
OpenCL_prg::OpenCL_prg(cl_context *ct,cl_device_id device_id, const char *source)
{
		int err;
		char *buffer;
		char *buffer2;
		size_t len;
	    hContext=ct;
		// try to find cached binary source
		{
			FILE *fi=fopen("res/opencl_cache","rb");
			if (fi)
			{
				char dev_name[1024];
				size_t l;
				int crc;
				int rcrc,rs;
				char *data=NULL;
				// form string to compare
				string s=string(source);
				clGetDeviceInfo(device_id,CL_DEVICE_NAME,1024,dev_name,&l);
				s+=string(dev_name);
				// get crc
				crc=Crc32((const unsigned char *)s.c_str(),s.length());
				// file format - hash-size-data
				// read file, search for crc
				while(fread(&rcrc,1,sizeof(int),fi))
				{
					fread(&rs,1,sizeof(int),fi);
					if (rcrc==crc)  // try to create program from binary source
					{
						data=new char[rs];
						size_t srs=rs;
						fread(data,1,rs,fi);
						hProgram = clCreateProgramWithBinary(hContext[0],1,&device_id,&srs,(const unsigned char **)&data,NULL,&err);
						if (!hProgram || err != CL_SUCCESS)
							goto end;
						err = clBuildProgram(hProgram, 0, NULL, NULL, NULL, NULL);
						if (err != CL_SUCCESS)
						{
							clReleaseProgram(hProgram);
							goto end;
						}
						delete [] data;
						fclose(fi);
						return;
					}
					else
						fseek(fi,rs,SEEK_CUR);
				}
end:
				if (data) delete [] data;
				fclose(fi);
			}
		}
		hProgram = clCreateProgramWithSource(hContext[0], 1, (const char **) &source, NULL, &err);
		if (!hProgram || err != CL_SUCCESS) SERROR_INT("OpenCL failed to Create program with source",err);
		err = clBuildProgram(hProgram, 0, NULL, NULL, NULL, NULL);
		if (err != CL_SUCCESS)
		{
			clGetProgramBuildInfo(hProgram, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
			buffer=new char[len+1];
			buffer2=new char[len+1024];
			clGetProgramBuildInfo(hProgram, device_id, CL_PROGRAM_BUILD_LOG, len, buffer, &len);
			sprintf(buffer2,"OpenCL failed to build program executable: %s\n", buffer);
			SERROR(buffer2);
		}
		// write binary to cache file
		FILE *fi=fopen("res/opencl_cache","ab");
		if (fi)
		{
			char dev_name[1024];
			size_t l;
			int crc;
			int rs;
			char *data;
			// form string to compare
			string s = string(source);
			clGetDeviceInfo(device_id, CL_DEVICE_NAME, 1024, dev_name, &l);
			s += string(dev_name);
			// get crc
			crc = Crc32((const unsigned char *)s.c_str(), s.length());
			// write data
			fwrite(&crc, 1, sizeof(int), fi);
			// get binary
			clGetProgramInfo(hProgram, CL_PROGRAM_BINARY_SIZES, sizeof(int), &rs, NULL);
			data = new char[rs];
			clGetProgramInfo(hProgram, CL_PROGRAM_BINARIES, sizeof(char *), &data, NULL);
			// write binary
			fwrite(&rs, 1, sizeof(int), fi);
			fwrite(data, 1, rs, fi);
			delete[] data;
			fclose(fi);
		}
}	
OpenCL_prg::~OpenCL_prg()
{
		clReleaseProgram(hProgram);
}
OpenCL_kernel::OpenCL_kernel(cl_context* ct, OpenCL_prg *hprog,const char *name)
	{
		int err;
		hProgram=&hprog->hProgram;
		hContext=ct;
		// create kernel
	    hKernel = clCreateKernel(hProgram[0], name, &err);
		if (!hKernel || err != CL_SUCCESS)	SERROR_INT("OpenCL failed to create kernel",err);
		if (opencl_debug)
		{
			nm=new char[strlen(name)+1];
			strcpy(nm,name);
			{
			char str[1024];
			sprintf(str,"OpenCL kernel created: name %s",name);
			SWARN(str);
			}
		}
	}
OpenCL_kernel::~OpenCL_kernel()
	{
	    clReleaseKernel(hKernel);
 	}
cl_int OpenCL_kernel::SetBufferArg(OpenCL_buffer *buf,int idx)
	{
		return clSetKernelArg(hKernel, idx, sizeof(cl_mem), (void *)&buf->buffer);
	}
cl_int OpenCL_kernel::SetArg(int idx,int size,void *buf)
	{
		return clSetKernelArg(hKernel, idx, size, buf);
	}
OpenCL_commandqueue::OpenCL_commandqueue(cl_context *ctxt,cl_device_id did,cl_command_queue_properties pr)
	{
	   cl_int err;
	   hContext=ctxt;
	   device_id=did;
	   props=pr;
	   if (opencl_debug) props|=CL_QUEUE_PROFILING_ENABLE;
	   hCmdQueue = clCreateCommandQueue(hContext[0], device_id, props, &err);
	   if (err!=CL_SUCCESS) SERROR_INT("OpenCL clCreateCommandQueue failed",err);
	}
OpenCL_commandqueue::~OpenCL_commandqueue()
	{
	   clReleaseCommandQueue(hCmdQueue);
	}
cl_event OpenCL_commandqueue::ExecuteKernel(OpenCL_kernel *krnl,int ndims,size_t *dims,size_t *ldims,int nwaitkernels,cl_event *evs,int noevent)
	{
		int err,local_ldims=0,i;
		size_t nthr;
		cl_event e,*ep=&e;
		// check for 0 dims
		for (int i=0;i<ndims;i++)
			if (dims[i]==0) return e;
		if (ldims==NULL)
		{
		    if (alloced_ldims_size<ndims)
		    {
		      if (alloced_ldims) delete [] alloced_ldims;
		      alloced_ldims_size=ndims;
		      alloced_ldims=new size_t[ndims];
		    }
	            ldims=alloced_ldims;
	  	    local_ldims=1;
		}
		// Get the maximum work-group size for executing the kernel on the device
		if (got_local==0)
		{
    		    err = clGetKernelWorkGroupInfo(krnl->hKernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);		    
		    if (err != CL_SUCCESS)	SERROR_INT("OpenCL clGetKernelWorkGroupInfo Failed",err);
		    got_local=1;
		}
		// adjust work-group size
		if (local_ldims==1)
		{
			int j=0;
			// initial set and adjust dims%ldims
			for (i=0;i<ndims;i++) ldims[i]=((dims[i]<local)?dims[i]:local);
			for (i=0;i<ndims;i++) 
			  while (dims[i]%ldims[i]) ldims[i]--;
			// adjust nthreads
			nthr=1;
			for (i=0;i<ndims;i++) nthr*=ldims[i];
			if (nthr>local)
			{
				do
				{
					if (ldims[j]<2) j=((j+1)%ndims);
					ldims[j]--;
					while (dims[j]%ldims[j]) ldims[j]--;
					j=((j+1)%ndims);
					nthr=1;
					for (i=0;i<ndims;i++) nthr*=ldims[i];
				}
				while (nthr>local);
			}
		}
		// execute kernel
		{
			// clear nulls from event list
			int newevssize=nwaitkernels;
			cl_event *evs0=evs;
			for (i=0;i<nwaitkernels;i++)
				if (evs[i]==NULL)
					newevssize--;
			int j=0;
			for (i=0;i<newevssize;i++)
			{
				while (evs[j]==NULL) j++;
				evs[i]=evs[j];
				j++;
			}
			if (newevssize==0) evs0=NULL;
			if (noevent) ep=NULL;
			if (opencl_debug)
			{
				char str[1024];
				sprintf(str,"OpenCL kernel is scheduled for execution (waiting %d events)",newevssize);
	    		SWARN(str);
	    		for (i=0;i<ndims;i++)
	    		{
	    		  sprintf(str,"dim %d - (%d,%d)",i,dims[i],ldims[i]);
	    		  SWARN(str);
	    		}
			}
			err = clEnqueueNDRangeKernel(hCmdQueue, krnl->hKernel, ndims, NULL, dims, ldims, newevssize, evs0, ep);
			if (opencl_debug)
				Finish();
		}
		if (err != CL_SUCCESS) SERROR_INT("OpenCL clEnqueueNDRangeKernel Failed",err);
		return e;
	}
void OpenCL_commandqueue::ReleaseEvent(cl_event e)
{
	clWaitForEvents(1,&e);
	if (props&CL_QUEUE_PROFILING_ENABLE) // print profiling info
	{
		cl_int err;
		cl_ulong info[4];
		err=clGetEventProfilingInfo(e,CL_PROFILING_COMMAND_QUEUED,sizeof(cl_ulong),&info[0],NULL);
		err|=clGetEventProfilingInfo(e,CL_PROFILING_COMMAND_SUBMIT,sizeof(cl_ulong),&info[1],NULL);
		err|=clGetEventProfilingInfo(e,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&info[2],NULL);
		err|=clGetEventProfilingInfo(e,CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&info[3],NULL);
		if (err!=CL_SUCCESS) SERROR_INT("OpenCL cGetEventProfilingInfo failed",err);
		{
			char str[4096];
			sprintf(str,"OpenCL profiling info for event %llu - (%llu,%llu,%llu,%llu)",(cl_ulong)(e),info[0],info[1],info[2],info[3]);
			SWARN(str);
		}
	}
	clReleaseEvent(e);
}
cl_int OpenCL_commandqueue::EnqueueBuffer(OpenCL_buffer *b,void *mem,int offset,int size)
	{
		if (size==0) size=b->size;
		return clEnqueueReadBuffer(hCmdQueue, b->buffer, CL_TRUE, offset, size, mem, 0, NULL, NULL);
	}
cl_int OpenCL_commandqueue::EnqueueWriteBuffer(OpenCL_buffer *b,void *mem,int offset,int size)
	{
		if (size==0) size=b->size;
		return clEnqueueWriteBuffer(hCmdQueue, b->buffer, CL_TRUE, offset, size, mem, 0, NULL, NULL);
	}
void OpenCL_commandqueue::Finish()
	{
	   clFinish(hCmdQueue);
	}
OpenCL_program::OpenCL_program(int gpu)
   {
	    int err;
		char buffer[2048];
		char buffer2[2048];
		size_t len;
		cl_platform_id *pids;
		cl_uint np,cp;		
		err = clGetPlatformIDs( 1, &platform, &np );
		if (err != CL_SUCCESS) SERROR_INT("OpenCL failed to get platform IDs",err);
		pids=new cl_platform_id[np];
		err = clGetPlatformIDs( np, pids, NULL );
		if (err != CL_SUCCESS) SERROR_INT("OpenCL failed to get platform IDs",err);
		cp=0;
a10:
		platform=pids[cp++];
		err = clGetDeviceIDs(platform, gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_ALL, 0, NULL, &n_devices);
		if (err != CL_SUCCESS)
		{
			if (gpu==1)
			{
				if (cp==np)
				{
				    gpu=0;
				    cp=0;
				}
				goto a10;
			}			
			SERROR_INT("OpenCL failed to get number of devices",err);
		}
		device_ids=new cl_device_id[n_devices];
		err = clGetDeviceIDs(platform, gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_ALL, n_devices, device_ids, NULL);
		if (err != CL_SUCCESS) SERROR_INT("OpenCL failed to get device ID",err);
		if (opencl_debug)
		for (int i=0;i<n_devices;i++)
		{
			err = clGetDeviceInfo(device_ids[i], CL_DEVICE_NAME, sizeof(buffer), buffer, &len);
			sprintf(buffer2,"CL_DEVICE_NAME: %s\n", buffer);
			SWARN(buffer2);
			err = clGetDeviceInfo(device_ids[i], CL_DEVICE_VENDOR, sizeof(buffer), buffer, &len);
			sprintf(buffer2,"CL_DEVICE_VENDOR: %s\n", buffer);
			SWARN(buffer2);
		}
		// create OpenCL device & context
		hContext = clCreateContext(0, n_devices, device_ids, NULL, NULL, &err);
		if (err != CL_SUCCESS) SERROR_INT("OpenCL failed to create context",err);
   }
int OpenCL_program::get_ndevices()
{
	return (int)n_devices;
}
cl_device_type OpenCL_program::get_device_type(int d)
{
	cl_device_type ret;
	cl_int err=clGetDeviceInfo(device_ids[d],CL_DEVICE_TYPE,sizeof(cl_device_type),&ret,0);
	if (err != CL_SUCCESS) SERROR_INT("OpenCL failed to get device type",err);
	return ret;
}
OpenCL_program::~OpenCL_program()
   {
	   clReleaseContext(hContext);
	   delete [] device_ids;
   }
OpenCL_buffer *OpenCL_program::create_buffer(cl_mem_flags flags,size_t sz,void * mem,OpenCL_buffer *b)
   {
	    if (b)
	    {
		b->realloc(&hContext,flags,sz,mem);
		return b;
	    }
	   OpenCL_buffer *buf=new OpenCL_buffer(&hContext,flags,sz,mem);
	   return buf;
   }
OpenCL_prg *OpenCL_program::create_program(const char *src)
   {
	   OpenCL_prg *prg=new OpenCL_prg(&hContext,device_ids[0],src);
	   return prg;
   }
OpenCL_kernel *OpenCL_program::create_kernel(OpenCL_prg *prog,const char *name)
   {
	   OpenCL_kernel *krnl=new OpenCL_kernel(&hContext,prog,name);
	   return krnl;
   }
OpenCL_commandqueue *OpenCL_program::create_queue(int device,cl_command_queue_properties props)
   {
	   device=device%n_devices;
	   OpenCL_commandqueue *queue=new OpenCL_commandqueue(&hContext,device_ids[device],props);
	   return queue;
   }
