// fractured media equation for P - d^a P/dt=ae d^b p/dx+tau_r d^a(d^b p/dx)/dt+q
// C equation - d^a C/dt = D d^b C/ dx- d/dx( v C)+qc, v=(k/mu)*D^b (p+tau_r D^a p)
// -- finite-difference solution
// -- short memory for calculating fractional derivatives
// -- linearization to three-diagonal system + Pickard iteration
// -- testing: analytic solution for (P,C) system
// -- opencl code with parallel reduction for solving threediagonal systems
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <cfloat>
#include <sstream>
#include <iostream>
#include <vector>
//////////////////////////////////////
//////////////////////////////////////
//////////////////////////////////////

// Note that the functions Gamma and LogGamma are mutually dependent.
double Gamma
(
    double x    // We require x > 0
);

double LogGamma
(
    double x    // x must be positive
)
{
	if (x <= 0.0)
	{
		std::stringstream os;
        os << "Invalid input argument " << x <<  ". Argument must be positive.";
        throw std::invalid_argument( os.str() ); 
	}

    if (x < 12.0)
    {
        return log(fabs(Gamma(x)));
    }

	// Abramowitz and Stegun 6.1.41
    // Asymptotic series should be good to at least 11 or 12 figures
    // For error analysis, see Whittiker and Watson
    // A Course in Modern Analysis (1927), page 252

    static const double c[8] =
    {
		 1.0/12.0,
		-1.0/360.0,
		1.0/1260.0,
		-1.0/1680.0,
		1.0/1188.0,
		-691.0/360360.0,
		1.0/156.0,
		-3617.0/122400.0
    };
    double z = 1.0/(x*x);
    double sum = c[7];
    for (int i=6; i >= 0; i--)
    {
        sum *= z;
        sum += c[i];
    }
    double series = sum/x;

    static const double halfLogTwoPi = 0.91893853320467274178032973640562;
    double logGamma = (x - 0.5)*log(x) - x + halfLogTwoPi + series;    
	return logGamma;
}
double Gamma
(
    double x    // We require x > 0
)
{
	if (x <= 0.0)
	{
		std::stringstream os;
        os << "Invalid input argument " << x <<  ". Argument must be positive.";
        throw std::invalid_argument( os.str() ); 
	}

    // Split the function domain into three intervals:
    // (0, 0.001), [0.001, 12), and (12, infinity)

    ///////////////////////////////////////////////////////////////////////////
    // First interval: (0, 0.001)
	//
	// For small x, 1/Gamma(x) has power series x + gamma x^2  - ...
	// So in this range, 1/Gamma(x) = x + gamma x^2 with error on the order of x^3.
	// The relative error over this interval is less than 6e-7.

	const double gamma = 0.577215664901532860606512090; // Euler's gamma constant

    if (x < 0.001)
        return 1.0/(x*(1.0 + gamma*x));

    ///////////////////////////////////////////////////////////////////////////
    // Second interval: [0.001, 12)
    
	if (x < 12.0)
    {
        // The algorithm directly approximates gamma over (1,2) and uses
        // reduction identities to reduce other arguments to this interval.
		
		double y = x;
        int n = 0;
        bool arg_was_less_than_one = (y < 1.0);

        // Add or subtract integers as necessary to bring y into (1,2)
        // Will correct for this below
        if (arg_was_less_than_one)
        {
            y += 1.0;
        }
        else
        {
            n = static_cast<int> (floor(y)) - 1;  // will use n later
            y -= n;
        }

        // numerator coefficients for approximation over the interval (1,2)
        static const double p[] =
        {
            -1.71618513886549492533811E+0,
             2.47656508055759199108314E+1,
            -3.79804256470945635097577E+2,
             6.29331155312818442661052E+2,
             8.66966202790413211295064E+2,
            -3.14512729688483675254357E+4,
            -3.61444134186911729807069E+4,
             6.64561438202405440627855E+4
        };

        // denominator coefficients for approximation over the interval (1,2)
        static const double q[] =
        {
            -3.08402300119738975254353E+1,
             3.15350626979604161529144E+2,
            -1.01515636749021914166146E+3,
            -3.10777167157231109440444E+3,
             2.25381184209801510330112E+4,
             4.75584627752788110767815E+3,
            -1.34659959864969306392456E+5,
            -1.15132259675553483497211E+5
        };

        double num = 0.0;
        double den = 1.0;
        int i;

        double z = y - 1;
        for (i = 0; i < 8; i++)
        {
            num = (num + p[i])*z;
            den = den*z + q[i];
        }
        double result = num/den + 1.0;

        // Apply correction if argument was not initially in (1,2)
        if (arg_was_less_than_one)
        {
            // Use identity gamma(z) = gamma(z+1)/z
            // The variable "result" now holds gamma of the original y + 1
            // Thus we use y-1 to get back the orginal y.
            result /= (y-1.0);
        }
        else
        {
            // Use the identity gamma(z+n) = z*(z+1)* ... *(z+n-1)*gamma(z)
            for (i = 0; i < n; i++)
                result *= y++;
        }

		return result;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Third interval: [12, infinity)

    if (x > 171.624)
    {
		// Correct answer too large to display. Force +infinity.
		double temp = DBL_MAX;
		return temp*2.0;
    }

    return exp(LogGamma(x));
}
//////////////////////////////////////
//////////////////////////////////////
//////////////////////////////////////
// variables
double beta; // 1<beta<2 - space derivative
double alpha; // 0<alpha<1 - time derivative
double tau;
int tstep;
// constants
int M=100;
double tau_r=0.01; // fracture parameter>0 - 0 - no fractures 
double ae=0.1; // filtration coefficient
double D=0.025; // constant diffusion coefficient
double kmu=0.25;
double h=1.0/(M+1.0);
double decay_rate=0; // -dr C term
// inputs (testing==0)
double u_inp_coef=1.0;
double u_inp_a=6;
double u_inp_b=0;
double c_inp_coef=1.0;
double c_inp_duration=0.5;
// arrays
double *U=NULL,*CU=NULL,*G=NULL,*CG=NULL;
std::vector<double *> Us,CUs;
//////////////////////////////////////
//////////////////////////////////////
//////////////////////////////////////
double tau_a,h_beta,h_bm1;
double g1a,g3b,g4,g4b,g5b,g2a;

int use_opencl=1;
int testing=0;
int bottom_cond=0; // 0 - f=0, 1 - d^{b-1}f/dx=0
// exact solution
double q0(double x,double t) { return x*x*(1.0-x)*g1a-ae*pow(x,2.0-beta)*((2.0/g3b)-(g4*x/g4b))*(1.0+pow(t,alpha)+tau_r*g1a);}
double u0(double x,double t) { return x*x*(1.0-x)*(1.0+pow(t,alpha));}
double qc0(double x,double t) { return x*x*(1.0-x)*g1a-D*pow(x,2.0-beta)*((2.0/g3b)-(g4*x/g4b))*(1.0+pow(t,alpha))+
				kmu*(1.0+pow(t,alpha))*(1.0+pow(t,alpha)+tau_r*g1a)*(((2.0/g4b)-(g4*x/g5b))*((5-beta)*pow(x,4.0-beta)-
				(6-beta)*pow(x,5.0-beta))-pow(x,5.0-beta)*(1-x)*(g4/g5b)); }
double uc0(double x,double t) { return x*x*(1.0-x)*(1.0+pow(t,alpha)); }
// initial-boundary value problem 
double q1(double x,double t) { return 0.0; }
double u1(double x,double t) { return u_inp_coef*(1-cos(u_inp_a*t+u_inp_b)); }
double qc1(double x,double t) { return 0.0; }
double uc1(double x,double t) { if ((t<c_inp_duration)&&(x==0.0)) return c_inp_coef*1.0; return 0.0; }
// proxy
double q(double x,double t) { return (testing==0)?q1(x,t):q0(x,t); }
double u(double x,double t) { return (testing==0)?u1(x,t):u0(x,t); }
double qc(double x,double t) { return (testing==0)?qc1(x,t):qc0(x,t); }
double uc(double x,double t) { return (testing==0)?uc1(x,t):uc0(x,t); }

double bk(double k) { if (k==0) return 1.0; else return pow(k+1.0,1.0-alpha)-pow(k,1.0-alpha); }
double gs(int s,double beta) 
{
	if (s==0) 
		return 1.0;
	return ((s-beta-1.0)/(double)s)*gs(s-1,beta);
}
double AU,BU,SU; // constant three-diagonal matrix for P
double CAU,CBU,CSU; // constant three-diagonal matrix for C
std::vector<double> bks;
double fr_err=1e-10; // for short-memory calculations
double gs_err=1e-20; // for Gauss-Seidel algorithm
int gs_maxiter=10000;
double pick_err=1e-12; // for Pickard iterations
int pick_maxiter=1000;
double FU(int i) 
{
    double sum1=0,sum2=0,fij=0;
    // d^a/dt
    if ((tstep>=1)&&(alpha!=1.0))
	for (int k=1;k<=tstep;k++)
	{
	    double v=bks[k]*(Us[tstep+1-k][i]-Us[tstep-k][i]);
	    sum1+=v;
	    if (fabs(bks[k])<fr_err)
		break;
	}
    // d^ad^b/dx/dt
    if (tau_r!=0.0)
    {
	for (int k=0;k<=tstep;k++)
	{
	    double sum=0;
	    for (int s=0;s<=i+1;s++)
	    {
		double v=G[s]*(((k==0)?((s<3)?0:U[i-s+1]):Us[tstep+1-k][i-s+1])-Us[tstep-k][i-s+1]);
		sum+=v;
		if (fabs(G[s])<fr_err)
		    break;
	    }
	    double v=bks[k]*sum;
	    fij+=v;
	    if (fabs(bks[k])<fr_err)
		break;
	}
	fij/=tau_a*h_beta*g2a;
    }
    //d^b/dx
    if (i>=2)
	for (int s=3;s<=i+1;s++)
	{
	    double v=G[s]*U[i-s+1];
	    sum2+=v;
	    if (fabs(G[s])<fr_err)
		break;
	}
    return ((-Us[tstep][i]+sum1)/(tau_a*g2a))-(ae*sum2/h_beta)-(ae*tau_r*fij)-q(i*h,tau*tstep);
}
double FCU(int i) 
{
    double sum1=0,sum2=0,sum3=0,vi=0,vim1=0,si=0,sim1=0;
    // d^a/dt
    if ((tstep>=1)&&(alpha!=1.0))
	for (int k=1;k<=tstep;k++)
	{
	    double v=bks[k]*(CUs[tstep+1-k][i]-CUs[tstep-k][i]);
	    sum1+=v;
	    if (fabs(bks[k])<fr_err)
		break;
	}
    // d^b/dx
    if (i>=2)
	for (int s=3;s<=i+1;s++)
	{
	    double v=G[s]*CU[i-s+1];
	    sum2+=v;
	    if (fabs(G[s])<fr_err)
		break;
	}
    // velocities
    for (int s=0;s<=i+1;s++)
    {
		vi+=CG[s]*U[i-s+1];
		if (s<=i) vim1+=CG[s]*U[i-s];
		if (fabs(CG[s])<fr_err)
			break;
    }
    vi/=h_bm1;
    vim1/=h_bm1;
    if (tau_r!=0.0)
    {
		for (int k=0;k<=tstep;k++)
		{
			double s1=0,s2=0;
			for (int s=0;s<=i+1;s++)
			{
				s1+=CG[s]*(((k==0)?U[i-s+1]:Us[tstep+1-k][i-s+1])-Us[tstep-k][i-s+1]);
				if (s<=i) s2+=CG[s]*(((k==0)?U[i-s]:Us[tstep+1-k][i-s])-Us[tstep-k][i-s]);
				if (fabs(CG[s])<fr_err)
					break;
			}
			si+=bks[k]*s1;
			sim1+=bks[k]*s2;
			if (fabs(bks[k])<fr_err)
				break;
		}
		si/=tau_a*h_bm1*g2a/tau_r;
		sim1/=tau_a*h_bm1*g2a/tau_r;
    }
    vi=-kmu*(vi+si);
    vim1=-kmu*(vim1+sim1);
    sum3=(vi*CU[i]-vim1*CU[i-1])/h; 
    return ((-CUs[tstep][i]+sum1)/(tau_a*g2a))-(D*sum2/h_beta)-sum3-qc(i*h,tau*tstep)+decay_rate*CU[i];
}
// Gauss-Seidel
double *Fs=NULL,*Rp=NULL,*ssU=NULL;
double *CFs=NULL,*CRp=NULL,*ssCU=NULL;
void solve_alloc()
{
	if (Fs==NULL) 
	{
	    Fs=new double[M+2];
	    Rp=new double[M+2];
	    ssU=new double[M+2];
	    CFs=new double[M+2];
	    CRp=new double[M+2];
	    ssCU=new double[M+2];
	}
}
int iteration_counter=0;
void solveU2()
{
	double err2;
	int iter2=0;
	solve_alloc();
	// Pickard iteration
	do
	{
	    memcpy(ssU,U,(M+2)*sizeof(double));
	    memcpy(ssCU,CU,(M+2)*sizeof(double));
#pragma omp parallel for
	    for (int i=1;i<=((M/2)+1);i++)
	    {
		Fs[i]=FU(i);
	        Fs[M-i+1]=FU(M-i+1);
		CFs[i]=FCU(i);
	        CFs[M-i+1]=FCU(M-i+1);
	    }
	    double err=0.0;
	    int iter=0;
	    if (bottom_cond==0) 
		U[M+1]=u((M+1)*h,(tstep+1)*tau);
	    else
	    {
		double sum3=0;
	        for (int s=1;s<=M+1;s++)
		{
			double v=CG[s]*U[M+1-s];
			sum3+=v;
			if (fabs(CG[s])<fr_err)
				break;
		}
		U[M+1]=-sum3/CG[0];
	    }
	    U[0]=u(0,(tstep+1)*tau);
	    if (bottom_cond==0) 
		CU[M+1]=uc((M+1)*h,(tstep+1)*tau);
	    else
	    {
		double sum3=0;
	        for (int s=1;s<=M+1;s++)
		{
			double v=CG[s]*CU[M+1-s];
			sum3+=v;
			if (fabs(CG[s])<fr_err)
				break;
		}
		CU[M+1]=-sum3/CG[0];
	    }
	    CU[0]=uc(0,(tstep+1)*tau);
	    do
	    {
		// P equation
		Rp[0]=Rp[M+1]=0.0;
		for (int i=1;i<=M;i++) Rp[i]=Fs[i]-AU*U[i+1];
		for (int i=1;i<=M;i++) U[i]=-(Rp[i]-BU*U[i-1])/SU;
		// C equation
		CRp[0]=CRp[M+1]=0.0;
		for (int i=1;i<=M;i++) CRp[i]=CFs[i]-CAU*CU[i+1];
		for (int i=1;i<=M;i++) CU[i]=-(CRp[i]-CBU*CU[i-1])/CSU;

		// error
		err=0.0;
		for (int i=1;i<=M;i++) 
		{
		    // P
		    double l=AU*U[i+1]-SU*U[i]+BU*U[i-1];
		    err+=(Fs[i]-l)*(Fs[i]-l);
		    // C
		    double cl=CAU*CU[i+1]-CSU*CU[i]+CBU*CU[i-1];
		    err+=(CFs[i]-cl)*(CFs[i]-cl);
		}
		err/=2*M;
		if ((iter++)==gs_maxiter) {printf("GAUSS-SEIDEL FAULT %g\n",err); break;}
//		printf("%d %g\n",iter,err);
	    }
	    while (err>gs_err);
	    err2=0.0;
	    for (int i=1;i<=M;i++) 
	        err2+=(U[i]-ssU[i])*(U[i]-ssU[i])+(CU[i]-ssCU[i])*(CU[i]-ssCU[i]);
	    err2/=2*M;
	    if ((iter2++)==pick_maxiter) {printf("PICKARD ITERATION FAULT %g\n",err2); break;}
//	    printf("%g %g %g - %g %g %g - %d %g - %d %g - %g %g\n",AU,BU,SU,CAU,CBU,CSU,iter,err,iter2,err2, U[1],CU[1]);
	    iteration_counter++;
	}
	while (err2>pick_err);
}
/////////////////////////////////////////////////////////////
//////////////// opencl solver //////////////////////////////
/////////////////////////////////////////////////////////////
#ifdef USE_OPENCL
#include "opencl_class.h"
int double_ext=1;
#define GS 32
const char *input_opencl_text = 
(char *)"#pragma OPENCL EXTENSION %s : enable \n\
#define GS %d\n\
// exact solution \n\
double q0(double x,double t,double alpha,double beta,double tau_r,double ae,double D,double kmu,double g4,double g1a,double g5b,double g3b,double g4b)\n\
{ return x*x*(1.0-x)*g1a-ae*pow(x,2.0-beta)*((2.0/g3b)-(g4*x/g4b))*(1.0+pow(t,alpha)+tau_r*g1a); }\n\
double u0(double x,double t,double alpha,double beta,double tau_r,double ae,double D,double kmu,double g4,double g1a,double g5b,double g3b,double g4b)\n\
{ return x*x*(1.0-x)*(1.0+pow(t,alpha)); }\n\
double qc0(double x,double t,double alpha,double beta,double tau_r,double ae,double D,double kmu,double g4,double g1a,double g5b,double g3b,double g4b)\n\
{ return x*x*(1.0-x)*g1a-D*pow(x,2.0-beta)*((2.0/g3b)-(g4*x/g4b))*(1.0+pow(t,alpha))+\n\
kmu*(1.0+pow(t,alpha))*(1.0+pow(t,alpha)+tau_r*g1a)*(((2.0/g4b)-(g4*x/g5b))*((5-beta)*pow(x,4.0-beta)-(6-beta)*pow(x,5.0-beta))-pow(x,5.0-beta)*(1-x)*(g4/g5b)); }\n\
double uc0(double x,double t,double alpha,double beta,double tau_r,double ae,double D,double kmu,double g4,double g1a,double g5b,double g3b,double g4b)\n\
{ return x*x*(1.0-x)*(1.0+pow(t,alpha)); }\n\
// initial-boundary value problem\n\
double q1(double x,double t,double alpha,double beta,double tau_r,double ae,double D,double kmu,double g4,double g1a,double g5b,double g3b,double g4b)\n\
{ return 0.0; }\n\
double u1(double x,double t,double alpha,double beta,double tau_r,double ae,double D,double kmu,double g4,double g1a,double g5b,double g3b,double g4b)\n\
{ return %g*(1-cos(%g*t+%g)); }\n\
double qc1(double x,double t,double alpha,double beta,double tau_r,double ae,double D,double kmu,double g4,double g1a,double g5b,double g3b,double g4b)\n\
{ return 0.0; }\n\
double uc1(double x,double t,double alpha,double beta,double tau_r,double ae,double D,double kmu,double g4,double g1a,double g5b,double g3b,double g4b)\n\
{ if ((t<%g)&&(x==0.0)) return %g*1.0; return 0.0; }\n\
// proxy\n\
double q(double x,double t,int testing, double alpha,double beta,double tau_r,double ae,double D,double kmu,double g4,double g1a,double g5b,double g3b,double g4b)\n\
{ return (testing==0)?q1(x,t,alpha,beta,tau_r,ae,D,kmu,g4,g1a,g5b,g3b,g4b):q0(x,t,alpha,beta,tau_r,ae,D,kmu,g4,g1a,g5b,g3b,g4b); }\n\
double u(double x,double t,int testing, double alpha,double beta,double tau_r,double ae,double D,double kmu,double g4,double g1a,double g5b,double g3b,double g4b)\n\
{ return (testing==0)?u1(x,t,alpha,beta,tau_r,ae,D,kmu,g4,g1a,g5b,g3b,g4b):u0(x,t,alpha,beta,tau_r,ae,D,kmu,g4,g1a,g5b,g3b,g4b); }\n\
double qc(double x,double t,int testing, double alpha,double beta,double tau_r,double ae,double D,double kmu,double g4,double g1a,double g5b,double g3b,double g4b)\n\
{ return (testing==0)?qc1(x,t,alpha,beta,tau_r,ae,D,kmu,g4,g1a,g5b,g3b,g4b):qc0(x,t,alpha,beta,tau_r,ae,D,kmu,g4,g1a,g5b,g3b,g4b); }\n\
double uc(double x,double t,int testing, double alpha,double beta,double tau_r,double ae,double D,double kmu,double g4,double g1a,double g5b,double g3b,double g4b)\n\
{ return (testing==0)?uc1(x,t,alpha,beta,tau_r,ae,D,kmu,g4,g1a,g5b,g3b,g4b):uc0(x,t,alpha,beta,tau_r,ae,D,kmu,g4,g1a,g5b,g3b,g4b); }\n\
double FU(int t0,int t1,int i,int tstep,int testing, double alpha,double beta,double tau_r,double ae,double D,double kmu,double g4,double g1a,double g5b,double g3b,double g4b,double tau_a,double h_beta,double g2a,double fr_err,__global double *Us,__global double *U,__global double *bks,__global double *G,int M,double tau,double h) \n\
{\n\
    double sum1=0,sum2=0,fij=0;\n\
    // d^a/dt\n\
    if ((tstep>=1)&&(alpha!=1.0))\n\
	if (((t0==0)?1:t0)<=t1)\n\
	for (int k=((t0==0)?1:t0);k<=t1;k++)\n\
	{\n\
	    double v=bks[k]*(Us[(M+2)*(tstep+1-k)+i]-Us[(M+2)*(tstep-k)+i]);\n\
	    sum1+=v;\n\
	    if (fabs(bks[k])<fr_err)\n\
		break;\n\
	}\n\
    // d^ad^b/dx/dt\n\
    if (tau_r!=0.0)\n\
    {\n\
	if (t0<=t1)\n\
	for (int k=t0;k<=t1;k++)\n\
	{\n\
	    double sum=0;\n\
	    for (int s=0;s<=i+1;s++)\n\
	    {\n\
		double v=G[s]*(((k==0)?((s<3)?0:U[i-s+1]):Us[(M+2)*(tstep+1-k)+i-s+1])-Us[(M+2)*(tstep-k)+i-s+1]);\n\
		sum+=v;\n\
		if (fabs(G[s])<fr_err)\n\
		    break;\n\
	    }\n\
	    double v=bks[k]*sum;\n\
	    fij+=v;\n\
	    if (fabs(bks[k])<fr_err)\n\
		break;\n\
	}\n\
	fij/=tau_a*h_beta*g2a;\n\
    }\n\
    //d^b/dx\n\
    if ((i>=2)&&(t0==0))\n\
	for (int s=3;s<=i+1;s++)\n\
	{\n\
	    double v=G[s]*U[i-s+1];\n\
	    sum2+=v;\n\
	    if (fabs(G[s])<fr_err)\n\
		break;\n\
	}\n\
    double ret=(sum1/(tau_a*g2a))-(ae*tau_r*fij);\n\
    if (t0==0) ret+=((-Us[(M+2)*tstep+i])/(tau_a*g2a))-(ae*sum2/h_beta)-q(i*h,tau*tstep,testing,alpha,beta,tau_r,ae,D,kmu, g4,g1a,g5b,g3b,g4b);\n\
    return ret;\n\
}\n\
double FCU(int t0,int t1,int i,int tstep,int testing, double alpha,double beta,double tau_r,double ae,double D,double kmu,double g4,double g1a,double g5b,double g3b,double g4b,double tau_a,double h_beta,double g2a,double h,double fr_err,__global double *CUs,__global double *U,__global double *CU,__global double *bks,__global double *G,__global double *CG,int M,double tau,double h_bm1,__global double *Us) \n\
{\n\
    double sum1=0,sum2=0,sum3=0,sum4=0,vi=0,vim1=0,si=0,sim1=0;\n\
    // d^a/dt\n\
    if ((tstep>=1)&&(alpha!=1.0))\n\
	if (((t0==0)?1:t0)<=t1)\n\
	for (int k=((t0==0)?1:t0);k<=t1;k++)\n\
	{\n\
	    double v=bks[k]*(CUs[(M+2)*(tstep+1-k)+i]-CUs[(M+2)*(tstep-k)+i]);\n\
	    sum1+=v;\n\
	    if (fabs(bks[k])<fr_err)\n\
		break;\n\
	}\n\
    // d^b/dx\n\
    if ((i>=2)&&(t0==0))\n\
	for (int s=3;s<=i+1;s++)\n\
	{\n\
	    double v=G[s]*CU[i-s+1];\n\
	    sum2+=v;\n\
	    if (fabs(G[s])<fr_err)\n\
		break;\n\
	}\n\
    // velocities\n\
    if (t0==0)\n\
    {\n\
   for (int s=0;s<=i+1;s++)\n\
    {\n\
		vi+=CG[s]*U[i-s+1];\n\
		if (s<=i) vim1+=CG[s]*U[i-s];\n\
		if (fabs(CG[s])<fr_err)\n\
			break;\n\
    }\n\
	vi/=h_bm1;\n\
	vim1/=h_bm1;\n\
	vi=-kmu*vi;\n\
	vim1=-kmu*vim1;\n\
	sum4=(vi*CU[i]-vim1*CU[i-1])/h; \n\
    }\n\
    if (tau_r!=0.0)\n\
    {\n\
		if (t0<=t1)\n\
		for (int k=t0;k<=t1;k++)\n\
		{\n\
			double s1=0,s2=0;\n\
			for (int s=0;s<=i+1;s++)\n\
			{\n\
				s1+=CG[s]*(((k==0)?U[i-s+1]:Us[(M+2)*(tstep+1-k)+i-s+1])-Us[(M+2)*(tstep-k)+i-s+1]);\n\
				if (s<=i) s2+=CG[s]*(((k==0)?U[i-s]:Us[(M+2)*(tstep+1-k)+i-s])-Us[(M+2)*(tstep-k)+i-s]);\n\
				if (fabs(CG[s])<fr_err)\n\
					break;\n\
			}\n\
			si+=bks[k]*s1;\n\
			sim1+=bks[k]*s2;\n\
			if (fabs(bks[k])<fr_err)\n\
				break;\n\
		}\n\
		si/=tau_a*h_bm1*g2a/tau_r;\n\
		sim1/=tau_a*h_bm1*g2a/tau_r;\n\
    }\n\
	vi=-kmu*si;\n\
	vim1=-kmu*sim1;\n\
    sum3=(vi*CU[i]-vim1*CU[i-1])/h; \n\
    double ret=(sum1/(tau_a*g2a))-sum3;\n\
    if (t0==0) ret+=(-CUs[(M+2)*tstep+i]/(tau_a*g2a))-sum4-(D*sum2/h_beta)-qc(i*h,tau*tstep,testing,alpha,beta,tau_r,ae,D,kmu, g4,g1a,g5b,g3b,g4b)+%g*CU[i];\n\
    return ret;\n\
}\n\
__kernel void Fill(int tstep,int testing, double alpha,double beta,double tau_r,double ae,double D,double kmu,double g4,double g1a,double g5b,double g3b,double g4b,double tau_a,double h_beta,double g2a,double h,double fr_err,__global double *Us,__global double *CUs,__global double *U,__global double *CU,__global double *bks,__global double *G,__global double *CG,__global double *Fs,__global double *CFs,int M,double tau,double h_bm1,int bottom_cond)\n\
{\n\
    int i;\n\
    int gi=get_global_id(0)%%GS;\n\
    __local double fsi[GS],cfsi[GS];\n\
    int st=(tstep/GS)+1;\n\
    int t0=gi*st+((gi==0)?0:1);\n\
    int t1=(gi+1)*st;\n\
    if (t1>tstep) t1=tstep;\n\
    // two points per thread - from opposite sides \n\
    for (int zu=0;zu<=1;zu++) \n\
    {\n\
	if (zu==0) i=(get_global_id(0)/GS);\n\
	if (zu==1) i=M-(get_global_id(0)/GS)+1;\n\
	if ((i!=0)&&(i!=M+1))\n\
	{\n\
		barrier(CLK_LOCAL_MEM_FENCE);\n\
		fsi[gi]=FU(t0,t1,i,tstep,testing, alpha,beta,tau_r,ae,D,kmu,g4,g1a,g5b,g3b,g4b,tau_a,h_beta,g2a,fr_err,Us,U,bks,G,M,tau,h); \n\
		cfsi[gi]=FCU(t0,t1,i,tstep,testing,alpha,beta,tau_r,ae,D,kmu,g4,g1a,g5b,g3b,g4b,tau_a,h_beta,g2a,h,fr_err,CUs,U,CU,bks,G,CG,M,tau,h_bm1,Us);\n\
		barrier(CLK_LOCAL_MEM_FENCE);\n\
		if (gi==0)\n\
		{\n\
		    Fs[i]=fsi[0];\n\
		    CFs[i]=cfsi[0];\n\
		    for (int ii=1;ii<GS;ii++)\n\
		    {\n\
			Fs[i]+=fsi[ii];\n\
			CFs[i]+=cfsi[ii];\n\
		    }\n\
		}\n\
		barrier(CLK_LOCAL_MEM_FENCE);\n\
	}\n\
	// boundary conditions \n\
	if ((i==0)&&(gi==0)) \n\
	{\n\
		Fs[0]=u(0,(tstep+1)*tau,testing,alpha,beta,tau_r,ae,D,kmu, g4,g1a,g5b,g3b,g4b);\n\
		CFs[0]=uc(0,(tstep+1)*tau,testing,alpha,beta,tau_r,ae,D,kmu, g4,g1a,g5b,g3b,g4b);\n\
	}\n\
	if ((i==M+1)&&(gi==0))\n\
	{\n\
		if (bottom_cond==0)\n\
			Fs[M+1]=u((M+1)*h,(tstep+1)*tau,testing,alpha,beta,tau_r,ae,D,kmu, g4,g1a,g5b,g3b,g4b);\n\
		else\n\
		{\n\
			double sum3=0;\n\
		    for (int s=1;s<=M+1;s++)\n\
			{\n\
				double v=CG[s]*U[M+1-s];\n\
				sum3+=v;\n\
				if (fabs(CG[s])<fr_err)\n\
					break;\n\
			}\n\
			Fs[M+1]=-sum3/CG[0];\n\
		}\n\
		if (bottom_cond==0) \n\
			CFs[M+1]=uc((M+1)*h,(tstep+1)*tau,testing,alpha,beta,tau_r,ae,D,kmu, g4,g1a,g5b,g3b,g4b);\n\
		else\n\
		{\n\
			double sum3=0;\n\
		    for (int s=1;s<=M+1;s++)\n\
			{\n\
				double v=CG[s]*CU[M+1-s];\n\
				sum3+=v;\n\
				if (fabs(CG[s])<fr_err)\n\
					break;\n\
			}\n\
			CFs[M+1]=-sum3/CG[0];\n\
		}\n\
	}\n\
    }\n\
}\n\
__kernel void Cdiff(__global double *C,__global double *Cprev,__global double *s,int N)\n\
{\n\
	int k=get_global_id(0);\n\
	__local double ss[GS];\n\
	ss[k]=0.0;\n\
	for (int i=(k*(N+1)/GS);i<((k+1)*(N+1)/GS);i++)\n\
	    if (i<N)\n\
		ss[k]+=(Cprev[i]-C[i])*(Cprev[i]-C[i]);\n\
	barrier(CLK_LOCAL_MEM_FENCE);\n\
	if (k==0) \n\
	{\n\
		for (int i=1;i<GS;i++)\n\
			ss[0]+=ss[i];\n\
		s[0]=ss[0];\n\
	}\n\
} \n\
";
OpenCL_program *prg=NULL;
OpenCL_prg *prog; // program object for matrix filler
cl_program cpProgram; // program object for linear system solver
#include "nvidia_pcr.h"
OpenCL_commandqueue *queue=NULL;
OpenCL_kernel *kFill,*kCdiff;
OpenCL_buffer *bUs=NULL,*bCUs=NULL,*bU=NULL,*bCU=NULL,*bbks=NULL,*bG=NULL,*bCG=NULL,*bs=NULL;
OpenCL_buffer *bUs2,*bCUs2,*bbks2,*bsU=NULL,*bsCU=NULL;
OpenCL_buffer *bAv=NULL,*bCAv=NULL,*bBv=NULL,*bCBv=NULL,*bSv=NULL,*bCSv=NULL,*bFv=NULL,*bCFv=NULL; // linear systems
OpenCL_buffer *bAv2=NULL,*bCAv2=NULL,*bBv2=NULL,*bCBv2=NULL,*bSv2=NULL,*bCSv2=NULL,*bFv2=NULL,*bCFv2=NULL; // linear systems
int prev_size=100;

int padded_size;
double *zp=NULL;
double *auxp=NULL;
double *op=NULL;
double *ss=NULL;

void init_opencl()
{
	int iv;
	if (prg==NULL) // code
	{
		prg = new OpenCL_program(1);
		char *text = new char[strlen(input_opencl_text)* 2];
		sprintf(text, input_opencl_text, ((double_ext==0)?"cl_amd_fp64":"cl_khr_fp64"),GS,u_inp_coef,u_inp_a,u_inp_b,c_inp_duration,c_inp_coef,decay_rate);
		prog = prg->create_program(text);
		delete[] text;
		pcr_small_systems_init_program(prog->hContext[0],0);
		//printf("OpenCL programs built\n");
	}
	if (queue==NULL) // queue and kernels
	{
		queue = prg->create_queue(0, 0);
		kFill = prg->create_kernel(prog, "Fill");
		kCdiff = prg->create_kernel(prog, "Cdiff");
		pcr_small_systems_init_kernels(prog->hContext[0],0);
		//printf("OpenCL queue and kernels created\n");
	}
	// cpu memory
	padded_size = pow(2.0,ceil(log2(M + 2)));
	if (zp) delete [] zp;
	if (auxp) delete [] auxp;
	if (op) delete [] op;
	if (ss) delete [] ss;
	ss = new double[GS];
	zp = new double[padded_size];
	auxp = new double[padded_size];
	op = new double[padded_size];
	for (int i = 0;i < padded_size;i++)
	{
		zp[i] = 0.0;
		op[i] = 1.0;
	}
	// gpu solutions
	bU = prg->create_buffer(CL_MEM_READ_WRITE , sizeof(double)*padded_size,NULL,bU);
	bCU = prg->create_buffer(CL_MEM_READ_WRITE , sizeof(double)*padded_size,NULL,bCU);
	bsU = prg->create_buffer(CL_MEM_READ_WRITE , sizeof(double)*padded_size,NULL,bsU);
	bsCU = prg->create_buffer(CL_MEM_READ_WRITE , sizeof(double)*padded_size,NULL,bsCU);
	bUs = prg->create_buffer(CL_MEM_READ_WRITE , sizeof(double)*(M+2)*prev_size,NULL,bUs);
	bCUs = prg->create_buffer(CL_MEM_READ_WRITE , sizeof(double)*(M+2)*prev_size,NULL,bCUs);
	queue->EnqueueWriteBuffer(bU, U,0,sizeof(double)*(M+2));
	queue->EnqueueWriteBuffer(bCU, CU,0,sizeof(double)*(M+2));
	// gpu linear systems
	for (int i = 0;i < padded_size;i++) auxp[i]=BU;
	auxp[0]=auxp[M+1]=0;
	bAv = prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(double)*padded_size, auxp,bAv);
	bAv2 = prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(double)*padded_size, zp,bAv2);
	queue->Finish();
	for (int i = 0;i < padded_size;i++) auxp[i]=CBU;
	auxp[0]=auxp[M+1]=0;
	bCAv = prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR , sizeof(double)*padded_size, auxp,bCAv);
	bCAv2 = prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR , sizeof(double)*padded_size, zp,bCAv2);
	queue->Finish();
	for (int i = 0;i < padded_size;i++) auxp[i]=AU;
	auxp[0]=auxp[M+1]=0;
	bBv = prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR , sizeof(double)*padded_size, auxp,bBv);
	bBv2 = prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR , sizeof(double)*padded_size, zp,bBv2);
	queue->Finish();
	for (int i = 0;i < padded_size;i++) auxp[i]=CAU;
	auxp[0]=auxp[M+1]=0;
	bCBv = prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR , sizeof(double)*padded_size, auxp,bCBv);
	bCBv2 = prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR , sizeof(double)*padded_size, zp,bCBv2);
	queue->Finish();
	for (int i = 0;i < padded_size;i++) auxp[i]=-SU;
	auxp[0]=auxp[M+1]=1;
	bSv = prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR , sizeof(double)*padded_size, auxp,bSv);
	bSv2 = prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR , sizeof(double)*padded_size, op,bSv2);
	queue->Finish();
	for (int i = 0;i < padded_size;i++) auxp[i]=-CSU;
	auxp[0]=auxp[M+1]=1;
	bCSv = prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR , sizeof(double)*padded_size, auxp,bCSv);
	bCSv2 = prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR , sizeof(double)*padded_size, op,bCSv2);
	bFv = prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR , sizeof(double)*padded_size, zp,bFv);
	bFv2 = prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR , sizeof(double)*padded_size, zp,bFv2);
	bCFv = prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR , sizeof(double)*padded_size, zp,bCFv);
	bCFv2 = prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR , sizeof(double)*padded_size, zp,bCFv2);
	// gpu aux
	bG = prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR , sizeof(double)*(M+2), G,bG);
	bCG = prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR , sizeof(double)*(M+2), CG,bCG);
	bbks= prg->create_buffer(CL_MEM_READ_WRITE , sizeof(double)*prev_size, NULL,bbks);
	bs = prg->create_buffer(CL_MEM_READ_WRITE , sizeof(double)*GS, NULL,bs);
	//printf("OpenCL memory allocated\n");
	// set fixed kernels args
	int N=M+2;
	kCdiff->SetArg(3, sizeof(int), &N);
	kCdiff->SetBufferArg(bs, 2);
	kFill->SetArg(1, sizeof(int), &testing);
	kFill->SetArg(2, sizeof(double), &alpha);
	kFill->SetArg(3, sizeof(double), &beta);
	kFill->SetArg(4, sizeof(double), &tau_r);
	kFill->SetArg(5, sizeof(double), &ae);
	kFill->SetArg(6, sizeof(double), &D);
	kFill->SetArg(7, sizeof(double), &kmu);
	kFill->SetArg(8, sizeof(double), &g4);
	kFill->SetArg(9, sizeof(double), &g1a);
	kFill->SetArg(10, sizeof(double), &g5b);
	kFill->SetArg(11, sizeof(double), &g3b);
	kFill->SetArg(12, sizeof(double), &g4b);
	kFill->SetArg(13, sizeof(double), &tau_a);
	kFill->SetArg(14, sizeof(double), &h_beta);
	kFill->SetArg(15, sizeof(double), &g2a);
	kFill->SetArg(16, sizeof(double), &h);
	kFill->SetArg(17, sizeof(double), &fr_err);
	kFill->SetBufferArg(bUs, 18);
	kFill->SetBufferArg(bCUs, 19);
	kFill->SetBufferArg(bU, 20);
	kFill->SetBufferArg(bCU, 21);
	kFill->SetBufferArg(bbks, 22);
	kFill->SetBufferArg(bG, 23);
	kFill->SetBufferArg(bCG, 24);
	kFill->SetBufferArg(bFv, 25);
	kFill->SetBufferArg(bCFv, 26);
	kFill->SetArg(27, sizeof(int), &M);
	kFill->SetArg(28, sizeof(double), &tau);
	kFill->SetArg(29, sizeof(double), &h_bm1);
	kFill->SetArg(30, sizeof(int), &bottom_cond);
	//printf("OpenCL fixed kernel parameters set\n");
	queue->Finish();
	printf("OpenCL initialization finished M %d size %d\n",M+2,padded_size);
}
void fr_save_previous()
{
	if (tstep>=(prev_size-2)) // realloc
	{
		double *a1=new double[(M+2)*prev_size*2];
		double *a2=new double[(M+2)*prev_size*2];
		double *a3=new double[prev_size*2];
		queue->EnqueueBuffer(bUs, a1,0,sizeof(double)*(M+2)*prev_size);
		queue->EnqueueBuffer(bCUs, a2,0,sizeof(double)*(M+2)*prev_size);
		queue->EnqueueBuffer(bbks, a3,0,sizeof(double)*prev_size);
		bUs = prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR , sizeof(double)*(M+2)*prev_size*2,a1,bUs);
		bCUs = prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR , sizeof(double)*(M+2)*prev_size*2,a2,bCUs);
		bbks = prg->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR , sizeof(double)*prev_size*2,a3,bbks);
		queue->Finish();
		prev_size*=2;
		kFill->SetBufferArg(bUs, 18);
		kFill->SetBufferArg(bCUs, 19);
		kFill->SetBufferArg(bbks, 22);
		delete [] a1;
		delete [] a2;
		delete [] a3;
		//printf("OpenCL buffers reallocated %d %d\n",tstep,prev_size);
	}
	int err=clEnqueueCopyBuffer(queue->hCmdQueue,bU->buffer,bUs->buffer,0,tstep*(M+2)*sizeof(double),(M+2)*sizeof(double),0,NULL,NULL);
	err=clEnqueueCopyBuffer(queue->hCmdQueue,bCU->buffer,bCUs->buffer,0,tstep*(M+2)*sizeof(double),(M+2)*sizeof(double),0,NULL,NULL);
	double v=bk(tstep);
	queue->EnqueueWriteBuffer(bbks, &v,sizeof(double)*tstep,sizeof(double));
	queue->Finish();
	//printf("OpenCL solutions saved to the array %d\n",tstep);
}
double fr_diff_c_prevc()
{
	double ret;
	kCdiff->SetBufferArg(bU, 0);
	kCdiff->SetBufferArg(bsU, 1);
	size_t nth=GS,lsize=GS;
	queue->ReleaseEvent(queue->ExecuteKernel(kCdiff, 1, &nth, &lsize));
	queue->EnqueueBuffer(bs, ss);
	queue->Finish();
	ret=ss[0];
	kCdiff->SetBufferArg(bCU, 0);
	kCdiff->SetBufferArg(bsCU, 1);
	queue->ReleaseEvent(queue->ExecuteKernel(kCdiff, 1, &nth, &lsize));
	queue->EnqueueBuffer(bs, ss);
	queue->Finish();
	ret+=ss[0];
	//printf("OpenCL difference calculated %d %g\n",tstep,ret);
	return ret;
}
void fr_getC()
{
	queue->EnqueueBuffer(bU, U,0,sizeof(double)*(M+2));
	queue->EnqueueBuffer(bCU, CU,0,sizeof(double)*(M+2));
	queue->Finish();
	//printf("OpenCL solutions saved to CPU memory %d\n",tstep);
}
void solveU2_opencl()
{
	solve_alloc();
	// save
	fr_save_previous();
	int iter=0;
	do
	{
		// save to compare
		int err=clEnqueueCopyBuffer(queue->hCmdQueue,bU->buffer,bsU->buffer,0,0,(M+2)*sizeof(double),0,NULL,NULL);
		err=clEnqueueCopyBuffer(queue->hCmdQueue,bCU->buffer,bsCU->buffer,0,0,(M+2)*sizeof(double),0,NULL,NULL);
		queue->Finish();
		//printf("OpenCL solutions saved for further comparison %d %d\n",tstep,iter);
		// fill
		size_t nth=ceil((M+2)/2.0)*GS, lsize=GS;
		kFill->SetArg(0, sizeof(int), &tstep);
		queue->ReleaseEvent(queue->ExecuteKernel(kFill, 1, &nth, &lsize));
		//printf("OpenCL right part filled %d %d\n",tstep,iter);
		// solve three-diagonal linear system
		pcr_solver(bAv->buffer,bSv->buffer,bBv->buffer,bFv->buffer,bU->buffer, padded_size,&queue->hCmdQueue,0,bAv2->buffer,bSv2->buffer,bBv2->buffer,bFv2->buffer);
		pcr_solver(bCAv->buffer,bCSv->buffer,bCBv->buffer,bCFv->buffer,bCU->buffer, padded_size,&queue->hCmdQueue,0,bCAv2->buffer,bCSv2->buffer,bCBv2->buffer,bCFv2->buffer);
		//printf("OpenCL linear systems solved %d %d\n",tstep,iter);
		queue->Finish();
		if ((iter++)==pick_maxiter) break;
		iteration_counter++;
//		printf("------ %g\n",fr_diff_c_prevc());
	}
	while ((fr_diff_c_prevc()/(2*M))>pick_err);
}
#endif
///////////////////////////////////////////////////////////////
////////////////////// time stepping procedure ////////////////
///////////////////////////////////////////////////////////////
void calc_step()
{
	double *sU=new double[M+2];
	double *sCU=new double[M+2];
	memcpy(sU,U,(M+2)*sizeof(double));
	memcpy(sCU,CU,(M+2)*sizeof(double));
	Us.push_back(sU);
	CUs.push_back(sCU);
	bks.push_back(bk(tstep));

	if (use_opencl==0)
		solveU2();
	else
	{
#ifdef USE_OPENCL
		solveU2_opencl();
#else
		solveU2();
#endif
	}
	tstep++;
}
void initialize()
{
	if (U!=NULL) delete [] U;
	if (CU!=NULL) delete [] CU;
	if (G!=NULL) delete [] G;
	if (CG!=NULL) delete [] CG;
	U=new double[M+2];
	CU=new double[M+2];
	G=new double[M+2];
	CG=new double[M+2];

	h=1.0/(M+1.0);
	tau_a=pow(tau,alpha);
	h_beta=pow(h,beta);
	h_bm1=pow(h,beta-1);
	g1a=Gamma(1.0+alpha);
	g3b=Gamma(3.0-beta);
	g4=Gamma(4.0);
	g4b=Gamma(4.0-beta);
	g5b=Gamma(5.0-beta);
	g2a=Gamma(2.0-alpha);

	AU=ae*(1+tau_r/(tau_a*g2a))/h_beta;
	BU=AU*beta*(beta-1.0)/2.0; 
	SU=(ae*(1+tau_r/(tau_a*g2a))*beta/h_beta)+(1.0/(tau_a*g2a)); 

	CAU=D/h_beta;
	CBU=CAU*beta*(beta-1.0)/2.0; 
	CSU=(D*beta/h_beta)+(1.0/(tau_a*g2a)); 

	for (int i=0;i<=M+1;i++)
	{
		G[i]=gs(i,beta);
		CG[i]=gs(i,beta-1);
	}

	for (int i=0;i<M+2;i++)
	{
		U[i]=u(i*h,0.0);
		CU[i]=uc(i*h,0.0);
	}
	for (int i=1;i<Us.size();i++)
	        delete [] Us[i];
	for (int i=1;i<CUs.size();i++)
	        delete [] CUs[i];
	Us.clear();
	CUs.clear();
	bks.clear();
	tstep=0;

#ifdef USE_OPENCL
	if (use_opencl)
		init_opencl();
#endif
}
void solve(int m,double a,double b,double t,int n,int ns,double t_r,double aa)
{
	M=m;
	alpha=a;
	beta=b;
	tau=t;
	tau_r=t_r;
	ae=aa;
	iteration_counter=0;
	initialize();
	for (int i=0;i<n;i++)
	{
		calc_step();
		if ((i%ns)==0)
		{
#ifdef USE_OPENCL
		    if (use_opencl)
			fr_getC();
#endif
		    double err=0;
		    double maxrel=0,relerr,relerr2,avg_rel=0.0;
		    int nrel=0;
		    for (int j=0;j<=M+1;j++)
		    {
			err+=(u(j*h,(i+1)*t)-U[j])*(u(j*h,(i+1)*t)-U[j])+(uc(j*h,(i+1)*t)-CU[j])*(uc(j*h,(i+1)*t)-CU[j]);
			if ((u(j*h,(i+1)*t)!=0.0)&&(uc(j*h,(i+1)*t)!=0.0))
			{
				relerr=100.0*fabs((u(j*h,(i+1)*t)-U[j])/u(j*h,(i+1)*t));
				relerr2=100.0*fabs((uc(j*h,(i+1)*t)-CU[j])/uc(j*h,(i+1)*t));
				if (relerr>maxrel) maxrel=relerr;
				if (relerr2>maxrel) maxrel=relerr2;
				avg_rel+=relerr+relerr2;
				nrel+=2;
			}
			printf("a %g b %g ae %g tr %g tau %g t %g x %g U %g Ua %g C %g Ca %g\n",a,b,ae,tau_r,t,(i+1)*t,j*h,U[j],u(j*h,(i+1)*t),CU[j],uc(j*h,(i+1)*t));
		    }
		    if (nrel!=0) avg_rel/=nrel;
		    printf("a %g b %g tr %g ae %g M %d tau %g t %g avgabs %g maxrel %g avgrel %g iters %d\n",a,b,tau_r,ae,M,t,(i+1)*t,sqrt(err/(2*M)),maxrel,avg_rel,iteration_counter);
		}
	}
}

//////////////////////////////////////
//////////////////////////////////////
//////////////////////////////////////
int main(int argc, char**argv)
{
    if (argc>=13)
	u_inp_coef=atof(argv[12]);
    if (argc>=14)
	c_inp_duration=atof(argv[13]);
    if (argc>=15)
	c_inp_coef=atof(argv[14]);
    if (argc>=16)
	D=atof(argv[15]);
    if (argc>=17)
	kmu=atof(argv[16]);
    if (argc>=18)
	decay_rate=atof(argv[17]);
    if (argc>=19)
	u_inp_a=atof(argv[18]);
    if (argc>=20)
	u_inp_b=atof(argv[19]);
    if (argc>=12)
    {
	testing=atoi(argv[9]);
	bottom_cond=atoi(argv[10]);
	use_opencl=atoi(argv[11]);
	if (testing==0) printf("u_inp %g c_dur %g c_inp %g D %g kmu %g decay_rate %g u_a %g u_b %g\n",u_inp_coef,c_inp_duration,c_inp_coef,D,kmu,decay_rate,u_inp_a,u_inp_b);
        solve(atoi(argv[8]),atof(argv[1]),atof(argv[2]),atof(argv[3]),atoi(argv[4]),atoi(argv[5]),atof(argv[6]),atof(argv[7]));
    }
    return 0;
}