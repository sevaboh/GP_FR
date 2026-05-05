#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <memory.h>
#include "gamma.h"
#include <vector>
#include <map>
#include <functional>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <random>
// constants
double max_val=1e300; // maximal/penalty fitness function value
double max_c=10; // maximal constant node value
int t_max_depth=5; // maximal generated tree depth
int max_nodes=32; // maximal number of nodes in a tree
int pop_size=200; // population size
double mut_prob=0.05; // probability of mutation
double mut_exp=0.1; // exponent for higher mutation probability on dense populations (0 - constant probability)
double shiftc_size=0.05; // size of random constants shifts
int n_shiftc_periter=10; // number of constants shifts per iteration
int max_iters=20000; // maximal number of iterations
double eps1=1e-4; // min difference between max and min f.f. values in the population
double eps2=1e-3; // min fitness function value
double eps3=1e-4; // min difference of subtree value in x points to consider the subtree as constant
double eps4=0.01; // min difference between derivative orders to see them as equal
double max_err_to_stop=0.7; // max accumulated error to stop calculating f.f. and return penalty value
double w_mult=10; // multiplier for crossover weight function
double regularization_a=0.1;
int min_i_to_stop=50; // minimal number of points to be processes before possible f.f. calculation stop
int fff=0; // fitness function through F
int fit_f=0; // fit f=g(*) not df/dt=g(*)
int no_x=0; // no x in fitting
int ff_nx=0; // number of x points to calculate f.f. (0 - all points)
int is_der=1; // 0 -no derivative, 1 - only first-order, 2 - fractional der and itegral
int debug=0;
int silent=1;
// GP tuning
double elitism=0.05;
int tournament=3; // for crossover: 0 - for random weighted, >0 - purely random 
// GA for floats
double ga_mm_mult=1.0; // multiplier for constant range
int ga_n=40; // population size
int ga_iters=10; // maximal number of iterations 
int ga_repeat=8; // repeat with optimization afterwards
double regularization_b=0.1; // for distance to initial
double use_soldb=0; // use previous possible solutions db 0- no, not 0 - fitness limit to save
std::string soldb_suffix=""; // suffix to add in the end of each solution db record
std::vector<std::pair<double,double> > ga_minmax; // externally set limits
int use_gradient=0; // 0 - no, 1 - yes, 2 - only GD
int ga_mask=0; // do not change set bits
// clustering distance for substitution effect assessments
double clustering_dist=15;
double similarity_dist=0.5;
double eff_minimum=-0.5; // -1 - account for everything, 0 - account only for positive
double effect_mult=1.0; // 0 - don't account for, 1 - >0 - always, 2 - 0 - 0.5, 1 - 1
int max_n_tries=10;
// input data
double *xX;
double **yY;
int maxf=1; // number of functions
int xys; // number of points
// init from previously saved candidates
int n_read_candidates=0;
// statistics
int stat[14]={0,0,0,0,0,0,0,0,0,0,0,0,0,0};
////////////////////////////////////////////////
////////////// genetic programming /////////////
////////////////////////////////////////////////
struct cache_item
{
    double g2o;
    double g1o;
    double *bs;
};
// DCaputo discretization coefficients
double b(double t0,double t1,double t2,double alpha) 
{ 
    if (alpha==0)
    {
	if (t2==t0) 
	    return 1;
	return 0;
    }
    return pow(t0-t1,alpha)-pow(t0-t2,alpha);
}
std::map<double, cache_item> caputo_cache;
cache_item precalc_caputo(double o)
{
    cache_item ret;
    ret.g2o=Gamma(2.0-o);
    ret.g1o=Gamma(1.0-o);
    ret.bs=new double[xys*xys];
    for (int i=0;i<xys;i++)
    for (int k=0;k<i;k++)
	if (o>=0) 
	    ret.bs[i*xys+k]=b(xX[i],xX[k],xX[k+1],1.0-o);
	else
	    ret.bs[i*xys+k]=b(xX[i],xX[k],xX[k+1],-o);
    return ret;
}
// numerical df/dt or DCaputo^o or I^o
double difff(int i,int mainf,double *yf=NULL,double o=1.0)
{
    double *y=yY[mainf];
    if (yf) y=yf;
    if (o>1.0) o=1.0;
    if (o<-1.0) o=-1.0;
    if (o!=1.0) 
    {
	cache_item cc;
	int found=0;
#pragma omp critical
	{
	    auto ccp=caputo_cache.find((int)(o/eps4)*eps4);
	    if (ccp!=caputo_cache.end())
	    {
		found=1;
		cc=ccp->second;
	    }
	    if (found==0)
	    {
		cc=precalc_caputo((int)(o/eps4)*eps4);
		caputo_cache.insert(std::pair<double,cache_item>((int)(o/eps4)*eps4,cc));
	    }
	}
	if (o>=0) // DCaputo^o
	{
	    if (i==0) return (y[1]-y[0])/(xX[1]-xX[0]);
	    double sum=0;
	    for (int k=0;k<i;k++)
		sum+=cc.bs[i*xys+k]*((y[k+1]-y[k])/(xX[k+1]-xX[k]));
	    sum/=cc.g2o;
	    return sum;
	}
	else // I^o
	{
	    if (i==0) return y[0];
	    double sum=0;
	    for (int k=0;k<i;k++)
		sum+=cc.bs[i*xys+k]*y[k+1];
	    sum/=cc.g1o;
	    return sum;
	}
    }
    if (i==0) return (y[1]-y[0])/(xX[1]-xX[0]);
    return (y[i]-y[i-1])/(xX[i]-xX[i-1]); // df/dt
}
// classes
class node
{
public:
    node *a,*b;
    int mainf;
    int type; // 0 - a+b, 1 - a*b, 2 - a^b, 3 - sin, 4 - c, 5 - x, 6 -f, 7 - a-b, 8 - a/b, 9 - exp, 10 - D^a f
    double c,_c2;
    node(int mf) {a=b=NULL;type=4;c=0.0;mainf=mf;}
    void clear() 
    {
		if (a) delete a;
		if (b) delete b; 
		a=b=NULL;
    }
    ~node() { clear(); }
    double val(int i,int *con=NULL,int opt=0,double *yf=NULL,int print=0,double **of=NULL) // calculates tree value
    {
		double v=0.0;
		int c1=1,c2=1;
		if (con) con[0]=0;
		if (type==4)
		{
		     v=c;
		     if (con) con[0]=1; // report than the result is a constant
		}
		double x=xX[i];
		if (type==5) v=x;
		if (type==6) { if (of) v=of[(int)c][i]; else {if ((yf)&&((int)c==mainf)) v=yf[i]; else v=yY[(int)c][i];} }
		if (type==7) v=a->val(i,&c1,opt,yf,print,of)-b->val(i,&c2,opt,yf,print,of);
		if (type==8) v=a->val(i,&c1,opt,yf,print,of)/b->val(i,&c2,opt,yf,print,of);
		if (type==0) v=a->val(i,&c1,opt,yf,print,of)+b->val(i,&c2,opt,yf,print,of);
		if (type==1) v=a->val(i,&c1,opt,yf,print,of)*b->val(i,&c2,opt,yf,print,of);
		if (type==2) v=pow(a->val(i,&c1,opt,yf,print,of),b->val(i,&c2,opt,yf,print,of));
		if (type==3) v=sin(a->val(i,&c1,opt,yf,print,of));
		if (type==9) v=exp(a->val(i,&c1,opt,yf,print,of));
		if (type==10) {if (_c2>1.0) _c2=1.0; if (_c2<-1.0) _c2=-1.0;} 
		if (type==10) { if (of) v=difff(i,(int)c,of[(int)c],_c2); else {if ((yf)&&((int)c==mainf)) v=difff(i,(int)c,yf,_c2); else v=difff(i,(int)c,NULL,_c2); }}
		if (print) printf("%d %d %g %g - %g\n",i,type,c,_c2,v);
		// subtree optimization
		if (opt)
		{
		    if (((type<=3)||(type>=7))&&(type!=10)) // simplify if arguments are constants
		    if (c1 && c2)
		    {
				c=v;
				delete a;
				a=NULL;
				if (type!=3)
				{
					delete b;
					b=NULL;    
				}
				type=4;
				if (con) con[0]=1; // report than the optimized result is a constant
				stat[0]++;
		    }
		    if ((type==0)||(type==1)||(type==7)||(type==8)) // simplify x+0,x-0, x*1, x/1
		    {
				int sa=0,sb=0;
				if ((a->type==4)&&(fabs(a->c)<eps3)&&((type==0)||(type==7))) // x+0, x-0
					sb=1;
				if ((b->type==4)&&(fabs(b->c)<eps3)&&((type==0)||(type==7))) // x+0, x-0
					sa=1;
				if ((a->type==4)&&(fabs(a->c-1)<eps3)&&((type==1)||(type==8))) // x*1, x/1
					sb=1;
				if ((b->type==4)&&(fabs(b->c-1)<eps3)&&((type==1)||(type==8))) // x*1, x/1
					sa=1;
				if (sb)
				{
					delete a;
					if (b->a) a=b->a->copy(); else a=NULL;
					type=b->type;
					c=b->c;
					_c2=b->_c2;
					mainf=b->mainf;
					node *sb=b;
					if (b->b) b=b->b->copy(); else b=NULL;
					delete sb;    
					stat[1]++;
				}
				else
				if (sa)
				{
					delete b;
					if (a->b) b=a->b->copy(); else b=NULL;
					type=a->type;
					c=a->c;
					_c2=a->_c2;
					mainf=a->mainf;
					node *sa=a;
					if (a->a) a=a->a->copy(); else a=NULL;
					delete sa;    
					stat[3]++;
				}
				    
		    }
		    if (type==1) // open paranthesis: a*(b +- f)=(ab) +- a*f
		    {
				double sd;
				if ((a->type==4)&&((b->type==0)||(b->type==7)))
				{
					if (b->a->type==4)
					{
						type=b->type;
						sd=a->c;
						a->c*=b->a->c;
						b->a->c=sd;
						b->type=1;
						stat[1]++;
					}
					else
					if (b->b->type==4)
					{
						type=b->type;
						sd=a->c;
						a->c*=b->b->c;
						b->b->c=sd;
						b->type=1;
						stat[2]++;
					}
				}
				else
				if ((b->type==4)&&((a->type==0)||(a->type==7)))
				{
					if (a->a->type==4)
					{
						type=a->type;
						sd=b->c;
						b->c*=a->a->c;
						a->a->c=sd;
						a->type=1;
						stat[3]++;
					}
					else
					if (a->b->type==4)
					{
						type=a->type;
						sd=b->c;
						b->c*=a->b->c;
						a->b->c=sd;
						a->type=1;
						stat[4]++;
					}
				}
		    }
		    if ((type==7)||(type==0)) // minus: a +/- ((-b)*f) => a -/+ (b*f)
		    if (b->type==1)
		    {
			if ((b->a->type==4)&&(b->a->c<0))
			{
			    if (type==7) type=0; else type=7;
			    b->a->c=-b->a->c;
			    stat[3]++;
			}
			else
			if ((b->b->type==4)&&(b->b->c<0))
			{
			    if (type==7) type=0; else type=7;
			    b->b->c=-b->b->c;
			    stat[4]++;
			}
		    }
		    // move constants up in a +/-(b +/- c) chains 
		    if ((type==7)||(type==0))
		    {
			// for b - const: (b+c)+a=(a+c)+b; (b-c)+a=(a-c)+b; (b+c)-a=(c-a)+b; (b-c)-a=b-(a+c)
			// for c - const: (b+c)+a=(b+a)+c; (b-c)+a=(b+a)-c; (b+c)-a=(b-a)+c; (b-c)-a=(b-a)-c
			if ((a->type==7)||(a->type==0))
			{
			    if (a->a->type==4)
			    {
				node *s=b;
				b=a->a;
				a->a=s;
				if ((type==7)&&(a->type==0)) { node *s2=a->b; a->b=a->a; a->a=s2; type=0; a->type=7; }
				if ((type==7)&&(a->type==7)) { node *s2=b; b=a; a=s2; b->type=0; }
				stat[1]++;
			    }
			    else
			    if (a->b->type==4)
			    {
				node *s=b;
				b=a->b;
				a->b=s;
				if ((type==0)&&(a->type==7)) { type=7; a->type=0; }
				if ((type==7)&&(a->type==0)) { type=0; a->type=7; }
				stat[2]++;
			    }
			}
			else
			// for b - const: a+(b+c)=b+(a+c); a+(b-c)=b+(a-c); a-(b+c)=(-b)+(a-c); a-(b-c)=(-b)+(a+c)
			// for c - const: a+(b+c)=c+(b+c); a+(b-c)=(-c)+(b+a); a-(b+c)=(-c)-(b-a); a-(b-c)=c-(b-a)
			if ((b->type==7)||(b->type==0))
			{
			    if (b->a->type==4)
			    {
				node *s=a;
				a=b->a;
				b->a=s;
				if ((type==7)&&(b->type==0)) { a->c=-a->c; type=0; b->type=7; }
				if ((type==7)&&(b->type==7)) { a->c=-a->c; type=0; b->type=0; }
				stat[3]++;
			    }
			    else
			    if (b->b->type==4)
			    {
				node *s=a;
				a=b->b;
				b->b=s;
				if ((type==0)&&(b->type==7)) { a->c=-a->c; b->type=0; }
				if ((type==7)&&(b->type==0)) { a->c=-a->c; b->type=7; }
				stat[4]++;
			    }
			}
		    }
		    if ((type==0)||(type==1)||(type==7)||(type==8)) // simplify +,-,*,/ constant chains
		    {
				if ((a->type==4)&&(b->type==type))
				{
					if (b->a->type==4)
					{
						if (type==0) a->c+=b->a->c;
						if (type==1) a->c*=b->a->c;
						if (type==7) a->c-=b->a->c;
						if (type==8) a->c/=b->a->c;
						node *bb=b->b;
						b->b=NULL;
						delete b;
						b=bb;
						stat[1]++;
					}
					else
					if (b->b->type==4)
					{
						if (type==0) a->c+=b->b->c;
						if (type==1) a->c*=b->b->c;
						if (type==7) a->c-=b->b->c;
						if (type==8) a->c/=b->b->c;
						node *bb=b->a;
						b->a=NULL;
						delete b;
						b=bb;				
						stat[2]++;
					}
				}
				else
				if ((b->type==4)&&(a->type==type))
				{
					if (a->a->type==4)
					{
						if (type==0) b->c+=a->a->c;
						if (type==1) b->c*=a->a->c;
						if (type==7) b->c-=a->a->c;
						if (type==8) b->c/=a->a->c;
						node *bb=a->b;
						a->b=NULL;
						delete a;
						a=bb;				
						stat[3]++;
					}
					else
					if (a->b->type==4)
					{
						if (type==0) b->c+=a->b->c;
						if (type==1) b->c*=a->b->c;
						if (type==7) b->c-=a->b->c;
						if (type==8) b->c/=a->b->c;
						node *bb=a->a;
						a->a=NULL;
						delete a;
						a=bb;				
						stat[4]++;
					}
				}
		    }
		    if (type==3) // sin(a-b*f)=sin((pi-a)+b*f)
		    if (a->type==7)
		    if (a->a->type==4)
		    {
			a->type=0;
			a->a->c=M_PI-a->a->c;
			stat[4]++;
		    }
		    if (type==0) // a+sin(pi+b)=a-sin(b)
		    if (b->type==3)
		    if (b->a->type==0)
		    if (b->a->a->type==4)
		    if (fabs(b->a->a->c-M_PI)<eps3)
		    {
			type=7;
			node *sa=b->a->b->copy();
			delete b->a;
			b->a=sa;
			stat[4]++;
		    }
		    if (type==3) // sin(a +/- b) - put a in [0,2*pi]
		    if ((a->type==7)||(a->type==0))
		    {
			if (a->a->type==4)
			if (finite(a->a->c))
			{
				if (a->a->c<100000)
				while (a->a->c>2*M_PI)
					a->a->c-=2*M_PI;
				if (a->a->c>-100000)
				while (a->a->c<0)
					a->a->c+=2*M_PI;
				stat[0]++;
			}
			if (a->b->type==4)
			if (finite(a->b->c))
			{
				if (a->b->c<100000)
				while (a->b->c>2*M_PI)
					a->b->c-=2*M_PI;
				if (a->b->c>-100000)
				while (a->b->c<0)
					a->b->c+=2*M_PI;
				stat[0]++;
			}
		    }
		    // sorting order in sums of derivatives: d^a f + d^b f = *if b>a* = d^b f + d^a f
		    if (type==0)
		    if ((a->type==10)&&(b->type==10))
		    if (a->c==b->c)
		    if (b->_c2>a->_c2)
		    {
			double sc=b->_c2;
			b->_c2=a->_c2;
			a->_c2=sc;
			stat[1]++;
		    }
		    // sorting order in sums of derivatives: d^a f + (c + d^b f) = *if b>a* = d^b f + (c + d^a f)
		    if (type==0)
		    if ((a->type==10)||(b->type==10))
		    if ((a->type==0)||(b->type==0))
		    {
			if (a->type==10)
			{
			    if (b->a->type==10)
			    if (a->c==b->a->c)
			    if (b->a->_c2<a->_c2)
			    {
				double sc=b->a->_c2;
				b->a->_c2=a->_c2;
				a->_c2=sc;
				stat[1]++;
			    }
			    if (b->b->type==10)
			    if (a->c==b->b->c)
			    if (b->b->_c2<a->_c2)
			    {
				double sc=b->b->_c2;
				b->b->_c2=a->_c2;
				a->_c2=sc;
				stat[2]++;
			    }
			}
			if (b->type==10)
			{
			    if (a->a->type==10)
			    if (b->c==a->a->c)
			    if (a->a->_c2<b->_c2)
			    {
				double sc=a->a->_c2;
				a->a->_c2=b->_c2;
				b->_c2=sc;
				stat[3]++;
			    }
			    if (a->b->type==10)
			    if (b->c==a->b->c)
			    if (a->b->_c2<b->_c2)
			    {
				double sc=a->b->_c2;
				a->b->_c2=b->_c2;
				b->_c2=sc;
				stat[4]++;
			    }
			}
		    }
		    // sorting order in sums of derivatives: d^a f + (d^b f - c) = *if b>a* = d^b f + (d^a f - c)
		    if (type==0)
		    if ((a->type==10)||(b->type==10))
		    if ((a->type==7)||(b->type==7))
		    {
			if (a->type==10)
			{
			    if (b->a->type==10)
			    if (a->c==b->a->c)
			    if (b->a->_c2<a->_c2)
			    {
				double sc=b->a->_c2;
				b->a->_c2=a->_c2;
				a->_c2=sc;
				stat[1]++;
			    }
			}
			if (b->type==10)
			{
			    if (a->a->type==10)
			    if (b->c==a->a->c)
			    if (a->a->_c2<b->_c2)
			    {
				double sc=a->a->_c2;
				a->a->_c2=b->_c2;
				b->_c2=sc;
				stat[2]++;
			    }
			}
		    }
		    // remove equal derivatives: d^a f - d^a f =0
		    if (type==7)
		    if ((a->type==10)&&(b->type==10))
		    if (a->c==b->c)
		    if (fabs(a->_c2-b->_c2)<eps4)
		    {
			delete a;
			delete b;
			a=b=NULL;
			type=4;
			c=0;
			stat[1]++;
		    }
		    // remove equal derivatives: d^a f + (b - d^a f) =b
		    if (type==0)
		    if ((a->type==10)||(b->type==10))
		    if ((a->type==7)||(b->type==7))
		    {
			if (b->type==7)
			{
			if (b->b->type==10)
			if (a->c==b->b->c)
			if (fabs(a->_c2-b->b->_c2)<eps4)
			{
			    node *sb=b;
			    delete a;
			    type=sb->a->type;
			    c=sb->a->c;
			    _c2=sb->a->_c2;
			    if (sb->a->a) a=sb->a->a->copy(); else a=NULL;
			    if (sb->a->b) b=sb->a->b->copy(); else b=NULL;
			    delete sb;
			    stat[1]++;
			}
			}
			else
			if (a->type==7)
			if (a->b->type==10)
			if (b->c==a->b->c)
			if (fabs(b->_c2-a->b->_c2)<eps4)
			{
			    node *sa=a;
			    delete b;
			    type=sa->a->type;
			    c=sa->a->c;
			    _c2=sa->a->_c2;
			    if (sa->a->a) a=sa->a->a->copy(); else a=NULL;
			    if (sa->a->b) b=sa->a->b->copy(); else b=NULL;
			    delete sa;
			    stat[2]++;
			}
		    }
		    // remove equal derivatives:  (b + d^a f) - d^a f =b
		    if (type==7)
		    if (b->type==10)
		    if (a->type==0)
		    {
			int done=0;
			if (a->b->type==10)
			if (b->c==a->b->c)
			if (fabs(b->_c2-a->b->_c2)<eps4)
			{
			    node *sa=a;
			    delete b;
			    type=sa->a->type;
			    c=sa->a->c;
			    _c2=sa->a->_c2;
			    if (sa->a->a) a=sa->a->a->copy(); else a=NULL;
			    if (sa->a->b) b=sa->a->b->copy(); else b=NULL;
			    delete sa;
			    done=1;
			    stat[1]++;
			}
			if (done==0)
			if (a->a->type==10)
			if (b->c==a->a->c)
			if (fabs(b->_c2-a->a->_c2)<eps4)
			{
			    node *sa=a;
			    delete b;
			    type=sa->b->type;
			    c=sa->b->c;
			    _c2=sa->b->_c2;
			    if (sa->b->a) a=sa->b->a->copy(); else a=NULL;
			    if (sa->b->b) b=sa->b->b->copy(); else b=NULL;
			    delete sa;
			    stat[2]++;
			}
		    }
		    // remove equal derivatives: d^a f - (d^a f-b) =b
		    if (type==7)
		    if (a->type==10)
		    if (b->type==7)
			if (b->a->type==10)
			if (a->c==b->a->c)
			if (fabs(a->_c2-b->a->_c2)<eps4)
			{
			    node *sb=b;
			    delete a;
			    type=sb->b->type;
			    c=sb->b->c;
			    _c2=sb->b->_c2;
			    if (sb->b->a) a=sb->b->a->copy(); else a=NULL;
			    if (sb->b->b) b=sb->b->b->copy(); else b=NULL;
			    delete sb;
			    stat[1]++;
			}
		    // remove equal derivatives: (d^a f-b) -d^a f=0-b
		    if (type==7)
		    if (b->type==10)
		    if (a->type==7)
			if (a->a->type==10)
			if (b->c==a->a->c)
			if (fabs(b->_c2-a->a->_c2)<eps4)
			{
			    delete b;
			    b=a->b->copy();
			    delete a->a;
			    delete a->b;
			    a->a=a->b=NULL;
			    a->type=4;
			    a->c=0;
			    stat[1]++;
			}
		    // merge equal derivatives: d^a f +/- b*d^a f =(1+/-b)*d^a f
		    if ((type==7)||(type==0))
		    {
			if ((a->type==10)&&(b->type==1))
			{
			    if (b->a->type==10) // d +/- d * b = d * (1 +/- b)
			    {
			    if (a->c==b->a->c)
			    if (fabs(a->_c2-b->a->_c2)<eps4)
			    {
				b->type=type;
				type=1;
				b->a->type=4;
				b->a->c=1.0;
				b->a->_c2=0.0;
				stat[1]++;
			    }
			    }
			    else
			    if (b->b->type==10) // d +/- b * d = d * (1 +/- b)
			    {
			    if (a->c==b->b->c)
			    if (fabs(a->_c2-b->b->_c2)<eps4)
			    {
				b->type=type;
				type=1;
				b->b->type=4;
				b->b->c=1.0;
				b->b->_c2=0.0;
				node *sb=b->b;
				b->b=b->a;
				b->a=sb;
				stat[2]++;
			    }
			    }
			}
			else
			if ((b->type==10)&&(a->type==1))
			{
			    if (a->a->type==10) // d * b +/- d = (b +/- 1) * d
			    {
			    if (b->c==a->a->c)
			    if (fabs(b->_c2-a->a->_c2)<eps4)
			    {
				a->type=type;
				type=1;
				a->a->type=4;
				a->a->c=1.0;
				a->a->_c2=0.0;
				node *sb=a->b;
				a->b=a->a;
				a->a=sb;
				stat[3]++;
			    }
			    }
			    else
			    if (a->b->type==10) // b * d +/- d = (b +/- 1) * d
			    {
			    if (b->c==a->b->c)
			    if (fabs(b->_c2-a->b->_c2)<eps4)
			    {
				a->type=type;
				type=1;
				a->b->type=4;
				a->b->c=1.0;
				a->b->_c2=0.0;
				stat[4]++;
			    }
			    }
			}
		    }
		    // sum equal derivatives: d^a f + d^a f = 2* d^a f
		    if (type==0)
		    if ((a->type==10)&&(b->type==10))
		    if (a->c==b->c)
		    if (fabs(a->_c2-b->_c2)<eps4)
		    {
			type=1;
			a->type=4;
			a->c=2.0;
			a->_c2=0.0;
			stat[1]++;
		    }
		    // sum equal derivatives: d^a f - (b- d^a f) =2*d^a f - b
		    if (type==7)
		    if (a->type==10)
		    if (b->type==7)
			if (b->b->type==10)
			if (a->c==b->b->c)
			if (fabs(a->_c2-b->b->_c2)<eps4)
			{
			    node *sba=b->a->copy();
			    if (b->a->a) { delete b->a->a; b->a->a=NULL; }
			    if (b->a->b) { delete b->a->b; b->a->b=NULL; }
			    b->type=1;
			    b->a->type=4;
			    b->a->c=2.0;
			    b->a->_c2=0.0;
			    delete a;
			    a=b;
			    b=sba;
			    stat[2]++;
			}
		    // sum equal derivatives: (b- d^a f) - d^a f =b- 2*d^a f
		    if (type==7)
		    if (b->type==10)
		    if (a->type==7)
			if (a->b->type==10)
			if (b->c==a->b->c)
			if (fabs(b->_c2-a->b->_c2)<eps4)
			{
			    node *saa=a->a->copy();
			    if (a->a->a) { delete a->a->a; a->a->a=NULL; }
			    if (a->a->b) { delete a->a->b; a->a->b=NULL; }
			    a->type=1;
			    a->a->type=4;
			    a->a->c=2.0;
			    a->a->_c2=0.0;
			    delete b;
			    b=a;
			    a=saa;
			    stat[3]++;
			}
		    // sum equal derivatives: d^a f + (d^a f +/-b) =d^a f *2+/- b
		    if (type==0)
		    if (a->type==10)
		    if ((b->type==7)||(b->type==0))
			if (b->a->type==10)
			if (a->c==b->a->c)
			if (fabs(a->_c2-b->a->_c2)<eps4)
			{
			    type=b->type;
			    node *sba=b->b->copy();
			    if (b->b->a) { delete b->b->a; b->b->a=NULL; }
			    if (b->b->b) { delete b->b->b; b->b->b=NULL; }
			    b->b->type=4;
			    b->b->c=2.0;
			    b->b->_c2=0.0;
			    type=7;
			    delete a;
			    a=b;
			    b=sba;
			    stat[4]++;
			}
		    // sum equal derivatives: (d^a f+/-b) + d^a f =d^a f*2+/-b
		    if (type==0)
		    if (b->type==10)
		    if ((a->type==7)||(a->type==0))
			if (a->a->type==10)
			if (b->c==a->a->c)
			if (fabs(b->_c2-a->a->_c2)<eps4)
			{
			    type=a->type;
			    node *saa=a->b->copy();
			    if (a->b->a) { delete a->b->a; a->b->a=NULL; }
			    if (a->b->b) { delete a->b->b; a->b->b=NULL; }
			    a->type=1;
			    a->b->type=4;
			    a->b->c=2.0;
			    a->b->_c2=0.0;
			    delete b;
			    b=saa;
			    stat[1]++;
			}
		    // sum equal derivatives: (b+d^a f) + d^a f =b+d^a f*2
		    if (type==0)
		    if (b->type==10)
		    if (a->type==0)
			if (a->b->type==10)
			if (b->c==a->b->c)
			if (fabs(b->_c2-a->b->_c2)<eps4)
			{
			    node *saa=a->a->copy();
			    if (a->a->a) { delete a->a->a; a->a->a=NULL; }
			    if (a->a->b) { delete a->a->b; a->a->b=NULL; }
			    a->type=1;
			    a->a->type=4;
			    a->a->c=2.0;
			    a->a->_c2=0.0;
			    delete b;
			    b=a;
			    a=saa;
			    stat[1]++;
			}
		}
		return v;
    }
    void optimize() // should be called from root
    {
		val(0,NULL,1);
    }
    void optimize2(double err) // subtrees with the value close to constant -> to constant node / positive - change all <0 to 0
    {
	double min=val(0),max=min;
	for (int i=1;i<xys;i++)
	{
	    double v=val(i);
	    if (v<min) min=v;
	    if (v>max) max=v;
	}
	if ((max-min)<err)
	{
	    clear();
	    type=4;
	    c=max;
	    stat[9]++;
	}
	else
	{
	    if (a) a->optimize2(err);
	    if (b) b->optimize2(err);
	}
    }
    int depth() // returns depth of the tree
    {
	if ((type==4)||(type==5)||(type==6)||(type==10))
		return 1;
	if ((type==3)||(type==9))
		return 1+a->depth();
	int ad=a->depth();
	int bd=b->depth();
	if (bd>ad) ad=bd;
	return 1+ad;
    }
    int n_nodes() const // returns number of nodes in the tree
    {
		if ((type==4)||(type==5)||(type==6)||(type==10))
			return 1;
		if ((type==3)||(type==9))
			return 1+a->n_nodes();
		return 1+a->n_nodes()+b->n_nodes();
    }
    node *get_node(int i) // returns node with number i (left search)
    {
		if (i<=0)
			return this;
		if (((type<4)||(type>=7))&&(type!=10))
		{
			node *l=a->get_node(i-1);
			if (l)
				return l;
			if ((type==3)||(type==9))
				return l;
			return b->get_node(i-1-a->n_nodes());
		}
		return NULL;
    }
    void gen_subtree(int mf,int max_depth,int depth=0) // build random tree with depth up to max_depth
    {
		mainf=mf;
		if (max_depth<0) max_depth=0;
a10:
		do
		do
			type=rand()%11;
		while ((max_depth==depth)&&(type!=4)&&(type!=5)&&(type!=6)&&(type!=10));
		while ((no_x==1)&&(type==5));
		c=0;
		_c2=1;
		a=b=NULL;
		if (type==4) 
			c=max_c*(2.0*((rand()%1000000)/1000000.0)-1.0);
		if (type==6) 
		{
		    if ((fit_f==1)&&(maxf==1)) goto a10;
		    do
			c=(int)(maxf*((rand()%1000000)/1000000.0));
		    while ((fit_f==1)&&(c==mainf)); // no f[mainf] when f is fitted
		}
		if (type==10)
		{
		    if (is_der==0) goto a10;
		    if ((fit_f==0)&&(maxf==1)) goto a10;
		    do
			c=(int)(maxf*((rand()%1000000)/1000000.0));
		    while ((c==mainf)&&(fit_f==0)); // no Df[mainf] when df is fitted
		    if (is_der==2)
		    {
			do
			    _c2=(2.0*((rand()%1000000)/1000000.0)-1.0);
			while(fabs(_c2)<0.5);
		    }
		    if (is_der==1)
			    _c2=1.0;
		}
		if (((type<=3)||(type>=7))&&(type!=10))
		{
			a=new node(0,mainf,max_depth,depth+1);
			if ((type!=3)&&(type!=9))
				b=new node(0,mainf,max_depth,depth+1);
		}
		stat[6]++;
    }
    int good_node() // check if there if node is permissible
    {
	if (depth()>t_max_depth+1)
		return 0;
	if (n_nodes()>max_nodes+1)
		return 0;
	return 1;
    }
    node(int check,int mf,int max_depth,int depth=0)
    {
		int i=0;
		do
		{
			if (i) clear(); i++;
			gen_subtree(mf,max_depth,depth);
			if (check==0) break;
		}
		while (good_node()==0);
    }
    void mutate(int max_depth) // change subtree to a randomly generated subtree with given maximal depth
    {
		clear();
		gen_subtree(mainf,max_depth);
    }
    void replace(node *n) // replace current node with node n
    {
		type=n->type;
		a=n->a;
		b=n->b;
		c=n->c;
		_c2=n->_c2;
    }
    node *copy()
    {
		node *res=new node(mainf);
		res->type=type;
		res->c=c;
		res->_c2=_c2;
		res->a=res->b=NULL;
		if (((type<=3)||(type>=7))&&(type!=10))
		{
			if (a) res->a=a->copy();
			if ((type!=3)&&(type!=9)) if (b) res->b=b->copy();
		}
		return res;
    }
    void print()
    {
		if (type==4) printf("%g",c);
		if (type==5) printf("x");
		if (type==6) printf("f[%d]",(int)c);
		if (type==10) printf("df^(%g)[%d]",_c2,(int)c);
		if (type==0) { printf("(");a->print();printf(")+(");b->print();printf(")"); }
		if (type==1) { printf("(");a->print();printf(")*(");b->print();printf(")"); }
		if (type==2) { printf("(");a->print();printf(")^(");b->print();printf(")"); }
		if (type==3) { printf("sin(");a->print();printf(")"); }
		if (type==9) { printf("exp(");a->print();printf(")"); }
		if (type==7) { printf("(");a->print();printf(")-(");b->print();printf(")"); }
		if (type==8) { printf("(");a->print();printf(")/(");b->print();printf(")"); }
    }
    void serialize(char *name,FILE *fi=NULL)
    {
		FILE *f=fi;
		if (fi==NULL)
			f=fopen(name,"wt");
		fprintf(f,"%d %lg %lg\n",type,c,_c2);
		if (((type<=3)||(type>=7))&&(type!=10))
		{
			a->serialize(name,f);
			if ((type!=3)&&(type!=9))
				b->serialize(name,f);
		}
		if (fi==NULL)
			fclose(f);
    }
///////////// for sorting, distancing, and usage in a map
    void canonical_form(std::vector<int>& out, int depth = 0) const
    {
	out.push_back(type);
	out.push_back(depth);
	if (a) a->canonical_form(out, depth + 1);
	if (b) b->canonical_form(out, depth + 1);
    }
    double edit_distance(const node* other, int no_swap=1) const
    {
	if (!other)
    	    return 1.0;
	// substitution cost
	double cost = (type == other->type) ? 0.0 : 1.0;
	// constant penalty
	if (type == 4 && other->type == 4)
    	    cost += fabs(c - other->c) / max_c;
	if (type == 10 && other->type == 10)
    	    cost += fabs(_c2 - other->_c2);
	// leaf cases
	if (!a && !b && !other->a && !other->b)
    	    return cost;
	// unary operators
	if ((type == 3 || type == 9) &&
    	    (other->type == 3 || other->type == 9))
	{
    	    return cost + a->edit_distance(other->a,no_swap);
	}
	// binary operators
	if (a && b && other->a && other->b)
	{
    	    double d1 = a->edit_distance(other->a,no_swap) + b->edit_distance(other->b,no_swap);
            // optional: allow swapped children for commutative ops
	    if (no_swap)
	    if (type == 0 || type == 1) // + or *
	    {
        	double d2 = a->edit_distance(other->b,no_swap) + b->edit_distance(other->a,no_swap);
                return cost + std::min(d1, d2);
	    }
            return cost + d1;
	}
        // structure mismatch - insertion/deletion
	return cost + fabs(n_nodes() - other->n_nodes());
    }
};
/// tree signature
struct TreeSignature
{
    int nodes;
    int depth;
    node *_node;
    std::array<int, 16> op_hist{};
    std::vector<int> canonical;
};
TreeSignature make_signature(node* t)
{
    TreeSignature s;
    s.nodes = t->n_nodes();
    s.depth = t->depth();
    s._node=t->copy();

    std::function<void(const node*)> hist =
        [&](const node* n)
        {
            s.op_hist[n->type]++;
            if (n->a) hist(n->a);
            if (n->b) hist(n->b);
        };

    hist(t);
    t->canonical_form(s.canonical);
    return s;
}
struct TreePairComparator
{
    bool operator()(const std::pair<TreeSignature,TreeSignature>& A,
                    const std::pair<TreeSignature,TreeSignature>& B) const
    {
        if (A.first.nodes != B.first.nodes)
            return A.first.nodes < B.first.nodes;
        if (A.first.depth != B.first.depth)
            return A.first.depth < B.first.depth;
        if (A.first.op_hist != B.first.op_hist)
            return A.first.op_hist < B.first.op_hist;
	if (A.first.canonical != B.first.canonical)
    	    return A.first.canonical < B.first.canonical;
        if (A.second.nodes != B.second.nodes)
            return A.second.nodes < B.second.nodes;
        if (A.second.depth != B.second.depth)
            return A.second.depth < B.second.depth;
        if (A.second.op_hist != B.second.op_hist)
            return A.second.op_hist < B.second.op_hist;
    	return A.second.canonical < B.second.canonical;
    }
};
/// substitution effect assessment
// substitution pair (first out, second in) to <average effect (difference of goal function values before and after substitution, number of substitutions>
std::map< std::pair<TreeSignature,TreeSignature> , std::pair<double,int> , TreePairComparator> substitution_effect; 
omp_lock_t se_lock;
void add_to_substitution_effect_map(node *old_n,node *new_n,double eff)
{
    TreeSignature s1=make_signature(old_n);
    TreeSignature s2=make_signature(new_n);
    auto p=std::pair<TreeSignature,TreeSignature>(s1,s2);
omp_set_lock(&se_lock);
    auto upper_bound_it = substitution_effect.lower_bound(p);
    if (upper_bound_it == substitution_effect.end())
    {
	if (substitution_effect.size()<10000)
	    substitution_effect.insert(std::pair<std::pair<TreeSignature,TreeSignature> , std::pair<double,int> >(p,std::pair<double,int>(eff,1))); // add new pair
    }
    else
    {
	double dist=upper_bound_it->first.first._node->edit_distance(old_n);
	dist+=upper_bound_it->first.second._node->edit_distance(new_n);
	if (dist<clustering_dist) // change average effect
	{
	    upper_bound_it->second.first=(upper_bound_it->second.first*upper_bound_it->second.second+eff)/(upper_bound_it->second.second+1);
	    upper_bound_it->second.second++;
	}
	else // add new pair
	{
	    if (substitution_effect.size()<10000)
		substitution_effect.insert(std::pair<std::pair<TreeSignature,TreeSignature> , std::pair<double,int> >(p,std::pair<double,int>(eff,1)));
	}
    }
omp_unset_lock(&se_lock);
}
double get_substitution_effect_assessment(node *old_n,node *new_n)
{
    TreeSignature s1=make_signature(old_n);
    TreeSignature s2=make_signature(new_n);
    auto p=std::pair<TreeSignature,TreeSignature>(s1,s2);
omp_set_lock(&se_lock);
    auto upper_bound_it = substitution_effect.lower_bound(p);
    double ret=-max_val;
    if (upper_bound_it!=substitution_effect.end())
    {
	double dist=upper_bound_it->first.first._node->edit_distance(old_n);
	dist+=upper_bound_it->first.second._node->edit_distance(new_n);
	if (dist<clustering_dist) // return effect
	    ret=upper_bound_it->second.first;
    }
omp_unset_lock(&se_lock);
    return ret;
}
/// deserialize
node *deserialize(char *name,int mf,FILE *fi)
{
	node *n=new node(mf);
	FILE *f=fi;
	if (fi==NULL)
	    f=fopen(name,"rt");
	if (f==NULL)
	    return NULL;
	fscanf(f,"%d %lg %lg\n",&n->type,&n->c,&n->_c2);
	if (((n->type<=3)||(n->type>=7))&&(n->type!=10))
	{
		n->a=deserialize(name,mf,f);
		if ((n->type!=3)&&(n->type!=9))
		    n->b=deserialize(name,mf,f);
	}
	if (fi==NULL)
	    fclose(f);
	return n;
}
// genetic programming class
// fitting dF[mainf]/dt=G / F[mainf]=G / DCaputo^a(t)F[mainf]=G
class genetic_programming;
void optimize_floats_ga(genetic_programming *gp, node* tree, int pop, int gens);
class genetic_programming
{
public:
	node **population; // G
	double *ffs; // fitness function values
	double *prevV; // calculated F values
	int iter;
	int mainf;
	double fitness_function(node *n,int no_opt=0,int print=0)
	{
	    double err=0.0,sum=0.0,v;
	    stat[5]++;
	    if (no_opt==0) n->optimize();
	    v=yY[mainf][0];
	    prevV[0]=v;
	    prevV[1]=yY[mainf][1];
	    for (int i=0;i<(ff_nx?ff_nx:xys-1);i++)
	    {
		double df,vi;
		if ((fff==0)&&(fit_f==0)) // df/dt - g
		{
			df=difff(i,mainf);
			vi=n->val(i,NULL,0,NULL,print)-df;
		}
		else // f - g
		{
			if (fit_f==1) { v=n->val(i,NULL,0,prevV,print); prevV[i+1]=v; } // g is a function
			df=yY[mainf][i];
			vi=v-df;
		}
		sum+=df*df;
		err+=vi*vi;
		// check accumulated value
		if (i>min_i_to_stop)
		{
		    double r=sqrt(err)/sqrt(sum);
		    if (r>max_err_to_stop)
			 return max_val;
		}
		if (fff) // calculate f solving diff(f) =g
		{
		    v+=n->val(i,NULL,0,prevV,print)*(xX[i+1]-xX[i]);
		    prevV[i+1]=v;
		}
	    }
	    if (!finite(err))
		return max_val;
	    return (sqrt(err)/sqrt(sum))+regularization_a*(n->n_nodes()/max_nodes);
	}
	genetic_programming(int mf) 
	{
	    iter=0;
	    mainf=mf;
	    prevV=new double[xys];
	    population=new node*[pop_size];
	    ffs=new double[pop_size];
	    for (int i=0;i<pop_size;i++)
	    {
			node *afunc=NULL;
			population[i]=new node(mainf);
			int try1=1;
			do
			{
				do
				{
				    if ((try1==1)&&(i<n_read_candidates)) // read from file
				    {
					char str[1024];
					node *n;
					sprintf(str,"candidate_%d.txt",i);
					n=deserialize(str,mainf,NULL);
					if (n)
					{
					    for (int zu=0;zu<ga_repeat;zu++)
					    {
						node *n2=NULL;
						optimize_floats_ga(this, n,ga_n,ga_iters);
						if (max_iters!=0)
						{
						    int ni=0;
						    do
						    {
							if (n2) delete n2;
							n2=n->copy();
							n->optimize();
							if ((ni++)>100) break;
						    }
						    while (n->edit_distance(n2,0)!=0.0);
						    n->optimize2(eps3);
						}
					    }
					    population[i]=n;
					}
					else
					{
					    population[i]->clear();
					    population[i]->gen_subtree(mainf,t_max_depth);
					}
				    }
				    else // generate
				    {
					population[i]->clear();
					population[i]->gen_subtree(mainf,t_max_depth);
				    }
				    try1=0;
				}
				while (population[i]->good_node()==0);
				if (max_iters!=0) population[i]->optimize2(eps3);
				ffs[i]=fitness_function(population[i],((max_iters==0)?1:0));
				if ((!finite(ffs[i]))||(ffs[i]>=max_val)||(population[i]->type==4 && ffs[i]>eps2))
				{
					population[i]->clear();
				}
				else
					break;
			}
			while (1);
			if (silent==0) printf("init %d %g\n",i,ffs[i]);
	    }
	}
	~genetic_programming()
	{
	    for (int i=0;i<pop_size;i++)
		delete population[i];
	    delete [] population;
	    delete [] prevV;
	}
	double weight(double x) // weights when selecting node for crossover
	{
		return 1.0/log(2.0+w_mult*x);
	}
	// mask for elitism
	void compute_mask(const double* ffs, size_t N, double n_percent, bool* mask)
	{
	    size_t k = static_cast<size_t>(std::ceil(N * n_percent));
	    std::vector<size_t> idx(N);
	    for (size_t i = 0; i < N; ++i) idx[i] = i;
	    std::nth_element(
		idx.begin(), idx.begin() + k, idx.end(),
	        [&](size_t a, size_t b) { return ffs[a] < ffs[b]; }
	    );
	    std::fill(mask, mask + N, false);
	    for (size_t j = 0; j < k; ++j)
		mask[idx[j]] = true;
	}
	void do_iteration(double mmdiff)
	{
	    // get max/min fitness function value and index of the corresponding node
	    double max=0,minv=max_val;
	    int imax=-1,imin=-1;
	    bool *mask=new bool[pop_size];
	    compute_mask(ffs,pop_size,elitism,mask);
	    for (int i=0;i<pop_size;i++)
	    {
	        if (ffs[i]>=max) {max=ffs[i];imax=i;}
		if (ffs[i]<minv) {minv=ffs[i];imin=i;}
	    }
	    // crossover
	    // crossover on G
	    int s1,s2,r1,r2,mn,n1,n2,no_do;
	    node *nn1,*old_n,*new_n;
	    double eff_assessment,r3;
	    int n_tries=max_n_tries;
	    do {
		// wighted randomly / tournament select two nodes
		do {
		    n1=-1,n2=-1; // indices of the selected nodes
		    if (tournament==0)
		    {
			double sum=0;
			for (int i=0;i<pop_size;i++)
			    sum+=weight(ffs[i]);
			double r1=sum*((rand()%1000000)/1000000.0);
			double r2=sum*((rand()%1000000)/1000000.0);
			for (int i=0;i<pop_size;i++)
			{
	    		    r1-=weight(ffs[i]);
	    		    if ((n1==-1)&&(r1<=0))
				n1=i;
			    r2-=weight(ffs[i]);
			    if ((n2==-1)&&(r2<=0)&&(i!=n1))
				n2=i;
			}
		    }
		    else
		    {
			for (int i=0;i<tournament;i++)
			{
			    double r1=pop_size*((rand()%1000000)/1000000.0);
			    double r2=pop_size*((rand()%1000000)/1000000.0);
			    if (n1==-1) n1=(int)r1; else if (ffs[(int)r1]<ffs[n1]) n1=(int)r1;
			    if (n2==-1) n2=(int)r2; else if (ffs[(int)r2]<ffs[n2]) n2=(int)r2;
			}
		    }
		} while (! ((n1!=n2)&&(n1!=-1)&&(n2!=-1)) );
		// randomly select two subtrees
		s1=population[n1]->n_nodes();
		s2=population[n2]->n_nodes();
		r1=rand()%s1;
		r2=rand()%s2;
		// penalty for similar trees
		double dist=population[n1]->edit_distance(population[n2]);
		// replace a subtree from node 1 with a subtree from node 2 in the copy of the node 1
		nn1=population[n1]->copy();
		old_n=nn1->get_node(r1)->copy();
		new_n=nn1->get_node(r1);
		delete nn1->get_node(r1)->a;
		delete nn1->get_node(r1)->b;
		nn1->get_node(r1)->replace(population[n2]->get_node(r2)->copy());
		// mix constant nodes that have the same indices between the node 2 and a new node into a new node
		int mn=nn1->n_nodes();
		if (population[n2]->n_nodes()<mn) mn=population[n2]->n_nodes();
		for (int i=0;i<mn;i++)
		{
			node *s1=nn1->get_node(i);
			node *s2=population[n2]->get_node(i);
			if ((s1->type==4)&&(s2->type==4))
			{
				double r=((rand()%1000000)/1000000.0);
				s1->c=s1->c*r+s2->c*(1.0-r);
			}
		}
		new_n=new_n->copy();
		eff_assessment=-get_substitution_effect_assessment(old_n,new_n);
		if (eff_assessment>1) eff_assessment=1;
		if (eff_assessment<eff_minimum) eff_assessment=eff_minimum;
		r3=effect_mult*((rand()%1000000)/1000000.0);
		// penalty for similar trees
		no_do=0;
		if ((similarity_dist!=0.0)&&(dist<similarity_dist))
		{
		    double r4=((rand()%1000000)/1000000.0);
		    if (r4>(dist/similarity_dist))
			no_do=1;
		}
		if ((ffs[n1]==max_val)||(ffs[n2]==max_val)) // pass through errorneous nodes
			break;
		if ((--n_tries)==0)
			break;
		if ((r3>(1.0+eff_assessment))||(no_do==1))
		{
		    delete nn1;
		    delete old_n;
		    delete new_n;
		    if (debug) printf("eff_assessment: %g %g %d %g size %d\n",r3,1+eff_assessment,no_do,dist,substitution_effect.size());
		    stat[11]++;
		    continue;
		}
		break;
	    }
	    while (1);
	    // recalculate fitness function
	    nn1->optimize2(eps3);
	    double fnn1=fitness_function(nn1);
	    if (!finite(fnn1)) fnn1=max_val;
	    if (fnn1>max_val) fnn1=max_val;
	    if (debug) printf("c(%d[%d] %d[%d])/effa[%g]/ ",n1,r1,n2,r2,eff_assessment);
	    if (nn1->good_node() && (nn1->type!=4 || fnn1<eps2)) // replace maximal ff valued node
	    {
		add_to_substitution_effect_map(old_n,new_n,fnn1-ffs[n1]);
		delete population[imax];
		population[imax]=nn1;
		ffs[imax]=fnn1;
		stat[7]++;
	    }
	    else
	    {
		if (debug) printf("c1/");
		delete nn1;
	    }
	    delete old_n;
	    delete new_n;
	    // mutation
	    double r=((rand()%1000000)/1000000.0);
	    if (r<exp(log(mut_prob)*pow(max-minv,mut_exp)))
	    {
		    node *nn1,*old_n,*new_n;
		    double sf1,eff_assessment,r3;
		    int n_tries=max_n_tries;
		    do {
			// select random node (except with minimal f.f. value + elitism accounting) and random subtree
			do
			{
			    n1=rand()%pop_size;
			    if (mask[n1]) stat[12]++;
			    if (debug) if (mask[n1]) printf("elitism - %d %g\n",n1,ffs[n1]);
			}
			while ((n1==imin)||(mask[n1]));
			// mutation on G
			n2=rand()%population[n1]->n_nodes();
			nn1=population[n1]->copy();
			sf1=ffs[n1];
			old_n=population[n1]->get_node(n2)->copy();
			new_n=population[n1]->get_node(n2);
			population[n1]->get_node(n2)->mutate(t_max_depth);
			new_n=new_n->copy();
			eff_assessment=-get_substitution_effect_assessment(old_n,new_n);
			if (eff_assessment>1) eff_assessment=1;
			if (eff_assessment<eff_minimum) eff_assessment=eff_minimum;
			r3=effect_mult*((rand()%1000000)/1000000.0);
			if (ffs[n1]==max_val) // pass through errorneous node
				break;
			if ((--n_tries)==0)
				break;
			if (r3>(1.0+eff_assessment))
			{
			    delete population[n1];
			    population[n1]=nn1;
			    delete old_n;
			    delete new_n;
			    stat[13]++;
			    if (debug) printf("eff_assessment mut - %g %g size %d\n",r3,1.0+eff_assessment,substitution_effect.size());
			}
		    } 
		    while (r3>(1.0+eff_assessment));
			// mutation on a(t)
			// recalculate fitness function
			population[n1]->optimize2(eps3);
			ffs[n1]=fitness_function(population[n1]);
			if (!finite(ffs[n1])) ffs[n1]=max_val;
			if (ffs[n1]>max_val) ffs[n1]=max_val;
			if (debug) printf("m(%d[%d])/effs[%g]/ ",n1,n2,eff_assessment);
			if ((ffs[n1]==max_val)||(population[n1]->good_node()==0)||(population[n1]->type==4 && ffs[n1]>eps2))
			{
				delete population[n1];
				population[n1]=nn1;
				ffs[n1]=sf1;
				if (debug) printf("c1");
			}
			else
			{
				add_to_substitution_effect_map(old_n,new_n,ffs[n1]-sf1);
				stat[8]++;
				delete nn1;
			}
			delete old_n;
			delete new_n;
	    }
	    // shift constants
	    for (int k=0;k<n_shiftc_periter;k++)
	    {
		int j=rand()%pop_size;
		// shift on G
		node *nn1=population[j]->copy();
		int nn=population[j]->n_nodes();
		for (int i=0;i<nn;i++)
		{
			node *s1=population[j]->get_node(i);
			if (s1->type==4)
			{
				double r=shiftc_size*max_c*(2.0*((rand()%1000000)/1000000.0)-1);
				s1->c+=r;
			}
		}
		// recalculate fitness function, save if new value is less than the previous one
		double nf=fitness_function(population[j]);
		if (nf<ffs[j])
		{
			delete nn1;
			ffs[j]=nf;
			stat[10]++;
		}
		else
		{
			delete population[j];
			population[j]=nn1;
		}
	    }
	    delete [] mask;
	}
	// iterative process
	node *optimize(double &minval,int &niter)
	{
	    node *res=NULL;
	    double oldmin=max_val,oldmax=0;
	    for (int i=0;i<14;i++) stat[i]=0;
	    do
	    {
			double max=0,min=max_val;
			if (max_iters!=0)
				do_iteration(oldmax-oldmin);
			for (int i=0;i<pop_size;i++)
			{
				if (ffs[i]<min) {min=ffs[i];res=population[i];}
				if (ffs[i]>max) max=ffs[i];
			}
			minval=min;
			if ((oldmin!=min)||(oldmax!=max))
			{
				if (silent==0) printf ("%d->(%g,%g)\n",iter,min,max);
				oldmin=min;
				oldmax=max;
			}
			if (((max-min)<eps1)&&(max!=max_val))
				break;
			if (min<eps2)
				break;
			iter++;
	    }
	    while (iter<max_iters);
	    niter=iter;
	    if (max_iters!=0) optimize_floats_ga(this, res,ga_n,ga_iters);
	    return res;
	}
};
// genetic algorithm for fitting floating point values
struct FloatParam
{
    double* ptr;
    double minv;
    double maxv;
};
struct Individual
{
    std::vector<double> genes;
    double fitness,fitness0;
};
void collect_float_params(node* n, std::vector<FloatParam>& out,node *u1=NULL,node *u2=NULL)
{
    if (!n) return;
    // type 4 → constant
    if (n->type == 4)
    {
        FloatParam p;
        p.ptr  = &n->c;
        p.minv = -max_c*ga_mm_mult;
        p.maxv =  max_c*ga_mm_mult;
	if (u1 && u2) // sin (a+/b) - a in [0,2*pi]
	if ((u1->type==7)||(u1->type==0))
	if (u2->type==3)
	{
	    p.minv=0;
	    p.maxv=2.0*M_PI;
	}
	if (u1 && u2) // sin (a*/b) - constant [0,2*pi)
	if ((u1->type==1)||(u1->type==8))
	if (u2->type==3)
	{
	    p.minv=0;
	    p.maxv=2.0*M_PI;
	}
	if (ga_minmax.size()>out.size())
	{
	    p.minv=ga_minmax[out.size()].first;
	    p.maxv=ga_minmax[out.size()].second;
	}
        out.push_back(p);
    }
    // type 10 → fractional order stored in _c2
    if (n->type == 10)
    {
        FloatParam p;
        p.ptr  = &n->_c2;
        // bounds for fractional order
        if (is_der == 2)
        {
            p.minv = -1.0+1e-10;
            p.maxv =  1.0-1e-10;
	    if (ga_minmax.size()>out.size())
	    {
		p.minv=ga_minmax[out.size()].first;
		p.maxv=ga_minmax[out.size()].second;
	    }
	    out.push_back(p);
        }
    }
    if (n->a) collect_float_params(n->a, out,n,u1);
    if (n->b) collect_float_params(n->b, out,n,u1);
}
double evaluate_float_individual(genetic_programming *gp,node* original,std::vector<FloatParam>& params, const std::vector<double>& genes, const std::vector<double>& base, double &f0,int print =0)
{
    node* copy_tree = original->copy();

    std::vector<FloatParam> new_params;
    collect_float_params(copy_tree, new_params);

    double l1=0,l2=0,l;
    for (size_t i = 0; i < genes.size(); ++i)
    {
        *new_params[i].ptr = genes[i];
	double div=std::max(fabs(params[i].maxv),fabs(params[i].minv));
	if (div>max_c) div=max_c;
	l1+=fabs(genes[i]-base[i])/div; // favour solution close to the base
	if (params[i].minv==-1.0+1e-10) 
	{
	    if (genes[i]>0) // penalty for fractional derivatives - favour integrals
		l2+=genes.size();
	    l2+=fabs(0.5-fabs(genes[i]))/div; // favour absolute values close to 0.5
	}
	else
	    l2+=fabs(genes[i])/div; // favour solution with minimal absolute value
    }
    l1/=genes.size();
    l2/=genes.size();
    l=(regularization_b*l1+regularization_a*l2)/2.0;
    f0=gp->fitness_function(copy_tree,1);
    double f = f0+l;
    if (print)
    {
	copy_tree->print();
        printf(" - f1 %g l %g l1 %g l2 %g f0 %g\n",f,l,l1,l2,f0);
    }
    delete copy_tree;
    return f;
}
Individual tournament_select(const std::vector<Individual>& pop, int tsize)
{
    Individual best = pop[rand()%pop.size()];
    for (int i = 1; i < tsize; ++i)
    {
        Individual cand = pop[rand()%pop.size()];
        if (cand.fitness < best.fitness)
                best = cand;
    }
    return best;
}
// Gradient descent
void refine_floats_gd(genetic_programming *gp,node* tree, std::vector<FloatParam> &params, std::vector<double> &base, std::vector<double> &genes, int steps = 50, double lr = 0.05, double eps = 1e-6)
{
    double f00,fb,f_new;
    int dim = params.size();

    fb = evaluate_float_individual(gp, tree, params, genes,base,f00);
    std::vector<double> x=genes;
    for (int step = 0; step < steps; step++)
    {
        std::vector<double> grad(dim, 0.0);

        double f0 = evaluate_float_individual(gp, tree, params, x,base,f00);

        // finite difference gradient
        for (int d = 0; d < dim; d++)
	if ((ga_mask & (1<<d))==0)
        {
            double old = x[d];

            double h = 1e-5 * (params[d].maxv - params[d].minv);

            x[d] = old + h;
            double f1 = evaluate_float_individual(gp, tree, params, x,base,f00);

            grad[d] = (f1 - f0) / h;

            x[d] = old;
        }

        // gradient step
        for (int d = 0; d < dim; d++)
    	if ((ga_mask & (1<<d))==0)
    	    x[d] -= lr * grad[d];

        f_new = evaluate_float_individual(gp, tree, params, x,base,f00);

        // simple adaptive step
        if (f_new > f0)
            lr *= 0.5;
        else
            lr *= 1.05;

        if (fabs(f_new - f0) < eps)
            break;
    }

    // write back
    if (f_new<fb)
    for (int d = 0; d < dim; d++)
        genes[d] = x[d];
}
std::vector<std::vector<double> > solution_db;
void optimize_floats_ga(genetic_programming *gp, node* tree, int pop, int gens)
{
    std::vector<FloatParam> params;
    std::vector<double> base;
    collect_float_params(tree, params);
    if (params.empty())
        return;
    int dim = params.size();
    std::vector<Individual> population(pop);
    double init_gf=gp->fitness_function(tree,1);
    // --- Initialize ---
    for (int i = 0; i < pop; ++i)
    {
        population[i].genes.resize(dim);
	if (i!=0)
	{
            for (int d = 0; d < dim; ++d)
	    if ((ga_mask & (1<<d))==0)
	    {
		int got=0;
    	        double r = (rand()%1000000)/1000000.0;
		if (solution_db.size()!=0)
		if (i<pop/2) // get half of the initial population randomly from solutions db
		if (solution_db[(int)(r*solution_db.size())].size()>=dim)
		{
		    double sv=solution_db[(int)(r*solution_db.size())][d]; // only if within limits if they are set
		    int in_limits=1;
		    if (ga_minmax.size()>d)
		    {
			if (sv<ga_minmax[d].first) in_limits=0;
			if (sv>ga_minmax[d].second) in_limits=0;
		    }
		    if (in_limits)
		    {
	    		population[i].genes[d] = sv;
			got=1;
		    }
		}
		if (got==0)
		    population[i].genes[d] = params[d].minv + r*(params[d].maxv - params[d].minv);
	    }
	    else
	        population[i].genes[d] = params[d].ptr[0];
	}
	else  // put values from initial tree and save them as a base
	{
            for (int d = 0; d < dim; ++d)
	        population[i].genes[d] = params[d].ptr[0];
	    if (use_gradient!=0)
		refine_floats_gd(gp,tree,params,population[i].genes, population[i].genes);
	    base=population[i].genes;
	}
        population[i].fitness = evaluate_float_individual(gp,tree, params, population[i].genes,base,population[i].fitness0);
    }
    // --- Evolution ---
    if (use_gradient!=2)
    for (int g = 0; g < gens; ++g)
    {
        std::vector<Individual> newpop;

        auto best = *std::min_element(
            population.begin(), population.end(),
            [](const Individual& a, const Individual& b)
            { return a.fitness < b.fitness; });
        auto worst = *std::min_element(
            population.begin(), population.end(),
            [](const Individual& a, const Individual& b)
            { return ((a.fitness==max_val)?0.0:a.fitness > b.fitness); });
	double avg=0;
	for (int i = 0; i < pop; ++i)
	if (population[i].fitness<1)
	    avg+=population[i].fitness;
	else 
	    avg+=1.0;
	avg/=pop;

	// dynamic change of mutation probability and tournament size
	int tsize=3;
	double mut=0.1;
	if ((avg-best.fitness)<0.5)
	{
	    mut=0.1+0.9*(exp(1.0-(avg-best.fitness)*2)-1.0)/(exp(1.0)-1.0);
	    tsize=(int)(1+2*(avg-best.fitness)*2);
	}

        newpop.push_back(best);

        while ((int)newpop.size() < pop)
        {
	    // crossover
            Individual p1 = tournament_select(population,tsize);
            Individual p2 = tournament_select(population,tsize);

            Individual child = p1;

            for (int d = 0; d < dim; ++d)
	    if ((ga_mask & (1<<d))==0)
            {
                double cmin = std::min(p1.genes[d], p2.genes[d]);
                double cmax = std::max(p1.genes[d], p2.genes[d]);
                double range = cmax - cmin;
                double alpha = 0.5; 

                double lower = cmin - alpha*range;
                double upper = cmax + alpha*range;

                double r = (rand()%1000000)/1000000.0;

                child.genes[d] = lower + r*(upper-lower);

		if (ga_minmax.size()==0)
		{
            	    if (child.genes[d] < params[d].minv) child.genes[d] = params[d].minv;
            	    if (child.genes[d] > params[d].maxv) child.genes[d] = params[d].maxv;
		}
            }
            // mutation
            for (int d = 0; d < dim; ++d)
	    if ((ga_mask & (1<<d))==0)
            {
                double r = (rand()%1000000)/1000000.0;
                if (r < mut)
                {
                    double shift =
                        0.1*(params[d].maxv - params[d].minv)*
                        (2.0*((rand()%1000000)/1000000.0)-1.0);

                    child.genes[d] += shift;

		    if (ga_minmax.size()==0)
		    {
            		if (child.genes[d] < params[d].minv) child.genes[d] = params[d].minv;
            		if (child.genes[d] > params[d].maxv) child.genes[d] = params[d].maxv;
		    }
                }
            }
            child.fitness = evaluate_float_individual(gp, tree, params, child.genes,base,child.fitness0);
	    newpop.push_back(child);
        }

        population = newpop;
    }
    auto best = *std::min_element(
        population.begin(), population.end(),
        [](const Individual& a, const Individual& b)
        { 
		return a.fitness < b.fitness; 
	});
    // save possible solutions dataset
#pragma omp critical
    if (use_soldb!=0.0)
    {
	FILE *fi=fopen("solutions_db.txt","a");
	if (best.fitness0<use_soldb)
	{
	    for (int j=0;j<dim;j++)
		fprintf(fi,"%g ",best.genes[j]);
	    fprintf(fi,"%s\n",soldb_suffix.data());
	}
    }

    // --- Write best back ---
    //evaluate_float_individual(gp, tree, params, best.genes,base,best.fitness0,1);
    if (best.fitness0<init_gf)
        for (int d = 0; d < dim; ++d)
	    *params[d].ptr = best.genes[d];
}
////////////////////////////////////////////////////////////////////
int main(int argc,char *argv[])
{
    node *best=NULL;
    if (argc<2) return 1;
    // read config
    char str[4096];
    FILE *fi=fopen(argv[1],"rt");
    fgets(str,4096,fi); max_val=atof(str); printf("max_val %g\n",max_val);
    fgets(str,4096,fi); max_c=atof(str); printf("max_c %g\n",max_c);
    fgets(str,4096,fi); t_max_depth=atoi(str); printf("t_max_depth %d\n",t_max_depth);
    fgets(str,4096,fi); max_nodes=atoi(str); printf("max_nodes %d\n",max_nodes);
    fgets(str,4096,fi); pop_size=atoi(str); printf("pop_size %d\n",pop_size);
    fgets(str,4096,fi); mut_prob=atof(str); printf("mut_prob %g\n",mut_prob);
    fgets(str,4096,fi); mut_exp=atof(str); printf("mut_exp %g\n",mut_exp);
    fgets(str,4096,fi); shiftc_size=atof(str); printf("shiftc_size %g\n",shiftc_size);
    fgets(str,4096,fi); n_shiftc_periter=atoi(str); printf("n_shiftc_periter %d\n",n_shiftc_periter);
    fgets(str,4096,fi); max_iters=atoi(str); printf("max_iters %d\n",max_iters);
    fgets(str,4096,fi); eps1=atof(str); printf("eps1 %g\n",eps1);
    fgets(str,4096,fi); eps2=atof(str); printf("eps2 %g\n",eps2);
    fgets(str,4096,fi); eps3=atof(str); printf("eps3 %g\n",eps3);
    fgets(str,4096,fi); max_err_to_stop=atof(str); printf("max_err_to_stop %g\n",max_err_to_stop);
    fgets(str,4096,fi); w_mult=atof(str); printf("w_mult %g\n",w_mult);
    fgets(str,4096,fi); regularization_a=atof(str); printf("regularization_a %g\n",regularization_a);
    fgets(str,4096,fi); min_i_to_stop=atoi(str); printf("min_i_to_stop %d\n",min_i_to_stop);
    fgets(str,4096,fi); fff=atoi(str); printf("fff %d\n",fff);
    fgets(str,4096,fi); fit_f=atoi(str); printf("fit_f %d\n",fit_f);
    fgets(str,4096,fi); no_x=atoi(str); printf("no_x %d\n",no_x);
    fgets(str,4096,fi); ff_nx=atoi(str); printf("ff_nx %d\n",ff_nx);
    fgets(str,4096,fi); debug=atoi(str); printf("debug %d\n",debug);
    fgets(str,4096,fi); silent=atoi(str); printf("silent %d\n",silent);
    fgets(str,4096,fi); is_der=atoi(str); printf("is_der %d",is_der);

    if (fgets(str,4096,fi)) ga_mm_mult=atof(str); printf(" ga_mm_mult %f",ga_mm_mult);
    if (fgets(str,4096,fi)) ga_n=atoi(str); printf(" ga_n %d",ga_n);
    if (fgets(str,4096,fi)) ga_iters=atoi(str); printf(" ga_iters %d",ga_iters);
    if (fgets(str,4096,fi)) ga_repeat=atoi(str); printf(" ga_rep %d",ga_repeat);
    if (fgets(str,4096,fi)) eps4=atof(str); printf(" eps4 %g",eps4);
    if (fgets(str,4096,fi)) regularization_b=atof(str); printf(" regularization_b %g",regularization_b);
    if (fgets(str,4096,fi)) use_soldb=atof(str); printf(" use_soldb %g",use_soldb);
    if (fgets(str,4096,fi)) use_gradient=atoi(str); printf(" use_GD %d",use_gradient);

    if (fgets(str,4096,fi)) elitism=atof(str); printf(" elitism %f",elitism);
    if (fgets(str,4096,fi)) tournament=atoi(str); printf(" tourn %d",tournament);
    if (fgets(str,4096,fi)) clustering_dist=atof(str); printf(" cl_dist %f",clustering_dist);
    if (fgets(str,4096,fi)) similarity_dist=atof(str); printf(" sm_dist %f",similarity_dist);
    if (fgets(str,4096,fi)) eff_minimum=atof(str); printf(" eff_min %f",eff_minimum);
    if (fgets(str,4096,fi)) effect_mult=atof(str); printf(" effect_mult %f",effect_mult);
    if (fgets(str,4096,fi)) max_n_tries=atoi(str); printf(" max_n_tries %d",max_n_tries);
    fclose(fi);
    printf("\n");

    // read or generate data
    if (argc==2)
    {
		// generate X,Y
		double L=5.0;
		xys=100;
		xX=new double[xys];
		yY=new double *[1];
		yY[0]=new double[xys];
		for (int i=0;i<xys;i++)
		{
			xX[i]=i*(L/xys);
			yY[0][i]=xX[i]*xX[i]*sin(xX[i]);
		}
    }
    else // read from argv[2] (delimiter - space, line - X, Y[0], ...., Y[maxf-1]
    {
		char str[4096];
		char *p,*p2;
		FILE *fi=fopen(argv[2],"rt");
		fgets(str,4096,fi);
		p=&str[0];
		maxf=0;
		while (p[0]) {if (p[0]==' ') maxf++;p++;}
		printf("maxf %d\n",maxf);
		fseek(fi,0,SEEK_SET);
		xys=0;
		while(fgets(str,4096,fi)) xys++;
		printf("xys %d\n",xys);
		xX=new double[xys];
		yY=new double *[maxf];
		for (int i=0;i<maxf;i++)
			yY[i]=new double [xys];
		int l=0;
		fseek(fi,0,SEEK_SET);
		while(fgets(str,4096,fi))
		{
			int i=0,e;
			p=p2=&str[0];
			while (p[0] && p[0]!=' ') p++;
			e=0; if (p[0]==' ') p[0]=0; else e=1;
			xX[l]=atof(p2);
			while (e==0)
			{
			p2=++p;
			while (p[0] && p[0]!=' ') p++;
			e=0; if (p[0]==' ') p[0]=0; else e=1;
			yY[i][l]=atof(p2);
			i++;
			}	    
			l++;
		}
		fclose(fi);
    }
	int f=0; // function to solve for
	int no_norm=0; //  1 - don't normalize
	if (argc>=4) f=atoi(argv[3]);
	if (argc>=6) no_norm=atoi(argv[5]);
	// normalize: change time scale to make dF[f]/dt range length = 1
	if (fit_f==0)
	{
	    double min1=difff(0,f),max1=difff(0,f);
	    for (int i=1;i<xys;i++)
	    {
		if (difff(i,f)<min1) min1=difff(i,f);
		if (difff(i,f)>max1) max1=difff(i,f);
	    }
	    if (no_norm) { max1=1;min1=0;}
	    printf("%g - %g %g\n",max1-min1,max1,min1);
	    for (int j=0;j<xys;j++)
		xX[j]*=(max1-min1);
	}
	else // normalize F[f] to [0,1] 
	{
	    double min1=yY[f][0],max1=yY[f][0];
	    for (int i=1;i<xys;i++)
	    {
		if (yY[f][i]<min1) min1=yY[f][i];
		if (yY[f][i]>max1) max1=yY[f][i];
	    }
	    if (no_norm) { max1=1;min1=0;}
	    printf("%g - %g\n",max1,min1);
	    for (int j=0;j<xys;j++)
		yY[f][j]=(yY[f][j]-min1)/(max1-min1);
	}
	if (argc>=5) // read function
	    if (argv[4][0]!='-')
		best=deserialize(argv[4],f,NULL);
	int n_bests_to_save=1; // number of best trees to save
	if (argc>=7) 
	    n_bests_to_save=atoi(argv[6]);
	if (argc>=8) // read previously generated candidates on initialization
	    n_read_candidates=atoi(argv[7]);
	if (argc>=9) // read min max for GA
	{
	  std::ifstream file(argv[8]);
          if (file.is_open()) 
	  {
	    std::string line;
	    while (std::getline(file, line)) {
    		if (line.empty()) continue;
    		std::stringstream ss(line);
    		double m1,m2;
		ss >> m1;
		ss >> m2;
		ga_minmax.push_back(std::pair<double,double>(m1,m2));
	    }
	    file.close();
	  }
	}
	if (argc>=10) // read min max for GA
	    soldb_suffix=std::string(argv[9]);
	if (argc>=11) // GA mask
	    ga_mask=atoi(argv[10]);
	//////////////////////
	//// read database of previously determined vectors of possible float parameter values
	if (use_soldb!=0.0)
        {
	  std::ifstream file("solutions_db.txt");
          if (file.is_open()) 
	  {
		   // Get file size once
        file.seekg(0, std::ios::end);
        unsigned long long file_size = static_cast<unsigned long long>(file.tellg());
        // Random generator
        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution< unsigned long long> dist(0, file_size - 1);
	    std::string line, dummy;
        for (int i = 0; i < ga_n; ++i) // read only random ga_n lines
        {
            if (!file.good())
                file.clear();
            file.seekg(dist(rng));
            std::getline(file, dummy);
            if (!std::getline(file, line))
                continue;
            if (line.empty())
                continue;
            std::stringstream ss(line);
            std::vector<double> row;
            double value;
            while (ss >> value)
                row.push_back(value);
            solution_db.push_back(row);
        }
        file.close();
		  }
	}
	// do GP fitting
    omp_init_lock(&se_lock);
    if (best==NULL)
    {
	srand(time(NULL));
	int ni1;
	double oldmin=0.0;
#pragma omp parallel for
	for (int it=0;it<24;it++)
	{
		double min;
		genetic_programming *g=new genetic_programming(f);
		node *opt=g->optimize(min,ni1);
		printf("%d %g(%d) ",it,min,ni1);
		for (int i=0;i<14;i++) printf("%d ",stat[i]);
		printf("\n");
#pragma omp critical
		if ((best==NULL)||(min<oldmin))
		if (opt)
		{
			oldmin=min;
			if (best) delete best;
			best=opt->copy();
			// save other trees
			bool *mask=new bool[pop_size];
			g->compute_mask(g->ffs,pop_size,(double)n_bests_to_save/pop_size,mask);
			int idx=0;
			for (int i=0;i<pop_size;i++)
			if (mask[i])
			{
				char str[1024];
				sprintf(str,"candidate_%d.txt",idx++);
				g->population[i]->serialize(str);
			}
		}
		delete g;
	}
    }
	// apply and check fitted or given best 
    if (best)
    {
		best->print();
		printf("\n");
		best->serialize("best.txt");
		printf("\n");
		double *prevV=new double[2*xys];
		double v=prevV[0]=yY[f][0];
		prevV[1]=yY[f][1];
		for (int i=0;i<((fit_f==0)?xys-1:xys);i++)
		{
			if (fit_f==0)
			{
			    printf("%g f %g - %g difff %g <- %g err %g\%\n",xX[i],yY[f][i],v,difff(i,f),best->val(i,NULL,0,prevV),100.0*fabs(best->val(i,NULL,0,prevV)-difff(i,f))/fabs(difff(i,f)));
			    if (i!=(xys-1)) v+=best->val(i,NULL,0,prevV)*(xX[i+1]-xX[i]);
			    prevV[i+1]=v;
			}
			else
			{
			    v=best->val(i,NULL,0,prevV);
			    printf("%g f %g - %g err %g\%\n",xX[i],yY[f][i],v,100.0*fabs(v-yY[f][i])/fabs(yY[f][i]));
			    prevV[i+1]=v;
			}
		}
		delete prevV;
		delete best;
    }
    omp_destroy_lock(&se_lock);
    return 0;
}
