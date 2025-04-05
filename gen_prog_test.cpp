#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <memory.h>
#include "gamma.h"
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
// input data
double *xX;
double **yY;
int maxf=1; // number of functions
int xys; // number of points
// statistics
int stat[11]={0,0,0,0,0,0,0,0,0,0,0};
////////////////////////////////////////////////
////////////// genetic programming /////////////
////////////////////////////////////////////////
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
// numerical df/dt or DCaputo^o or I^o
double difff(int i,int mainf,double *yf=NULL,double o=1.0)
{
    double *y=yY[mainf];
    if (yf) y=yf;
    if (o!=1.0) 
    {
	if (o>=0) // DCaputo^o
	{
	    if (i==0) return (y[1]-y[0])/(xX[1]-xX[0]);
	    double sum=0;
	    for (int k=0;k<i;k++)
		sum+=b(xX[i],xX[k],xX[k+1],1.0-o)*((y[k+1]-y[k])/(xX[k+1]-xX[k]));
	    sum/=Gamma(2-o);
	    return sum;
	}
	else // I^o
	{
	    if (i==0) return y[0];
	    double sum=0;
	    for (int k=0;k<i;k++)
		sum+=b(xX[i],xX[k],xX[k+1],-o)*y[k+1];
	    sum/=Gamma(1-o);
	    return sum;
	}
    }
    if (i==0) return (y[1]-y[0])/(xX[1]-xX[0]);
    return (y[i]-y[i-1])/(xX[i]-xX[i-1]); // df/dt
}
// classes
class node;
node *deserialize(char *name,int mf,FILE *fi=NULL);
class node
{
friend node *deserialize(char *name,int mf,FILE *fi);
    node *a,*b;
    int mainf;
public:
    int type; // 0 - a+b, 1 - a*b, 2 - a^b, 3 - sin, 4 - c, 5 - x, 6 -f, 7 - a-b, 8 - a/b
    double c,_c2;
    node(int mf) {a=b=NULL;type=4;c=0.0;mainf=mf;}
    void clear() 
    {
		if (a) delete a;
		if (b) delete b; 
		a=b=NULL;
    }
    ~node() { clear(); }
    double val(int i,int *con=NULL,int opt=0,double *yf=NULL) // calculates tree value
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
		if (type==6) { if ((yf)&&((int)c==mainf)) v=yf[i]; else v=yY[(int)c][i]; }
		if (type==7) v=a->val(i,&c1,opt,yf)-b->val(i,&c2,opt,yf);
		if (type==8) v=a->val(i,&c1,opt,yf)/b->val(i,&c2,opt,yf);
		if (type==0) v=a->val(i,&c1,opt,yf)+b->val(i,&c2,opt,yf);
		if (type==1) v=a->val(i,&c1,opt,yf)*b->val(i,&c2,opt,yf);
		if (type==2) v=pow(a->val(i,&c1,opt,yf),b->val(i,&c2,opt,yf));
		if (type==3) v=sin(a->val(i,&c1,opt,yf));
		if (type==9) v=exp(a->val(i,&c1,opt,yf));
		if (type==10) { if ((yf)&&((int)c==mainf)) v=difff(i,(int)c,yf,_c2); else v=difff(i,(int)c,NULL,_c2); }
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
    int n_nodes() // returns number of nodes in the tree
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
};
node *deserialize(char *name,int mf,FILE *fi)
{
	node *n=new node(mf);
	FILE *f=fi;
	if (fi==NULL)
	    f=fopen(name,"rt");
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
// fitting dF[mainf]/dt=G / F[mainf]=G / DCaputo^a(t)F[mainf]=G
class genetic_programming
{
	node **population; // G
	double *ffs; // fitness function values
	double *prevV; // calculated F values
	int iter;
	int mainf;
public:
	double fitness_function(node *n)
	{
	    double err=0.0,sum=0.0,v;
	    stat[5]++;
	    n->optimize();
	    v=yY[mainf][0];
	    prevV[0]=v;
	    prevV[1]=yY[mainf][1];
	    for (int i=0;i<(ff_nx?ff_nx:xys-1);i++)
	    {
		double df,vi;
		if ((fff==0)&&(fit_f==0)) // df/dt - g
		{
			df=difff(i,mainf);
			vi=n->val(i)-df;
		}
		else // f - g
		{
			if (fit_f==1) { v=n->val(i,NULL,0,prevV); prevV[i+1]=v; } // g is a function
			df=yY[mainf][i];
			vi=v-df;
		}
		sum+=df*df;
		err+=vi*vi;
		// check accumulated value
		if (i>min_i_to_stop)
		{
		    double r=sqrt(err)/sqrt(sum);
		    if (r>max_err_to_stop) return max_val;
		}
		if (fff) // calculate f solving diff(f) =g
		{
		    v+=n->val(i,NULL,0,prevV)*(xX[i+1]-xX[i]);
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
			do
			{
				do
				{
					population[i]->clear();
					population[i]->gen_subtree(mainf,t_max_depth);
				}
				while (population[i]->good_node()==0);
				population[i]->optimize2(eps3);
				ffs[i]=fitness_function(population[i]);
				if ((!finite(ffs[i]))||(ffs[i]>=max_val)||(population[i]->type==4 && ffs[i]>eps2))
					population[i]->clear();
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
	void do_iteration(double mmdiff)
	{
	    // get max/min fitness function value and index of the corresponding node
	    double max=0,minv=max_val;
	    int imax=-1,imin=-1;
	    for (int i=0;i<pop_size;i++)
	    {
	        if (ffs[i]>=max) {max=ffs[i];imax=i;}
		if (ffs[i]<minv) {minv=ffs[i];imin=i;}
	    }
	    // crossover
	    // wighted randomly select two nodes
	    double sum=0;
	    for (int i=0;i<pop_size;i++)
			sum+=weight(ffs[i]);
	    int n1=-1,n2=-1; // indices of the selected nodes
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
	    if ((n1!=n2)&&(n1!=-1)&&(n2!=-1))
	    {
			// crossover on G
			// randomly select two subtrees
			int s1=population[n1]->n_nodes();
			int s2=population[n2]->n_nodes();
			int r1=rand()%s1;
			int r2=rand()%s2;
			// replace a subtree from node 1 with a subtree from node 2 in the copy of the node 1
			node *nn1=population[n1]->copy();
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
			// recalculate fitness function
			nn1->optimize2(eps3);
			double fnn1=fitness_function(nn1);
			if (!finite(fnn1)) fnn1=max_val;
			if (fnn1>max_val) fnn1=max_val;
			if (debug) printf("c(%d[%d] %d[%d]) ",n1,r1,n2,r2);
			if (nn1->good_node() && (nn1->type!=4 || fnn1<eps2)) // replace maximal ff valued node
			{
				delete population[imax];
				population[imax]=nn1;
				ffs[imax]=fnn1;
				stat[7]++;
			}
			else
			{
				if (debug) printf("c1");
				delete nn1;
			}
	    }
	    // mutation
	    double r=((rand()%1000000)/1000000.0);
	    if (r<exp(log(mut_prob)*pow(max-minv,mut_exp)))
	    {
			// select random node (except with miniman f.f. value) and random subtree
			do
			    n1=rand()%pop_size;
			while (n1==imin);
			// mutation on G
			n2=rand()%population[n1]->n_nodes();
			node *nn1=population[n1]->copy();
			double sf1=ffs[n1];
			population[n1]->get_node(n2)->mutate(t_max_depth);
			// mutation on a(t)
			// recalculate fitness function
			population[n1]->optimize2(eps3);
			ffs[n1]=fitness_function(population[n1]);
			if (!finite(ffs[n1])) ffs[n1]=max_val;
			if (ffs[n1]>max_val) ffs[n1]=max_val;
			if (debug) printf("m(%d[%d]) ",n1,n2);
			if ((ffs[n1]==max_val)||(population[n1]->good_node()==0)||(population[n1]->type==4 && ffs[n1]>eps2))
			{
				delete population[n1];
				population[n1]=nn1;
				ffs[n1]=sf1;
				if (debug) printf("c1");
			}
			else
			{
				stat[8]++;
				delete nn1;
			}
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
	}
	// iterative process
	node *optimize(double &minval,int &niter)
	{
	    node *res=NULL;
	    double oldmin=max_val,oldmax=0;
	    for (int i=0;i<11;i++) stat[i]=0;
	    do
	    {
			double max=0,min=max_val;
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
	    return res;
	}
};
int main(int argc,char *argv[])
{
    node *best=NULL;
    if (argc<2) return 1;
    // read config
    char str[1024];
    FILE *fi=fopen(argv[1],"rt");
    fgets(str,1024,fi); max_val=atof(str); printf("max_val %g\n",max_val);
    fgets(str,1024,fi); max_c=atof(str); printf("max_c %g\n",max_c);
    fgets(str,1024,fi); t_max_depth=atoi(str); printf("t_max_depth %d\n",t_max_depth);
    fgets(str,1024,fi); max_nodes=atoi(str); printf("max_nodes %d\n",max_nodes);
    fgets(str,1024,fi); pop_size=atoi(str); printf("pop_size %d\n",pop_size);
    fgets(str,1024,fi); mut_prob=atof(str); printf("mut_prob %g\n",mut_prob);
    fgets(str,1024,fi); mut_exp=atof(str); printf("mut_exp %g\n",mut_exp);
    fgets(str,1024,fi); shiftc_size=atof(str); printf("shiftc_size %g\n",shiftc_size);
    fgets(str,1024,fi); n_shiftc_periter=atoi(str); printf("n_shiftc_periter %d\n",n_shiftc_periter);
    fgets(str,1024,fi); max_iters=atoi(str); printf("max_iters %d\n",max_iters);
    fgets(str,1024,fi); eps1=atof(str); printf("eps1 %g\n",eps1);
    fgets(str,1024,fi); eps2=atof(str); printf("eps2 %g\n",eps2);
    fgets(str,1024,fi); eps3=atof(str); printf("eps3 %g\n",eps3);
    fgets(str,1024,fi); max_err_to_stop=atof(str); printf("max_err_to_stop %g\n",max_err_to_stop);
    fgets(str,1024,fi); w_mult=atof(str); printf("w_mult %g\n",w_mult);
    fgets(str,1024,fi); regularization_a=atof(str); printf("regularization_a %g\n",regularization_a);
    fgets(str,1024,fi); min_i_to_stop=atoi(str); printf("min_i_to_stop %d\n",min_i_to_stop);
    fgets(str,1024,fi); fff=atoi(str); printf("fff %d\n",fff);
    fgets(str,1024,fi); fit_f=atoi(str); printf("fit_f %d\n",fit_f);
    fgets(str,1024,fi); no_x=atoi(str); printf("no_x %d\n",no_x);
    fgets(str,1024,fi); ff_nx=atoi(str); printf("ff_nx %d\n",ff_nx);
    fgets(str,1024,fi); debug=atoi(str); printf("debug %d\n",debug);
    fgets(str,1024,fi); silent=atoi(str); printf("silent %d\n",silent);
    fgets(str,1024,fi); is_der=atoi(str); printf("is_der %d\n",is_der);
    fclose(fi);
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
		char str[1024];
		char *p,*p2;
		FILE *fi=fopen(argv[2],"rt");
		fgets(str,1024,fi);
		p=&str[0];
		maxf=0;
		while (p[0]) {if (p[0]==' ') maxf++;p++;}
		printf("maxf %d\n",maxf);
		fseek(fi,0,SEEK_SET);
		xys=0;
		while(fgets(str,1024,fi)) xys++;
		printf("xys %d\n",xys);
		xX=new double[xys];
		yY=new double *[maxf];
		for (int i=0;i<maxf;i++)
			yY[i]=new double [xys];
		int l=0;
		fseek(fi,0,SEEK_SET);
		while(fgets(str,1024,fi))
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
	if (argc>=4) f=atoi(argv[3]);
	// normalize: change time scale to make dF[f]/dt range length = 1
	if (fit_f==0)
	{
	    double min1=difff(0,f),max1=difff(0,f);
	    for (int i=1;i<xys;i++)
	    {
		if (difff(i,f)<min1) min1=difff(i,f);
		if (difff(i,f)>max1) max1=difff(i,f);
	    }
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
	    printf("%g - %g\n",max1,min1);
	    for (int j=0;j<xys;j++)
		yY[f][j]=(yY[f][j]-min1)/(max1-min1);
	}
	if (argc>=5) // read function
	    best=deserialize(argv[4],f,NULL);
	//////////////////////
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
		for (int i=0;i<11;i++) printf("%d ",stat[i]);
		printf("\n");
#pragma omp critical
		if ((best==NULL)||(min<oldmin))
		if (opt)
		{
			oldmin=min;
			if (best) delete best;
			best=opt->copy();
		}
		delete g;
	}
    }
    if (best)
    {
		best->print();
		printf("\n");
		best->serialize("best.txt");
		printf("\n");
		double *prevV=new double[xys];
		double v=prevV[0]=yY[f][0];
		prevV[1]=yY[f][1];
		for (int i=0;i<xys-1;i++)
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
    return 0;
}
