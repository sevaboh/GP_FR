#include <vector>
#include <algorithm>
bool AbsGreat(double x, double y) {
    return fabs(x) > fabs(y);
}
bool GreaterPair(std::pair<double,double> x, std::pair<double,double> y) {
    return x.first > y.first;
}
double sum_sort_pyramid(std::vector<double> values)
{
	std::sort(values.begin(),values.end(),AbsGreat);
	int old_s=0;
	int old_e=0;
	do
	{
		old_s=old_e;
		old_e=values.size();
		for (int i=old_s;i<(old_e-(old_e-old_s)%2);i+=2)
			values.push_back(values[i]+values[i+1]);
		if ((old_e-old_s)%2)
			values[values.size()-1]+=values[old_e-1];
	}
	while (values.size()-old_e>1);
	return values[values.size()-1];
}
// integrate by linear approximation and recursive subdivision
double integrate(double (*func)(double a,double b,double z,double r),double a,double b,double z,double r0,double r1,double eps,int sd=1,double prev=0.0)
{
	double v1=0.5*(r1-r0)*(func(a,b,z,r0)+func(a,b,z,r1));
	double v=0.0;
	for (int i=1;i<=sd;i++)
		v+=func(a,b,z,r0+(i/((double)(sd+1)))*(r1-r0));
	v*=(r1-r0);
	v+=v1;
	v/=(double)(sd+1);
	if (sd<50000) // recursive subdivition
	{
		if (sd==1)
			prev=v1;
		if (fabs(v-prev)<eps)
			return v;
		else
			return integrate(func,a,b,z,r0,r1,eps,sd*2.0,v);
	}
	return v;
}
#define PI 3.14159265358979323846
double K(double a,double b,double z,double r)
{
	double k1=(1.0/(PI*a))*pow(r,(1-b)/a)*exp(-pow(r,1.0/a));
	double k2=r*sin(PI*(1-b))-z*sin(PI*(1-b+a));
	double k3=r*r-2*r*z*cos(PI*a)+z*z;
	return k1*k2/k3;
}
// two parametric mittag-leffler function
double mittag_leffler(double a,double b,double z,double eps)
{
	int k0=log(eps*(1-z))/log(z);
	int kmin=1+(int)((1-b)/a);
	if ((z>=1.0)||(z<0.0)) k0=-2.0*log(eps);
	if (k0<kmin) k0=kmin;
	double res=0.0;
	double zz=1.0;
	if (z>=0)
	{
	// do simple sum for z>0
	for (int i=0;i<k0;i++)
	{
		double v=zz/Gamma(b+a*i);
		if (finite(v))
			res+=v;
		else
			break;
		zz*=z;
	}
	}
	else
	{
		double v1,v2;
		if (((fabs(z)>3.14*a)&&(b<(1+a)))&&((a>0)&&(a<1)))
		{
			double r1=1.0;
			v1=2.0*fabs(z);
			if (v1>r1) r1=v1;
			v2=pow(-log(PI*eps/6.0),a);
			if (v2>r1) r1=v2;
			res=integrate(K,a,b,z,0.0,r1,eps);
		}
		else
		{
			zz=1.0;
			std::vector<double> values;
			// for z<0 sum + and - elements and save to array
			for (int i=0;i<k0;i+=2)
			{
				v1=zz/Gamma(b+a*i);
				zz*=z;
				v2=zz/Gamma(b+a*(i+1));
				zz*=z;
				if (finite(v1+v2))
					values.push_back(v1+v2);
				else
					break;
			}
			// sum array with sorting and pyramidal summing
			res=sum_sort_pyramid(values);
		}
	}
	return res;
}
double mittag_leffler_series(double x,double a,double b)
{
    if ((a==1.0)&&(b==1.0)) return exp(x);
    return mittag_leffler(a,b,x,1e-20);
}
