#include "mex.h"
#include <math.h>

/*////////////////////////////////////
// Written By: Einar Heiberg       //
// Last modified : 2002-03-10      //
//                                 //
// Copyrights: All rights reserved //
////////////////////////////////////*/

#define MAXDIM 10
#define MYMAX(a,b) (a>b?a;b)
#define MYMIN(a,b) (a<b?a;b)
#define KNOWN 0 /* Do not change */
#define UNKNOWN 1 /* Do not change */
#define TENTATIVE 2 /* Do not change */

/* BEGIN */

/* Declarations */
typedef struct point {
	double cost;  /* total cost */
	int ind; /* address to point */
} pointtype;

typedef struct queue {
	pointtype *basepointer;
	int elements;
	int maxelements;
} queuetype;

/* Queue Functions */
queuetype *init_queue(int s)
{
	pointtype p;
	queuetype *qp;
	
	qp = (queuetype *)mxCalloc(1,sizeof(queuetype));
	p.cost = 1E10; /* Initialize with large value */
	qp->basepointer = (pointtype *) mxCalloc(s,sizeof(pointtype));
	qp->basepointer[0] = p; /*High cost node should indicate error */
	qp->elements = 0;
	qp->maxelements = s-1;
	return(qp);
}

/* ------------ */
void delete_queue(queuetype *qp)
{
	/* Free the memory */
	mxFree(qp);
}

/* ------------ */
void put_queue(pointtype p,queuetype *qp)
{
	pointtype p2;
	int i;
	/* Check if queue is full. */
	if (qp->elements>=qp->maxelements)
		mexErrMsgTxt("Queue full.");
	else
	{
			/* Ok lets go.*/
		
			/* Update number of elements in heap. */
		qp->elements++;
		
			/* Place element last in heap. */
		qp->basepointer[qp->elements] = p;
		
			/* Rearange heap - sift up */
		i = qp->elements; /* i is the index of current position. */
		while ( (i>1)&&(qp->basepointer[i].cost < qp->basepointer[i/2].cost) )
		{
		/* Push up element by exchanging parents of larger costs. */
			p2 = qp->basepointer[i];
			qp->basepointer[i]=qp->basepointer[i/2];
			qp->basepointer[i/2] = p2;
			i = i/2;
		}
	} /* Else clasue of queue full. */
}

/* -------------- */
pointtype get_queue(queuetype *qp)
{
	pointtype p1,p2;
	int i,j;
	
	/* Return first element and reorder queue */
	if (qp->elements==0)
	{
		mexErrMsgTxt("Queue empty.");
		return(qp->basepointer[0]);    /* Return something.*/
	}
	p1 = qp->basepointer[1];           /* Always return first element*/
	p2 = qp->basepointer[qp->elements]; /* Last element*/
	qp->basepointer[1] = p2;           /* Move the last element to first position*/
	
	/* Update number of elements */
	qp->elements--;
	
	/* Restore order in heap - sift the element down */
	i = 1; /* The position of the root (which shall be sifted down */
	while (i<=(qp->elements/2))
	{
			/* Push element down the tree */
		if ( ( (qp->basepointer[2*i].cost) < (qp->basepointer[2*i+1].cost) ) ||
		( 2*i == qp->elements ))
			j = 2*i; else
				j = 2*i+1;
			/* j is now the child of i having a smaller cost */
		if (qp->basepointer[i].cost > qp->basepointer[j].cost )
		{
		/* Exchange elements */
			p2 = qp->basepointer[i];
			qp->basepointer[i] = qp->basepointer[j];
			qp->basepointer[j] = p2;
			i = j;
		} else
		{
			/* Cannot push further */
			return(p1);
		}
	}
	
	/* Return the first element (extracted before) */
	return(p1);
}

/* Add neighboors */
void addn(int size[MAXDIM],
int step[MAXDIM+1],
int dims,
int ind,
float *im,
char *pointstatus,
queuetype *qp,
double *arrivaltime)
{
	pointtype p;
	int       coord;
	int       newind;
	int       d;
	
	/* Loop over all dimensions */
	newind = ind;
	for (d=(dims-1);d>=0;d--)
	{
		coord = newind/step[d];
		newind = newind%step[d];
		
			/* Dont bother if singleton dimension */
		if (size[d]!=1)
		{
			if (coord>0)
			{
				p.ind = ind-step[d];
				if (pointstatus[p.ind]!=KNOWN)
				{
					p.cost = arrivaltime[ind]+im[p.ind];
					if (p.cost<arrivaltime[p.ind])
					{
						put_queue(p,qp);
						arrivaltime[p.ind]=p.cost;
						pointstatus[p.ind]=TENTATIVE;
					}
				}
			}
			
			if (coord<(size[d]-1))
			{
				p.ind = ind+step[d];
				if (pointstatus[p.ind]!=KNOWN)
				{
					p.cost = arrivaltime[ind]+im[p.ind];
					if (p.cost<arrivaltime[p.ind])
					{
						put_queue(p,qp);
						arrivaltime[p.ind]=p.cost;
						pointstatus[p.ind]=TENTATIVE;
					}
				}
			}
			
		}
	}
	
} /* addn */

/* Main function */
void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{
	mxArray       *startind_ptr,*maxendcost_ptr,*im_ptr;
	mxArray       *arrivaltime_ptr;
	int           dims;
	const int     *tempsize;
	int           size[MAXDIM];
	int           step[MAXDIM+1];
	int           numel;
	int           startpoints;
	int           loop;
	char          *pointstatus;
	double         *arrivaltime;
	float         *startind;
	float         *im;
	double        maxendcost;
	queuetype     *qp;
	pointtype     p;
	
	/* Create temporary matrix */
	arrivaltime_ptr = mxCreateDoubleMatrix(0,0,mxREAL);
	
	/* Assign the output with the matrix */
	plhs[0] = arrivaltime_ptr;
	
	/* Check number of arguments */
	if (nrhs==0)
	{
		printf("Usage: arrivaltime = fastmarch(im,startind,maxendcost), im and startind shall be single.\n");
		printf("Copyright Einar Heiberg 2002\n");
		return;
	}
	
	if (nrhs!=3)
		mexErrMsgTxt("Fastmarch requires three input arguments.");
	
	/* Extract pointers */
	im_ptr = (mxArray *)prhs[0];
	startind_ptr = (mxArray *)prhs[1];
	maxendcost_ptr = (mxArray *)prhs[2];
	
	/* Check argument types im*/
	if (mxIsSingle(im_ptr)==0)
		mexErrMsgTxt("im should be numeric and single.");
	
	dims = mxGetNumberOfDimensions(im_ptr);
	tempsize = mxGetDimensions(im_ptr);
	
	if (dims>MAXDIM)
		mexErrMsgTxt("Too many input dimensions.");
	
	/* Get size*/
	numel = 1;
	step[0]=1;
	
	for(loop=0;loop<MAXDIM;loop++)
	{
		size[loop]=1;
		if (loop<dims)
			size[loop]=tempsize[loop];
		numel = numel*size[loop];
		step[loop+1]=step[loop]*size[loop];
	}
	
	/* check startind */
	if (mxIsSingle(startind_ptr)==0)
		mexErrMsgTxt("Startind should be numeric and single.");
	
	startpoints = ((int) mxGetN(startind_ptr))*((int) mxGetM(startind_ptr));
	
	if (mxIsDouble(maxendcost_ptr)==0)
		mexErrMsgTxt("Maxendcost should be numeric.");
	
	/* Create output matrix */
	arrivaltime_ptr = mxCreateNumericArray(dims,
	size,
	mxDOUBLE_CLASS,
	mxREAL); /* Create matrix */
	
	arrivaltime = mxGetPr(arrivaltime_ptr); /* Address to elements in matrix */
	
	/* Assign the output with the matrix */
	plhs[0] = arrivaltime_ptr;
	
	/* Extract more pointers */
	im = (float*) mxGetData(im_ptr);
	startind = (float*) mxGetData(startind_ptr);
	
	maxendcost = *mxGetPr(maxendcost_ptr);
	
	/* Create pointstatus matrix */
	pointstatus = mxCalloc(numel,sizeof(char));
	
	/* Start of calculations */
	
	for (loop=0;loop<numel;loop++)
	{
		arrivaltime[loop]=maxendcost;
		pointstatus[loop]=UNKNOWN;
	}
	
	/* Initialize the queue */
	qp = init_queue(numel);
	
	/* Initialize startpoint(s) */
	for(loop=0;loop<startpoints;loop++)
	{
		
			/* Check out of range */
		if (startind[loop]>numel)
			mexErrMsgTxt("Out of range startpoint.");
		
			/* Set arrivaltime to zero at startpoints*/
		arrivaltime[((int)startind[loop])-1]=0;
		
			/* Set pointstatus to known at startpoints */
		pointstatus[((int)startind[loop])-1]=KNOWN;
		
			/* Add neighbours as active nodes */
		addn(size,step,dims,(int)(startind[loop])-1,im,pointstatus,qp,arrivaltime);
		
	} /* Initalizing startpoints */
	
	/* --- Main loop --- */
	
	p.cost = 0;
	while ( ((qp->elements)>0)&&(p.cost<maxendcost) )
	{
		p = get_queue(qp);
		
			/* Check if point already known => ignore */
		if (pointstatus[p.ind]!=KNOWN)
		{

			
		/* Ok not known => make it */
			pointstatus[p.ind] = KNOWN;
			
		/* update cost */
			arrivaltime[p.ind] = p.cost;
			
		/* add new neigbhboors */
			addn(size,step,dims,p.ind,im,pointstatus,qp,arrivaltime);
			
		}
		
	}
	
	/* Free allocated stuff */
	delete_queue(qp);
	mxFree(pointstatus);
}

/* END */
