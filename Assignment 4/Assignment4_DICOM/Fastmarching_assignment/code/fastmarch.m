%arrivalmap = fastmarch(im,startind,maxendcost);
%Implements a fastmarching code with a simplistic stencil implementation. 
%Note that the code is generic on dimensionality.
%
%Please note that the im should be single. The code is implemented as a mex
%file and needs to be compiled.
%
%Einar Heiberg 2002
