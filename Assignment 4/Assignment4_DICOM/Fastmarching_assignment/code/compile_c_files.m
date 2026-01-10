% This script will compile all the C files
clear msfm2d;
mex('msfm2d.c');
clear msfm3d;
mex('msfm3d.c');
clear fastmarch.c;
mex('fastmarch.c');
