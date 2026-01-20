function [varargout] = interactive(arg,varargin)
%Simple interactive medical image segmentation tool. 
%
%Part of assignment in the course Medical Image Analysis
%FMAN30, Centre of Mathematical Sciences, Engineering Faculty, Lund University.

%Einar Heiberg

%#ok<*GVMIS> 

if nargin==0
  %Called with no input arguments then initialize
  init;
else
  %Call the appropriate subfunction
  [varargout{1:nargout}] = feval(arg,varargin{:}); % FEVAL switchyard  
end

%------------
function init
%------------
global GUI 

GUI = []; %Start out clean;

GUI.Fig = openfig('interactive.fig');

%Extract handles
GUI.Handles = guihandles(GUI.Fig);

%Initialize
GUI.IM = []; %Loaded by open_Callback
GUI.Info = []; %Set by open_Callback
GUI.SPEED = [];
GUI.MAP = [];
GUI.Slider1 = 0; %Updated later by slider_Callback
GUI.Slider2 = 0;
GUI.Slider3 = 0; 
GUI.ArrivalTime = 0;
GUI.XSeed = []; %Updated by load_Callback and xxx
GUI.YSeed = [];

%Get values from sliders
slider_Callback;

%Make the images invisible
set(GUI.Handles.imaxes,'visible','off');
set(GUI.Handles.mapaxes,'visible','off');

return;

%---------------------
function open_Callback 
%---------------------
%Called when user clicks on open button
global GUI

%Ask for filename
[filename, pathname] = uigetfile('*.dcm', 'Pick a DICOM file');

[info,im] = mydicomread([pathname filesep filename]);

%Test if mydicomread is not implemted yet
if isempty(im)
  msgbox('mydicomread does not seem to be implemented yet. Loading standard image.');
  
  %MR case
  %load('images/testmr.mat'); %Just load one file
  %im = im(1:100,:);
  
  %CT
  im = zeros(100,100);
  im(40:70,40:90) = 1;
  
end

%Store data
GUI.Info = info;
GUI.IM = double(im);

%Make the images visible
set(GUI.Handles.imaxes,'visible','on');
set(GUI.Handles.mapaxes,'visible','on');

%--- Display the data

%Consider to change here to get correct image proportions..
GUI.Handles.image = image(getimage,'parent',GUI.Handles.imaxes);
axis(GUI.Handles.imaxes,'image','off');

%Add seed point
GUI.XSeed = round(size(GUI.IM,2)/2);
GUI.YSeed = round(size(GUI.IM,1)/2);

hold(GUI.Handles.imaxes,'on');
GUI.Handles.seedpoint = plot(GUI.Handles.imaxes,GUI.XSeed,GUI.YSeed,'yo');
hold(GUI.Handles.imaxes,'off');

%Set callback for button down
set(GUI.Handles.image,'ButtonDownFcn','interactive(''buttondown_Callback'')');

%Calculate speed image and update
calculatespeedimage; %Store into GUI
update_Callback;

%---------------------
function im = getimage
%---------------------
%Returns the loaded image as RGB image with/without overlay
global GUI

minvalue = min(GUI.IM(:));
maxvalue = max(GUI.IM(:));

%If constant image return zero image
if isequal(minvalue,maxvalue)
  im = uint8(zeros([size(GUI.IM) 3]));
  return;  
end

%Calculate the RGB images
imr = uint8(255*(GUI.IM-minvalue)/(maxvalue-minvalue));
img = imr;
imb = imr;

%Apply overlay
if get(GUI.Handles.overlaycheckbox,'value')    
  if ~(isempty(GUI.MAP) || isempty(GUI.Thres))
    overlay = (GUI.MAP<GUI.Thres);
    imb(overlay) = uint8(255);
  end
end

%Return make it a n*m*3 array
im = cat(3,imr,img,imb);

%-----------------------
function update_Callback
%-----------------------
global GUI

if isempty(GUI.SPEED) || isempty(GUI.MAP)
  return;
end

%Get the position of the listbox selector
v = get(GUI.Handles.displaylistbox,'value');

%The two different display options
switch v
  case 1
    %Display the speed image
    imagesc(GUI.SPEED,'parent',GUI.Handles.mapaxes);    
  case 2
    %Display 1/speed image
    imagesc(1./GUI.SPEED,'parent',GUI.Handles.mapaxes);        
  case 3
    %Display arrival time map
    imagesc(GUI.MAP,'parent',GUI.Handles.mapaxes);    
  otherwise
    error('Unknown option.');
end

%Update the standard image
set(GUI.Handles.image,'cdata',getimage);

colorbar('peer',GUI.Handles.mapaxes);
axis(GUI.Handles.mapaxes,'image','off');

%---------------------------
function buttondown_Callback 
%---------------------------
global GUI

if isempty(GUI.IM)
  return;
end

type = get(GUI.Fig,'SelectionType');

p = get(GUI.Handles.imaxes,'CurrentPoint');
y = p(3);
x = p(1);

switch type
  case 'normal' 
    %Left click
    GUI.XSeed = round(x);
    GUI.YSeed = round(y);
    set(GUI.Handles.seedpoint,'xdata',round(x),'ydata',round(y));    
    calculatespeedimage; %Store into GUI
    update_Callback;
  case 'alt'
    %Right click

    %Extract speed
    yedge = round(y);
    xedge = round(x);
    GUI.Thres = GUI.MAP(yedge,xedge);
    
    %Update slider & text
    maxvalue = max(GUI.MAP(:));
    percentage = 100*GUI.Thres/maxvalue;
    set(GUI.Handles.arrivalslider,'value',percentage);
    set(GUI.Handles.arrivaltext,'string',sprintf('%0.5g',GUI.ArrivalTime));
    
    %figure(22);
    %imagesc(GUI.MAP<GUI.Thres);
end

%Update graphically
update_Callback;

%-----------------------
function slider_Callback
%-----------------------
%This function is called when the user changes the slider

global GUI

GUI.Slider1 = get(GUI.Handles.slider1,'value');
GUI.Slider2 = get(GUI.Handles.slider2,'value');
GUI.Slider3 = get(GUI.Handles.slider3,'value');
GUI.ArrivalTime = get(GUI.Handles.arrivalslider,'value');

%Calculate the threshold
maxarrival = max(GUI.MAP(:));
GUI.Thres = (GUI.ArrivalTime/100)*maxarrival;

%Update edit boxes
set(GUI.Handles.slider1text,'string',sprintf('%0.5g',GUI.Slider1));
set(GUI.Handles.slider2text,'string',sprintf('%0.5g',GUI.Slider2));
set(GUI.Handles.slider3text,'string',sprintf('%0.5g',GUI.Slider3));
set(GUI.Handles.arrivaltext,'string',sprintf('%0.5g',GUI.ArrivalTime));

%If no image present the do no more.
if isempty(GUI.IM)
  return;
end

calculatespeedimage; %Store into GUI
update_Callback;

%----------------------
function close_Callback 
%----------------------
%Close the user interface
global GUI

close(GUI.Fig);

%---------------------------
function calculatespeedimage
%---------------------------
global GUI

I = double(GUI.IM);

% Seed intensity
I0 = I(GUI.YSeed, GUI.XSeed);

% Slider 1: intensity tolerance 
sigma = 1 + 5*GUI.Slider1;

% Speed: intensity similarity to seed 
region = exp( - (I - I0).^2 / (2*sigma^2) );

% Avoid zero speed
eps0 = 1e-3;
GUI.SPEED = eps0 + region;
 
% Run fast marching
tic;
switch get(GUI.Handles.methodlistbox,'value')
    case 1
        GUI.MAP = msfm2d(GUI.SPEED,[GUI.YSeed;GUI.XSeed],true,true);
    case 2
        GUI.MAP = msfm2d(GUI.SPEED,[GUI.YSeed;GUI.XSeed],false,true);
    case 3
        GUI.MAP = msfm2d(GUI.SPEED,[GUI.YSeed;GUI.XSeed],true,false);
    case 4
        GUI.MAP = msfm2d(GUI.SPEED,[GUI.YSeed;GUI.XSeed],false,false);
end
t = toc;
set(GUI.Handles.calculationtimetext,'string',sprintf('Calculation time: %0.5g [s]',t));

