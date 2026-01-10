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
%Calculate speed image. Here, use your imagination to calculate a good
%speed image based on image intensity at the seed, slider settings,
%and different features from the image.

global GUI

%%%% --- Calculate speed image
%Replace this code. This is likely going to be several lines of code
%involving one or more sliders and mix pre-processing such as filtering,
%edge detection or whatever you can come up with. Note that your function
%may very well be nonlinear. Remember the speed function should be high for
%values that you want to include and low were you want the expansion to 
%stop/slow down. Have fun!

%Calculate the intensity at the seed in case it is useful.
intensityatseed = GUI.IM(GUI.YSeed,GUI.XSeed);

%Just anything for now, replace here!
GUI.SPEED = (GUI.IM-GUI.IM-intensityatseed)*GUI.Slider1;
GUI.SPEED(GUI.SPEED<1e-2) = 1e-2;

tic;
switch get(GUI.Handles.methodlistbox,'value')
  case 1
    %Full code
    GUI.MAP = msfm2d(GUI.SPEED,[GUI.YSeed;GUI.XSeed],true,true); %Call the fast marching code.
  case 2
    %No curve    
    GUI.MAP = msfm2d(GUI.SPEED,[GUI.YSeed;GUI.XSeed],false,true); %Call the fast marching code.    
  case 3
    %No cross
    GUI.MAP = msfm2d(GUI.SPEED,[GUI.YSeed;GUI.XSeed],true,false); %Call the fast marching code.        
  case 4
    GUI.MAP = msfm2d(GUI.SPEED,[GUI.YSeed;GUI.XSeed],false,false); %Call the fast marching code.            
  case 5
    startind = round(GUI.XSeed)*size(GUI.IM,1)+round(GUI.YSeed); %Calculate startindex
    try
      GUI.MAP = fastmarch(single(1./GUI.SPEED),single(startind),1e10); %Call the fast marching code.    
    catch me
      error('The function fastmarch does not seem to be compiled.');
    end
end
t = toc;
set(GUI.Handles.calculationtimetext,'string',sprintf('Calculation time: %0.5g [s]',t));
