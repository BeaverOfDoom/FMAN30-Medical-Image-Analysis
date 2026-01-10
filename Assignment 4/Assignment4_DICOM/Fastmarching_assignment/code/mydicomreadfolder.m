function [im,info] = mydicomreadfolder(foldername)
%[im,info] = mydicomreadfolder(foldername)
%Reads all dicom files in a folder into an image volume.
%
%-im is a three dimensional array of image data
%-info is a struct containing suitable data for voxel sizes etc.
%
%See also MYDICOMINFO, MYDICOMREAD
%
%This function is just a stub and you need to write it.

%Stub written by Einar Heiberg

%Hint:
%To get all files called in a folder, use the function 
%f = dir([foldername filesep '*.dcm'])

%Hint: Consider preallocating data for the sake of speed.

%Hint: waitbar is a good way of updating regarding progress.

%--- Initialize
im = [];
info = [];

%If called without input argument then ask for folder.
if nargin==0
  foldername = uigetdir(pwd, 'Select a folder');
end;

%Display folder name
disp(sprintf('Reading the folder %s.',foldername)); %#ok<DSPS>

% Listing the DICOM files
 f = dir([foldername filesep '*.dcm']);
 n = numel(f);
 if n == 0
     error('No .dcm files found in the folder :3, %s', foldername);
 end

 % Read first slice to get size and base information

 firstfile = fullfile(foldername, f(1).name);
 [info1, im1] = mydicomread(firstfile);

 % Preallocate volume
im = zeros(size(im1,1), size(im1,2), n, 'double');
im(:,:,1) = im1;

% Collect slice sorting key (InstanceNumber)
inst = nan(n,1);
if isfield(info1,'InstanceNumber')
    inst(1) = double(info1.InstanceNumber);
end

% Also collect z-position if available (more robust)
zpos = nan(n,1);
if isfield(info1,'ImagePositionPatient') && numel(info1.ImagePositionPatient)>=3
    zpos(1) = double(info1.ImagePositionPatient(3));
end

%--- Read remaining slices
h = waitbar(0,'Reading DICOM folder...');
cleanupObj = onCleanup(@() close(h));

for k = 2:n
    filename = fullfile(foldername, f(k).name);
    [infok, imk] = mydicomread(filename);
    im(:,:,k) = imk;

    if isfield(infok,'InstanceNumber')
        inst(k) = double(infok.InstanceNumber);
    end
    if isfield(infok,'ImagePositionPatient') && numel(infok.ImagePositionPatient)>=3
        zpos(k) = double(infok.ImagePositionPatient(3));
    end

    waitbar(k/n, h);
end

%--- Sort slices
% Prefer z-position, otherwise InstanceNumber, otherwise leave as-is
if all(~isnan(zpos))
    [~, order] = sort(zpos, 'ascend');
elseif all(~isnan(inst))
    [~, order] = sort(inst, 'ascend');
else
    order = 1:n;
end

im = im(:,:,order);

%--- Build output info struct (voxel size etc.)
info = struct();
info.Rows = info1.Rows;
info.Columns = info1.Columns;
info.PixelSpacing = info1.PixelSpacing;

% z spacing
if isfield(info1,'SpacingBetweenSlices') && ~isnan(info1.SpacingBetweenSlices)
    info.SliceSpacing = double(info1.SpacingBetweenSlices);
elseif all(~isnan(zpos))
    info.SliceSpacing = median(abs(diff(sort(zpos))));
elseif isfield(info1,'SliceThickness') && ~isnan(info1.SliceThickness)
    info.SliceSpacing = double(info1.SliceThickness);
else
    info.SliceSpacing = NaN;
end

% voxel size as [dy dx dz] in mm
info.VoxelSize = [double(info1.PixelSpacing(1)), double(info1.PixelSpacing(2)), double(info.SliceSpacing)];

info.NumSlices = n;
info.FileNames = {f(order).name};




