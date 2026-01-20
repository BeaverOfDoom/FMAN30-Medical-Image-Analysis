function [im,info] = mydicomreadfolder(foldername)
%[im,info] = mydicomreadfolder(foldername)
%Reads all dicom files in a folder into an image volume.
%
%-im is a three dimensional array of image data
%-info is a struct containing suitable data for voxel sizes etc.
%
%See also MYDICOMINFO, MYDICOMREAD
%
%Stub written by Einar Heiberg

%--- Initialize
im = [];
info = [];

%If called without input argument then ask for folder.
if nargin==0
    foldername = uigetdir(pwd, 'Select a folder');
end
if isequal(foldername,0)
    error('No folder selected.');
end

%--- List DICOM files
f = dir([foldername filesep '*.dcm']); % Put slices in folder
n = length(f);
if n == 0
    error('No .dcm files found in the folder');
end

% --- Read info from all slices (only metadata, for sorting)
zpos = zeros(n,1);
for k = 1:n
    filename = fullfile(foldername, f(k).name);
    infok = mydicominfo(filename);

    % Use ImagePositionPatient (z-coordinate) if it exists
    if isfield(infok,'ImagePositionPatient') && numel(infok.ImagePositionPatient) >= 3
        zpos(k) = infok.ImagePositionPatient(3);
    else
        % fallback if missing
        zpos(k) = k;
    end
end

% Sort slices by z-position
[~,order] = sort(zpos,'ascend');
f = f(order);

% --- Read first slice to know image size
firstfile = fullfile(foldername, f(1).name);
[info1, im1] = mydicomread(firstfile);

rows = size(im1,1);
cols = size(im1,2);

% Preallocate volume
im = zeros(rows, cols, n);
im(:,:,1) = im1;

% Read remaining slices
h = waitbar(0,'Reading DICOM slices...');
for k = 2:n
    filename = fullfile(foldername, f(k).name);
    [~, imk] = mydicomread(filename);
    im(:,:,k) = imk;
    waitbar(k/n, h);
end
close(h);

%--- Output info
info.Rows = info1.Rows;
info.Columns = info1.Columns;
info.PixelSpacing = info1.PixelSpacing;
info.NumSlices = n;
info.FileNames = {f.name};

% --- Slice spacing dz
if isfield(info1,'SpacingBetweenSlices') && ~isnan(info1.SpacingBetweenSlices)
    dz = double(info1.SpacingBetweenSlices);
elseif isfield(info1,'SliceThickness') && ~isnan(info1.SliceThickness)
    dz = double(info1.SliceThickness);
else
    % fallback using zpos differences
    dz = median(abs(diff(sort(zpos))));
end

info.SliceSpacing = dz;
info.VoxelSize = [double(info1.PixelSpacing(1)), double(info1.PixelSpacing(2)), dz];
end
