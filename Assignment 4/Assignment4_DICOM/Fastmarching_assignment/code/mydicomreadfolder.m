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

%Display folder name
fprintf('Reading the folder %s.\n', foldername);

%--- List DICOM files
f = dir([foldername filesep '*.dcm']);
n = numel(f);
if n == 0
    error('No .dcm files found in folder: %s', foldername);
end

%--- Read first slice
firstfile = fullfile(foldername, f(1).name);
[info1, im1] = mydicomread(firstfile);

%--- Preallocate volume (double to keep scaled values)
im = zeros(size(im1,1), size(im1,2), n);
im(:,:,1) = im1;

%--- Helpers for sorting
inst = nan(n,1);
zpos = nan(n,1);

if isfield(info1,'InstanceNumber') && ~isempty(info1.InstanceNumber)
    inst(1) = double(info1.InstanceNumber);
end
if isfield(info1,'ImagePositionPatient') && numel(info1.ImagePositionPatient) >= 3
    zpos(1) = double(info1.ImagePositionPatient(3));
end

%--- Read remaining slices
h = waitbar(0,'Reading DICOM folder...');
for k = 2:n
    filename = fullfile(foldername, f(k).name);
    [infok, imk] = mydicomread(filename);
    im(:,:,k) = imk;

    if isfield(infok,'InstanceNumber') && ~isempty(infok.InstanceNumber)
        inst(k) = double(infok.InstanceNumber);
    end
    if isfield(infok,'ImagePositionPatient') && numel(infok.ImagePositionPatient) >= 3
        zpos(k) = double(infok.ImagePositionPatient(3));
    end

    waitbar(k/n, h);
end
close(h);

%--- Sort slices (simple)
if all(~isnan(zpos))
    [~, order] = sort(zpos, 'ascend');
elseif all(~isnan(inst))
    [~, order] = sort(inst, 'ascend');
else
    order = 1:n;
end
im = im(:,:,order);

%--- Output info
info = struct();
info.Rows = info1.Rows;
info.Columns = info1.Columns;
info.PixelSpacing = info1.PixelSpacing;
info.NumSlices = n;
info.FileNames = {f(order).name};

%--- Z spacing (simple + robust enough for assignment)
% Prefer header spacing values before position-diff fallback
if isfield(info1,'SpacingBetweenSlices') && ~isnan(info1.SpacingBetweenSlices)
    dz = double(info1.SpacingBetweenSlices);
elseif isfield(info1,'SliceThickness') && ~isnan(info1.SliceThickness)
    dz = double(info1.SliceThickness);
elseif all(~isnan(zpos))
    dz = median(abs(diff(sort(zpos))));
else
    dz = NaN;
end

info.SliceSpacing = dz;
info.VoxelSize = [double(info1.PixelSpacing(1)), double(info1.PixelSpacing(2)), dz];
