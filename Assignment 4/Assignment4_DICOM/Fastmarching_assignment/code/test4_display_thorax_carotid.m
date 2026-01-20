% Part 2 - Step 1: Display central slices in axial/transversal, coronal, sagittal

[vol, info] = mydicomreadfolder();   % pick folder MR-thorax-transversal

dy = info.VoxelSize(1);   % mm per pixel in rows
dx = info.VoxelSize(2);   % mm per pixel in cols
dz = info.VoxelSize(3);   % mm per slice

[Ny, Nx, Nz] = size(vol);

% Pixel coords
x = (0:Nx-1) * dx;   % Left/Right axis in mm (image columns)
y = (0:Ny-1) * dy;   % Anterior/Posterior axis in mm (image rows)
z = (0:Nz-1) * dz;   % Feet/Head axis in mm (slice index)

% Central indices
ix = round(Nx/2);
iy = round(Ny/2);
iz = round(Nz/2);

% Flip
flipLR = true;  % left-right flip (columns)
flipAP = true;  % anterior-posterior flip (rows)

V = vol;
if flipLR
    V = V(:, end:-1:1, :);
    x = x(end:-1:1);
end
if flipAP
    V = V(end:-1:1, :, :);
    y = y(end:-1:1);
end

% 1) Transversal view: central slice in z
axial = V(:,:,iz);

figure;
imagesc(x, y, axial);
axis image;
colormap gray; colorbar;
xlabel('Right/Left (mm)');
ylabel('Anterior/Posterior (mm)');
title(sprintf('Transversal (axial) central slice, k=%d', iz));

% 2) Coronal view: slice in y (rows) 

coronal = squeeze(V(iy,:,:));  

figure;
imagesc(x, z, coronal');  % transpose so x is horizontal, z vertical
axis image;
colormap gray; colorbar;
xlabel('Right/Left (mm)');
ylabel('Feet/Head (mm)');
title(sprintf('Coronal central slice, y=%d', iy));

% 3) Sagittal view: slice in x (columns) 
sagittal = squeeze(V(:,ix,:)); 

figure;
imagesc(y, z, sagittal');  % transpose so y is horizontal, z vertical
axis image;
colormap gray; colorbar;
xlabel('Anterior/Posterior (mm)');
ylabel('Feet/Head (mm)');
title(sprintf('Sagittal central slice, x=%d', ix));

%% Part 2: Dataset 2 (MR-carotid-coronal) - Maximum Intensity Projection MIP

% Load carotid volume
[carVol, carInfo] = mydicomreadfolder();

dy = carInfo.VoxelSize(1);
dx = carInfo.VoxelSize(2);
dz = carInfo.VoxelSize(3);

[Ny, Nx, Nz] = size(carVol);

x = (0:Nx-1) * dx;   % Right/Left (mm)
y = (0:Ny-1) * dy;   % Anterior/Posterior (mm)
z = (0:Nz-1) * dz;   % Feet/Head (mm)

% Physical size in mm 
size_RL = Nx * dx;
size_AP = Ny * dy;
size_FH = Nz * dz;

fprintf('\n--- Carotid volume size (mm) ---\n');
fprintf('Right/Left: %.1f mm\n', size_RL);
fprintf('Anterior/Posterior: %.1f mm\n', size_AP);
fprintf('Feet/Head: %.1f mm\n', size_FH);

% MIP in transversal (axial) plane
mip_axial = max(carVol, [], 3);

figure;
imagesc(x, y, mip_axial);
axis image;
colormap gray; colorbar;
xlabel('Right/Left (mm)');
ylabel('Anterior/Posterior (mm)');
title('Carotid MIP - Transversal (axial)');

% MIP in coronal plane: max along y
mip_coronal = squeeze(max(carVol, [], 1)); 

figure;
imagesc(x, z, mip_coronal');  % transpose so x horizontal, z vertical
axis image;
colormap gray; colorbar;
xlabel('Right/Left (mm)');
ylabel('Feet/Head (mm)');
title('Carotid MIP - Coronal');

% MIP in sagittal plane: max along x 
mip_sagittal = squeeze(max(carVol, [], 2));

figure;
imagesc(y, z, mip_sagittal'); % transpose so y horizontal, z vertical
axis image;
colormap gray; colorbar;
xlabel('Anterior/Posterior (mm)');
ylabel('Feet/Head (mm)');
title('Carotid MIP - Sagittal');



