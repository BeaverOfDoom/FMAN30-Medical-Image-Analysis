% Step 2 test: mydicomread on CT + MR and display with colorbar + grayscale

% --- Select CT file ---
[fn,pn] = uigetfile('*.dcm','Select CT-thorax-single.dcm');
if isequal(fn,0), error('Cancelled'); end
ctfile = fullfile(pn,fn);

% --- Select MR file ---
[fn,pn] = uigetfile('*.dcm','Select MR-heart-single.dcm');
if isequal(fn,0), error('Cancelled'); end
mrfile = fullfile(pn,fn);

% --- Read CT ---
[infoCT, Ict] = mydicomread(ctfile);
disp('--- CT summary ---')
disp(infoCT)
fprintf('CT image size: %d x %d\n', size(Ict,1), size(Ict,2));
fprintf('CT min/max: %.1f / %.1f\n', min(Ict(:)), max(Ict(:)));

figure;
imagesc(Ict); axis image off;
colormap gray; colorbar;
title('CT-thorax-single (scaled, HU)');

% --- Read MR ---
[infoMR, Imr] = mydicomread(mrfile);
disp('--- MR summary ---')
disp(infoMR)
fprintf('MR image size: %d x %d\n', size(Imr,1), size(Imr,2));
fprintf('MR min/max: %.1f / %.1f\n', min(Imr(:)), max(Imr(:)));

figure;
imagesc(Imr); axis image off;
colormap gray; colorbar;
title('MR-heart-single');
%%

exportgraphics(gcf,'figures/CT_Thorax_single.png','Resolution',300)
