[im, info] = mydicomreadfolder();   % pick folder

disp(info)

% show middle slice
k = round(size(im,3)/2);
figure;
imagesc(im(:,:,k)); axis image off;
colormap gray; colorbar;
title(sprintf('Middle slice (k=%d)', k));
