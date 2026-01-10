% STEP 1: Test that mydicominfo finds the tags

[fn,pn] = uigetfile('*.dcm','Select CT-thorax-single.dcm');
if isequal(fn,0), error('Cancelled'); end
ctfile = fullfile(pn,fn);

infoCT = mydicominfo(ctfile);

% Show only the important fields
%disp('--- CT info ---')
disp(['Rows: ' num2str(infoCT.Rows)])
disp(['Columns: ' num2str(infoCT.Columns)])
disp(['BitsAllocated: ' num2str(infoCT.BitsAllocated)])
disp(['PixelRepresentation: ' num2str(infoCT.PixelRepresentation)])
disp(['PixelSpacing: ' mat2str(infoCT.PixelSpacing)])
disp(['RescaleSlope: ' num2str(infoCT.RescaleSlope)])
disp(['RescaleIntercept: ' num2str(infoCT.RescaleIntercept)])
disp(['StartOfPixelData: ' num2str(infoCT.StartOfPixelData)])
