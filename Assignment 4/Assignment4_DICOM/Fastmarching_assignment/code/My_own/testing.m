infoCT = mydicominfo('CT-thorax-single.dcm');

disp(infoCT.Rows)
disp(infoCT.Columns)
disp(infoCT.BitsAllocated)
disp(infoCT.PixelSpacing)
disp(infoCT.RescaleSlope)
disp(infoCT.RescaleIntercept)
disp(infoCT.StartOfPixelData)

infoMR = mydicominfo('MR-heart-single.dcm');
disp(infoMR.Rows)
disp(infoMR.Columns)
disp(infoMR.BitsAllocated)
disp(infoMR.PixelSpacing)
disp(infoMR.StartOfPixelData)
