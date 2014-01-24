function SaveDataPts( cX, cFileName)

%
% function SaveDataPts( cX, cFileName)
%
% cX    : N by D matrix of N points in D ddimensions
% cFileName
%

lHeader = [size(cX,1) size(cX,2)];

fid = fopen(cFileName,'wb');
fwrite(fid,lHeader,'integer*8');
fwrite(fid,cX','real*8');
fclose(fid);

return;