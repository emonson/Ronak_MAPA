function X = Read3MatrixFromFile( cFileName )

lFile = fopen(cFileName,'rb');
lDims = fread( lFile,3,'integer*8')';

X = zeros(lDims);

for k=1:lDims(1),
    for j=1:lDims(2),
        X(k,j,:) = fread(lFile,lDims(3),'real*8');
    end;
end;

return;