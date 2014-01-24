function X = mSVD_ReadS( cFileName )

lFile = fopen(cFileName,'rb');

X.N = fread( lFile,1,'integer*8=>ulong')';
X.J = fread( lFile,1,'integer*8=>ulong')';

for j = 1:X.J,
    X.Nets(j).n = fread( lFile,1,'integer*8=>ulong' );
    X.Nets(j).NetStats.nS = fread( lFile,1,'integer*8=>ulong' );
    X.Nets(j).idxs = fread( lFile,uint32(X.Nets(j).n),'integer*8=>ulong' );
    try
    X.Nets(j).NetStats.S = reshape(fread( lFile, double(X.Nets(j).n * X.Nets(j).NetStats.nS), 'real*8=>double' ),[X.Nets(j).NetStats.nS,X.Nets(j).n]);
    catch
        1,
    end;
end;

%figure;plot(X.Nets(10).NetStats.S);

fclose(lFile);

return;