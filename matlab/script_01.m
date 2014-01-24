% Data set
pDataSetName = 'D-Sphere';

for k = 4,
    system('rm GeometricMultiResolutionAnalysis.out');
    k,
    X = GenerateDataSets(pDataSetName, struct('NumberOfPoints',10000,'Dim',k,'EmbedDim',200,'NoiseType','Gaussian','NoiseParam',0.0) )';
    % Save in appropriate format
    SaveDataPts(X,'Sphere.pts');
    
    % Call C program
    [lRet,lOutput]=system('/Users/mauromaggioni/Library/Developer/Xcode/DerivedData/GeometricMultiResolutionAnalysis-ecqfwburzbqitucmpzymvhsiiqmg/Build/Products/Debug/GeometricMultiResolutionAnalysis -df Sphere.pts');
    
    % Load results
    %S = Read3MatrixFromFile('GeometricMultiResolutionAnalysis.out');
    %Sv{k} = squeeze(mean(S,1));    
    MSVD = mSVD_ReadS('GeometricMultiResolutionAnalysis.out');
    
    for j = 1:length(MSVD.Nets),
        S(j,1:size(MSVD.Nets(j).NetStats.S,1)) = mean(MSVD.Nets(j).NetStats.S,2);
    end;
    
    figure;plot(S);title(sprintf('MSVD for %s, dimension %d.',pDataSetName,k));
end;