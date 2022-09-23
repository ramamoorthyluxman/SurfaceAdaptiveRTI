function [path, file] = study_dynamicRatioStats(Acquisition)
    
    directory = 'Dynamic_Ratio';
    file = 'dynamic Ratio.txt';
    
    path = fullfile(Acquisition.Path, directory);
    if ~exist(path, 'dir')
            mkdir(path)
    end
    
    fileID = fopen(fullfile(path, file),'w');
    if fileID == -1
      error('Author:Function:OpenFile', 'Cannot open file: %s', fullfile(path, file));
    end
    
    [~, meanRatio, stdRatio, minRatio, maxRatio] = process_intraPixelsStats(Acquisition.Data);

    fprintf(fileID, 'intra-pixels mean : %f\n',     meanRatio);
    fprintf(fileID, 'intra-pixels std : %f\n',          stdRatio);
    fprintf(fileID, 'intra-pixels min : %f\n',        minRatio);
    fprintf(fileID, 'intra-pixels max : %f\n',        maxRatio);

    [meanRatio, stdRatio, minRatio, maxRatio] = process_intraImagesStats(Acquisition.Data);

    fprintf(fileID, 'intra-images mean : %f\n',    meanRatio);
    fprintf(fileID, 'intra-images std : %f\n',         stdRatio);
    fprintf(fileID, 'intra-images min : %f\n',       minRatio);
    fprintf(fileID, 'intra-images max : %f\n',       maxRatio);

    if contains(Acquisition.Path, 'HDR')
        d = dir(fullfile(Acquisition.Path, 'LDR*.png'));
        fprintf(fileID, 'Number of LDR images %d\n', size(d, 1) );

        h = zeros(size(Acquisition.Data, 1), 1);
        for i = 1:size(d, 1)
            h(str2num(d(i).name(5:7)) + 1) = h(str2num(d(i).name(5:7)) + 1) + 1;
        end

        fprintf(fileID, 'Maximum number image LDR %d\n', max(h(:)));
        fprintf(fileID, 'Minimum number image LDR %d\n', min(h(:)));
        fprintf(fileID, 'Mean number image LDR %f\n', mean(h(:)));
        fprintf(fileID, 'Std number image LDR %f\n', std(h(:)));
    end
    
    fclose(fileID);
    
end

