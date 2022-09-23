function study_dynamicRatioHist(Acquisition1, Acquisition2)

    directory = 'Dynamic_Ratio';
    if ~exist(fullfile(Acquisition1.Path, directory), 'dir')
            mkdir(fullfile(Acquisition1.Path, directory))
    end
    
    if ~exist(fullfile(Acquisition2.Path, directory), 'dir')
            mkdir(fullfile(Acquisition2.Path, directory))
    end
       
    [Acquisition1_Values, ~, ~, ~, ~] = process_intraPixelsStats(Acquisition1.Data);
    [Acquisition2_Values, ~, ~, ~, ~] = process_intraPixelsStats(Acquisition2.Data);

    fig = figure; 
    [n, xout] = hist(Acquisition1_Values, 300);
    bar(xout, n, 'barwidth', 1, 'basevalue', 1);
    set(gca, 'YScale', 'log')
    xlabel('Dynamic Ratio','FontSize',16);
    ylabel('log(Nb Pixels)','FontSize',16); 
    axis([0 max(cat(1, Acquisition1_Values(:), Acquisition2_Values(:))) 0 inf]);
    io_saveFigure(fig, fullfile(Acquisition1.Path, directory, Acquisition2.Name, 'Intra-pixel_dynamic_ratio.png'));
    delete(fig);

    fig = figure;
    [n, xout] = hist(Acquisition2_Values, 300);
    bar(xout, n, 'barwidth', 1, 'basevalue', 1);
    set(gca, 'YScale', 'log')
    xlabel('Dynamic Ratio','FontSize',16);
    ylabel('log(Nb Pixels)','FontSize',16); 
    axis([0 max(cat(1, Acquisition1_Values(:), Acquisition2_Values(:))) 0 inf]);
    io_saveFigure(fig, fullfile(Acquisition2.Path, directory, Acquisition1.Name, 'Intra-pixel_dynamic_ratio.png'));
    delete(fig);

end

