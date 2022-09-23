% Print data like historam 
%
% Data the data to print like a histogram
% varagin :
% 'NBins' 
% 'Scale' linear or log
% 'Color' the color of the histogram
% 'Labels' The names of the axis

function [fig, ax] = gui_histData(Data, varargin)
    
    NBins = 100;
    Scales = ['linear', 'linear'];
    Color = 'black';
    Labels = {'X', 'Y'};

    for i=1:2:length(varargin)
        switch(varargin{i})
            case 'NBins'
                NBins = varargin{i+1};
            case 'Scale'
                Scales = varargin{i+1};
            case 'Color'
                Color = varargin{i+1};
            case 'Labels'
                Labels = varargin{i+1};
            otherwise
                error(strcat(dbstack.name, ": Erreur argument ", num2str(i)));
        end
    end
    
    fig = figure;
    switch(Scales{1})
        case 'linear'
            h = histogram(Data, NBins, 'FaceColor', Color, 'EdgeColor', Color);
        case 'log'
            [~,edges] = histcounts(log10(Data));
            h = histogram(Data, 10.^edges, 'FaceColor', Color, 'EdgeColor', Color);
    end
    
    ax = gca;
    
    ax.XScale = Scales{1};
    ax.YScale = Scales{2};
    
    xlabel(Labels{1});
    ylabel(Labels{2});
        
end

