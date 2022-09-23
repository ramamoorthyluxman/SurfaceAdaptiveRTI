function [fig, ax, clb] = gui_plotData(Data, X, Y, varargin)
    
    Colormaping = 'parula';
    Optimized = false;
    Unit = '';
    
    for i=1:2:length(varargin)
        switch(varargin{i})
            case 'Colormap'
                Colormaping = varargin{i+1};
            case 'Optimized'
                Optimized = varargin{i+1};
            case 'Unit'
                Unit = varargin{i+1};
            otherwise
                error(strcat(dbstack.name, ": Erreur argument ", num2str(i)));
        end
    end

    fig = figure;
    sc = scatter3(X, Y, Data, 50, Data, 'filled');
    
    colormap(Colormaping);
    clb = colorbar;
    title(clb, Unit);
    
    ax = sc.Parent;
    
    view(2)
    
    xlabel('lu');
    ylabel('lv');
    zlabel('NG');
    
    if Optimized
        gui_optimizeColorMap(ax, clb, Data);
    end
    
end

