function [fig, ax, clb] = gui_surfData(Data, ImageSize, varargin)
   
    PixelSize = [1 1];
    Colormaping = 'jet';
    Optimized = false;
    Unit = '';
    
    for i=1:2:length(varargin)
        switch(varargin{i})
            case 'PixelSize'
                PixelSize = varargin{i+1};
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
    img = surf(reshape(Data, ImageSize));
    
    shading interp;
    colormap(Colormaping);
    clb = colorbar;
    title(clb, Unit);
    
    ax = img.Parent;
   
    if ~isequal(PixelSize, [1 1])
        convertX=@(x) sprintf('%.2f', x * PixelSize(1));
        convertY =@(x) sprintf('%.2f', x * PixelSize(2));

        ax.XTickLabel = cellfun(convertX, num2cell(ax.XTick'), 'UniformOutput', false);
        ax.YTickLabel = cellfun(convertY, num2cell(ax.YTick'), 'UniformOutput', false);
        ax.XTickLabelRotation = 45;
        ax.XLabel.String = 'mm';
        ax.YLabel.String = 'mm';    
    end
    
    if Optimized
        gui_optimizeColorMap(ax, clb, Data);
    end
    
end

