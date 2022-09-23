% Print data like image 
%
% Data the data to print like an image
% ImageSize the size of the data [height width]
% varagin :
% 'PixelSize' print a ruler to visualize the size of the data in
% millimeter (In default nothing is print) 
% 'Unit' print the unit of the colorbar (in default nothing is print)
% 'Colormap' colormap to use (in default 'gray')
% 'Optimized' cutoff the colormap mean +- 4*std

function [fig, ax, clb] = gui_imageData(Data, ImageSize, varargin)

    PixelSize = [1 1];
    Colormaping = 'gray';
    Optimized = 'none';
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
    
    if strcmp(Optimized, 'Data')
        Data = gui_optimizeData(Data);
    end
    
    fig = figure;
    if(size(Data, 1) == 1)
        img = imagesc(flipud(reshape(Data, ImageSize)));
        colormap(Colormaping);
        clb = colorbar;
        title(clb, Unit);
    else
        Optimized = false;
        tmp (1, :) = (Data(1, :) + 1.0 )* 125.7;
        tmp (2, :) = (Data(2, :) + 1.0 ) * 125.7;
        tmp (3, :) = Data(3, :) * 255;
        img = imagesc(flipud(reshape((tmp / 255)', [ImageSize 3])));
        clb = NaN;
    end

    ax = img.Parent;
    ax.YDir = 'normal';
    axis image;
    
    if ~isequal(PixelSize, [1 1])
        gui_addSize(ax, ImageSize, PixelSize);
    end
    
    if strcmp(Optimized, 'Colormap')
        gui_optimizeColorMap(ax, clb, Data);
    end
    
end

