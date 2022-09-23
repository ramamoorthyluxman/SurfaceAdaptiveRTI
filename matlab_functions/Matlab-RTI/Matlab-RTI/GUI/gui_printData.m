% Print data like an image, a plot or a surface
% if a second data is given, they share the same dynamic
%
% Mode : the printing mode (like an image 'Image', a plot 'Plot' or a surface 'Surface')
% DataRef : the first data. It will take the same dynamic than the second
% DataCmp : the second data. It will give its dynamic to the first one
% varagin : Look at gui_imageData/gui_plotData/gui_surfData
% if DataCmp is empty DataRef keep its dynamic

% TODO : Add histogram
function [figures, axis, colorbars] = gui_printData(Mode, DataRef, DataCmp, varargin)
    
    if isequal(DataCmp, [])
        if strcmp(Mode, 'Image')
            [figures(1), axis(1), colorbars(1)] = gui_imageData(DataRef, varargin{1}, varargin{2:end});
        elseif strcmp(Mode, 'Plot')
            [figures(1), axis(1), colorbars(1)] = gui_plotData(DataRef, varargin{1}, varargin{2:end});
        elseif strcmp(Mode, 'Surface')
            [figures(1), axis(1), colorbars(1)] = gui_surfData(DataRef, varargin{1}, varargin{2:end});
        elseif strcmp(Mode, 'Histogram')
            [figures(1), axis(1)] = gui_histData(DataRef, varargin{1:end});
        end
    else
        if strcmp(Mode, 'Image')
            [figures(1), axis(1), colorbars(1)] = gui_imageData(DataRef, varargin{1}, varargin{2:end});
            [figures(2), axis(2), colorbars(2)] = gui_imageData(DataCmp, varargin{1}, 'Optimized', true, varargin{2:end});
            gui_optimizeColorMap(axis(1), colorbars(1), DataCmp);
        elseif strcmp(Mode, 'Plot')
            [figures(1), axis(1), colorbars(1)] = gui_plotData(DataRef, varargin{1}, varargin{2}, varargin{3:end});
            [figures(2), axis(2), colorbars(2)] = gui_plotData(DataCmp, varargin{1}, varargin{2}, 'Optimized', true, varargin{3:end});
            gui_optimizeColorMap(axis(1), colorbars(1), DataCmp);
        elseif strcmp(Mode, 'Surface')
            [figures(1), axis(1), colorbars(1)] = gui_surfData(DataRef, DataSize, varargin{:});
            [figures(2), axis(2), colorbars(2)] = gui_surfData(DataCmp, DataSize, 'Optimized', true, varargin{:});
            gui_optimizeColorMap(axis(1), colorbars(1), DataCmp);
        elseif strcmp(Mode, 'Histogram')
            [figures(1), axis(1)] = gui_histData(DataRef, varargin{1:end});
            [figures(2), axis(2)] = gui_histData(DataCmp, varargin{1:end});
            axis(1).XLim = axis(2).XLim;
            axis(1).YLim = axis(2).YLim;
        end

    end
    
end

