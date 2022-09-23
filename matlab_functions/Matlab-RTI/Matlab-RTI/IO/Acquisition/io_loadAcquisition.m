function [Acquisition] = io_loadAcquisition(varargin)

    if nargin == 0
        Path = uigetdir;
        if isequal(Path, 0)
             disp('Selection canceled');
             return;
        end
    end
    
    Type = 'single';
    Channel = 'Gray';
    
    if nargin > 0
        Path = varargin{1};
    end
    
    if nargin > 1
        Type = varargin{2};
    end
    
    if nargin > 2
        Channel = varargin{3};
    end
    
    if nargin > 3
        Crop = varargin{4};
    end
    
    if nargin > 4
        Zoom = varargin{5};
    end
    
    Acquisition = struct_acquisition();
    pathSplited = split(Path, '\');
    
    Acquisition.Path = Path;
    if contains(Acquisition.Path, 'RGB')
        Acquisition.Modality = 'RGB-RTI';
    elseif contains(Acquisition.Path, 'HDR')
        Acquisition.Modality = 'HD-RTI';
    else
        Acquisition.Modality = 'RTI';
    end
    Acquisition.Name = pathSplited{end};

    Acquisition.LP = io_loadLP(Path);
    Acquisition.Channel = Channel;
    
    [Acquisition.Data, Acquisition.DataSize(1), Acquisition.DataSize(2)] = io_loadImages(Acquisition.Path, Acquisition.LP.Names, Type, Channel);
    
    if exist('Zoom', 'var') && ~isempty(Zoom)
        [Acquisition.PixelSize(1), Acquisition.PixelSize(2)] = process_computeSize((Zoom - 1) / (12.5 - 1), [4112 3008]);
        Acquisition.Zoom = Zoom;
    else
        Acquisition.PixelSize = [1 1];
        Acquisition.Zoom = NaN;
    end
    
    if exist('Crop', 'var') && ~isempty(Crop)
        [Acquisition, Acquisition.Cropped] = process_cropAcquisition(Acquisition, Crop);
    else
        Acquisition.Cropped = [];
    end
    
    Acquisition.Data = Acquisition.Data / 255; 
    Acquisition.NbData = size(Acquisition.Data, 1);
end

