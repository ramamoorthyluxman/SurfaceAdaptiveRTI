function [Maps] = map_generateByName(MapName, Maps, Data, varargin)
   
    PixelSize = [];
    DataSize = [];
    LP = [];
    
    for i=1:2:length(varargin)
        switch(varargin{i})
            case 'PixelSize'
                PixelSize = varargin{i+1};
            case 'DataSize'
                DataSize = varargin{i+1};
            case 'LP'
                LP = varargin{i+1};
            otherwise
                error(strcat(dbstack.name, ": Erreur argument ", num2str(i)));
        end
    end

    MapsData = {'Mean', 'Median', 'StandardDeviation', ...
                'Skewness', 'Kurtosis', 'VariationCoefficient', ...
                'Energy', 'Entropy', 'Max', 'Min', 'Mode', ...
                'Quantile1', 'Quantile3'};

    MapsLP = {'Normal', 'Dx', 'Dy', 'Dz', ...
              'DipAngle'};

    MapsDataSize = {'Kxx', 'Kxy', 'Kyy', 'Kmin', 'Kmax'...
                    'Kmean', 'Kgaussian', 'KMehlum', 'ShapeIndex', ...
                     'Curvedness', 'KminVector', 'KmaxVector'};

    MapsPixelSize = {'RiseX', 'RiseY', 'RiseXY'};

    map_function = str2func(Maps.Functions.(MapName));
    switch MapName
        case MapsData
            Maps = map_function(Maps, Data);
        case MapsLP
            Maps = map_function(Maps, Data, LP);
        case MapsDataSize
            Maps = map_function(Maps, Data, LP, DataSize);
        case MapsPixelSize
            Maps = map_function(Maps, Data, LP, PixelSize);
    end
end

