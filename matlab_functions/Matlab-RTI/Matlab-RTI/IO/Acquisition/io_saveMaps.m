function io_saveMaps(Acquisition1, Acquisition2, MapsNames, colormap)

        options = {Acquisition1.DataSize, 'PixelSize', Acquisition1.PixelSize, 'Optimized', 'Data', 'Colormap', colormap, 'Visible', 'off'};

        if ~isempty(Acquisition2)
            
            dir1 = fullfile(process_generatePathAcquisition(Acquisition1), 'Maps', 'Compared_With', Acquisition2.Name);
            dir2 = fullfile(process_generatePathAcquisition(Acquisition2), 'Maps', 'Compared_With', Acquisition1.Name);
                
            for i=1:length(MapsNames)
                if ~strcmp(MapsNames{i}, 'Normal') && ~strcmp(MapsNames{i}, 'DMDInvariant') && ~strcmp(MapsNames{i}, 'PTM') && ~strcmp(MapsNames{i}, 'HSH') && ~strcmp(MapsNames{i}, 'RBF') && ~strcmp(MapsNames{i}, 'AngularNormal') && ~strcmp(MapsNames{i}, 'AzimuthAngle')
                    mapPath1 = fullfile(dir1, MapsNames{i}, strcat(MapsNames{i}, '_', colormap, '.png'));
                    mapPath2 = fullfile(dir2, MapsNames{i}, strcat(MapsNames{i}, '_', colormap, '.png'));
                    if ~exist(mapPath1, 'file') || ~exist(mapPath2, 'file')
                        if map_isComputed(Acquisition1.Maps, MapsNames{i}) && map_isComputed(Acquisition2.Maps, MapsNames{i})
                            io_saveData('Image', Acquisition1.Maps.Data.(MapsNames{i}), mapPath1, Acquisition2.Maps.Data.(MapsNames{i}), mapPath2, options{:});
                        end
                    end
                end
            end
            
            dir = fullfile(process_generatePathAcquisition(Acquisition2), 'Maps'); 

            
            for i=1:length(MapsNames)
                mapPath2 = fullfile(dir, MapsNames{i}, strcat(MapsNames{i}, '_', colormap, '.png'));
                if ~exist(mapPath2, 'file')
                    if map_isComputed(Acquisition2.Maps, MapsNames{i})
                        io_saveData('Image', Acquisition2.Maps.Data.(MapsNames{i}), mapPath2, [], [], options{:});
                    end
                end
            end
        end
        
        dir = fullfile(process_generatePathAcquisition(Acquisition1), 'Maps'); 
            
        for i=1:length(MapsNames)
            mapPath1 = fullfile(dir, MapsNames{i}, strcat(MapsNames{i}, '_', colormap, '.png'));
            if map_isComputed(Acquisition1.Maps, MapsNames{i})
                if ~strcmp(MapsNames{i}, 'DMDInvariant') && ~strcmp(MapsNames{i}, 'PTM') && ~strcmp(MapsNames{i}, 'HSH') && ~strcmp(MapsNames{i}, 'RBF') && ~strcmp(MapsNames{i}, 'AngularNormal') && ~strcmp(MapsNames{i}, 'AzimuthAngle')
                    io_saveData('Image', {Acquisition1.Maps.Data.(MapsNames{i})}, {mapPath1}, options{:});
                elseif strcmp(MapsNames{i}, 'AngularNormal')
                    io_saveData('Image', {Acquisition1.Maps.Data.(MapsNames{i})}, {mapPath1}, options{:}, 'Colormap', 'HSV');
                elseif strcmp(MapsNames{i}, 'AzimuthAngle')
                    io_saveData('Image', {cat(1, Acquisition1.Maps.Data.('AzimuthAngle')/360, ones(size(Acquisition1.Maps.Data.('DipAngle'))), ones(size(Acquisition1.Maps.Data.('DipAngle'))))}, {mapPath1}, options{:}, 'Colormap', 'HSV');
                else
                    if strcmp(MapsNames{i}, 'RBF')
                        for j=1:size(Acquisition1.Maps.Data.(MapsNames{i}).rbfcoeff, 2)
                            mapPath1 = fullfile(dir, MapsNames{i}, strcat(MapsNames{i}, '_', num2str(j), '_', colormap, '.png'));
                            io_saveData('Image', {Acquisition1.Maps.Data.(MapsNames{i}).rbfcoeff(:,j)'}, {mapPath1}, options{:});
                        end
                    else
                        for j=1:size(Acquisition1.Maps.Data.(MapsNames{i}), 1)
                            mapPath1 = fullfile(dir, MapsNames{i}, strcat(MapsNames{i}, '_', num2str(j), '_', colormap, '.png'));
                            io_saveData('Image', {Acquisition1.Maps.Data.(MapsNames{i})(j,:)}, {mapPath1}, options{:});
                        end
                    end
                end
            end
        end
end

