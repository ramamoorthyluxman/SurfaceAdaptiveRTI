function [MapsRef, MapsCmp, SaliencyRef, SaliencyCmp, WeigthRef, WeigthCmp] = study_saliency(Acquisition_Ref, Acquisition_Cmp, DistanceAlgorithm, Descriptors_Names, Percentage, Radius, STD, Step)

% 99.9% 
% 99.5%
% 95 % 

    directory = 'Saliency';
    if ~exist('Percentage', 'var')
        Percentage = 100;
    end
    
    path{1} = fullfile(Acquisition_Ref.Path, directory, Acquisition_Cmp.Name, strcat('Sp_', num2str(Percentage)));
    path{2} = fullfile(Acquisition_Cmp.Path, directory, Acquisition_Ref.Name, strcat('Sp_', num2str(Percentage)));
    
    if exist('Radius', 'var') && Radius ~= 0
        path{1} = strcat(path{1}, '_Radius_', num2str(Radius));
        path{2} = strcat(path{2}, '_Radius_', num2str(Radius));
    end
    
    if exist('STD', 'var') && STD ~= 0
        path{1} = strcat(path{1}, '_STD_', num2str(STD));
        path{2} = strcat(path{2}, '_STD_', num2str(STD));
    end
    
    if exist('Step', 'var') && Step ~= 0
        path{1} = strcat(path{1}, '_Step_', num2str(Step));
        path{2} = strcat(path{2}, '_Step_', num2str(Step));
    end
        
    descriptors_Ref = [];
    descriptors_Cmp = [];
    fileName = 'Saliency';
    
     Acquisition_Ref.Maps = map_generateByNames(Descriptors_Names, Acquisition_Ref.Maps, Acquisition_Ref.Data, 'DataSize', Acquisition_Ref.Data_Size, 'PixelSize', Acquisition_Ref.Pixel_Size, 'LP', Acquisition_Ref.LP);
     Acquisition_Cmp.Maps = map_generateByNames(Descriptors_Names, Acquisition_Cmp.Maps, Acquisition_Cmp.Data, 'DataSize', Acquisition_Cmp.Data_Size, 'PixelSize', Acquisition_Cmp.Pixel_Size, 'LP', Acquisition_Cmp.LP);

    for i=1:length(Descriptors_Names)
        descriptors_Ref = cat(1, descriptors_Ref, Acquisition_Ref.Maps.Data.(Descriptors_Names{i}));
        descriptors_Cmp = cat(1, descriptors_Cmp, Acquisition_Cmp.Maps.Data.(Descriptors_Names{i}));
        
        fileName = strcat(fileName, '_', Descriptors_Names{i});
    end
    
    if exist('Radius', 'var') && Radius ~= 0
        [SaliencyRef, WeigthRef] = map_saliency(descriptors_Ref, Acquisition_Ref.Data_Size, DistanceAlgorithm, Percentage, Radius, STD, Step);
        [SaliencyCmp, WeigthCmp] = map_saliency(descriptors_Cmp, Acquisition_Cmp.Data_Size, DistanceAlgorithm, Percentage, Radius, STD, Step);
        
        pathWeight{1} = fullfile(path{1}, strcat(fileName, '_Weight.png'));
        pathWeight{2} = fullfile(path{2}, strcat(fileName, '_Weight.png'));
        io_saveData('Image', WeigthRef, pathWeight{1}, WeigthCmp, pathWeight{2}, Acquisition_Cmp.Data_Size, 'PixelSize', Acquisition_Cmp.Pixel_Size, 'Colormap', 'parula');
    
        [GxRef,GyRef] = gradient(reshape(WeigthRef, Acquisition_Ref.Data_Size(1), Acquisition_Ref.Data_Size(2)));
        GxRef = reshape(GxRef, 1, []);
        GyRef = reshape(GyRef, 1, []);

        [GxCmp,GyCmp] = gradient(reshape(WeigthCmp, Acquisition_Ref.Data_Size(1), Acquisition_Ref.Data_Size(2)));
        GxCmp = reshape(GxCmp, 1, []);
        GyCmp = reshape(GyCmp, 1, []); 
        
        pathGx{1} = fullfile(path{1}, strcat(fileName, '_Weight_Gx.png'));
        pathGx{2} = fullfile(path{2}, strcat(fileName, '_Weight_Gx.png'));
        io_saveData('Image', GxRef, pathGx{1}, GxCmp, pathGx{2}, Acquisition_Cmp.Data_Size, 'PixelSize', Acquisition_Cmp.Pixel_Size, 'Colormap', 'gray');
    
        pathGy{1} = fullfile(path{1}, strcat(fileName, '_Weight_Gy.png'));
        pathGy{2} = fullfile(path{2}, strcat(fileName, '_Weight_Gy.png'));
        io_saveData('Image', GyRef, pathGy{1}, GyCmp, pathGy{2}, Acquisition_Cmp.Data_Size, 'PixelSize', Acquisition_Cmp.Pixel_Size, 'Colormap', 'gray');

    else
        [SaliencyRef, WeigthRef] = map_saliency(descriptors_Ref, Acquisition_Ref.Data_Size, DistanceAlgorithm, Percentage);
        [SaliencyCmp, WeigthCmp] = map_saliency(descriptors_Cmp, Acquisition_Cmp.Data_Size, DistanceAlgorithm, Percentage);
    end
     
    [GxRef,GyRef] = gradient(reshape(SaliencyRef, Acquisition_Ref.Data_Size(1), Acquisition_Ref.Data_Size(2)));
    GxRef = reshape(GxRef, 1, []);
    GyRef = reshape(GyRef, 1, []);
    
    [GxCmp,GyCmp] = gradient(reshape(SaliencyCmp, Acquisition_Ref.Data_Size(1), Acquisition_Ref.Data_Size(2)));
    GxCmp = reshape(GxCmp, 1, []);
    GyCmp = reshape(GyCmp, 1, []);
    
    pathGx{1} = fullfile(path{1}, strcat(fileName, '_Gx.png'));
    pathGx{2} = fullfile(path{2}, strcat(fileName, '_Gx.png'));
    io_saveData('Image', GxRef, pathGx{1}, GxCmp, pathGx{2}, Acquisition_Cmp.Data_Size, 'PixelSize', Acquisition_Cmp.Pixel_Size, 'Colormap', 'gray');
    
    pathGy{1} = fullfile(path{1}, strcat(fileName, '_Gy.png'));
    pathGy{2} = fullfile(path{2}, strcat(fileName, '_Gy.png'));
    io_saveData('Image', GyRef, pathGy{1}, GyCmp, pathGy{2}, Acquisition_Cmp.Data_Size, 'PixelSize', Acquisition_Cmp.Pixel_Size, 'Colormap', 'gray');
    
    pathSaliency{1} = fullfile(path{1}, strcat(fileName, '.png'));
    pathSaliency{2} = fullfile(path{2}, strcat(fileName, '.png'));
    io_saveData('Image', SaliencyRef, pathSaliency{1}, SaliencyCmp, pathSaliency{2}, Acquisition_Cmp.Data_Size, 'PixelSize', Acquisition_Cmp.Pixel_Size, 'Colormap', 'parula');

    %% Histogramme
    
    Mean_Saliency1 = mean(SaliencyRef, 2);
    STD_Saliency1 = std(SaliencyRef, 0, 2);

    Mean_Saliency2 = mean(SaliencyCmp, 2);
    STD_Saliency2 = std(SaliencyCmp, 0, 2);

    [figures, axis] = gui_printData('Histogram', SaliencyRef, SaliencyCmp, 'NBins', 1000, 'Scale', {'log', 'log'}, 'Labels', {'Data saliency', 'Nb Pixels'}, 'Color', 'blue');
       
    annotation(figures(1), 'textbox',[.60 .60 .3 .3],'String',['\mu = ' num2str(Mean_Saliency1, 3) newline '\sigma = ' num2str(STD_Saliency1, 3)], 'FitBoxToText','on', 'FontSize', 13);
    annotation(figures(2), 'textbox',[.60 .60 .3 .3],'String',['\mu = ' num2str(Mean_Saliency2, 3) newline '\sigma = ' num2str(STD_Saliency2, 3)], 'FitBoxToText','on', 'FontSize', 13);

    io_saveAxis(axis(1), fullfile(path{1}, strcat(fileName, '_Histogramme.png')));
    io_saveAxis(axis(2), fullfile(path{2}, strcat(fileName, '_Histogramme.png')));

    close(figures(1));
    close(figures(2));
        
    MapsRef = Acquisition_Ref.Maps;
    MapsCmp = Acquisition_Cmp.Maps;

end

