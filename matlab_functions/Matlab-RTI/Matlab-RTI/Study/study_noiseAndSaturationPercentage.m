function study_noiseAndSaturationPercentage(Acquisition1, Acquisition2, limits, colors)
    
    directory = 'Dynamic_Limitation';
    
    if contains(Acquisition1.Path, 'LDR')
        
        % =======================================
        %% If the first acquisition is LDR so we 
        %% save images with limitation colorized
        % =======================================
                
        if isequal(Acquisition2, []) || ~contains(Acquisition2.Path, 'LDR')
            
            % =======================================
            %% If the second acquisition is empty or HDR
            %% so we save limitation information normally
            % =======================================
            
            options = {'Optimized', true, 'Colormap', 'parula', 'Unit', '%'};
            imageOptions = [{Acquisition1.Data_Size, 'PixelSize', Acquisition1.Pixel_Size}, options(:)'];
            plotOptions = [{Acquisition1.LP.X, Acquisition1.LP.Y}, options(:)'];
            
            NoisedValuesIntraPixels = process_noisedValueIntraPixels(Acquisition1.Data, limits(1));
            NoisedValuesIntraPixelsPath = fullfile(Acquisition1.Path, directory, 'IntraPixels_Percent_Noised.png');

            SaturatedValuesIntraPixels = process_saturatedValueIntraPixels(Acquisition1.Data, limits(2));
            SaturatedValuesIntraPixelsPath = fullfile(Acquisition1.Path, directory, 'IntraPixels_Percent_Saturated.png');

            BadValuesIntraPixels = NoisedValuesIntraPixels + SaturatedValuesIntraPixels;
            BadValuesIntraPixelsPath = fullfile(Acquisition1.Path, directory, 'IntraPixels_Percent_Noised_and_Saturated.png');

            NoiseValuesIntraImages = process_noisedValueIntraImages(Acquisition1.Data, limits(1));
            NoiseValuesIntraImagesPath = fullfile(Acquisition1.Path, directory, 'IntraImages_Percent_Noised.png');

            SaturatedValuesIntraImages = process_saturatedValueIntraImages(Acquisition1.Data, limits(2));
            SaturatedValuesIntraImagesPath = fullfile(Acquisition1.Path, directory, 'IntraImages_Percent_Saturated.png');

            BadValuesIntraImages = NoiseValuesIntraImages + SaturatedValuesIntraImages;
            BadValuesIntraImagesPath = fullfile(Acquisition1.Path, directory, 'IntraImages_Percent_Noised_and_Saturated.png');
            
            io_saveData('Image', NoisedValuesIntraPixels, NoisedValuesIntraPixelsPath, [], [], imageOptions{:})
            io_saveData('Image', BadValuesIntraPixels, BadValuesIntraPixelsPath, [], [], imageOptions{:})
            io_saveData('Image', SaturatedValuesIntraPixels, SaturatedValuesIntraPixelsPath, [], [], imageOptions{:})
            
            io_saveData('Plot', NoiseValuesIntraImages, NoiseValuesIntraImagesPath, [], [], plotOptions{:})
            io_saveData('Plot', BadValuesIntraImages, BadValuesIntraImagesPath, [], [], plotOptions{:})
            io_saveData('Plot', SaturatedValuesIntraImages, SaturatedValuesIntraImagesPath, [], [], plotOptions{:})
         
        elseif contains(Acquisition2.Path, 'LDR')
            
            % =======================================
            %% If the second acquisition is not empty and LDR
            %% so we save limitation information with same dynamic
            % =======================================
                        
            options = {'Colormap', 'parula', 'Unit', '%'};
            imageOptions = [{Acquisition1.Data_Size, 'PixelSize', Acquisition1.Pixel_Size}, options(:)'];
            plotOptions = [{Acquisition1.LP.X, Acquisition1.LP.Y}, options(:)'];
            
            NoisedValuesIntraPixels(1, :) = process_noisedValueIntraPixels(Acquisition1.Data, limits(1));
            NoisedValuesIntraPixelsPath = fullfile(Acquisition1.Path, directory, Acquisition2.Name, 'IntraPixels_Percent_Noised.png');

            SaturatedValuesIntraPixels(1, :) = process_saturatedValueIntraPixels(Acquisition1.Data, limits(2));
            SaturatedValuesIntraPixelsPath = fullfile(Acquisition1.Path, directory, Acquisition2.Name, 'IntraPixels_Percent_Saturated.png');

            BadValuesIntraPixels(1, :) = NoisedValuesIntraPixels + SaturatedValuesIntraPixels;
            BadValuesIntraPixelsPath = fullfile(Acquisition1.Path, directory, Acquisition2.Name, 'IntraPixels_Percent_Noised_and_Saturated.png');

            NoiseValuesIntraImages(1, :) = process_noisedValueIntraImages(Acquisition1.Data, limits(1));
            NoiseValuesIntraImagesPath = fullfile(Acquisition1.Path, directory, Acquisition2.Name, 'IntraImages_Percent_Noised.png');

            SaturatedValuesIntraImages(1, :) = process_saturatedValueIntraImages(Acquisition1.Data, limits(2));
            SaturatedValuesIntraImagesPath = fullfile(Acquisition1.Path, directory, Acquisition2.Name, 'IntraImages_Percent_Saturated.png');

            BadValuesIntraImages(1, :) = NoiseValuesIntraImages + SaturatedValuesIntraImages;
            BadValuesIntraImagesPath = fullfile(Acquisition1.Path, directory, Acquisition2.Name, 'IntraImages_Percent_Noised_and_Saturated.png');
            
            NoisedValuesIntraPixels(2, :) = process_noisedValueIntraPixels(Acquisition2.Data, limits(1));
            NoisedValuesIntraPixelsPath = {NoisedValuesIntraPixelsPath, fullfile(Acquisition2.Path, directory, Acquisition1.Name, 'IntraPixels_Percent_Noised.png')};
            
            SaturatedValuesIntraPixels(2, :) = process_saturatedValueIntraPixels(Acquisition2.Data, limits(2));
            SaturatedValuesIntraPixelsPath = {SaturatedValuesIntraPixelsPath, fullfile(Acquisition2.Path, directory, Acquisition1.Name, 'IntraPixels_Percent_Saturated.png')};
            
            BadValuesIntraPixels(2, :) = NoisedValuesIntraPixels(2, :) + SaturatedValuesIntraPixels(2, :);
            BadValuesIntraPixelsPath = {BadValuesIntraPixelsPath, fullfile(Acquisition2.Path, directory, Acquisition1.Name, 'IntraPixels_Percent_Noised_and_Saturated.png')};
            
            NoiseValuesIntraImages(2, :) = process_noisedValueIntraImages(Acquisition2.Data, limits(1));
            NoiseValuesIntraImagesPath = {NoiseValuesIntraImagesPath, fullfile(Acquisition2.Path, directory, Acquisition1.Name, 'IntraImages_Percent_Noised.png')};
            
            SaturatedValuesIntraImages(2, :) = process_saturatedValueIntraImages(Acquisition2.Data, limits(2));
            SaturatedValuesIntraImagesPath = {SaturatedValuesIntraImagesPath, fullfile(Acquisition2.Path, directory, Acquisition1.Name, 'IntraImages_Percent_Saturated.png')};
            
            BadValuesIntraImages(2, :) = NoiseValuesIntraImages(2, :) + SaturatedValuesIntraImages(2, :);
            BadValuesIntraImagesPath = {BadValuesIntraImagesPath, fullfile(Acquisition2.Path, directory, Acquisition1.Name, 'IntraImages_Percent_Noised_and_Saturated.png')};
            
            io_saveData('Image', NoisedValuesIntraPixels(1, :), NoisedValuesIntraPixelsPath{1}, NoisedValuesIntraPixels(2, :), NoisedValuesIntraPixelsPath{2}, imageOptions{:});
            io_saveData('Image', BadValuesIntraPixels(1, :), BadValuesIntraPixelsPath{1}, BadValuesIntraPixels(2, :), BadValuesIntraPixelsPath{2}, imageOptions{:});
            io_saveData('Image', SaturatedValuesIntraPixels(1, :), SaturatedValuesIntraPixelsPath{1}, SaturatedValuesIntraPixels(2, :), SaturatedValuesIntraPixelsPath{2}, imageOptions{:});
            
            io_saveData('Plot', NoiseValuesIntraImages(1, :), NoiseValuesIntraImagesPath{1}, NoiseValuesIntraImages(2, :), NoiseValuesIntraImagesPath{2}, plotOptions{:});
            io_saveData('Plot', BadValuesIntraImages(1, :), BadValuesIntraImagesPath{1}, BadValuesIntraImages(2, :), BadValuesIntraImagesPath{2}, plotOptions{:});
            io_saveData('Plot', SaturatedValuesIntraImages(1, :), SaturatedValuesIntraImagesPath{1}, SaturatedValuesIntraImages(2, :), SaturatedValuesIntraImagesPath{2}, plotOptions{:});
         
        end
    end
    
end

