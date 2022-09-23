function [Saliency, Weigth] = map_saliency(Data, Data_Size, DistanceAlgorithm, Percentage, Radius, STD, Step, Scale)        
    
        if ~exist('Radius', 'var') || Radius == 0
            switch DistanceAlgorithm
                case 'Mahalanobis'
                    Saliency = sqrt(mahal(Data', Data'))';
                otherwise
                    error(strcat(dbstack.name, ": Erreur argument ", DistanceAlgorithm));
            end
            Weigth = [];
            
            if Percentage < 100
                buffer = sort(Saliency);
                thresholdRef = buffer(1, round(numel(Saliency) * Percentage / 100));
                
                d = Data(:, (Saliency(:) <= thresholdRef));
                Saliency = sqrt(mahal(Data', d'))';    
            end
        else
            [Data, DataSizeNew] = process_mirrorEdge(Data, Data_Size, Radius);
            
            Data = reshape(Data, DataSizeNew);
            Saliency = zeros(DataSizeNew);
            Weigth = zeros(DataSizeNew);
            
            SlideWindowSize = Radius*2+1;
            Gaussian = [process_gaussian2D(SlideWindowSize, STD)];
            
            Offset = Radius+1;
            ROI = [offset:DataSizeNew(1)-Offset, Offset:DataSizeNew(2)-Offset];
            
            SlideWindowsCoords = combvec(Offset:Step:Data_Size(2)+Offset, Offset:Step:Data_Size(1)+Offset);
            
            RelativeCoords = combvec([-Radius:Radius], [-Radius:Radius]);
            PixelsCoords = repmat(SlideWindowsCoords, 1, 1, size(RelativeCoords, 2)) - reshape(RelativeCoords, 2, 1, size(RelativeCoords, 2));
            PixelsIndex = sub2ind(DataSizeNew, PixelsCoords(1,:,:), PixelsCoords(2,:,:));
            
            for i=1:size(PixelsCoords, 3)
                if exist('Scale', 'var') && strcmp(Scale,'PixelWise') 
                    pixel = sub2ind(Data_Size, j, i);
                    Saliency(pixel) = sqrt(mahal(Data(:,pixel)', data'))';
                    Weigth(pixel) = 1;
                else
                    Saliency(PixelsIndex(:,:,i)) = Saliency(PixelsIndex(:,:,i)) +  map_saliency(Data(PixelsIndex(:,:,i)), Data_Size, DistanceAlgorithm, Percentage);
                    Weigth(PixelsIndex(:,:,i)) = Weigth(PixelsIndex(:,:,i)) + Gaussian;
                end
            end
            
            
            for i=SlideWindowsCoord(1)
                %disp(strcat(num2str(i/Data_Size(2) * 100), '%'));
                parfor j=SlideWindowsCoord(2)
                    coordinates = combvec(i-Radius:i+Radius, j-Radius:j+Radius);
                    weights = Gaussian(:)';
                    
                    weights(:, sum(coordinates < 1, 1) > 0) = [];
                    coordinates(:, sum(coordinates < 1, 1) > 0) = [];
                    
                    weights(:, sum(coordinates(1,:) > Data_Size(2), 1) > 0) = [];
                    coordinates(:, sum(coordinates(1,:) > Data_Size(2), 1) > 0) = [];
                    
                    weights(:, sum(coordinates(2,:) > Data_Size(1), 1) > 0) = [];
                    coordinates(:, sum(coordinates(2,:) > Data_Size(1), 1) > 0) = [];
                    
                    subscripts = sub2ind(Data_Size, coordinates(2,:), coordinates(1,:));
%                     data = Data(:, subscripts);
                    
                    if exist('Scale', 'var') && strcmp(Scale,'PixelWise') 
                        pixel = sub2ind(Data_Size, j, i);
                        Saliency(pixel) = sqrt(mahal(Data(:,pixel)', data'))';
                        Weigth(pixel) = 1;
                    else
%                         Saliency(subscripts) = Saliency(subscripts) +  map_saliency(data, Data_Size, DistanceAlgorithm, Percentage);
                        Weigth(subscripts) = Weigth(subscripts) + weights;
                    end
                end
            end
            
            Saliency = Saliency ./ Weigth;
        end
end

