function [KminVector] = map_kminVector(Maps, Data, LP, Data_Size)
    
    if ~map_isComputed(Maps, 'KminVector')

        Maps = map_kxx(Maps, Data, LP, Data_Size);
        Maps = map_kxy(Maps, Data, LP, Data_Size);
        Maps = map_kyy(Maps, Data, LP, Data_Size);

        Tensor = [reshape(Maps.Data.('Kxx'), 1, 1, []) reshape(Maps.Data.('Kxy'), 1, 1, []); ...
                  reshape(Maps.Data.('Kxy'), 1, 1, []) reshape(Maps.Data.('Kyy'), 1, 1, [])];
        KminVector = zeros(2, size(Data, 2));
        parfor i=1:size(Data, 2)
            [EigenVectors, ~] = eig(Tensor(:,:,i));
            KminVector(:, i) = EigenVectors(1, :);
        end
        Maps.Data.('KminVector') = KminVector;

    end
   
end

