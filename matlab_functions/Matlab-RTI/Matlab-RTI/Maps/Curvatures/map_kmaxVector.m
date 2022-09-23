function [KmaxVector] = map_kmaxVector(Maps, Data, LP, Data_Size)
    
    if ~map_isComputed(Maps, 'KmaxVector')

        Maps = map_kxx(Maps, Data, LP, Data_Size);
        Maps = map_kxy(Maps, Data, LP, Data_Size);
        Maps = map_kyy(Maps, Data, LP, Data_Size);

        Tensor = [reshape(Maps.Data.('Kxx'), 1, 1, []) reshape(Maps.Data.('Kxy'), 1, 1, []); ...
                  reshape(Maps.Data.('Kxy'), 1, 1, []) reshape(Maps.Data.('Kyy'), 1, 1, [])];
        KmaxVector = zeros(2, size(Data, 2));
        parfor i=1:size(Data, 2)
            [EigenVectors, ~] = eig(Tensor(:,:,i));
            KmaxVector(:, i) = EigenVectors(2, :);
        end
        Maps.Data.('KmaxVector') = KmaxVector;

    end

end

