function [Maps] = map_entropy(Maps, Data)

    if ~map_isComputed(Maps, 'Entropy')

        Entropy = zeros(1,size(Data,2));
        parfor i=1:size(Data,2)
            Entropy(1,i) = entropy(double(Data(:,i))); 
        end
        Maps.Data.('Entropy') = Entropy;

    end
end