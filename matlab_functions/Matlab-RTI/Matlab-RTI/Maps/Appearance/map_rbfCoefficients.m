function [Maps] = map_rbfCoefficients(Maps, Data, LP, Function)
    
    Li = cat(2, LP.X, LP.Y);
    %     R = mean(std(Li, 0, 1));

    rbf = rbfcreate(Li', Data', 'RBFFunction', Function);
    Maps.('RBF').Coefficients = transpose(rbf.('rbfcoeff'));
    
end

