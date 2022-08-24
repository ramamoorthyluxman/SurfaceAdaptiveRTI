function P_N = PTM_terms(dirs)

P_N=[ dirs(:,1).^2  dirs(:,2).^2 dirs(:,1).*dirs(:,2)...
      dirs(:,1) dirs(:,2) ones(length(dirs(:,1)),1) ];
end