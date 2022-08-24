function [h_ax]= plotLP(LP)
% temp = f;
% temp1=temp;
% temp(temp<1) = -1./temp(temp<1)+2; 
% maxv1 = max([abs(max(temp)-1) abs(min(temp)-1)])
 %figure;
 h_ax = scatter3(LP(:,1),LP(:,2),LP(:,3),'filled','SizeData',100,'MarkerEdgeColor','k');

 %labels = num2str (f,'%.2f');
  %       text(Images.LP(:,1), Images.LP(:,2), Images.LP(:,3), labels,...
   %              'horizontal','left', 'vertical','bottom')
%caxis([1-maxv1, 1+maxv1]);
hc = colorbar;
colormap(parula);
axis equal
grid on
xlim([-1 1])
xlim([-1 1])
zlim([0 1])
xticks([-1 -0.5 0 0.5 1])
yticks([-1 -0.5 0 0.5 1])
zticks([0 1])
xlabel('l_u') % x-axis label
ylabel('l_v') % y-axis label
% 
end