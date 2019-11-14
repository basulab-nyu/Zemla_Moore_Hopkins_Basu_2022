function [outputArg1,outputArg2] = auc_scatterplots(path_dir)

for ee=1:size(path_dir,2)
    load_data_path_auc{ee} = fullfile(path_dir{ee},'cumul_analysis','auc.mat');
    auc_data{ee} = load(string(load_data_path_auc{ee}));
end

%add colormap to this with cbrewer
%blues (A trials)
cmap_blue = cbrewer('seq','Blues',size(path_dir,2)+2);
%reds (B trials)
cmap_red = cbrewer('seq','Reds',size(path_dir,2)+2);

%don't do range of color gradients, but single color connected by gray
%lines

%% Mean AUC/min for each animal

for ee=1:size(path_dir,2)
    mean_AUC.A(ee,:) = mean(auc_data{ee}.total_AUC_min.A,2)';
    mean_AUC.B(ee,:) = mean(auc_data{ee}.total_AUC_min.B,2)';
end


%% Plot
figure;
hold on
axis square
xlim([0 10])
ylim([0 10])
xticks(0:2:10)
yticks(0:2:10)
set(gca,'FontSize',16)
set(gca,'LineWidth',2)
xlabel('AUC/min - A trials')
ylabel('AUC/min - B trials')
title('Activity rate for task selective place cells')

%plot connecting lines for same animal points
for ee=1:size(path_dir,2)
    plot([mean_AUC.A(ee,1),mean_AUC.B(ee,1)],[mean_AUC.A(ee,2),mean_AUC.B(ee,2)] ,'Color',[1 1 1]*0.5,'LineWidth',0.5)
end

%A selective
%for ee=1:size(path_dir,2)
s1 = scatter(mean_AUC.A(:,1),mean_AUC.A(:,2),'filled','MarkerFaceColor',[65,105,225]./255);
%end

%B selective
%for ee=1:size(path_dir,2)
s2 = scatter(mean_AUC.B(:,1),mean_AUC.B(:,2),'filled','MarkerFaceColor',[220,20,60]./255);
%end

%plot center line
plot([0 10], [0 10],'Color',[1 1 1]*0,'LineStyle', '--','LineWidth',2)

legend([s1 s2],{'A','B'},'location','southeast')

end

