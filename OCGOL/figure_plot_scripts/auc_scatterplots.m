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

%% Combine into cumulative plot (all animals - 2 columns)

%run
for ee=1:size(path_dir,2)
    cum_AUC.run.Asel{ee} = auc_data{ee}.total_AUC_min.run.Asel';
    cum_AUC.run.Bsel{ee} = auc_data{ee}.total_AUC_min.run.Bsel';
    cum_AUC.run.AB{ee} = auc_data{ee}.total_AUC_min.run.si_ts.AB';
end

%no run
for ee=1:size(path_dir,2)
    cum_AUC.norun.Asel{ee} = auc_data{ee}.total_AUC_min.norun.Asel';
    cum_AUC.norun.Bsel{ee} = auc_data{ee}.total_AUC_min.norun.Bsel';
    cum_AUC.norun.AB{ee} = auc_data{ee}.total_AUC_min.norun.si_ts.AB';
end

Asel_norun = cell2mat(cum_AUC.norun.Asel');
Bsel_norun = cell2mat(cum_AUC.norun.Bsel');

%% Plot scatter of no run activity (some neurons are sp tuned in one, yet very active - participate in ctx discrimation) 
figure
subplot(1,2,1)
hold on
axis square
title('A selective')
xlabel('A laps')
ylabel('B laps')
scatter(Asel_norun(:,1),Asel_norun(:,2),5,'filled')
subplot(1,2,2)
hold on
xlim([-1 20])
ylim([-1 50])
axis square
title('B selective')
xlabel('A laps')
ylabel('B laps')
scatter(Bsel_norun(:,1),Bsel_norun(:,2),5,'filled','r')

% %A lap threshold
% thresA = mean(Bsel_norun(:,1)) +2*std(Bsel_norun(:,1))
% %B lap threshold
% thresB = mean(Bsel_norun(:,2)) -2*std(Bsel_norun(:,2))
% 
% find( & (Bsel_norun(:,1) > thresA) ==1)
% 
% find(((Bsel_norun(:,2) < mean(Bsel_norun(:,2))) & (Bsel_norun(:,1) > thresA))==1) 
% 
% find((Bsel_norun(:,2) == 0) ==1 )
% 
% ax = cell2mat(cum_AUC.run.Asel')
% bx = cell2mat(cum_AUC.run.Bsel')


% figure
% hold on
% %histogram(ax(:,1),60)
% histogram(ax(:,2),60)
% 
% thres = mean(ax(:,1))+(2*std(ax(:,1)));
% 
% %b selective
% thres = mean(bx(:,2))+(2*std(bx(:,2)));
% 
% find(ax(:,2)> thres)
% 
% figure
% hold on
% xlim([0 3])
% scatter(2*ones(1,size(ax,1)),ax(:,2))
% scatter(1*ones(1,size(ax,1)),ax(:,1))
% 
% mean(ax(:,1))

%% Mean AUC/min for each animal

%run
for ee=1:size(path_dir,2)
    mean_AUC.run.Asel(ee,:) = mean(auc_data{ee}.total_AUC_min.run.Asel,2)';
    mean_AUC.run.Bsel(ee,:) = mean(auc_data{ee}.total_AUC_min.run.Bsel,2)';
    mean_AUC.run.AB(ee,:) = mean(auc_data{ee}.total_AUC_min.run.si_ts.AB,2)';
end

%no run
for ee=1:size(path_dir,2)
    mean_AUC.norun.Asel(ee,:) = mean(auc_data{ee}.total_AUC_min.norun.Asel,2)';
    mean_AUC.norun.Bsel(ee,:) = mean(auc_data{ee}.total_AUC_min.norun.Bsel,2)';
    mean_AUC.norun.AB(ee,:) = mean(auc_data{ee}.total_AUC_min.norun.si_ts.AB,2)';
end

%distinct colors for marking each animal (rather than using lines which is
%messy
dist_colormap = distinguishable_colors(64);

%% Color scheme for figures
color_mat = [65,105,225; 220,20,60; 139, 0, 139; 128 128 128]./255;

%% Mean bar plots with sem - RUN and NO RUN

%RUN
%means
grouped_means_run = [mean(mean_AUC.run.Asel,1);
mean(mean_AUC.run.Bsel,1);
mean(mean_AUC.run.AB,1)];
%sem
grouped_sem_run = [std(mean_AUC.run.Asel,0,1)./sqrt(size(path_dir,2));
                    std(mean_AUC.run.Bsel,0,1)./sqrt(size(path_dir,2));
                    std(mean_AUC.run.AB,0,1)./sqrt(size(path_dir,2))];

%NO RUN
%mean
grouped_means_norun = [mean(mean_AUC.norun.Asel,1);
mean(mean_AUC.norun.Bsel,1);
mean(mean_AUC.norun.AB,1)];

%sem
grouped_sem_norun = [std(mean_AUC.norun.Asel,0,1)./sqrt(size(path_dir,2));
                    std(mean_AUC.norun.Bsel,0,1)./sqrt(size(path_dir,2));
                    std(mean_AUC.norun.AB,0,1)./sqrt(size(path_dir,2))];

%% Collect mean data for statistics (Wilcoxon with mc correction)

mean_run = [mean_AUC.run.Asel, mean_AUC.run.Bsel, mean_AUC.run.AB];
mean_norun = [mean_AUC.norun.Asel, mean_AUC.norun.Bsel, mean_AUC.norun.AB];

%1 vs. 2 (A sel - A vs. B)
p_run(1) = signrank(mean_run(:,1), mean_run(:,2));
%3 vs. 4 (B sel - A vs. B)
p_run(2) = signrank(mean_run(:,3), mean_run(:,4));
%5 vs. 6 (A&B - A vs. B)
p_run(3) = signrank(mean_run(:,5), mean_run(:,6));


%1 vs. 2 (A sel - A vs. B)
p_norun(1) = signrank(mean_norun(:,1), mean_norun(:,2));
%3 vs. 4 (B sel - A vs. B)
p_norun(2) = signrank(mean_norun(:,3), mean_norun(:,4));
%5 vs. 6 (A&B - A vs. B)
p_norun(3) = signrank(mean_norun(:,5), mean_norun(:,6));

%% Plot bar plots - RUN and NO RUN
figure('Position',[2390 331 426 532]);
subplot(2,1,1)
hold on;
%axis square
title('Run');
%bar the mean for each group
b = bar(1:3,grouped_means_run,'FaceColor', 'flat');
pause(0.1)
ylabel('AUC/min') 
xlim([0.5 3.5])
ylim([0 10])
%plot the sem for each mean for each group
for ib = 1:numel(b)
    %XData property is the tick labels/group centers; XOffset is the offset
    %of each distinct group
    if ib ==1
        xData(1,:) = b(ib).XData + b(ib).XOffset;
    elseif ib ==2
        xData(2,:) = b(ib).XData + b(ib).XOffset;
    elseif ib ==3
        xData(3,:) = b(ib).XData + b(ib).XOffset;
    elseif ib ==4
        xData(4,:) = b(ib).XData + b(ib).XOffset;
    end
    errorbar(xData(ib,:),grouped_means_run(:,ib)',grouped_sem_run(:,ib),'k.','LineWidth',1)
end

%set A group bars to blue
b(1).CData(1:3,:) =  repmat(color_mat(1,:),3,1);
%set B group bars to red
b(2).CData(1:3,:) =  repmat(color_mat(2,:),3,1);
%set B group bars to red
%b(3).CData(1:3,:) =  repmat(color_mat(3,:),3,1);
%set B group bars to red
%b(4).CData(1:3,:) =  repmat(color_mat(4,:),3,1);

xticks([1 2 3]);
xticklabels({'A sel.','B sel.','A&B'});

legend('A laps','B laps')

set(gca,'FontSize',16)
set(gca,'LineWidth',1.5)

subplot(2,1,2)
hold on;
%axis square
title('No run');
%bar the mean for each group
b2 = bar(1:3,grouped_means_norun,'FaceColor', 'flat');
pause(0.1)
ylabel('AUC/min') 
xlim([0.5 3.5])
ylim([0 2])
%plot the sem for each mean for each group
for ib = 1:numel(b2)
    %XData property is the tick labels/group centers; XOffset is the offset
    %of each distinct group
    if ib ==1
        xData(1,:) = b2(ib).XData + b2(ib).XOffset;
    elseif ib ==2
        xData(2,:) = b2(ib).XData + b2(ib).XOffset;
    elseif ib ==3
        xData(3,:) = b2(ib).XData + b2(ib).XOffset;
    elseif ib ==4
        xData(4,:) = b2(ib).XData + b2(ib).XOffset;
    end
    errorbar(xData(ib,:),grouped_means_norun(:,ib)',grouped_sem_norun(:,ib),'k.','LineWidth',1)
end

%set A group bars to blue
b2(1).CData(1:3,:) =  repmat(color_mat(1,:),3,1);
%set B group bars to red
b2(2).CData(1:3,:) =  repmat(color_mat(2,:),3,1);
%set B group bars to red
%b2(3).CData(1:3,:) =  repmat(color_mat(3,:),3,1);
%set B group bars to red
%b(4).CData(1:3,:) =  repmat(color_mat(4,:),3,1);

xticks([1 2 3]);
xticklabels({'A sel.','B sel.','A&B'});
ylabel('AUC/min');
legend('A laps','B laps')

set(gca,'FontSize',16)
set(gca,'LineWidth',1.5)

%%


%% Plot - RUN
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
title('A, B selective RUN')

%plot connecting lines for same animal points
% for ee=1:size(path_dir,2)
%     plot([ mean_AUC.run.Bsel(ee,1), mean_AUC.run.AB(ee,1), mean_AUC.run.Asel(ee,1), ],...
%         [mean_AUC.run.Bsel(ee,2), mean_AUC.run.AB(ee,2),mean_AUC.run.Asel(ee,2), ] ,'Color',[1 1 1]*0.5,'LineWidth',0.5)
% end

%A selective
for ee=1:size(path_dir,2)
    s1 = scatter(mean_AUC.run.Asel(ee,1),mean_AUC.run.Asel(ee,2),'filled','MarkerFaceColor',color_mat(1,:));
end

%B selective
for ee=1:size(path_dir,2)
    s2 = scatter(mean_AUC.run.Bsel(ee,1),mean_AUC.run.Bsel(ee,2),'filled','MarkerFaceColor',color_mat(2,:));
end

for ee=1:size(path_dir,2)
    s3 = scatter(mean_AUC.run.AB(ee,1),mean_AUC.run.AB(ee,2),'filled','MarkerFaceColor',color_mat(3,:));
end

%plot center line
plot([0 10], [0 10],'Color',[1 1 1]*0,'LineStyle', '--','LineWidth',2)

legend([s1 s2 s3],{'A','B','A&B'},'location','southeast')

%% Plot NO RUN
figure;
hold on
axis square
xlim([0 2])
ylim([0 2])
xticks(0:1:2)
yticks(0:1:2)
set(gca,'FontSize',16)
set(gca,'LineWidth',2)
xlabel('AUC/min - A trials')
ylabel('AUC/min - B trials')
title('A, B selective NO RUN')

%plot connecting lines for same animal points
% for ee=1:size(path_dir,2)
%     plot([ mean_AUC.run.Bsel(ee,1), mean_AUC.run.AB(ee,1), mean_AUC.run.Asel(ee,1), ],...
%         [mean_AUC.run.Bsel(ee,2), mean_AUC.run.AB(ee,2),mean_AUC.run.Asel(ee,2), ] ,'Color',[1 1 1]*0.5,'LineWidth',0.5)
% end

%A selective
for ee=1:size(path_dir,2)
    s1 = scatter(mean_AUC.norun.Asel(ee,1),mean_AUC.norun.Asel(ee,2),'filled','MarkerFaceColor',color_mat(1,:));
end

%B selective
for ee=1:size(path_dir,2)
    s2 = scatter(mean_AUC.norun.Bsel(ee,1),mean_AUC.norun.Bsel(ee,2),'filled','MarkerFaceColor',color_mat(2,:));
end

for ee=1:size(path_dir,2)
    s3 = scatter(mean_AUC.norun.AB(ee,1),mean_AUC.norun.AB(ee,2),'filled','MarkerFaceColor',color_mat(3,:));
end

%plot center line
plot([0 10], [0 10],'Color',[1 1 1]*0,'LineStyle', '--','LineWidth',2)

legend([s1 s2 s3],{'A','B','A&B'},'location','southeast')


end

