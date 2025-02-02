function [stats_out] = lap_speed_by_animal(path_dir)

%% Load in lap speed data

%read relevant data
for ee=1:size(path_dir,2)
    load_data_path{ee} = fullfile(path_dir{ee},'cumul_analysis','lap_and_event_speed.mat');
    speed_data{ee} = load(string(load_data_path{ee}),'mean_bin_speed');
end

%% Generate mean difference between bins

%get mean across each bin for A and B laps
for ii=1:size(path_dir,2)
    %get mean for A and B animal
    %A laps
    mean_speed.A(ii,:) = nanmean(speed_data{ii}.mean_bin_speed.A,1);
    %B laps
    mean_speed.B(ii,:) = nanmean(speed_data{ii}.mean_bin_speed.B,1);
end

%get mean difference between A and B laps
mean_diff = mean_speed.A - mean_speed.B;


%% Reviewer speed analysis
%for each animal
%split into bin 30-35 - B zone (complete zone - 25-40)
%split into bin 70-75 - A zone ( complete zone - 65-80)

pos = [2143.40000000000	153.800000000000	708	973.600000000000];
%pos = [1320, 256, 404, 341];

fig = figure('Position',pos);
%master layout
gridSize = [11,4];
t2 = tiledlayout(fig,gridSize(1),gridSize(2),'TileSpacing','normal','Padding','compact','Units','centimeters');
%mean +/- sem with color shading plot for each animal
%11 x 4 tabular graph
%create input function to calculate standard error of mean
sem = @(x) nanstd(x,0,1)./sqrt(size(x,1));
list_order = (1:44);
list_order = reshape(list_order,[4,11])';

% A reward zone, A trials
for ii=1:11
    nexttile(t2,list_order(ii,1),[1,1])
    s1 = shadedErrorBar(60:85,speed_data{ii}.mean_bin_speed.A(:,60:85),{@nanmean,@nanstd},'lineprops','-','transparent',true,'patchSaturation',0.20);
    set(s1.edge,'LineWidth',0.2,'LineStyle','-','Color',[[65,105,225]/255, 0.2]) %last value add transparency value
    s1.mainLine.LineWidth = 2;
    s1.mainLine.Color = [65,105,225]/255;
    s1.patch.FaceColor = [65,105,225]/255;
    ylim([0 30])
    xlim([60 85])
    yticks([0 15 30])
    xline(70,'Color',[65,105,225]/255,'LineWidth',1);
    xline(74,'Color',[65,105,225]/255,'LineWidth',1);
    if ii ==5
        ylabel('Speed [cm/s]')
    end

    if ii==11
        xticks([72])
        xticklabels("A reward zone")
        xlabel('Spatial bin')
    else
        xticks([])
    end
end
%B reward zone, A trials
for ii=1:11
    nexttile(t2,list_order(ii,2),[1,1])
    s1 = shadedErrorBar(20:45,speed_data{ii}.mean_bin_speed.A(:,20:45),{@nanmean,@nanstd},'lineprops','-','transparent',true,'patchSaturation',0.20);
    set(s1.edge,'LineWidth',0.2,'LineStyle','-','Color',[[65,105,225]/255, 0.2]) %last value add transparency value
    s1.mainLine.LineWidth = 2;
    s1.mainLine.Color = [65,105,225]/255;
    s1.patch.FaceColor = [65,105,225]/255;
    ylim([0 30])
    xlim([20 45])
    yticks([])
    xline(30,'Color',[220,20,60]/255,'LineWidth',1);
    xline(34,'Color',[220,20,60]/255,'LineWidth',1);

    if ii==11
        xticks([32])
        xticklabels("B reward zone")
        xlabel('Spatial bin')
    else
        xticks([])
    end

end

%B reward zone, B trials
for ii=1:11
    nexttile(t2,list_order(ii,3),[1,1])
    s1 = shadedErrorBar(20:45,speed_data{ii}.mean_bin_speed.B(:,20:45),{@nanmean,@nanstd},'lineprops','-','transparent',true,'patchSaturation',0.20);
    set(s1.edge,'LineWidth',0.2,'LineStyle','-','Color',[[220,20,60]/255, 0.2]) %last value add transparency value
    s1.mainLine.LineWidth = 2;
    s1.mainLine.Color = [220,20,60]/255;
    s1.patch.FaceColor = [220,20,60]/255;
    ylim([0 30])
    xlim([20 45])
    yticks([])
    xline(30,'Color',[220,20,60]/255,'LineWidth',1);
    xline(34,'Color',[220,20,60]/255,'LineWidth',1);

    if ii==11
        xticks([32])
        xticklabels("B reward zone")
        xlabel('Spatial bin')
    else
        xticks([])
    end
end

%A reward zone, A trials
for ii=1:11
    nexttile(t2,list_order(ii,4),[1,1])
    s2 = shadedErrorBar(60:85,speed_data{ii}.mean_bin_speed.B(:,60:85),{@nanmean,@nanstd},'lineprops','-','transparent',true,'patchSaturation',0.20);
    set(s2.edge,'LineWidth',0.2,'LineStyle','-','Color',[[220,20,60]/255, 0.2]) %last value add transparency value
    s2.mainLine.LineWidth = 2;
    s2.mainLine.Color = [220,20,60]/255;
    s2.patch.FaceColor = [220,20,60]/255;
    ylim([0 30])
    xlim([60 85])
    yticks([])
    xline(70,'Color',[65,105,225]/255,'LineWidth',1);
    xline(74,'Color',[65,105,225]/255,'LineWidth',1);

    if ii==11
        xticks([72])
        xticklabels("A reward zone")
        xlabel('Spatial bin')
    else
        xticks([])
    end
end
%set axis font/label and font size
set(findobj(gcf,'type','axes'),'FontName','Arial','FontSize',12, ...
    'FontWeight','normal', 'LineWidth', 1.5,'layer','top')

%% Plot of pre vs. post speed around reward zone for A vs.B with paired t-test
for ii=1:11
    %A trails A zones
    pre_post_Az.A(ii,1) = nanmean(nanmean(speed_data{ii}.mean_bin_speed.A(:,60:69),2),1)
    pre_post_Az.A(ii,2) = nanmean(nanmean(speed_data{ii}.mean_bin_speed.A(:,75:84),2),1)
    %B trials B zone
    pre_post_Bz.B(ii,1) = nanmean(nanmean(speed_data{ii}.mean_bin_speed.B(:,20:29),2),1)
    pre_post_Bz.B(ii,2) = nanmean(nanmean(speed_data{ii}.mean_bin_speed.B(:,35:44),2),1)

end

%% calculate stats here for comparing pre and post speed - paired t test
[h,p,~,stats]= ttest(pre_post_Az.A(:,1),pre_post_Az.A(:,2));
stats_out.Atrial_Azone.p = p;
stats_out.Atrial_Azone.stats = stats;
stats_out.Atrial_Azone.input_data = pre_post_Az.A;

[h,p,~,stats]= ttest(pre_post_Bz.B(:,1),pre_post_Bz.B(:,2));
stats_out.Btrial_Bzone.p = p;
stats_out.Btrial_Bzone.stats = stats;
stats_out.Btrial_Bzone.input_data = pre_post_Bz.B;

%% 
figure
hold on
ylim([0, 25])
plot([1,2],pre_post_Bz.B)

n_sample = ee;

mean_A = mean(pre_post_Az.A,1);
sem_A = std(pre_post_Az.A,0,1)./sqrt(size(n_sample));

mean_B = mean(pre_post_Bz.B,1);
sem_B = std(pre_post_Bz.B,0,1)./sqrt(size(n_sample));

pos = [2465	537	708	297.600000000000];
%pos = [1320, 256, 404, 341];

fig = figure('Position',pos);
%master layout
gridSize = [1,2];
t2 = tiledlayout(fig,gridSize(1),gridSize(2),'TileSpacing','normal','Padding','compact','Units','centimeters');

nexttile(t2,1,[1,1])
hold on
axis square
title('A trials')
plot([1,2],pre_post_Az.A,'Color', [0.7 0.7 0.7])
errorbar(mean_A,sem_A,'LineWidth',1.5,'LineStyle','-','Color',[[65,105,225]/255, 1.0])
ylim([0, 25])
xlim([0.75, 2.25])
xticks([1,2])
xticklabels({'Pre','Post'})
xlabel('A reward zone')
ylabel("Speed [cm/s]")

nexttile(t2,2,[1,1])
hold on
axis square
title('B trials')
plot([1,2],pre_post_Bz.B,'Color', [0.7 0.7 0.7])
errorbar(mean_B,sem_B,'LineWidth',1.5,'LineStyle','-','Color',[[220,20,60]/255, 1.0])
ylim([0, 25])
xlim([0.75, 2.25])
xticks([1,2])
xticklabels({'Pre','Post'})
xlabel('B reward zone')
ylabel("Speed [cm/s]")

%set axis font/label and font size
set(findobj(gcf,'type','axes'),'FontName','Arial','FontSize',12, ...
    'FontWeight','normal', 'LineWidth', 1.5,'layer','top')

%% Generate subplot for all animals with mean speed on A/B laps across all bins

figure('Position',[2308 118 1148 861])
for ii=1:size(path_dir,2)
    subplot(3,4,ii)
    hold on
    axis square
    title(num2str(ii))
    ylim([0 40])
    yticks([0 10 20 30]);
    xlabel('Normalized position');
    xticks([1 50 100])
    xticklabels({'0','0,5','1'});
    ylabel('Speed [cm/s]');

    %create input function to calculate standard error of mean
    sem = @(x) nanstd(x,0,1)./sqrt(size(x,1));

    s1 = shadedErrorBar(1:100,speed_data{ii}.mean_bin_speed.A,{@nanmean,@nanstd},'lineprops','-','transparent',true,'patchSaturation',0.20);
    set(s1.edge,'LineWidth',0.2,'LineStyle','-','Color',[[65,105,225]/255, 0.2]) %last value add transparency value
    s1.mainLine.LineWidth = 2;
    s1.mainLine.Color = [65,105,225]/255;
    s1.patch.FaceColor = [65,105,225]/255;

    s2 = shadedErrorBar(1:100,speed_data{ii}.mean_bin_speed.B,{@nanmean,@nanstd},'lineprops','-','transparent',true,'patchSaturation',0.20);
    set(s2.edge,'LineWidth',0.2,'LineStyle','-','Color',[[220,20,60]/255, 0.2]) %last value add transparency value
    s2.mainLine.LineWidth = 2;
    s2.mainLine.Color = [220,20,60]/255;
    s2.patch.FaceColor = [220,20,60]/255;

    %plot legend only for first animal
    if ii==1
        legend('A laps','B laps','location','northeast','AutoUpdate','off');
    end

    %plot reward zone A and B markers
    %reward zone A
    plot([70 70],[0 40],'--','Color',[65,105,225]/255)
    %reward zone B
    plot([30 30],[0 40],'--','Color',[220,20,60]/255)

    set(gca,'FontSize',11)
    set(gca,'LineWidth',1.5)
end

subplot(3,4,12)
hold on
axis square
%title(num2str(ii))
ylim([-15 15])
yticks([-15 -10 -5 0 5 10 15]);
xlabel('Normalized position');
xticks([1 50 100])
xticklabels({'0','0,5','1'});
ylabel('Mean Speed Difference (A-B) \newline  [cm/s]');
s1 = shadedErrorBar(1:100,mean_diff,{@nanmean,@nanstd},'lineprops','-','transparent',true,'patchSaturation',0.20);
set(s1.edge,'LineWidth',1.5,'LineStyle','-','Color',[[34,139,34]/255, 0.2]) %last value add transparency value
s1.mainLine.LineWidth = 2;
s1.mainLine.Color = [34,139,34]./255;
s1.patch.FaceColor = [34,139,34]./255;

%plot reward zone A and B markers
%reward zone A
plot([70 70],[-15 15],'--','Color',[65,105,225]/255)
%reward zone B
plot([30 30],[-15 15],'--','Color',[220,20,60]/255)

%plot 5cm/s difference lines
plot([1 100],[5 5],'-','Color','k')
plot([1 100],[-5 -5],'-','Color','k')

set(gca,'FontSize',11)
set(gca,'LineWidth',1.5)


end

