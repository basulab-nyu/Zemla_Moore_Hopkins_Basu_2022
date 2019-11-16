function [outputArg1,outputArg2] = si_ts_score_distributions(path_dir)


%% Load the scores for each animal
for aa=1:size(path_dir,2)
    load_data_path{aa} = fullfile(path_dir{aa},'cumul_analysis','tuning_scores.mat');
    scores{aa} = load(string(load_data_path{aa}));
end

%% Create cumulative distribution and histograms for each class of neurons
%A,B,A&B,Neither

%first column - Alaps; second column - Blaps
for aa=1:size(path_dir,2)
    
    %SI
    %A only neurons
    si.Aonly{aa}(:,1) = scores{aa}.tuning_scores.si.Aonly.Alaps;
    si.Aonly{aa}(:,2) = scores{aa}.tuning_scores.si.Aonly.Blaps;
    
    %B only neurons
    si.Bonly{aa}(:,1) = scores{aa}.tuning_scores.si.Bonly.Alaps;
    si.Bonly{aa}(:,2) = scores{aa}.tuning_scores.si.Bonly.Blaps;
    
    %A&B neurons
    si.AB{aa}(:,1) = scores{aa}.tuning_scores.si.AB.Alaps;
    si.AB{aa}(:,2) = scores{aa}.tuning_scores.si.AB.Blaps;
    %neither neurons
    si.N{aa}(:,1) = scores{aa}.tuning_scores.si.neither.Alaps;
    si.N{aa}(:,2) = scores{aa}.tuning_scores.si.neither.Blaps;
    
    %TS
    %A only neurons
    ts.Aonly{aa}(:,1) = scores{aa}.tuning_scores.ts.Aonly.Alaps;
    ts.Aonly{aa}(:,2) = scores{aa}.tuning_scores.ts.Aonly.Blaps;
    
    %B only neurons
    ts.Bonly{aa}(:,1) = scores{aa}.tuning_scores.ts.Bonly.Alaps;
    ts.Bonly{aa}(:,2) = scores{aa}.tuning_scores.ts.Bonly.Blaps;
    
    %A&B neurons
    ts.AB{aa}(:,1) = scores{aa}.tuning_scores.ts.AB.Alaps;
    ts.AB{aa}(:,2) = scores{aa}.tuning_scores.ts.AB.Blaps;
    %neither neurons
    ts.N{aa}(:,1) = scores{aa}.tuning_scores.ts.neither.Alaps;
    ts.N{aa}(:,2) = scores{aa}.tuning_scores.ts.neither.Blaps;

end

%collapse into single matrix
%SI
si.Aonly_cumul = cell2mat(si.Aonly');
si.Bonly_cumul = cell2mat(si.Bonly');
si.AB_cumul = cell2mat(si.AB');
si.N_cumul = cell2mat(si.N');

%TS
ts.Aonly_cumul = cell2mat(ts.Aonly');
ts.Bonly_cumul = cell2mat(ts.Bonly');
ts.AB_cumul = cell2mat(ts.AB');
ts.N_cumul = cell2mat(ts.N');

%% Get histogram distributions around centerline - S.I.

%for si
options.xlims = [-0.2 0.2];
unity_hist_scatter_spatial_scores(si,options)

%for ts
options.xlims = [-1 1];
unity_hist_scatter_spatial_scores(ts,options)

%% Plot scatter plots of all neurons for each category
marker_size = 5;

color_codes = [65,105,225; 220,20,60; 139, 0, 139; 128 128 128]./255;

%try scatterplot
figure('Position',[2208 244 1198 512])
%si
subplot(1,2,1)
hold on
title('S.I. score')
axis square
xlim([-0.01 0.25])
ylim([-0.01 0.25])
xticks([0 0.1 0.2])
yticks([0 0.1 0.2])
xlabel('A laps')
ylabel('B laps')
scatter(si.N_cumul(:,1),si.N_cumul(:,2),marker_size,'MarkerFaceColor',color_codes(4,:),'MarkerEdgeColor',color_codes(4,:))
scatter(si.Aonly_cumul(:,1),si.Aonly_cumul(:,2),marker_size,'MarkerFaceColor',color_codes(1,:),'MarkerEdgeColor',color_codes(1,:))
scatter(si.Bonly_cumul(:,1),si.Bonly_cumul(:,2),marker_size,'MarkerFaceColor',color_codes(2,:),'MarkerEdgeColor',color_codes(2,:))
scatter(si.AB_cumul(:,1),si.AB_cumul(:,2),marker_size,'MarkerFaceColor',color_codes(3,:),'MarkerEdgeColor',color_codes(3,:))
set(gca,'FontSize',14)
set(gca,'LineWidth',1.5)

%center line
plot([-0.01 0.25],[-0.01 0.25],'k--')

%ts
subplot(1,2,2)
hold on
title('T.S. score')
axis square
xlim([0 1.1])
ylim([0 1.1])
xticks([0 0.5 1])
yticks([0 0.5 1])
xlabel('A laps')
ylabel('B laps')
scatter(ts.N_cumul(:,1),ts.N_cumul(:,2),marker_size,'MarkerFaceColor',color_codes(4,:),'MarkerEdgeColor',color_codes(4,:))
scatter(ts.Aonly_cumul(:,1),ts.Aonly_cumul(:,2),marker_size,'MarkerFaceColor',color_codes(1,:),'MarkerEdgeColor',color_codes(1,:))
scatter(ts.Bonly_cumul(:,1),ts.Bonly_cumul(:,2),marker_size,'MarkerFaceColor',color_codes(2,:),'MarkerEdgeColor',color_codes(2,:))
scatter(ts.AB_cumul(:,1),ts.AB_cumul(:,2),marker_size,'MarkerFaceColor',color_codes(3,:),'MarkerEdgeColor',color_codes(3,:))
set(gca,'FontSize',14)
set(gca,'LineWidth',1.5)

%center line
plot([0 1.1],[0 1.1],'k--')

end

