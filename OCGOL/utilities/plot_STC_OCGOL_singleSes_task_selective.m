function [task_sel_STC] = plot_STC_OCGOL_singleSes_task_selective(animal_data, tunedLogical,task_selective_ROIs,options)

%% Import variables

%% Define tuned cell combinations across trials

%make conditional here for si or ts tuned neurons
switch options.tuning_criterion

    case 'selective_filtered'
        %neurons idxs associated with selective filtering for
        %task-selectivity
        select_filt_ROIs.A = task_selective_ROIs.A.idx;
        select_filt_ROIs.B = task_selective_ROIs.B.idx;
        
end

%% Extract mean STC map in each spatial bin (not normalized and not occupancy divided) (100 bins)
%for each session
%correct only
for ss =1:size(animal_data,2)
    A_df{ss} = animal_data{ss}.Place_cell{1}.Spatial_tuning_curve;
    B_df{ss} = animal_data{ss}.Place_cell{2}.Spatial_tuning_curve;
    
    %Gs smoothed, but not normalized (nn) to itself
    A_STC_nn{ss} = animal_data{ss}.Place_cell{1}.Spatial_Info.rate_map_smooth{8};
    B_STC_nn{ss} = animal_data{ss}.Place_cell{2}.Spatial_Info.rate_map_smooth{8};
    
    %A_df_both{ss} = A_df{ss}(:,AandB_tuned{ss});
    %B_df_both{ss} = B_df{ss}(:,AandB_tuned{ss});
    
    %A_df_onlyA{ss} = A_df{ss}(:,onlyA_tuned{ss});
    %B_df_onlyA{ss} = B_df{ss}(:,onlyA_tuned{ss});
end

%% Normalize each STC ROI across both trials in non-norm STCs

for ss =1:size(animal_data,2)
        %get max value for each ROIs between trials
        max_STC_across_trials{ss} = max([A_STC_nn{ss};B_STC_nn{ss}]);
        %normalize each matrix to these values (tn = trial normalized)
        A_STC_tn{ss} = A_STC_nn{ss}./max_STC_across_trials{ss};
        B_STC_tn{ss} = B_STC_nn{ss}./max_STC_across_trials{ss};
    %for max value to normalize to by 1
    %in future, do normalization range (0,1)
end

%% A vs. B on early vs late training (A or B tuned)

%ALL NEURONS in each session that meet criteria - tuned to either A or B
%early day
%for each type fo trials
for tt=1:2
    for ss =1:size(animal_data,2)
        if tt ==1
            STC_norm_self_AB{ss}{tt} = [A_df{ss}(:,select_filt_ROIs.A)', B_df{ss}(:,select_filt_ROIs.A)'];
            STC_norm_trials_AB{ss}{tt} = [A_STC_tn{ss}(:,select_filt_ROIs.A)', B_STC_tn{ss}(:,select_filt_ROIs.A)'];
        elseif tt ==2
            STC_norm_self_AB{ss}{tt} = [A_df{ss}(:,select_filt_ROIs.B)', B_df{ss}(:,select_filt_ROIs.B)'];
            STC_norm_trials_AB{ss}{tt} = [A_STC_tn{ss}(:,select_filt_ROIs.B)', B_STC_tn{ss}(:,select_filt_ROIs.B)'];
            
        end
        %STC_norm_trials_AB{ss} = [A_STC_tn{ss}(:,neither_tuned{ss})', B_STC_tn{ss}(:,neither_tuned{ss})'];
    end
end

%sort each session by A map
for tt=1:2
    for ss =1:size(animal_data,2)
        %change sort order depending of A or B selective neurons being
        %looked at
        %maxBin - spatial bin where activity is greatest for each ROI
        if tt==1
            [~,maxBin_all_AB{ss}{tt}] = max(STC_norm_trials_AB{ss}{tt}(:,1:100)', [], 1);
            %sortIdx - arrangment of ROIs after sorting by max spatial bin acitivity
            [~,sortOrder_all_AB{ss}{tt}] = sort(maxBin_all_AB{ss}{tt},'ascend');
        elseif tt==2
            [~,maxBin_all_AB{ss}{tt}] = max(STC_norm_trials_AB{ss}{tt}(:,101:200)', [], 1);
            %sortIdx - arrangment of ROIs after sorting by max spatial bin acitivity
            [~,sortOrder_all_AB{ss}{tt}] = sort(maxBin_all_AB{ss}{tt},'ascend');
        end
    end
end


%% Do the raster plot plot side by side; day by day
% figure;
% %subplot(2,1,1)
% imagesc(STC_norm_self_AB{1}{1}(sortOrder_all_AB{1}{1},:))
% %title('5A5B')
% hold on
% colormap('jet')
% caxis([0 1])
% %A/B vertical separator line
% plot([100 100],[1,size(STC_norm_self_AB{1}{1},1)], 'k','LineWidth', 1.5);


%% Split the rasters by blue and red color for A and B trials

%define colors maps for rasters
cmap_blue=cbrewer('seq', 'Blues', 32);
cmap_red=cbrewer('seq', 'Reds', 32);

%%%%% A selective neurons - STC on A laps and STC on B laps %%%%%

%set 0 value to white for both blue and red
cmap_blue(1,:) = [1 1 1];
cmap_red(1,:) = [1 1 1];

%input raster for display (A and B neighboring)
%for A selective neurons
figure('Position',[2106 101 761 857])
%A
subplot(2,2,1)
%A laps
input_matrix = STC_norm_trials_AB{1}{1}(sortOrder_all_AB{1}{1},1:100);
%create blank alpha shading matrix where 
imAlpha=ones(size(input_matrix));
imAlpha(isnan(input_matrix))=0;
imagesc(input_matrix,'AlphaData',imAlpha);
hold on
title('Asel - A laps')
%set background axis color to black
set(gca,'color',0*[1 1 1]);
%set colormap to 
caxis([0 1])
colormap(gca,cmap_blue);
cbar= colorbar;
cbar.Label.String = 'Normalized activity';
cbar.Ticks = [0 0.5 1];
ax1 = gca;
ylabel('Neuron #');
xlabel('Normalized position');
ax1.XTick = [1 100];
ax1.XTickLabel = {'0','1'};

%make ticks invisible
set(ax1, 'TickLength', [0 0]);

set(gca,'FontSize',14)
set(gca,'LineWidth',1.5)

%B laps as input
input_matrix = STC_norm_trials_AB{1}{1}(sortOrder_all_AB{1}{1},101:200);

%B 
subplot(2,2,2)
%create blank alpha shading matrix where 
imAlpha=ones(size(input_matrix));
imAlpha(isnan(input_matrix))=0;
imagesc(input_matrix,'AlphaData',imAlpha);
hold on
title('Asel - B laps')
%set background axis color to black
set(gca,'color',0*[1 1 1]);
%set colormap to 
caxis([0 1])
colormap(gca,cmap_red);
cbar2 = colorbar;
cbar2.Label.String = 'Normalized activity';
cbar2.Ticks = [0 0.5 1];
ax2 = gca;
ylabel('Neuron #');
xlabel('Normalized position');
ax2.XTick = [1 100];
ax2.XTickLabel = {'0','1'};
set(gca,'FontSize',14)
set(gca,'LineWidth',1.5)

%make ticks invisible
set(ax2, 'TickLength', [0 0]);


%%%%% B selective neurons - STC on A laps and STC on B laps %%%%%

% %set 0 value to white for both blue and red
% cmap_blue(1,:) = [1 1 1];
% cmap_red(1,:) = [1 1 1];

%input raster for display (A and B neighboring)
%for A selective neurons
%figure('Position',[2182 356 934 559])
%A
subplot(2,2,3)
%A laps
input_matrix = STC_norm_trials_AB{1}{2}(sortOrder_all_AB{1}{2},1:100);

%create blank alpha shading matrix where 
imAlpha=ones(size(input_matrix));
imAlpha(isnan(input_matrix))=0;
imagesc(input_matrix,'AlphaData',imAlpha);
hold on
title('Bsel - A laps')
%set background axis color to black
set(gca,'color',0*[1 1 1]);
%set colormap to 
caxis([0 1])
colormap(gca,cmap_blue);
cbar= colorbar;
cbar.Label.String = 'Normalized activity';
cbar.Ticks = [0 0.5 1];
ax1 = gca;
ylabel('Neuron #');
xlabel('Normalized position');
ax1.XTick = [1 100];
ax1.XTickLabel = {'0','1'};

%make ticks invisible
set(ax1, 'TickLength', [0 0]);

set(gca,'FontSize',14)
set(gca,'LineWidth',1.5)

%B laps as input
input_matrix = STC_norm_trials_AB{1}{2}(sortOrder_all_AB{1}{2},101:200);

%B 
subplot(2,2,4)
%create blank alpha shading matrix where 
imAlpha=ones(size(input_matrix));
imAlpha(isnan(input_matrix))=0;
imagesc(input_matrix,'AlphaData',imAlpha);
hold on
title('Bsel - B laps')
%set background axis color to black
set(gca,'color',0*[1 1 1]);
%set colormap to 
caxis([0 1])
colormap(gca,cmap_red);
cbar2 = colorbar;
cbar2.Label.String = 'Normalized activity';
cbar2.Ticks = [0 0.5 1];
ax2 = gca;
ylabel('Neuron #');
xlabel('Normalized position');
ax2.XTick = [1 100];
ax2.XTickLabel = {'0','1'};
set(gca,'FontSize',14)
set(gca,'LineWidth',1.5)

%make ticks invisible
set(ax2, 'TickLength', [0 0]);


%% Export STCs for cumulative plot

task_sel_STC.maps = STC_norm_trials_AB; 
task_sel_STC.sortOrder = sortOrder_all_AB;

%%
if 0
%STC normalized across trials
f= figure('Position', [2090 415 1240 420]);
subplot(1,2,1)
imagesc(STC_norm_trials_AB{1}{1}(sortOrder_all_AB{1}{1},:))
%title('Normalized according to trials')
hold on
caxis([0 1])
colormap('jet')
cbar= colorbar;
cbar.Label.String = 'Normalized activity';
cbar.Ticks = [0 0.5 1];
ax1 = gca;
ylabel('Neuron #');
xlabel('Normalized position');
ax1.XTick = [1 100 200];
ax1.XTickLabel = {'0','1','1'};
%A/B vertical separator line
plot([100 100],[1,size(STC_norm_trials_AB{1}{1},1)], 'k','LineWidth', 1.5);
hold off

subplot(1,2,2)
imagesc(STC_norm_trials_AB{1}{2}(sortOrder_all_AB{1}{2},:))
hold on
caxis([0 1])
colormap('jet')
cbar= colorbar;
cbar.Label.String = 'Normalized activity';
cbar.Ticks = [0 0.5 1];
ax1 = gca;
ylabel('Neuron #');
xlabel('Normalized position');
ax1.XTick = [1 100 200];
ax1.XTickLabel = {'0','1','1'};
%A/B vertical separator line
plot([100 100],[1,size(STC_norm_trials_AB{1}{1},1)], 'k','LineWidth', 1.5);
hold off
end

% subplot(2,1,2)
% imagesc(dF_maps_all_AB_early_late{2}(sortOrder_all_AB{2},:))
% hold on
% title('Random AB')
% colormap('jet')
% caxis([0 1])
% %A/B vertical separator line
% plot([100 100],[1,size(dF_maps_all_AB_early_late{2},1)], 'k','LineWidth', 1.5);


%hold off
%% PV correlation plot below
if 0
%PVcorr = corr(A_STC_nn{session_nb}(:,tuning_selection{session_nb}),B_STC_nn{session_nb}(:,tuning_selection{session_nb}))
PVcorr = corr(A_STC_nn{1}',B_STC_nn{1}');


figure('Position',[1350, 90, 500 860]);
subplot(2,1,1)
imagesc(PVcorr)
hold on
title('Population vector correlation');
xlabel('Spatial bin')
ylabel('Spatial bin')
colormap('jet')
caxis([0 1])
xticks([20 40 60 80 100]);
yticks([20 40 60 80 100]);
axis('square')
cbar2 = colorbar;
cbar2.Label.String = 'Correlation coefficient';
cbar2.Ticks = [0 0.5 1];

subplot(2,1,2)
hold on
title('Population vector correlation');
plot(diag(PVcorr),'k','LineWidth',1.5)
xlabel('Spatial bin')
ylabel('Correlation coef.');
plot([30 30],[0 1],'r--','LineWidth',1.5);
text([31 31],[0.9 0.9],'Reward zone B','Color','r')
plot([70 70],[0 1],'b--','LineWidth',1.5);
text([71 71],[0.3 0.3],'Reward zone A','Color','b')
plot([10 10],[0 1],'g--','LineWidth',1.5);
text([11 11],[0.9 0.9],'Odor zone\newline end','Color','g')

end

%% Make matching ROI list with tuning criteria for both sessions

%tuned to A or B on either sessions
% AorB_idx{1} = find(AorB_tuned{1} ==1);
% AorB_idx{2} = find(AorB_tuned{2} ==1);
% 
% %intersect with
% [tuned_match_idx{1},match_idx{1},~] = intersect(matching_list(:,1),AorB_idx{1},'stable');
% [tuned_match_idx{2},match_idx{2},~] = intersect(matching_list(:,2),AorB_idx{2},'stable');
% 
% %create not logical for nan exclusion from copied matrix assignement below
% include_log{1} = false(1,size(matching_list,1));
% include_log{1}(match_idx{1}) = 1; 
% %session 2 
% include_log{2} = false(1,size(matching_list,1));
% include_log{2}(match_idx{2}) = 1;
% 
% %make copy
% tuned_matching_ROI_list = matching_list;
% %nan first session that are not tuned and last session that are not
% %tuned
% tuned_matching_ROI_list(~include_log{1},1) = nan;
% tuned_matching_ROI_list(~include_log{2},2) = nan;
% 
% %which neurons to remove based on tuning criterion
% keep_ROI = sum(isnan(tuned_matching_ROI_list),2) == 0;
% 
% %retain only tuned and matched ROIs
% tuned_matching_ROI_list(~keep_ROI,:) = [];

%% Generate maps based on tuned matching ROI list
%row - session
%column - trial type
% 
% session_matched_tuned_dF_maps{1,1} = A_df{1}(:,tuned_matching_ROI_list(:,1))';
% session_matched_tuned_dF_maps{1,2} = B_df{1}(:,tuned_matching_ROI_list(:,1))';
% session_matched_tuned_dF_maps{2,1} = A_df{2}(:,tuned_matching_ROI_list(:,2))';
% session_matched_tuned_dF_maps{2,2} = B_df{2}(:,tuned_matching_ROI_list(:,2))';

% %combined 2x2
% combined_maps_2x2 = cell2mat(session_matched_tuned_dF_maps);
% 
% %single matrix (ses 1 A, B, ses 2 A,B) - in 1 row
% combined_maps_row = cell2mat(reshape(session_matched_tuned_dF_maps,1,4));
% 
% %sort by A trials on session 1
% [~,matched_maxBin_1] = max(session_matched_tuned_dF_maps{1,1}(:,1:100)', [], 1);
% %sortIdx - arrangment of ROIs after sorting by max spatial bin acitivity
% [~,matched_sortOrder_1] = sort(matched_maxBin_1,'ascend');
% 
% %sort by A trials on session 2
% [~,matched_maxBin_2] = max(session_matched_tuned_dF_maps{2,1}(:,1:100)', [], 1);
% %sortIdx - arrangment of ROIs after sorting by max spatial bin acitivity
% [~,matched_sortOrder_2] = sort(matched_maxBin_2,'ascend');

%session 1 above and session 2 below
% figure;
% subplot(2,2,1)
% imagesc(session_matched_tuned_dF_maps{1,1}(matched_sortOrder_1,:))
% colormap('jet')
% caxis([0 1]);
% subplot(2,2,2)
% imagesc(session_matched_tuned_dF_maps{1,2}(matched_sortOrder_1,:))
% colormap('jet')
% caxis([0 1]);
% subplot(2,2,3)
% imagesc(session_matched_tuned_dF_maps{2,1}(matched_sortOrder_1,:))
% colormap('jet')
% caxis([0 1]);
% subplot(2,2,4)
% imagesc(session_matched_tuned_dF_maps{2,2}(matched_sortOrder_1,:))
% colormap('jet')
% caxis([0 1]);

% %horizontal and vertical separation lines
% plot([100 100],[1,size(session_matched_tuned_dF_maps{1,1},1)*2], 'k','LineWidth', 1.5);
% plot([100 100],[1,size(dF_maps_all_AB_early_late{1},1)], 'k','LineWidth', 1.5);
% 

%% Extract STCs with tuned ROIs - in nontuned neurons will scale the weakest signal to 1 regardless of tuning

%definition of spatial tuning curve
%Gaussian smoothed onset rate map / spatial bin occupancy time (sec)
%Normalization from (0-1) for each ROI (ROI-by-ROI)

%these contain NaNs
% STC_A = animal_data{1}.Place_cell{1}.Spatial_tuning_curve;
% STC_B = animal_data{1}.Place_cell{2}.Spatial_tuning_curve;
% 
% A_STC_both = STC_A(:,AandB_tuned);
% B_STC_both = STC_B(:,AandB_tuned);
% 
% A_STC_onlyA = STC_A(:,onlyA_tuned);
% B_STC_onlyA = STC_B(:,onlyA_tuned);



end

