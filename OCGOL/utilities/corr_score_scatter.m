function [r_global] = corr_score_scatter(path_dir)
%%
%output: export global remapping neuron r values (Grant)


%% Load place cells defined according to correlation sig criteria (updated)
%correctly identified neurons will be in the final sub-struct
for ee=1:size(path_dir,2)
    load_remapping_ROI{ee} = fullfile(path_dir{ee},'cumul_analysis','remap_corr_idx.mat');
    remapping_ROI_data{ee} = load(string(load_remapping_ROI{ee}));
end

%% Load r-scores and p-values from all animals

%%%export the r values and p values associated with each rate map correlation
for ee=1:size(path_dir,2)
    load_corr_scores{ee} = fullfile(path_dir{ee},'cumul_analysis','tun_curve_corr.mat');
    r_scores_p_val{ee} = load(string(load_corr_scores{ee}));
end

%% Load in display rate maps

for ee=1:size(path_dir,2)
    load_maps{ee} = fullfile(path_dir{ee},'cumul_analysis','rate_maps_display.mat');
    rate_maps{ee} = load(string(load_maps{ee}));
end


%% Place maps into shorter variables
%circulary smoothed again (from STC_nonNorm)
for ee=1:size(path_dir,2)
    rate_map_smooth{ee}.A = rate_maps{ee}.rate_maps_display.A_smooth_norm_circ;
    rate_map_smooth{ee}.B = rate_maps{ee}.rate_maps_display.B_smooth_norm_circ;
end

%% Split the variables for remapping neuron idxs and r_scores/p_val into shorter names

%idxs of each class of remapping neurons
for ee=1:size(path_dir,2)
    common_idx{ee} = remapping_ROI_data{ee}.remapping_corr_idx.final.common;
    global_idx{ee} = remapping_ROI_data{ee}.remapping_corr_idx.final.global;
    partial_idx{ee} = remapping_ROI_data{ee}.remapping_corr_idx.final.partial;
    rate_idx{ee} = remapping_ROI_data{ee}.remapping_corr_idx.final.rate_remap_all;
end

%r values and p values for each class
for ee=1:size(path_dir,2)
    r_val{ee} = r_scores_p_val{ee}.tun_curve_corr.r;
    p_val{ee} = r_scores_p_val{ee}.tun_curve_corr.p_val;
end

%% Pair common, global, and partial with respective index from each animal

%r correlation values and p values for each of 4 classes

for ee=1:size(path_dir,2)
    %common
    r_common{ee} = r_val{ee}(common_idx{ee});
    p_common{ee} = p_val{ee}(common_idx{ee});
    
    %global
    r_global{ee} = r_val{ee}(global_idx{ee});
    p_global{ee} = p_val{ee}(global_idx{ee});
    
    %partial
    r_partial{ee} = r_val{ee}(partial_idx{ee});
    p_partial{ee} = p_val{ee}(partial_idx{ee});
    
    %rate
    r_rate{ee} = r_val{ee}(rate_idx{ee});
    p_rate{ee} = p_val{ee}(rate_idx{ee});
end

%% Merge into 1 vector r and p values

%global merge
r_global_merge = cell2mat(r_global);
p_global_merge = cell2mat(p_global);

%common merge
r_common_merge = cell2mat(r_common);
p_common_merge = cell2mat(p_common);

%partial merge
r_partial_merge = cell2mat(r_partial);
p_partial_merge = cell2mat(p_partial);

%rate merge
r_rate_merge = cell2mat(r_rate);
p_rate_merge = cell2mat(p_rate);

%% Extract smoothed STCs for each animal for common and global neurons

for ee=1:size(path_dir,2)
    %common maps
    rate_map_smooth_common{ee}.A = rate_map_smooth{ee}.A(:,common_idx{ee});
    rate_map_smooth_common{ee}.B = rate_map_smooth{ee}.B(:,common_idx{ee});
    
    %global maps
    rate_map_smooth_global{ee}.A = rate_map_smooth{ee}.A(:,global_idx{ee});
    rate_map_smooth_global{ee}.B = rate_map_smooth{ee}.B(:,global_idx{ee});    
end

%% Scroll through a few neurons and pick examples of common and global
%animal 1; neuron 8 11 17 or 52 - good common
if 0
    figure
    for ee=1:11
        for ii=1:size(rate_map_smooth_common{ee}.A,2)
            hold on
            title([num2str(ee),' ',num2str(ii)])
            plot(rate_map_smooth_common{ee}.A(:,ii),'b')
            plot(rate_map_smooth_common{ee}.B(:,ii),'r')
            pause
            clf
        end
    end
end

%good globals
%animal 1; neuron 5
%animal 4; neurons 4 7 10
%animal 6; neuron 21
if 0
    figure
    for ee=1:11
        for ii=1:size(rate_map_smooth_global{ee}.A,2)
            hold on
            title([num2str(ee),' ',num2str(ii)])
            plot(rate_map_smooth_global{ee}.A(:,ii),'b')
            plot(rate_map_smooth_global{ee}.B(:,ii),'r')
            pause
            clf
        end
        
    end
end

%% Plot scatter of correlation value vs -log(10) p value
%raise the values to the power of -1 to get the negative to the log10 when
%you use 'yscale' log

%darkorchid - common
%orchid - global
%paper colors
cmap_paper = return_paper_colormap;

%scatter colors
cmap_remapping_type = [147,112,219;...
                    255,0,255]./255;
% -log10 of p value
p_log10_thres = -log10(0.05);

marker_size =8;
%try log scale without logging the values
figure('Position',[1999 231 1454 538])
subplot(1,3,1)
hold on
axis square
title('Common')
xticks([1 100])
xticklabels({'0','1'})
xlabel('Normalized position');
yticks([0 0.5 1])
ylabel('Normalized activity')
p1 = plot(rate_map_smooth_common{1}.A(:,8),'Color',cmap_paper(1,:),'LineWidth',2);
p2 = plot(rate_map_smooth_common{1}.B(:,8),'Color',cmap_paper(2,:),'LineWidth',2);
set(gca,'FontSize',14)
set(gca,'LineWidth',1.5)
legend([p1 p2],{'A','B'},'location','northeast','AutoUpdate','off')

%plot odor dash
plot([10 10],[0 1 ],'k--')
%plot reward zone A start
plot([71 71],[0 1 ],'--','Color',cmap_paper(1,:))
%plot reward zone B start
plot([30 30],[0 1 ],'--','Color',cmap_paper(2,:))

subplot(1,3,2)
hold on
axis square
title('Global')
xticks([1 100])
xticklabels({'0','1'})
xlabel('Normalized position');
yticks([0 0.5 1])
ylabel('Normalized activity')
p1 = plot(rate_map_smooth_global{1}.A(:,5),'Color',cmap_paper(1,:),'LineWidth',2);
p2 = plot(rate_map_smooth_global{1}.B(:,5),'Color',cmap_paper(2,:),'LineWidth',2);
set(gca,'FontSize',14)
set(gca,'LineWidth',1.5)
legend([p1 p2],{'A','B'},'location','northeast','AutoUpdate','off')

%plot odor dash
plot([10 10],[0 1 ],'k--')
%plot reward zone A start
plot([71 71],[0 1 ],'--','Color',cmap_paper(1,:))
%plot reward zone B start
plot([30 30],[0 1 ],'--','Color',cmap_paper(2,:))

subplot(1,3,3)
axis square
hold on
xlim([-0.7 1.05])
ylim([-5 55])
%ylim([10^(-2) 10^60])
xlabel('Correlation score')
ylabel('-log_1_0(p)')
s1 = scatter(r_common_merge,-log10(p_common_merge),marker_size,'filled','MarkerFaceColor',cmap_remapping_type(1,:));
s2 = scatter(r_global_merge,-log10(p_global_merge),marker_size,'filled','MarkerFaceColor',cmap_remapping_type(2,:));
%plot p-threshold
plot([-1 2],[p_log10_thres, p_log10_thres],'k--','LineWidth',1)
%plot 0 correlation threshold
plot([0 0],[-5 55],'k--')
set(gca,'FontSize',14)
set(gca,'LineWidth',1.5)
%define legend

%map the two neurons with crosses on the scatter
%common neuron
scatter(r_common{1}(8),-log10(p_common{1}(8)),250, '+','MarkerFaceColor','k','MarkerEdgeColor','k')
%global neuron
scatter(r_global{1}(5),-log10(p_global{1}(5)),250, '+','MarkerFaceColor','k','MarkerEdgeColor','k')

legend([s1 s2],{'Common','Global'},'location','northwest')

%scatter(r_common_merge,p_common_merge,marker_size,'filled','MarkerFaceColor',cmap_remapping_type(1,:))
%scatter(r_common_merge,p_common_merge.^(-1),marker_size,'filled','MarkerFaceColor',cmap_remapping_type(1,:))
%scatter(r_rate_merge,p_rate_merge.^(-1),marker_size,'filled','MarkerFaceColor',cmap_remapping_type(3,:))
%scatter(r_partial_merge,-log10(p_partial_merge),14,'filled')

%set(gca,'yscale','log')




% 
% figure
% hold on
% xlim([-0.7 1.1])
% %ylim([-2 50])
% xlabel('Correlation score')
% ylabel('-log10(p)')
% scatter(r_common_merge,(p_common_merge),14,'filled')
% scatter(r_global_merge,(p_global_merge),14,'filled')
% scatter(r_rate_merge,(p_rate_merge),14,'filled')
% 



end

