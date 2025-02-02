function [remap_corr_idx,tun_curve_corr] = remapping_correlations(session_vars,tunedLogical,task_selective_ROIs,ROI_idx_tuning_class,task_remapping_ROIs,pf_count_filtered,options)


%TO-DO:
%1) get A and B tuned by either criterion correlation for comparison in
%boxplot 

%% Define tuned cell combinations across trials

%load in logicals into these variables
%S.I.
%for each session
for ss =1:size(session_vars,2)
    %spatial information criterion
    Atuned.si{ss} = ROI_idx_tuning_class.si.log.Aonly | ROI_idx_tuning_class.si.log.AB;
    Btuned.si{ss} = ROI_idx_tuning_class.si.log.Bonly | ROI_idx_tuning_class.si.log.AB;
    
    %AorB_tuned{ss} = tunedLogical(ss).si.AorB_tuned;
    onlyA_tuned.si{ss} = ROI_idx_tuning_class.si.log.Aonly;
    onlyB_tuned.si{ss} = ROI_idx_tuning_class.si.log.Bonly;
    AandB_tuned.si{ss} = ROI_idx_tuning_class.si.log.AB;
    neither_tuned.si{ss} = ROI_idx_tuning_class.si.log.N;
end

%T.S.
for ss =1:size(session_vars,2)
    %spatial information criterion
    Atuned.ts{ss} = ROI_idx_tuning_class.ts.log.Aonly | ROI_idx_tuning_class.ts.log.AB;
    Btuned.ts{ss} = ROI_idx_tuning_class.ts.log.Bonly | ROI_idx_tuning_class.ts.log.AB;

    onlyA_tuned.ts{ss} = ROI_idx_tuning_class.ts.log.Aonly;
    onlyB_tuned.ts{ss} = ROI_idx_tuning_class.ts.log.Bonly;
    AandB_tuned.ts{ss} = ROI_idx_tuning_class.ts.log.AB;
    neither_tuned.ts{ss} = ROI_idx_tuning_class.ts.log.N;
    %AorB_tuned{ss} = tunedLogical(ss).ts.AorB_tuned;
end

%% A&B tuned neurons by either criterion

Atuned.si_ts{1} = Atuned.si{1} | Atuned.ts{1};
Btuned.si_ts{1} = Btuned.si{1} | Btuned.ts{1};
AandB_tuned.si_ts{1} = Atuned.si_ts{1} & Btuned.si_ts{1};


%% Task-selective neuron idx
%add task-selective neurons (additional filters)
%neurons idxs associated with selective filtering for
%task-selectivity
select_filt_ROIs.A = task_selective_ROIs.A.idx;
select_filt_ROIs.B = task_selective_ROIs.B.idx;

%% Single field neuron criteria (TS tuned to A&B) and min 5 events per field - only these are used

%neurons tuned by TS in both trials
ts_tuned_single_PF = ROI_idx_tuning_class.ts.AB;
%single pf place fields
single_pf_idx = find(sum(pf_count_filtered==1,1) ==2);

ts_tuned_single_PF = intersect( single_pf_idx,ts_tuned_single_PF);

%% Extract mean STC map in each spatial bin (not normalized and not occupancy divided) (100 bins)
%for each session
%correct only
for ss =1:size(session_vars,2)
    A_STC{ss} = session_vars{ss}.Place_cell{1}.Spatial_tuning_curve;
    B_STC{ss} = session_vars{ss}.Place_cell{2}.Spatial_tuning_curve;
    
    %Gs smoothed, but not normalized (nn) to itself
    A_STC_nn{ss} = session_vars{ss}.Place_cell{1}.Spatial_Info.rate_map_smooth{8};
    B_STC_nn{ss} = session_vars{ss}.Place_cell{2}.Spatial_Info.rate_map_smooth{8};
    
    %A_STC_both{ss} = A_STC{ss}(:,AandB_tuned{ss});
    %B_STC_both{ss} = B_STC{ss}(:,AandB_tuned{ss});
    
    %A_STC_onlyA{ss} = A_STC{ss}(:,onlyA_tuned{ss});
    %B_STC_onlyA{ss} = B_STC{ss}(:,onlyA_tuned{ss});
end

%% Normalize each STC ROI across both trials in non-norm STCs
for ss =1:size(session_vars,2)
        %get max value for each ROIs between trials
        max_STC_across_trials{ss} = max([A_STC_nn{ss};B_STC_nn{ss}]);
        %min STC across trials
        min_STC_across_trials{ss} = min([A_STC_nn{ss};B_STC_nn{ss}]);
        
        %normalize each matrix to these values (tn = trial normalized)
        A_STC_tn{ss} = (A_STC_nn{ss} - min_STC_across_trials{ss})./(max_STC_across_trials{ss} - min_STC_across_trials{ss});
        B_STC_tn{ss} = (B_STC_nn{ss} - min_STC_across_trials{ss})./(max_STC_across_trials{ss} - min_STC_across_trials{ss});
end

%% Calculate correlation coeficient and associated p-value
%for each ROI, get correlation coef and p v-value
for rr=1:size(A_STC_nn{1},2)
    temp = A_STC_nn{1}(:,rr) + B_STC_nn{1}(:,rr);
    valid = temp>0;
    [R_mat{rr},p_mat{rr},~,~] = corrcoef(A_STC_nn{1}(valid,rr), B_STC_nn{1}(valid,rr));
    %extra singular coef and p-value
    r_score(rr) = R_mat{rr}(1,2);
    p_val(rr) = p_mat{rr}(1,2); 
end

%find common neurons
high_r_val = find(r_score >=0.5);
low_r_val =  find(r_score <0.5 & r_score >=0);
neg_r_val = find(r_score < 0); 
sig_p_val = find(p_val <0.05);
nosig_p_val = find(p_val >=0.05);

%intersect with TS criterion and min 5 events
%non-global neurons
high_pos_r_sig_p = intersect(intersect(high_r_val,sig_p_val),ts_tuned_single_PF);
low_pos_r_sig_p = intersect(intersect(low_r_val,sig_p_val),ts_tuned_single_PF);

non_global_idx = [high_pos_r_sig_p';low_pos_r_sig_p'];

%global
neg_r_sig_p = intersect(intersect(neg_r_val,sig_p_val),ts_tuned_single_PF);
no_sig_p = intersect(nosig_p_val,ts_tuned_single_PF);

global_idx = [neg_r_sig_p'; no_sig_p'];


%% Show tuning curves

figure
%for rr=all_r_nosig_p
%for rr=pos_r_sig_p
%for rr=low_pos_r_sig_p
for rr=no_sig_p
    hold on
    title([num2str(rr),'\newline',num2str(r_score(rr)),'\newline',num2str(p_val(rr))])
    plot(A_STC_nn{1}(:,rr))
    plot(B_STC_nn{1}(:,rr))
    %pause
    %clf
end


%% Calculate PV and TC correlation matrixes for tuned subcategories
%correlations are done on non normalized STCs

% %PV correlation - all neurons
% PVcorr = corr(A_STC_nn{1}',B_STC_nn{1}', 'Rows', 'complete');
% 
% %all neurons regardless of tuning status
% TCcorr.all = corr(A_STC_nn{1},B_STC_nn{1});
% 
% %TC correlation
% %SI
% TCcorr.si.Aonly = corr(A_STC_nn{1}(:,onlyA_tuned.si{1}),B_STC_nn{1}(:,onlyA_tuned.si{1}), 'Rows', 'complete');
% TCcorr.si.Bonly = corr(A_STC_nn{1}(:,onlyB_tuned.si{1}),B_STC_nn{1}(:,onlyB_tuned.si{1}), 'Rows', 'complete');
% TCcorr.si.AB = corr(A_STC_nn{1}(:,AandB_tuned.si{1}),B_STC_nn{1}(:,AandB_tuned.si{1}), 'Rows', 'complete');
% TCcorr.si.N = corr(A_STC_nn{1}(:,neither_tuned.si{1}),B_STC_nn{1}(:,neither_tuned.si{1}), 'Rows', 'complete');
% 
% %TS
% TCcorr.ts.Aonly = corr(A_STC_nn{1}(:,onlyA_tuned.ts{1}),B_STC_nn{1}(:,onlyA_tuned.ts{1}), 'Rows', 'complete');
% TCcorr.ts.Bonly = corr(A_STC_nn{1}(:,onlyB_tuned.ts{1}),B_STC_nn{1}(:,onlyB_tuned.ts{1}), 'Rows', 'complete');
% TCcorr.ts.AB = corr(A_STC_nn{1}(:,AandB_tuned.ts{1}),B_STC_nn{1}(:,AandB_tuned.ts{1}), 'Rows', 'complete');
% TCcorr.ts.N = corr(A_STC_nn{1}(:,neither_tuned.ts{1}),B_STC_nn{1}(:,neither_tuned.ts{1}), 'Rows', 'complete');
% 
% %TC correlations for A-selective and B-selective filtered 
% TCcorr.Aselective =  corr(A_STC_nn{1}(:,select_filt_ROIs.A),B_STC_nn{1}(:,select_filt_ROIs.A), 'Rows', 'complete');
% TCcorr.Bselective =  corr(A_STC_nn{1}(:,select_filt_ROIs.B),B_STC_nn{1}(:,select_filt_ROIs.B), 'Rows', 'complete');
% 
% %TC correlation for A&B tuned by either SI or TS criteria
% TCcorr.si_ts.AB = corr(A_STC_nn{1}(:,AandB_tuned.si_ts{1}),B_STC_nn{1}(:,AandB_tuned.si_ts{1}), 'Rows', 'complete');
% 
% 
% %mean(diag(TCcorr.si_ts.AB))
% 
% % nanmean(diag(TCcorr.Aonly))
% % nanmean(diag(TCcorr.Bonly))
% % nanmean(diag(TCcorr.AB))
% 
% %sort TC correlation by ROI
% diagTC = diag(TCcorr.all);
% 
% [sort_tc_val, I_sort_tc] = sort(diagTC,'ascend');
% 
% %TC correlation following reordering by Pearson
% TCcorr_sort = corr(A_STC_nn{1}(:,I_sort_tc),B_STC_nn{1}(:,I_sort_tc));
% 
% figure; 
% subplot(2,1,1)
% imagesc(PVcorr)
% hold on
% title('PV correlation - all neurons')
% hold off
% subplot(2,1,2)
% imagesc(TCcorr.all(I_sort_tc,I_sort_tc))
% hold on
% title('TC correlation - all neurons')
% 
% figure;
% %check that sort worked as expected (same result)
% plot(diag(TCcorr.all(I_sort_tc,I_sort_tc)),'b')
% plot(diag(TCcorr_sort),'r')

%% Export correlation struct

%PV correlation
remap_corr_idx.non_global_idx = non_global_idx;

%TC correlation
remap_corr_idx.global_idx = global_idx;

%export tuning correlation R coef and p value for each neuron in struct
tun_curve_corr.r = r_score;
tun_curve_corr.p_val = p_val;

end

