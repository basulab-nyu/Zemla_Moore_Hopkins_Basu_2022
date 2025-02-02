function [scores] = si_ts_score_category(session_vars,ROI_idx_tuning_class,task_selective_ROIs,options)


%% Define the indices for each category of neurons

%% SI
%logicals for si tuned
% ROI_log.si.Aonly = tunedLogical.si.onlyA_tuned;
% ROI_log.si.Bonly = tunedLogical.si.onlyB_tuned;
% ROI_log.si.AB = tunedLogical.si.AandB_tuned;
% ROI_log.si.neither = tunedLogical.si.neither;

%get ROI idxs corresponding to the logicals
ROI_idx.si.Aonly = ROI_idx_tuning_class.si.Aonly;
ROI_idx.si.Bonly = ROI_idx_tuning_class.si.Bonly;
ROI_idx.si.AB = ROI_idx_tuning_class.si.AB;
ROI_idx.si.neither = ROI_idx_tuning_class.si.N;

%nb neurons in each category
ROI_nb.si(1) = size(ROI_idx.si.Aonly,2);
ROI_nb.si(2) = size(ROI_idx.si.Bonly,2);
ROI_nb.si(3) = size(ROI_idx.si.AB,2);
ROI_nb.si(4) = size(ROI_idx.si.neither,2);

%% TS
%logicals for ts tuned
% ROI_log.ts.Aonly = tunedLogical.ts.onlyA_tuned;
% ROI_log.ts.Bonly = tunedLogical.ts.onlyB_tuned;
% ROI_log.ts.AB = tunedLogical.ts.AandB_tuned;
% ROI_log.ts.neither = tunedLogical.ts.neither;

%get ROI idxs corresponding to the logicals
ROI_idx.ts.Aonly = ROI_idx_tuning_class.ts.Aonly;
ROI_idx.ts.Bonly = ROI_idx_tuning_class.ts.Bonly;
ROI_idx.ts.AB = ROI_idx_tuning_class.ts.AB;
ROI_idx.ts.neither = ROI_idx_tuning_class.ts.N;

%nb neurons in each category
ROI_nb.ts(1) = size(ROI_idx.ts.Aonly,2);
ROI_nb.ts(2) = size(ROI_idx.ts.Bonly,2);
ROI_nb.ts(3) = size(ROI_idx.ts.AB,2);
ROI_nb.ts(4) = size(ROI_idx.ts.neither,2);

%check that numbers add up to total id'd neurons
sum(ROI_nb.si)
sum(ROI_nb.ts)

%% Task selective
%Get task selective neuron idxs
task_sel_idx.A = task_selective_ROIs.A.idx;
task_sel_idx.B = task_selective_ROIs.B.idx;


%% Extract tuning scores for each class of neurons
% score -- type -- class of neurons -- lap type (all correct)
%% S.I. score (across 100 bins)
for tt=options.selectTrial
    %if A laps
    if tt == 1
        %all neurons
        scores.si.all.Alaps = session_vars{1}.Place_cell{tt}.Spatial_Info.Spatial_Info(8,:);
        
        %each si subcategory
        scores.si.Aonly.Alaps = session_vars{1}.Place_cell{tt}.Spatial_Info.Spatial_Info(8,ROI_idx.si.Aonly);
        scores.si.Bonly.Alaps = session_vars{1}.Place_cell{tt}.Spatial_Info.Spatial_Info(8,ROI_idx.si.Bonly);
        scores.si.AB.Alaps = session_vars{1}.Place_cell{tt}.Spatial_Info.Spatial_Info(8,ROI_idx.si.AB);
        scores.si.neither.Alaps = session_vars{1}.Place_cell{tt}.Spatial_Info.Spatial_Info(8,ROI_idx.si.neither);
        
        %selective neurons
        scores.si.task_sel_A.Alaps = session_vars{1}.Place_cell{tt}.Spatial_Info.Spatial_Info(8,task_sel_idx.A);
        scores.si.task_sel_B.Alaps = session_vars{1}.Place_cell{tt}.Spatial_Info.Spatial_Info(8,task_sel_idx.B);
    %if B laps    
    elseif tt == 2
        disp('x')
        %all neurons
        scores.si.all.Blaps = session_vars{1}.Place_cell{tt}.Spatial_Info.Spatial_Info(8,:);
        
        %each si subcategory
        scores.si.Aonly.Blaps = session_vars{1}.Place_cell{tt}.Spatial_Info.Spatial_Info(8,ROI_idx.si.Aonly);
        scores.si.Bonly.Blaps = session_vars{1}.Place_cell{tt}.Spatial_Info.Spatial_Info(8,ROI_idx.si.Bonly);
        scores.si.AB.Blaps = session_vars{1}.Place_cell{tt}.Spatial_Info.Spatial_Info(8,ROI_idx.si.AB);
        scores.si.neither.Blaps = session_vars{1}.Place_cell{tt}.Spatial_Info.Spatial_Info(8,ROI_idx.si.neither);
        
        %selective neurons
        scores.si.task_sel_A.Blaps = session_vars{1}.Place_cell{tt}.Spatial_Info.Spatial_Info(8,task_sel_idx.A);
        scores.si.task_sel_B.Blaps = session_vars{1}.Place_cell{tt}.Spatial_Info.Spatial_Info(8,task_sel_idx.B);        
    end
end

%% T.S. score (across 100 bins)
for tt=options.selectTrial
    %if A laps
    if tt == 1
        %all neurons
        scores.ts.all.Alaps = session_vars{1}.Place_cell{tt}.Tuning_Specificity.tuning_specificity;
        
        %each si subcategory
        scores.ts.Aonly.Alaps = session_vars{1}.Place_cell{tt}.Tuning_Specificity.tuning_specificity(ROI_idx.ts.Aonly);
        scores.ts.Bonly.Alaps = session_vars{1}.Place_cell{tt}.Tuning_Specificity.tuning_specificity(ROI_idx.ts.Bonly);
        scores.ts.AB.Alaps = session_vars{1}.Place_cell{tt}.Tuning_Specificity.tuning_specificity(ROI_idx.ts.AB);
        scores.ts.neither.Alaps = session_vars{1}.Place_cell{tt}.Tuning_Specificity.tuning_specificity(ROI_idx.ts.neither);
        
        %selective neurons
        scores.ts.task_sel_A.Alaps = session_vars{1}.Place_cell{tt}.Tuning_Specificity.tuning_specificity(task_sel_idx.A);
        scores.ts.task_sel_B.Alaps = session_vars{1}.Place_cell{tt}.Tuning_Specificity.tuning_specificity(task_sel_idx.B);
    %if B laps    
    elseif tt == 2
        %all neurons
        scores.ts.all.Blaps = session_vars{1}.Place_cell{tt}.Tuning_Specificity.tuning_specificity;
        
        %each si subcategory
        scores.ts.Aonly.Blaps = session_vars{1}.Place_cell{tt}.Tuning_Specificity.tuning_specificity(ROI_idx.ts.Aonly);
        scores.ts.Bonly.Blaps = session_vars{1}.Place_cell{tt}.Tuning_Specificity.tuning_specificity(ROI_idx.ts.Bonly);
        scores.ts.AB.Blaps = session_vars{1}.Place_cell{tt}.Tuning_Specificity.tuning_specificity(ROI_idx.ts.AB);
        scores.ts.neither.Blaps = session_vars{1}.Place_cell{tt}.Tuning_Specificity.tuning_specificity(ROI_idx.ts.neither);
        
        %selective neurons
        scores.ts.task_sel_A.Blaps = session_vars{1}.Place_cell{tt}.Tuning_Specificity.tuning_specificity(task_sel_idx.A);
        scores.ts.task_sel_B.Blaps = session_vars{1}.Place_cell{tt}.Tuning_Specificity.tuning_specificity(task_sel_idx.B);        
    end
end

%% Do sample scatterplot for task selective neurons
figure
subplot(1,2,1)
hold on
title('S.I. score \newline A vs. B criteria tuned neurons')
xlim([0 0.2])
xlabel('A laps')
ylabel('B laps')
xticks(0:0.05:0.2)
yticks(0:0.05:0.2)
axis square
ylim([0 0.2])
scatter(scores.si.task_sel_A.Alaps, scores.si.task_sel_A.Blaps,'MarkerFaceColor','b')
scatter(scores.si.task_sel_B.Alaps, scores.si.task_sel_B.Blaps,'MarkerFaceColor','r')
%plot center line
plot([0 0.2],[0 0.2],'k--')

subplot(1,2,2)
hold on
title('T.S. score \newline A vs. B selective neurons')
xlim([0 1])
xlabel('A laps')
ylabel('B laps')
xticks(0:0.2:1)
yticks(0:0.2:1)
axis square
ylim([0 1])
scatter(scores.ts.task_sel_A.Alaps, scores.ts.task_sel_A.Blaps,'MarkerFaceColor','b')
scatter(scores.ts.task_sel_B.Alaps, scores.ts.task_sel_B.Blaps,'MarkerFaceColor','r')
%plot center line
plot([0 1],[0 1],'k--')

%% Do sample scatterplot for neurons by tuning category
figure
subplot(1,2,1)
hold on
title('S.I. score \newline A,B,A&B, neither')
xlim([0 0.2])
xlabel('A laps')
ylabel('B laps')
xticks(0:0.05:0.2)
yticks(0:0.05:0.2)
axis square
ylim([0 0.2])
scatter(scores.si.Aonly.Alaps, scores.si.Aonly.Blaps,'MarkerFaceColor','b')
scatter(scores.si.Bonly.Alaps, scores.si.Bonly.Blaps,'MarkerFaceColor','r')
scatter(scores.si.AB.Alaps, scores.si.AB.Blaps,'MarkerFaceColor','m')
scatter(scores.si.neither.Alaps, scores.si.neither.Blaps,'MarkerFaceColor',[1 1 1]*0.5)
%plot center line
plot([0 0.2],[0 0.2],'k--')

subplot(1,2,2)
hold on
title('T.S. score \newline A,B,A&B, neither')
xlim([0 1])
xlabel('A laps')
ylabel('B laps')
xticks(0:0.2:1)
yticks(0:0.2:1)
axis square
ylim([0 1])
scatter(scores.ts.Aonly.Alaps, scores.ts.Aonly.Blaps,'MarkerFaceColor','b')
scatter(scores.ts.Bonly.Alaps, scores.ts.Bonly.Blaps,'MarkerFaceColor','r')
scatter(scores.ts.AB.Alaps, scores.ts.AB.Blaps,'MarkerFaceColor','m')
scatter(scores.ts.neither.Alaps, scores.ts.neither.Blaps,'MarkerFaceColor',[1 1 1]*0.5)
%plot center line
plot([0 1],[0 1],'k--')


end

