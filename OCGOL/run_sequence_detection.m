%% RUN sequence detection assembly

%% Load in animal data
%lab workstation
%input directories to matching function
%path_dir = {'G:\OCGOL_training\I56_RLTS_041019\5A5B'};
path_dir = {'G:\OCGOL_training\I56_RLTS_041019\ABrand_no_punish_041619'};
%cross session directory
%crossdir = 'G:\OCGOL_training\I56_RLTS_041019\crossSession';

%load place cell variables for each session
%get mat directories in each output folder
for ii=1:size(path_dir,2)
    %get matfile names for each session
    matfiles{ii} = dir([path_dir{ii},'\output','\*.mat']);
end
%load in place cell variables (and others later)
for ii = 1:size(path_dir,2)
    %add event variables
    session_vars{ii} = load(fullfile(matfiles{ii}.folder,matfiles{ii}.name),'Place_cell', 'Behavior',...
        'Behavior_split_lap','Behavior_split','Events_split','Events_split_lap',...
        'Imaging_split', 'Imaging');
end

%% Input data
%try this on a single lap for each trial and then aggregate

%small set for testing
%columns - ROIS
%rows - frame time
%pca_input =traces(st_idx:end_idx,input_neuron_idxs_sorted);

%A traces
pca_input = session_vars{1, 1}.Imaging_split{1, 4}.trace_restricted;

%make raw copy
pca_input_raw = pca_input;
%all restricted time and traces
%pca_input =traces;

%get speed
time_choice = session_vars{1}.Behavior_split{4}.resampled.time;
time = session_vars{1, 1}.Behavior.resampled.time;
[~,select_speed_idx,~] = intersect(time,time_choice,'stable');
speed = session_vars{1, 1}.Behavior.speed;
%overwrite
speed = speed(select_speed_idx);

%% Load SCE detection data
if 1
sce_detection = load(fullfile(path_dir{1},'sequence_analysis','SCE_detection_output.mat'));
%name variables for local workspace
SCE_ROIs = sce_detection.SCE_ROIs;
sync_idx = sce_detection.sync_idx;
sce_threshold = sce_detection.sce_threshold;
end

%% Smooth calcium traces with gaussian with sigma = 5

%5 sigma gaussian kernel
%5seconds -  150 frames
options.sigma_filter =  150;
gaussFilter = define_Gaussian_kernel(options);

for rr=1:size(pca_input,2)
    pca_input(:,rr) =conv(pca_input(:,rr),gaussFilter, 'same');
end


%% Run basic PCA

%calculate mean of each ROI - get from PCA as mu

%pca
[coef, score, latent,~,explained,mu] = pca(pca_input);

%extract 1st component (highest explained variance)
%principal component * loading
figure;
for ii=1:5
compNb = ii;
first_comp_recon = score(:,compNb)*coef(:,compNb)';
% add mean for each ROI
%first_comp_recon = bsxfun(@plus,first_comp_recon,mu);
first_comp_recon = first_comp_recon + mu;

%plot as colormap - display first 5 comps
subplot(1,5,ii)
imagesc(first_comp_recon')
hold on;
title(['PCA - component: ', num2str(compNb)]);
colorbar;
caxis([0 1])
end

%% Derivative of principal componenet for use as template

%derivate of first PCA
diff_comp = diff(score(:,2));

%% Display PCA

%% Correlate to each neuron (smoothed? or raw) - try both
%pad with 
%same result if raw or smoothed
corr_values = corr(pca_input, [0;diff_comp]);

%% Otsu's method for correlation threshold calculation (done on entire population of active cells)

%4 bins works well 10 - select only those with positive correlation?
[corr_counts,em] = histcounts(corr_values,4);
%calculate Otsu' threshold
T = otsuthresh(corr_counts);

%neurons above threshold
recurring_neuron_idx = find(corr_values > T);

%% Plot histogram with labeled Otsu threshold
figure;
histogram(corr_values, 10,'Normalization','probability')
hold on
title('Red - otsu thresold')
%plot otsu threshold
plot([T T],[0 0.3],'r--')

%% Plot dF/F of neurons above threshold

figure;
subplot(2,1,1)
imagesc(pca_input_raw(:,recurring_neuron_idx)')
hold on
title('RUN sequence identified neurons')
caxis([0 1]);
colormap('jet');
subplot(2,1,2)
hold on
xlim([1 size(speed,1)])
plot(speed,'r')

%% Insert Run sequence detection script here - return order vector 

%create sort vector
recurring_neuron_idx_sort; %import from cell activate RUN sort
remaining_neurons = setdiff(1:size(pca_input_raw,2),recurring_neuron_idx_sort');
%sort vector
sort_vector = [recurring_neuron_idx_sort', remaining_neurons];


%% Plot dF/F of neurons above threshold and show SCE for all events

figure;
subplot(4,1,1)
hold on
xlim([1 size(speed,1)])
title('Position')
ylabel('Normalized position');
plot(position_norm,'k')

subplot(4,1,2)
hold on
xlim([1 size(speed,1)])
ylim([-10 30])
title('Speed')
ylabel('[cm/s]');
plot(speed,'k')

%lap start
%plot([lap_on_off(:,1), lap_on_off(:,1)] , [-10 ,30],'r','LineStyle','-')
%lap end
%plot([lap_on_off(:,2), lap_on_off(:,2)] , [-10 ,30],'r','LineStyle','-')

subplot(4,1,3)
%imagesc(pca_input_raw(:,sort_vector)')
imagesc(pca_input_raw(:,recurring_neuron_idx_sort)')
hold on
ylabel('Neuron #');
xlim([1 size(speed,1)])
title('RUN sequence identified neurons')
caxis([0 0.7]);
colormap('hot');
%start
plot([lap_on_off(:,1), lap_on_off(:,1)] , [1 ,size(idx_med_onset,2)],'r','LineStyle','-')
%end
plot([lap_on_off(:,2), lap_on_off(:,2)] , [1 ,size(idx_med_onset,2)],'r','LineStyle','-')

%select SCE indices
sce_nbs =[4 18 54 73 83 93 103 118 131 154 174 191 203];

for ss=1:size(sce_nbs,2)
    sce_nb = sce_nbs(ss);
    
    %plot selected sync events - plot start of 200ms interval
    plot([sync_idx(sce_nb )-3 sync_idx(sce_nb)-3],[1 size(idx_med_onset,2)],'w--');
    %plot([sync_idx(sce_nb )+3 sync_idx(sce_nb)+3],[1 size(idx_med_onset,2)],'w--');
end

%colorbar
subplot(4,1,4)
hold on
title('Calcium onsets')
xlim([1 size(speed,1)])
ylim([0 40])
ylabel('Synchronous event count')
area(sce_event_count,'FaceColor', 'k', 'EdgeColor','k')
%minimum cell involvement line
p1 = plot([1 size(thres_traces,1)],[min_cell_nb min_cell_nb],'r--');
%threshold determined by shuffle
p2 = plot([1 size(thres_traces,1)],[sce_threshold sce_threshold],'b--');
legend([p1 p2],'5 cells','Temporal shuffle threshold')





%% See how many neurons in RUN sequence are present in given SCE

%for each sync event, find max
for ss=1:size(SCE_ROIs,2)
[neurons_participating{ss},~,~] = intersect(SCE_ROIs{ss},recurring_neuron_idx,'stable');
end

%take the length of each sequence
seq_overlap = cell2mat(cellfun(@length,neurons_participating,'UniformOutput',false));

%take the ones where there are at least 5 in RUN sequence
SCE_run_seqs = find(seq_overlap > 5);


%% Save relevant variables for subsequent analyssi here


%% FOR LATER INTEGRATION - don't see much (any) difference with oPCA
%{

%% For offset PCA - similar to PCA test - 
%construct covariance matrix from original dF/F and 1 timeframes shifted
%matrix (circshift) - 2 separate matrices


%use cov to construct covariance matrix
%use pcacov to run pca on cov matrix

%try this approach first to get same output as with regular PCA
%need to stack the two matrixs together and select correct set do PCA; if
%loaded separate will run as two long vectors

%original matrix
orig_mat = pca_input;
%mean of each ROI (as vector)
meanROI = mean(pca_input,1);
%replicate vectors for matrix subtraction from whole dataset
data_mean = repmat(mean(pca_input,1),size(pca_input,1),1);
%mean-subtraced matrix
meanSub_mat = orig_mat - data_mean;

%remove last row
%orig_mat = orig_mat(1:end-1,:);

%shifted matrices for oPCA
shift_mat_fwd = circshift(meanSub_mat,6);
shift_mat_rev = circshift(meanSub_mat,-6);

%remove first row
shift_mat_fwd = shift_mat_fwd(2:end,:);
shift_mat_rev = shift_mat_rev(1:end-1,:);

%cov_matrix input
cov_1_mat = cov(shift_mat_fwd);
cov_2_mat = cov(shift_mat_rev);

%combined estimator
cov_input = 0.5*(cov_1_mat + cov_2_mat);

%pca on covariance matrix (same explained variance as reg pca)
[coef_pcacov, latened_pcacov, explained_pca_cov] = pcacov(cov_input);

%scores - representation of X (data) in principal component space
%need to generate this to reconstruct the data

scores_cov = orig_mat*coef_pcacov;
%reconstruct using chosen component
compNb = 2;
first_comp_recon_pca_cov = scores_cov(:,compNb)*coef_pcacov(:,compNb)'  + data_mean;

%plot as colormap
figure;
imagesc(first_comp_recon_pca_cov')
hold on;
title(['PCA - component: ', num2str(compNb)]);
colorbar;


%center the inputs as well
%cov_pca =  cov([orig_mat-data_mean,shift_mat-data_mean]);
%cov_pca1 =  cov([pca_input,pca_input]);
%extract correct sub-matrix that corresponds to the cross-covariance
%cov_cross = cov_pca1(1:16,17:32);
%cov_cross = cov_pca(569:end,1:568);

%cov_pca2 = cov([pca_input,circshift(pca_input,3)]);

%% Derivative of principal componenet for use as template

%derivate of first PCA
diff_comp_cov = diff(scores_cov(:,2));

%% Display PCA

%% Correlate to each neuron (smoothed? or raw) - try both
%pad with 
%same result if raw or smoothed
corr_values = corr(pca_input, [0;diff_comp_cov]);

%plot histogram
figure;
hold on
histogram(corr_values, 10)


%% Otsu's method for correlation threshold calculation (done on entire population of active cells)

[corr_counts,em] = histcounts(corr_values,10);
%calculate Otsu' threshold
T = otsuthresh(corr_counts);

%neurons above threshold
recurring_neuron_idx = find(corr_values > T);

for ss=1:size(SCE_ROIs,2)
[neurons_participating{ss},~,~] = intersect(SCE_ROIs{ss},recurring_neuron_idx,'stable');
end

%take the length of each sequence
seq_overlap = cell2mat(cellfun(@length,neurons_participating,'UniformOutput',false));

%take the ones where there are at least 5 in RUN sequence
SCE_run_seqs = find(seq_overlap >5);


%% Plot dF/F of neurons above threshold

figure;
subplot(2,1,1)
imagesc(pca_input_raw(:,recurring_neuron_idx)')
hold on
title('RUN sequence identified neurons')
caxis([0 1]);
colormap('jet');
subplot(2,1,2)
hold on
xlim([1 size(speed,1)])
plot(speed,'r')


%% Correlate to each neuron (smoothed? or raw) - try both
%pad with 
%same result if raw or smoothed
corr_values_cov = corr(orig_mat, [0;diff_comp_cov]);

%plot histogram
figure;
hold on
histogram(corr_values_cov)


%% Otsu's method for correlation threshold calculation (done on entire population of active cells)

[corr_counts_cov,~] = histcounts(corr_values_cov,100);
%calculate Otsu' threshold
T = otsuthresh(corr_counts_cov);

find(corr_values_cov >  T)

%% OLD BELOW %%%%%%%%%%%%%%%%%%%

%% PCA using covariance matrix (no double matrix for offset test) - as good as above
data_mean = repmat(mean(pca_input,1),size(pca_input,1),1);
%center the inputs as well
%cov_pca1 =  cov(pca_input-data_mean);
cov_pca1 =  cov(pca_input);

%pca on covariance matrix (same explained variance as reg pca)
[coef_pcacov, latened_pcacov, explained_pca_cov] = pcacov(cov_pca1);

%scores - representation of X (data) in principal component space
%need to generate this to reconstruct the data

scores_cov = pca_input*coef_pcacov;
%reconstruct using chosen component
compNb = 1;
first_comp_recon_pca_cov = scores_cov(:,compNb)*coef_pcacov(:,compNb)'  + data_mean;

%plot as colormap
figure;
imagesc(first_comp_recon_pca_cov')
hold on;
title(['PCA - component: ', num2str(compNb)]);
colorbar;

diff_mat = first_comp_recon_pca_cov' - first_comp_recon';

%}
