function [SCE] = detect_SCE(session_vars,options)

%% Input options
sessionSelect = options.sessionSelect;
selectTrial = options.selectTrial;

%% Input data
%input data (time x ROI)
%input_data = traces;

%preallocate SCE output cell
SCE = cell(1,size(options.sessionSelect,2));
%session # 
for ss=sessionSelect
    %ss =1;
    %take in all data from all laps (regardless of trial
    trialType = 3;
    
    % the number of frames (number of rows should be the same for vars below)
    % all laps (calcium trace - time x ROI)
    %REPLACE WITH UPDATED_DFF IN THE FUTURE
    imaging_traces = session_vars{ss}.Imaging_split{trialType}.trace_restricted;
    %normalized position
    position_norm = session_vars{ss}.Behavior_split{trialType}.resampled.position_norm;
    %run epoch binary
    run_epoch = session_vars{ss}.Behavior_split{trialType}.run_ones;
    
    %select only run sequence neurons in input data
    %input_data = input_data(:,recurring_neuron_idx);
    %input_data = traces(st_idx:end_idx,input_neuron_idxs_sorted);
    %copy of above
    %input_data_all = imaging_traces;

    % Run SCE detection for each animal with shuffle
    disp(['Running session:',num2str(ss)]);
    [SCE{ss}] = run_detect_SCE_per_session(imaging_traces, position_norm,run_epoch,options)
end


%% REUSE CODE BELOW

%{

%% Save relevant variables in folder for future analysis
%SCE_ROIs - ROIs associated with each frame SCE
%sync_idx - indices of SCEs
%sce_threshold - shuffle determined threshold for # of events needed to
%sce_event_count - number of events within 200 ms window on each frame 
%consider SCE statistically signifant
%final_events - event onsets with filtered events according to criteria
%above

%make shared folder for sequence analysis
cd(path_dir{1})
mkdir('sequence_analysis')

save(fullfile(path_dir{1},'sequence_analysis','SCE_detection_output.mat'),'SCE_ROIs','sync_idx',...
    'sce_threshold','sce_event_count','final_events');


%% For each SCE, get fraction of neurons that are tuned by respective criteria

%1 - A tuned
%2 - B tuned
%3 - A&B tuned
%4 - neither

%onlyA_idx = find(tunedLogical.si.onlyA_tuned ==  1);
%after filter
onlyA_idx = task_selective_ROIs.A.idx;
%onlyB_idx = find(tunedLogical.si.onlyB_tuned ==  1);
%after filter
onlyB_idx = task_selective_ROIs.B.idx;
AandB_idx = find(tunedLogical.ts.AandB_tuned == 1);

%centroid diff associated with with A&B tuned neurons
%find(tunedLogical.ts.AandB_tuned ==1)

for ss=1:size(SCE_ROIs,2)
    log_A{ss} = ismember(SCE_ROIs{ss},onlyA_idx);
    %percentrage A of all ROIs in SCE
    percentages(ss,1) = size(find(log_A{ss}==1),2)/size(log_A{ss},2);
    log_B{ss} = ismember(SCE_ROIs{ss},onlyB_idx);
    %percentrage B of all ROIs in SCE
    percentages(ss,2) = size(find(log_B{ss}==1),2)/size(log_B{ss},2);
    [log_AB{ss},cent_idx{ss}] = ismember(SCE_ROIs{ss},AandB_idx);
    %get mean centroid diff for involved neurons
    mean_cent_diff(ss) = rad2deg(nanmean(cent_diff_AandB.angle_diff(cent_idx{ss}(log_AB{ss}))));
    %percentrage A&B of all ROIs in SCE
    percentages(ss,3) = size(find(log_AB{ss}==1),2)/size(log_AB{ss},2);
end

%add centroid diff to percentrages
percentages = [percentages, mean_cent_diff'];

%% Extract sync intervals, neurons involved, and plot

%plot entire dF/F for each sync event ROIs
figure
imagesc(traces(:,SCE_ROIs{77}  )')
hold on
axis normal;
caxis([0 1]);
ylabel('Neuron #');
colormap(gca,'jet')
hold off

input_data_A = session_vars{1}.Imaging_split{4}.trace_restricted;
input_data_B = session_vars{1}.Imaging_split{5}.trace_restricted;

%dF/F
figure
subplot(1,2,1)
imagesc(input_data_A(:,SCE_ROIs{78})')
hold on
title('A')
axis normal;
caxis([0 1]);
ylabel('Neuron #');
colormap(gca,'jet')
hold off

subplot(1,2,2)
imagesc(input_data_B(:,SCE_ROIs{78})')
hold on
title('B')
axis normal;
caxis([0 1]);
ylabel('Neuron #');
colormap(gca,'jet')
hold off

%% Same code as in early simple version search

%A correct
time_choice = session_vars{1}.Behavior_split{1}.resampled.time;
time = session_vars{1, 1}.Behavior.resampled.time;

[~,select_speed_idx,~] = intersect(time,time_choice,'stable');

speed = session_vars{1, 1}.Behavior.speed;

%overwrite
speed = speed(select_speed_idx);

run_onsets = session_vars{1}.Events_split{1}.Run;
norun_onsets = session_vars{1}.Events_split{1}.NoRun;

%binary onset logicals - using Dombeck search
%run_binary = run_onsets.run_onset_binary;
%norun_binary = norun_onsets.norun_onset_binary;
norun_binary = final_events;

%combine binary onsets:
combined_binary = norun_binary;

% onset_ROIs_log (override with all tuned ROI to both or either trial)
%onset_ROIs_log = tunedLogical.si.AorB_tuned;
%all neurons
onset_ROIs_log = logical(ones(1,size(input_data_all,2)));

%input neurons from RUN sequence analysis
%onset_ROIs_log = false(1,size(AorB_tuned,2));
%onset_ROIs_log(thres_neuron_idx) = 1; 

%start and end points of sorted (10-15 frame range) - 500 ms = 15
sync_idx_for_range_sort = 78;

st_evt_sort = sync_idx(sync_idx_for_range_sort)-3;
% st_evt_sort = 11511;
 end_evt_sort = sync_idx(sync_idx_for_range_sort)+2;

plot_range = [1500, 1500];

%start idx (absolute)
st_idx =st_evt_sort-plot_range(1);
%end idx (absolute)
end_idx = st_evt_sort+plot_range(2);

%input_dFF_matrix = traces(2360+700:2900+700,onset_ROIs_log)';

%get absolute idx's of neurons
input_neuron_idxs = find(onset_ROIs_log ==1);

%1500-2000 +700
input_events_matrix = combined_binary(st_evt_sort:end_evt_sort,onset_ROIs_log)';

%sort by event onset
%for each ROI
for rr=1:size(input_events_matrix,1)
    if  ~isempty(find(input_events_matrix(rr,:) > 0,1))
        loc_ROI(rr) = find(input_events_matrix(rr,:) > 0,1);
    else
        loc_ROI(rr) = 0;
    end
end

%sort by index
[M_sort,I] =sort(loc_ROI,'ascend');

%remove neurons that do not have an event in region of interest
I(M_sort == 0) = [];

%neuron idx (global) sorted
input_neuron_idxs_sorted = input_neuron_idxs(I);

figure;
imagesc(input_events_matrix(I,:))
hold on

% Plot speed, position and dF/F trace of neurons prior to SCE in no-run epoch

figure
subplot(4,1,1)
hold on;
xlim([1,end_idx-st_idx])
%speed
plot(speed(st_idx:end_idx),'k');
%plot(time(st_idx:end_idx),speed(st_idx:end_idx),'k');
ylabel('Speed [cm/s]');
xlabel('Time [s]');

subplot(4,1,2)
%position (norm)
hold on
xlim([1,end_idx-st_idx])
%xlim([st_idx,end_idx])
ylabel('Normalized Position');
plot(position_norm(st_idx:end_idx),'k');
%plot(time(st_idx:end_idx),norm_position(st_idx:end_idx),'k');
hold off

%dF/F
subplot(4,1,[3 4])
imagesc(imaging_traces(st_idx:end_idx,input_neuron_idxs_sorted)')
hold on
axis normal;
caxis([0 1]);
ylabel('Neuron #');
colormap(gca,'jet')
hold off

%% Plot traces as line plot (vs imagesc)
figure;
hold on;
stepSize = 2;
step = 0;
for ii=1:size(input_neuron_idxs_sorted,2)
    plot(imaging_traces(st_idx:end_idx,input_neuron_idxs_sorted(ii))-step, 'k', 'LineWidth', 1.5)
    step = step - stepSize;
end

ylabel('Normalized Position');
plot(norm_position(st_idx:end_idx)-step,'r');
%}

end

