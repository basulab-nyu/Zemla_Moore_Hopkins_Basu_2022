%% Sequential replay detection

%% Import/define variables

%calcium traces
traces =session_vars{1, 1}.Imaging_split{1, 4}.trace_restricted;

%ROIs associated with SCEs
SCE_ROIs 
%indices of SCEs
sync_idx


sce_nb=6;
%sync_idx(sce_nb) %--> nice replay
%SCE_ROIs{1}

%convert ROIs to logical
SCE_ROI_logi = false(1,size(final_events,2));
SCE_ROI_logi(SCE_ROIs{sce_nb}) = true; 


%% Extract calcium transient of each involved cell over 2s time windows
%2s window = 60 frames (30 before and 29 after (involved center point)
fr_range = 60;
%use frames of sce as center for calcium traces
SCE_traces = traces(sync_idx(6)-(fr_range/2):sync_idx(6)+((fr_range/2) - 1),SCE_ROIs{6});
%calculate reference as the median transient among cells involved
ref_transient = median(SCE_traces,2);


%% Plot involved traces as line plot 
figure;
subplot(2,1,1)
hold on
title('All transients involved in SCE')
xlim([1 fr_range])
stepSize = 2;
step = 0;
for ii=1:size(SCE_traces,2)
    plot(SCE_traces(:,ii), 'LineWidth', 1.5)
    %plot(dur_filtered_event(:,ii)-step, 'r', 'LineWidth', 1.5)
    %step = step - stepSize;
end

subplot(2,1,2)
hold on
title('Reference transient vs first transient')
xlim([1 fr_range])
%plot reference
plot(ref_transient,'k');
%plot first transient in series
plot(SCE_traces(:,1),'b')


%% Calculate normalized covariance (correlation) for delay between -200ms and 200 ms) (cross-correlation)
%cross covariance lag in frames
fr_delay =6;
[r,lags] = xcorr(ref_transient,SCE_traces(:,1),fr_delay);

%time diff between frames (seconds)
dt = 0.0334;
time_lag = [0:dt:1.03];
time_lag = [-1*fliplr(time_lag),time_lag(2:end)];

%fit parabola to (2nd order polynomial to covariance within lag range)
%insert x in second time domain
p = polyfit(time_lag(25:37)',r,2);

%insert a 100x expanded timebase for max extrapolation from parbola fit
expanded_time =[0:dt/100:1.03];
expanded_time = [-1*fliplr(expanded_time),expanded_time(2:end)];

%generate parabola
%x1 = time_lag(25:37);
x1 = expanded_time;
y1 = polyval(p,x1);

%find max time
[~,idx_max] = max(y1);
max_onset = x1(idx_max);

%plot
figure
hold on
%plot original fit range
plot(time_lag(25:37),r,'k')
%time extrapolated fit
plot(x1,y1,'r')
%maximum point
scatter(max_onset, y1(idx_max),'b');


%% Plot example SCE

time_choice = session_vars{1}.Behavior_split{4}.resampled.time;
time = session_vars{1, 1}.Behavior.resampled.time;

[~,select_speed_idx,~] = intersect(time,time_choice,'stable');


speed = session_vars{1, 1}.Behavior.speed;
%overwrite
speed = speed(select_speed_idx);

run_onsets = session_vars{1}.Events_split{4}.Run;
norun_onsets = session_vars{1}.Events_split{4}.NoRun;

%binary onset logicals - using Dombeck search
%run_binary = run_onsets.run_onset_binary;
%norun_binary = norun_onsets.norun_onset_binary;
norun_binary = final_events;

%combine binary onsets:
combined_binary = norun_binary;

% onset_ROIs_log (override with all tuned ROI to both or either trial)
onset_ROIs_log = SCE_ROI_logi;

%input neurons from RUN sequence analysis
%onset_ROIs_log = false(1,size(AorB_tuned,2));
%onset_ROIs_log(thres_neuron_idx) = 1; 

%start and end points of sorted (10-15 frame range) - 500 ms = 15
st_evt_sort = sync_idx(sce_nb);
% st_evt_sort = 11511;
 end_evt_sort = st_evt_sort+15;

plot_range = [500, 2000];

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
%input_neuron_idxs_sorted = input_neuron_idxs(I);
input_neuron_idxs_sorted = input_neuron_idxs;
figure;
%imagesc(input_events_matrix(I,:))
imagesc(input_events_matrix)
hold on


% Plot speed, position and dF/F trace of neurons prior to SCE in no-run epoch

%start_idx

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
imagesc(input_data(st_idx:end_idx,input_neuron_idxs_sorted)')
hold on
axis normal;
caxis([0 1]);
ylabel('Neuron #');
colormap(gca,'jet')
hold off

