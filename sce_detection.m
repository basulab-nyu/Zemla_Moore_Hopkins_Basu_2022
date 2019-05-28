
%% Input data
%input data (time x ROI)
%input_data = traces;

% all A laps (calcium trace - time x ROI)
input_data = session_vars{1}.Imaging_split{4}.trace_restricted;
%normalized position
position_norm = session_vars{1}.Behavior_split{4}.resampled.position_norm;
%run epoch binary
run_epoch = session_vars{1}.Behavior_split{4}.run_ones;

%input_data = traces(st_idx:end_idx,input_neuron_idxs_sorted);

%% 3rd order Savitzky-Golay filter frame size 500ms (~15 frames)

%order of filter
sg_order = 3;
%window of filter
sg_window = 15;
%filter along first dimension (across rows)
filtered_traces = sgolayfilt(input_data,sg_order,sg_window,[],1);

%plot side by side (sample ROI
ROI = 1;
figure;
hold on
plot(input_data(:,ROI), 'k')
plot(filtered_traces(:,ROI)-1,'r');

%% Plot traces as line plot (vs imagesc)
figure;
subplot(1,2,1)
hold on;
%ylim([-0.5 1])
title('Savitzky-Golay filtered calcium traces')
stepSize = 2;
step = 0;
%first 30 ROIs
for ii=1:30size(filtered_traces,2)
    plot(filtered_traces(:,ii)-step, 'k', 'LineWidth', 1.5)
    step = step - stepSize;
end

%ylabel('Normalized Position');
%plot(norm_position(st_idx:end_idx)-step,'r');

subplot(1,2,2)
hold on;
%ylim([-0.5 1])
title('Non-filtered calcium traces')
stepSize = 2;
step = 0;
for ii=1:30%size(filtered_traces,2)
    plot(input_data(:,ii)-step, 'k', 'LineWidth', 1.5)
    step = step - stepSize;
end

%% Detect events based on threshold
%for each cell
%sum of the median value with 3x interquartile range calculated within:
%sliding window -2/+2 s 
%2s = 60 frames
win_width = 60;

%moving median
med_traces = movmedian(filtered_traces,[win_width win_width],1);

%create blank iqr vector for 1 neuron (for all neurons - ROIx time)
iqr_range = zeros(size(filtered_traces,2),size(filtered_traces,1));
%interquartile range (start at first index will be win_width+1 etc
for ii=1:size(filtered_traces,1)-2*win_width %(set range here and add edges detection in the future
    iqr_range(:,win_width+ii) = iqr(filtered_traces(ii:((2*win_width)+ii),:),1);
end

%set threshold for detection later on
event_thres = 3*iqr_range + med_traces';
%set to time x ROI format
event_thres = event_thres';

%%
%try different ROIs:
ROI = 4;

%% Plot trace, median and irq range - works
figure
hold on
stepSize = 2;
step = 0;
for ii =1:30
    ROI=ii;
    %plot all
    %trace
    plot(filtered_traces(:,ROI) - step,'k');
    %sliding median
    plot(med_traces(:,ROI) - step, 'r');
    
    %sliding interquartile range
    %plot(iqr_range, 'g');
    plot((3*iqr_range(ROI,:) + med_traces(:,ROI)') - step, 'b');
    step = step - stepSize;
end

%position
plot(position_norm-step+2,'r');

%% Detect events above 3x IQR interval

%get logical of all points where event exceeds threshold
thres_traces = filtered_traces > event_thres;

figure;
imagesc(thres_traces')

%% Select only events in noRun intervals

%select binary interval for select processing interval
run_binary_interval = run_epoch;

%remove run events
noRun_thres_traces = thres_traces & ~repmat(logical(run_binary_interval),1,size(thres_traces,2));

figure;
subplot(3,1,1)
imagesc(noRun_thres_traces')
subplot(3,1,2)
imagesc(~repmat(logical(run_binary_interval),1,size(thres_traces,2))')
subplot(3,1,3)
hold on
title('No run interval')
ylim([0 2]);
xlim([1 size(thres_traces,1)]);
plot(~run_binary_interval,'r')

%% Select individual onsets separated at least 1s - WORKS!

%frames - 1s = 30 frames
min_dur = 30;

%get onsets and offsets
diff_thres = diff(double(noRun_thres_traces),1,1);

%only get onsets
diff_thres = diff_thres == 1; 

%get index of each onset and check if event is within 
%for each ROI
for rr = 1:size(diff_thres,2)
    event_idx{rr} = find(diff_thres(:,rr) == 1);
end

%run diff, if diff < min_dur, remove event
%do this iteratively for each ROI
%set iterative flag
check_events = 1;

for  rr = 1:size(diff_thres,2)
    diff_events{rr} = diff(event_idx{rr});
    
    while check_events == 1
        
        %find first diff less than frame space duration, remove recalulate diff
        temp_dur_flag = find(diff_events{rr} < min_dur,1);
        
        %if dur_flag not empty
        if ~isempty(temp_dur_flag)
            %remove that +1 event
            event_idx{rr}(temp_dur_flag+1) = [];
            %recalulate diff on updated event list
            diff_events{rr} = diff(event_idx{rr});
            %keep flag on
            check_events = 1;
        else
            %reset flag
            check_events = 0;
        end
    end
    %reset flag for next ROI
    check_events = 1;
end

%reconstruct events
%create blank 
dur_filtered_event = zeros(size(diff_thres,1),size(diff_thres,2));
%reconstruct filtered events for each ROI
for rr=1:size(event_idx,2)
    dur_filtered_event(event_idx{rr},rr) = 1;
end

%filter out events with in 1s of one another - lots of code when  more than
%1 events that are clustered events
%sum_dur_mat = movsum(diff_thres, [0 min_dur-1],1);

%% Plot final filter
figure;
subplot(2,1,1)
hold on
xlim([1 size(thres_traces,1)])
stepSize = 2;
step = 0;
for ii=1:30%size(filtered_traces,2)
    plot(noRun_thres_traces(:,ii)-step, 'k', 'LineWidth', 1.5)
    step = step - stepSize;
end
%check onset and duration separation filter
subplot(2,1,2)
hold on
xlim([1 size(thres_traces,1)])
stepSize = 2;
step = 0;
for ii=1:30%size(filtered_traces,2)
    plot(diff_thres(:,ii)-step, 'k', 'LineWidth', 1.5)
    plot(dur_filtered_event(:,ii)-step, 'r', 'LineWidth', 1.5)
    step = step - stepSize;
end

%% Separate plot to check sync events
figure
hold on
xlim([1 size(thres_traces,1)])
stepSize = 2;
step = 0;
for ii=1:100%size(filtered_traces,2) %first 100 ROIs to check
    plot(diff_thres(:,ii)-step, 'k', 'LineWidth', 1.5)
    plot(dur_filtered_event(:,ii)-step, 'r', 'LineWidth', 1.5)
    step = step - stepSize;
end

%% Pad one index with 0 that's lost with derivative processing above

final_events = [zeros(1,size(dur_filtered_event,2));dur_filtered_event];

%% Detect SCE by running moving sum across the filtered traces and 

%minimum cells that must particupate in SCE
min_cell_nb = 5;

%width of sync events - 200ms = 6 frames
%5 will give 2 behind, center and 2 ahead
%6 will gives 3 beind, cetner
sync_window_width = 6;

%collapse all the no run SCE events 
summed_events_SCE = sum(final_events,2);

%count sync events within time window witdh
sce_event_count = movsum(summed_events_SCE,sync_window_width);

%plot
figure;
hold on
ylabel('Synchronous event count')
plot(sce_event_count)
plot([1 size(thres_traces,1)],[min_cell_nb min_cell_nb],'r--')

%% Find neurons involved in each SCE
%within 200 ms (6 frames)

%get of SCE frame indices; use 5 as a cutoff for now - use shuffle-based
%threshold later
sync_idx = find(sce_event_count >= 5);

%use all the indices for now, reconsider later (i.e. exclude some/all of
%the neighboring intervals (i.e. filter here in the future)
%proposal: for each segment of frame neighboring ROIs, select 1 index which
%has the most ROIs involved in SCE

%for each SCE, identify the neurons involved (based on window width of 6
%frames)
%defined above

%for each index, take frame range (3 idx before until 2 idx after)
%take frame slide of processed activity, sum and include ROI idx
for ss=1:size(sync_idx,1)
    SCE_ROIs{ss} = find(sum(final_events(sync_idx(ss)-3:sync_idx(ss)+2,:),1) > 0);
end

%% Relevant variables for further replay detection script

%ROIs associated with SCEs
SCE_ROIs 
%indices of SCEs
sync_idx

%% Extract sync intervals, neurons involved, and plot


%% Same code as in early simple version search

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
onset_ROIs_log = AorB_tuned;

%input neurons from RUN sequence analysis
%onset_ROIs_log = false(1,size(AorB_tuned,2));
%onset_ROIs_log(thres_neuron_idx) = 1; 

%start and end points of sorted (10-15 frame range) - 500 ms = 15
st_evt_sort = sync_idx(1);
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
input_neuron_idxs_sorted = input_neuron_idxs(I);

figure;
imagesc(input_events_matrix(I,:))
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

%% Plot traces as line plot (vs imagesc)
figure;
hold on;
stepSize = 2;
step = 0;
for ii=1:size(input_neuron_idxs_sorted,2)
    plot(input_data(st_idx:end_idx,input_neuron_idxs_sorted(ii))-step, 'k', 'LineWidth', 1.5)
    step = step - stepSize;
end

ylabel('Normalized Position');
plot(norm_position(st_idx:end_idx)-step,'r');

%% Temporal shuffle
%SCE corresponded to sync calcium events that involved more cells than
%expected by chance within a 200ms time window (i.e. > 3 std after temporal
%reshuffling of cell activity)...
%and with a minimum 5 cells participation thresgold (this is already)

%shuffle event onsets of all cells with the no run epochs (for each cell)

%recalculate the SCE count - store - do 1000x times - get distribution and
%find where threshold is at p=0.05;





