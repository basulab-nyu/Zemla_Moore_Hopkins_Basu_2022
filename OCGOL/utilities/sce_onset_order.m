function [SCE] = sce_onset_order(session_vars,SCE,options)
% SCE order detection using parabolic fit (Malvache et al. 2016)

%% Define parameters
sessionSelect = options.sessionSelect;


%% Import/define variables

%returns sorted indexes of corresponding ROIs in each SCE for each session
for ss = sessionSelect
    disp(['Session: ', num2str(ss)]);
    %calcium traces - input into get_SCE_order (all trials)
    traces = session_vars{ss}.Imaging.trace_restricted;
    
    %run order sorting for each session
    [SCE] = get_SCE_order(traces,SCE,ss);
end



%% Plot example SCE - move to separate script

if 0
%calcium traces - input into get_SCE_order (all trials)
traces = session_vars{ss}.Imaging.trace_restricted;


figure;
for cc=34
    sce_nb = cc;
    
    %temporary generate times and speed for select input inverval
    time = session_vars{ss}.Imaging.time_restricted;
    speed = session_vars{ss}.Behavior.speed;
    pos_norm = session_vars{ss}.Behavior.resampled.normalizedposition;
    
    %# of frames before and after onset of SCE
    plot_range = [7500, 27500];
    
    %absolute time frame of SCE start (all trials)
    start_SCE_frame = SCE{ss}.sync_idx(SCE{ss}.sync_range(sce_nb,1));
    
    %start and end points of sorted (10-15 frame range) - 500 ms = 15
    st_evt_sort = start_SCE_frame;
    % st_evt_sort = 11511;
    end_evt_sort = st_evt_sort+15;
    
    %start idx (absolute)
    st_idx =st_evt_sort-plot_range(1);
    %end idx (absolute)
    end_idx = st_evt_sort+plot_range(2);
    subplot(3,1,1)
    imagesc(traces(st_idx:end_idx,SCE{ss}.SCE_unique_ROIs_sorted{sce_nb})')
    hold on;
    xlim([0 sum(plot_range)])
    title('All SCE ROIS traces sorted by onset time')
    colormap('hot')
    caxis([0 1])
    
    subplot(3,1,2)
    hold on
    ylabel('Normalized position')
    %plot(pos_norm(st_idx:end_idx),'k-','LineWidth',2)
    xlim([0 sum(plot_range)])
    %plot frames where sync event occurred
    plot([plot_range(1),plot_range(1)],[0 1],'Color',[0.5 0.5 0.5],'LineStyle', '--')
    %color blue A trials
    fr_assign_cut_A  = pos_norm(st_idx:end_idx);
    plot(find(frame_trial_assign(st_idx:end_idx)==2),fr_assign_cut_A(find(frame_trial_assign(st_idx:end_idx)==2)),'b')
    %color red B trials
    fr_assign_cut_B  = pos_norm(st_idx:end_idx);
    plot(find(frame_trial_assign(st_idx:end_idx)==3),fr_assign_cut_B(find(frame_trial_assign(st_idx:end_idx)==3)),'r')
    
    subplot(3,1,3)
    hold on
    plot(speed(st_idx:end_idx),'k','LineWidth',1)
    xlim([0 sum(plot_range)])
    ylabel('Speed [cm/s]');
    %plot frames where sync event occurred
    plot([plot_range(1),plot_range(1)],[-5 25],'Color',[0.5 0.5 0.5],'LineStyle', '--')
    
    pause
    clf;
end
end

%% Return the onset order for each SCE



%get overlap between run sequence neurons and SCE neurons
%[run_sce_neurons, run_sce_idx,~] = intersect(SCE_ROIs{sce_nb},neurons_participating{sce_nb},'stable');

%only ROIs in SCE that are part of run sequence
%run_SCE_ROIs = SCE_ROIs{sce_nb}(run_sce_idx);

%only SCE onsets of ROIS that are part of run sequence
%run_SCE_onsets = max_onset{sce_nb}(run_sce_idx);

%sort only run sequence involved neurons
%[~,I_sce_run] = sort(max_onset{sce_nb}(run_sce_idx),'ascend');

%sort SCE inputs by onset time
%[~,I_sce] = sort(max_onset{sce_nb},'ascend');




%time = session_vars{1, 1}.Behavior.resampled.time;

%[~,select_speed_idx,~] = intersect(time,time_choice,'stable');

%overwrite
%speed = speed(select_speed_idx);

%%
%{
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
imagesc(traces(st_idx:end_idx,SCE_ROIs{sce_nb}(I_sce))')
hold on
axis normal;
caxis([0 1]);
ylabel('Neuron #');
colormap(gca,'jet')
hold off

% only with RUN sequence neurons
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
imagesc(traces(st_idx:end_idx,run_SCE_ROIs(I_sce_run))')
hold on
title('RUN sequence neurons only');
axis normal;
caxis([0 1]);
ylabel('Neuron #');
colormap(gca,'jet')
hold off

%% Correlate SCE activation onset time with median (try mean)

%median normalized position onset across laps (run epochs)
median_run_seq_onset
%indices of RUN sequence neurons
recurring_neuron_idx
%neuron idxs of those involved in SCE and RUN sequence
run_SCE_ROIs

%relative onsets of neurons in SCEs (that are also in RUN sequence)
run_SCE_onsets

[~,~,recur_idx_pos] = intersect(run_SCE_ROIs,recurring_neuron_idx,'stable');

%correlate SCE onsets with median position of firing
[rho,p] =  corr(run_SCE_onsets', median_run_seq_onset(recur_idx_pos)','Type','Spearman')
%if p less than 0.05 and positive --> forward replay; 
%if spearman correlation negative --> reverse replay;
%}

end

