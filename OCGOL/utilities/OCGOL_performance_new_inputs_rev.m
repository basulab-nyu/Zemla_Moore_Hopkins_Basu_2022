function [Behavior] = OCGOL_performance_new_inputs_rev(Behavior)
%Get performance of the animal on the OCGOL trials

%% Assign variables
%cell with time start and stop of each lap
lap = Behavior.lap;

%time
time = Behavior.time;

%number of complete laps
lap_nb = size(lap,2);

lick = Behavior.lick;
%lap type (trial) based on reward location and trial type signal
lap_id = Behavior.lap_id;

%lap indices to which the 
reward_idx = Behavior.reward;

%time and position of each set of rewards - no longer used
rewards = Behavior.rewards;

%collect rewards
reward_coll = Behavior.reward_coll;

%start and stop lap idx's for constraining variables
start_lap_idx = find(time == lap{1}(1));
end_lap_idx = find(time == lap{end}(2));

%% Set parameters

%reward zone size (cm)
rew_size = 10;

%% lick position and time split by complete laps

for ll = 1:size(lap,2)
   lick_idx_temp = find(lick.time >= lap{ll}(1) &   lick.time <= lap{ll}(2));
   lick_lap{ll}.time = lick.time(lick_idx_temp);
   lick_lap{ll}.position = lick.position(lick_idx_temp);
   %set temp variable to empty
   lick_idx_temp = [];

end   

% %QC of licks across plots
% figure;
% for ll=1:size(lap,2)
% subplot(5,8,ll)
% hold on
% title(['Licks \newline lap: ', num2str(ll)])
% xlim([0,200])
% %stem(lick_lap{ll}.position, ones(1,size(lick_lap{ll}.position,1)),'r')
% hold off
% end

%% reward location (not collection)
%get reward position on each lap
%reward postion
reward_position = zeros(size(lap,2),1);
reward_pos_idx = zeros(size(lap,2),1);

%early rewards - B trials
reward_position(reward_idx.reward_early) = rewards{1}.position;
%idxs of associated reward positions
reward_pos_idx(reward_idx.reward_early) = rewards{1}.idx;
%late rewards - A trials
reward_position(reward_idx.reward_late) = rewards{2}.position;
%idxs of associated reward positions
reward_pos_idx(reward_idx.reward_late) = rewards{2}.idx;

%filter rewards that occur within full laps
reward_position((reward_pos_idx < start_lap_idx) | (reward_pos_idx > end_lap_idx)) = [];


%calulate mean and median of A and B trial reward zones onsets
%median
med_A_reward = median(rewards{2}.position);
med_B_reward = median(rewards{1}.position);

%mean
mean_A_reward = mean(rewards{2}.position);
mean_B_reward = mean(rewards{1}.position);

%% reward delivery/collected location - by lap

for ll = 1:size(lap,2)
   reward_coll_idx_temp = find(reward_coll.time >= lap{ll}(1) &   reward_coll.time <= lap{ll}(2));
   reward_coll_lap{ll}.time = reward_coll.time(reward_coll_idx_temp);
   reward_coll_lap{ll}.position = reward_coll.position(reward_coll_idx_temp);
   %set temp variable to empty
   reward_coll_idx_temp = [];

end

%number of rewards collected in each lap
for ll = 1:size(lap,2)
    nb_reward_coll_lap(ll) = size(reward_coll_lap{ll}.time,1); 
end

% %QC of reward collected across plots
% figure;
% for ll=1:size(lap,2)
%     subplot(5,8,ll)
%     hold on
%     title(['Rew col \newline lap: ', num2str(ll)])
%     xlim([0,200])
%     if ~isempty(reward_coll_lap{ll}.time)
%         stem(reward_coll_lap{ll}.position, ones(1,size(reward_coll_lap{ll}.position,1)),'r')
%     end
%     hold off
% end


%% separate the lick matrix into separate cells based on lapTimes matrix

%bin edges
edges = (0:2:200);

%bin licks on each lap
for ll = 1:size(lap,2)
    [N_licks(ll,:),~,~] = histcounts(lick_lap{ll}.position,edges,'Normalization','count');
end

%trial order - starting from first lap (transpose to be column vector)
trialOrder = lap_id.trial_based';

%assign to string names for plotting based on trial layout
for ii=1:size(trialOrder,1)
    if (trialOrder(ii) == 2)
        trialName{ii} = 'Ar'; %A
    elseif (trialOrder(ii) == 3)
        trialName{ii} = 'Br'; %B
    end
end

%% performance calculations - based on reward zone onset and lick signal

%correct criteria:
%1) lick in reward zone of lap trial type
%2) not lick in anticipatory zone
%3) at least 1 reward collected in reward zone associated with that lap

%this section checks that

%determine which laps were correct/wrong
%for each lap
for ii = 1:size(lap,2)
    %if A trial - check if licks in B ant or reward zone
    if trialOrder(ii) == 3
        %if there are licks in reward zone of A (2, far reward)
        if ~isempty(find(lick_lap{ii}.position >= reward_position(ii) & lick_lap{ii}.position <= reward_position(ii)+rew_size))
            %and not in ant or reward zone of B (based on mean reward
            %position)
            if isempty(find(lick_lap{ii}.position >= (mean_B_reward - rew_size) & lick_lap{ii}.position <= mean_B_reward+rew_size))
                trialCorrect(ii) = 1;
                trialCorrName{ii} = 'Y';
            else
                trialCorrect(ii) = 0;
                trialCorrName{ii} = 'N';mean_A_reward
                trialOrder(ii) = 30;
            end
            
        else
            trialCorrect(ii) = 0;
            trialCorrName{ii} = 'N';
            trialOrder(ii) = 30;
        end
        %if B trial
    elseif trialOrder(ii) == 2
        %if there are licks in reward zone of B (3, near reward)
        if ~isempty(find(lick_lap{ii}.position >= reward_position(ii) & lick_lap{ii}.position <= reward_position(ii)+rew_size))
            %and not in ant or reward zone of A (based on mean reward
            %position)
            if isempty(find(lick_lap{ii}.position >= (mean_A_reward - rew_size) & lick_lap{ii}.position <= mean_A_reward+rew_size))
                trialCorrect(ii) = 1;
                trialCorrName{ii} = 'Y';
            else
                trialCorrect(ii) = 0;
                trialCorrName{ii} = 'N';
                trialOrder(ii) = 20;
            end
            
        else
            trialCorrect(ii) = 0;
            trialCorrName{ii} = 'N';
            trialOrder(ii) = 20;
        end
        
    end
end

%% Check that at least 1 reward was collected in 

%determine which laps were correct/wrong
%for each lap
for ii = 1:size(lap,2)
    %if A trial, check that at least 1 reward delivered in reward zone area
    
    if trialOrder(ii) == 3
        if  ~isempty(find(reward_coll_lap{ii}.position >= reward_position(ii) & reward_coll_lap{ii}.position <= reward_position(ii)+rew_size))
            reward_collected_check(ii) = 1;
        else
            reward_collected_check(ii) = 0;
        end
        %if B trial, check that at least 1 reward delivered in reward zone area
    elseif trialOrder(ii) == 2
        if  ~isempty(find(reward_coll_lap{ii}.position >= reward_position(ii) & reward_coll_lap{ii}.position <= reward_position(ii)+rew_size))
            reward_collected_check(ii) = 1;
        else
            reward_collected_check(ii) = 0;
        end
    elseif trialOrder(ii) == 20 || trialOrder(ii) == 30 %wrong A or B trial
         if ~isempty(find(reward_coll_lap{ii}.position >= reward_position(ii) & reward_coll_lap{ii}.position <= reward_position(ii)+rew_size) | ...
                find(reward_coll_lap{ii}.position >= reward_position(ii) & reward_coll_lap{ii}.position <= reward_position(ii)+rew_size))   
            reward_collected_check(ii) = 1;
        else
            reward_collected_check(ii) = 0;
         end
    end
end

%check at least 1 reward collected on each lap
if sum(reward_collected_check) == lap_nb
    disp('At least 1 reward collected on each lap/trial.');
else
    disp('Missed reward collection on at least 1 lap/trial');
end

%% Missed lap == wrong lap (possible change in future)

%% Display fraction correct trials in command line

%all
frac_correct = sum(trialCorrect)./lap_nb;
%A trials
A_trial_idx = find(lap_id.trial_based == 2);
A_nb = length(A_trial_idx);

A_corr = trialCorrect(A_trial_idx);
frac_A = sum(A_corr)./A_nb;

%B trials
B_trial_idx = find(lap_id.trial_based == 3);
B_nb = length(B_trial_idx);

B_corr = trialCorrect(B_trial_idx);
frac_B = sum(B_corr)./B_nb;

%print resutls to command line
fprintf('Fraction of correct laps/trials: %0.2f \n', frac_correct);
fprintf('Total laps/trials: %d \n', lap_nb);
fprintf('Fraction of correct A trials: %0.2f \n', frac_A);
fprintf('Total A trials: %d \n', A_nb);
fprintf('Fraction of correct B trials: %0.2f \n', frac_B);
fprintf('Total B trials: %d \n', B_nb); 

%% plot
%as single plot with trials increase along y axis

% %raster plot indicating trial type and whether correct or not
% figure('Position',[150 150 900 600]);
% imagesc(N_licks);
% hold on;
% %trial type
% text(105*ones(lap_nb,1),1:lap_nb,trialName);
% %correct/wrong
% tc = text(110*ones(lap_nb,1),1:lap_nb,trialCorrName);
% %add red color to wrong trials
% 
% title('Lick map')
% ylabel('Lap #');
% xlabel('Spatial bin');
% colormap('jet');
% hold off

%% Save to structure
%performance.trialOrder = column vector with 3 (B),2 (A) corresponding to correct trials
%30 - (first number is trial type 3 - B trial, 2 - A trial, 0 - wrong, 1 - missed)
%performance.trialCorrect = columns vector 1 if correct, 0 = if wrong, -1 = missed

Behavior.performance.trialOrder = trialOrder;
Behavior.performance.trialCorrect = trialCorrect';


%export lick data for histogram figures
%number of licks = binned position (100 bins; 0-1 position)
Behavior.lick.bin_licks_lap = N_licks;
%position and time of licks on each lap
Behavior.lick.lick_lap = lick_lap;



end

