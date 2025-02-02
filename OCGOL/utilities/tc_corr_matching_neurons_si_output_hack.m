function [tc_corr_match] = tc_corr_matching_neurons_si_output_hack(session_vars,registered,options,tc_corr_match)


%% Parameters
sessionSelect = options.sessionSelect;
selectTrial = options.selectTrial;


%% Get STC maps
for ss = options.sessionSelect%1:size(animal_data,2)
    %normalized to A or B trials independently
    A_STC{ss} = session_vars{ss}.Place_cell{options.selectTrial(1)}.Spatial_tuning_curve;
    B_STC{ss} = session_vars{ss}.Place_cell{options.selectTrial(2)}.Spatial_tuning_curve;
    
    %Gs smoothed, but not normalized (nn) to itself
    %animal_data{ss}.Place_cell{1}.Spatial_Info.rate_map_smooth{8}
    %same as the following
    A_STC_noNorm{ss} = session_vars{ss}.Place_cell{options.selectTrial(1)}.Spatial_tuning_curve_no_norm;
    B_STC_noNorm{ss} = session_vars{ss}.Place_cell{options.selectTrial(2)}.Spatial_tuning_curve_no_norm;
  
    %dF/F (like STC - normalized) 
    %A_df{ss} = animal_data{ss}.Place_cell{1}.Spatial_tuning_dF;
    %B_df{ss} = animal_data{ss}.Place_cell{2}.Spatial_tuning_dF;
    
    %not occupancy normalized mean dF values across respective A and B
    %trials (not Gaussian smoothed)
    A_df_non_oc{ss} = session_vars{ss}.Place_cell{options.selectTrial(1)}.Spatial_Info.mean_dF_map{8};
    B_df_non_oc{ss} = session_vars{ss}.Place_cell{options.selectTrial(2)}.Spatial_Info.mean_dF_map{8};
    
end

%% Normalized to max STC value across A and B trials (for each ROI)
%for each session, take max value for each ROI
for ss=options.sessionSelect
    %make cumulative matrix for that session
    comb_STC = [A_STC_noNorm{ss}; B_STC_noNorm{ss}];
    %get min and max value for each ROI (min should all be 0 for STC based on event
    %map)
    min_STC = min(comb_STC,[],1);
    max_STC = max(comb_STC,[],1);
    %get max - min difference for each ROI for [0-1] normalization below
    diff_max_min_STC = max_STC - min_STC;
    
    %feature scale/normalize (0-1 range)
    A_STC_norm{ss} = (A_STC_noNorm{ss} - min_STC)./(diff_max_min_STC);
    B_STC_norm{ss} = (B_STC_noNorm{ss} - min_STC)./(diff_max_min_STC);
end

%% Get matching ROIs idxs for ts/event filtered neurons for A and B trials

%# of ROIs in match list
nbROI_match_list = size(registered.multi.assigned_filtered,1);

% (1)Need to change this to get a hack workaround for SI scores
%ROIs for A and B trials (TS)
matchROI_ts_event.A = registered.multi.matching_list_filtered.si_Aall_filt_event_filt(:,1:sessionSelect(end));
matchROI_ts_event.B = registered.multi.matching_list_filtered.si_Ball_filt_event_filt(:,1:sessionSelect(end));

% (2) Need to change this to get a hack workaround for SI scores
%construct ROI matrix for both A and B tuned  (TS)
A_filt_log = ~isnan(registered.multi.matching_list_filtered.si_Aall_filt_event_filt(:,1:sessionSelect(end)));
B_filt_log = ~isnan(registered.multi.matching_list_filtered.si_Ball_filt_event_filt(:,1:sessionSelect(end)));

%combined A and B filtered logical
AB_filt_log = A_filt_log & B_filt_log;

%create empty with nan
matchROI_ts_event.AB = nan(nbROI_match_list, sessionSelect(end));
%populate
matchROI_ts_event.AB(find(AB_filt_log == 1)) = matchROI_ts_event.A(find(AB_filt_log == 1));

%% Construct matching STC matrices for neighboring day and all cross-day arrangments

%relative to day 1
for ii=2:sessionSelect(end)
    %A ts/event tuned
    matching_ROI_d1_idx.ts.A{1,ii} = matchROI_ts_event.A(sum(~isnan(matchROI_ts_event.A(:,[1, ii])),2) == 2,[1, ii]);
    %B ts/event tuned
    matching_ROI_d1_idx.ts.B{1,ii} = matchROI_ts_event.B(sum(~isnan(matchROI_ts_event.B(:,[1, ii])),2) == 2,[1, ii]);
    %A/B ts/event tuned
    matching_ROI_d1_idx.ts.AB{1,ii} = matchROI_ts_event.AB(sum(~isnan(matchROI_ts_event.AB(:,[1, ii])),2) == 2,[1, ii]);
end

%relative to all session combinations
%swap the index variables around (for consistency - shouldn't affect
%results)

for ii=sessionSelect
    for jj = sessionSelect
        %A ts/event tuned
        matching_ROI_all_day_idx.ts.A{ii,jj} = matchROI_ts_event.A(sum(~isnan(matchROI_ts_event.A(:,[ii, jj])),2) == 2,[ii, jj]);
        %B ts/event tuned
        matching_ROI_all_day_idx.ts.B{ii,jj} = matchROI_ts_event.B(sum(~isnan(matchROI_ts_event.B(:,[ii, jj])),2) == 2,[ii, jj]);
        %A/B ts/event tuned
        matching_ROI_all_day_idx.ts.AB{ii,jj} = matchROI_ts_event.AB(sum(~isnan(matchROI_ts_event.AB(:,[ii, jj])),2) == 2,[ii, jj]);
    end
end

%get position of microtextures
%use session 1 as reference (position should ~ same across session)
for tex=1:4
    tex_pos(tex) = nanmean(session_vars{1}.Behavior.textures{tex}.position_norm);
end

%get position of reward onsets
for rew=1:2
    rew_pos(rew) = nanmean(session_vars{1}.Behavior.rewards{rew}.position_norm);
end

%% Construct STCs for all A (ts/event) matching neurons and all B (ts/event) matching neurons (all day combinations) (not A&B analysis)

for ii=sessionSelect
    for jj=sessionSelect

        %get trial normalized STC matrices - sample D1 vs. D3
        %A
        matching_ROI_all_day_STC.ts.A{ii,jj}{1} = A_STC_norm{ii}(:,matching_ROI_all_day_idx.ts.A{ii,jj}(:,1));
        matching_ROI_all_day_STC.ts.A{ii,jj}{2} = A_STC_norm{jj}(:,matching_ROI_all_day_idx.ts.A{ii,jj}(:,2));
        
        %B
        matching_ROI_all_day_STC.ts.B{ii,jj}{1} = B_STC_norm{ii}(:,matching_ROI_all_day_idx.ts.B{ii,jj}(:,1));
        matching_ROI_all_day_STC.ts.B{ii,jj}{2} = B_STC_norm{jj}(:,matching_ROI_all_day_idx.ts.B{ii,jj}(:,2));
        
        %convert STCs to cell (ROI x 200 bins - 100 first ses; 100 next ses)
        STC_mat_A{ii,jj} = cell2mat(matching_ROI_all_day_STC.ts.A{ii,jj}')';
        STC_mat_B{ii,jj} = cell2mat(matching_ROI_all_day_STC.ts.B{ii,jj}')';
        
        %get count of neurons in each session
        %A to A matches ROIs 
        TC_corr_counts.ts.A(ii,jj) = size(matching_ROI_all_day_idx.ts.A{ii,jj},1);
        %B to B matches ROIs
        TC_corr_counts.ts.B(ii,jj) = size(matching_ROI_all_day_idx.ts.B{ii,jj},1);
        
        %sort by d1
        %maxBin - spatial bin where activity is greatest for each ROI
        [~,maxBin_all_A] = max(STC_mat_A{ii,jj}(:,1:100)', [], 1,'includenan');
        %sortIdx - arrangment of ROIs after sorting by max spatial bin acitivity
        [~,sortOrder_all_A] = sort(maxBin_all_A,'ascend');
        
        STC_mat_A_sort{ii,jj} = STC_mat_A{ii,jj}(sortOrder_all_A,:);
        
        %maxBin - spatial bin where activity is greatest for each ROI
        [~,maxBin_all_B] = max(STC_mat_B{ii,jj}(:,1:100)', [], 1,'includenan');
        %sortIdx - arrangment of ROIs after sorting by max spatial bin acitivity
        [~,sortOrder_all_B] = sort(maxBin_all_B,'ascend');
        
        STC_mat_B_sort{ii,jj} = STC_mat_B{ii,jj}(sortOrder_all_B,:);
        
        %% Correlate matching A and cross-correlate STC (correlated between sorted variables)
        %correlation for A
        if ~isempty(STC_mat_A_sort{ii,jj})
            %this is using sorted data
            TC_corr_all_day{ii,jj}.A = corr(STC_mat_A_sort{ii,jj}(:,1:100)',STC_mat_A_sort{ii,jj}(:,101:200)','Type','Pearson','rows','all');
            %this is using unsorted data
            TC_corr_all_day_nonsorted{ii,jj}.A = corr(STC_mat_A{ii,jj}(:,1:100)',STC_mat_A{ii,jj}(:,101:200)','Type','Pearson','rows','all');
            %this is sorted
            PV_corr_all_day{ii,jj}.A = corr(STC_mat_A_sort{ii,jj}(:,1:100),STC_mat_A_sort{ii,jj}(:,101:200),'Type','Pearson','rows','all');
        else
            TC_corr_all_day{ii,jj}.A = [];
            PV_corr_all_day{ii,jj}.A = [];
            TC_corr_all_day_nonsorted{ii,jj}.A = [];
        end
        
        %correlation for B
        if ~isempty(STC_mat_B_sort{ii,jj})
            TC_corr_all_day{ii,jj}.B = corr(STC_mat_B_sort{ii,jj}(:,1:100)',STC_mat_B_sort{ii,jj}(:,101:200)','Type','Pearson','rows','all');
            %this is using unsorted data
            TC_corr_all_day_nonsorted{ii,jj}.B = corr(STC_mat_B{ii,jj}(:,1:100)',STC_mat_B{ii,jj}(:,101:200)','Type','Pearson','rows','all');            
            
            PV_corr_all_day{ii,jj}.B = corr(STC_mat_B_sort{ii,jj}(:,1:100),STC_mat_B_sort{ii,jj}(:,101:200),'Type','Pearson','rows','all');
        else
            TC_corr_all_day{ii,jj}.B = [];
            PV_corr_all_day{ii,jj}.B = [];
            
            TC_corr_all_day_nonsorted{ii,jj}.B = [];
        end
    end
end

%QC check
%check if they are the same
% mean(diag(TC_corr_all_nonsorted{2,7}.A)),mean(diag(TC_corr_all_day{2,7}.A))
% mean(diag(TC_corr_all_nonsorted{2,7}.B)),mean(diag(TC_corr_all_day{2,7}.B))



%check if non-sorted yields same results as sorted

% %For visualization
% test_STC  =cell2mat(matching_ROI_all_day_STC.ts.A{1, 6}')';
% 
% [~,test_maxBin] = max(test_STC(:,101:200)', [], 1,'includenan');
% %sortIdx - arrangment of ROIs after sorting by max spatial bin acitivity
% [~,test_sortOrder] = sort(test_maxBin,'ascend');
% 
% test_STC_sort = test_STC(test_sortOrder,:);
% 
% figure
% imagesc(test_STC_sort)


%% Do for A&B matching neurons

for ii=sessionSelect
    for jj=sessionSelect
        
        %get trial normalized STC matrices - sample D1 vs. D3
        %A
        matching_ROI_all_day_STC.ts.AB.A{ii, jj}{1} = A_STC_norm{ii}(:,matching_ROI_all_day_idx.ts.AB{ii, jj}(:,1));
        matching_ROI_all_day_STC.ts.AB.A{ii, jj}{2} = A_STC_norm{jj}(:,matching_ROI_all_day_idx.ts.AB{ii, jj}(:,2));
        
        %B
        matching_ROI_all_day_STC.ts.AB.B{ii, jj}{1} = B_STC_norm{ii}(:,matching_ROI_all_day_idx.ts.AB{ii, jj}(:,1));
        matching_ROI_all_day_STC.ts.AB.B{ii, jj}{2} = B_STC_norm{jj}(:,matching_ROI_all_day_idx.ts.AB{ii, jj}(:,2));
        
        % index these STCs -TODO
        STC_mat_AB_A{ii,jj} = cell2mat(matching_ROI_all_day_STC.ts.AB.A{ii, jj}')';
        STC_mat_AB_B{ii,jj} = cell2mat(matching_ROI_all_day_STC.ts.AB.B{ii, jj}')';
        
        %A&B matches
        TC_corr_counts.ts.AB(ii,jj) = size(matching_ROI_all_day_idx.ts.AB{ii,jj},1);
        
        %sort by d1
        %maxBin - spatial bin where activity is greatest for each ROI
        [~,maxBin_all_AB_A] = max(STC_mat_AB_A{ii,jj}(:,1:100)', [], 1,'includenan');
        %sortIdx - arrangment of ROIs after sorting by max spatial bin acitivity
        [~,sortOrder_all_AB_A] = sort(maxBin_all_AB_A,'ascend');
        
        STC_mat_AB_A_sort{ii,jj} = STC_mat_AB_A{ii,jj}(sortOrder_all_AB_A,:);
        
        
        %maxBin - spatial bin where activity is greatest for each ROI
        [~,maxBin_all_AB_B] = max(STC_mat_AB_B{ii,jj}(:,1:100)', [], 1,'includenan');
        %sortIdx - arrangment of ROIs after sorting by max spatial bin acitivity
        [~,sortOrder_all_AB_B] = sort(maxBin_all_AB_B,'ascend');
        
        STC_mat_AB_B_sort{ii,jj} = STC_mat_AB_B{ii,jj}(sortOrder_all_AB_B,:);
        
        %sort by A order for A vs. B comparison
        STC_mat_AB_B_sortRelA{ii,jj} = STC_mat_AB_B{ii,jj}(sortOrder_all_AB_A,:);
        
        %% Correlate matching A and cross-correlate STC
        %A against A for A&B neurons
        if ~isempty(STC_mat_AB_A_sort{ii,jj})
            TC_corr_all_day{ii, jj}.AB_A = corr(STC_mat_AB_A_sort{ii,jj}(:,1:100)',STC_mat_AB_A_sort{ii,jj}(:,101:200)','Type','Pearson','rows','all');
            %nonsorted A vs. A correlation
            TC_corr_all_day_nonsorted{ii, jj}.AB_A = corr(STC_mat_AB_A{ii,jj}(:,1:100)',STC_mat_AB_A{ii,jj}(:,101:200)','Type','Pearson','rows','all');
            
            PV_corr_all_day{ii, jj}.AB_A = corr(STC_mat_AB_A_sort{ii,jj}(:,1:100),STC_mat_AB_A_sort{ii,jj}(:,101:200),'Type','Pearson','rows','all');
            %PV_corr{sel_day}.AB_A = corr(STC_mat_AB_A_sort(:,1:100),STC_mat_AB_A_sort(:,101:200),'Type','Pearson','rows','all');
        else
            TC_corr_all_day{ii, jj}.AB_A = [];
            TC_corr_all_day_nonsorted{ii, jj}.AB_A = [];
            PV_corr_all_day{ii, jj}.AB_A = [];
        end
        
        %B against B for A&B neurons
        if ~isempty(STC_mat_AB_B_sort{ii,jj})
            TC_corr_all_day{ii, jj}.AB_B = corr(STC_mat_AB_B_sort{ii,jj}(:,1:100)',STC_mat_AB_B_sort{ii,jj}(:,101:200)','Type','Pearson','rows','all');
            %unsorted B vs. B correlation
            TC_corr_all_day_nonsorted{ii, jj}.AB_B = corr(STC_mat_AB_B{ii,jj}(:,1:100)',STC_mat_AB_B{ii,jj}(:,101:200)','Type','Pearson','rows','all');
            
            PV_corr_all_day{ii, jj}.AB_B = corr(STC_mat_AB_B_sort{ii,jj}(:,1:100),STC_mat_AB_B_sort{ii,jj}(:,101:200),'Type','Pearson','rows','all');
        else            
            TC_corr_all_day{ii, jj}.AB_B = [];
            TC_corr_all_day_nonsorted{ii, jj}.AB_B = [];
            PV_corr_all_day{ii, jj}.AB_B = [];
        end
        %A against B for A&B neurons (first session vs. later session)
        
        if ~isempty(STC_mat_AB_A_sort{ii,jj}) || ~isempty(STC_mat_AB_B_sortRelA{ii,jj})
            %A vs.B correlation on each day
            %TC
            TC_corr_all_day{ii, jj}.AB_AB_early = corr(STC_mat_AB_A_sort{ii,jj}(:,1:100)',STC_mat_AB_B_sortRelA{ii,jj}(:,1:100)','Type','Pearson','rows','all');
            TC_corr_all_day{ii, jj}.AB_AB_later = corr(STC_mat_AB_A_sort{ii,jj}(:,101:200)',STC_mat_AB_B_sortRelA{ii,jj}(:,101:200)','Type','Pearson','rows','all');
            
            %non-sorted A vs. B
            TC_corr_all_day_nonsorted{ii, jj}.AB_AB_early = corr(STC_mat_AB_A{ii,jj}(:,1:100)',STC_mat_AB_B{ii,jj}(:,1:100)','Type','Pearson','rows','all');
            TC_corr_all_day_nonsorted{ii, jj}.AB_AB_later = corr(STC_mat_AB_A{ii,jj}(:,101:200)',STC_mat_AB_B{ii,jj}(:,101:200)','Type','Pearson','rows','all');
            
            %PV
            PV_corr_all_day{ii, jj}.AB_AB_early = corr(STC_mat_AB_A_sort{ii,jj}(:,1:100),STC_mat_AB_B_sortRelA{ii,jj}(:,1:100),'Type','Pearson','rows','all');
            PV_corr_all_day{ii, jj}.AB_AB_later = corr(STC_mat_AB_A_sort{ii,jj}(:,101:200),STC_mat_AB_B_sortRelA{ii,jj}(:,101:200),'Type','Pearson','rows','all');
            
        else
            TC_corr_all_day{ii, jj}.AB_AB_early = [];
            TC_corr_all_day{ii, jj}.AB_AB_later = [];
            
            TC_corr_all_day_nonsorted{ii, jj}.AB_AB_early = [];
            TC_corr_all_day_nonsorted{ii, jj}.AB_AB_later = [];
            
            PV_corr_all_day{ii, jj}.AB_AB_early = [];
            PV_corr_all_day{ii, jj}.AB_AB_later = [];
        end
        
    end
end

%QC check
%check if they are the same
%A&B A vs. A
%A&B B vs. B
%A&B A vs. B

%QC checked
%A&B A vs. A
% mean(diag(TC_corr_all_day{2,7}.AB_A)),mean(diag(TC_corr_all_day_nonsorted{2,7}.AB_A))
% %A&B B vs. B
% mean(diag(TC_corr_all_day{2,7}.AB_B)),mean(diag(TC_corr_all_day_nonsorted{2,7}.AB_B))
% %early
% mean(diag(TC_corr_all_day{2,7}.AB_AB_early)),mean(diag(TC_corr_all_day_nonsorted{2,7}.AB_AB_early))
% %late
% mean(diag(TC_corr_all_day{2,7}.AB_AB_later)),mean(diag(TC_corr_all_day_nonsorted{2,7}.AB_AB_later))


% TC_corr_all_day{ii, jj}.AB_A
% TC_corr_all_day_nonsorted{ii, jj}.AB_A
% 
% TC_corr_all_day{ii, jj}.AB_B
% TC_corr_all_day_nonsorted{ii, jj}.AB_B
% 
% TC_corr_all_day{ii, jj}.AB_AB_early
% TC_corr_all_day{ii, jj}.AB_AB_later
% 
% TC_corr_all_day_nonsorted{ii, jj}.AB_AB_early
% TC_corr_all_day_nonsorted{ii, jj}.AB_AB_later

% %For visualization
% test_STC  =cell2mat({matching_ROI_all_day_STC.ts.AB.A{1, 3}{2} matching_ROI_all_day_STC.ts.AB.B{1, 3}{2} }')';
% 
% [~,test_maxBin] = max(test_STC(:,1:100)', [], 1,'includenan');
% %sortIdx - arrangment of ROIs after sorting by max spatial bin acitivity
% [~,test_sortOrder] = sort(test_maxBin,'ascend');
% 
% test_STC_sort = test_STC(test_sortOrder,:);
% 
% figure
% imagesc(test_STC_sort)



%% Plot A vs. A anfigure
for ii=sessionSelect
    subplot(sessionSelect(end),1,ii)
    hold on
    title('all A and all B tuned matching neurons')
    ylim([-0.1 0.8])

    p1 = plot(diag(PV_corr_all_day{1,ii}.A),'b');
    p2 = plot(diag(PV_corr_all_day{1,ii}.B),'r');

    %plot microtexture locations
    for tex=1:4
    plot([tex_pos(tex), tex_pos(tex)].*100,[-0.1 0.8],'k--')
    end
    
    %plot B zone
    plot([rew_pos(1), rew_pos(1)].*100,[-0.1 0.8],'r--')
    %plot A zone
    plot([rew_pos(2), rew_pos(2)].*100,[-0.1 0.8],'b--')
    
        %add legend for day labeling
    legend([p1 p2], {'A', 'B'},'Location','northeast')
end
%d B vs. B correlations for matching ROIs all A and all B tuned ROIs - rel d1

%% Plot A vs. A and B vs. B correlations for matching ROIs A&B tuned ROIs
y_lim_range = [-0.2 1];

figure
for ii=sessionSelect
    subplot(sessionSelect(end),1,ii)
    hold on
    title('A&B tuned matching neurons')
    ylim(y_lim_range)
    p1 = plot(diag(PV_corr_all_day{1,ii}.AB_A),'b');
    p2 = plot(diag(PV_corr_all_day{1,ii}.AB_B),'r');
    %plot microtexture locations
    for tex=1:4
    plot([tex_pos(tex), tex_pos(tex)].*100,y_lim_range,'k--')
    end
    
    %plot B zone
    plot([rew_pos(1), rew_pos(1)].*100,y_lim_range,'r--')
    %plot A zone
    plot([rew_pos(2), rew_pos(2)].*100,y_lim_range,'b--')
    
        %add legend for day labeling
    legend([p1 p2], {'A', 'B'},'Location','northeast')
end

%% Plot A vs. B correlations for matching ROIs A&B tuned ROIs

y_lim_range = [-0.2 1];

figure
for ii=sessionSelect
    subplot(sessionSelect(end),1,ii)
    hold on
    title('A vs. B TC correlation for A&B tuned across days');
    xlabel('Position')
    ylim(y_lim_range)
    p1 = plot(diag(PV_corr_all_day{1, ii}.AB_AB_early),'Color',[139, 0, 139]./255,'LineStyle','-');
    p2 = plot(diag(PV_corr_all_day{1, ii}.AB_AB_later),'Color',[139, 0, 139]./255,'LineStyle','--');
    
    %plot microtexture locations
    for tex=1:4
        plot([tex_pos(tex), tex_pos(tex)].*100,y_lim_range,'k--')
    end
    
    %plot B zone
    plot([rew_pos(1), rew_pos(1)].*100,y_lim_range,'r--')
    %plot A zone
    plot([rew_pos(2), rew_pos(2)].*100,y_lim_range,'b--')
    
    %add legend for day labeling
    legend([p1 p2], {'Day 1', ['Day ', num2str(ii)]},'Location','southeast')
end


%% Export correlations for analysis

%TC correlations are most important here
%correlation matrix that is not sorted (same results as sorted one, but
%can match to ROIs later)
tc_corr_match.si.TC_corr_all_day_nonsorted = TC_corr_all_day_nonsorted;
tc_corr_match.si.TC_corr_all_day = TC_corr_all_day;

%spatial tuning curves for A&B tuned neurons
tc_corr_match.si.STC_mat_AB_A = STC_mat_AB_A;
tc_corr_match.si.STC_mat_AB_B = STC_mat_AB_B;

%spatial tuning curves for A tuned neurons
tc_corr_match.si.STC_mat_A  = STC_mat_A;
%spatial tuning curves for B tuned neurons
tc_corr_match.si.STC_mat_B = STC_mat_B;

%number of neurons counted for each A,B, and A&B correlation
tc_corr_match.si.TC_corr_counts = TC_corr_counts;

%export matching indices (already paired for each day)
tc_corr_match.si.matching_ROI_all_day_idx = matching_ROI_all_day_idx.ts;


%export computed values
tc_corr_match.si.PV_corr_all_day = PV_corr_all_day;


tc_corr_match.tex_pos = tex_pos;
tc_corr_match.rew_pos = rew_pos;

% all STCs below
tc_corr_match.si.matching_ROI_all_day_STC = matching_ROI_all_day_STC;




end

