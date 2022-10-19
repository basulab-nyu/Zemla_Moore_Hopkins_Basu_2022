cleer;
colormap parula;

LOAD = 1;

if(LOAD==1)
    base = '\\research-cifs.nyumc.org\Research\basulab\basulabspace\RZ\data_for_Jason\matching_binary_events_all_lap_learn';
else
    base = 'F:\MATLAB\D_Drive\RZ_Data\matching_binary_events_all_lap_learn';
end
% base = '\\research-cifs.nyumc.org\Research\basulab\basulabspace\RZ\data_for_Jason\matching_binary_events_all_lap_recall';
mice = dir(base);
mice = arrayfun(@(x) x.name, mice(3:end-1), 'UniformOutput', false);


totalMice = 6;
maxSessions = 9;

ALL_TEMPLATE = cell(totalMice, maxSessions);
ALL_BIN_RATE = cell(totalMice, maxSessions);
ALL_TRUE_POS = cell(totalMice, maxSessions);
ALL_TRUE_RUN = cell(totalMice, maxSessions);
ALL_TRUE_TYPE = cell(totalMice, maxSessions);
for i_mouse = 1:totalMice
    load(fullfile(base,mice{i_mouse},'matched_transients_all_ses.mat'),'matching_ROI_bin_transient_lap_data');

    %%
    these = matching_ROI_bin_transient_lap_data;

    TRUE_POS = cell(1,maxSessions);
    TRUE_RUN = cell(1,maxSessions);
    TRUE_TYPE = cell(1,maxSessions);
    TEMPLATE = cell(1,maxSessions);
    BIN_RATE = cell(1,maxSessions);    
    for i_ses = 1:length(these)
    %     i_ses = 6;

        this = these{i_ses};

        N_BINS = 40;
        edgesPosition = linspace(0,1,N_BINS+1);
        centersPosition = center(edgesPosition);

        SM = 1;
        DT = nanmedian(diff(this.time));
        N_CELLS = size(this.transient_mat,1);


        cumulativeDistance = unwrap(this.position_norm*2*pi)/(2*pi);
        midDistance = cumulativeDistance(jzeroCrossing(cumulativeDistance-cumulativeDistance(end)/2));

        HALF = 3;
        if(HALF==1)
            valid1 = cumulativeDistance<midDistance;    
            valid2 = cumulativeDistance>=midDistance;    
        elseif(HALF==2)
            valid1 = cumulativeDistance>=midDistance;
            valid2 = cumulativeDistance<midDistance;    
        else
            valid1 = true(size(cumulativeDistance));
            valid2 = true(size(cumulativeDistance));
        end

        occA = DT*histcn(this.position_norm(this.run_state==1 & this.trialType==2 & valid1),edgesPosition);
        occA = jmm_smooth_1d_cor_circ(occA,SM);
        occA = occA(1:N_BINS);

        occB = DT*histcn(this.position_norm(this.run_state==1 & this.trialType==3 & valid1),edgesPosition);
        occB = jmm_smooth_1d_cor_circ(occB,SM);
        occB = occB(1:N_BINS);

        timeBin = 0.25;
        SM2 = ceil(timeBin/DT);

        timeEdges = this.time(1):timeBin:this.time(end)+timeBin;
        timeCenters = center(timeEdges);
        coarseRate = NaN(N_CELLS,length(timeEdges)-1);
        eventsA = NaN(N_CELLS,N_BINS);
        eventsB = NaN(N_CELLS,N_BINS);
        for i_cell = 1:N_CELLS
            eventBinaryAll = (this.transient_mat(i_cell,:).*(this.run_state==1 & valid2)'==1);
            eventBinaryA = (this.transient_mat(i_cell,:).*(this.run_state==1 & this.trialType==2 & valid1)'==1);
            eventBinaryB = (this.transient_mat(i_cell,:).*(this.run_state==1 & this.trialType==3 & valid1)'==1);
        
            eventPosA = this.position_norm(eventBinaryA);
            eventPosB = this.position_norm(eventBinaryB);

            eventA = histcn(eventPosA,edgesPosition);
            eventA = jmm_smooth_1d_cor_circ(eventA,SM);
            eventA = eventA(1:N_BINS);
            eventsA(i_cell,:) = eventA;

            eventB = histcn(eventPosB,edgesPosition);
            eventB = jmm_smooth_1d_cor_circ(eventB,SM);
            eventB = eventB(1:N_BINS);
            eventsB(i_cell,:) = eventB;

            smoothedEventRate = jmm_smooth_1d_cor(double(eventBinaryAll), 2*SM2)/timeBin;
            coarseRate(i_cell,:) = interp1(this.time,smoothedEventRate,timeCenters);
        end

        rateA = bsxfun(@rdivide, eventsA, occA);
        rateB = bsxfun(@rdivide, eventsB, occB);
        rateTogether = [rateA rateB];
        %
        TEMPLATE{i_ses} = rateTogether;
        BIN_RATE{i_ses} = coarseRate;
                        
        downsampledPosition = interp1(this.time, this.position_norm, timeCenters,'nearest');
        downsampledRunState = interp1(this.time, 1*(this.run_state==1 & this.trialType>0), timeCenters,'nearest');
        downsampledPosition(downsampledRunState==0) = NaN;
        downsampledTrialType = interp1(this.time, this.trialType, timeCenters,'nearest');
        downsampledTrialType(downsampledRunState==0) = NaN;

        TRUE_POS{i_ses} = downsampledPosition;
        TRUE_RUN{i_ses} = downsampledRunState;
        TRUE_TYPE{i_ses} = downsampledTrialType;
        
       
    end
    ALL_TEMPLATE(i_mouse,:) = TEMPLATE;
    ALL_BIN_RATE(i_mouse,:) = BIN_RATE;
    ALL_TRUE_POS(i_mouse,:) = TRUE_POS;
    ALL_TRUE_RUN(i_mouse,:) = TRUE_RUN;
    ALL_TRUE_TYPE(i_mouse,:) = TRUE_TYPE;
end


%%
DECODING_L = NaN(totalMice, size(ALL_TEMPLATE,2), size(ALL_TEMPLATE,2));
ERROR_L = NaN(totalMice, size(ALL_TEMPLATE,2), size(ALL_TEMPLATE,2));
PREDICTED_POS = cell(totalMice, size(ALL_TEMPLATE,2), size(ALL_TEMPLATE,2));
for i_mouse = 1:totalMice
    for i_train = 1:size(ALL_TEMPLATE,2)
        this_template = ALL_TEMPLATE{i_mouse, i_train};
        if(isempty(this_template))
            continue;
        end           
        for i_pred = 1:size(ALL_TEMPLATE,2)
            this_data = ALL_BIN_RATE{i_mouse, i_pred};
            this_true_pos = ALL_TRUE_POS{i_mouse, i_pred};
            this_true_type = ALL_TRUE_TYPE{i_mouse, i_pred};
            this_true_run = ALL_TRUE_RUN{i_mouse, i_pred};

            if(isempty(this_data))
                continue;
            end
            tp2 = 200*(this_true_pos+this_true_type-2);
            tp2(this_true_run==0) = NaN;
            
            centersPosition2 = 200*[centersPosition centersPosition+1];
            nValid = sum(sum(this_template,2)>0);
            c = zscore(this_template)'*zscore(this_data)/(nValid-1);
            [~, maxBin] = max(c);
            confidence = max(c) - quantile(c,0.85);

            pp2 = centersPosition2(maxBin);
            pp2(this_true_run==0) = NaN;
            
            PREDICTED_POS{i_mouse, i_train, i_pred} = pp2;
            AA = sum(tp2<200 & pp2<200);
            AB = sum(tp2<200 & pp2>=200);
            BA = sum(tp2>=200 & pp2<200);
            BB = sum(tp2>=200 & pp2>=200);
    
            DECODING_L(i_mouse, i_train, i_pred) = (AA+BB)/(AA+AB+BA+BB);
    
            error = 100*abs(angDiff(2*pi*pp2/200,2*pi*tp2/200))/pi;
        
            ERROR_L(i_mouse, i_train, i_pred) = nanmean(error);
            
        end
    end
end
ERROR_L(3,2,:) = NaN;
ERROR_L(3,:,2) = NaN;
DECODING_L(3,2,:) = NaN;
DECODING_L(3,:,2) = NaN;

ALL_TRUE_POS_0 = ALL_TRUE_POS;
ALL_TRUE_RUN_0 = ALL_TRUE_RUN;
ALL_TRUE_TYPE_0 = ALL_TRUE_TYPE;
ALL_PREDICTED_POS_0 = PREDICTED_POS;
%%
diags = -8:8;
totalMice = 6;
decoding_dayDecay_L = NaN(totalMice, length(diags));
error_dayDecay_L = NaN(totalMice, length(diags));

decoding_dayDecay2_L = NaN(totalMice, maxSessions);
error_dayDecay2_L = NaN(totalMice, maxSessions);


for i_mouse = 1:totalMice
    if(i_mouse>size(ERROR_L,1))
        continue;
    end        
    this = squeeze(ERROR_L(i_mouse,:,:));
    error_dayDecay_L(i_mouse,:) = arrayfun(@(x) nanmean(diag(this,x)), diags);
    
    temp = diag(this,1);
    error_dayDecay2_L(i_mouse,1:length(temp)) = temp;
    
    this = squeeze(DECODING_L(i_mouse,:,:));
    decoding_dayDecay_L(i_mouse,:) = arrayfun(@(x) nanmean(diag(this,x)), diags);
    temp = diag(this,1);
    decoding_dayDecay2_L(i_mouse,1:length(temp)) = temp;    
end

CA1 = [0.47 0.97];
CA2 = [3.0 34.4];
CA2 = -CA2(end:-1:1);







%%
if(LOAD==1)
    base = '\\research-cifs.nyumc.org\Research\basulab\basulabspace\RZ\data_for_Jason\matching_binary_events_all_lap_learn';
else
    base = 'F:\MATLAB\D_Drive\RZ_Data\matching_binary_events_all_lap_learn';
end
% base = '\\research-cifs.nyumc.org\Research\basulab\basulabspace\RZ\data_for_Jason\matching_binary_events_all_lap_recall';
mice = dir(base);
mice = arrayfun(@(x) x.name, mice(3:end-1), 'UniformOutput', false);

if(LOAD==1)
    load('\\research-cifs.nyumc.org\Research\basulab\basulabspace\RZ\_revision_scripts_data_Jason\tuned_match_ROI_learn_recall_data.mat', 'short_term_learn');
    load('\\research-cifs.nyumc.org\Research\basulab\basulabspace\RZ\_revision_scripts_data_Jason\tuned_match_ROI_learn_recall_data.mat', 'reg_learn');
else
    load('F:\MATLAB\D_Drive\RZ_Data\tuned_match_ROI_learn_recall_data.mat', 'short_term_learn');
    load('F:\MATLAB\D_Drive\RZ_Data\tuned_match_ROI_learn_recall_data.mat', 'reg_learn');
end
totalMice = 6;
maxSessions = 9;

ALL_TEMPLATE = cell(totalMice, maxSessions);
ALL_TEMPLATE_FULL = cell(totalMice, maxSessions);
ALL_BIN_RATE = cell(totalMice, maxSessions);
ALL_TRUE_POS = cell(totalMice, maxSessions);
ALL_TRUE_RUN = cell(totalMice, maxSessions);
ALL_TRUE_TYPE = cell(totalMice, maxSessions);
for i_mouse = 1:totalMice
    load(fullfile(base,mice{i_mouse},'matched_transients_all_ses.mat'),'matching_ROI_bin_transient_lap_data');

    %%
    these = matching_ROI_bin_transient_lap_data;

    TRUE_POS = cell(1,maxSessions);
    TRUE_RUN = cell(1,maxSessions);
    TRUE_TYPE = cell(1,maxSessions);
    TEMPLATE = cell(1,maxSessions);
    TEMPLATE_FULL = cell(1,maxSessions);
    BIN_RATE = cell(1,maxSessions);    
    for i_ses = 1:length(these)
    %     i_ses = 6;

        this = these{i_ses};

        N_BINS = 40;
        edgesPosition = linspace(0,1,N_BINS+1);
        centersPosition = center(edgesPosition);

        SM = 1;
        DT = nanmedian(diff(this.time));
        N_CELLS = size(this.transient_mat,1);


        cumulativeDistance = unwrap(this.position_norm*2*pi)/(2*pi);
        midDistance = cumulativeDistance(jzeroCrossing(cumulativeDistance-cumulativeDistance(end)/2));

        HALF = 1;
        if(HALF==1)
            valid1 = cumulativeDistance<midDistance;    
            valid2 = cumulativeDistance>=midDistance;    
        elseif(HALF==2)
            valid1 = cumulativeDistance>=midDistance;
            valid2 = cumulativeDistance<midDistance;    
        else
            valid1 = true(size(cumulativeDistance));
            valid2 = true(size(cumulativeDistance));
        end

        occA = DT*histcn(this.position_norm(this.run_state==1 & this.trialType==2 & valid1),edgesPosition);
        occA = jmm_smooth_1d_cor_circ(occA,SM);
        occA = occA(1:N_BINS);

        occB = DT*histcn(this.position_norm(this.run_state==1 & this.trialType==3 & valid1),edgesPosition);
        occB = jmm_smooth_1d_cor_circ(occB,SM);
        occB = occB(1:N_BINS);

        occA_full = DT*histcn(this.position_norm(this.run_state==1 & this.trialType==2),edgesPosition);
        occA_full = jmm_smooth_1d_cor_circ(occA_full,SM);
        occA_full = occA_full(1:N_BINS);

        occB_full = DT*histcn(this.position_norm(this.run_state==1 & this.trialType==3),edgesPosition);
        occB_full = jmm_smooth_1d_cor_circ(occB_full,SM);
        occB_full = occB_full(1:N_BINS);
        
        timeBin = 0.25;
        SM2 = ceil(timeBin/DT);

        timeEdges = this.time(1):timeBin:this.time(end)+timeBin;
        timeCenters = center(timeEdges);
        coarseRate = NaN(N_CELLS,length(timeEdges)-1);
        eventsA = NaN(N_CELLS,N_BINS);
        eventsB = NaN(N_CELLS,N_BINS);
        eventsA_full = NaN(N_CELLS,N_BINS);
        eventsB_full = NaN(N_CELLS,N_BINS);
        for i_cell = 1:N_CELLS
            eventBinaryAll = (this.transient_mat(i_cell,:).*(this.run_state==1 & valid2)'==1);
            eventBinaryA = (this.transient_mat(i_cell,:).*(this.run_state==1 & this.trialType==2 & valid1)'==1);
            eventBinaryB = (this.transient_mat(i_cell,:).*(this.run_state==1 & this.trialType==3 & valid1)'==1);
                   
            eventPosA = this.position_norm(eventBinaryA);
            eventPosB = this.position_norm(eventBinaryB);

            eventA = histcn(eventPosA,edgesPosition);
            eventA = jmm_smooth_1d_cor_circ(eventA,SM);
            eventA = eventA(1:N_BINS);
            eventsA(i_cell,:) = eventA;

            eventB = histcn(eventPosB,edgesPosition);
            eventB = jmm_smooth_1d_cor_circ(eventB,SM);
            eventB = eventB(1:N_BINS);
            eventsB(i_cell,:) = eventB;

            smoothedEventRate = jmm_smooth_1d_cor(double(eventBinaryAll), 2*SM2)/timeBin;
            coarseRate(i_cell,:) = interp1(this.time,smoothedEventRate,timeCenters);
            
            
            eventBinaryA = (this.transient_mat(i_cell,:).*(this.run_state==1 & this.trialType==2)'==1);
            eventBinaryB = (this.transient_mat(i_cell,:).*(this.run_state==1 & this.trialType==3)'==1);
                   
            eventPosA = this.position_norm(eventBinaryA);
            eventPosB = this.position_norm(eventBinaryB);

            eventA = histcn(eventPosA,edgesPosition);
            eventA = jmm_smooth_1d_cor_circ(eventA,SM);
            eventA = eventA(1:N_BINS);
            eventsA_full(i_cell,:) = eventA;

            eventB = histcn(eventPosB,edgesPosition);
            eventB = jmm_smooth_1d_cor_circ(eventB,SM);
            eventB = eventB(1:N_BINS);
            eventsB_full(i_cell,:) = eventB;
            
        end

        rateA = bsxfun(@rdivide, eventsA, occA);
        rateB = bsxfun(@rdivide, eventsB, occB);
        rateTogether = [rateA rateB];
        
        rateA = bsxfun(@rdivide, eventsA_full, occA_full);
        rateB = bsxfun(@rdivide, eventsB_full, occB_full);
        rateTogether_full = [rateA rateB];
        %
        TEMPLATE{i_ses} = rateTogether;
        TEMPLATE_FULL{i_ses} = rateTogether_full;
        BIN_RATE{i_ses} = coarseRate;
                        
        downsampledPosition = interp1(this.time, this.position_norm, timeCenters,'nearest');
        downsampledRunState = interp1(this.time, 1*(this.run_state==1 & this.trialType>0), timeCenters,'nearest');
        downsampledPosition(downsampledRunState==0) = NaN;
        downsampledTrialType = interp1(this.time, this.trialType, timeCenters,'nearest');
        downsampledTrialType(downsampledRunState==0) = NaN;

        TRUE_POS{i_ses} = downsampledPosition;
        TRUE_RUN{i_ses} = downsampledRunState;
        TRUE_TYPE{i_ses} = downsampledTrialType;
        
       
    end
    ALL_TEMPLATE(i_mouse,:) = TEMPLATE;
    ALL_TEMPLATE_FULL(i_mouse,:) = TEMPLATE_FULL;
    ALL_BIN_RATE(i_mouse,:) = BIN_RATE;
    ALL_TRUE_POS(i_mouse,:) = TRUE_POS;
    ALL_TRUE_RUN(i_mouse,:) = TRUE_RUN;
    ALL_TRUE_TYPE(i_mouse,:) = TRUE_TYPE;
end


%%
N_resam = 10;
N_included = 20:20:300;
DECODING = NaN(totalMice, size(ALL_TEMPLATE,2), length(N_included), N_resam);
ERROR = NaN(totalMice, size(ALL_TEMPLATE,2), length(N_included), N_resam);
ALL_PREDICTED_POS_1 = cell(totalMice, size(ALL_TEMPLATE,2), length(N_included), N_resam);
fprintf('\n');
for i_mouse = 1:totalMice
    fprintf('\nMouse %d: ',i_mouse);
    for i_ses = 1:size(ALL_TEMPLATE,2)
        fprintf('%d...',i_ses);
        full_template = ALL_TEMPLATE{i_mouse, i_ses};
        full_data = ALL_BIN_RATE{i_mouse, i_ses};
        if(isempty(full_template))
            continue;
        end 
        this_true_pos = ALL_TRUE_POS{i_mouse, i_ses};
        this_true_type = ALL_TRUE_TYPE{i_mouse, i_ses};
        this_true_run = ALL_TRUE_RUN{i_mouse, i_ses};
        for i_include = 1:length(N_included)
            valid = find(sum(full_template,2)>0);
            nValid = length(valid);
            if(N_included(i_include)>nValid)
                continue;
            end
            cells_to_include = N_included(i_include);

            for i_resam = 1:N_resam
                r = randperm(nValid);
                include = valid(r(1:cells_to_include));
                this_template = full_template(include,:);
                this_data = full_data(include,:);
                
                if(isempty(this_data))
                    continue;
                end
                
                tp2 = 200*(this_true_pos+this_true_type-2);
                tp2(this_true_run==0) = NaN;

                centersPosition2 = 200*[centersPosition centersPosition+1];
                nValid2 = sum(sum(this_template,2)>0);
                c = zscore(this_template)'*zscore(this_data)/(nValid2-1);
                [~, maxBin] = max(c);
                confidence = max(c) - quantile(c,0.85);

                pp2 = centersPosition2(maxBin);
                pp2(this_true_run==0) = NaN;
                pp2(confidence==0) = NaN;
                
                AA = sum(tp2<200 & pp2<200);
                AB = sum(tp2<200 & pp2>=200);
                BA = sum(tp2>=200 & pp2<200);
                BB = sum(tp2>=200 & pp2>=200);

                DECODING(i_mouse, i_ses, i_include, i_resam) = (AA+BB)/(AA+AB+BA+BB);

                error = 100*abs(angDiff(2*pi*pp2/200,2*pi*tp2/200))/pi;

                ERROR(i_mouse, i_ses, i_include, i_resam) = nanmedian(error);
            
                ALL_PREDICTED_POS_1{i_mouse, i_ses, i_include, i_resam} = pp2;
            end
        end
    end
end
fprintf('\n');           

           
ERROR(3,2,:,:) = NaN;
DECODING(3,2,:,:) = NaN;
ERROR_1 = squeeze(nanmean(ERROR,4));
DECODING_1 = squeeze(nanmean(DECODING,4));




%%
clf;
subpanels = 'A     B  C';
NCOLS = 3;
NROWS = 4;

% WIDTH = 6.722*2.54;
% HEIGHT = 3.693*2.54;

WIDTH = 5.3*2.54;
HEIGHT = 6*2.54;
set(0,'defaultaxesfontsize',14)
close all;
[panel_pos, fig, all_ax, HH] = panelFigureSetup2( NCOLS, NROWS, subpanels,WIDTH,HEIGHT, 0.5, 1.5);

HH(1).FontSize = 20;
HH(7).FontSize = 20;
HH(10).FontSize = 20;
% HH(3).FontSize = 24;
% HH(9).FontSize = 24;
% HH(11).FontSize = 24;
% HH(13).FontSize = 24;
% HH(15).FontSize = 24;
% HH(17).FontSize = 24;
% HH(19).FontSize = 24;

HH(1).Position = HH(1).Position+[-0.02 0.04 0];
HH(7).Position = HH(7).Position+[-0.02 0.01 0];
HH(10).Position = HH(10).Position+[-0.02 -0.01 0];
% HH(9).Position = HH(9).Position+[-0.03 0.01 0];
% HH(13).Position = HH(13).Position+[-0.03 -0.03 0];
% HH(17).Position = HH(17).Position+[-0.03 0.01 0];

% HH(3).Position = HH(3).Position+[0.03 0.01 0];
% HH(11).Position = HH(11).Position+[0.03 -0.01 0];
% HH(19).Position = HH(19).Position+[0.03 -0.02 0];
% HH(1).Position = HH(1).Position+[-0.03 0.01 0];
colormap parula;
SCALE = 1.18;

SCALE = 1;
W = 1;
SHIFT_D = 0.1;
SHIFT_U = 0.2;
mDECODING = squeeze(nanmean(DECODING,4));
mERROR = squeeze(nanmean(ERROR,4));
xxx = 1:7;
% XL = [min(xxx)-0.5 max(xxx)+0.5];
XL = [min(xxx)-0.1 max(xxx)+0.1];
YL = [0.53 0.975];
YL2 = [2 18];
YLAB = 'Identification\nScore';
YLAB2 = 'Decoding Error';
% colors = [linspace(0,1,8)' zeros(8,1) linspace(1,0,8)'];
colors = [linspace(1,0,8)' linspace(0,0.9,8)' linspace(1,0,8)'];
% colormap(colors);


MS1 = 5;
MS2 = 10;
LW = 2;

n = 2;
P = panel_pos(7,:);
P(1) = P(1)+P(3)*0.1;
P(2) = P(2)+P(4)*SHIFT_U;
P(3) = P(3)*W;
P(3:4) = P(3:4)*SCALE;
axes('position', P);

c1 = 0.75*[1 1 1];

% plot(xxx,squeeze(mDECODING(:,xxx,n))','.-','Color',(0.95*[1 1 1]+colors(n,:))/2,'Markersize',MS1);
plot(xxx,squeeze(mDECODING(:,xxx,n))','-','Color',c1,'Markersize',MS1);
hold on;
p = seplot(xxx,squeeze(mDECODING(:,xxx,n))',colors(n,:),1);
p(2).LineStyle = '--';
p(3).LineStyle = '--';
% plot(xxx,squeeze(nanmean(mDECODING(:,xxx,n))),'.-','Linewidth',LW,'Markersize',MS2,'Color',colors(n,:));
% xlabel('Session #');
ylabel(sprintf('Identification\nScore'));
% title(sprintf('%d Neurons',N_included(n)));
xlim(XL);
ylim(YL);
text(mean(xlim),max(ylim),sprintf('%d Neurons',N_included(n)),'HorizontalAlign','Center','VerticalAlign','Top','FontSize',12);
% axis square;
xlabel('Day');
set(gca,'ytick',0.6:0.1:1);
set(gca,'xtick',1:7);

n = 2;
P = panel_pos(10,:);
P(1) = P(1)+P(3)*0.1;
P(2) = P(2)-P(4)*SHIFT_D;
P(3) = P(3)*W;
P(3:4) = P(3:4)*SCALE;
axes('position', P);

plot(xxx,squeeze(mERROR(:,xxx,n))','-','Color',c1,'Markersize',MS1);
hold on;
p = seplot(xxx,squeeze(mERROR(:,xxx,n))',colors(n,:),1);
p(2).LineStyle = '--';
p(3).LineStyle = '--';

% plot(xxx,squeeze(mERROR(:,xxx,n))','.-','Color',(0.95*[1 1 1]+colors(n,:))/2,'Markersize',MS1);
% hold on;
% plot(xxx,squeeze(nanmean(mERROR(:,xxx,n))),'.-','Linewidth',LW,'Markersize',MS2,'Color',colors(n,:));
% xlabel('Session #');
% ylabel(YLAB2);
% title(sprintf('%d Neurons',N_included(n)));
ylabel(sprintf('Decoding\nError (cm)'));
xlim(XL);
ylim(YL2);
xlabel('Day');
text(mean(xlim),max(ylim),sprintf('%d Neurons',N_included(n)),'HorizontalAlign','Center','VerticalAlign','Top','FontSize',12);
set(gca,'xtick',1:7);
% axis square;

% P = panel_pos(16,:);
% P(1) = P(1)+P(3)*0.3;
% P(3:4) = P(3:4)*SCALE;
% axes('position', P);
% 
% n = 5;
% plot(xxx,squeeze(mDECODING(:,xxx,n))','.-','Color',(0.95*[1 1 1]+colors(n,:))/2,'Markersize',MS1);
% hold on;
% plot(xxx,squeeze(nanmean(mDECODING(:,xxx,n))),'.-','Linewidth',LW,'Markersize',MS1,'Color',colors(n,:));
% xlabel('Session #');
% ylabel('Decoding Score');
% title(sprintf('%d Neurons',N_included(n)));
% xlim(XL);
% ylim(YL);
% axis square;


P = panel_pos(8,:);
P(1) = P(1)+P(3)*0.2;
P(2) = P(2)+P(4)*SHIFT_U;
P(3) = P(3)*W;
P(3:4) = P(3:4)*SCALE;
axes('position', P);

n = 8;
plot(xxx,squeeze(mDECODING(:,xxx,n))','-','Color',c1,'Markersize',MS1);
hold on;
p = seplot(xxx,squeeze(mDECODING(:,xxx,n))',colors(n,:),1);
p(2).LineStyle = '--';
p(3).LineStyle = '--';

% plot(xxx,squeeze(mDECODING(:,xxx,n))','.-','Color',(0.95*[1 1 1]+colors(n,:))/2,'Markersize',MS1);
% hold on;
% plot(xxx,squeeze(nanmean(mDECODING(:,xxx,n))),'.-','Linewidth',LW,'Markersize',MS2,'Color',colors(n,:));
xlabel('Day');
% ylabel('                                   Identification Score');
% title(sprintf('%d Neurons',N_included(n)));
xlim(XL);
ylim(YL);
text(mean(xlim),max(ylim),sprintf('%d Neurons',N_included(n)),'HorizontalAlign','Center','VerticalAlign','Top','FontSize',12);
% axis square;
% set(gca,'yticklabel','');
set(gca,'ytick',0.6:0.1:1);
set(gca,'xtick',1:7);

n = 8;
P = panel_pos(11,:);
P(1) = P(1)+P(3)*0.2;
P(2) = P(2)-P(4)*SHIFT_D;
P(3) = P(3)*W;
P(3:4) = P(3:4)*SCALE;
axes('position', P);

plot(xxx,squeeze(mERROR(:,xxx,n))','-','Color',c1,'Markersize',MS1);
hold on;
p = seplot(xxx,squeeze(mERROR(:,xxx,n))',colors(n,:),1);
p(2).LineStyle = '--';
p(3).LineStyle = '--';

% plot(xxx,squeeze(mERROR(:,xxx,n))','.-','Color',(0.95*[1 1 1]+colors(n,:))/2,'Markersize',MS1);
% hold on;
% plot(xxx,squeeze(nanmean(mERROR(:,xxx,n))),'.-','Linewidth',LW,'Markersize',MS2,'Color',colors(n,:));
xlabel('Day');
% ylabel('                                  Decoding Error');
% title(sprintf('%d Neurons',N_included(n)));
xlim(XL);
ylim(YL2);
text(mean(xlim),max(ylim),sprintf('%d Neurons',N_included(n)),'HorizontalAlign','Center','VerticalAlign','Top','FontSize',12);
set(gca,'xtick',1:7);

% axis square;
% set(gca,'yticklabel','');

% subplot(2,3,5);
% n = 8;
% plot(xxx,squeeze(mDECODING(:,xxx,n))','.-','Color',(0.95*[1 1 1]+colors(n,:))/2);
% hold on;
% plot(xxx,squeeze(nanmean(mDECODING(:,xxx,n))),'.-','Linewidth',4,'Markersize',40,'Color',colors(n,:));
% xlabel('Session #');
% ylabel('Decoding Score');
% title(sprintf('%d Neurons',N_included(n)));
% xlim(XL);
% ylim(YL);
% axis square;

P = panel_pos(9,:);
P(1) = P(1)+P(3)*0.3;
P(2) = P(2)+P(4)*SHIFT_U;
P(3) = P(3)*W;
P(3:4) = P(3:4)*SCALE;
axes('position', P);

% yyy = linspace(0.575,0.75,8);
yyy = [0.575 0.63 0.645 0.68 0.7 0.715 0.7325 0.76];
for i_inc = 1:8
    plot(xxx,squeeze(nanmean(mDECODING(:,xxx,i_inc))),'.-','Linewidth',LW,'Markersize',MS2,'Color',colors(i_inc,:));
%     text(0.5,squeeze(nanmean(mDECODING(:,1,i_inc))),sprintf('%d',N_included(i_inc)),'HorizontalAlign','Right','Color',colors(i_inc,:),'Fontsize',16);
    if(isint(log2(i_inc)))
        text(0.7,yyy(i_inc),sprintf('%d',N_included(i_inc)),'HorizontalAlign','Right','Color',colors(i_inc,:),'Fontsize',10);
    end
    hold on;
end
xlabel('Day');
% axis square;
xlim([-0.95 7.1]);
ylim(YL);
set(gca,'xtick',xxx);
text(0.6, 0.85,'N','Color','k','HorizontalAlign','Right','Fontsize',15);
% ylabel(YLAB);
% colorbar;
xxx2 = repmat(xxx,[size(mDECODING,3)-2 1]);
yyy = squeeze(nanmean(mDECODING(:,xxx,1:end-2),1))';
g = repmat((1:13)',[1 length(xxx)]);
valid = ~isnan(yyy);
[a3, b_3, stats3] = anovan(yyy(valid), {g(valid) xxx2(valid)}, 'continuous', 2, 'display', 'off', 'model', 'full', 'varnames', char('Group', 'X-factor'));
% set(gca,'yticklabel','');
set(gca,'ytick',0.6:0.1:1);


P = panel_pos(12,:);
P(1) = P(1)+P(3)*0.3;
P(2) = P(2)-P(4)*SHIFT_D;
P(3) = P(3)*W;
P(3:4) = P(3:4)*SCALE;
axes('position', P);

% colors = [linspace(0,1,8)' zeros(8,1) linspace(1,0,8)'];
colors = [linspace(1,0,8)' linspace(0,0.9,8)' linspace(1,0,8)'];
% yyy = linspace(0.575,0.75,8);
yyy = [13 9 7.5 6.75 6.2 5.6 5.1 5];
for i_inc = 1:8
    plot(xxx,squeeze(nanmean(mERROR(:,xxx,i_inc))),'.-','Linewidth',LW,'Markersize',MS2,'Color',colors(i_inc,:));
%     text(0.5,squeeze(nanmean(mDECODING(:,1,i_inc))),sprintf('%d',N_included(i_inc)),'HorizontalAlign','Right','Color',colors(i_inc,:),'Fontsize',16);
    if(isint(log2(i_inc)))
        text(0.7,yyy(i_inc),sprintf('%d',N_included(i_inc)),'HorizontalAlign','Right','Color',colors(i_inc,:),'Fontsize',10);
    end
    hold on;
end
xlabel('Day');
% axis square;
xlim([-0.95 7.1]);
ylim(YL2);
set(gca,'xtick',xxx);
text(0.63, 16,'N','Color','k','HorizontalAlign','Right','Fontsize',15);
% ylabel(YLAB2);
% set(gca,'yticklabel','');

xxx2 = repmat(xxx,[size(mERROR,3)-2 1]);
yyy = squeeze(nanmean(mERROR(:,xxx,1:end-2),1))';
g = repmat((1:13)',[1 length(xxx)]);
valid = ~isnan(yyy);
[a4, b_4, stats4] = anovan(yyy(valid), {g(valid) xxx2(valid)}, 'continuous', 2, 'display', 'off', 'model', 'full', 'varnames', char('Group', 'X-factor'));



c1 = 0.75*[1 1 1];
% I_MOUSE = 1;
% REF = 3;
% I_SES = REF+[0 1 2 3];
% panels = [1 2 5 6];

% S1 = 0.24;
% S2 = 0.77;
% for i_ses = 1:length(panels)
%     P = panel_pos(panels(i_ses),:);
%     P(2) = P(2)+P(4)*0.3;
%     P(3:4) = P(3:4)*SCALE;
%     axes('position', P);
%     this = (ALL_TRUE_POS_0{I_MOUSE,I_SES(i_ses)}+(ALL_TRUE_TYPE_0{I_MOUSE,I_SES(i_ses)}-2))*200;
%     this(ALL_TRUE_RUN_0{I_MOUSE,I_SES(i_ses)}==0) = NaN;
%     this2 = ALL_PREDICTED_POS_0{I_MOUSE,REF,I_SES(i_ses)};
%     t = timeBin*(1:length(this));
%     plot(t,this,'.','Markersize',3,'Color',c1);
%     hold on;
%     plot(t,this2,'r.','Markersize',3);
%     plot(xlim,200*[1 1],'k');
% %     title(sprintf('Train: D%d, Test: D%d',REF,I_SES(i_ses)));
%     title(sprintf('A: %.02f  E: %.01f',DECODING_L(I_MOUSE,REF,I_SES(i_ses)),ERROR_L(I_MOUSE,REF,I_SES(i_ses))));
%     patch([min(xlim) min(xlim)+range(xlim)*S1 min(xlim)+range(xlim)*S1 min(xlim)],[min(ylim)+range(ylim)*S2 min(ylim)+range(ylim)*S2 max(ylim) max(ylim)],'w');
%     patch([min(xlim)+range(xlim)*(1-S1) max(xlim) max(xlim) min(xlim)+range(xlim)*(1-S1)],[min(ylim)+range(ylim)*S2 min(ylim)+range(ylim)*S2 max(ylim) max(ylim)],'w');
%     text(min(xlim),max(ylim),sprintf('D%d',REF),'VerticalAlign','Top','Fontsize',16);
%     text(max(xlim),max(ylim),sprintf('D%d',I_SES(i_ses)),'VerticalAlign','Top','HorizontalAlign','Right','Fontsize',16);
%     plot((min(xlim)+range(xlim)*S1)*[1 1],[min(ylim)+range(ylim)*S2 max(ylim)],'k','Linewidth',1);
%     plot([min(xlim) min(xlim)+range(xlim)*S1],(min(ylim)+range(ylim)*S2)*[1 1],'k','Linewidth',1);
%     plot((min(xlim)+range(xlim)*(1-S1))*[1 1],[min(ylim)+range(ylim)*S2 max(ylim)],'k','Linewidth',1);
%     plot([min(xlim)+range(xlim)*(1-S1) max(xlim)],(min(ylim)+range(ylim)*S2)*[1 1],'k','Linewidth',1);
%     set(gca,'ytick',[0 100 200 300 400]);
%     set(gca,'yticklabel',{'0','100','200','100','200'});
% end


W = 1.6;
H = 0.8;
MS = 5;
SHIFT_U = 0.1;
I_MOUSE = 3;
SES1 = 1;
SES2 = 7;
S1 = 0.17;
S2 = 0.69;
REP = 2;
P = panel_pos(1,:);
P(1) = P(1)+P(3)*0.1;
P(2) = P(2)+P(4)*(0.3+SHIFT_U);
P(3) = P(3)*W;
P(4) = P(4)*H;
P(3:4) = P(3:4)*SCALE;
axes('position', P);
I_SES = SES1;
N = 2;
this = (ALL_TRUE_POS{I_MOUSE,I_SES}+(ALL_TRUE_TYPE{I_MOUSE,I_SES}-2))*200;
this(ALL_TRUE_RUN{I_MOUSE,I_SES}==0) = NaN;
this2 = ALL_PREDICTED_POS_1{I_MOUSE,I_SES,N,REP};
start = find(~isnan(this2),1,'first');
this = this(start:end);
this2 = this2(start:end);
t = timeBin*(1:length(this));
plot(t,this,'.','Color',c1,'Markersize',MS-1);
hold on;
plot(t,this2,'.','Markersize',MS,'Color',colors(N,:));
axis tight;
ylim([0 400]);
plot(xlim,200*[1 1],'k');
title(sprintf('I: %.02f  E: %.01f',DECODING(I_MOUSE,I_SES,N,REP),ERROR(I_MOUSE,I_SES,N,REP)));
patch([min(xlim) min(xlim)+range(xlim)*S1 min(xlim)+range(xlim)*S1 min(xlim)],[min(ylim)+range(ylim)*S2 min(ylim)+range(ylim)*S2 max(ylim) max(ylim)],'w');
patch([min(xlim)+range(xlim)*(0.92-S1) max(xlim) max(xlim) min(xlim)+range(xlim)*(0.92-S1)],[min(ylim)+range(ylim)*S2 min(ylim)+range(ylim)*S2 max(ylim) max(ylim)],'w');
text(min(xlim),max(ylim),sprintf('D:%d',I_SES),'VerticalAlign','Top','Fontsize',14);
text(max(xlim),max(ylim),sprintf('N:%d',N_included(N)),'VerticalAlign','Top','HorizontalAlign','Right','Fontsize',14);
plot((min(xlim)+range(xlim)*S1)*[1 1],[min(ylim)+range(ylim)*S2 max(ylim)],'k','Linewidth',1);
plot([min(xlim) min(xlim)+range(xlim)*S1],(min(ylim)+range(ylim)*S2)*[1 1],'k','Linewidth',1);
plot((min(xlim)+range(xlim)*(0.92-S1))*[1 1],[min(ylim)+range(ylim)*S2 max(ylim)],'k','Linewidth',1);
plot([min(xlim)+range(xlim)*(0.92-S1) max(xlim)],(min(ylim)+range(ylim)*S2)*[1 1],'k','Linewidth',1);   
set(gca,'ytick',[0 100 200 300 400]);
set(gca,'yticklabel',{'0','100','200','100','200'});
% ylabel('Position (cm)');
set(gca,'xtick',[]);
P2 = [P(1)+P(3)-P(3)*0.04 P(2) P(3)*0.1 P(4)];
axes('position', P2);
plot([0 0],[0 0.5],'Color',[0 0 0.85],'Linewidth',5);
text(0.9,0.2,'A','Fontsize',12,'Rotation',90);
hold on;
plot([0 0],[0.5 1],'Color',[1 0 0],'Linewidth',5);
text(0.9,0.7,'B','Fontsize',12,'Rotation',90);
axis off;

P = panel_pos(1,:);
P(1) = P(1)+P(3)*0.1;
P(2) = P(2)-P(4)*0.1;
P(3) = P(3)*W;
P(4) = P(4)*H;
P(3:4) = P(3:4)*SCALE;
axes('position', P);
% ylim([0 400]);
plot([15 65],[25 25],'k','Linewidth',2);
text(40,5,'50s','HorizontalAlign','Center','Fontsize',14);
xlim([0.25 300.5]);
ylim([-50 85]);
axis off;


P = panel_pos(4,:);
P(1) = P(1)+P(3)*0.1;
P(2) = P(2)+P(4)*0.3;
P(3) = P(3)*W;
P(4) = P(4)*H;
P(3:4) = P(3:4)*SCALE;
axes('position', P);
I_SES = SES2;
N = 2;
this = (ALL_TRUE_POS{I_MOUSE,I_SES}+(ALL_TRUE_TYPE{I_MOUSE,I_SES}-2))*200;
this(ALL_TRUE_RUN{I_MOUSE,I_SES}==0) = NaN;
this2 = ALL_PREDICTED_POS_1{I_MOUSE,I_SES,N,REP};
start = find(~isnan(this2),1,'first');
this = this(start:end);
this2 = this2(start:end);
t = timeBin*(1:length(this));
plot(t,this,'.','Color',c1,'Markersize',MS-1);
hold on;
plot(t,this2,'.','Markersize',MS,'Color',colors(N,:));
axis tight;
ylim([0 400]);
plot(xlim,200*[1 1],'k');
title(sprintf('I: %.02f  E: %.01f',DECODING(I_MOUSE,I_SES,N,REP),ERROR(I_MOUSE,I_SES,N,REP)));
patch([min(xlim) min(xlim)+range(xlim)*S1 min(xlim)+range(xlim)*S1 min(xlim)],[min(ylim)+range(ylim)*S2 min(ylim)+range(ylim)*S2 max(ylim) max(ylim)],'w');
patch([min(xlim)+range(xlim)*(0.92-S1) max(xlim) max(xlim) min(xlim)+range(xlim)*(0.92-S1)],[min(ylim)+range(ylim)*S2 min(ylim)+range(ylim)*S2 max(ylim) max(ylim)],'w');
text(min(xlim),max(ylim),sprintf('D:%d',I_SES),'VerticalAlign','Top','Fontsize',14);
text(max(xlim),max(ylim),sprintf('N:%d',N_included(N)),'VerticalAlign','Top','HorizontalAlign','Right','Fontsize',14);
plot((min(xlim)+range(xlim)*S1)*[1 1],[min(ylim)+range(ylim)*S2 max(ylim)],'k','Linewidth',1);
plot([min(xlim) min(xlim)+range(xlim)*S1],(min(ylim)+range(ylim)*S2)*[1 1],'k','Linewidth',1);
plot((min(xlim)+range(xlim)*(0.92-S1))*[1 1],[min(ylim)+range(ylim)*S2 max(ylim)],'k','Linewidth',1);
plot([min(xlim)+range(xlim)*(0.92-S1) max(xlim)],(min(ylim)+range(ylim)*S2)*[1 1],'k','Linewidth',1);   
set(gca,'ytick',[0 100 200 300 400]);
set(gca,'yticklabel',{'0','100','200','100','200'});
ylabel('                                 Position (cm)');
set(gca,'xtick',[]);
P2 = [P(1)+P(3)-P(3)*0.04 P(2) P(3)*0.1 P(4)];
axes('position', P2);
plot([0 0],[0 0.5],'Color',[0 0 0.85],'Linewidth',5);
text(0.9,0.2,'A','Fontsize',12,'Rotation',90);
hold on;
plot([0 0],[0.5 1],'Color',[1 0 0],'Linewidth',5);
text(0.9,0.7,'B','Fontsize',12,'Rotation',90);
axis off;


P = panel_pos(3,:);
P(1) = P(1)-P(3)*0.3;
P(2) = P(2)+P(4)*(0.3+SHIFT_U);
P(3) = P(3)*W;
P(4) = P(4)*H;
P(3:4) = P(3:4)*SCALE;
axes('position', P);
I_SES = SES1;
N = 8;
this = (ALL_TRUE_POS{I_MOUSE,I_SES}+(ALL_TRUE_TYPE{I_MOUSE,I_SES}-2))*200;
this(ALL_TRUE_RUN{I_MOUSE,I_SES}==0) = NaN;
this2 = ALL_PREDICTED_POS_1{I_MOUSE,I_SES,N,REP};
start = find(~isnan(this2),1,'first');
this = this(start:end);
this2 = this2(start:end);
t = timeBin*(1:length(this));
plot(t,this,'.','Color',c1,'Markersize',MS-1);
hold on;
plot(t,this2,'.','Markersize',MS,'Color',colors(N,:));
axis tight;
ylim([0 400]);
plot(xlim,200*[1 1],'k');
title(sprintf('I: %.02f  E: %.01f',DECODING(I_MOUSE,I_SES,N,REP),ERROR(I_MOUSE,I_SES,N,REP)));
patch([min(xlim) min(xlim)+range(xlim)*S1 min(xlim)+range(xlim)*S1 min(xlim)],[min(ylim)+range(ylim)*S2 min(ylim)+range(ylim)*S2 max(ylim) max(ylim)],'w');
patch([min(xlim)+range(xlim)*(0.88-S1) max(xlim) max(xlim) min(xlim)+range(xlim)*(0.88-S1)],[min(ylim)+range(ylim)*S2 min(ylim)+range(ylim)*S2 max(ylim) max(ylim)],'w');
text(min(xlim),max(ylim),sprintf('D:%d',I_SES),'VerticalAlign','Top','Fontsize',14);
text(max(xlim),max(ylim),sprintf('N:%d',N_included(N)),'VerticalAlign','Top','HorizontalAlign','Right','Fontsize',14);
plot((min(xlim)+range(xlim)*S1)*[1 1],[min(ylim)+range(ylim)*S2 max(ylim)],'k','Linewidth',1);
plot([min(xlim) min(xlim)+range(xlim)*S1],(min(ylim)+range(ylim)*S2)*[1 1],'k','Linewidth',1);
plot((min(xlim)+range(xlim)*(0.88-S1))*[1 1],[min(ylim)+range(ylim)*S2 max(ylim)],'k','Linewidth',1);
plot([min(xlim)+range(xlim)*(0.88-S1) max(xlim)],(min(ylim)+range(ylim)*S2)*[1 1],'k','Linewidth',1);   
set(gca,'ytick',[0 100 200 300 400]);
% set(gca,'yticklabel',{'0','100','200','100','200'});
set(gca,'yticklabel','');
set(gca,'xtick',[]);
P2 = [P(1)+P(3)-P(3)*0.04 P(2) P(3)*0.1 P(4)];
axes('position', P2);
plot([0 0],[0 0.5],'Color',[0 0 0.85],'Linewidth',5);
text(0.9,0.2,'A','Fontsize',12,'Rotation',90);
hold on;
plot([0 0],[0.5 1],'Color',[1 0 0],'Linewidth',5);
text(0.9,0.7,'B','Fontsize',12,'Rotation',90);
axis off;


P = panel_pos(6,:);
P(1) = P(1)-P(3)*0.3;
P(2) = P(2)+P(4)*0.3;
P(3) = P(3)*W;
P(4) = P(4)*H;
P(3:4) = P(3:4)*SCALE;
axes('position', P);
I_SES = SES2;
N = 8;
this = (ALL_TRUE_POS{I_MOUSE,I_SES}+(ALL_TRUE_TYPE{I_MOUSE,I_SES}-2))*200;
this(ALL_TRUE_RUN{I_MOUSE,I_SES}==0) = NaN;
this2 = ALL_PREDICTED_POS_1{I_MOUSE,I_SES,N,REP};
start = find(~isnan(this2),1,'first');
this = this(start:end);
this2 = this2(start:end);
t = timeBin*(1:length(this));
plot(t,this,'.','Color',c1,'Markersize',MS-1);
hold on;
plot(t,this2,'.','Markersize',MS,'Color',colors(N,:));
axis tight;
ylim([0 400]);
plot(xlim,200*[1 1],'k');
title(sprintf('I: %.02f  E: %.01f',DECODING(I_MOUSE,I_SES,N,REP),ERROR(I_MOUSE,I_SES,N,REP)));
patch([min(xlim) min(xlim)+range(xlim)*S1 min(xlim)+range(xlim)*S1 min(xlim)],[min(ylim)+range(ylim)*S2 min(ylim)+range(ylim)*S2 max(ylim) max(ylim)],'w');
patch([min(xlim)+range(xlim)*(0.88-S1) max(xlim) max(xlim) min(xlim)+range(xlim)*(0.88-S1)],[min(ylim)+range(ylim)*S2 min(ylim)+range(ylim)*S2 max(ylim) max(ylim)],'w');
text(min(xlim),max(ylim),sprintf('D:%d',I_SES),'VerticalAlign','Top','Fontsize',14);
text(max(xlim),max(ylim),sprintf('N:%d',N_included(N)),'VerticalAlign','Top','HorizontalAlign','Right','Fontsize',14);
plot((min(xlim)+range(xlim)*S1)*[1 1],[min(ylim)+range(ylim)*S2 max(ylim)],'k','Linewidth',1);
plot([min(xlim) min(xlim)+range(xlim)*S1],(min(ylim)+range(ylim)*S2)*[1 1],'k','Linewidth',1);
plot((min(xlim)+range(xlim)*(0.88-S1))*[1 1],[min(ylim)+range(ylim)*S2 max(ylim)],'k','Linewidth',1);
plot([min(xlim)+range(xlim)*(0.88-S1) max(xlim)],(min(ylim)+range(ylim)*S2)*[1 1],'k','Linewidth',1);   
set(gca,'ytick',[0 100 200 300 400]);
% set(gca,'yticklabel',{'0','100','200','100','200'});
set(gca,'yticklabel','');
set(gca,'xtick',[]);
P2 = [P(1)+P(3)-P(3)*0.04 P(2) P(3)*0.1 P(4)];
axes('position', P2);
plot([0 0],[0 0.5],'Color',[0 0 0.85],'Linewidth',5);
text(0.9,0.2,'A','Fontsize',12,'Rotation',90);
hold on;
plot([0 0],[0.5 1],'Color',[1 0 0],'Linewidth',5);
text(0.9,0.7,'B','Fontsize',12,'Rotation',90);
axis off;

%%
figure_path_primary = '.';
capstr = '';
fig = gcf;
savefig(fullfile(figure_path_primary, 'RZ_Rebut_J3_v2'), fig, false , capstr, 600);
print(fig, fullfile(figure_path_primary, 'RZ_Rebut_J3_v2'), '-painters', '-dsvg');

