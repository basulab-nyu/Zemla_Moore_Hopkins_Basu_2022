%% PV Decoding Figure, but replacing trial type with slow/fast running

cleer;
colormap parula;

% base = '\\research-cifs.nyumc.org\Research\basulab\basulabspace\RZ\data_for_Jason\matching_binary_events_all_lap_learn';
base = 'F:\MATLAB\D_Drive\RZ_Data\matching_binary_events_all_lap_learn';
% base = '\\research-cifs.nyumc.org\Research\basulab\basulabspace\RZ\data_for_Jason\matching_binary_events_all_lap_recall';
mice = dir(base);
mice = arrayfun(@(x) x.name, mice(3:end-1), 'UniformOutput', false);

i_mouse = 6;

totalMice = 6;
maxSessions = 9;

PPP_POS_SPEED = NaN(totalMice,maxSessions);
DDD_POS_SPEED = NaN(totalMice,maxSessions);
EEE_POS_SPEED = NaN(totalMice,maxSessions);
CCC_POS_SPEED = NaN(totalMice,maxSessions);
NNN_POS_SPEED = NaN(totalMice,maxSessions);
EE_POS_SPEED = NaN(80,totalMice,maxSessions);
CC_POS_SPEED = NaN(80,totalMice,maxSessions);


speedEdges = [2 12.2245 31.2];
for i_mouse = 1:totalMice
    load(fullfile(base,mice{i_mouse},'matched_transients_all_ses.mat'),'matching_ROI_bin_transient_lap_data');

    %%
    these = matching_ROI_bin_transient_lap_data;

    PERFORMANCE = NaN(1,maxSessions);
    DECODING = NaN(1,maxSessions);
    ERROR = NaN(1,maxSessions);
    CONFUSION = NaN(1,maxSessions);
    DECODING_SPEED = NaN(1,maxSessions);
    ERROR_SPEED = NaN(1,maxSessions);
    CONFUSION_SPEED = NaN(1,maxSessions);
    
    for i_ses = 1:length(these)
    %     i_ses = 6;

        this = these{i_ses};

        N_BINS = 40;
        edgesPosition = linspace(0,1,N_BINS+1);
        centersPosition = center(edgesPosition);
        edgesSpeed = linspace(-5,30,N_BINS+1);
        centersSpeed = center(edgesSpeed);
        
        SM = 1;
        DT = nanmedian(diff(this.time));
        N_CELLS = size(this.transient_mat,1);

        [~,speedBin] = histc(this.speed,speedEdges);
        
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

        occA = DT*histcn(this.position_norm(this.run_state==1 & speedBin==1 & valid1),edgesPosition);
        occA = jmm_smooth_1d_cor_circ(occA,SM);
        occA = occA(1:N_BINS);

        occB = DT*histcn(this.position_norm(this.run_state==1 & speedBin==2 & valid1),edgesPosition);
        occB = jmm_smooth_1d_cor_circ(occB,SM);
        occB = occB(1:N_BINS);
       
        timeBin = 0.25;
        SM2 = ceil(timeBin/DT);

        timeEdges = this.time(1):timeBin:this.time(end)+timeBin;
        timeCenters = center(timeEdges);
        coarseRate = NaN(N_CELLS,length(timeEdges)-1);
        eventsA = NaN(N_CELLS,N_BINS);
        eventsB = NaN(N_CELLS,N_BINS);
        eventsA_speed = NaN(N_CELLS,N_BINS);
        eventsB_speed = NaN(N_CELLS,N_BINS);
        for i_cell = 1:N_CELLS
            eventBinaryAll = (this.transient_mat(i_cell,:).*(this.run_state==1 & valid2)'==1);
            eventBinaryA = (this.transient_mat(i_cell,:).*(this.run_state==1 & speedBin==1 & valid1)'==1);
            eventBinaryB = (this.transient_mat(i_cell,:).*(this.run_state==1 & speedBin==2 & valid1)'==1);
        %   
        %     eventRun = this.run_state(eventBinary)==1;
            eventPosA = this.position_norm(eventBinaryA);
            eventPosB = this.position_norm(eventBinaryB);

            eventSpeedA = this.speed(eventBinaryA);
            eventSpeedB = this.speed(eventBinaryB);
            
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
        goodCells = nansum(rateTogether,2)>0;
        selectRateTogether = rateTogether(goodCells,:);
        selectCoarseRate = coarseRate(goodCells,:);
        N_selectCells = sum(goodCells);

        NNN_POS_SPEED(i_mouse,i_ses) = sum(goodCells);
        instCorr = (zscore(selectRateTogether)'*zscore(selectCoarseRate))/(N_selectCells-1);
        [~, maxBin] = max(instCorr);
        
        confidence = max(instCorr) - quantile(instCorr,0.85);
        centersPosition2 = [centersPosition centersPosition+1];
        predPos = centersPosition2(maxBin);
        predPos(confidence==0) = NaN;
        
        downsampledPosition = interp1(this.time, this.position_norm, timeCenters,'nearest');
        downsampledRunState = interp1(this.time, 1*(this.run_state==1 & this.trialType>0), timeCenters,'nearest');
        downsampledPosition(downsampledRunState==0) = NaN;
        downsampledSpeedBin = interp1(this.time, speedBin, timeCenters,'nearest');
        downsampledSpeedBin(downsampledRunState==0) = NaN;

%          clf;
        % subplot(2,1,1);
        % plot(timeCenters, downsampledPosition, 'b.');
        % hold on;
        % plot(timeCenters, predPos, 'r.');
        pp2 = predPos(downsampledRunState==1)*200;
        ic2 = instCorr(:,downsampledRunState==1);
        cf2 = confidence(downsampledRunState==1);
        tt2 = downsampledSpeedBin(downsampledRunState==1);
        dp2 = (downsampledPosition(downsampledRunState==1) + (tt2-1))*200;
        t2 = timeBin*(1:length(pp2));
%         subplot(2,1,1);
%         plot(t2, dp2, 'b.');
%         hold on;
%         plot(t2, pp2, 'r.');
%         xlabel('Running Time (s)');
%         ylabel('Position (cm)');
%         axis tight;
%         plot(xlim,200*[1 1],'k');
%         legend('Actual','Decoded');
%         set(gca,'ytick',50:50:400);
%         set(gca,'yticklabel',{'50','100','150','200','50','100','150','200'});

%         subplot(2,4,5);
%         plot(dp2+randn(size(dp2)),pp2+randn(size(dp2)),'.','Color',purple,'Markersize',3);
%         hold on;
%         plot([0 200],[0 200],'k');
%         axis square;
%         xlabel('Actual Position (cm)');
%         ylabel('Decoded Position (cm)');
%         [c,p] = corrcoef(dp2,pp2,'rows','complete');
%         title(sprintf('Cells: %d\nR = %.02f', N_selectCells, c(1,2)));
%         set(gca,'xtick',100:100:400);
%         set(gca,'xticklabel',{'100','200','100','200'});
%         set(gca,'ytick',100:100:400);
%         set(gca,'yticklabel',{'100','200','100','200'});
%         xlim([0 400]);
%         ylim([0 400]);
%         plot(xlim,200*[1 1],'k');
%         plot(200*[1 1],ylim,'k');

%         subplot(2,3,4);
        b2 = linspace(0,400,2*N_BINS/2+1);
        b2 = [0 200 400.5];
        [N,~,~,L] = histcn([dp2(:) pp2(:)],b2,b2);        
        N = N';
        L(L==0) = NaN;
        N = bsxfun(@rdivide, N, sum(N));
%         imagesc(center(b2),center(b2),(N));
%         set(gca,'ydir','normal');
%         axis square;
%         xlabel('Actual Position (cm)');
%         ylabel('Decoded Position (cm)');
%         set(gca,'xtick',100:100:400);
%         set(gca,'xticklabel',{'100','200','100','200'});
%         set(gca,'ytick',100:100:400);
%         set(gca,'yticklabel',{'100','200','100','200'});
%         hold on;
%         plot(xlim,200*[1 1],'w');
%         plot(200*[1 1],ylim,'w');
%         caxis([0 1]);
%         p = get(gca,'position');        
%         colorbar;
%         set(gca,'position',p+[-0.02 0 0 0]);
%         plot(dp2+randn(size(dp2)),pp2+randn(size(dp2)),'o','Color','k','Markersize',3);
%         
        AA = sum(dp2<200 & pp2<200);
        AB = sum(dp2<200 & pp2>=200);
        BA = sum(dp2>=200 & pp2<200);
        BB = sum(dp2>=200 & pp2>=200);

        F1_1 = 2*AA/(2*AA+AB+BA);
        F1_2 = 2*BB/(2*BB+AB+BA);
        F1_3 = (F1_1+F1_2)/2;
        F1_4 = (AA+BB)/(AA+AB+BA+BB);
        correctIncorrect = this.lap_data.performance.trialCorrect;
        perf = sum(correctIncorrect==1)/sum(correctIncorrect>=0);
%         title(sprintf('Performance: %.02f\nF1: %.02f', perf, F1_3));

        PERFORMANCE(i_ses) = perf;
        DECODING(i_ses) = F1_4;
        %
        b2 = linspace(0,400,2*N_BINS+1);
%         subplot(2,3,5);
        error = 100*abs(angDiff(2*pi*pp2/200,2*pi*dp2/200))/pi;
        E = histcn(dp2(:),b2,'AccumData',error,'Fun',@nanmedian);
        E = E(1:(length(b2)-1));
        ERROR(i_ses) = nanmean(error);
        EE_POS_SPEED(:,i_mouse,i_ses) = E;
%         plot(center(b2),jmm_smooth_1d_cor_circ(E,2));
%         jAXIS;
%         xlim([0 400]);
%         ylim([0 max(ylim)]);
%         xlabel('Position (cm)');
%         ylabel('Median Decoding Error (cm)');
%         title(sprintf('Median Error: %.02f cm', nanmedian(error)));
%         axis square;
%         set(gca,'xtick',100:100:400);
%         set(gca,'xticklabel',{'100','200','100','200'});
%         hold on;
%         plot(200*[1 1],ylim,'k');

%         subplot(2,3,6);
        
        
%         100*abs(angDiff(2*pi*pp2/200,2*pi*dp2/200))/pi;
        
%         conf = abs(diff(L,[],2));
        
        conf = histcn(dp2(:),b2,'AccumData',abs(diff(L,[],2)),'Fun',@nanmean);
%         conf = histcn(dp2(:),b2,'AccumData',cf2,'Fun',@median);
        conf = conf(1:(length(b2)-1));
%         plot(center(b2),jmm_smooth_1d_cor_circ(conf,2));
%         ylim([0 max(ylim)]);
%         xlabel('Position (cm)');
%         ylabel('Mean Confusion');
%         title(sprintf('Mean Confusion: %.02f', nanmean(conf)));
%         axis square;
%         set(gca,'xtick',100:100:400);
%         set(gca,'xticklabel',{'100','200','100','200'});
%         hold on;
%         plot(200*[1 1],ylim,'k');
        CONFUSION(i_ses) = nanmean(conf);
        CC_POS_SPEED(:,i_mouse,i_ses) = conf;
        
        drawnow;
    %     pause;
    end
    PPP_POS_SPEED(i_mouse,:) = PERFORMANCE;
    DDD_POS_SPEED(i_mouse,:) = DECODING;
    EEE_POS_SPEED(i_mouse,:) = ERROR;
    CCC_POS_SPEED(i_mouse,:) = CONFUSION;
    
  
end

%%
% base = '\\research-cifs.nyumc.org\Research\basulab\basulabspace\RZ\data_for_Jason\matching_binary_events_all_lap_learn';
base = 'F:\MATLAB\D_Drive\RZ_Data\matching_binary_events_all_lap_learn';
% base = '\\research-cifs.nyumc.org\Research\basulab\basulabspace\RZ\data_for_Jason\matching_binary_events_all_lap_recall';
mice = dir(base);
mice = arrayfun(@(x) x.name, mice(3:end-1), 'UniformOutput', false);

i_mouse = 6;

totalMice = 6;
maxSessions = 9;

PPP = NaN(totalMice,maxSessions);
DDD = NaN(totalMice,maxSessions);
EEE = NaN(totalMice,maxSessions);
CCC = NaN(totalMice,maxSessions);
NNN = NaN(totalMice,maxSessions);
EE = NaN(80,totalMice,maxSessions);
CC = NaN(80,totalMice,maxSessions);

DDD_SPEED_TRIAL = NaN(totalMice,maxSessions);
EEE_SPEED_TRIAL = NaN(totalMice,maxSessions);
CCC_SPEED_TRIAL = NaN(totalMice,maxSessions);
EE_SPEED_TRIAL = NaN(80,totalMice,maxSessions);
CC_SPEED_TRIAL = NaN(80,totalMice,maxSessions);

for i_mouse = 1:totalMice
    load(fullfile(base,mice{i_mouse},'matched_transients_all_ses.mat'),'matching_ROI_bin_transient_lap_data');

    %%
    these = matching_ROI_bin_transient_lap_data;

    PERFORMANCE = NaN(1,maxSessions);
    DECODING = NaN(1,maxSessions);
    ERROR = NaN(1,maxSessions);
    CONFUSION = NaN(1,maxSessions);
    DECODING_SPEED = NaN(1,maxSessions);
    ERROR_SPEED = NaN(1,maxSessions);
    CONFUSION_SPEED = NaN(1,maxSessions);
    
    for i_ses = 1:length(these)
    %     i_ses = 6;

        this = these{i_ses};

        N_BINS = 40;
        edgesPosition = linspace(0,1,N_BINS+1);
        centersPosition = center(edgesPosition);
        edgesSpeed = linspace(-5,30,N_BINS+1);
        centersSpeed = center(edgesSpeed);
        
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

        occA_speed = DT*histcn(this.speed(this.run_state==1 & this.trialType==2 & valid1),edgesSpeed);
        occA_speed = jmm_smooth_1d_cor_circ(occA_speed,SM);
        occA_speed = occA_speed(1:N_BINS);

        occB_speed = DT*histcn(this.speed(this.run_state==1 & this.trialType==3 & valid1),edgesSpeed);
        occB_speed = jmm_smooth_1d_cor_circ(occB_speed,SM);
        occB_speed = occB_speed(1:N_BINS);
        
        timeBin = 0.25;
        SM2 = ceil(timeBin/DT);

        timeEdges = this.time(1):timeBin:this.time(end)+timeBin;
        timeCenters = center(timeEdges);
        coarseRate = NaN(N_CELLS,length(timeEdges)-1);
        eventsA = NaN(N_CELLS,N_BINS);
        eventsB = NaN(N_CELLS,N_BINS);
        eventsA_speed = NaN(N_CELLS,N_BINS);
        eventsB_speed = NaN(N_CELLS,N_BINS);
        for i_cell = 1:N_CELLS
            eventBinaryAll = (this.transient_mat(i_cell,:).*(this.run_state==1 & valid2)'==1);
            eventBinaryA = (this.transient_mat(i_cell,:).*(this.run_state==1 & this.trialType==2 & valid1)'==1);
            eventBinaryB = (this.transient_mat(i_cell,:).*(this.run_state==1 & this.trialType==3 & valid1)'==1);
        %   
        %     eventRun = this.run_state(eventBinary)==1;
            eventPosA = this.position_norm(eventBinaryA);
            eventPosB = this.position_norm(eventBinaryB);

            eventSpeedA = this.speed(eventBinaryA);
            eventSpeedB = this.speed(eventBinaryB);
            
            eventA = histcn(eventPosA,edgesPosition);
            eventA = jmm_smooth_1d_cor_circ(eventA,SM);
            eventA = eventA(1:N_BINS);
            eventsA(i_cell,:) = eventA;

            eventB = histcn(eventPosB,edgesPosition);
            eventB = jmm_smooth_1d_cor_circ(eventB,SM);
            eventB = eventB(1:N_BINS);
            eventsB(i_cell,:) = eventB;

            eventA_speed = histcn(eventSpeedA,edgesSpeed);
            eventA_speed = jmm_smooth_1d_cor(eventA_speed,SM);
            eventA_speed = eventA_speed(1:N_BINS);
            eventsA_speed(i_cell,:) = eventA_speed;

            eventB_speed = histcn(eventSpeedB,edgesSpeed);
            eventB_speed = jmm_smooth_1d_cor(eventB_speed,SM);
            eventB_speed = eventB_speed(1:N_BINS);
            eventsB_speed(i_cell,:) = eventB_speed;
            
            smoothedEventRate = jmm_smooth_1d_cor(double(eventBinaryAll), 2*SM2)/timeBin;
            coarseRate(i_cell,:) = interp1(this.time,smoothedEventRate,timeCenters);
        end

        rateA = bsxfun(@rdivide, eventsA, occA);
        rateB = bsxfun(@rdivide, eventsB, occB);
        rateTogether = [rateA rateB];
        
        rateA_speed = bsxfun(@rdivide, eventsA_speed, occA_speed);
        rateB_speed = bsxfun(@rdivide, eventsB_speed, occB_speed);
        rateTogether_speed = [rateA_speed rateB_speed];
        %
        goodCells = nansum(rateTogether,2)>0;
        selectRateTogether = rateTogether(goodCells,:);
        selectRateTogether_speed = rateTogether_speed(goodCells,:);
        selectCoarseRate = coarseRate(goodCells,:);
        N_selectCells = sum(goodCells);

        NNN(i_mouse,i_ses) = sum(goodCells);
        instCorr = (zscore(selectRateTogether)'*zscore(selectCoarseRate))/(N_selectCells-1);
        instCorr_speed = (zscore(selectRateTogether_speed)'*zscore(selectCoarseRate))/(N_selectCells-1);
        [~, maxBin] = max(instCorr);
        [~, maxBin_speed] = max(instCorr_speed);
        
        confidence = max(instCorr) - quantile(instCorr,0.85);
        confidence_speed = max(instCorr_speed) - quantile(instCorr_speed,0.85);
        centersPosition2 = [centersPosition centersPosition+1];
        centersSpeed2 = [centersSpeed centersSpeed+35];
        predPos = centersPosition2(maxBin);
        predPos(confidence==0) = NaN;
        predSpeed = centersSpeed2(maxBin_speed);
        predSpeed(confidence_speed==0) = NaN;
        
        downsampledPosition = interp1(this.time, this.position_norm, timeCenters,'nearest');
        downsampledSpeed = interp1(this.time, this.speed, timeCenters,'nearest');
        downsampledRunState = interp1(this.time, 1*(this.run_state==1 & this.trialType>0), timeCenters,'nearest');
        downsampledPosition(downsampledRunState==0) = NaN;
        downsampledSpeed(downsampledRunState==0) = NaN;
        downsampledTrialType = interp1(this.time, this.trialType, timeCenters,'nearest');
        downsampledTrialType(downsampledRunState==0) = NaN;

%         clf;
        % subplot(2,1,1);
        % plot(timeCenters, downsampledPosition, 'b.');
        % hold on;
        % plot(timeCenters, predPos, 'r.');
        pp2 = predPos(downsampledRunState==1)*200;
        ps2 = predSpeed(downsampledRunState==1);
        ic2 = instCorr(:,downsampledRunState==1);
        ics2 = instCorr_speed(:,downsampledRunState==1);
        cf2 = confidence(downsampledRunState==1);
        cfs2 = confidence_speed(downsampledRunState==1);
        tt2 = downsampledTrialType(downsampledRunState==1);
        dp2 = (downsampledPosition(downsampledRunState==1) + (tt2-2))*200;
        ds2 = (downsampledSpeed(downsampledRunState==1) + (tt2-2)*35);
        t2 = timeBin*(1:length(pp2));
%         subplot(2,2,1);
%         plot(t2, dp2, 'b.');
%         hold on;
%         plot(t2, pp2, 'r.');
%         xlabel('Running Time (s)');
%         ylabel('Position (cm)');
%         axis tight;
%         plot(xlim,200*[1 1],'k');
%         legend('Actual','Decoded');
%         set(gca,'ytick',50:50:400);
%         set(gca,'yticklabel',{'50','100','150','200','50','100','150','200'});

%         subplot(2,2,2);
%         plot(t2, ds2, 'b.');
%         hold on;
%         plot(t2, ps2, 'r.');
%         xlabel('Running Time (s)');
%         ylabel('Speed (cm/s)');
%         axis tight;
%         plot(xlim,30*[1 1],'k');
%         legend('Actual','Decoded');
%         set(gca,'ytick',[0 15 30 35 50 65]);
%         set(gca,'yticklabel',{'0','15','30','0','15','30'});
%         

%         subplot(2,6,7);
        b2 = linspace(0,400,2*N_BINS/2+1);
        b2 = [0 200 400.5];
        [N,~,~,L] = histcn([dp2(:) pp2(:)],b2,b2);        
        N = N';
        L(L==0) = NaN;
        N = bsxfun(@rdivide, N, sum(N));
%         imagesc(center(b2),center(b2),(N));
%         set(gca,'ydir','normal');
%         axis square;
%         xlabel('Actual Position (cm)');
%         ylabel('Decoded Position (cm)');
%         set(gca,'xtick',100:100:400);
%         set(gca,'xticklabel',{'100','200','100','200'});
%         set(gca,'ytick',100:100:400);
%         set(gca,'yticklabel',{'100','200','100','200'});
%         hold on;
%         plot(xlim,200*[1 1],'w');
%         plot(200*[1 1],ylim,'w');
%         caxis([0 1]);
%         p = get(gca,'position');        
%         colorbar;
%         set(gca,'position',p+[-0.02 0 0 0]);
%         plot(dp2+randn(size(dp2)),pp2+randn(size(dp2)),'o','Color','k','Markersize',3);
%         
        AA = sum(dp2<200 & pp2<200);
        AB = sum(dp2<200 & pp2>=200);
        BA = sum(dp2>=200 & pp2<200);
        BB = sum(dp2>=200 & pp2>=200);

        F1_1 = 2*AA/(2*AA+AB+BA);
        F1_2 = 2*BB/(2*BB+AB+BA);
        F1_3 = (F1_1+F1_2)/2;
        F1_4 = (AA+BB)/(AA+AB+BA+BB);
        correctIncorrect = this.lap_data.performance.trialCorrect;
        perf = sum(correctIncorrect==1)/sum(correctIncorrect>=0);
%         title(sprintf('Performance: %.02f\nF1: %.02f', perf, F1_3));

        PERFORMANCE(i_ses) = perf;
        DECODING(i_ses) = F1_4;
        
        
%         subplot(2,6,10);
        b2 = linspace(-5,65,2*N_BINS/2+1);
        b2 = [-5 30 65.5];
        [N_SPEED,~,~,L_SPEED] = histcn([ds2(:) ps2(:)],b2,b2);        
        N_SPEED = N_SPEED';
        L_SPEED(L_SPEED==0) = NaN;
        N_SPEED = bsxfun(@rdivide, N_SPEED, sum(N_SPEED));
%         imagesc(center(b2),center(b2),(N_SPEED));
%         set(gca,'ydir','normal');
%         axis square;
%         xlabel('Actual Speed (cm/s)');
%         ylabel('Decoded Speed (cm/s)');
%         set(gca,'xtick',[0 15 30 35 50 65]);
%         set(gca,'xticklabel',{'0','15','30','0','15','30'});
%         set(gca,'ytick',[0 15 30 35 50 65]);
%         set(gca,'yticklabel',{'0','15','30','0','15','30'});
%         hold on;
%         plot(xlim,30*[1 1],'w');
%         plot(30*[1 1],ylim,'w');
%         caxis([0 1]);
%         p = get(gca,'position');        
%         colorbar;
%         set(gca,'position',p+[-0.02 0 0 0]);
%         plot(ds2+randn(size(ds2)),ps2+randn(size(ds2)),'o','Color','k','Markersize',3);
        
        AA = sum(ds2<30 & ps2<30);
        AB = sum(ds2<30 & ps2>=30);
        BA = sum(ds2>=30 & ps2<30);
        BB = sum(ds2>=30 & ps2>=30);

        F1_1 = 2*AA/(2*AA+AB+BA);
        F1_2 = 2*BB/(2*BB+AB+BA);
        F1_3 = (F1_1+F1_2)/2;
        F1_4 = (AA+BB)/(AA+AB+BA+BB);
        correctIncorrect = this.lap_data.performance.trialCorrect;
        perf = sum(correctIncorrect==1)/sum(correctIncorrect>=0);
%         title(sprintf('Performance: %.02f\nF1: %.02f', perf, F1_3));
        
        DECODING_SPEED(i_ses) = F1_4;
        
        
        
        %
%         subplot(2,6,8);
        b2 = linspace(0,400,2*N_BINS+1);
        count = histc(dp2(:),b2);
        count = count(1:(length(b2)-1));
        error = 100*abs(angDiff(2*pi*pp2/200,2*pi*dp2/200))/pi;
        E = histcn(dp2(:),b2,'AccumData',error,'Fun',@nanmedian);
        E = E(1:(length(b2)-1));
        E(count==0) = NaN;
        ERROR(i_ses) = nanmedian(error);
        E(isnan(E)) = 0;
        E2 = jmm_smooth_1d_cor_circ(E,2);        
        E2(count==0) = NaN;        
        E(count==0) = NaN;
        EE(:,i_mouse,i_ses) = E;
%         plot(center(b2),E2);
%         jAXIS;
%         xlim([0 400]);
%         ylim([0 max(ylim)]);
%         xlabel('Position (cm)');
%         ylabel('Median Decoding Error (cm)');
%         title(sprintf('Median Error: %.02f cm', ERROR(i_ses)));
%         axis square;
%         set(gca,'xtick',100:100:400);
%         set(gca,'xticklabel',{'100','200','100','200'});
%         hold on;
%         plot(200*[1 1],ylim,'k');

                       
%         subplot(2,6,9);                
        error = abs(diff(L,[],2));
        conf = histcn(dp2(:),b2,'AccumData',error,'Fun',@nanmean);
%         conf = histcn(dp2(:),b2,'AccumData',cf2,'Fun',@median);
        conf = conf(1:(length(b2)-1));
        conf(isnan(conf)) = 0;
        conf2 = jmm_smooth_1d_cor_circ(conf,2);
        conf2(count==0) = NaN;
        conf(count==0) = NaN;
        CC(:,i_mouse,i_ses) = conf;        
%         plot(center(b2),conf2);
%         ylim([0 max(ylim)]);
%         xlabel('Position (cm)');
%         ylabel('Mean Confusion');
%         title(sprintf('Mean Confusion: %.02f', nanmean(error)));
%         axis square;
%         set(gca,'xtick',100:100:400);
%         set(gca,'xticklabel',{'100','200','100','200'});
%         hold on;
%         plot(200*[1 1],ylim,'k');
        CONFUSION(i_ses) = nanmean(error);
        
        
        
        
%         subplot(2,6,11);
        b2 = linspace(-5,65,2*N_BINS+1);
        count = histc(ds2(:),b2);
        count = count(1:(length(b2)-1));
        error = 35*abs(angDiff(2*pi*(ps2+5)/35,2*pi*(ds2+5)/35))/(2*pi);
        E = histcn(ds2(:),b2,'AccumData',error,'Fun',@nanmedian);
        E = E(1:(length(b2)-1));
        ERROR_SPEED(i_ses) = nanmedian(error);        
        E(isnan(E)) = 0;
        E2 = jmm_smooth_1d_cor(E,2);
        E2(count==0) = NaN;
        E(count==0) = NaN;
        EE_SPEED_TRIAL(:,i_mouse,i_ses) = E;
%         plot(center(b2),E2);
%         jAXIS;
%         xlim([-5 65]);
%         ylim([0 max(ylim)]);
%         xlabel('Speed (cm/s)');
%         ylabel('Median Decoding Error (cm)');
%         title(sprintf('Median Error: %.02f cm/s', ERROR_SPEED(i_ses)));
%         axis square;
%         set(gca,'xtick',[0 15 30 35 50 65]);
%         set(gca,'xticklabel',{'0','15','30','0','15','30'});        
%         hold on;
%         plot(30*[1 1],ylim,'k');
        
        
%         subplot(2,6,12);                
        error = abs(diff(L_SPEED,[],2));
        conf = histcn(ds2(:),b2,'AccumData',abs(diff(L_SPEED,[],2)),'Fun',@nanmean);
        conf = conf(1:(length(b2)-1));
        CONFUSION_SPEED(i_ses) = nanmean(error);        
        conf(isnan(conf)) = 0;
        conf2 = jmm_smooth_1d_cor(conf,2);
        conf2(count==0) = NaN;
        conf(count==0) = NaN;
        CC_SPEED_TRIAL(:,i_mouse,i_ses) = conf;
%         plot(center(b2),conf2);
%         ylim([0 max(ylim)]);
%         xlabel('Position (cm)');
%         ylabel('Mean Confusion');
%         title(sprintf('Mean Confusion: %.02f', nanmean(error)));
%         axis square;
%         set(gca,'xtick',[0 15 30 35 50 65]);
%         set(gca,'xticklabel',{'0','15','30','0','15','30'});        
%         hold on;
%         plot(30*[1 1],ylim,'k');
        
        
        drawnow;
    %     pause;
    end
    PPP(i_mouse,:) = PERFORMANCE;
    DDD(i_mouse,:) = DECODING;
    EEE(i_mouse,:) = ERROR;
    CCC(i_mouse,:) = CONFUSION;
    
    DDD_SPEED_TRIAL(i_mouse,:) = DECODING_SPEED;
    EEE_SPEED_TRIAL(i_mouse,:) = ERROR_SPEED;
    CCC_SPEED_TRIAL(i_mouse,:) = CONFUSION_SPEED;
end



%%
subpanels = 'A BC   DE FG   HI J';
NCOLS = 4;
NROWS = 5;

WIDTH = 8.25*2.54;
HEIGHT = 8.25*2.54*5/4;
set(0,'defaultaxesfontsize',12)
close all;
[panel_pos, fig, all_ax, HH] = panelFigureSetup2( NCOLS, NROWS, subpanels,WIDTH,HEIGHT, 0.5, 1.5);

HH(1).Position = HH(1).Position+[-0.01 0.01 0];
HH(3).Position = HH(3).Position+[-0.04 0.01 0];
HH(4).Position = HH(4).Position+[0.02 0.01 0];
HH(8).Position = HH(8).Position+[0.02 0.005 0];

HH(9).Position = HH(9).Position+[-0.01 -0.025 0];
HH(11).Position = HH(11).Position+[-0.04 -0.025 0];
HH(12).Position = HH(12).Position+[0.02 -0.025 0];
HH(16).Position = HH(16).Position+[0.02 -0.03 0];
HH(17).Position = HH(17).Position+[-0.01 -0.045 0];
HH(19).Position = HH(19).Position+[0.05 -0.045 0];

FSZ = 20;
HH(1).FontSize = FSZ;
HH(3).FontSize = FSZ;
HH(4).FontSize = FSZ;
HH(8).FontSize = FSZ;
HH(9).FontSize = FSZ;
HH(11).FontSize = FSZ;
HH(12).FontSize = FSZ;
HH(16).FontSize = FSZ;
HH(17).FontSize = FSZ;
HH(19).FontSize = FSZ;


SCALE = 0.7;
% clf;
fsz = 18;
colors = {'b','r','k',orange,purple,green};
shapes = {'.','^','s','o','*','d'};
sizes = [15 9 9 9 9 9];
% clf;
y = DDD_POS_SPEED(:);
x = PPP_POS_SPEED(:);
n = NNN_POS_SPEED(:);
g = reshape(repmat((1:6)',[1 9]),[],1);
valid = ~isnan(y) & n>50;
[a1,b_1,stats1] = anovan(y(valid), {g(valid) x(valid)}, 'continuous', 2, 'display', 'off', 'model', 'full', 'varnames', char('Group', 'X-factor'));

P = panel_pos(12,:);
P(1) = P(1)+P(3)*0.35;
P(2) = P(2)+P(4)*0.1;
P(3:4) = P(3:4)*SCALE;
% P(4) = P(4)*2;
axes('position', P);
[C,p,P,ttl,h] = corplot(x(valid), y(valid), 1);
h(1).MarkerSize = 1;
h(1).Color = [1 1 1];
hold on;

for i_mouse = 1:totalMice
    valid2 = NNN_POS_SPEED(i_mouse,:)>50;    
    plot(PPP_POS_SPEED(i_mouse,valid2),DDD_POS_SPEED(i_mouse,valid2),shapes{i_mouse},'Color',colors{i_mouse},'Markersize',sizes(i_mouse));
    hold on;
end
xlim([0 1]);
ylim([0.5 1]);
    
% subplot(3,3,2);
if(a1(2)<1e-3)
    t = title(sprintf('R = %.02f, p = %.02e',C,a1(2)));
else
    t = title(sprintf('R = %.02f, p = %.02g',C,a1(2)));
end
t.Position = t.Position+[0 range(ylim)*0.02 0];
xlabel('Performance');
ylabel(sprintf('Indentification\nScore'));
% text(min(xlim)-0.05*range(xlim),max(ylim)-0.05*range(ylim),'f','HorizontalAlign','Right','VerticalAlign','Bottom','FontSize',fsz);


%
colors = {'b','r','k',orange,purple,green};
shapes = {'.','^','s','o','*','d'};
sizes = [15 9 9 9 9 9];
% clf;
y = EEE_POS_SPEED(:);
x = PPP_POS_SPEED(:);
n = NNN_POS_SPEED(:);
g = reshape(repmat((1:6)',[1 9]),[],1);
valid = ~isnan(y) & n>50;
[a2,b_2,stats2] = anovan(y(valid), {g(valid) x(valid)}, 'continuous', 2, 'display', 'off', 'model', 'full', 'varnames', char('Group', 'X-factor'));

P = panel_pos(16,:);
P(1) = P(1)+P(3)*0.35;
P(2) = P(2)+P(4)*0.1;
P(3:4) = P(3:4)*SCALE;
axes('position', P);
[C,p,P,ttl,h] = corplot(x(valid), y(valid), 1);
h(1).MarkerSize = 1;
h(1).Color = [1 1 1];
hold on;

for i_mouse = 1:totalMice
    valid2 = NNN_POS_SPEED(i_mouse,:)>50;
    plot(PPP_POS_SPEED(i_mouse,valid2),EEE_POS_SPEED(i_mouse,valid2),shapes{i_mouse},'Color',colors{i_mouse},'Markersize',sizes(i_mouse));
    hold on;
end
xlim([0 1]);
ylim([0.5 14.9]);
    
% subplot(3,3,2);
if(a2(2)<1e-3)
    t = title(sprintf('R = %.02f, p = %.02e',C,a2(2)));
else
    t = title(sprintf('R = %.02f, p = %.02g',C,a2(2)));
end
t.Position = t.Position+[0 range(ylim)*0.02 0];
xlabel('Performance');
ylabel(sprintf('Decoding\nError (cm)'));
% text(min(xlim)-0.05*range(xlim),max(ylim)-0.05*range(ylim),'g','HorizontalAlign','Right','VerticalAlign','Bottom','FontSize',fsz);


P = panel_pos(1,:);
P(2) = P(2)+P(4)*0.4;
P(3:4) = P(3:4)*SCALE;
P(3) = P(3)*2.3;
axes('position', P);
title('Decode Speed; Identify A/B');
axis off;

P = panel_pos(9,:);
P(2) = P(2)+P(4)*0.2;
P(3:4) = P(3:4)*SCALE;
P(3) = P(3)*2.3;
axes('position', P);
title('Decode Position; Identify Slow/Fast');
axis off;
%

P = panel_pos(9,:);
P(1) = P(1)-P(3)*0.2;
P(2) = P(2)+P(4)*0.65;
P(3) = P(3)*4.75;
axes('position', P);
plot([0 1],[1 1],'k-');
axis off;

P = panel_pos(17,:);
P(1) = P(1)-P(3)*0.2;
P(2) = P(2)+P(4)*0.45;
P(3) = P(3)*4.75;
axes('position', P);
plot([0 1],[1 1],'k-');
axis off;


colormap parula;
MS = 5;
selectSessions = [1 5];
for i_mouse = 4
    load(fullfile(base,mice{i_mouse},'matched_transients_all_ses.mat'),'matching_ROI_bin_transient_lap_data');

    %%
    these = matching_ROI_bin_transient_lap_data;

    PERFORMANCE = NaN(1,maxSessions);
    DECODING = NaN(1,maxSessions);
    ERROR = NaN(1,maxSessions);
    CONFUSION = NaN(1,maxSessions);
    for iii = 1:length(selectSessions)
        i_ses = selectSessions(iii);
    %     i_ses = 6;

        this = these{i_ses};

        N_BINS = 40;
        edgesPosition = linspace(0,1,N_BINS+1);
        centersPosition = center(edgesPosition);

        SM = 1;
        DT = nanmedian(diff(this.time));
        N_CELLS = size(this.transient_mat,1);

        [~,speedBin] = histc(this.speed,speedEdges);
        this.trialType = speedBin+1;
        
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
        %   
        %     eventRun = this.run_state(eventBinary)==1;
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
        goodCells = nansum(rateTogether,2)>0;
        selectRateTogether = rateTogether(goodCells,:);
        selectCoarseRate = coarseRate(goodCells,:);
        N_selectCells = sum(goodCells);

        NNN_POS_SPEED(i_mouse,i_ses) = sum(goodCells);
        instCorr = (zscore(selectRateTogether)'*zscore(selectCoarseRate))/(N_selectCells-1);
        [~, maxBin] = max(instCorr);
        confidence = max(instCorr) - quantile(instCorr,0.85);
        centersPosition2 = [centersPosition centersPosition+1];
        predPos = centersPosition2(maxBin);
        predPos(confidence==0) = NaN;
        downsampledPosition = interp1(this.time, this.position_norm, timeCenters,'nearest');
        downsampledRunState = interp1(this.time, 1*(this.run_state==1 & this.trialType>0), timeCenters,'nearest');
        downsampledPosition(downsampledRunState==0) = NaN;
        downsampledTrialType = interp1(this.time, this.trialType, timeCenters,'nearest');
        downsampledTrialType(downsampledRunState==0) = NaN;

%         clf;
        % subplot(2,1,1);
        % plot(timeCenters, downsampledPosition, 'b.');
        % hold on;
        % plot(timeCenters, predPos, 'r.');
        pp2 = predPos(downsampledRunState==1)*200;
        ic2 = instCorr(:,downsampledRunState==1);
        cf2 = confidence(downsampledRunState==1);
        tt2 = downsampledTrialType(downsampledRunState==1);
        dp2 = (downsampledPosition(downsampledRunState==1) + (tt2-2))*200;
        t2 = timeBin*(1:length(pp2));
        
        
        P = panel_pos(9+4*(iii-1),:);
%         P(2) = P(2)+(iii==1)*P(4)*0.2;
        P(2) = P(2)+P(4)*0.1;
        P(3:4) = P(3:4)*SCALE;
        P(3) = P(3)*2.3;
        axes('position', P);
%         subplot(4,2,1+2*(iii-1));
        
        start = t2(find(~isnan(pp2),1,'first'));        
        plot(t2-start, dp2, 'b.','Markersize',MS);
        hold on;
        plot(t2-start, pp2, 'r.','Markersize',MS);
        
        xlabel('Running Time (s)');
        ylabel('Position (cm)');
        axis tight;
        xlim([0 max(xlim)]);
        plot(xlim,200*[1 1],'k');
        ylim([-0.5 400.5]);
        legend('Actual','Decoded','Location','SouthEast');
%         set(gca,'ytick',50:50:400);
%         set(gca,'yticklabel',{'50','100','150','200','50','100','150','200'});
        set(gca,'ytick',0:100:400);
        set(gca,'yticklabel',{'0','100','200','100','200'});
        title(sprintf('Session %d',i_ses));
%         text(min(xlim)-0.05*range(xlim),max(ylim)+0.05*range(ylim),sprintf('a_%d',iii),'HorizontalAlign','Right','VerticalAlign','Bottom','FontSize',fsz);

        P2 = [P(1)+P(3)-P(3)*0.04 P(2) P(3)*0.1 P(4)];
        axes('position', P2);
        plot([0 0],[0 0.5],'Color',[0 0.7 0],'Linewidth',5);
        text(0.75,0.1,'Slow','Fontsize',12,'Rotation',90);
        hold on;
        plot([0 0],[0.5 1],'Color',[0.5 0.95 0.5],'Linewidth',5);
        text(0.75,0.6,'Fast','Fontsize',12,'Rotation',90);
        axis off;
        
        
        P = panel_pos(11+4*(iii-1),:);
        P(1) = P(1)+P(3)*0.1;
        P(2) = P(2)+P(4)*0.1;
%         P(2) = P(2)+(iii==1)*P(4)*0.2;
        P(3:4) = P(3:4)*SCALE;
        axes('position', P);
        b2 = linspace(0,400,2*N_BINS/2+1);
        b2 = [0 200 400.5];
        [N,~,~,L] = histcn([dp2(:) pp2(:)],b2,b2);        
        N = N';
        L(L==0) = NaN;
        N = bsxfun(@rdivide, N, sum(N));
        imagesc(center(b2),center(b2),(N));
        set(gca,'ydir','normal');
        axis square;
%         if(iii==2)
        xlabel('Actual Position (cm)');
%         end
        ylabel(sprintf('Decoded\nPosition (cm)'));
        set(gca,'xtick',100:100:400);
        set(gca,'xticklabel',{'100','200','100','200'});
        set(gca,'ytick',100:100:400);
        set(gca,'yticklabel',{'100','200','100','200'});
        hold on;
        plot(xlim,200*[1 1],'w');
        plot(200*[1 1],ylim,'w');
        caxis([0 1]);
        p = get(gca,'position');        
        colorbar;
        set(gca,'position',p+[-0.02 0 0 0]);
        plot(dp2+randn(size(dp2)),pp2+randn(size(dp2)),'o','Color','k','Markersize',2);
        
        AA = sum(dp2<200 & pp2<200);
        AB = sum(dp2<200 & pp2>=200);
        BA = sum(dp2>=200 & pp2<200);
        BB = sum(dp2>=200 & pp2>=200);

        F1_1 = 2*AA/(2*AA+AB+BA);
        F1_2 = 2*BB/(2*BB+AB+BA);
        F1_3 = (F1_1+F1_2)/2;
        F1_4 = (AA+BB)/(AA+AB+BA+BB);
        correctIncorrect = this.lap_data.performance.trialCorrect;
        perf = sum(correctIncorrect==1)/sum(correctIncorrect>=0);
%         title(sprintf('Performance: %.02f\nF1: %.02f', perf, F1_3));
        t = title(sprintf('Performance: %.02f', perf));
        t.Position = t.Position+[0 range(ylim)*0.02 0];
%         text(min(xlim)-0.05*range(xlim),max(ylim)+0.05*range(ylim),sprintf('b_%d',iii),'HorizontalAlign','Right','VerticalAlign','Bottom','FontSize',fsz);

        PERFORMANCE(i_ses) = perf;
        DECODING(i_ses) = F1_4;
 
    end
%     P = 
end



y = DDD_SPEED_TRIAL(:);
x = PPP(:);
n = NNN(:);
g = reshape(repmat((1:6)',[1 9]),[],1);
valid = ~isnan(y) & n>50;
[a1,b_1,stats1] = anovan(y(valid), {g(valid) x(valid)}, 'continuous', 2, 'display', 'off', 'model', 'full', 'varnames', char('Group', 'X-factor'));

P = panel_pos(4,:);
P(1) = P(1)+P(3)*0.35;
P(2) = P(2)+P(4)*0.3;
P(3:4) = P(3:4)*SCALE;
% P(4) = P(4)*2;
axes('position', P);
[C,p,P,ttl,h] = corplot(x(valid), y(valid), 1);
h(1).MarkerSize = 1;
h(1).Color = [1 1 1];
hold on;

for i_mouse = 1:totalMice
    valid2 = NNN(i_mouse,:)>50;    
    plot(PPP(i_mouse,valid2),DDD_SPEED_TRIAL(i_mouse,valid2),shapes{i_mouse},'Color',colors{i_mouse},'Markersize',sizes(i_mouse));
    hold on;
end
xlim([0 1]);
ylim([0.5 1]);
    
% subplot(3,3,2);
if(a1(2)<1e-3)
    t= title(sprintf('R = %.02f, p = %.02e',C,a1(2)));
else
    t = title(sprintf('R = %.02f, p = %.02g',C,a1(2)));
end
t.Position = t.Position+[0 range(ylim)*0.02 0];
xlabel('Performance');
ylabel(sprintf('Identification\nScore'));
% text(min(xlim)-0.05*range(xlim),max(ylim)-0.05*range(ylim),'f','HorizontalAlign','Right','VerticalAlign','Bottom','FontSize',fsz);


%
colors = {'b','r','k',orange,purple,green};
shapes = {'.','^','s','o','*','d'};
sizes = [15 9 9 9 9 9];
% clf;
y = EEE_SPEED_TRIAL(:);
x = PPP(:);
n = NNN(:);
g = reshape(repmat((1:6)',[1 9]),[],1);
valid = ~isnan(y) & n>50;
[a2,b_2,stats2] = anovan(y(valid), {g(valid) x(valid)}, 'continuous', 2, 'display', 'off', 'model', 'full', 'varnames', char('Group', 'X-factor'));

P = panel_pos(8,:);
P(1) = P(1)+P(3)*0.35;
P(2) = P(2)+P(4)*0.3;
P(3:4) = P(3:4)*SCALE;
axes('position', P);
[C,p,P,ttl,h] = corplot(x(valid), y(valid), 1);
h(1).MarkerSize = 1;
h(1).Color = [1 1 1];
hold on;

for i_mouse = 1:totalMice
    valid2 = NNN(i_mouse,:)>50;
    plot(PPP(i_mouse,valid2),EEE_SPEED_TRIAL(i_mouse,valid2),shapes{i_mouse},'Color',colors{i_mouse},'Markersize',sizes(i_mouse));
    hold on;
end
xlim([0 1]);
ylim([0.5 8.2]);
    
% subplot(3,3,2);
if(a2(2)<1e-3)
    t = title(sprintf('R = %.02f, p = %.02e',C,a2(2)));
else
    t = title(sprintf('R = %.02f, p = %.02g',C,a2(2)));
end
t.Position = t.Position+[0 range(ylim)*0.02 0];
xlabel('Performance');
ylabel(sprintf('Decoding\nError (cm/s)'));
% text(min(xlim)-0.05*range(xlim),max(ylim)-0.05*range(ylim),'g','HorizontalAlign','Right','VerticalAlign','Bottom','FontSize',fsz);


MS = 5;
selectSessions = [1 5];
for i_mouse = 4
%     load(fullfile(base,mice{i_mouse},'matched_transients_all_ses.mat'),'matching_ROI_bin_transient_lap_data');

    %%
    these = matching_ROI_bin_transient_lap_data;

    PERFORMANCE = NaN(1,maxSessions);
    DECODING = NaN(1,maxSessions);
    ERROR = NaN(1,maxSessions);
    CONFUSION = NaN(1,maxSessions);
    for iii = 1:length(selectSessions)
        i_ses = selectSessions(iii);
    %     i_ses = 6;

        this = these{i_ses};

        N_BINS = 40;
        edgesPosition = linspace(0,1,N_BINS+1);
        centersPosition = center(edgesPosition);
        edgesSpeed = linspace(-5,30,N_BINS+1);
        centersSpeed = center(edgesSpeed);
        
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

        occA_speed = DT*histcn(this.speed(this.run_state==1 & this.trialType==2 & valid1),edgesSpeed);
        occA_speed = jmm_smooth_1d_cor_circ(occA_speed,SM);
        occA_speed = occA_speed(1:N_BINS);

        occB_speed = DT*histcn(this.speed(this.run_state==1 & this.trialType==3 & valid1),edgesSpeed);
        occB_speed = jmm_smooth_1d_cor_circ(occB_speed,SM);
        occB_speed = occB_speed(1:N_BINS);
        
        timeBin = 0.25;
        SM2 = ceil(timeBin/DT);

        timeEdges = this.time(1):timeBin:this.time(end)+timeBin;
        timeCenters = center(timeEdges);
        coarseRate = NaN(N_CELLS,length(timeEdges)-1);
        eventsA = NaN(N_CELLS,N_BINS);
        eventsB = NaN(N_CELLS,N_BINS);
        eventsA_speed = NaN(N_CELLS,N_BINS);
        eventsB_speed = NaN(N_CELLS,N_BINS);
        for i_cell = 1:N_CELLS
            eventBinaryAll = (this.transient_mat(i_cell,:).*(this.run_state==1 & valid2)'==1);
            eventBinaryA = (this.transient_mat(i_cell,:).*(this.run_state==1 & this.trialType==2 & valid1)'==1);
            eventBinaryB = (this.transient_mat(i_cell,:).*(this.run_state==1 & this.trialType==3 & valid1)'==1);
        %   
        %     eventRun = this.run_state(eventBinary)==1;
            eventPosA = this.position_norm(eventBinaryA);
            eventPosB = this.position_norm(eventBinaryB);

            eventSpeedA = this.speed(eventBinaryA);
            eventSpeedB = this.speed(eventBinaryB);
            
            eventA = histcn(eventPosA,edgesPosition);
            eventA = jmm_smooth_1d_cor_circ(eventA,SM);
            eventA = eventA(1:N_BINS);
            eventsA(i_cell,:) = eventA;

            eventB = histcn(eventPosB,edgesPosition);
            eventB = jmm_smooth_1d_cor_circ(eventB,SM);
            eventB = eventB(1:N_BINS);
            eventsB(i_cell,:) = eventB;

            eventA_speed = histcn(eventSpeedA,edgesSpeed);
            eventA_speed = jmm_smooth_1d_cor(eventA_speed,SM);
            eventA_speed = eventA_speed(1:N_BINS);
            eventsA_speed(i_cell,:) = eventA_speed;

            eventB_speed = histcn(eventSpeedB,edgesSpeed);
            eventB_speed = jmm_smooth_1d_cor(eventB_speed,SM);
            eventB_speed = eventB_speed(1:N_BINS);
            eventsB_speed(i_cell,:) = eventB_speed;
            
            smoothedEventRate = jmm_smooth_1d_cor(double(eventBinaryAll), 2*SM2)/timeBin;
            coarseRate(i_cell,:) = interp1(this.time,smoothedEventRate,timeCenters);
        end

        rateA = bsxfun(@rdivide, eventsA, occA);
        rateB = bsxfun(@rdivide, eventsB, occB);
        rateTogether = [rateA rateB];
        
        rateA_speed = bsxfun(@rdivide, eventsA_speed, occA_speed);
        rateB_speed = bsxfun(@rdivide, eventsB_speed, occB_speed);
        rateTogether_speed = [rateA_speed rateB_speed];
        
        %
        goodCells = nansum(rateTogether,2)>0;
        selectRateTogether = rateTogether(goodCells,:);
        selectRateTogether_speed = rateTogether_speed(goodCells,:);
        selectCoarseRate = coarseRate(goodCells,:);
        N_selectCells = sum(goodCells);

        NNN(i_mouse,i_ses) = sum(goodCells);
        instCorr = (zscore(selectRateTogether)'*zscore(selectCoarseRate))/(N_selectCells-1);
        instCorr_speed = (zscore(selectRateTogether_speed)'*zscore(selectCoarseRate))/(N_selectCells-1);
        [~, maxBin] = max(instCorr);
        [~, maxBin_speed] = max(instCorr_speed);
        
        confidence = max(instCorr) - quantile(instCorr,0.85);
        confidence_speed = max(instCorr_speed) - quantile(instCorr_speed,0.85);
        centersPosition2 = [centersPosition centersPosition+1];
        centersSpeed2 = [centersSpeed centersSpeed+35];
        predPos = centersPosition2(maxBin);
        predPos(confidence==0) = NaN;
        predSpeed = centersSpeed2(maxBin_speed);
        predSpeed(confidence_speed==0) = NaN;
                
        downsampledPosition = interp1(this.time, this.position_norm, timeCenters,'nearest');
        downsampledSpeed = interp1(this.time, this.speed, timeCenters,'nearest');
        downsampledRunState = interp1(this.time, 1*(this.run_state==1 & this.trialType>0), timeCenters,'nearest');
        downsampledPosition(downsampledRunState==0) = NaN;
        downsampledSpeed(downsampledRunState==0) = NaN;
        downsampledTrialType = interp1(this.time, this.trialType, timeCenters,'nearest');
        downsampledTrialType(downsampledRunState==0) = NaN;

%         clf;
        % subplot(2,1,1);
        % plot(timeCenters, downsampledPosition, 'b.');
        % hold on;
        % plot(timeCenters, predPos, 'r.');
        pp2 = predPos(downsampledRunState==1)*200;
        ps2 = predSpeed(downsampledRunState==1);
        ic2 = instCorr(:,downsampledRunState==1);
        ics2 = instCorr_speed(:,downsampledRunState==1);
        cf2 = confidence(downsampledRunState==1);
        cfs2 = confidence_speed(downsampledRunState==1);
        tt2 = downsampledTrialType(downsampledRunState==1);
        dp2 = (downsampledPosition(downsampledRunState==1) + (tt2-2))*200;
        ds2 = (downsampledSpeed(downsampledRunState==1) + (tt2-2)*35);
        t2 = timeBin*(1:length(pp2));
        
        P = panel_pos(1+4*(iii-1),:);
        P(2) = P(2)+P(4)*0.3;
        P(3:4) = P(3:4)*SCALE;
        P(3) = P(3)*2.3;
        axes('position', P);
        
        start = t2(find(~isnan(ps2),1,'first'));        
        plot(t2-start, ds2, 'b.','Markersize',MS);
        hold on;
        plot(t2-start, ps2, 'r.','Markersize',MS);
        xlabel('Running Time (s)');
        ylabel('Speed (cm/s)');
        axis tight;
        xlim([0 max(xlim)]);
        plot(xlim,30*[1 1],'k');
        ylim([-5 65.5]);
        legend('Actual','Decoded','Location','SouthEast');
%         set(gca,'ytick',50:50:400);
%         set(gca,'yticklabel',{'50','100','150','200','50','100','150','200'});
        set(gca,'ytick',[15 30 50 65]);
        set(gca,'yticklabel',{'15','30','15','30'});        
        title(sprintf('Session %d',i_ses));
%         text(min(xlim)-0.05*range(xlim),max(ylim)+0.05*range(ylim),sprintf('a_%d',iii),'HorizontalAlign','Right','VerticalAlign','Bottom','FontSize',fsz);

        P2 = [P(1)+P(3)-P(3)*0.04 P(2) P(3)*0.1 P(4)];
        axes('position', P2);
        plot([0 0],[0 0.5],'Color',[0 0 0.85],'Linewidth',5);
        text(0.75,0.2,'A','Fontsize',12,'Rotation',90);
        hold on;
        plot([0 0],[0.5 1],'Color',[1 0 0],'Linewidth',5);
        text(0.75,0.7,'B','Fontsize',12,'Rotation',90);
        axis off;
        
%         subplot(2,4,5);
%         plot(dp2+randn(size(dp2)),pp2+randn(size(dp2)),'.','Color',purple,'Markersize',3);
%         hold on;
%         plot([0 200],[0 200],'k');
%         axis square;
%         xlabel('Actual Position (cm)');
%         ylabel('Decoded Position (cm)');
%         [c,p] = corrcoef(dp2,pp2,'rows','complete');
%         title(sprintf('Cells: %d\nR = %.02f', N_selectCells, c(1,2)));
%         set(gca,'xtick',100:100:400);
%         set(gca,'xticklabel',{'100','200','100','200'});
%         set(gca,'ytick',100:100:400);
%         set(gca,'yticklabel',{'100','200','100','200'});
%         xlim([0 400]);
%         ylim([0 400]);
%         plot(xlim,200*[1 1],'k');
%         plot(200*[1 1],ylim,'k');

        P = panel_pos(3+4*(iii-1),:);
        P(1) = P(1)+P(3)*0.1;
        P(2) = P(2)+P(4)*0.3;
        P(3:4) = P(3:4)*SCALE;
        axes('position', P);
        b2 = linspace(-5,65,2*N_BINS/2+1);
        b2 = [-5 30 65.5];
        [N_SPEED,~,~,L_SPEED] = histcn([ds2(:) ps2(:)],b2,b2);        
        N_SPEED = N_SPEED';
        L_SPEED(L_SPEED==0) = NaN;
        N_SPEED = bsxfun(@rdivide, N_SPEED, sum(N_SPEED));
        imagesc(center(b2),center(b2),(N_SPEED));
        set(gca,'ydir','normal');
        axis square;
        xlabel('Actual Speed (cm/s)');
        ylabel(sprintf('Decoded\nSpeed (cm/s)'));
        set(gca,'xtick',[15 30 50 65]);
        set(gca,'xticklabel',{'15','30','15','30'});
        set(gca,'ytick',[15 30 50 65]);
        set(gca,'yticklabel',{'15','30','15','30'});
        hold on;
        plot(xlim,30*[1 1],'w');
        plot(30*[1 1],ylim,'w');
        caxis([0 1]);
        p = get(gca,'position');        
        colorbar;
        set(gca,'position',p+[-0.02 0 0 0]);
        plot(ds2+randn(size(ds2)),ps2+randn(size(ds2)),'o','Color','k','Markersize',2);
        
        AA = sum(ds2<30 & ps2<30);
        AB = sum(ds2<30 & ps2>=30);
        BA = sum(ds2>=30 & ps2<30);
        BB = sum(ds2>=30 & ps2>=30);

        F1_1 = 2*AA/(2*AA+AB+BA);
        F1_2 = 2*BB/(2*BB+AB+BA);
        F1_3 = (F1_1+F1_2)/2;
        F1_4 = (AA+BB)/(AA+AB+BA+BB);
        correctIncorrect = this.lap_data.performance.trialCorrect;
        perf = sum(correctIncorrect==1)/sum(correctIncorrect>=0);
%         title(sprintf('Performance: %.02f\nF1: %.02f', perf, F1_3));
        t = title(sprintf('Performance: %.02f', perf));
        t.Position = t.Position+[0 range(ylim)*0.02 0];
%         text(min(xlim)-0.05*range(xlim),max(ylim)+0.05*range(ylim),sprintf('b_%d',iii),'HorizontalAlign','Right','VerticalAlign','Bottom','FontSize',fsz);

%         PERFORMANCE(i_ses) = perf;
%         DECODING(i_ses) = F1_4;
        %
        b2 = linspace(-5,65,2*N_BINS+1);
%         subplot(4,6,6+6*(iii-1));        
        error = 35*abs(angDiff(2*pi*(ps2+5)/35,2*pi*(ds2+5)/35))/(2*pi);
        count = histc(ds2(:),b2);
        count = count(1:(length(b2)-1));
        E = histcn(ds2(:),b2,'AccumData',error,'Fun',@nanmedian);
        E = E(1:(length(b2)-1));
        E(isnan(E)) = 0;
%         E(count==0) = NaN;
        E2 = jmm_smooth_1d_cor(E,2);
        E2(count==0) = NaN;
%         E(E==0) = NaN;
%         ERROR(i_ses) = nanmean(error);
%         EE(:,i_mouse,i_ses) = E;
%         E(isnan(E)) = 0;
%         plot(center(b2),E2);
%         jAXIS;
%         xlim([-5 65]);
%         ylim([0 max(ylim)]);
%         xlabel('Speed (cm/s)');
%         ylabel('Decoding Error (cm/s)');
%         title(sprintf('Median Error: %.02f cm', nanmedian(error)));
%         axis square;
%         set(gca,'xtick',[0 15 30 35 50 65]);
%         set(gca,'xticklabel',{'0','15','30','0','15','30'});
%         hold on;
%         plot(30*[1 1],ylim,'k');
%         text(min(xlim)-0.05*range(xlim),max(ylim)+0.05*range(ylim),sprintf('d_%d',iii),'HorizontalAlign','Right','VerticalAlign','Bottom','FontSize',fsz);

%         subplot(4,6,5+6*(iii-1));        
        
        
%         100*abs(angDiff(2*pi*pp2/200,2*pi*dp2/200))/pi;
        
%         conf = abs(diff(L,[],2));
        count = histc(ds2(:),b2);
        count = count(1:(length(b2)-1));
        error = 1-abs(diff(L_SPEED,[],2));
        conf = histcn(ds2(:),b2,'AccumData',error,'Fun',@nanmean);
%         conf = histcn(dp2(:),b2,'AccumData',cf2,'Fun',@median);
        conf(count==0) = NaN;
        conf = conf(1:(length(b2)-1));
        conf(isnan(conf)) = 1;
        conf2 = jmm_smooth_1d_cor(conf,2);
        conf2(count==0) = NaN;
%         plot(center(b2),conf2);
%         ylim([0 max(ylim)]);
%         xlabel('Speed (cm/s)');
%         ylabel('Decoding Score');
%         axis square;
%         set(gca,'xtick',[0 15 30 35 50 65]);
%         set(gca,'xticklabel',{'0','15','30','0','15','30'});        
%         hold on;
%         plot(30*[1 1],ylim,'k');
% %         CONFUSION(i_ses) = nanmean(conf);
% %         CC(:,i_mouse,i_ses) = conf;
%         text(min(xlim)-0.05*range(xlim),max(ylim)+0.05*range(ylim),sprintf('c_%d',iii),'HorizontalAlign','Right','VerticalAlign','Bottom','FontSize',fsz);
%         title(sprintf('Mean Decoding: %.02f', nanmean(error)));
%         
%         drawnow;
    %     pause;
    end
    
end

%%
colors1 = {'b','r',[0 0.5 0]};
P = panel_pos(19,:);
P(1) = P(1)+P(3)*0.25;
P(2) = P(2)-P(4)*0.07;
P(3) = P(3)*0.9;
P(4) = P(4)*0.85;
axes('position', P);
valid = NNN>50;
x = PPP(valid);
y1 = 100*EEE(valid)/200;
y3 = 100*EEE_POS_SPEED(valid)/200;
y2 = 100*EEE_SPEED_TRIAL(valid)/35;
y_1 = {y1,y2,y3};
ms = 12;
lw = 2;
for i_y = 1:length(y_1)
    plot(x,y_1{i_y},'.','Color',colors1{i_y},'Markersize',ms);
    hold on;
end
for i_y = 1:length(y_1)
    p = polyfit(x,y_1{i_y},1);
    plot(xlim,p(2)+xlim*p(1),'Color',colors1{i_y},'Linewidth',lw);
end
ylim([0 19.5]);
plot(0.7,18,'.','Color',colors1{1},'Markersize',ms);
plot(0.7,15.5,'.','Color',colors1{2},'Markersize',ms);
plot(0.7,13,'.','Color',colors1{3},'Markersize',ms);
text(0.73,18,'P_A_/_B','Fontsize',10);
text(0.73,15.5,'Sp_A_/_B','Fontsize',10);
text(0.73,13,'P_S_/_F','Fontsize',10);
xlabel('Performance');
ylabel('Norm. Decoding Error');

xxx = repmat(x,3,1);
yyy = cell2mat(y_1');
g = reshape(repmat((1:3),[43 1]),[],1);
[a1,b_1,stats1] = anovan(yyy, {g xxx}, 'continuous', 2, 'display', 'off', 'model', 'full', 'varnames', char('Group', 'X-factor'));


colors2 = {[0 0 0.75],[0.75 0 0],[0 0.4 0]};
P = panel_pos(17,:);
P(2) = P(2)-P(4)*0.07;
P(3) = P(3)*0.9;
P(4) = P(4)*0.85;
axes('position', P);
valid = NNN>50;
y4 = (DDD(valid));
y6 = (DDD_POS_SPEED(valid));
y5 = (DDD_SPEED_TRIAL(valid));
y_2 = {y4,y5,y6};
for i_y = 1:length(y_2)
    plot(x,y_2{i_y},'.','Color',colors2{i_y},'Markersize',ms);
    hold on;
end
for i_y = 1:length(y_2)
    p = polyfit(x,y_2{i_y},1);
    plot(xlim,p(2)+xlim*p(1),'Color',colors2{i_y},'Linewidth',lw);
end
ylim([55 110]/100);
% legend('6G','S11C','S11G');
plot(0.08,1.06,'.','Color',colors2{1},'Markersize',ms);
plot(0.08,0.995,'.','Color',colors2{2},'Markersize',ms);
plot(0.08,0.93,'.','Color',colors2{3},'Markersize',ms);
text(0.11,1.06,'P_A_/_B','Fontsize',10);
text(0.11,0.995,'Sp_A_/_B','Fontsize',10);
text(0.11,0.93,'P_S_/_F','Fontsize',10);
xlabel('Performance');
% plot(x,y4,'.');
% hold on;
% plot(x,y5,'.');
% plot(x,y6,'.');
ylabel('Identification Score');

xxx = repmat(x,3,1);
yyy = cell2mat(y_2');
g = reshape(repmat((1:3),[43 1]),[],1);
[a2, b_2, stats2] = anovan(yyy, {g xxx}, 'continuous', 2, 'display', 'off', 'model', 'full', 'varnames', char('Group', 'X-factor'));




P = panel_pos(20,:);
P(1) = P(1)+P(3)*0.25;
P(2) = P(2)-P(4)*0.07;
P(3) = P(3)*0.9;
P(4) = P(4)*0.85;
valid2 = NNN>50 & PPP>0.5;
axes('position', P);
MS1 = 8;
LW = 2;
y1 = 100*EEE(valid2)/100;
y3 = 100*EEE_POS_SPEED(valid2)/100;
y2 = 100*EEE_SPEED_TRIAL(valid2)/35;
y_1 = {y1,y2,y3};
[V1, M1, CI1, P1] = violinStats(y_1, MS1, colors1, LW, @(x,y) signrank(x,y));
xlim([0.25 3.75]);
set(gca,'xticklabel',{'P_A_/_B','Sp_A_/_B','P_S_/_F'});
ylim([0 17]);
markSig2(P1,(1:3)-0.2,0.1);

P = panel_pos(18,:);
P(1) = P(1)+P(3)*0.0;
P(2) = P(2)-P(4)*0.07;
P(3) = P(3)*0.9;
P(4) = P(4)*0.85;
axes('position', P);
MS1 = 8;
LW = 2;
y4 = (DDD(valid2));
y6 = (DDD_POS_SPEED(valid2));
y5 = (DDD_SPEED_TRIAL(valid2));
y_2 = {y4,y5,y6};
[V2, M2, CI2, P2] = violinStats(y_2, MS1, colors2, LW, @(x,y) signrank(x,y));
xlim([0.25 3.75]);
set(gca,'xticklabel',{'P_A_/_B','Sp_A_/_B','P_S_/_F'});
ylim([0.55 1.09]);
markSig2(P2,(1:3)-0.2,0.1);
%%
figure_path_primary = '.';
capstr = '';
fig = gcf;
savefig(fullfile(figure_path_primary, 'RZ_Rebut_J2_v2'), fig, false , capstr, 600);
print(fig, fullfile(figure_path_primary, 'RZ_Rebut_J2_v2'), '-painters', '-dsvg');


