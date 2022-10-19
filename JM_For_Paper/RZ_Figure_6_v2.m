cleer;
colormap parula;

base = '\\research-cifs.nyumc.org\Research\basulab\basulabspace\RZ\data_for_Jason\matching_binary_events_all_lap_learn';
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

DDD_SPEED = NaN(totalMice,maxSessions);
EEE_SPEED = NaN(totalMice,maxSessions);
CCC_SPEED = NaN(totalMice,maxSessions);
EE_SPEED = NaN(80,totalMice,maxSessions);
CC_SPEED = NaN(80,totalMice,maxSessions);

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
            
            smoothedEventRate = jmm_smooth_1d_cor(eventBinaryAll, 2*SM2)/timeBin;
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

        clf;
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
        subplot(2,2,1);
        plot(t2, dp2, 'b.');
        hold on;
        plot(t2, pp2, 'r.');
        xlabel('Running Time (s)');
        ylabel('Position (cm)');
        axis tight;
        plot(xlim,200*[1 1],'k');
        legend('Actual','Decoded');
        set(gca,'ytick',50:50:400);
        set(gca,'yticklabel',{'50','100','150','200','50','100','150','200'});

        subplot(2,2,2);
        plot(t2, ds2, 'b.');
        hold on;
        plot(t2, ps2, 'r.');
        xlabel('Running Time (s)');
        ylabel('Speed (cm/s)');
        axis tight;
        plot(xlim,30*[1 1],'k');
        legend('Actual','Decoded');
        set(gca,'ytick',[0 15 30 35 50 65]);
        set(gca,'yticklabel',{'0','15','30','0','15','30'});
        
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

        subplot(2,6,7);
        b2 = linspace(0,400,2*N_BINS/2+1);
        b2 = [0 200 400.5];
        [N,~,~,L] = histcn([dp2(:) pp2(:)],b2,b2);        
        N = N';
        L(L==0) = NaN;
        N = bsxfun(@rdivide, N, sum(N));
        imagesc(center(b2),center(b2),(N));
        set(gca,'ydir','normal');
        axis square;
        xlabel('Actual Position (cm)');
        ylabel('Decoded Position (cm)');
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
        plot(dp2+randn(size(dp2)),pp2+randn(size(dp2)),'o','Color','k','Markersize',3);
        
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
        title(sprintf('Performance: %.02f\nF1: %.02f', perf, F1_3));

        PERFORMANCE(i_ses) = perf;
        DECODING(i_ses) = F1_4;
        
        
        subplot(2,6,10);
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
        ylabel('Decoded Speed (cm/s)');
        set(gca,'xtick',[0 15 30 35 50 65]);
        set(gca,'xticklabel',{'0','15','30','0','15','30'});
        set(gca,'ytick',[0 15 30 35 50 65]);
        set(gca,'yticklabel',{'0','15','30','0','15','30'});
        hold on;
        plot(xlim,30*[1 1],'w');
        plot(30*[1 1],ylim,'w');
        caxis([0 1]);
        p = get(gca,'position');        
        colorbar;
        set(gca,'position',p+[-0.02 0 0 0]);
        plot(ds2+randn(size(ds2)),ps2+randn(size(ds2)),'o','Color','k','Markersize',3);
        
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
        title(sprintf('Performance: %.02f\nF1: %.02f', perf, F1_3));
        
        DECODING_SPEED(i_ses) = F1_4;
        
        
        
        %
        subplot(2,6,8);
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
        plot(center(b2),E2);
        jAXIS;
        xlim([0 400]);
        ylim([0 max(ylim)]);
        xlabel('Position (cm)');
        ylabel('Median Decoding Error (cm)');
        title(sprintf('Median Error: %.02f cm', ERROR(i_ses)));
        axis square;
        set(gca,'xtick',100:100:400);
        set(gca,'xticklabel',{'100','200','100','200'});
        hold on;
        plot(200*[1 1],ylim,'k');

                       
        subplot(2,6,9);                
        error = abs(diff(L,[],2));
        conf = histcn(dp2(:),b2,'AccumData',error,'Fun',@nanmean);
%         conf = histcn(dp2(:),b2,'AccumData',cf2,'Fun',@median);
        conf = conf(1:(length(b2)-1));
        conf(isnan(conf)) = 0;
        conf2 = jmm_smooth_1d_cor_circ(conf,2);
        conf2(count==0) = NaN;
        conf(count==0) = NaN;
        CC(:,i_mouse,i_ses) = conf;        
        plot(center(b2),conf2);
        ylim([0 max(ylim)]);
        xlabel('Position (cm)');
        ylabel('Mean Confusion');
        title(sprintf('Mean Confusion: %.02f', nanmean(error)));
        axis square;
        set(gca,'xtick',100:100:400);
        set(gca,'xticklabel',{'100','200','100','200'});
        hold on;
        plot(200*[1 1],ylim,'k');
        CONFUSION(i_ses) = nanmean(error);
        
        
        
        
        subplot(2,6,11);
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
        EE_SPEED(:,i_mouse,i_ses) = E;
        plot(center(b2),E2);
        jAXIS;
        xlim([-5 65]);
        ylim([0 max(ylim)]);
        xlabel('Speed (cm/s)');
        ylabel('Median Decoding Error (cm)');
        title(sprintf('Median Error: %.02f cm/s', ERROR_SPEED(i_ses)));
        axis square;
        set(gca,'xtick',[0 15 30 35 50 65]);
        set(gca,'xticklabel',{'0','15','30','0','15','30'});        
        hold on;
        plot(30*[1 1],ylim,'k');
        
        
        subplot(2,6,12);                
        error = abs(diff(L_SPEED,[],2));
        conf = histcn(ds2(:),b2,'AccumData',abs(diff(L_SPEED,[],2)),'Fun',@nanmean);
        conf = conf(1:(length(b2)-1));
        CONFUSION_SPEED(i_ses) = nanmean(error);        
        conf(isnan(conf)) = 0;
        conf2 = jmm_smooth_1d_cor(conf,2);
        conf2(count==0) = NaN;
        conf(count==0) = NaN;
        CC_SPEED(:,i_mouse,i_ses) = conf;
        plot(center(b2),conf2);
        ylim([0 max(ylim)]);
        xlabel('Position (cm)');
        ylabel('Mean Confusion');
        title(sprintf('Mean Confusion: %.02f', nanmean(error)));
        axis square;
        set(gca,'xtick',[0 15 30 35 50 65]);
        set(gca,'xticklabel',{'0','15','30','0','15','30'});        
        hold on;
        plot(30*[1 1],ylim,'k');
        
        
        drawnow;
    %     pause;
    end
    PPP(i_mouse,:) = PERFORMANCE;
    DDD(i_mouse,:) = DECODING;
    EEE(i_mouse,:) = ERROR;
    CCC(i_mouse,:) = CONFUSION;
    
    DDD_SPEED(i_mouse,:) = DECODING_SPEED;
    EEE_SPEED(i_mouse,:) = ERROR_SPEED;
    CCC_SPEED(i_mouse,:) = CONFUSION_SPEED;
end

%%
fsz = 18;
colors = {'b','r','k',orange,purple,green};
shapes = {'.','^','s','o','*','d'};
sizes = [15 9 9 9 9 9];

subpanels = 'A  B    CDIJEF  GH';
NCOLS = 4;
NROWS = 6;

WIDTH = 8.35*2.54;
HEIGHT = 9*2.54;
set(0,'defaultaxesfontsize',12)
close all;
[panel_pos, fig, all_ax, HH] = panelFigureSetup2( NCOLS, NROWS, subpanels,WIDTH,HEIGHT, 0.5, 1.5);

FSZ = 20;
HH(1).FontSize = FSZ;
HH(4).FontSize = FSZ;
HH(9).FontSize = FSZ;
HH(10).FontSize = FSZ;
HH(11).FontSize = FSZ;
HH(12).FontSize = FSZ;
HH(13).FontSize = FSZ;
HH(14).FontSize = FSZ;
HH(17).FontSize = FSZ;
HH(18).FontSize = FSZ;

HH(1).Position = HH(1).Position+[-0.03 0.03 0];
HH(4).Position = HH(4).Position+[0 0.03 0];
HH(9).Position = HH(9).Position+[0 -0.03 0];
HH(10).Position = HH(10).Position+[0 -0.03 0];
HH(13).Position = HH(13).Position+[0 -0.035 0];
HH(14).Position = HH(14).Position+[0 -0.035 0];
HH(17).Position = HH(17).Position+[0 -0.05 0];
HH(18).Position = HH(18).Position+[0 -0.05 0];
HH(11).Position = HH(11).Position+[0.04 -0.06 0];
HH(12).Position = HH(12).Position+[0.04 -0.06 0];

SCALE = 0.9;
SHIFT = 0.2;
H = 0.9;
H2 = 0.8;
y = DDD(:);
x = PPP(:);
n = NNN(:);
g = reshape(repmat((1:6)',[1 9]),[],1);
valid = ~isnan(y) & n>50;
[a1,b_1,stats1] = anovan(y(valid), {g(valid) x(valid)}, 'continuous', 2, 'display', 'off', 'model', 'full', 'varnames', char('Group', 'X-factor'));

P = panel_pos(13,:);
P(2) = P(2)-P(4)*0.1;
P(4) = P(4)*H2;
P(3:4) = P(3:4)*SCALE;
axes('position', P);
[C,p,P,ttl,h] = corplot(x(valid), y(valid), 1);
h(1).MarkerSize = 1;
h(1).Color = [1 1 1];
hold on;

for i_mouse = 1:totalMice
%     subplot(3,3,i_mouse+3);
    valid2 = NNN(i_mouse,:)>50;
%     [~,~,~,~,h] = corplot(PPP(i_mouse,valid2),DDD(i_mouse,valid2),'.');   
%     h(1).MarkerSize = 1;
%     plot(PPP(i_mouse,valid2),DDD(i_mouse,valid2),shapes{i_mouse},'Color',colors{i_mouse},'Markersize',sizes(i_mouse));
%     xlim([0 1]);
%     ylim([0.5 1]);
%     xlabel('Performance');
%     ylabel('Decoding');
%     text(min(xlim)+0.01*range(xlim),max(ylim),sprintf('Mouse %d', i_mouse),'Fontsize',14,'VerticalAlign','Top');
%     subplot(4,4,13);
    plot(PPP(i_mouse,valid2),DDD(i_mouse,valid2),shapes{i_mouse},'Color',colors{i_mouse},'Markersize',sizes(i_mouse));
    hold on;
end
xlim([0 1]);
ylim([0.5 1]);
    
% subplot(3,3,2);
if(a1(2)<1e-3)
    title(sprintf('R = %.02f, p = %.02e',C,a1(2)));
else
    title(sprintf('R = %.02f, p = %.02g',C,a1(2)));
end
xlabel('Performance');
ylabel(sprintf('Identification\nScore'));
% text(min(xlim)-0.05*range(xlim),max(ylim)-0.05*range(ylim),'g','HorizontalAlign','Right','VerticalAlign','Bottom','FontSize',fsz);


%
colors = {'b','r','k',orange,purple,green};
shapes = {'.','^','s','o','*','d'};
sizes = [15 9 9 9 9 9];
% clf;
y = EEE(:);
x = PPP(:);
n = NNN(:);
g = reshape(repmat((1:6)',[1 9]),[],1);
valid = ~isnan(y) & n>50;
[a2,b_2,stats2] = anovan(y(valid), {g(valid) x(valid)}, 'continuous', 2, 'display', 'off', 'model', 'full', 'varnames', char('Group', 'X-factor'));

P = panel_pos(14,:);
P(1) = P(1)+P(3)*SHIFT;
P(2) = P(2)-P(4)*0.1;
P(4) = P(4)*H2;
P(3:4) = P(3:4)*SCALE;
axes('position', P);
[C,p,P,ttl,h] = corplot(x(valid), y(valid), 1);
h(1).MarkerSize = 1;
h(1).Color = [1 1 1];
hold on;

for i_mouse = 1:totalMice
%     subplot(3,3,i_mouse+3);
    valid2 = NNN(i_mouse,:)>50;
%     [~,~,~,~,h] = corplot(PPP(i_mouse,valid2),EEE(i_mouse,valid2),'.');   
%     h(1).MarkerSize = 1;
%     plot(PPP(i_mouse,valid2),EEE(i_mouse,valid2),shapes{i_mouse},'Color',colors{i_mouse},'Markersize',sizes(i_mouse));
%     xlim([0 1]);
%     ylim([0.5 15.1]);
%     xlabel('Performance');
%     ylabel('Error (cm)');
%     text(min(xlim)+0.01*range(xlim),max(ylim),sprintf('Mouse %d', i_mouse),'Fontsize',14,'VerticalAlign','Top');
%     subplot(4,4,14);
    plot(PPP(i_mouse,valid2),EEE(i_mouse,valid2),shapes{i_mouse},'Color',colors{i_mouse},'Markersize',sizes(i_mouse));
    hold on;
end
xlim([0 1]);
ylim([0.5 14.9]);
    
% subplot(3,3,2);
if(a2(2)<1e-3)
    title(sprintf('R = %.02f, p = %.02e',C,a2(2)));
else
    title(sprintf('R = %.02f, p = %.02g',C,a2(2)));
end
xlabel('Performance');
ylabel(sprintf('Decoding\nError (cm)'));
% text(min(xlim)-0.05*range(xlim),max(ylim)-0.05*range(ylim),'h','HorizontalAlign','Right','VerticalAlign','Bottom','FontSize',fsz);


colormap parula;
P = panel_pos(18,:);
P(1) = P(1)+P(3)*SHIFT;
P(2) = P(2)-P(4)*0.275;
P(3) = P(3)*0.85;
P(4) = P(4)*H;
P(3:4) = P(3:4)*SCALE;
axes('position', P);

imagesc(centersPosition2*200,[],squeeze(nanmedian(EE(:,:,1:6),2))');
set(gca,'xtick',100:100:400);
set(gca,'xticklabel',{'100','200','100','200'});
hold on;
plot(200*[1 1],ylim,'w','Linewidth',2);
% axis square;
p0 = get(gca,'position');
c = colorbar;
caxis([0 max(caxis)]);
set(gca,'position',p0);
c.Position = c.Position+[-0.015 0 0 0];
xlabel('Position (cm)');
ylabel('Day');
title('Decoding Error (cm)');
% text(min(xlim)-0.05*range(xlim),min(ylim)+0.05*range(ylim),'j','HorizontalAlign','Right','VerticalAlign','Bottom','FontSize',fsz);

P = panel_pos(22,:);
P(1) = P(1)+P(3)*SHIFT;
P(2) = P(2)-P(4)*0.1;
P(3) = P(3)*0.85;
P(4) = P(4)*H;
P(3:4) = P(3:4)*SCALE;
axes('position', P);
temp = nanmedian(squeeze(nanmedian(EE(:,:,1:6),2)),2);
% plot(centersPosition2*200, jmm_smooth_1d_cor_circ(temp,1),'k','Linewidth',2);
plot(centersPosition2*200, jmm_smooth_1d_cor_circ(squeeze(nanmedian(EE(:,:,1),2)),1),'b');
hold on;
plot(centersPosition2*200, jmm_smooth_1d_cor_circ(squeeze(nanmedian(EE(:,:,6),2)),1),'r');
set(gca,'xtick',100:100:400);
set(gca,'xticklabel',{'100','200','100','200'});
hold on;
p = get(gca,'position');
c = colorbar;
set(gca,'position',p);
c.Visible = false;
ylim([0 19]);
plot(200*[1 1],ylim,'k','Linewidth',2);
xlabel('Position (cm)');
ylabel(sprintf('Decoding\nError (cm)'));
% legend('Day 1', 'Day 6');

P = panel_pos(17,:);
P(1) = P(1)+P(3)*0.05;
P(2) = P(2)-P(4)*0.275;
P(3) = P(3)*0.85;
P(4) = P(4)*H;
P(3:4) = P(3:4)*SCALE;
axes('position', P);
q = gca;
q.Position = q.Position + [-0.01 0 0 0];
imagesc(centersPosition2*200,[],squeeze(nanmedian(1-CC(:,:,1:6),2))');
set(gca,'xtick',100:100:400);
set(gca,'xticklabel',{'100','200','100','200'});
hold on;
plot(200*[1 1],ylim,'w','Linewidth',2);
p0 = get(gca,'position');
c = colorbar;
caxis([0.3 max(caxis)]);
set(gca,'position',p0);
c.Position = c.Position+[-0.015 0 0 0];
xlabel('Position (cm)');
ylabel('Day');
title('Identification Score');
% text(min(xlim)-0.05*range(xlim),min(ylim)+0.05*range(ylim),'i','HorizontalAlign','Right','VerticalAlign','Bottom','FontSize',fsz);


P = panel_pos(21,:);
P(1) = P(1)+P(3)*0.05;
P(2) = P(2)-P(4)*0.1;
P(3) = P(3)*0.85;
P(4) = P(4)*H;
P(3:4) = P(3:4)*SCALE;
axes('position', P);
q = gca;
q.Position = q.Position + [-0.01 0 0 0];
temp = nanmedian(squeeze(nanmedian(1-CC(:,:,1:6),2)),2);
% plot(centersPosition2*200, jmm_smooth_1d_cor_circ(temp,1),'k','Linewidth',2);
plot(centersPosition2*200, jmm_smooth_1d_cor_circ(squeeze(nanmedian(1-CC(:,:,1),2)),1),'b');
hold on;
plot(centersPosition2*200, jmm_smooth_1d_cor_circ(squeeze(nanmedian(1-CC(:,:,6),2)),1),'r');
set(gca,'xtick',100:100:400);
set(gca,'xticklabel',{'100','200','100','200'});
hold on;
p = get(gca,'position');
c = colorbar;
set(gca,'position',p);
c.Visible = false;
ylim([0 1.1]);
plot(200*[1 1],ylim,'k','Linewidth',2);
xlabel('Position (cm)');
ylabel(sprintf('Identification\nScore'));
plot([235 285],0.3*[1 1],'b');
plot([235 285],0.15*[1 1],'r');
text(290,0.3,'Day 1','Fontsize',8);
text(290,0.15,'Day 6','Fontsize',8);
% legend('Day 1', 'Day 6','Location','SouthEast');

% clf;
i_mouse = 4;
P = panel_pos(9,:);
P(2) = P(2)-P(4)*0.05;
P(4) = P(4)*H2;
P(3:4) = P(3:4)*SCALE;
axes('position', P);
plot(PPP(i_mouse,:),'Linewidth',2);
hold on;
plot(DDD(i_mouse,:),'r','Linewidth',2);
xlabel('Day');
ylabel(sprintf('Performance or\nIdentification Score'));
plot([6 7.5],0.3*[1 1],'b','Linewidth',2);
plot([6 7.5],0.15*[1 1],'r','Linewidth',2);
text(7.75,0.3,'Perf.','Fontsize',8);
text(7.75,0.15,'Iden.','Fontsize',8);
% legend('Perf.','Iden.','Location','SouthEast');
title(sprintf('Mouse %d', i_mouse));
% text(min(xlim)-0.05*range(xlim),max(ylim)-0.05*range(ylim),'d','HorizontalAlign','Right','VerticalAlign','Bottom','FontSize',fsz);
xlim([0.75 9.25]);

P = panel_pos(10,:);
P(1) = P(1)+P(3)*SHIFT;
P(2) = P(2)-P(4)*0.05;
P(4) = P(4)*H2;
P(3:4) = P(3:4)*SCALE;
axes('position', P);
corplot(PPP(i_mouse,:),DDD(i_mouse,:),1);
plot(PPP(i_mouse,:),DDD(i_mouse,:),'.','Color',purple,'Markersize',25);
xlabel('Performance');
ylabel(sprintf('Identification\nScore'));
% text(min(xlim)-0.05*range(xlim),max(ylim)-0.05*range(ylim),'e','HorizontalAlign','Right','VerticalAlign','Bottom','FontSize',fsz);


%
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

            smoothedEventRate = jmm_smooth_1d_cor(eventBinaryAll, 2*SM2)/timeBin;
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

        NNN(i_mouse,i_ses) = sum(goodCells);
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

        P = panel_pos(1+NCOLS*(iii-1),:);
        P(2) = P(2)+P(4)*(0.05+0.2*(2-iii));
        P(3) = P(3)*3;        
        axes('position', P);
        
        start = t2(find(~isnan(pp2),1,'first'));        
        plot(t2-start, dp2, 'b.','Markersize',MS);
        hold on;
        plot(t2-start, pp2, 'r.','Markersize',MS);
        if(iii==2)
            xlabel('Running Time (s)');
            ylabel('                                   Position (cm)');
        end
        axis tight;
        xlim([0 max(xlim)]);
        plot(xlim,200*[1 1],'k');
        ylim([-0.5 400.5]);
        legend('Actual','Decoded','Location','SouthEast');
%         set(gca,'ytick',50:50:400);
%         set(gca,'yticklabel',{'50','100','150','200','50','100','150','200'});
        set(gca,'ytick',0:100:400);
        set(gca,'yticklabel',{'0','100','200','100','200'});
        title(sprintf('Day %d',i_ses));
%         text(min(xlim)-0.05*range(xlim),max(ylim)+0.05*range(ylim),sprintf('a_%d',iii),'HorizontalAlign','Right','VerticalAlign','Bottom','FontSize',fsz);
        P2 = [P(1)+P(3)-P(3)*0.05 P(2) P(3)*0.1 P(4)];
        axes('position', P2);
        plot([0 0],[0 0.5],'Color',[0 0 0.85],'Linewidth',5);
        text(0.4,0.2,'A','Fontsize',12,'Rotation',90);
        hold on;
        plot([0 0],[0.5 1],'Color',[1 0 0],'Linewidth',5);
        text(0.4,0.7,'B','Fontsize',12,'Rotation',90);
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

        P = panel_pos(NCOLS*iii,:);    
        P(1) = P(1)+P(3)*0.15;
        P(2) = P(2)+P(4)*(0.05+0.2*(2-iii));
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
        if(iii==2)
            xlabel('Actual Position (cm)');
            ylabel('                                Decoded Position (cm)');
        end
        set(gca,'xtick',100:100:400);
        set(gca,'xticklabel',{'100','200','100','200'});
        set(gca,'ytick',100:100:400);
        set(gca,'yticklabel',{'100','200','100','200'});
        hold on;
        plot(xlim,200*[1 1],'w');
        plot(200*[1 1],ylim,'w');
        caxis([0 1]);
        p = get(gca,'position');        
        c = colorbar;
        c.Label.String = 'Proportion';
        set(gca,'position',p+[-0.02 0 0 0]);
        plot(dp2+randn(size(dp2)),pp2+randn(size(dp2)),'o','Color','k','Markersize',2);
        
        c.Position = c.Position+[-0.01 0 0 0];
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
        t.Position = t.Position+[0 range(ylim)*0.025 0];
%         text(min(xlim)-0.05*range(xlim),max(ylim)+0.05*range(ylim),sprintf('b_%d',iii),'HorizontalAlign','Right','VerticalAlign','Bottom','FontSize',fsz);

        PERFORMANCE(i_ses) = perf;
        DECODING(i_ses) = F1_4;
        %
        
        
%         subplot(4,6,6+6*(iii-1));        
%         b2 = linspace(0,400,2*N_BINS+1);
%         count = histc(dp2(:),b2);
%         count = count(1:(length(b2)-1));                
%         error = 100*abs(angDiff(2*pi*pp2/200,2*pi*dp2/200))/pi;
%         E = histcn(dp2(:),b2,'AccumData',error,'Fun',@nanmedian);
%         E = E(1:(length(b2)-1));
%         E(isnan(E)) = 0;
%         E2 = jmm_smooth_1d_cor_circ(E,2);
%         E2(count==0) = NaN;
%         plot(center(b2),E2);
%         jAXIS;
%         xlim([0 400]);
%         ylim([0 max(ylim)]);
%         xlabel('Position (cm)');
%         ylabel('Decoding Error (cm)');
%         title(sprintf('Median Error: %.02f cm', nanmedian(error)));
%         axis square;
%         set(gca,'xtick',100:100:400);
%         set(gca,'xticklabel',{'100','200','100','200'});
%         hold on;
%         plot(200*[1 1],ylim,'k');
%         text(min(xlim)-0.05*range(xlim),max(ylim)+0.05*range(ylim),sprintf('d_%d',iii),'HorizontalAlign','Right','VerticalAlign','Bottom','FontSize',fsz);
% 
%         subplot(4,6,5+6*(iii-1));        
%         
%         
% %         100*abs(angDiff(2*pi*pp2/200,2*pi*dp2/200))/pi;
%         
% %         conf = abs(diff(L,[],2));
%         
%         count = histc(dp2(:),b2);
%         count = count(1:(length(b2)-1));
%         error = 1-abs(diff(L,[],2));
%         conf = histcn(dp2(:),b2,'AccumData',error,'Fun',@nanmean);
%         conf(count==0) = NaN;
%         conf = conf(1:(length(b2)-1));
%         conf(isnan(conf)) = 1;
%         conf2 = jmm_smooth_1d_cor_circ(conf,2);
%         conf2(count==0) = NaN;                
%         plot(center(b2),conf2);
%         ylim([0 max(ylim)]);
%         xlabel('Position (cm)');
%         ylabel('Decoding Score');
%         title(sprintf('Mean Decoding: %.02f', nanmean(conf)));
%         axis square;
%         set(gca,'xtick',100:100:400);
%         set(gca,'xticklabel',{'100','200','100','200'});
%         hold on;
%         plot(200*[1 1],ylim,'k');
% %         CONFUSION(i_ses) = nanmean(conf);
% %         CC(:,i_mouse,i_ses) = conf;
%         text(min(xlim)-0.05*range(xlim),max(ylim)+0.05*range(ylim),sprintf('c_%d',iii),'HorizontalAlign','Right','VerticalAlign','Bottom','FontSize',fsz);

        drawnow;
    %     pause;
    end
    
end

%%
MODE = 1;

if(MODE==1)
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
% CA2 = -CA2(end:-1:1);


%%
SCALE = 1;
SHIFT = 0.35;

P = panel_pos(11,:);
P(1) = P(1)+P(3)*SHIFT;
P(2) = P(2)-P(4)*0.5;
P(3:4) = P(3:4)*SCALE;
axes('position', P);
imagesc(squeeze(nanmean(DECODING_L,1))');
caxis(CA1);
% xlim([0.5 7.5]);
% ylim([0.5 7.5]);
axis square;
colorbar;
xlabel('Train Day');
xlim([0.5 6.5]);
ylim([0.5 6.5]);
y = ylabel('Decode Day');
y.Position = y.Position+[-0.25 0 0];
title('Identification Score');

P = panel_pos(12,:);
P(1) = P(1)+P(3)*SHIFT;
P(2) = P(2)-P(4)*0.5;
P(3:4) = P(3:4)*SCALE;
axes('position', P);
imagesc(squeeze(nanmean(ERROR_L,1))');
caxis(CA2);
% xlim([0.5 7.5]);
% ylim([0.5 7.5]);
axis square;
c = colorbar;
% c.TickLabels = {'30','20','10'};
xlabel('Train Day');
xlim([0.5 6.5]);
ylim([0.5 6.5]);
y = ylabel('Decode Day');
y.Position = y.Position+[-0.25 0 0];
title('Decoding Error (cm)');

H = 1.2;
SCALE = 0.8;

c1 = 0.75*[1 1 1];
c2 = [1 0 1];
% clf;
P = panel_pos(19,:);
P(1) = P(1)+P(3)*SHIFT;
P(2) = P(2)-P(4)*1.15;
P(3:4) = P(3:4)*SCALE;
P(4) = P(4)*H;
axes('position', P);
plot(1:maxSessions,decoding_dayDecay2_L,'Color',c1);
hold on;
p = seplot(1:maxSessions,decoding_dayDecay2_L',c2,1);
p(2).LineStyle = '--';
p(3).LineStyle = '--';
% plot(1:maxSessions,nanmedian(decoding_dayDecay2),'Color',c2,'Linewidth',5);
% axis square;
jAXIS;
xlabel('Day comparison');
ylabel('                                    Identification Score');
% ylabel('Decoding Accuracy');
xlim([1 5]);
set(gca,'xtick',1:5);
set(gca,'xticklabel',{'1 vs. 2','2 vs. 3','3 vs. 4','4 vs. 5','5 vs. 6'});
set(gca,'XTickLabelRotation',30);
ylim([0.55 0.91]);

xxx = repmat(1:5,6,1);
yyy = decoding_dayDecay2_L(:,1:5);
g = repmat((1:6)',[1 5]);
valid = ~isnan(yyy);
[a3, b_3, stats3] = anovan(yyy(valid), {g(valid) xxx(valid)}, 'continuous', 2, 'display', 'off', 'model', 'full', 'varnames', char('Group', 'X-factor'));



P = panel_pos(20,:);
P(1) = P(1)+P(3)*SHIFT;
P(2) = P(2)-P(4)*1.15;
P(3:4) = P(3:4)*SCALE;
P(4) = P(4)*H;
axes('position', P);
plot(1:maxSessions,error_dayDecay2_L,'Color',c1);
hold on;
p = seplot(1:maxSessions,error_dayDecay2_L',c2,1);
p(2).LineStyle = '--';
p(3).LineStyle = '--';
% axis square;
jAXIS;
xlabel('Day Comparison'); 
ylabel('                                     Decoding Error (cm)');
% ylabel('Decoding Error');
xlim([1 5]);
set(gca,'xtick',1:5);
set(gca,'xticklabel',{'1 vs. 2','2 vs. 3','3 vs. 4','4 vs. 5','5 vs. 6'});
set(gca,'XTickLabelRotation',30);
ylim([5 27]);
% set(gca,'ydir','reverse');
% colorbar('Visible','off');
% xlim([1 6]);
xxx = repmat(1:5,6,1);
yyy = error_dayDecay2_L(:,1:5);
g = repmat((1:6)',[1 5]);
valid = ~isnan(yyy);
[a4, b_4, stats4] = anovan(yyy(valid), {g(valid) xxx(valid)}, 'continuous', 2, 'display', 'off', 'model', 'full', 'varnames', char('Group', 'X-factor'));


P = panel_pos(15,:);
P(1) = P(1)+P(3)*SHIFT;
P(2) = P(2)-P(4)*0.75;
P(3:4) = P(3:4)*SCALE;
P(4) = P(4)*H;
axes('position', P);
plot(diags,decoding_dayDecay_L,'Color',c1);
hold on;
p = seplot(diags,decoding_dayDecay_L',c2,1);
p(2).LineStyle = '--';
p(3).LineStyle = '--';
% axis square;
xlabel('Relative Day');
jAXIS;
xlim([-5 5]);
set(gca,'xtick',[-4 -2 0 2 4]);
% xlim([-6 6])
ylim([0.5 1]);
% colorbar('Visible','off');

P = panel_pos(16,:);
P(1) = P(1)+P(3)*SHIFT;
P(2) = P(2)-P(4)*0.75;
P(3:4) = P(3:4)*SCALE;
P(4) = P(4)*H;
axes('position', P);
plot(diags,error_dayDecay_L,'Color',c1);
hold on;
p = seplot(diags,error_dayDecay_L',c2,1);
p(2).LineStyle = '--';
p(3).LineStyle = '--';
% axis square;
xlabel('Relative Day');
jAXIS;
xlim([-5 5]);
set(gca,'xtick',[-4 -2 0 2 4]);
ylim([0 34]);
% set(gca,'ydir','reverse');


%%
figure_path_primary = '.';
capstr = '';
fig = gcf;
savefig(fullfile(figure_path_primary, 'RZ_Rebut_Fig_6_v2'), fig, false , capstr, 600);
print(fig, fullfile(figure_path_primary, 'RZ_Rebut_Fig_6_v2'), '-painters', '-dsvg');

