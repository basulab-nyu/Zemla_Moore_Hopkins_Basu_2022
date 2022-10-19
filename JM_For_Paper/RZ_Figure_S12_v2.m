cleer;
colormap parula;

LOAD = 1;

if(LOAD==1)
    base = '\\research-cifs.nyumc.org\Research\basulab\basulabspace\RZ\data_for_Jason\rev\matching_binary_events_all_lap_learn';

else
    base = 'F:\MATLAB\D_Drive\RZ_Data\rev\matching_binary_events_all_lap_learn';
end

mice = dir(base);
mice = arrayfun(@(x) x.name, mice(3:end), 'UniformOutput', false);

i_mouse = 1;

totalMice = length(mice);
maxSessions = 9;

PPP = NaN(totalMice,maxSessions);
DDD = NaN(totalMice,maxSessions);
NNN = NaN(totalMice,maxSessions);
EEE = NaN(totalMice,maxSessions);
EE = NaN(80,totalMice,maxSessions);
CC = NaN(80,totalMice,maxSessions);
CCC = NaN(totalMice,maxSessions);
for i_mouse = 1:totalMice
    load(fullfile(base,mice{i_mouse},'matched_transients_all_ses.mat'),'matching_ROI_bin_transient_lap_data');

    %%
    these = matching_ROI_bin_transient_lap_data;

    PERFORMANCE = NaN(1,maxSessions);
    DECODING = NaN(1,maxSessions);
    ERROR = NaN(1,maxSessions);
    CONFUSION = NaN(1,maxSessions);
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

        clf;
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
        subplot(2,1,1);
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

        subplot(2,3,4);
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
        %
        b2 = linspace(0,400,2*N_BINS+1);
        subplot(2,3,5);
        error = 100*abs(angDiff(2*pi*pp2/200,2*pi*dp2/200))/pi;
        E = histcn(dp2(:),b2,'AccumData',error,'Fun',@nanmedian);
        E = E(1:(length(b2)-1));
        ERROR(i_ses) = nanmean(error);
        EE(:,i_mouse,i_ses) = E;
        plot(center(b2),jmm_smooth_1d_cor_circ(E,2));
        jAXIS;
        xlim([0 400]);
        ylim([0 max(ylim)]);
        xlabel('Position (cm)');
        ylabel('Median Decoding Error (cm)');
        title(sprintf('Median Error: %.02f cm', nanmedian(error)));
        axis square;
        set(gca,'xtick',100:100:400);
        set(gca,'xticklabel',{'100','200','100','200'});
        hold on;
        plot(200*[1 1],ylim,'k');

        subplot(2,3,6);
        
        
%         100*abs(angDiff(2*pi*pp2/200,2*pi*dp2/200))/pi;
        
%         conf = abs(diff(L,[],2));
        
        conf = histcn(dp2(:),b2,'AccumData',abs(diff(L,[],2)),'Fun',@nanmean);
%         conf = histcn(dp2(:),b2,'AccumData',cf2,'Fun',@median);
        conf = conf(1:(length(b2)-1));
        plot(center(b2),jmm_smooth_1d_cor_circ(conf,2));
        ylim([0 max(ylim)]);
        xlabel('Position (cm)');
        ylabel('Mean Confusion');
        title(sprintf('Mean Confusion: %.02f', nanmean(conf)));
        axis square;
        set(gca,'xtick',100:100:400);
        set(gca,'xticklabel',{'100','200','100','200'});
        hold on;
        plot(200*[1 1],ylim,'k');
        CONFUSION(i_ses) = nanmean(conf);
        CC(:,i_mouse,i_ses) = conf;
        
        drawnow;
    %     pause;
    end
    PPP(i_mouse,:) = PERFORMANCE;
    DDD(i_mouse,:) = DECODING;
    EEE(i_mouse,:) = ERROR;
    CCC(i_mouse,:) = CONFUSION;
end
PPP(1,1) = 1;
%%
fsz = 18;
colors = {'b','r','k',orange,purple,green};
shapes = {'.','^','s','o','*','d'};
sizes = [15 9 9 9 9 9];

subpanels = 'A B CD          EFG H IJ';
NCOLS = 6;
NROWS = 6;

WIDTH = 8.35*2.54;
HEIGHT = 9*2.54;
set(0,'defaultaxesfontsize',12)
close all;
[panel_pos, fig, all_ax, HH] = panelFigureSetup2( NCOLS, NROWS, subpanels,WIDTH,HEIGHT, 0.5, 1.5);

HH(1).Position = HH(1).Position+[-0.03 0.03 0];
HH(3).Position = HH(3).Position+[-0.01 0.03 0];
HH(5).Position = HH(5).Position+[-0.065 0.03 0];
HH(6).Position = HH(6).Position+[0.01 0.03 0];
HH(17).Position = HH(17).Position+[-0.065 0.1 0];
HH(18).Position = HH(18).Position+[0.01 0.1 0];
HH(19).Position = HH(19).Position+[-0.025 0 0];
HH(21).Position = HH(21).Position+[-0.05 0 0];
HH(23).Position = HH(23).Position+[-0.1 0 0];
HH(24).Position = HH(24).Position+[-0.01 0 0];
% HH(25).Position = HH(25).Position+[-0.025 -0.04 0];

HH(1).FontSize = 20;
HH(3).FontSize = 20;
HH(5).FontSize = 20;
HH(6).FontSize = 20;
HH(17).FontSize = 20;
HH(18).FontSize = 20;
HH(19).FontSize = 20;
HH(21).FontSize = 20;
HH(23).FontSize = 20;
HH(24).FontSize = 20;
% HH(25).FontSize = 20;

H1 = 1.2;
W = 1.15;
SHIFT = 1.85;
OFFSET = 0.3;
y = DDD(:);
x = PPP(:);
n = NNN(:);
g = reshape(repmat((1:totalMice)',[1 9]),[],1);
valid = ~isnan(y) & n>50;
[a1,b_1,stats1] = anovan(y(valid), {g(valid) x(valid)}, 'continuous', 2, 'display', 'off', 'model', 'full', 'varnames', char('Group', 'X-factor'));

P = panel_pos(17,:);
% P(1) = P(1)+(OFFSET+2*SHIFT)*P(3);
P(1) = P(1)-P(3)*0.25;
P(2) = P(2)+P(4)*0.5;
P(3) = P(3)*W;
P(4) = P(4)*H1;
axes('position', P);

[C,p,P,ttl,h] = corplot(x(valid), y(valid), 1);
h(1).MarkerSize = 1;
h(1).Color = [1 1 1];
hold on;

for i_mouse = 1:totalMice
    valid2 = NNN(i_mouse,:)>50;
    plot(PPP(i_mouse,valid2),DDD(i_mouse,valid2),shapes{i_mouse},'Color',colors{i_mouse},'Markersize',sizes(i_mouse));
    hold on;
end
xlim([0 1]);
ylim([0.6 1]);
    
% subplot(3,3,2);
if(a1(2)<1e-3)
    title(sprintf('R = %.02f, p = %.02e',C,a1(2)));
else
    title(sprintf('R = %.02f, p = %.02g',C,a1(2)));
end
xlabel('Performance');
ylabel('Identification Score');
% text(min(xlim)-0.05*range(xlim),max(ylim)-0.05*range(ylim),'f','HorizontalAlign','Right','VerticalAlign','Bottom','FontSize',fsz);


%
colors = {'b','r','k',orange,purple,green};
shapes = {'.','^','s','o','*','d'};
sizes = [15 9 9 9 9 9];
% clf;
y = EEE(:);
x = PPP(:);
n = NNN(:);
g = reshape(repmat((1:totalMice)',[1 9]),[],1);
valid = ~isnan(y) & n>50;
[a2,b_2,stats2] = anovan(y(valid), {g(valid) x(valid)}, 'continuous', 2, 'display', 'off', 'model', 'full', 'varnames', char('Group', 'X-factor'));

P = panel_pos(18,:);
P(1) = P(1)+P(3)*0.25;
P(2) = P(2)+P(4)*0.5;
P(3) = P(3)*W;
P(4) = P(4)*H1;
axes('position', P);

[C,p,P,ttl,h] = corplot(x(valid), y(valid), 1);
h(1).MarkerSize = 1;
h(1).Color = [1 1 1];
hold on;

for i_mouse = 1:totalMice
    valid2 = NNN(i_mouse,:)>50;
    plot(PPP(i_mouse,valid2),EEE(i_mouse,valid2),shapes{i_mouse},'Color',colors{i_mouse},'Markersize',sizes(i_mouse));
    hold on;
end
xlim([0 1]);
ylim([3 14.9]);
    
% subplot(3,3,2);
if(a2(2)<1e-3)
    title(sprintf('R = %.02f, p = %.02e',C,a2(2)));
else
    title(sprintf('R = %.02f, p = %.02g',C,a2(2)));
end
xlabel('Performance');
ylabel('Decoding Error (cm)');
% text(min(xlim)-0.05*range(xlim),max(ylim)-0.05*range(ylim),'g','HorizontalAlign','Right','VerticalAlign','Bottom','FontSize',fsz);

W2 = 1.2;
H2 = 1.4;

colormap parula;
P = panel_pos(26,:);
P(1) = P(1)+P(3)*0.8;
P(2) = P(2)+P(4)*0.6;
P(3) = P(3)*W2;
P(4) = P(4)*H2;
axes('position', P);
imagesc(centersPosition2*200,[],squeeze(nanmedian(EE(:,:,1:9),2))');
set(gca,'xtick',100:100:400);
set(gca,'xticklabel',{'100','200','100','200'});
hold on;
plot(200*[1 1],ylim,'w','Linewidth',2);
% axis square;
p0 = get(gca,'position');
colorbar;
caxis([0 max(caxis)]);
set(gca,'position',p0);
% xlabel('Position (cm)');
ylabel('Day');
title('Decoding Error (cm)');
% text(min(xlim)-0.05*range(xlim),min(ylim)+0.05*range(ylim),'i','HorizontalAlign','Right','VerticalAlign','Bottom','FontSize',fsz);

P = panel_pos(32,:);
P(1) = P(1)+P(3)*0.8;
P(2) = P(2)-P(4)*0;
P(3) = P(3)*W2;
P(4) = P(4)*H2;
axes('position', P);
temp = nanmedian(squeeze(nanmedian(EE(:,:,1:6),2)),2);
% plot(centersPosition2*200, jmm_smooth_1d_cor_circ(temp,1),'k','Linewidth',2);
plot(centersPosition2*200, jmm_smooth_1d_cor_circ(squeeze(nanmedian(EE(:,:,1),2)),1),'b');
hold on;
plot(centersPosition2*200, jmm_smooth_1d_cor_circ(squeeze(nanmedian(EE(:,:,4),2)),1),'r');
plot(centersPosition2*200, jmm_smooth_1d_cor_circ(squeeze(nanmedian(EE(:,:,7),2)),1),'m');
set(gca,'xtick',100:100:400);
set(gca,'xticklabel',{'100','200','100','200'});
hold on;
p = get(gca,'position');
c = colorbar;
set(gca,'position',p);
% c.Visible = false;
c.Visible = 'off';
ylim([0 19]);
plot(200*[1 1],ylim,'k','Linewidth',2);
xlabel('Position (cm)');
ylabel('Decoding Error (cm)');
% legend('Median', 'Session 1', 'Session 4', 'Session 7');

P = panel_pos(25,:);
P(1) = P(1)+P(3)*0;
P(2) = P(2)+P(4)*0.6;
P(3) = P(3)*W2;
P(4) = P(4)*H2;
axes('position', P);
q = gca;
q.Position = q.Position + [-0.01 0 0 0];
imagesc(centersPosition2*200,[],squeeze(nanmedian(1-CC(:,:,1:9),2))');
set(gca,'xtick',100:100:400);
set(gca,'xticklabel',{'100','200','100','200'});
hold on;
plot(200*[1 1],ylim,'w','Linewidth',2);
p0 = get(gca,'position');
colorbar;
caxis([0.3 1.01*max(caxis)]);
set(gca,'position',p0);
% xlabel('Position (cm)');
ylabel('Day');
title('Identification Score');
% text(min(xlim)-0.05*range(xlim),min(ylim)+0.05*range(ylim),'h','HorizontalAlign','Right','VerticalAlign','Bottom','FontSize',fsz);


P = panel_pos(31,:);
P(1) = P(1)+P(3)*0;
P(2) = P(2)-P(4)*0;
P(3) = P(3)*W2;
P(4) = P(4)*H2;
axes('position', P);
q = gca;
q.Position = q.Position + [-0.01 0 0 0];
temp = nanmedian(squeeze(nanmedian(1-CC(:,:,1:9),2)),2);
% plot(centersPosition2*200, jmm_smooth_1d_cor_circ(temp,1),'k','Linewidth',2);
plot(centersPosition2*200, jmm_smooth_1d_cor_circ(squeeze(nanmedian(1-CC(:,:,1),2)),1),'b');
hold on;
plot(centersPosition2*200, jmm_smooth_1d_cor_circ(squeeze(nanmedian(1-CC(:,:,4),2)),1),'r');
plot(centersPosition2*200, jmm_smooth_1d_cor_circ(squeeze(nanmedian(1-CC(:,:,7),2)),1),'m');
set(gca,'xtick',100:100:400);
set(gca,'xticklabel',{'100','200','100','200'});
hold on;
p = get(gca,'position');
c = colorbar;
set(gca,'position',p);
% c.Visible = false;
c.Visible = 'off';
ylim([0 1.1]);
plot(200*[1 1],ylim,'k','Linewidth',2);
xlabel('Position (cm)');
ylabel('Identification Score');

fz = 8;
% plot([75 175],0.35*[1 1],'k','Linewidth',2);
plot([25 80],0.3*[1 1],'b');
plot([25 80],0.2*[1 1],'r');
plot([25 80],0.1*[1 1],'m');
% text(205, 0.35, 'Median','Fontsize',fz);
text(90, 0.3, 'Day 1','Fontsize',fz);
text(90, 0.2, 'Day 4','Fontsize',fz);
text(90, 0.1, 'Day 7','Fontsize',fz);
% legend('Median', 'Session 1', 'Session 4', 'Session 7','Location','SouthEast');

% clf;
I_MOUSE = 2;
i_mouse = I_MOUSE;
P = panel_pos(5,:);
% P(1) = P(1)+(OFFSET+0*SHIFT)*P(3);
P(1) = P(1)-P(3)*0.25;
% P(2) = P(2)+P(4)*0.5;
P(3) = P(3)*W;
P(4) = P(4)*H1;
axes('position', P);
plot(PPP(i_mouse,:),'Linewidth',2);
hold on;
plot(DDD(i_mouse,:),'r','Linewidth',2);
xlabel('Day');
ylabel(sprintf('Performance or\nIdentification Score'));
title(sprintf('Mouse %d', i_mouse));
% text(min(xlim)-0.05*range(xlim),max(ylim)-0.05*range(ylim),'d','HorizontalAlign','Right','VerticalAlign','Bottom','FontSize',fsz);
plot(4*[1 1],ylim,'k--');
% plot(
% legend('Perf.','Iden.','Location','SouthEast');
plot([6 7],0.525*[1 1],'b','Linewidth',2);
plot([6 7],0.45*[1 1],'r','Linewidth',2);
text(9,0.525,'Perf.','Fontsize',8,'HorizontalAlign','Right');
text(9,0.45,'Iden.','Fontsize',8,'HorizontalAlign','Right');
xlim([0.75 9.25]);

P = panel_pos(6,:);
P(1) = P(1)+P(3)*0.25;
% P(2) = P(2)+P(4)*0.5;
P(3) = P(3)*W;
P(4) = P(4)*H1;
axes('position', P);
corplot(PPP(i_mouse,:),DDD(i_mouse,:),1);
plot(PPP(i_mouse,:),DDD(i_mouse,:),'.','Color',purple,'Markersize',25);
xlabel('Performance');
ylabel('Identification Score');
% text(min(xlim)-0.05*range(xlim),max(ylim)-0.05*range(ylim),'e','HorizontalAlign','Right','VerticalAlign','Bottom','FontSize',fsz);


%
MS = 5;
selectSessions = [1 4 7];
for i_mouse = I_MOUSE
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


        pp2 = predPos(downsampledRunState==1)*200;
        ic2 = instCorr(:,downsampledRunState==1);
        cf2 = confidence(downsampledRunState==1);
        tt2 = downsampledTrialType(downsampledRunState==1);
        dp2 = (downsampledPosition(downsampledRunState==1) + (tt2-2))*200;
        t2 = timeBin*(1:length(pp2));
        
        P = panel_pos(1+NCOLS*(iii-1),:);
        P(2) = P(2)+P(4)*0.4;
        P(3) = P(3)*1.9;
        P(4) = P(4)*0.9;
        axes('position', P);

        
        start = t2(find(~isnan(pp2),1,'first'));        
        plot(t2-start, dp2, 'b.','Markersize',MS);
        hold on;
        plot(t2-start, pp2, 'r.','Markersize',MS);
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
        
        if(iii==2)
            ylabel('Position (cm)');
        end
        if(iii<3)
            set(gca,'xticklabel','');
        else
            xlabel('Running Time (s)');
        end
            
        xlim([0 270]);
        if(iii==1)
            P2 = [P(1)+P(3)-P(3)*0.03 P(2) P(3)*0.1 P(4)];
            axes('position', P2);
            plot([0 0],[0 0.5],'Color',[0 0 0.85],'Linewidth',5);
            text(1,0.2,'A','Fontsize',12,'Rotation',90);
            hold on;
            plot([0 0],[0.5 1],'Color',[1 0 0],'Linewidth',5);
            text(1,0.7,'B','Fontsize',12,'Rotation',90);
            axis off;
        else
            P2 = [P(1)+P(3)-P(3)*0.03 P(2) P(3)*0.1 P(4)];
            axes('position', P2);
            plot([0 0],[0 0.5],'Color',[0 0.5 0.85],'Linewidth',5);
            text(1,0.2,'A''','Fontsize',12,'Rotation',90);
            hold on;
            plot([0 0],[0.5 1],'Color',[1 0.5 0],'Linewidth',5);
            text(1,0.7,'B''','Fontsize',12,'Rotation',90);
            axis off;
        end
        
        P = panel_pos(4+NCOLS*(iii-1),:);
        P(1) = P(1)-P(3)*2;
        P(2) = P(2)+P(4)*0.4;
        P(3) = P(3)*3.25;
        P(4) = P(4)*0.9;
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
        set(gca,'position',p+[-0.02 0 0 0]);
        plot(dp2+randn(size(dp2)),pp2+randn(size(dp2)),'o','Color','k','Markersize',2);
        if(iii==2)
            ylabel('Decoded Position (cm)');
        end
        if(iii==3)
            xlabel('Actual Position (cm)');
        else
            set(gca,'xticklabel','');
        end
        if(iii==3)
            c.Position = c.Position + [-0.0125 0 0 0];
        end
        
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
        b2 = linspace(0,400,2*N_BINS+1);
        drawnow;
    end
    
end


%%
base = '\\research-cifs.nyumc.org\Research\basulab\basulabspace\RZ\data_for_Jason\rev\matching_binary_events_all_lap_learn';
% base = '\\research-cifs.nyumc.org\Research\basulab\basulabspace\RZ\data_for_Jason\matching_binary_events_all_lap_recall';
mice = dir(base);
mice = arrayfun(@(x) x.name, mice(3:end), 'UniformOutput', false);


totalMice = 3;
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

            smoothedEventRate = jmm_smooth_1d_cor(eventBinaryAll, 2*SM2)/timeBin;
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
% ERROR_L(3,2,:) = NaN;
% ERROR_L(3,:,2) = NaN;
% DECODING_L(3,2,:) = NaN;
% DECODING_L(3,:,2) = NaN;




%%
diags = -8:8;
totalMice = 3;
decoding_dayDecay_L = NaN(totalMice, length(diags));
error_dayDecay_L = NaN(totalMice, length(diags));

decoding_dayDecay2_L = NaN(totalMice, maxSessions);
error_dayDecay2_L = NaN(totalMice, maxSessions);

% decoding_dayDecay_R = NaN(totalMice, length(diags));
% error_dayDecay_R = NaN(totalMice, length(diags));
% 
% decoding_dayDecay2_R = NaN(totalMice, maxSessions);
% error_dayDecay2_R = NaN(totalMice, maxSessions);


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


CA1 = [0.32 0.99];
CA2 = [3.0 44.4];
% CA2 = -CA2(end:-1:1);


%%
colormap parula;
SCALE = 1.73;



H = 0.65;
P = panel_pos(22,:);
P(1) = P(1)+P(3)*0.5;
P(2) = P(2)-P(4)*0.05;
P(3:4) = P(3:4)*SCALE;
% P(3) = P(3)*1.2;
P(4) = P(4)*H;
axes('position', P);
imagesc(squeeze(nanmean(DECODING_L,1))');
caxis(CA1);
axis square;
colorbar;%('NorthOutside');
xlabel('Train Day');
y = ylabel('Decode Day');
y.Position = y.Position+[-0.25 0 0];
title('Identification Score');


P = panel_pos(24,:);
% P(1) = P(1)-P(3)*0.1;
P(2) = P(2)-P(4)*0.05;
P(3:4) = P(3:4)*SCALE;
P(4) = P(4)*H;
axes('position', P);
imagesc(squeeze(nanmean(ERROR_L,1))');
caxis(CA2);
% xlim([0.5 7.5]);
% ylim([0.5 7.5]);
axis square;
c = colorbar;%('NorthOutside');
% c.TickLabels = {'40','30','20','10'};
xlabel('Train Day');
y = ylabel('Decode Day');
y.Position = y.Position+[-0.25 0 0];
title('Decoding Error');


SCALE = 1.2;

c1 = 0.75*[1 1 1];
c2 = [1 0 1];
% clf;
P = panel_pos(34,:);
P(1) = P(1)+P(3)*0.5;
P(2) = P(2)+P(4)*0.05;
% P(1) = P(1)-P(3)*0.8;
% P(2) = P(2)+P(4)*0.7;
P(3:4) = P(3:4)*SCALE;
P(4) = P(4)*H;
axes('position', P);
plot(1:maxSessions,decoding_dayDecay2_L,'Color',c1);
hold on;
p = seplot(1:maxSessions,decoding_dayDecay2_L',c2,1);
p(2).LineStyle = '--';
p(3).LineStyle = '--';
% axis square;
jAXIS;
xlabel('Day Comparison');
ylabel('                                Identification Score');
set(gca,'xtick',1:maxSessions);
set(gca,'xticklabel',{'1 vs. 2','2 vs. 3','3 vs. 4','4 vs. 5','5 vs. 6','6 vs. 7','7 vs. 8','8 vs. 9'});
set(gca,'XTickLabelRotation',45);
ylim(CA1);

P = panel_pos(36,:);
% P(1) = P(1)-P(3)*0.8;
P(2) = P(2)+P(4)*0.05;
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
ylabel('                                 Decoding Error (cm)');
set(gca,'xtick',1:maxSessions);
set(gca,'xticklabel',{'1 vs. 2','2 vs. 3','3 vs. 4','4 vs. 5','5 vs. 6','6 vs. 7','7 vs. 8','8 vs. 9'});
set(gca,'XTickLabelRotation',45);
ylim([5 37]);
% set(gca,'ydir','reverse');

P = panel_pos(28,:);
P(1) = P(1)+P(3)*0.5;
% P(1) = P(1)-P(3)*0.4;
% P(2) = P(2)-P(4)*0.05;
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
ylim(CA1);
% ylabel('Identification Score');

P = panel_pos(30,:);
% P(1) = P(1)+P(3)*0.05;
% P(1) = P(1)-P(3)*0.4;
% P(2) = P(2)+P(4)*0.15;
% P(2) = P(2)-P(4)*0.05;
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
ylim([0 46]);
% set(gca,'ydir','reverse');
% ylabel('Decoding Error');




%%
figure_path_primary = '.';
capstr = '';
fig = gcf;
savefig(fullfile(figure_path_primary, 'RZ_Rebut_J4_v2'), fig, false , capstr, 600);
print(fig, fullfile(figure_path_primary, 'RZ_Rebut_J4_v2'), '-painters', '-dsvg');

