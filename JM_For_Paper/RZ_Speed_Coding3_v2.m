cleer;
colormap parula;
bigFont;

base = '\\research-cifs.nyumc.org\Research\basulab\basulabspace\RZ\data_for_Jason\matching_binary_events_all_lap_learn';
% base = '\\research-cifs.nyumc.org\Research\basulab\basulabspace\RZ\data_for_Jason\matching_binary_events_all_lap_recall';
mice = dir(base);
mice = arrayfun(@(x) x.name, mice(3:end-1), 'UniformOutput', false);

load('\\research-cifs.nyumc.org\Research\basulab\basulabspace\RZ\_revision_scripts_data_Jason\tuned_match_ROI_learn_recall_data.mat', 'short_term_learn');
load('\\research-cifs.nyumc.org\Research\basulab\basulabspace\RZ\_revision_scripts_data_Jason\tuned_match_ROI_learn_recall_data.mat', 'reg_learn');

totalMice = 6;
maxSessions = 9;

speedQuantiles = NaN(totalMice,maxSessions,3);
nCells = NaN(1,totalMice);
for i_mouse = 1:totalMice
    i_mouse
    load(fullfile(base,mice{i_mouse},'matched_transients_all_ses.mat'),'matching_ROI_bin_transient_lap_data');

    %%
    these = matching_ROI_bin_transient_lap_data;

    a = cell2mat(arrayfun(@(x) quantile(these{x}.speed(these{x}.run_state==1 & these{x}.speed>2),[0 0.5 1]),1:length(these),'UniformOutput', false)');
    speedQuantiles(i_mouse,1:size(a,1),:) = a;      
    nCells(i_mouse) = size(these{1}.transient_mat,1);
end
temp = reshape(speedQuantiles,[],size(speedQuantiles,3));
speedEdges = nanmean(temp);
speedEdges(1) = min(temp(:,1));
speedEdges(end) = max(temp(:,end));

%%
N_BINS = 40;
MAPS = NaN(nansum(nCells), maxSessions, N_BINS, 8);
N_EVENTS = NaN(nansum(nCells), maxSessions, 8);
TOTAL_TIME = NaN(nansum(nCells), maxSessions, 8);
ONLY_A = NaN(nansum(nCells), maxSessions);
ONLY_B = NaN(nansum(nCells), maxSessions);
ALL_A = NaN(nansum(nCells), maxSessions);
ALL_B = NaN(nansum(nCells), maxSessions);
AB = NaN(nansum(nCells), maxSessions);
NEITHER = NaN(nansum(nCells), maxSessions);

count = 1;
for i_mouse = 1:totalMice
    load(fullfile(base,mice{i_mouse},'matched_transients_all_ses.mat'),'matching_ROI_bin_transient_lap_data');
    these = matching_ROI_bin_transient_lap_data;
    for i_ses = 1:length(these)
        this = these{i_ses};

        edgesPosition = linspace(0,1,N_BINS+1);
%         edgesPosition2 = linspace(0,1,N_BINS+1);
        SM = 1;
        DT = nanmedian(diff(this.time));
        N_CELLS = size(this.transient_mat,1);
        
        [~,speedBin] = histc(this.speed,speedEdges);
%         ONLY_A(count:count+size(rate,1)-1,i_ses) = short_term_learn.tuned_log{}.tuned_logicals.tuned_log_filt_si{1}.onlyA
        
        VALID = {this.run_state==1 & this.trialType==2 & speedBin>0;
                this.run_state==1 & this.trialType==3 & speedBin>0;
                this.run_state==1 & speedBin==1;
                this.run_state==1 & speedBin==2;
                this.run_state==1 & this.trialType==2 & speedBin==1;
                this.run_state==1 & this.trialType==3 & speedBin==1;
                this.run_state==1 & this.trialType==2 & speedBin==2;
                this.run_state==1 & this.trialType==3 & speedBin==2};

        temp = reg_learn{i_mouse}.registered.multi.assigned_filtered(:,i_ses);
        bad = isnan(temp);
        temp(bad) = 1;
        
        temp2 = short_term_learn.tuned_log{i_mouse}.tuned_logicals.tuned_log_filt_si{i_ses};        
        
        temp3 = double(temp2.onlyA(temp));
        temp3(bad) = NaN;
        ONLY_A(count:count+length(temp)-1, i_ses) = temp3;
        
        temp3 = double(temp2.onlyB(temp));
        temp3(bad) = NaN;
        ONLY_B(count:count+length(temp)-1, i_ses) = temp3;
        
        temp3 = double(temp2.allA(temp));
        temp3(bad) = NaN;
        ALL_A(count:count+length(temp)-1, i_ses) = temp3;
        
        temp3 = double(temp2.allB(temp));
        temp3(bad) = NaN;
        ALL_B(count:count+length(temp)-1, i_ses) = temp3;
        
        temp3 = double(temp2.AB(temp));
        temp3(bad) = NaN;
        AB(count:count+length(temp)-1, i_ses) = temp3;
        
        temp3 = double(temp2.neither(temp));
        temp3(bad) = NaN;
        NEITHER(count:count+length(temp)-1, i_ses) = temp3;
        
        for i_valid = 1:length(VALID)
            occ = DT*histcn(this.position_norm(VALID{i_valid}),edgesPosition);
            TOTAL_TIME(count:count+length(temp)-1,i_ses,i_valid) = nansum(occ);
            occ = gaussian_smooth_1d_circ(occ,SM);
            occ = occ(1:N_BINS);
            
            events = NaN(N_CELLS,N_BINS);
            for i_cell = 1:N_CELLS
                eventBinary = (this.transient_mat(i_cell,:).*(VALID{i_valid})'==1);
                eventPos = this.position_norm(eventBinary);
                event = histcn(eventPos,edgesPosition);
                N_EVENTS(count+i_cell-1,i_ses,i_valid) = length(eventPos);
                event = gaussian_smooth_1d_circ(event,SM);
                event = event(1:N_BINS);
                events(i_cell,:) = event;
            end
            rate = bsxfun(@rdivide, events, occ);
            MAPS(count:count+size(rate,1)-1,i_ses,:,i_valid) = rate;
        end                      
        
    end
    count = count + size(rate,1);
end





%%
clf;
subpanels = 'G HI';
NCOLS = 5;
NROWS = 1;

% WIDTH = 6.722*2.54;
% HEIGHT = 3.693*2.54;

WIDTH = 10*2.54;
HEIGHT = 2.8*2.54;
set(0,'defaultaxesfontsize',14)
close all;
[panel_pos, fig, all_ax, HH] = panelFigureSetup2( NCOLS, NROWS, subpanels,WIDTH,HEIGHT, 0.5, 1.5);

FSZ = 20;
HH(1).Position = HH(1).Position + [-0.02 0.06 0];
HH(1).FontSize = FSZ;
HH(3).Position = HH(3).Position + [0.015 0.06 0];
HH(3).FontSize = FSZ;
HH(4).Position = HH(4).Position + [0.03 0.06 0];
HH(4).FontSize = FSZ;
% HH(5).Position = HH(5).Position + [-0.02 -0.06 0];
% HH(5).FontSize = FSZ;
% HH(7).Position = HH(7).Position + [-0.02 -0.06 0];
% HH(7).FontSize = FSZ;
% HH(11).Position = HH(11).Position + [-0.02 -0.18 0];
% HH(11).FontSize = FSZ;


i_ses = 6;
colors = {'b',[0.25 0.75 1],'r','m'};

MS1 = 8;
LW = 2;
MS2 = 12;
MS3 = 6;
valid = ONLY_A(:,i_ses)==1;
rate_A_slow = squeeze(N_EVENTS(valid,i_ses,5)./TOTAL_TIME(valid,i_ses,5));
rate_A_fast = squeeze(N_EVENTS(valid,i_ses,7)./TOTAL_TIME(valid,i_ses,7));
rate_B_slow = squeeze(N_EVENTS(valid,i_ses,6)./TOTAL_TIME(valid,i_ses,6));
rate_B_fast = squeeze(N_EVENTS(valid,i_ses,8)./TOTAL_TIME(valid,i_ses,8));

H = 1.2;
W = 1;
SHIFT_U = -0.1;
P = panel_pos(1,:);
P(2) = P(2)+P(4)*SHIFT_U;
P(3) = P(3)*W;
P(4) = P(4)*H;
axes('position', P);
[V1, M1, CI1, P1] = violinStats({60*rate_A_slow,60*rate_A_fast,60*rate_B_slow,60*rate_B_fast}, MS1, colors, LW, @(x,y) signrank(x,y));
N1 = arrayfun(@(x) sum(~isnan(x.ScatterPlot.YData)), V1);
set(gca,'xticklabel',{'A_s_l_o_w','A_f_a_s_t','B_s_l_o_w','B_f_a_s_t'});
ylabel('Events/min');
title('A-Selective');
xlim([0.5 4.5]);
ylim([0.1 15]);
% title(sprintf('A Only Selective Neurons: Learning Session %d', i_ses));

% P = panel_pos(3,:);
% P(1) = P(1)+P(3)*0.1;
% P(2) = P(2)+P(4)*0.3;
% P(4) = P(4)*0.95;
% axes('position', P);
% plot(60*rate_A_slow,60*rate_B_slow,'.','Color',[0 0.7 0],'Markersize',MS2);
% xlim([0.01 11]);
% ylim([0.01 11]);
% makeItSquare;
% xlabel('Events/min: A_s_l_o_w');
% ylabel('Events/min: B_s_l_o_w');
% 
% P = panel_pos(4,:);
% P(1) = P(1)+P(3)*0.2;
% P(2) = P(2)+P(4)*0.3;
% P(4) = P(4)*0.95;
% axes('position', P);
% plot(60*rate_A_fast,60*rate_B_fast,'.','Color',[0.5 0.95 0.5],'Markersize',MS2);
% xlim([0.01 22]);
% ylim([0.01 22]);
% makeItSquare;
% xlabel('Events/min: A_f_a_s_t');
% ylabel('Events/min: B_f_a_s_t');


valid = ONLY_B(:,i_ses)==1;
rate_A_slow = squeeze(N_EVENTS(valid,i_ses,5)./TOTAL_TIME(valid,i_ses,5));
rate_A_fast = squeeze(N_EVENTS(valid,i_ses,7)./TOTAL_TIME(valid,i_ses,7));
rate_B_slow = squeeze(N_EVENTS(valid,i_ses,6)./TOTAL_TIME(valid,i_ses,6));
rate_B_fast = squeeze(N_EVENTS(valid,i_ses,8)./TOTAL_TIME(valid,i_ses,8));

P = panel_pos(2,:);
P(1) = P(1)+P(3)*0;
P(2) = P(2)+P(4)*SHIFT_U;
P(3) = P(3)*W;
P(4) = P(4)*H;
axes('position', P);
[V2, M2, CI2, P2] = violinStats({60*rate_A_slow,60*rate_A_fast,60*rate_B_slow,60*rate_B_fast}, MS1, colors, LW, @(x,y) signrank(x,y));
N2 = arrayfun(@(x) sum(~isnan(x.ScatterPlot.YData)), V2);
set(gca,'xticklabel',{'A_s_l_o_w','A_f_a_s_t','B_s_l_o_w','B_f_a_s_t'});
% ylabel('Events/min');
title('B-Selective');
% title(sprintf('B Only Selective Neurons: Learning Session %d', i_ses));
xlim([0.5 4.5]);
ylim([0.1 15]);

% P = panel_pos(7,:);
% P(1) = P(1)+P(3)*0.1;
% P(4) = P(4)*0.95;
% axes('position', P);
% plot(60*rate_A_slow,60*rate_B_slow,'.','Color',[0 0.7 0],'Markersize',MS2);
% xlim([0.01 11]);
% ylim([0.01 11]);
% makeItSquare;
% xlabel('Events/min: A_s_l_o_w');
% ylabel('Events/min: B_s_l_o_w');
% 
% P = panel_pos(8,:);
% P(1) = P(1)+P(3)*0.2;
% P(4) = P(4)*0.95;
% axes('position', P);
% plot(60*rate_A_fast,60*rate_B_fast,'.','Color',[0.5 0.95 0.5],'Markersize',MS2);
% xlim([0.01 22]);
% ylim([0.01 22]);
% makeItSquare;
% xlabel('Events/min: A_f_a_s_t');
% ylabel('Events/min: B_f_a_s_t');



%
comparisons = [ 1 2;
                3 4;
                5 7;
                6 8;
                5 6;
                7 8];
colors = {[0.75 0 0.75],0.5*[1 1 1],'b','r',[0 0.5 0],[0.25 0.85 0.25]};
i_ses = 6;        
CCC = NaN(size(MAPS,1),size(comparisons,1));
for i_comp = 1:size(comparisons,1)
    A = squeeze(MAPS(:,i_ses,:,comparisons(i_comp,1)));
    B = squeeze(MAPS(:,i_ses,:,comparisons(i_comp,2)));
    
    C = restricted_corrcoef(A,B);
    d = diag(C);
    d(d==0) = NaN;
    CCC(:,i_comp) = d;
end

H = 0.5;
W = 1.8;

% P = panel_pos(11,:);
% P(1) = P(1)-P(3)*0.55;
% P(2) = P(2)-P(4)*0.6;
% P(3) = P(3)*W;
% P(4) = P(4)*H;
% axes('position', P);
% [V, M, CI, P] = violinStats({CCC(:,1),CCC(:,2),CCC(:,3),CCC(:,4),CCC(:,5),CCC(:,6)}, 10, colors, 2);
% set(gca,'xticklabel',{'A-B', 'S-F', 'A_S-A_F', 'B_S-B_F', 'A_S-B_S', 'A_F-B_F'});
% ylim([-1.2 1.2]);
% ylabel('Correlation Score');
% title('All Neurons');
% % title(sprintf('All Neurons - Learning Session %d',i_ses));
% XL = xlim;
% 
% include = CCC(:,1)<0.3;
% P = panel_pos(12,:);
% P(1) = P(1)-P(3)*0.25;
% P(2) = P(2)-P(4)*0.6;
% P(3) = P(3)*W;
% P(4) = P(4)*H;
% axes('position', P);
% [V, M, CI, P] = violinStats({CCC(include,1),CCC(include,2),CCC(include,3),CCC(include,4),CCC(include,5),CCC(include,6)}, 10, colors, 2);
% set(gca,'xticklabel',{'A-B', 'S-F', 'A_S-A_F', 'B_S-B_F', 'A_S-B_S', 'A_F-B_F'});
% ylim([-1.2 1.2]);
% % ylabel('Correlation Score');
% set(gca,'yticklabel','');
% title('Low A-B Correlation');
% plot([0.8 1.2],0.3*[1 1],'r-','Linewidth',4);

include = AB(:,i_ses)==1;
P = panel_pos(4,:);
P(1) = P(1)+P(3)*0.5;
P(2) = P(2)+P(4)*0.66;
P(3) = P(3)*W;
P(4) = P(4)*H;
axes('position', P);
[V3, M3, CI3, P3] = violinStats({CCC(include,1),CCC(include,2),CCC(include,3),CCC(include,4),CCC(include,5),CCC(include,6)}, MS3, colors, 2, @(x,y) signrank(x,y));
N3 = arrayfun(@(x) sum(~isnan(x.ScatterPlot.YData)), V3);
set(gca,'xticklabel',{'A-B', 's-f', 'A_s-A_f', 'B_s-B_f', 'A_s-B_s', 'A_f-B_f'});
ylim([-1.2 1.2]);
ylabel('Correlation');
title('A and B Selective Neurons');
xlim([0.35 6.3]);
set(gca,'ytick',-1:0.25:1);
set(gca,'yticklabel',{'-1','','0.5','','0','','0.5','','1'});

include = AB(:,i_ses)==1 & CCC(:,1)<0.3;
P = panel_pos(4,:);
P(1) = P(1)+P(3)*0.5;
P(2) = P(2)-P(4)*0.15;
P(3) = P(3)*W;
P(4) = P(4)*H;
axes('position', P);
[V4, M4, CI4, P4] = violinStats({CCC(include,1),CCC(include,2),CCC(include,3),CCC(include,4),CCC(include,5),CCC(include,6)}, MS3, colors, 2, @(x,y) signrank(x,y));
N4 = arrayfun(@(x) sum(~isnan(x.ScatterPlot.YData)), V4);
set(gca,'xticklabel',{'A-B', 's-f', 'A_s-A_f', 'B_s-B_f', 'A_s-B_s', 'A_f-B_f'});
ylim([-1.2 1.2]);
ylabel('Correlation');
% set(gca,'yticklabel','');
title('Low A-B Correlation');
plot([0.8 1.2],0.3*[1 1],'r-','Linewidth',4);
xlim([0.35 6.3]);
set(gca,'ytick',-1:0.25:1);
set(gca,'yticklabel',{'-1','','0.5','','0','','0.5','','1'});

%
% clf;
temp = find(AB(:,i_ses)==1);
i_cell = temp(203);

H = 0.4;
W = 0.9;
x = 200*center(edgesPosition);
LW = 2;
P = panel_pos(3,:);
P(1) = P(1)+P(3)*0.2;
P(2) = P(2)+P(4)*0.81;
P(3) = P(3)*W;
P(4) = P(4)*H;
axes('position', P);
plot(x,60*squeeze(MAPS(i_cell,i_ses,:,1)),'b','Linewidth',LW);
hold on;
plot(x,60*squeeze(MAPS(i_cell,i_ses,:,2)),'r','Linewidth',LW);
% ylabel('Events/minute');
% title('All Speeds');
% title(sprintf('All Speeds Included\nSample Neuron %d',i_cell));
set(gca,'xticklabel','');
ylim([0 64]);
set(gca,'ytick',[0 20 40]);
plot([160 175]+5,45*[1 1],'b','Linewidth',LW);
plot([160 175]+5,30*[1 1],'r','Linewidth',LW);
text(177+5,45,'A','Fontsize',10);
text(177+5,30,'B','Fontsize',10);
% legend('A','B');
xlim([0.1 199.9]);
text(mean(xlim),max(ylim),'All Speeds','HorizontalAlign','Center','VerticalAlign','Top','Fontsize',12);

P = panel_pos(3,:);
P(1) = P(1)+P(3)*0.2;
P(2) = P(2)+P(4)*0.38;
P(3) = P(3)*W;
P(4) = P(4)*H;
axes('position', P);
plot(x,60*squeeze(MAPS(i_cell,i_ses,:,5)),'Color',[0 0 0.75],'Linewidth',LW);
hold on;
plot(x,60*squeeze(MAPS(i_cell,i_ses,:,6)),'Color',[0.75 0 0],'Linewidth',LW);
plot([160 175]+5,45*[1 1],'Color',[0 0 0.75],'Linewidth',LW);
plot([160 175]+5,30*[1 1],'Color',[0.75 0 0],'Linewidth',LW);
text(177+5,45,'A','Fontsize',10);
text(177+5,30,'B','Fontsize',10);
ylabel('Events/min');
% title('Slow Speed');
set(gca,'xticklabel','');
ylim([0 64]);
set(gca,'ytick',[0 20 40]);
% legend('A','B');
xlim([0.1 199.9]);
text(mean(xlim),max(ylim),'Slow Speed','HorizontalAlign','Center','VerticalAlign','Top','Fontsize',12);

P = panel_pos(3,:);
P(1) = P(1)+P(3)*0.2;
P(2) = P(2)-P(4)*0.05;
P(3) = P(3)*W;
P(4) = P(4)*H;
axes('position', P);
plot(x,60*squeeze(MAPS(i_cell,i_ses,:,7)),'Color',[0.5 0.5 1],'Linewidth',LW);
hold on;
plot(x,60*squeeze(MAPS(i_cell,i_ses,:,8)),'Color',[1 0.5 0.5],'Linewidth',LW);
plot([160 175]+5,45*[1 1],'Color',[0.5 0.5 1],'Linewidth',LW);
plot([160 175]+5,30*[1 1],'Color',[1 0.5 0.5],'Linewidth',LW);
text(177+5,45,'A','Fontsize',10);
text(177+5,30,'B','Fontsize',10);
xlabel('Position (cm)');
% ylabel('Events/minute');
% title('Fast Speed');
ylim([0 64]);
set(gca,'ytick',[0 20 40]);
% legend('A','B');
xlim([0.1 199.9]);
text(mean(xlim),max(ylim),'Fast Speed','HorizontalAlign','Center','VerticalAlign','Top','Fontsize',12);



%%
figure_path_primary = '.';
capstr = '';
fig = gcf;
savefig(fullfile(figure_path_primary, 'RZ_Rebut_J1_v3'), fig, false , capstr, 600);
print(fig, fullfile(figure_path_primary, 'RZ_Rebut_J1_v3'), '-painters', '-dsvg');

