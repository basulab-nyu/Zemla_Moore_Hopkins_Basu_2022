function [outputArg1,outputArg2] = PV_TC_correlation_plotter(corr_analysis)

%% Export the structures

st_learn = corr_analysis.st_learn;
st_recall = corr_analysis.st_recall;
lt_recall = corr_analysis.lt_recall;


%% PV and TC (ts) by animal line plots for figures for Figure 4F

st_learn_day_range = 1:7;
st_recall_day_range = 1:9;

figure('Position',[2272 370 1007 420])
%PV plot - all neurons
subplot(1,2,1)
hold on 
axis square
title('PV correlation - all')
ylim([0 1])
xlim([0 10])
xticks(1:9)
yticks(0:0.2:1)
xlabel('Days')
ylabel('Correlation score')

%Learn
lA = errorbar(st_learn_day_range,st_learn.mean_PV.A(st_learn_day_range),st_learn.sem_PV.A(st_learn_day_range),'LineStyle','--','Linewidth',2,'Color',[65,105,225]/255);
lB = errorbar(st_learn_day_range,st_learn.mean_PV.B(st_learn_day_range),st_learn.sem_PV.B(st_learn_day_range),'LineStyle','--','Linewidth',2,'Color',[220,20,60]/255);

%Recall
rA = errorbar(st_recall_day_range,st_recall.mean_PV.all_corr.A(st_recall_day_range),st_recall.sem_PV.all_corr.A(st_recall_day_range),'LineStyle','-','Linewidth',2,'Color',[65,105,225]/255);
rB = errorbar(st_recall_day_range,st_recall.mean_PV.all_corr.B(st_recall_day_range),st_recall.sem_PV.all_corr.B(st_recall_day_range),'LineStyle','-','Linewidth',2,'Color',[220,20,60]/255);

set(gca,'FontSize',16)
set(gca,'Linewidth',2)

legend([lA,lB,rA,rB],{'Learning A','Learning B','Recall A','Recall B'},'location','northeast')

subplot(1,2,2)
hold on 
axis square
title('TC correlation - ts')
ylim([0 1])
xlim([0 10])
xticks(1:9)
yticks(0:0.2:1)
xlabel('Days')
ylabel('Correlation score')

 
%Learn
lA = errorbar(st_learn_day_range,st_learn.ts.mean_TC.animal.A(st_learn_day_range),st_learn.ts.sem_TC.animal.A(st_learn_day_range),'LineStyle','--','Linewidth',2,'Color',[65,105,225]/255);
lB = errorbar(st_learn_day_range,st_learn.ts.mean_TC.animal.B(st_learn_day_range),st_learn.ts.sem_TC.animal.B(st_learn_day_range),'LineStyle','--','Linewidth',2,'Color',[220,20,60]/255);

%Recall
rA = errorbar(st_recall_day_range,st_recall.ts.mean_TC.all_corr.A(st_recall_day_range),st_recall.ts.sem_TC.all_corr.A(st_recall_day_range),'LineStyle','-','Linewidth',2,'Color',[65,105,225]/255);
rB = errorbar(st_recall_day_range,st_recall.ts.mean_TC.all_corr.B(st_recall_day_range),st_recall.ts.sem_TC.all_corr.B(st_recall_day_range),'LineStyle','-','Linewidth',2,'Color',[220,20,60]/255);

set(gca,'FontSize',16)
set(gca,'Linewidth',2)

legend([lA,lB,rA,rB],{'Learning A','Learning B','Recall A','Recall B'},'location','northeast')

%% Organize data for input into plotter comparison script - NOT COMPLETE; NOT USED
%organize data for plotting
%ST learn data (only raw data; no fill ins were necessary)
% plot_inputs.A.PV.st_learn.range = [1 2 3 6 7 8 9];
% plot_inputs.A.PV.st_learn.mean = st_learn.mean_PV.A([1 2 3 6 7 8 9]);
% plot_inputs.A.PV.st_learn.sem = st_learn.sem_PV.A(([1 2 3 6 7 8 9]));
% 
% %ST recall with raw data (excluding D4/D5 that have substitutions for
% %equivalent 3 vs. 6 and 3 vs. 7)
% plot_inputs.A.PV.st_recall.d4_d5.range = [1 2 3 6 7 8 9];
% plot_inputs.A.PV.st_recall.d4_d5.mean = st_recall.mean_PV.A([1 2 3 6 7 8 9]);
% plot_inputs.A.PV.st_recall.d4_d5.sem = st_recall.sem_PV.A(([1 2 3 6 7 8 9]));


%% Organize PV data for plotting
%range, mean, sem (rows)
%A
PV_mean_sem.st_learn.A = [[1 2 3 4 5 6 7];st_learn.mean_PV.A([1 2 3 4 5 6 7]); st_learn.sem_PV.A(([1 2 3 4 5 6 7]))];
%B
PV_mean_sem.st_learn.B = [[1 2 3 4 5 6 7];st_learn.mean_PV.B([1 2 3 4 5 6 7]); st_learn.sem_PV.B(([1 2 3 4 5 6 7]))];

%raw (with substituted D4 and D5)
%A
PV_mean_sem.st_recall.d4_d5.A = [[1 2 3 6 7 8 9];st_recall.mean_PV.A([1 2 3 6 7 8 9]); st_recall.sem_PV.A(([1 2 3 6 7 8 9]))];
%B
PV_mean_sem.st_recall.d4_d5.B = [[1 2 3 6 7 8 9];st_recall.mean_PV.B([1 2 3 6 7 8 9]); st_recall.sem_PV.B(([1 2 3 6 7 8 9]))];

%all corr (all days)
%A
PV_mean_sem.st_recall.all_corr.A = [[1 2 3 4 5 6 7 8 9];st_recall.mean_PV.all_corr.A([1 2 3 4 5 6 7 8 9]); st_recall.sem_PV.all_corr.A(([1 2 3 4 5 6 7 8 9]))];
%B
PV_mean_sem.st_recall.all_corr.B = [[1 2 3 4 5 6 7 8 9];st_recall.mean_PV.all_corr.B([1 2 3 4 5 6 7 8 9]); st_recall.sem_PV.all_corr.B(([1 2 3 4 5 6 7 8 9]))];

%% Organize Neighboring PV data for plotting

%short term learn
%A
PV_mean_sem.neighbor.st_learn.A = [[1 2 3 4 5 6 7];st_learn.neighbor.mean_PV.A([1 2 3 4 5 6 7]); st_learn.neighbor.sem_PV.A(([1 2 3 4 5 6 7]))];
%B
PV_mean_sem.neighbor.st_learn.B = [[1 2 3 4 5 6 7];st_learn.neighbor.mean_PV.B([1 2 3 4 5 6 7]); st_learn.neighbor.sem_PV.B(([1 2 3 4 5 6 7]))];

%short term recall
%A
PV_mean_sem.neighbor.st_recall.A = [[1 2 6 7 8];st_recall.neighbor.mean_PV.A([1 2 6 7 8]); st_recall.neighbor.sem_PV.A(([1 2 6 7 8]))];
%B
PV_mean_sem.neighbor.st_recall.B = [[1 2 6 7 8];st_recall.neighbor.mean_PV.B([1 2 6 7 8]); st_recall.neighbor.sem_PV.B(([1 2 6 7 8]))];

%long term recall - ADD THIS


%% Organize TC data for plotting
%range, mean, sem (rows)
%TS learn A/B
%A
TC_mean_sem.ts.st_learn.A = [[1 2 3 4 5 6 7];st_learn.ts.mean_TC.animal.A([1 2 3 4 5 6 7]); st_learn.ts.sem_TC.animal.A(([1 2 3 4 5 6 7]))];
%B
TC_mean_sem.ts.st_learn.B = [[1 2 3 4 5 6 7];st_learn.ts.mean_TC.animal.B([1 2 3 4 5 6 7]); st_learn.ts.sem_TC.animal.B(([1 2 3 4 5 6 7]))];

%SI learn A/B
%A
TC_mean_sem.si.st_learn.A = [[1 2 3 4 5 6 7];st_learn.si.mean_TC.animal.A([1 2 3 4 5 6 7]); st_learn.si.sem_TC.animal.A(([1 2 3 4 5 6 7]))];
%B
TC_mean_sem.si.st_learn.B = [[1 2 3 4 5 6 7];st_learn.si.mean_TC.animal.B([1 2 3 4 5 6 7]); st_learn.si.sem_TC.animal.B(([1 2 3 4 5 6 7]))];

%TS recall A/B (raw) without D4 and D5
%A
TC_mean_sem.ts.st_recall.d4_d5.A = [[1 2 3 6 7 8 9]; st_recall.ts.mean_TC.animal.A([1 2 3 6 7 8 9]); st_recall.ts.sem_TC.animal.A(([1 2 3 6 7 8 9]))];
%B
TC_mean_sem.ts.st_recall.d4_d5.B = [[1 2 3 6 7 8 9]; st_recall.ts.mean_TC.animal.B([1 2 3 6 7 8 9]); st_recall.ts.sem_TC.animal.B(([1 2 3 6 7 8 9]))];

%SI recall A/B (raw) without D4 and D5
%A
TC_mean_sem.si.st_recall.d4_d5.A = [[1 2 3 6 7 8 9]; st_recall.si.mean_TC.animal.A([1 2 3 6 7 8 9]); st_recall.si.sem_TC.animal.A(([1 2 3 6 7 8 9]))];
%B
TC_mean_sem.si.st_recall.d4_d5.B = [[1 2 3 6 7 8 9]; st_recall.si.mean_TC.animal.B([1 2 3 6 7 8 9]); st_recall.si.sem_TC.animal.B(([1 2 3 6 7 8 9]))];


%TS recall A/B all corr (all days including D4 and D5)
%A
TC_mean_sem.ts.st_recall.all_corr.A = [[1:9]; st_recall.ts.mean_TC.all_corr.A([1:9]); st_recall.ts.sem_TC.all_corr.A(([1:9]))];
%B
TC_mean_sem.ts.st_recall.all_corr.B = [[1:9]; st_recall.ts.mean_TC.all_corr.B([1:9]); st_recall.ts.sem_TC.all_corr.B(([1:9]))];

%A
TC_mean_sem.si.st_recall.all_corr.A = [[1:9]; st_recall.si.mean_TC.all_corr.A([1:9]); st_recall.si.sem_TC.all_corr.A(([1:9]))];
%B
TC_mean_sem.si.st_recall.all_corr.B = [[1:9]; st_recall.si.mean_TC.all_corr.B([1:9]); st_recall.si.sem_TC.all_corr.B(([1:9]))];

%% Organize NEIGHBORING TC data for plotting
%range, mean, sem (rows)
%TS learn A/B
%A
TC_mean_sem.neighbor.ts.st_learn.A = [[1 2 3 4 5 6 7];st_learn.neighbor.ts.mean_TC.animal.A([1 2 3 4 5 6 7]); st_learn.neighbor.ts.sem_TC.animal.A(([1 2 3 4 5 6 7]))];
%B
TC_mean_sem.neighbor.ts.st_learn.B = [[1 2 3 4 5 6 7];st_learn.neighbor.ts.mean_TC.animal.B([1 2 3 4 5 6 7]); st_learn.neighbor.ts.sem_TC.animal.B(([1 2 3 4 5 6 7]))];

%SI learn A/B
%A
TC_mean_sem.neighbor.si.st_learn.A = [[1 2 3 4 5 6 7];st_learn.neighbor.si.mean_TC.animal.A([1 2 3 4 5 6 7]); st_learn.neighbor.si.sem_TC.animal.A(([1 2 3 4 5 6 7]))];
%B
TC_mean_sem.neighbor.si.st_learn.B = [[1 2 3 4 5 6 7];st_learn.neighbor.si.mean_TC.animal.B([1 2 3 4 5 6 7]); st_learn.neighbor.si.sem_TC.animal.B(([1 2 3 4 5 6 7]))];

%TS recall A/B
%A
TC_mean_sem.neighbor.ts.st_recall.A = [[1 2 6 7 8]; st_recall.neighbor.ts.mean_TC.animal.A([1 2 6 7 8]); st_recall.neighbor.ts.sem_TC.animal.A(([1 2 6 7 8]))];
%B
TC_mean_sem.neighbor.ts.st_recall.B = [[1 2 6 7 8]; st_recall.neighbor.ts.mean_TC.animal.B([1 2 6 7 8]); st_recall.neighbor.ts.sem_TC.animal.B(([1 2 6 7 8]))];

%SI recall A/B
%A
TC_mean_sem.neighbor.si.st_recall.A = [[1 2 6 7 8]; st_recall.neighbor.si.mean_TC.animal.A([1 2 6 7 8]); st_recall.neighbor.si.sem_TC.animal.A(([1 2 6 7 8]))];
%B
TC_mean_sem.neighbor.si.st_recall.B = [[1 2 6 7 8]; st_recall.neighbor.si.mean_TC.animal.B([1 2 6 7 8]); st_recall.neighbor.si.sem_TC.animal.B(([1 2 6 7 8]))];

%% Plot neighboring correlations for replacement of Figure 4G
figure('Position', [2272 370 1007 420])
%learn vs raw PV
subplot(1,2,1)
hold on
ylim([0 1])
xlabel('Day comparison')
ylabel('Correlation score')
xticks([1:8])
xlim([0 9])
xticklabels({'1 vs. 2','2 vs. 3','3 vs. 4','4 vs. 5','5 vs. 6','6 vs. 7','7 vs. 8','8 vs. 9'})
xtickangle(45)
title('PV correlation')
%learn
lA = plot_error_line(PV_mean_sem.neighbor.st_learn.A(:,1:6),'--',2,[65,105,225]/255);
lB = plot_error_line(PV_mean_sem.neighbor.st_learn.B(:,1:6),'--',2,[220,20,60]/255);
%recall
rA = plot_error_line(PV_mean_sem.neighbor.st_recall.A,'-',2,[65,105,225]/255);
rB = plot_error_line(PV_mean_sem.neighbor.st_recall.B,'-',2,[220,20,60]/255);

set(gca,'FontSize',16)
set(gca,'Linewidth',2)

legend([lA,lB,rA,rB],{'Learning A','Learning B','Recall A','Recall B'},'location','northeast')

subplot(1,2,2)
hold on
xlim([0 9])
ylim([0 1])
xlabel('Day comparison')
ylabel('Correlation score')
xticks([1:8])
xlim([0 9])
xticklabels({'1 vs. 2','2 vs. 3','3 vs. 4','4 vs. 5','5 vs. 6','6 vs. 7','7 vs. 8','8 vs. 9'})
xtickangle(45)
title('TC correlation - TS')
%learn
lA = plot_error_line(TC_mean_sem.neighbor.ts.st_learn.A(:,1:6),'--',2,[65,105,225]/255);
lB = plot_error_line(TC_mean_sem.neighbor.ts.st_learn.B(:,1:6),'--',2,[220,20,60]/255);
%recall
rA = plot_error_line(TC_mean_sem.neighbor.ts.st_recall.A,'-',2,[65,105,225]/255);
rB = plot_error_line(TC_mean_sem.neighbor.ts.st_recall.B,'-',2,[220,20,60]/255);

set(gca,'FontSize',16)
set(gca,'Linewidth',2)

legend([lA,lB,rA,rB],{'Learning A','Learning B','Recall A','Recall B'},'location','northeast')

set(gca,'FontSize',16)
set(gca,'Linewidth',2)


%% Plot neighboring PV with for ST learning
figure('Position', [2283 474 923 358])
%learn vs raw PV
subplot(1,2,1)
hold on
ylim([0 1])
xlabel('Day comparison')
ylabel('Corr. score')
xticks([1:6])
xlim([0 7])
xticklabels({'1 vs. 2','2 vs. 3','3 vs. 4','4 vs. 5','5 vs. 6','6 vs. 7'})
xtickangle(45)
title('PV correlation - learning')
%learn
lA = plot_error_line(PV_mean_sem.neighbor.st_learn.A(:,1:6),'--',2,[65,105,225]/255);
lB = plot_error_line(PV_mean_sem.neighbor.st_learn.B(:,1:6),'--',2,[220,20,60]/255);

set(gca,'FontSize',16)
set(gca,'Linewidth',2)

legend([lA,lB],{'Learning A','Learning B'},'location','northeast')

subplot(1,2,2)
hold on
xlim([0 9])
ylim([0 1])
xlabel('Day comparison')
xticks([1 2 6 7 8])
xticklabels({'1 vs. 2','2 vs. 3','6 vs. 7','7 vs. 8','8 vs. 9'})
xtickangle(45)
title('PV correlation - recall')
%recall
rA = plot_error_line(PV_mean_sem.neighbor.st_recall.A,'-',2,[65,105,225]/255);
rB = plot_error_line(PV_mean_sem.neighbor.st_recall.B,'-',2,[220,20,60]/255);

legend([rA,rB],{'Recall A','Recall B'},'location','northeast')

set(gca,'FontSize',16)
set(gca,'Linewidth',2)


%% Plot PV with raw results and cross correlated data (ST RECALL)
figure('Position', [2283 474 923 358])
%learn vs raw PV
subplot(1,2,1)
hold on
xlabel('Relative Day to D1')
ylabel('Corr. score')
xticks([1:9])
xticklabels({'0','1','2','3','4','5','6','7','8'})
title('PV correlation - raw')
%learn
lA = plot_error_line(PV_mean_sem.st_learn.A,'--',2,[65,105,225]/255);
lB = plot_error_line(PV_mean_sem.st_learn.B,'--',2,[220,20,60]/255);

%recall
rA = plot_error_line(PV_mean_sem.st_recall.d4_d5.A,'-',2,[65,105,225]/255);
rB = plot_error_line(PV_mean_sem.st_recall.d4_d5.B,'-',2,[220,20,60]/255);

legend([lA,lB,rA,rB],{'Learning A','Learning B','Recall A','Recall B'},'location','northeast')

set(gca,'FontSize',16)
set(gca,'Linewidth',2)

%learn vs cross-correlated ST RECALL PV
subplot(1,2,2)
hold on
xlabel('Relative Day to D1')
ylabel('Corr. score')
xticks([1:9])
xticklabels({'0','1','2','3','4','5','6','7','8'})
title('PV correlation - cross correlation')
%learn
lA = plot_error_line(PV_mean_sem.st_learn.A,'--',2,[65,105,225]/255);
lB = plot_error_line(PV_mean_sem.st_learn.B,'--',2,[220,20,60]/255);

%recall
rA = plot_error_line(PV_mean_sem.st_recall.all_corr.A,'-',2,[65,105,225]/255);
rB = plot_error_line(PV_mean_sem.st_recall.all_corr.B,'-',2,[220,20,60]/255);

legend([lA,lB,rA,rB],{'Learning A','Learning B','Recall A','Recall B'},'location','northeast')

set(gca,'FontSize',16)
set(gca,'Linewidth',2)

%% Plot TC with raw results and cross correlated data (ST RECALL)

%TC TS
figure('Position', [2050 107 829 824])
%learn vs raw PV
subplot(2,2,1)
hold on
xlabel('Relative Day to D1')
ylabel('Corr. score')
xticks([1:9])
xticklabels({'0','1','2','3','4','5','6','7','8'})
title('TC TS correlation - raw')
%learn
lA = plot_error_line(TC_mean_sem.ts.st_learn.A,'--',2,[65,105,225]/255);
lB = plot_error_line(TC_mean_sem.ts.st_learn.B,'--',2,[220,20,60]/255);

%recall
rA = plot_error_line(TC_mean_sem.ts.st_recall.d4_d5.A,'-',2,[65,105,225]/255);
rB = plot_error_line(TC_mean_sem.ts.st_recall.d4_d5.B,'-',2,[220,20,60]/255);

legend([lA,lB,rA,rB],{'Learning A','Learning B','Recall A','Recall B'},'location','northeast')

set(gca,'FontSize',16)
set(gca,'Linewidth',2)

%learn vs cross-correlated ST RECALL TC
subplot(2,2,2)
hold on
xlabel('Relative Day to D1')
ylabel('Corr. score')
xticks([1:9])
xticklabels({'0','1','2','3','4','5','6','7','8'})
title('TC TS correlation - cross correlation')
%learn
lA = plot_error_line(TC_mean_sem.ts.st_learn.A,'--',2,[65,105,225]/255);
lB = plot_error_line(TC_mean_sem.ts.st_learn.B,'--',2,[220,20,60]/255);

%recall
rA = plot_error_line(TC_mean_sem.ts.st_recall.all_corr.A,'-',2,[65,105,225]/255);
rB = plot_error_line(TC_mean_sem.ts.st_recall.all_corr.B,'-',2,[220,20,60]/255);

legend([lA,lB,rA,rB],{'Learning A','Learning B','Recall A','Recall B'},'location','northeast')

set(gca,'FontSize',16)
set(gca,'Linewidth',2)

%TC SI
subplot(2,2,3)
hold on
xlabel('Relative Day to D1')
ylabel('Corr. score')
xticks([1:9])
xticklabels({'0','1','2','3','4','5','6','7','8'})
title('TC SI correlation - raw')
%learn
lA = plot_error_line(TC_mean_sem.si.st_learn.A,'--',2,[65,105,225]/255);
lB = plot_error_line(TC_mean_sem.si.st_learn.B,'--',2,[220,20,60]/255);

%recall
rA = plot_error_line(TC_mean_sem.si.st_recall.d4_d5.A,'-',2,[65,105,225]/255);
rB = plot_error_line(TC_mean_sem.si.st_recall.d4_d5.B,'-',2,[220,20,60]/255);

legend([lA,lB,rA,rB],{'Learning A','Learning B','Recall A','Recall B'},'location','northeast')

set(gca,'FontSize',16)
set(gca,'Linewidth',2)


subplot(2,2,4)
hold on
xlabel('Relative Day to D1')
ylabel('Corr. score')
xticks([1:9])
xticklabels({'0','1','2','3','4','5','6','7','8'})
title('TC SI correlation - cross correlation')
%learn
lA = plot_error_line(TC_mean_sem.si.st_learn.A,'--',2,[65,105,225]/255);
lB = plot_error_line(TC_mean_sem.si.st_learn.B,'--',2,[220,20,60]/255);

%recall
rA = plot_error_line(TC_mean_sem.si.st_recall.all_corr.A,'-',2,[65,105,225]/255);
rB = plot_error_line(TC_mean_sem.si.st_recall.all_corr.B,'-',2,[220,20,60]/255);

legend([lA,lB,rA,rB],{'Learning A','Learning B','Recall A','Recall B'},'location','northeast')

set(gca,'FontSize',16)
set(gca,'Linewidth',2)



%% Equivalent plots for short term data for SI - supplement/thesis data
st_learn_day_range = 1:7;
st_recall_day_range = 1:9;

figure('Position',[2272 370 1007 420])
%PV plot - all neurons
subplot(1,2,1)
hold on 
axis square
title('PV correlation - all')
ylim([0 1])
xlim([0 10])
xticks(1:9)
yticks(0:0.2:1)
xlabel('Days')
ylabel('Correlation score')

%Learn
lA = errorbar(st_learn_day_range,st_learn.mean_PV.A(st_learn_day_range),st_learn.sem_PV.A(st_learn_day_range),'LineStyle','--','Linewidth',2,'Color',[65,105,225]/255);
lB = errorbar(st_learn_day_range,st_learn.mean_PV.B(st_learn_day_range),st_learn.sem_PV.B(st_learn_day_range),'LineStyle','--','Linewidth',2,'Color',[220,20,60]/255);

%Recall
rA = errorbar(st_recall_day_range,st_recall.mean_PV.all_corr.A(st_recall_day_range),st_recall.sem_PV.all_corr.A(st_recall_day_range),'LineStyle','-','Linewidth',2,'Color',[65,105,225]/255);
rB = errorbar(st_recall_day_range,st_recall.mean_PV.all_corr.B(st_recall_day_range),st_recall.sem_PV.all_corr.B(st_recall_day_range),'LineStyle','-','Linewidth',2,'Color',[220,20,60]/255);

set(gca,'FontSize',16)
set(gca,'Linewidth',2)

legend([lA,lB,rA,rB],{'Learning A','Learning B','Recall A','Recall B'},'location','northeast')

subplot(1,2,2)
hold on 
axis square
title('TC correlation - si')
ylim([0 1])
xlim([0 10])
xticks(1:9)
yticks(0:0.2:1)
xlabel('Days')
ylabel('Correlation score')

 
%Learn
lA = errorbar(st_learn_day_range,st_learn.si.mean_TC.animal.A(st_learn_day_range),st_learn.si.sem_TC.animal.A(st_learn_day_range),'LineStyle','--','Linewidth',2,'Color',[65,105,225]/255);
lB = errorbar(st_learn_day_range,st_learn.si.mean_TC.animal.B(st_learn_day_range),st_learn.si.sem_TC.animal.B(st_learn_day_range),'LineStyle','--','Linewidth',2,'Color',[220,20,60]/255);

%Recall
rA = errorbar(st_recall_day_range,st_recall.si.mean_TC.all_corr.A(st_recall_day_range),st_recall.si.sem_TC.all_corr.A(st_recall_day_range),'LineStyle','-','Linewidth',2,'Color',[65,105,225]/255);
rB = errorbar(st_recall_day_range,st_recall.si.mean_TC.all_corr.B(st_recall_day_range),st_recall.si.sem_TC.all_corr.B(st_recall_day_range),'LineStyle','-','Linewidth',2,'Color',[220,20,60]/255);

set(gca,'FontSize',16)
set(gca,'Linewidth',2)

legend([lA,lB,rA,rB],{'Learning A','Learning B','Recall A','Recall B'},'location','northeast')


%% Below are control\QC plots 


%% Short term - Plot TC as function of time with sem - T.S - by animal

%short term plots
figure('Position', [2136 350 1031 420])
subplot(1,2,1)
hold on
title('TC - ts - by animal (normalized overlay)')
ylim([0 1])
errorbar(1:7, st_learn.ts.mean_TC.animal.A(1:7),st_learn.ts.sem_TC.animal.A(1:7),'b')
errorbar(1:7, st_learn.ts.mean_TC.animal.B(1:7),st_learn.ts.sem_TC.animal.B(1:7),'r')

errorbar(1:9, st_recall.ts.mean_TC.animal.A(1:9),st_recall.ts.sem_TC.animal.A(1:9),'b')
errorbar(1:9, st_recall.ts.mean_TC.animal.B(1:9),st_recall.ts.sem_TC.animal.B(1:9),'r')
%overlay
errorbar(1:7, st_learn.ts.norm.mean_TC.animal.A(1:7),st_learn.ts.norm.sem_TC.animal.A(1:7),'c--')
errorbar(1:7, st_learn.ts.norm.mean_TC.animal.B(1:7),st_learn.ts.norm.sem_TC.animal.B(1:7),'m--')

errorbar(1:9, st_recall.ts.norm.mean_TC.animal.A(1:9),st_recall.ts.norm.sem_TC.animal.A(1:9),'c--')
errorbar(1:9, st_recall.ts.norm.mean_TC.animal.B(1:9),st_recall.ts.norm.sem_TC.animal.B(1:9),'m--')


%overlay normalized data as a control
subplot(1,2,2)
hold on
ylim([0 1])
title('TC - ts - by animal - correlation on normalized STCs for control (no overlay)')
errorbar(1:7, st_learn.ts.norm.mean_TC.animal.A(1:7),st_learn.ts.norm.sem_TC.animal.A(1:7),'--')
errorbar(1:7, st_learn.ts.norm.mean_TC.animal.B(1:7),st_learn.ts.norm.sem_TC.animal.B(1:7),'--')

errorbar(1:9, st_recall.ts.norm.mean_TC.animal.A(1:9),st_recall.ts.norm.sem_TC.animal.A(1:9),'--')
errorbar(1:9, st_recall.ts.norm.mean_TC.animal.B(1:9),st_recall.ts.norm.sem_TC.animal.B(1:9),'--')


%% Short term - Plot TC as function of time with sem - T.S - by animal (compare with cross day correlations)

%short term plots
figure('Position', [1921 41 1920 963])
subplot(2,3,1)
hold on
axis square
xlabel('Day')
ylabel('Correlation')
title('TC - ts - by animal ')
ylim([0 1])
errorbar(1:8, st_learn.ts.mean_TC.animal.A(1:8),st_learn.ts.sem_TC.animal.A(1:8),'b--')
errorbar(1:8, st_learn.ts.mean_TC.animal.B(1:8),st_learn.ts.sem_TC.animal.B(1:8),'r--')

errorbar(1:9, st_recall.ts.mean_TC.animal.A(1:9),st_recall.ts.sem_TC.animal.A(1:9),'b')
errorbar(1:9, st_recall.ts.mean_TC.animal.B(1:9),st_recall.ts.sem_TC.animal.B(1:9),'r')
legend({'A learn','B learn', 'A recall sub','B recall sub'})

%overlay normalized data as a control
subplot(2,3,2)
hold on
axis square
xlabel('Day')
ylabel('Correlation')
ylim([0 1])
title('TC - ts - by animal - \newline using recall data generated from distance-day correlations')
errorbar(1:8, st_learn.ts.mean_TC.animal.A(1:8),st_learn.ts.sem_TC.animal.A(1:8),'b--')
errorbar(1:8, st_learn.ts.mean_TC.animal.B(1:8),st_learn.ts.sem_TC.animal.B(1:8),'r--')

errorbar(1:9, st_recall.ts.mean_TC.all_corr.A(1:9),st_recall.ts.sem_TC.all_corr.A(1:9),'b-')
errorbar(1:9, st_recall.ts.mean_TC.all_corr.B(1:9),st_recall.ts.sem_TC.all_corr.B(1:9),'r-')
legend({'A learn','B learn', 'A recall all-corr','B recall all-corr'})

subplot(2,3,3)
hold on
axis square
xlabel('Day')
ylabel('Correlation')
ylim([0 1])
title('TC - ts - by animal - \newline overlay D4/D5 substituion and all-day correlation data')
%D4/D5 substitution
errorbar(1:9, st_recall.ts.mean_TC.animal.A(1:9),st_recall.ts.sem_TC.animal.A(1:9),'b')
errorbar(1:9, st_recall.ts.mean_TC.animal.B(1:9),st_recall.ts.sem_TC.animal.B(1:9),'r')

%all day cross correlations
errorbar(1:9, st_recall.ts.mean_TC.all_corr.A(1:9),st_recall.ts.sem_TC.all_corr.A(1:9),'b--')
errorbar(1:9, st_recall.ts.mean_TC.all_corr.B(1:9),st_recall.ts.sem_TC.all_corr.B(1:9),'r--')
legend({'A','B', 'A all-corr','B all-corr'})

% Short term - Plot TC as function of time with sem - S.I. - by animal (compare with cross day correlations)

%short term plots
%figure('Position', [2136 350 1031 420])
subplot(2,3,4)
hold on
axis square
xlabel('Day')
ylabel('Correlation')
title('TC - si - by animal ')
ylim([0 1])
errorbar(1:8, st_learn.si.mean_TC.animal.A(1:8),st_learn.si.sem_TC.animal.A(1:8),'b--')
errorbar(1:8, st_learn.si.mean_TC.animal.B(1:8),st_learn.si.sem_TC.animal.B(1:8),'r--')

errorbar(1:9, st_recall.si.mean_TC.animal.A(1:9),st_recall.si.sem_TC.animal.A(1:9),'b')
errorbar(1:9, st_recall.si.mean_TC.animal.B(1:9),st_recall.si.sem_TC.animal.B(1:9),'r')
legend({'A learn','B learn', 'A recall sub','B recall sub'})

%overlay normalized data as a control
subplot(2,3,5)
hold on
axis square
xlabel('Day')
ylabel('Correlation')
ylim([0 1])
title('TC - si - by animal - \newline using recall data generated from distance-day correlations')
errorbar(1:8, st_learn.si.mean_TC.animal.A(1:8),st_learn.si.sem_TC.animal.A(1:8),'b--')
errorbar(1:8, st_learn.si.mean_TC.animal.B(1:8),st_learn.si.sem_TC.animal.B(1:8),'r--')

errorbar(1:9, st_recall.si.mean_TC.all_corr.A(1:9),st_recall.si.sem_TC.all_corr.A(1:9),'b-')
errorbar(1:9, st_recall.si.mean_TC.all_corr.B(1:9),st_recall.si.sem_TC.all_corr.B(1:9),'r-')
legend({'A learn','B learn', 'A recall all-corr','B recall all-corr'})

subplot(2,3,6)
hold on
axis square
xlabel('Day')
ylabel('Correlation')
ylim([0 1])
title('TC - si - by animal - \newline overlay D4/D5 substituion and all-day correlation data')
%D4/D5 substitution
errorbar(1:9, st_recall.si.mean_TC.animal.A(1:9),st_recall.si.sem_TC.animal.A(1:9),'b')
errorbar(1:9, st_recall.si.mean_TC.animal.B(1:9),st_recall.si.sem_TC.animal.B(1:9),'r')

%all day cross correlations
errorbar(1:9, st_recall.si.mean_TC.all_corr.A(1:9),st_recall.si.sem_TC.all_corr.A(1:9),'b--')
errorbar(1:9, st_recall.si.mean_TC.all_corr.B(1:9),st_recall.si.sem_TC.all_corr.B(1:9),'r--')
legend({'A','B', 'A all-corr','B all-corr'})

%% Short term - Plot TC as function of time with sem - T.S - by neuron

%short term plots
subplot(1,2,2)
hold on
title('TC - ts - by neuron')
ylim([0 1])
errorbar(1:7, st_learn.ts.mean_TC.neuron.A(1:7),st_learn.ts.sem_TC.neuron.A(1:7))
errorbar(1:7, st_learn.ts.mean_TC.neuron.B(1:7),st_learn.ts.sem_TC.neuron.B(1:7))

errorbar(1:9, st_recall.ts.mean_TC.neuron.A(1:9),st_recall.ts.sem_TC.neuron.A(1:9))
errorbar(1:9, st_recall.ts.mean_TC.neuron.B(1:9),st_recall.ts.sem_TC.neuron.B(1:9))

%% Long term recall TC plots - TS - by animal

figure('Position', [2136 350 1031 420])
subplot(1,2,1)
hold on
title('TC correlation -  LT - TS - by animal')
ylim([0 1])
plot_days = [1 6 16 20 25 30];
xticks(plot_days)
errorbar(plot_days, lt_recall.ts.mean_TC.animal.A(plot_days),lt_recall.ts.sem_TC.animal.A(plot_days))
errorbar(plot_days, lt_recall.ts.mean_TC.animal.B(plot_days),lt_recall.ts.sem_TC.animal.B(plot_days))


%% Long term recall TC plots - TS - by neuron

subplot(1,2,2)
hold on
title('TC correlation -  LT - TS - by neuron')
ylim([0 1])
plot_days = [1 6 16 20 25 30];
xticks(plot_days)
errorbar(plot_days, lt_recall.ts.mean_TC.neuron.A(plot_days),lt_recall.ts.sem_TC.neuron.A(plot_days))
errorbar(plot_days, lt_recall.ts.mean_TC.neuron.B(plot_days),lt_recall.ts.sem_TC.neuron.B(plot_days))


%% Plot TC as function of time with sem - S.I. - by animal

figure('Position', [2136 350 1031 420])
subplot(1,2,1)
hold on
title('TC - si - by animal')
ylim([0 1])
errorbar(1:7, st_learn.si.mean_TC.animal.A(1:7),st_learn.si.sem_TC.animal.A(1:7))
errorbar(1:7, st_learn.si.mean_TC.animal.B(1:7),st_learn.si.sem_TC.animal.B(1:7))

errorbar(1:9, st_recall.si.mean_TC.animal.A(1:9),st_recall.si.sem_TC.animal.A(1:9))
errorbar(1:9, st_recall.si.mean_TC.animal.B(1:9),st_recall.si.sem_TC.animal.B(1:9))

%% Plot TC as function of time with sem - S.I. - by neuron

%short term plots
subplot(1,2,2)
hold on
title('TC - si - by neuron')
ylim([0 1])
errorbar(1:7, st_learn.si.mean_TC.neuron.A(1:7),st_learn.si.sem_TC.neuron.A(1:7))
errorbar(1:7, st_learn.si.mean_TC.neuron.B(1:7),st_learn.si.sem_TC.neuron.B(1:7))

errorbar(1:9, st_recall.si.mean_TC.neuron.A(1:9),st_recall.si.sem_TC.neuron.A(1:9))
errorbar(1:9, st_recall.si.mean_TC.neuron.B(1:9),st_recall.si.sem_TC.neuron.B(1:9))

%% Plot PV as function of time with sem; 
%compare against plot with correlation generated from all day correlation
%for short term recall

%short term plots
figure
subplot(1,2,1)
hold on
ylim([0 1])
%plot(mean_PV.A(1:7),'b--')
title('PV correlation - by animal')
errorbar(1:7, st_learn.mean_PV.A(1:7),st_learn.sem_PV.A(1:7))
errorbar(1:7, st_learn.mean_PV.B(1:7),st_learn.sem_PV.B(1:7))

errorbar(1:9, st_recall.mean_PV.A(1:9),st_recall.sem_PV.A(1:9))
errorbar(1:9, st_recall.mean_PV.B(1:9),st_recall.sem_PV.B(1:9))

subplot(1,2,2)
hold on
ylim([0 1])
%plot(mean_PV.A(1:7),'b--')
title('PV correlation - by animal - using all-day correlations for recall')
errorbar(1:7, st_learn.mean_PV.A(1:7),st_learn.sem_PV.A(1:7))
errorbar(1:7, st_learn.mean_PV.B(1:7),st_learn.sem_PV.B(1:7))

errorbar(1:9, st_recall.mean_PV.all_corr.A(1:9),st_recall.sem_PV.all_corr.A(1:9))
errorbar(1:9, st_recall.mean_PV.all_corr.B(1:9),st_recall.sem_PV.all_corr.B(1:9))


%long term plots
figure
hold on
title('PV correlation - by animal')
ylim([0 1])
plot_days = [1 6 16 20 25 30];
xticks(plot_days)
errorbar(plot_days, lt_recall.mean_PV.A(plot_days),lt_recall.sem_PV.A(plot_days))
errorbar(plot_days, lt_recall.mean_PV.B(plot_days),lt_recall.sem_PV.B(plot_days))

%% Compare PV plot with short term recall generated for substitution of day 4 and day 5 vs. correlation against each point

figure('Position', [2093 485 1601 420])
subplot(1,3,1)
hold on
ylim([0 1])
title('PV - ST recall with D4 & D5 substitution')
xlabel('Day')
ylabel('Correlation')

%short term learning data
errorbar(1:8, st_learn.mean_PV.A(1:8),st_learn.sem_PV.A(1:8),'b--')
errorbar(1:8, st_learn.mean_PV.B(1:8),st_learn.sem_PV.B(1:8),'r--')

%st recall with substitution for D4 and D5
errorbar(1:9, st_recall.mean_PV.A(1:9),st_recall.sem_PV.A(1:9),'b')
errorbar(1:9, st_recall.mean_PV.B(1:9),st_recall.sem_PV.B(1:9),'r')
legend({'A learn','B learn', 'A recall sub','B recall sub'})

subplot(1,3,2)
hold on
ylim([0 1])
title('PV - ST recall with all day correlation distance days')
xlabel('Day')
ylabel('Correlation')

%short term learning data
errorbar(1:8, st_learn.mean_PV.A(1:8),st_learn.sem_PV.A(1:8),'b')
errorbar(1:8, st_learn.mean_PV.B(1:8),st_learn.sem_PV.B(1:8),'r')

%st recall with all cross day correlations
errorbar(1:9, st_recall.mean_PV.all_corr.A(1:9),st_recall.sem_PV.all_corr.A(1:9),'b--')
errorbar(1:9, st_recall.mean_PV.all_corr.B(1:9),st_recall.sem_PV.all_corr.B(1:9),'r--')
legend({'A learn','B learn', 'A recall all-corr','B recall all-corr'})

subplot(1,3,3)
hold on
ylim([0 1])
title('PV - ST recall with D4 & D5 substitution \newline vs. ST recall with all day correlation distance days')
xlabel('Day')
ylabel('Correlation')

%st recall with all cross day correlations
errorbar(1:9, st_recall.mean_PV.all_corr.A(1:9),st_recall.sem_PV.all_corr.A(1:9),'b--')
errorbar(1:9, st_recall.mean_PV.all_corr.B(1:9),st_recall.sem_PV.all_corr.B(1:9),'r--')

%st recall with substitution for D4 and D5
errorbar(1:9, st_recall.mean_PV.A(1:9),st_recall.sem_PV.A(1:9),'b')
errorbar(1:9, st_recall.mean_PV.B(1:9),st_recall.sem_PV.B(1:9),'r')
legend({'A','B', 'A all-corr','B all-corr'})


end

