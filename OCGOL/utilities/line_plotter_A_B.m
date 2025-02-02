function [outputArg1,outputArg2] = line_plotter_A_B(PV_mean_sem)



%% Arrange data for PV and TC correlations into cells and generate label for each set in another set of cells



%%
figure
hold on 
axis square
title('Test')
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


end

