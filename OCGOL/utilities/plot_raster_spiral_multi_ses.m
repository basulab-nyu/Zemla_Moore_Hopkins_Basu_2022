function [outputArg1,outputArg2] = plot_raster_spiral_multi_ses(plot_raster_vars,session_vars,registered,ROI_zooms, ROI_outlines,ROI_categories,options)


%% Load variables

idxMin = plot_raster_vars.idxMin;
r_scaled = plot_raster_vars.r_scaled;
posVectorApprox = plot_raster_vars.posVectorApprox;
x = plot_raster_vars.x;


%% Plot raster, event spiral and matching ROIs from FOV
%modify this to see events serially

%% Plot raster , spiral, ROI FOV across all 6 session

%find day1 and da
if 0 
%order of plots
subplot_matrix = 1:18;
subplot_matrix = reshape(subplot_matrix,6,3)';

figure('Position',[1920 40 1920 960]);
for ii=1:size(registered.multi.assigned_filtered,1) %with nans where no match
    %size(registered.multi.assigned_all,1) - match on each ses %options.idx_show%
    
    %for each session
    for ss=1:6
        %ROI = registered.multi.assigned_all(ii,ss);
        ROI = registered.multi.assigned_filtered(ii,ss);
        %skip of nan value
        if ~isnan(ROI)
            subplot(3,6,subplot_matrix(1,ss))
            %imagesc(session_vars{ss}.Place_cell{1, 3}.dF_lap_map_ROI{ROI})
            hold on;
            title(num2str(ROI));
            ylabel('Lap #');
            xlabel('Spatial bin');
            caxis([0 2])
            colormap(gca,'jet');
            hold off;
            
            %spiral plot early in learning
            subplot(3,6,subplot_matrix(2,ss))
            polarplot(x{ss},r_scaled{ss},'k','Linewidth',1.5)
            hold on
            
            %plot A (2) trial events
            for ll=1:size(idxMin{ss}{1},2)
                polarscatter(angle(posVectorApprox{ss}{1}{ll}{ROI}),r_scaled{ss}(idxMin{ss}{1}{ll}{ROI}),'bo','MarkerFaceColor','b')
                %place field center
                %polarscatter(centerA_angle(ii), 20, 'b*','MarkerFaceColor','b');
            end
            
            %plot tuning specificity vector for all A trials
            polarplot([0+0i,15*session_vars{ss}.Place_cell{1, 4}.Tuning_Specificity.tuning_vector_specificity(ROI)],'b-','LineWidth',2)
            
            %plot B (3) trial events
            for ll=1:size(idxMin{ss}{2},2)
                polarscatter(angle(posVectorApprox{ss}{2}{ll}{ROI}),r_scaled{ss}(idxMin{ss}{2}{ll}{ROI}),'ro','MarkerFaceColor','r')
                %place field center
                %polarscatter(centerB_angle(ii), 20, 'r*','MarkerFaceColor','r');
            end
            
            %plot tuning specificity vector for all B trials
            polarplot([0+0i,15*session_vars{ss}.Place_cell{1, 5}.Tuning_Specificity.tuning_vector_specificity(ROI)],'r-','LineWidth',2)
            
            hold off
            
            subplot(3,6,subplot_matrix(3,ss))
            imagesc(ROI_zooms{ss}{ROI})
            hold on;
            colormap(gca, 'gray')
            xticks([])
            yticks([])
            b = bwboundaries(ROI_outlines{ss}{ROI},'noholes');
            plot(b{1}(:,2),b{1}(:,1),'r')
            hold off
        end
    end
     %mv_frame(ii) = getframe(gcf);
    pause();
    clf;
end

figure('Position',[1920 40 1920 960]);
for ii=1:size(registered.multi.matching_list_filtered.si_AB_filt_event_filt,1) %with nans where no match
    %size(registered.multi.assigned_all,1) - match on each ses %options.idx_show%
    
    %for each session
    for ss=1:6
        %ROI = registered.multi.assigned_all(ii,ss);
        ROI = registered.multi.matching_list_filtered.si_AB_filt_event_filt(ii,ss);
        %skip of nan value
        if ~isnan(ROI)
            subplot(3,6,subplot_matrix(1,ss))
            %imagesc(session_vars{ss}.Place_cell{1, 3}.dF_lap_map_ROI{ROI})
            hold on;
            title(num2str(ROI));
            ylabel('Lap #');
            xlabel('Spatial bin');
            caxis([0 2])
            colormap(gca,'jet');
            hold off;
            
            %spiral plot early in learning
            subplot(3,6,subplot_matrix(2,ss))
            polarplot(x{ss},r_scaled{ss},'k','Linewidth',1.5)
            hold on
            
            %plot A (2) trial events
            for ll=1:size(idxMin{ss}{1},2)
                polarscatter(angle(posVectorApprox{ss}{1}{ll}{ROI}),r_scaled{ss}(idxMin{ss}{1}{ll}{ROI}),'bo','MarkerFaceColor','b')
                %place field center
                %polarscatter(centerA_angle(ii), 20, 'b*','MarkerFaceColor','b');
            end
            
            %plot tuning specificity vector for all A trials
            polarplot([0+0i,15*session_vars{ss}.Place_cell{1, 4}.Tuning_Specificity.tuning_vector_specificity(ROI)],'b-','LineWidth',2)
            
            %plot B (3) trial events
            for ll=1:size(idxMin{ss}{2},2)
                polarscatter(angle(posVectorApprox{ss}{2}{ll}{ROI}),r_scaled{ss}(idxMin{ss}{2}{ll}{ROI}),'ro','MarkerFaceColor','r')
                %place field center
                %polarscatter(centerB_angle(ii), 20, 'r*','MarkerFaceColor','r');
            end
            
            %plot tuning specificity vector for all B trials
            polarplot([0+0i,15*session_vars{ss}.Place_cell{1, 5}.Tuning_Specificity.tuning_vector_specificity(ROI)],'r-','LineWidth',2)
            
            hold off
            
            subplot(3,6,subplot_matrix(3,ss))
            imagesc(ROI_zooms{ss}{ROI})
            hold on;
            colormap(gca, 'gray')
            xticks([])
            yticks([])
            b = bwboundaries(ROI_outlines{ss}{ROI},'noholes');
            plot(b{1}(:,2),b{1}(:,1),'r')
            hold off
        end
    end
     %mv_frame(ii) = getframe(gcf);
    pause();
    clf;
end



figure('Position',[2600,300,1200,1000]);
for ii=1:size(registered.multi.assigned_filtered,1)
    
    %ROI from session 1
    ROI = registered.multi.assigned_filtered(ii,1);
    
    subplot(2,3,1)
    %imagesc(session_vars{1}.Place_cell{1, 3}.dF_lap_map_ROI{ROI})
    hold on;
    title(num2str(ROI));
    ylabel('Lap #'); 
    xlabel('Spatial bin');
    caxis([0 2])
    colormap(gca,'jet');
    hold off;
    
    
    
    %spiral plot early in learning
    subplot(2,3,2)
    polarplot(x{1},r_scaled{1},'k','Linewidth',1.5)
    hold on
    
    %plot A (2) trial events
    for ll=1:size(idxMin{1}{1},2)
        polarscatter(angle(posVectorApprox{1}{1}{ll}{ROI}),r_scaled{1}(idxMin{1}{1}{ll}{ROI}),'bo','MarkerFaceColor','b')
        %place field center
        %polarscatter(centerA_angle(ii), 20, 'b*','MarkerFaceColor','b');
    end
    
    %plot tuning specificity vector for all A trials
    polarplot([0+0i,15*session_vars{1, 1}.Place_cell{1, 4}.Tuning_Specificity.tuning_vector_specificity(ROI)],'b-','LineWidth',2)


    %plot B (3) trial events
    for ll=1:size(idxMin{1}{2},2)
        polarscatter(angle(posVectorApprox{1}{2}{ll}{ROI}),r_scaled{1}(idxMin{1}{2}{ll}{ROI}),'ro','MarkerFaceColor','r')
        %place field center
        %polarscatter(centerB_angle(ii), 20, 'r*','MarkerFaceColor','r');
    end
    
    %plot tuning specificity vector for all B trials
    polarplot([0+0i,15*session_vars{1, 1}.Place_cell{1, 5}.Tuning_Specificity.tuning_vector_specificity(ROI)],'r-','LineWidth',2)


    hold off
    
    subplot(2,3,3)
    imagesc(ROI_zooms{ii,1})
    hold on;
    colormap(gca, 'gray')
    xticks([])
    yticks([])
    b = bwboundaries(ROI_outlines{ii,1},'noholes');
    plot(b{1}(:,2),b{1}(:,1),'r')
    hold off
    
    %ROI from session 2
    ROI = registered.multi.assigned_all(ii,2);
    subplot(2,3,4)
    imagesc(session_vars{2}.Place_cell{1, 3}.dF_lap_map_ROI{ROI})
    hold on;
    title(num2str(ROI));
    ylabel('Lap #'); 
    xlabel('Spatial bin');
    caxis([0 2])
    colormap(gca, 'jet');
    hold off;
    
   
    
    subplot(2,3,5)
    polarplot(x{2},r_scaled{2},'k','Linewidth',1.5)
    hold on
    %plot A (2) trial events
    for ll=1:size(idxMin{2}{1},2)
        polarscatter(angle(posVectorApprox{2}{1}{ll}{ROI}),r_scaled{2}(idxMin{2}{1}{ll}{ROI}),'bo','MarkerFaceColor','b')
        %place field center
        %polarscatter(centerA_angle(ii), 20, 'b*','MarkerFaceColor','b');
    end
    %plot tuning specificity vector for all A trials
    polarplot([0+0i,15*session_vars{2}.Place_cell{4}.Tuning_Specificity.tuning_vector_specificity(ROI)],'b-','LineWidth',2)

    
    %plot B (3) trial events
    for ll=1:size(idxMin{2}{2},2)
        polarscatter(angle(posVectorApprox{2}{2}{ll}{ROI}),r_scaled{2}(idxMin{2}{2}{ll}{ROI}),'ro','MarkerFaceColor','r')
        %place field center
        %polarscatter(centerB_angle(ii), 20, 'r*','MarkerFaceColor','r');
    end
    
    %plot tuning specificity vector for all A trials
    polarplot([0+0i,15*session_vars{2}.Place_cell{5}.Tuning_Specificity.tuning_vector_specificity(ROI)],'r-','LineWidth',2)

    hold off
    
    subplot(2,3,6)
    imagesc(ROI_zooms{ii,2})
    %imagesc(ROI_zooms{registered.multi.assigned_all(ii,2),2})
    hold on;
    colormap(gca, 'gray')
    xticks([])
    yticks([])
    %b = bwboundaries(ROI_outlines{registered.multi.assigned_all(ii,2),2},'noholes');
    b = bwboundaries(ROI_outlines{ii,2},'noholes');
    plot(b{1}(:,2),b{1}(:,1),'r')
    hold off
    
    %pause;
    %clf;
    
end
end


%% Plot raster , spiral, ROI FOV across  - 2 session comparison

%find day1 and da
if 0
%order of plots
subplot_matrix = 1:6;
subplot_matrix = reshape(subplot_matrix,2,3)';

figure('Position',[2230 30 780 960]);
for ii=1:size(registered.multi.assign_cell{1, 6},1) %with nans where no match
    %size(registered.multi.assigned_all,1) - match on each ses %options.idx_show%
    ses_nb = 1;
    %for each session
    for ss=[1,6]
        
        %ROI = registered.multi.assigned_all(ii,ss);
        %ROI = registered.multi.assigned(ii,ss);
        ROI = registered.multi.assign_cell{1,6}(ii,ses_nb);
        %skip of nan value
        if ~isnan(ROI)
            subplot(3,2,subplot_matrix(1,ses_nb))
            imagesc(session_vars{ss}.Place_cell{1, 3}.dF_lap_map_ROI{ROI})
            hold on;
            title(num2str(ROI));
            ylabel('Lap #');
            xlabel('Spatial bin');
            caxis([0 2])
            colormap(gca,'jet');
            hold off;
            
            %spiral plot early in learning
            subplot(3,2,subplot_matrix(2,ses_nb))
            polarplot(x{ss},r_scaled{ss},'k','Linewidth',1.5)
            hold on
            
            %plot A (2) trial events
            for ll=1:size(idxMin{ss}{1},2)
                polarscatter(angle(posVectorApprox{ss}{1}{ll}{ROI}),r_scaled{ss}(idxMin{ss}{1}{ll}{ROI}),'bo','MarkerFaceColor','b')
                %place field center
                %polarscatter(centerA_angle(ii), 20, 'b*','MarkerFaceColor','b');
            end
            
            %plot tuning specificity vector for all A trials
            polarplot([0+0i,15*session_vars{ss}.Place_cell{1, 4}.Tuning_Specificity.tuning_vector_specificity(ROI)],'b-','LineWidth',2)
            
            %plot B (3) trial events
            for ll=1:size(idxMin{ss}{2},2)
                polarscatter(angle(posVectorApprox{ss}{2}{ll}{ROI}),r_scaled{ss}(idxMin{ss}{2}{ll}{ROI}),'ro','MarkerFaceColor','r')
                %place field center
                %polarscatter(centerB_angle(ii), 20, 'r*','MarkerFaceColor','r');
            end
            
            %plot tuning specificity vector for all B trials
            polarplot([0+0i,15*session_vars{ss}.Place_cell{1, 5}.Tuning_Specificity.tuning_vector_specificity(ROI)],'r-','LineWidth',2)
            
            hold off
            
            subplot(3,2,subplot_matrix(3,ses_nb))
            imagesc(ROI_zooms{ss}{ROI})
            hold on;
            colormap(gca, 'gray')
            xticks([])
            yticks([])
            b = bwboundaries(ROI_outlines{ss}{ROI},'noholes');
            plot(b{1}(:,2),b{1}(:,1),'r')
            hold off
        end
        %update session number
         ses_nb = ses_nb +1;
    end
    pause(0.2)
    clf;
    disp(ii)
end
end


%% Single session for visualization of task remap categories

%which session to look at
selectSes = 3;

%create vector with  ROIs to display
selectROI = ROI_categories.task_selective_ROIs{selectSes}.B.idx;

%figure('Position',[2230 30 780 960]);
figure('Position',[1029 336 525 309])
for ii=1:size(selectROI,2) %with nans where no match
    %for each session
    for ss=selectSes
        ROI = selectROI(ii);
        %spiral plot early in learning
        %subplot(3,2,subplot_matrix(2,ses_nb))
        polarplot(x{ss},r_scaled{ss},'k','Linewidth',1.5)
        hold on
        
        %plot A (2) trial events
        for ll=1:size(idxMin{ss}{1},2)
            polarscatter(angle(posVectorApprox{ss}{1}{ll}{ROI}),r_scaled{ss}(idxMin{ss}{1}{ll}{ROI}),'bo','MarkerFaceColor','b')
            %place field center
            %polarscatter(centerA_angle(ii), 20, 'b*','MarkerFaceColor','b');
        end
        
        %plot tuning specificity vector for all A trials
        polarplot([0+0i,15*session_vars{ss}.Place_cell{1, 4}.Tuning_Specificity.tuning_vector_specificity(ROI)],'b-','LineWidth',2)
        
        %plot B (3) trial events
        for ll=1:size(idxMin{ss}{2},2)
            polarscatter(angle(posVectorApprox{ss}{2}{ll}{ROI}),r_scaled{ss}(idxMin{ss}{2}{ll}{ROI}),'ro','MarkerFaceColor','r')
            %place field center
            %polarscatter(centerB_angle(ii), 20, 'r*','MarkerFaceColor','r');
        end
        
        %plot tuning specificity vector for all B trials
        polarplot([0+0i,15*session_vars{ss}.Place_cell{1, 5}.Tuning_Specificity.tuning_vector_specificity(ROI)],'r-','LineWidth',2)
        
        hold off
       
    end
    pause;
    clf;
    disp(ii)
end

end

