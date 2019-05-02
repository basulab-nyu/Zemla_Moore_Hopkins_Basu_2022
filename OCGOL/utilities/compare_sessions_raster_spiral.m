function [outputArg1,outputArg2] = compare_sessions_raster_spiral(session_vars,registered,ROI_outlines,ROI_zooms)

%% Define/load variables for each session

%for each session
for ii = 1:size(session_vars,2)
    % behavior and imaging related variables
    Behavior_split_lap{ii} = session_vars{ii}.Behavior_split_lap;
    Events_split_lap{ii} = session_vars{ii}.Events_split_lap;
    Behavior_split{ii} = session_vars{ii}.Behavior_split;
    Event_split{ii} = session_vars{ii}.Events_split;
    Imaging_split{ii} = session_vars{ii}.Imaging_split;
    Place_cell{ii} = session_vars{ii}.Place_cell;
    Behavior_full{ii} = session_vars{ii}.Behavior;
    
    %all within run domain
    position{ii}  = Behavior_split_lap{ii}.Run.position;
    time{ii} = Behavior_split_lap{ii}.Run.time;
    events_full{ii} = Events_split_lap{ii}.Run.run_onset_binary;
    run_intervals{ii} = Behavior_split_lap{ii}.run_ones;
    
    %global trial type order across restricted laps
    trialOrder{ii} = Behavior_full{ii}.performance.trialOrder;
end

%for each session
for ss=1:size(session_vars,2)
    %for each lap
    for ii=1:size(run_intervals{ss},2)
        events{ss}{ii} = events_full{ss}{ii}(logical(run_intervals{ss}{ii}),:);
    end
end

%lap_times = Behavior_split_lap.lap;
%how many trial types are there
%set to 2 - A and B for now
%trialTypes = 2;

%make this a condition for the type of trials that are compared
%all correct; all regardless of correct

%for each session
for ss = 1:size(session_vars,2)
    %find times from trials related to specific trial type
    %A trials
    trialTypeIdx{ss}{1} = find(trialOrder{ss} == 2 | trialOrder{ss} == 20);
    %B trials
    trialTypeIdx{ss}{2} = find(trialOrder{ss} == 3 | trialOrder{ss} == 30);
end

%% Define the spiral parameters according to the number of laps
%equivalent to number of laps for each session
turns = size(trialOrder,1); %The number of turns the spiral will have (how many laps)

%x is the angle
x=[-1*pi*turns : 0.01 : pi*turns];
%r is that radius of the point
r=[0:1/(length(x)-1):1];

%scale to lap length
r_scaled = r.*turns;

%all parameters in the run frame domain
%find the frames index of event and position
%for trial type
for ii=1:size(trialTypeIdx,2)
    %for each lap belonging to that trial
    for ll= 1:size(trialTypeIdx{ii},1)
        %for each ROI
        for rr=1:size(events{trialTypeIdx{ii}(ll)},2)
            %event indices in run domain
            event_idx{ii}{ll}{rr} = find(events{trialTypeIdx{ii}(ll)}(:,rr) == 1);
            %position that corresponds to indices
            pos{ii}{ll}{rr} = position{trialTypeIdx{ii}(ll)}(event_idx{ii}{ll}{rr});
            %position vectors that will be used as input to spiral
            posVectors{ii}{ll}{rr} = trialTypeIdx{ii}(ll).*exp(1i.*((pos{ii}{ll}{rr}/200)*2*pi)).';
        end
    end
end

%make the nearest approximation to a point along the spiral vector defined above
%predefine to avoid empty cells at the end
valMin = posVectors;
idxMin = posVectors;
posVectorApprox = posVectors;
%for each ROI
%for each trial type
for kk = 1:size(trialTypeIdx,2)
    %for each lap belonging to that trial
    for ll = 1:size(pos{kk},2)
        %for each ROI
        for rr = 1:size(events{1},2)
            %for each event
            for ee=1:size(pos{kk}{ll}{rr},1)
                %fix empty cell clipping at endpoints
                [valMin{kk}{ll}{rr}(ee),idxMin{kk}{ll}{rr}(ee)] = min(abs( (r_scaled - ( (trialTypeIdx{kk}(ll)-1) + (pos{kk}{ll}{rr}(ee)/200) ) ) ));
                posVectorApprox{kk}{ll}{rr}(ee) = r_scaled(idxMin{kk}{ll}{rr}(ee))*exp(1i.*(pos{kk}{ll}{rr}(ee)/200)*2*pi);
            end
        end
    end
end

%% Plot raster, event spiral and matching ROIs from FOV

figure;
for ii=1:size(registered.multi.assigned_all,1)
    
    subplot(2,3,1)
    imagesc(session_vars{1}.Place_cell{1, 3}.dF_lap_map_ROI{registered.multi.assigned_all(ii,1)})
    hold on;
    ylabel('Lap #'); 
    xlabel('Spatial bin');
    caxis([0 2])
    colormap(gca,'jet');
    hold off;
    
    %spiral plot early in learning
    subplot(2,3,2)
    
    
    subplot(2,3,3)
    imagesc(ROI_zooms{ii,1})
    hold on;
    colormap(gca, 'gray')
    xticks([])
    yticks([])
    b = bwboundaries(ROI_outlines{ii,1},'noholes');
    plot(b{1}(:,2),b{1}(:,1),'r')
    hold off
    
    
    subplot(2,3,4)
    imagesc(session_vars{2}.Place_cell{1, 3}.dF_lap_map_ROI{registered.multi.assigned_all(ii,2)})
    hold on;
    ylabel('Lap #'); 
    xlabel('Spatial bin');
    caxis([0 2])
    colormap(gca, 'jet');
    hold off;
    
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
    
    pause;
    clf;
    
end


end

