function X = get_trials(x_all, H_all, tLen, varargin)
    if iscell(x_all)
        nbrSession = length(x_all);
        x = x_all;
        H = H_all;
    else
        nbrSession = 1;
        x{1} = x_all;
        H{1} = H_all;
    end
    if isempty(varargin)
        delay = 1; %in seconds
    else
        delay = varargin{1};
    end   
    event_types = [33024, 33025, 33026, 33027];     
    wind = [0 tLen]; %window of 5 sec
    trial_limits = wind+delay;
    types = event_types;  
    
    for typ = 1:length(types)
        Xtmp2 = [];
        for session = 1:nbrSession      
            markers = H{session}.EVENT.POS(find(H{session}.EVENT.TYP ==  types(typ)));

    %         %Add markers to get more trials in one.
    %         markers = markers+delay*256; %insert the delay
    %         markers2 = markers+2*256;
    %         markers3 = markers2+2*256;
    %         markersA = [markers markers2 markers3];
    %         markersF = reshape(markersA',24,1);
    %         wind = [0 2]; %window of 2 sec
    %         trial_limits = wind;
    %         [trials sz] = trigg(x, markersF, round(trial_limits(1)*(H.SampleRate)+1), round(trial_limits(2)*H.SampleRate)); %number of channels, trial length, number of trials
            %---------------------------------------------

            [trials sz] = trigg(x{session}, markers, round(trial_limits(1)*(H{session}.SampleRate)+1), round(trial_limits(2)*H{session}.SampleRate)); %number of channels, trial length, number of trials
            Xtmp1 = reshape(trials, sz);
            Xtmp2 = cat(3, Xtmp2, Xtmp1);
        end
        X{typ} = Xtmp2;
    end