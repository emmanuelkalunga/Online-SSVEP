%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%	Evaluate  How class probability varies with the number of epochs use
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
tLen = 4; %78.41
delay = 2;

for l = 1:length(tLen)
    for sub = 6:17
        clear x_all H_all P X Pm PSD

        %% Load data
        [S_all, H_all] = loaddata(sub); %Returns cells of data from all available sessions
        Fs = H_all{1}.SampleRate;
        nbrSessions = length(S_all);
        sessions = 1:nbrSessions;
        %% Preprocessing of all available sessions (Same for training and test data)
        % 1) Band pass filter
        for session = 1:nbrSessions
            x_all{session} = bandpass_filter_ext([12.95 13.05], [16.9 17.1], [20.9 21.1], S_all{session}, H_all{session}); %74.31
        end

        % 2) Rearange data per trial
        X = get_trials(x_all, H_all, tLen(l), delay);
        
        %get trials of raw data (not filtered)
        chan = [1:3;4:6;7:9;10:12;13:15;16:18;19:21;22:24];
        S = get_trials(S_all, H_all, tLen(l), delay);
        for k = 1:8
        for i = 1:size(S,2)
            for j = 1:size(S{1},3)
                [~, F, T, PSD{i}(chan(k,:),:,j)] = spectrogram(S{i}(k,:,j),rectwin(256),128,[13 17 21],256,'yaxis');
            end
            if k == 1
                Pm(:,:,i) = mean(PSD{i},3);
            end
        end
        end
        ylabels = {['NO SSVEP class']; ['13 Hz class'];['21 Hz class'];['17 Hz class']};
        

        % 3) Covariance matrices of all trialssummed up per class
            Nt = size(X{1},3); %Number of trial
            for k = 1:Nt %loop for evrey trial
                for cl = 1:4
                    P{cl}(:,:,k) = shcovft((X{cl}(:,:,k))'); % J. Schaefer Shrinkage covariance from Barachant toolbox
                end
            end 

        for testSession = 1:nbrSessions
            trials = 1:size(P{1},3);
            trialPerSession = size(P{1},3)/nbrSessions;
            
            testTrials = (trialPerSession*testSession-trialPerSession+1):(trialPerSession*testSession);
            trainTrials = setxor(trials, testTrials);
            
            %% TRAINING PHASE
            trainSessions = setxor(sessions, testSession);
            COVtrain = cat(3, P{1}(:,:,trainTrials), P{2}(:,:,trainTrials), P{3}(:,:,trainTrials), P{4}(:,:,trainTrials));
            Ytrain = [zeros(1,length(trainTrials)) ones(1,length(trainTrials)) 2*ones(1,length(trainTrials)) 3*ones(1,length(trainTrials))];

            %%                  EVALUATION PHASE                               **
            %********************************************************************
            N = 5;
            tLen2 = 3.6; 
            totLen = 9;
            tLimit = totLen - tLen2;
            step = 0.2;
            delays = 0:step:tLimit;
            conf = 0.7; % 70% confidence
            thresh = round(N*conf);
%             eps = 0.01;
            eps = 0;
      
            types = [33024 33025 33026 33027];
            
            for typ = 1:numel(types)
                ind(typ,:) = find(H_all{testSession}.EVENT.TYP==types(typ));
                pos(typ,:) = H_all{testSession}.EVENT.POS(ind(typ,:));
                class(typ,:) = (typ-1)*ones(size(pos(typ,:)));
            end
            
            class_v = class(:);
            pos_v = pos(:);
            [POS, I] = sort(pos_v); 
            CLASS = class_v(I);
            
            Labels(:,:,testSession,sub-5) = repmat(CLASS, 1, length(delays)-N);
            
            Fs = H_all{testSession}.SampleRate;
            markers = bsxfun(@plus, POS, round(delays*Fs));
            %markers_initial = markers(:,1:N);
            
            Nt = size(markers, 1); %Number of trials
            for tr = 1:Nt
                [wind sz] = trigg(x_all{testSession}, markers(tr,:), 0, round(tLen2*Fs)); %number of channels, trial length, number of trials
                Xtr = reshape(wind, sz);
                for win = 1:sz(3)
                    Ptr(:,:,win) = shcovft((Xtr(:,:,win))'); % J. Schaefer Shrinkage covariance from Barachant toolbox
                end  
                % Classification by Remannian Distance
                Ptr(isnan(Ptr)) = 0; %Avoid NaN in data matrices
                Ptr(isinf(Ptr)) = 999; %Avoid Inf in data matrices

                [Ytest_tmp d_tmp C] = mdm(Ptr(:,:,1:N),COVtrain,Ytrain);  %classifies N first segments 
                [M F] = mode(Ytest_tmp); %retuns the most occuring element in Ytest_tmp and its frequency of occurence
                classProb(:,1,tr,testSession,sub-5) = arrayfun(@(x)length(find(Ytest_tmp==x)), unique(Ytrain)) / length(Ytest_tmp);
                if ( F > thresh ) %Check if identified class has occured more than the threshold and that the SCMs are moving toward this class, or whether the identified class is 0, in which case the gradient is not checked
                    Ytest(tr,1) = M;
                else
                    Ytest(tr,1) = -1; %No classification
                end
                win = N+1;
                for sindex = 2:length(delays)-N
                    [y d] = mdm(Ptr(:,:,win),COVtrain,Ytrain); %classify one more segment (sliding window)
                    Ytest_tmp = [Ytest_tmp(2:end) y]; %concatenate new class while leaving out the oldest
                    [M F] = mode(Ytest_tmp); %retuns the most occuring element in Ytest_tmp and its frequency of occurence
                    classProb(:,sindex,tr,testSession,sub-5) = arrayfun(@(x)length(find(Ytest_tmp==x)), unique(Ytrain)) / length(Ytest_tmp);
                    if ( F > thresh ) %Check if identified class has occured more than the threshold
                        Ytest(tr,sindex) = M;
                    else
                        Ytest(tr,sindex) = -1; %No classification
                    end
                    win = win+1;
                end
                sprintf('subject %d,  session %d,  trial %d,  segment# is: %d ...',sub, testSession, tr, win)
                if all(Ytest(tr,:)==-1)
                        delay_fin(tr) = length(delays)-N;
                else
                        delay_fin(tr) = find(Ytest(tr,:)>=0,1);
                end
            end
            labels = repmat(CLASS, 1, length(delays)-N);
            ac(sub-5, testSession) = sum(sum((labels-Ytest)==0))/(numel(Ytest)-numel(find(Ytest==-1)));
            %Ytest_all(testSession, :, sub-5) = Ytest;
            delay_fin_all(testSession, :, sub-5) = delay_fin;
        end
    end
end

for i = 1:size(ac,1)
    acSi = ac(i,:);
    acSi = acSi(acSi~=0);
    subId(i) = i+5;
    subNbrOfSess(i) = length(acSi);
    subAcMean(i) = mean(acSi);
    subVar(i) = var(acSi);
    
    del_sub = delay_fin_all(:,:,i);
    del_sub = del_sub(:);
    del_sub = del_sub(del_sub~=0);
    del_sub_all(i) = mean((del_sub-1)*step);
end
resMatrix = [subId' subNbrOfSess' subAcMean' subVar'];
resMean = mean(resMatrix);
resMean(2) = sum(resMatrix(:,2));
resMean = resMean(2:end);

save('classProb_4class.mat', 'classProb', 'Labels', 'resMatrix', 'resMean', 'del_sub_all', 'subNbrOfSess');