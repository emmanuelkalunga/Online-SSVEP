%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%	Evaluate  How performance varies with window size (for 3-class, all SSVEP)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all

tLen = 4; 
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
%             x_all{session} = bandpass_filter_ext([12.9 13.1], [16.9 17.1], [20.9 21.1], S_all{session}, H_all{session}); %74.23
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
%                     P{cl}(:,:,k) = standardSCM((X{cl}(:,:,k))); %Standard SCM
%                     P{cl}(:,:,k) = NormalizedSCM((X{cl}(:,:,k))'); %As Provided in Barachant toolbox 
                end
            end 

        for testSession = 1:nbrSessions
            trials = 1:size(P{1},3);
            trialPerSession = size(P{1},3)/nbrSessions;
            
            testTrials = (trialPerSession*testSession-trialPerSession+1):(trialPerSession*testSession);
            trainTrials = setxor(trials, testTrials);
            
            %% TRAINING PHASE
            trainSessions = setxor(sessions, testSession);
            COVtrain = cat(3, P{2}(:,:,trainTrials), P{3}(:,:,trainTrials), P{4}(:,:,trainTrials));
            Ytrain = [zeros(1,length(trainTrials)) ones(1,length(trainTrials)) 2*ones(1,length(trainTrials))];

            %%                  EVALUATION PHASE                               **
            %********************************************************************
            lenIdx = 0;
            tlen = 1:0.2:8;
            for tLen2 = tlen
                lenIdx = lenIdx + 1;
                N = 5;
                %tLen2 = 3.6; 
                totLen = 9;
                tLimit = totLen - tLen2;
                step = 0.2;
                delays = 0:step:tLimit;
                %conf = 0.7; % 70% confidence
                %conf = 0.5;
                conf = 0.8;
                thresh = round(N*conf);
    %             eps = 0.01;
                eps = 0;

                %types = [33024 33025 33026 33027];
                types = [33025 33026 33027]; %-- Only SSVEP classes

                for typ = 1:numel(types)
                    ind(typ,:) = find(H_all{testSession}.EVENT.TYP==types(typ));
                    pos(typ,:) = H_all{testSession}.EVENT.POS(ind(typ,:));
                    class(typ,:) = (typ-1)*ones(size(pos(typ,:)));
                end

                class_v = class(:);
                pos_v = pos(:);
                [POS, I] = sort(pos_v); 
                CLASS = class_v(I);

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
                    %[Ytest_tmp d_tmp C] = mdm(Ptr(:,:,1:numel(delays)),COVtrain,Ytrain);  %classifies all segments available in a trial
                    [Ytest_tmp d_tmp C] = mdm(Ptr(:,:,1:N),COVtrain,Ytrain);  %classifies N first segments in a trial
                    [M F] = mode(Ytest_tmp); %retuns the most occuring element in Ytest_tmp and its frequency of occurence

                    d_norm = bsxfun(@rdivide, d_tmp, sum(d_tmp,2));
                    grad = sum(diff(d_norm));

                    if ( ((F > thresh) && (grad(M+1)<eps))) %Check if identified class has occured more than the threshold and that the SCMs are moving toward this class
                        Ytest(tr) = M;
                        delay_fin(tr) = N;

                    else
                        win = N+1;    
                        while ( (F <= thresh || grad(M+1) >= eps) && (win <= numel(delays)) ) % Check whether 1) the identified class has been majoritary in the last N data segments 2) whether the segments' SCM are moving into the direction of the identified class. And this can only be done within the available trial length determined by numel(delays)
                            [y d] = mdm(Ptr(:,:,win),COVtrain,Ytrain); %classify one more segment (sliding window)
                            Ytest_tmp = [Ytest_tmp(2:end) y]; %concatenate new class while leaving out the oldest
                            d_n = d/sum(d);
                            d_norm = [d_norm(2:end,:); d_n]; %concatenate new normalised distance while leaving out the oldest
                            grad = sum(diff(d_norm));
                            [M F] = mode(Ytest_tmp); %retuns the most occuring element in Ytest_tmp and its frequency of occurence
                            %thresh = round(numel(Ytest_tmp)*conf); %update treshold   
                            win = win+1;
                            sprintf('subject %d,  session %d,  tLen %d, trial %d,  segment# is: %d ...',sub, testSession, tLen2, tr, win)
                        end
                        if win > numel(delays) %No convergence within the trial length (9 sec)
                            Ytest(tr) = -1; %No class recognised;
                        else
                            Ytest(tr) = M;
                        end
                        delay_fin(tr) = win-N;
                    end
                end
                %Ytest_all(testSession, :, sub-5) = Ytest;
                Ytest_all(testSession, :, lenIdx, sub-5) = Ytest;
                %delay_fin_all(testSession, :, sub-5) = delay_fin;
                delay_fin_all(testSession, :, lenIdx, sub-5) = delay_fin;
                labels = CLASS';
                %ac(sub-5, testSession) = sum((labels-Ytest)==0)/(trialPerSession*4- numel(find(Ytest==-1)));
                %ac(testSession, lenIdx, sub-5) = sum((labels-Ytest)==0)/(trialPerSession*3- numel(find(Ytest==-1)));
                ac(testSession, lenIdx, sub-5) = sum((labels-Ytest)==0)/(trialPerSession*3);
    %             ac(testSession) = sum((labels-Ytest)==0)/(trialPerSession*4);
                %end
                % end
            end
        end
    end
end
for i = 1:size(ac,3)
    for j = 1:size(ac,2)
        acSi = ac(:,j,i);
        acSi = acSi(acSi~=0);
        subId(i) = i;
        subNbrOfSess(i) = length(acSi);
        subAcMean(j,i) = mean(acSi);
        subVar(j,i) = var(acSi);
    
        del_sub = delay_fin_all(:,:,j,i);
        del_sub = del_sub(:);
        del_sub = del_sub(del_sub~=0);
        del_sub_all(j,i) = mean((del_sub)*step);
    end
end
for j = 1:length(tlen)
    %po = subAcMean(j,:); 
    po = bsxfun(@min,subAcMean(j,:),0.9999999999999999);
    tLen = del_sub_all(j,:); 
    B = log2(3)+po.*log2(po)+(1-po).*log2((1-po)/(3-1));
    itr(j,:) = B.*(60./tLen);
end
%-- adding tlen to time
del_sub_all2 = bsxfun(@plus, del_sub_all',tlen)';
for j = 1:length(tlen)
    %po = subAcMean(j,:); 
    po = bsxfun(@min,subAcMean(j,:),0.9999999999999999);
    tLen = del_sub_all2(j,:); 
    B = log2(3)+po.*log2(po)+(1-po).*log2((1-po)/(3-1));
    itr2(j,:) = B.*(60./tLen);
end
classifWindow = 9-(tlen+1); %- The length over which the trial is actually classified.
classifNumb = classifWindow/0.2; %-- Number of classifications output in a trial
del_sub_all3 = bsxfun(@plus, del_sub_all',tlen./classifNumb)';
for j = 1:length(tlen)
    %po = subAcMean(j,:); 
    po = bsxfun(@min,subAcMean(j,:),0.9999999999999999);
    tLen = del_sub_all3(j,:); 
    B = log2(3)+po.*log2(po)+(1-po).*log2((1-po)/(3-1));
    itr3(j,:) = B.*(60./tLen);
end
save('online_curve_tlen_3class_new.mat', 'tlen', 'subAcMean', 'subVar', 'del_sub_all', 'itr', 'del_sub_all2', 'itr2', 'del_sub_all3', 'itr3');
