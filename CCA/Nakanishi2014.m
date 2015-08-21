% This is an implementation of Nakanishi et al. method for SSVEP target
% recognition: 
% Nakanishi, Masaki, et al. "A high-speed brain speller using steady-state 
% visual evoked potentials." International journal of neural systems 24.06 
% (2014): 1450019.
% Requirements:
% - Matlab 7 or later 
% - Biosig toolbox
% Author: Emmanuel K. Kalunga

clear all
%harmonicsV = [0 1 2 3 4 5 6];
%har = 6;
tLen = 4;
delay = 2;
wind = [0 tLen];
%harmonics = harmonicsV(har)+1;
harmonics = 6;
trial_limits = wind+delay;

target_frequencies = [12.989 21.051 17.017 ]; % [12.98, 21.07 17.07];
event_types = [33024, 33025, 33026, 33027];
noTarget = 1;
shift = 0;
bootstrap = 50;

%% Generate stimuli frequency
Fs = 256;
f = target_frequencies;
%t = ( 0:(length(x{1})*length(S)) )/Fs;
t = (0:tLen*Fs)/Fs;
t = t(1:end-1);
for fr = 1:length(f)
    for harm = 1:harmonics+1 % Harmonics + Fundamental
        y(harm*2-1,:) = sin(2*(harm)*pi*f(fr)*t);
        y(harm*2,:)   = cos(2*(harm)*pi*f(fr)*t);
    end
    Y{fr} = y;
end

for sub = 6:17
    clear x_all H_all S_all P X Pm PSD class
    %% Load data
    disp('********************************************************');
    disp(['Load data subject', num2str(sub)]);
    [S_all, H_all] = loaddata(sub); %Returns cells of data from all available sessions
    
    %enhancement = 'cca'; %cca or ica or none
    %classifier = 'svm'; %or 'svm'
    channels_number = 8;
    
    %Fs = H_all{1}.SampleRate;
    nbrSessions = length(S_all);
    sessions = 1:nbrSessions;
    
    %% PREPROCESSING
    disp('---------------------------------------------------------------')
    disp(['Preprocessing subject ', num2str(sub)]);
    % DESIGN FILTER
    Fn = Fs/2;
    n_butter = 10;
    [b, a] = butter(n_butter, 12./Fn, 'high');
    % REMOVE 50Hz & BAND-PASS FILTER 
    for sessions = 1:nbrSessions
        x_all{sessions} = filtfilt(b, a, S_all{sessions});
        x_all{sessions} = remove5060hz(x_all{sessions}, H_all{sessions}, 'PCA 50'); % Biosig function
    end
    X = get_trials(x_all, H_all, tLen, delay);
    
    %% BOOTSTRAP CROSS VALIDATION
    for testSession = 1:bootstrap
        clear r class
        fprintf('subject %d, bootstrap %d ...\n', sub, testSession);
        trials = 1:size(X{1},3);
        trialPerSession = size(X{1},3)/nbrSessions;
        testTrials = randsample(1:trialPerSession*nbrSessions, trialPerSession);
        trainTrials = setxor(trials, testTrials);
        %% TRAINING PHASE (Get the averaged reference signals and the cca based spatial filters)
        for cl = 1:length(target_frequencies) % Only SSVEP classes
            Xk{cl} = mean(X{cl+1}(:,:,trainTrials),3);
        end
        true_labels = [ones(1, trialPerSession) 2*ones(1, trialPerSession) 3*ones(1, trialPerSession)];
        Xtest = cat(3, X{2}(:,:,testTrials), X{3}(:,:,testTrials), X{4}(:,:,testTrials));
        for trial = 1:size(Xtest,3)        
            for k = 1:length(target_frequencies) % for ech feature (we only have frequency. No phases) 
                % 1) Get CCA
                [wx wy R] = cca(Xtest(:,:,trial),Y{k});
                coef1(k) = corrcoef(Xtest(:,:,trial)'*wx(:,8), Y{k}'*wy(:,8));
                % 2) Get W_XXk
                [W_XXk wy R] = cca(Xtest(:,:,trial),Xk{k});
                coef2(k) = corrcoef(Xtest(:,:,trial)'*W_XXk(:,8), Xk{k}'*W_XXk(:,8));
                % 3) Get W_XY
                [W_XY wy R] = cca(Xtest(:,:,trial),Y{k});
                coef3(k) = corrcoef(Xtest(:,:,trial)'*W_XY(:,8), Xk{k}'*W_XY(:,8));
                % 4) Get W_XkY
                [W_XkY wy R] = cca(Xk{k},Y{k});
                coef4(k) = corrcoef(Xtest(:,:,trial)'*W_XkY(:,8), Xk{k}'*W_XkY(:,8));
            end
            coef = sum([sign(coef1).*coef1.^2; sign(coef2).*coef2.^2; sign(coef3).*coef3.^2; sign(coef4).*coef4.^2]);
            [r(trial) class(trial)] = max(coef);
        end
        res = true_labels - class;
        ac(sub-5, testSession) = sum(sum(res == 0))/numel(true_labels);       
    end
end
%% REPORT RESULTS
acSubMean = mean(ac,2);
acSubStd = std(ac,[],2);
acMean = mean(acSubMean);
acStd = std(acSubMean); 

po = bsxfun(@min,acSubMean,0.9999);
B = log2(3)+po.*log2(po)+(1-po).*log2((1-po)/(3-1));
itr = B*(60/tLen);

subjects = {'sub 1', 'sub 2', 'sub 3', 'sub 4', 'sub 5', 'sub 6', 'sub 7', 'sub 8', 'sub 9', 'sub 10', 'sub 11', 'sub 12', 'Mean'};
headers = {'accuracy', 'error', 'itr'}
disp('---------------------------------------------------');
disp('Accuracy (%) of each subject');
disp('---------------------------------------------------');
displaytable([acSubMean*100, acSubStd*100, itr; acMean*100, acStd*100, mean(itr)],headers,10,{'.1f'},subjects)
disp('---------------------------------------------------');

save('nakanishi2014.mat', 'ac', 'acSubMean', 'acSubStd', 'acMean', 'acStd', 'itr');


