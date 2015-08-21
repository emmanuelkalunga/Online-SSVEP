% This is an implementation of CCA for SSVEP recognition described in Lin2007
% Lin Z. et al, Frequency Recognition Based on Canonical Correlation
% Analysis for SSVEP-Based BCIs, TBME, 2007
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
    
    %% CLASSIFICATION WITH CCA
    for cl = 1:length(target_frequencies)
        for trial = 1: size(X{cl+1},3)
            for fr = 1:length(target_frequencies) 
                [wx wy R] = cca(X{cl+1}(:,:,trial),Y{fr});
                coef(fr) = max(diag(R));
            end
            [r(trial,cl) class(trial,cl)] = max(coef);          
        end
    end
    true_labels = repmat([1 2 3],size(X{1},3),1);
    res = true_labels - class;
    ac(sub-5) = sum(sum(res == 0))/numel(true_labels);    
end
po = bsxfun(@min,ac,0.9999);
B = log2(3)+po.*log2(po)+(1-po).*log2((1-po)/(3-1));
itr = B*(60/tLen);

colhead = {'sub 1', 'sub 2', 'sub 3', 'sub 4', 'sub 5', 'sub 6', 'sub 7', 'sub 8', 'sub 9', 'sub 10', 'sub 11', 'sub 12', 'Mean'};
rowhead = {'acc', 'itr'}
disp('------------------------------------------------------------------');
disp('Accuracy (%) of each subject');
disp('------------------------------------------------------------------');
displaytable([ac*100 mean(ac)*100; itr mean(itr)],colhead,6,{'.1f'}, rowhead)
disp('------------------------------------------------------------------');

save('cca_Lin2007.mat','ac', 'itr', 'tLen');


