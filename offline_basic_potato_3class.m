clear all

tLen = 6; %78.41
delay = 0;
bootstrap = 50;

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
        chan = [1:3;4:6;7:9;10:12;13:15;16:18;19:21;22:24];
        S = get_trials(S_all, H_all, tLen(l), delay);
      
        % 3) Covariance matrices of all trialssummed up per class
        Nt = size(X{1},3); %Number of trial
        for k = 1:Nt %loop for evrey trial
            for cl = 1:4
                P{cl}(:,:,k) = shcovft((X{cl}(:,:,k))'); % J. Schaefer Shrinkage covariance from Barachant toolbox
            end
        end 

        for testSession = 1:bootstrap
            fprintf('subject %d, bootstrap %d ...\n', sub, testSession);
            trials = 1:size(P{1},3);
            trialPerSession = size(P{1},3)/nbrSessions;
            testTrials = randsample(1:trialPerSession*nbrSessions, trialPerSession); 
            %testTrials = (trialPerSession*testSession-trialPerSession+1):(trialPerSession*testSession);
            trainTrials = setxor(trials, testTrials);
            %% TRAINING PHASE
            %trainSessions = setxor(sessions, testSession);
            COVtrain = cat(3, P{2}(:,:,trainTrials), P{3}(:,:,trainTrials), P{4}(:,:,trainTrials));
            Ytrain = [ones(1,numel(trainTrials)) 2*ones(1,numel(trainTrials)) 3*ones(1,numel(trainTrials))];
            
            for rp = 1:2 % without and with potato
                if rp == 1
                    %- No Riem Potato
                    COVtrain_filt = COVtrain;
                    Ytrain_filt = Ytrain;
                else
                    %********FILTER OUT OUTLIERS FROM TRAINIG SET WITH RIEMANNIAN POTATO
                    for cl = 1:3
                        % get mean of class
                        P_filt{cl} = P{cl+1}(:,:,trainTrials);
                        cont = 1;
                        contIdx = 0;
                        while cont == 1
                            contIdx = contIdx + 1;
                            dis = [];
                            z = [];
                            Bc(:,:,cl) = mean_covariances(P_filt{cl},'riemann');
                            % get distance of each matrice to it class mean
                            for i = 1:size(P_filt{cl},3)
                                dis(i,cl) = distance(P_filt{cl}(:,:,i), Bc(:,:,cl), 'riemann');
                            end
                            % get the geometric mean of the distances
                            mu(cl) = exp( mean(log(dis(:,cl))) );
                            % get the arithmetic mean of the distances (to be used in Wiener Entropy)
                            mu_ar(cl) = mean(dis(:,cl));
                            if contIdx == 1                                
                                wiener_entropy(sub-5, testSession, cl,1) = -10*log10(mu(cl)/mu_ar(cl)); %Before filtering
                            end
                            % get geometric standard dev
                            sig(cl) = exp( sqrt(mean((log(dis(:,cl)/mu(cl))).^2)) );
                            % get the z-score
                            z(:,cl) = log(dis(:,cl)/mu(cl))/log(sig(cl));
                            % Threshold z-score
                            z_th(cl) = 2.2*sig(cl);
                            % Identify outliers (lying beyond z_th)
                            [outliers{cl} ind_out{cl}] = find(z(:,cl) > z_th(cl));
                            if isempty(ind_out{cl})
                                cont = 0;
                                wiener_entropy(sub-5, testSession, cl,2) = -10*log10(mu(cl)/mu_ar(cl)); %- Value after filtering
                            else
                                P_filt{cl} = P_filt{cl}(:,:, setxor([1:size(P_filt{cl},3)], outliers{cl})); 
                                Bc(:,:,cl) = mean_covariances(P_filt{cl},'riemann');
                            end      
                        end
                        Bc_filt(:,:,cl) = Bc(:,:,cl);
                        %P_filt{cl} = P{cl};
                    end
                    
                    A_filt = cat(3, P_filt{1}, P_filt{2}, P_filt{3});

                    COVtrain_filt = A_filt;
                    Ytrain_filt = [ones(1,size(P_filt{1},3)) 2*ones(1,size(P_filt{2},3)) 3*ones(1,size(P_filt{3},3))];
                end
                        
                B_filt = mean_covariances(COVtrain_filt,'riemann');

                %%   EVALUATION PHASE  		                               **
                %********************************************************************
                labels = [ones(1, trialPerSession) 2*ones(1, trialPerSession) 3*ones(1, trialPerSession)];
                COVtest = cat(3, P{2}(:,:,testTrials), P{3}(:,:,testTrials), P{4}(:,:,testTrials));
                %************FILTER OUT OUTLIERS FROM TEST SET WITH RIEMANNIAN POTATO
                B_test = mean_covariances(COVtest,'riemann');  

                % Classification by Remannian Distance
                [Ytest d C] = mdm(COVtest,COVtrain_filt,Ytrain_filt);
                ytest{testSession} = Ytest;
                %     reshape(Ytest,trialPerSession,4)'
                ac(sub-5, testSession, rp) = sum((labels-Ytest)==0)/(numel(Ytest));
                %numTr(sub-5, testSession) = numel(Ytest); 
            end
        end
        clear COVtrain_filt ind_out COVtest_filt ind_out2 dis dis2
    end
end

for rp = 1:2
    for i = 1:size(ac,1)
        acSi = ac(i,:,rp);
        acSi = acSi(acSi~=0);
        subId(rp,i) = i;
        subNbrOfSess(rp,i) = length(acSi);
        subAcMean(rp,i) = mean(acSi);
        subStd(rp,i) = std(acSi);
    end
    resMatrix(:,:,rp) = [subId(rp,:)' subNbrOfSess(rp,:)' subAcMean(rp,:)' subStd(rp,:)'];
    resMean0 = mean(resMatrix(:,:,rp));
    resMean0(2) = sum(resMatrix(:,2,rp));
    resMean(:,rp) = resMean0(2:end);
end

po = bsxfun(@min,resMatrix(:,3,1),0.9999);
B = log2(3)+po.*log2(po)+(1-po).*log2((1-po)/(3-1));
itr = B*(60/tLen);

subjects = {'sub 1', 'sub 2', 'sub 3', 'sub 4', 'sub 5', 'sub 6', 'sub 7', 'sub 8', 'sub 9', 'sub 10', 'sub 11', 'sub 12', 'Mean'};
headers = {'accuracy', 'error'}
disp('---------------------------------------------------');
disp('Accuracy (%) of each subject without outlier removal');
disp('---------------------------------------------------');
displaytable([resMatrix(:,[3,4],1)*100; resMean(2,1)*100 resMean(3,1)*100],headers,10,{'.1f'},subjects)
disp('---------------------------------------------------');

disp('---------------------------------------------------');
disp('Accuracy (%) of each subject with outlier removal');
disp('---------------------------------------------------');
displaytable([resMatrix(:,[3,4],2)*100; resMean(2,2)*100 resMean(3,2)*100],headers,10,{'.1f'},subjects)
disp('---------------------------------------------------');

save('offline_basic_potato_3class.mat', 'resMatrix', 'resMean', 'tLen', 'itr');

