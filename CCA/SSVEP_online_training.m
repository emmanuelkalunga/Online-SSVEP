function [chanSel, a, b, Hd, WX, WY, Training, Group, Pvector, svm_model, tresh1, tresh2, tresh3, WF] = SSVEP_online_training(S, H, harmonics, target_frequencies, event_types, trial_limits, channels_number, noTarget, classifier, wind, enhancement)
%% Extract

%% Filter eeg Signal

%% Design Filter
Fs = H{1}.SampleRate;
Fn = Fs/2;
n_butter = 10;

%!!!!!!!!!!!!!!!!!!!!!!FIND METHOD!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

d  = fdesign.notch('N,F0,Q,Ap',6,50/Fs,10,1);
Hd = design(d);



for sessions = 1:length(S)
   %filtfilt(Hd,S{sessions});
    S{sessions} = filter(Hd, S{sessions}); %Notch filter 50 Hz
end

% for sessions = 1:length(S)
%    [S{sessions}, H{sessions}] = remove5060hz(S{sessions}, H{sessions}, 'PCA 50');
% end





switch harmonics
    case 1
        [b, a] = butter(n_butter, [12 26]./Fn, 'bandpass');
    case 2
        [b, a] = butter(n_butter, [12 45]./Fn, 'bandpass');
    case 3
        [b, a] = butter(n_butter, [12 66]./Fn, 'bandpass');
    case 4
        [b, a] = butter(n_butter, [12 89]./Fn, 'bandpass');
    case 5
        [b, a] = butter(n_butter, [12 110]./Fn, 'bandpass');
    case 6
        [b, a] = butter(n_butter, [12 130]./Fn, 'bandpass');
end
%% Filter

for sessions = 1:length(S)
    x{sessions} = filtfilt(b, a, S{sessions});
end
% for sessions = 1:length(S)
%     x{sessions} = S{sessions};
% end

f = target_frequencies;

%% Generate stimuli frequency
if strcmp(enhancement,'cca')
        % f = [12.72, 21.05 16.8];
    t = ( 0:(length(x{1})*length(S)) )/Fs;      % SHOULD BE CHANGE TO length(x{1}) + length(x{2})
    for fr = 1:length(f)
        for harm = 1:harmonics
            y(harm*2-1,:) = sin(2*(harm)*pi*f(fr)*t);
            y(harm*2,:)   = cos(2*(harm)*pi*f(fr)*t);

        %         y(harm*2-1,:) = square(2*(harm)*pi*f(fr)*t);
        %         y(harm*2,:)   = square(2*(harm)*pi*f(fr)*t+(2*pi*(harm*f(fr))/4));
        end
        Y{fr} = y;
    end
end
%% TRAINING PHASE
%% Center the data
for sessions = 1:length(S)
    x_not_centered{sessions} = x{sessions};
    x{sessions} = bsxfun(@minus, x{sessions}, mean(x{sessions},1));
end

%% Arrange trials per classes
% types = [33025, 33026, 33027];
types = event_types(2:length(event_types));                 %CHECK WHERE THE TRIAL REALLY START
for typ = 1:length(types)
    Xtmp2 = [];
    for session = 1:length(S)
        markers = H{session}.EVENT.POS(find(H{session}.EVENT.TYP ==  types(typ)));
    %         markers = markers - 2*H{session}.SampleRate; % this is done to create the possibility of having a negative delays 
        [trials sz] = trigg(x{session}, markers, round(trial_limits(1)*(H{session}.SampleRate)+1), round(trial_limits(2)*H{session}.SampleRate));
    %         [trials sz] = trigg(x{session}, markers, round(1*(H{session}.SampleRate)+1), round(6*H{session}.SampleRate));

        Xtmp1 = reshape(trials, sz); 
        Xtmp2 = cat(3, Xtmp2, Xtmp1);
    end
    X{typ} = Xtmp2;
    
    for trial = 2:(size(X{typ}(1,:,:),3))
        for sh = 1:ceil(Fs*((  (1/f(1))/2  )))
       % for sh = 1:ceil(Fs*((  (1/f(typ))/2  )))
            Yt = circshift((X{typ}(1,:,trial))',sh);
            rshift(sh) = abs(corr(Yt,(X{typ}(1,:,1))'));
        end
        [rho position] = max(rshift);
        X{typ}(:,:,trial)  = (circshift((X{typ}(:,:,trial))', position))';
    end
    X_av{typ} = mean(X{typ},3);
    X{typ} = Xtmp2; 
end
% clear Y
% Y = X_av;
sz = size(Xtmp2);
ws = sz(2);

%% Channels selection
if strcmp(enhancement,'cca')
    N = channels_number;
    M = 5;
    for fr = 1:length(f)
        for ch = 1:H{1}.NS
            for ch2 = 1:H{1}.NS
                Edist(ch2) = dist(X_av{fr}(ch,:), (X_av{fr}(ch2,:))');
            end   
    %         figure, plot(Edist,'*-'), title(['channel ',num2str(ch)]);
    %         pause
            [A B] = sort(Edist);
            X_all = X_av{fr}(B(1:M+1),:);
            Yf_all = Y{fr}(:,1:ws);
            Xf = X_all;
            Yf = Yf_all(:,1:ws);
            [Wx Wy r] = cca(Xf,Yf);
            rd = diag(r);
            ind(fr,ch) = max(rd);
        end
    end
    ind_av = mean(ind);
    [A B] = sort(ind_av);
    chanSel = B(length(ind_av)-N+1:length(ind_av));
end
   
    %% CCA FILTERS
    for fr = 1:length(target_frequencies)
        Xf = X_av{fr}(chanSel,:);
        Yf = Y{fr}(:,1:ws);
        [WX{fr} WY{fr} R{fr}] = cca(Xf,Yf);
    end

    %% Selecting 1 cca projection vector among the 3
    for fr = 1:length(f)
        Yf = Y{1}(:,1:ws);
        for trial = 1:size(X{1},3)
            r1(fr,trial) = abs(corr ((X{1}(chanSel,:,trial))'*WX{fr}(:,channels_number), Yf'*WY{fr}(:,channels_number)) );
        end

        Yf = Y{2}(:,1:ws);
        for trial = 1:size(X{2},3)
            r2(fr,trial) = abs(corr ((X{2}(chanSel,:,trial))'*WX{fr}(:,channels_number), Yf'*WY{fr}(:,channels_number)) );
        end

        Yf = Y{3}(:,1:ws);
        for trial = 1:size(X{3},3)
            r3(fr,trial) = abs(corr ((X{3}(chanSel,:,trial))'*WX{fr}(:,channels_number), Yf'*WY{fr}(:,channels_number)) );
        end

    end
    temp = mean([mean(r1,2) mean(r2,2) mean(r3,2)],2);
    [val pos] = max(temp);
    Pvector = pos;
    mixmat = 0;
%end
%% Get ICA Demiming matrix
if strcmp(enhancement,'ica')
    combSig = [];
    for sess = 1:max(size(x))
        combSig = [combSig;x{sess}];
    end
    [icasig, A, W] = fastica(combSig');
    for p = 1:min(size(icasig))
        FF = fft(icasig(p,:));
        fftamp = 2*abs(FF(1:length(FF)/2+1));
        fftamp = fftamp./sqrt(sum(fftamp.^2)); %normalize the frequency domain data    
        freq = Fs/2*linspace(0,1,length(FF)/2+1);
        sums(p)=sum(fftamp);
        for fr = 1:length(target_frequencies)
            tmp = 0;
            for harm = 1:harmonics
                tmp = tmp + max(fftamp(find( freq >= (f(fr)*harm)-(0.5) & freq <= (f(fr)*harm)+(0.5))));
            end
            amp_tmp(fr) = tmp;
        end
        ampica(p,:) = amp_tmp;   
    end
    amprank = sum(ampica,2)./sums';
    [val ind] = sort(amprank);
    mixmat = sum(W(ind(length(ind)-0:length(ind)),:),1); %Take the tow best ICs
    
    chanSel = 0;
    WX = 1;
    WY = 1;
    Pvector = 1;

    %PLOT 
%     sigica = combSig*mixmat';
%     FF = fft(sigica');
%     fftamp = 2*abs(FF(1:length(FF)/2+1));
%     fftamp = fftamp./sqrt(sum(fftamp.^2));
%     figure(10)
%     plot(freq,fftamp), axis([0 120 0 0.07]);
end

%% Get Best Channel with no signal enhancement
if strcmp(enhancement,'none')
    combSig = [];
    for sess = 1:max(size(x))
        combSig = [combSig;x{sess}];
    end
   
    for p = 1:min(size(combSig))
        FF = fft(combSig(:,p)');
        fftamp = 2*abs(FF(1:length(FF)/2+1));
        fftamp = fftamp./sqrt(sum(fftamp.^2)); %normalize the frequency domain data    
        freq = Fs/2*linspace(0,1,length(FF)/2+1);
        sums(p)=sum(fftamp);
        for fr = 1:length(target_frequencies)
            tmp = 0;
            for harm = 1:harmonics
                tmp = tmp + max(fftamp(find( freq >= (f(fr)*harm)-(0.5) & freq <= (f(fr)*harm)+(0.5))));
            end
            amp_tmp(fr) = tmp;
        end
        ampnone(p,:) = amp_tmp;   
    end
    amprank = sum(ampnone,2)./sums';
    [val ind] = sort(amprank);
    chanSel = ind(length(ind));
    
    mixmat = 1;
    WX = 1;
    WY = 1;
    Pvector = 1;
    %PLOT 
%     sigica = combSig*mixmat';
%     FF = fft(sigica');
%     fftamp = 2*abs(FF(1:length(FF)/2+1));
%     fftamp = fftamp./sqrt(sum(fftamp.^2));
%     figure(10)
%     plot(freq,fftamp), axis([0 120 0 0.07]);
end







%%  no-target Trials
if noTarget == 1
    typ = event_types(1);
    Xtmp2 = [];
    for session = 1:length(S)
        markers = H{session}.EVENT.POS(find(H{session}.EVENT.TYP ==  typ));
        [trials sz] = trigg(x{session}, markers, round(trial_limits(1)*(H{session}.SampleRate)+1), round(trial_limits(2)*H{session}.SampleRate));
        Xtmp1 = reshape(trials, sz); 
        Xtmp2 = cat(3, Xtmp2, Xtmp1);
    end
    X_none = Xtmp2;
end
%%






%% k-NN training  & SVM training
if strcmp(classifier,'knn') || strcmp(classifier,'svm')
    for trial = 1:size(X{1},3)
        if strcmp(enhancement,'ica')
            filtSig = ( (X{1}(:,:,trial))'* mixmat' )'; %Project trial on ica filter (enhancement)
        elseif strcmp(enhancement,'cca')
            filtSig = ( (X{1}(chanSel,:,trial))'* WX{Pvector}(:,channels_number) )'; % Project trial on CCA filter (enhancement)
        elseif strcmp(enhancement,'none')
            filtSig = X{1}(chanSel,:,trial); % taking the best channel with no enhancement    
        end
        tmp = abs(fft( filtSig ));
        Training1(trial,:) = tmp(1:length(tmp)/4);
    end
    Group1 = 13*ones(size(X{1},3),1);
    
    for trial = 1:size(X{2},3)
         if strcmp(enhancement,'ica')
            filtSig = ( (X{2}(:,:,trial))'* mixmat' )'; %Project trial on ica filter (enhancement)
         elseif strcmp(enhancement,'cca')
            filtSig = ( (X{2}(chanSel,:,trial))'* WX{Pvector}(:,channels_number) )'; % Project trial on CCA filter (enhancement)
         elseif strcmp(enhancement,'none')
            filtSig = X{2}(chanSel,:,trial); % taking the best channel with no enhancement
         end  
         tmp = abs(fft( filtSig ));
         Training2(trial,:) = tmp(1:length(tmp)/4);
    end
    Group2 = 21*ones(size(X{1},3),1);
    
    for trial = 1:size(X{3},3)
         if strcmp(enhancement,'ica')
            filtSig = ( (X{3}(:,:,trial))'* mixmat' )'; %Project trial on ica filter (enhancement)
         elseif strcmp(enhancement,'cca')
            filtSig = ( (X{3}(chanSel,:,trial))'* WX{Pvector}(:,channels_number) )'; % Project trial on CCA filter (enhancement)
         elseif strcmp(enhancement,'none')
            filtSig = X{3}(chanSel,:,trial); % taking the best channel with no enhancement
         end  
         tmp = abs(fft( filtSig ));
         Training3(trial,:) = tmp(1:length(tmp)/4);
    end
    Group3 = 17*ones(size(X{1},3),1);
    
    if noTarget == 1
        for trial = 1:size(X_none,3)
            if strcmp(enhancement,'ica')
                filtSig = ( (X_none(:,:,trial))'* mixmat' )'; %Project trial on ica filter (enhancement)
            elseif strcmp(enhancement,'cca')
                filtSig = ( (X_none(chanSel,:,trial))'* WX{Pvector}(:,channels_number) )'; % Project trial on CCA filter (enhancement)
            elseif strcmp(enhancement,'none')
                filtSig = X_none(chanSel,:,trial); % taking the best channel with no enhancement
            end  
            tmp = abs(fft( filtSig ));
            Training4(trial,:) = tmp(1:length(tmp)/4);
        end
        Group4 = 0*ones(size(X_none,3),1);
    end  
    %Normalize data
    Training1 = bsxfun(@rdivide, Training1, sqrt(sum(Training1.^2, 2)));
    Training2 = bsxfun(@rdivide, Training2, sqrt(sum(Training2.^2, 2)));
    Training3 = bsxfun(@rdivide, Training3, sqrt(sum(Training3.^2, 2)));
    if noTarget == 1
        Training4 = bsxfun(@rdivide, Training4, sqrt(sum(Training4.^2, 2)));
    end
    
    % Training1 = bsxfun(@rdivide, Training1, max(Training1,[],2));
    % Training2 = bsxfun(@rdivide, Training2, max(Training2,[],2));
    % Training3 = bsxfun(@rdivide, Training3, max(Training3,[],2));
    
    Center1 = median(Training1);
    Center2 = median(Training2);
    Center3 = median(Training3);
    if noTarget == 1
        Center4 = median(Training4);
    end
    
    K = size(Training1,1) - 6;
    [sorted1,index1]  = distfun(Center1, Training1, K);
    [sorted2,index2]  = distfun(Center2, Training2, K);
    [sorted3,index3]  = distfun(Center3, Training3, K);
    if noTarget == 1
        [sorted4,index4]  = distfun(Center4, Training4, K);
    end
    
    Training1 = Training1(index1,:);
    Training2 = Training2(index2,:);
    Training3 = Training3(index3,:);
    if noTarget == 1
        Training4 = Training4(index4,:);
    end
    
    %disp('size Training1 is')
   % disp(size(Training1));
    
    Group1 = Group1(index1);
    Group2 = Group2(index2);
    Group3 = Group3(index3);
    if noTarget == 1
        Group4 = Group4(index4);
    end
    
    if noTarget == 0
        Training = [Training1; Training2; Training3];
        Group = [Group1; Group2; Group3];
    elseif noTarget == 1
        Training = [Training1; Training2; Training3; Training4];
        Group = [Group1; Group2; Group3; Group4];
    end
    
    WF = 1;
    
    if strcmp(classifier,'svm')
        %svm_model = svmtrain(Group, Training);
        %svm_model = svmtrain(Group, Training, '-c 1 -g 0.07 -b 1');
        svm_model = svmtrain(Group, Training, '-c 1 -g 0.5 -b 1');
    else
        svm_model = 0;
    end
    disp('SVM and KNN trained');
end

if ( strcmp(classifier,'maxfreq') || strcmp(classifier,'cca') )
    for fr = 1:length(f) % plus 1 for no-target
        for trial = 1:size(X{fr},3)
            if strcmp(enhancement,'ica')
                Xt = ( (X{fr}(:,:,trial))'* mixmat' )'; %Project trial on ica filter (enhancement)
            elseif strcmp(enhancement,'cca')
                Xt = ( (X{fr}(chanSel,:,trial))'* WX{Pvector}(:,channels_number) )'; % Project trial on CCA filter (enhancement)
            elseif strcmp(enhancement,'none')
                Xt = X{fr}(chanSel,:,trial); % taking the best channel with no enhancement
            end
            bff = fft(Xt,1000);
            ff = Fs/2*linspace(0,1,length(bff)/2+1);
            fftamp = 2*abs(bff(1:length(bff)/2+1));
            fftamp = fftamp./sqrt(sum(fftamp.^2)); %normalize the frequency domain data
            tmp = 0;
            tmp_noise = 0;
            for harm = 1:harmonics
                tmp = tmp + max(fftamp(find( ff >= (f(fr)*harm)-(0.5) & ff <= (f(fr)*harm)+(0.5))));
                tmp_noise = tmp_noise + max(fftamp(find( ff >= (f(fr)*harm)-(1.5) & ff < (f(fr)*harm)-(0.5)))) + max(fftamp(find( ff > (f(fr)*harm)+(0.5) & ff <= (f(fr)*harm)+10.5)));
            end
            %____Get the fft amplitude of the other two frequecies_____
            i1 = fr + 1; %i1 and i2 locate frequency other than the one indicated by fr
            if i1 > 3
                i1 = 1;
            end
            i2 = fr - 1;
            if i2 < 1
                i2 = 3;
            end
            tmp_others = 0;
            for harm = 1:harmonics
                tmp_others = tmp_others + max(fftamp(find( ff >= (f(i1)*harm)-(0.5) & ff <= (f(i1)*harm)+(0.5)))) + max(fftamp(find( ff >= (f(i2)*harm)-(0.5) & ff <= (f(i2)*harm)+(0.5))));
            end
            %__________________________________________________________
            amp(trial) = tmp;
            amp_others(trial) = tmp_others;
            total(trial) = sum(fftamp);
            amp_noise(trial) = tmp_noise;
        end
        amp_all(fr, :) = amp;
        amp_others_all(fr, :) = amp_others;
        total_all(fr, :) = total;
        amp_noise_all(fr, :) = amp_noise;
    end
    %_______Select best trials_________________________________________
    ratio1 = amp_all./total_all;
    ratio2 = amp_all./amp_others_all;
    ratio3 = amp_all./amp_noise_all;
    
    [v1 p1] = sort(ratio1,2);
    [v2 p2] = sort(ratio2,2);
    [v3 p3] = sort(ratio3,2);
    
    selected_trials = p1(:,length(p1)-4:length(p1)); % n = 5
    selected_ratios1 = v1(:,length(v1)-4:length(v1)); % n = 5;
    selected_ratios2 = v2(:,length(v2)-4:length(v2)); % n = 5;
    selected_ratios3 = v3(:,length(v3)-4:length(v3)); % n = 5;
    
    tresh3 = selected_ratios3(:,1);
    
    if (strcmp(classifier,'maxfreq'))
        for fr = 1:length(f)
            amp_all_selected(fr,:) = amp_all(fr,selected_trials(fr,:));
        end
            amp_norm = median(amp_all_selected,2)/max(median(amp_all_selected,2)); %Normalized amp
            BL = amp_norm; %Base Line
    end
    if (strcmp(classifier,'cca'))
        if strcmp(enhancement,'cca')
            for fr = 1:length(f)
                for trial = 1:length(selected_trials(fr,:))
                    Xt = X{fr}(chanSel,:,selected_trials(fr,trial));
                    Yf = Y{fr}(:,1:ws);
                    [Wx Wy r] = cca(Xt, Yf); 
                    rd = diag(r);
                    rho(trial) = max(rd); 
                end
                rho_all(fr, :) = rho;
            end
            rho_norm = mean(rho_all,2)/max(mean(rho_all,2)); %Normalized rho
            BL = rho_norm; %Base Line
        end
    end
        WF = BL.^(-1); %Weighting Factor
        Training = 0;
        Group = 0;
        svm_model = 0;
end

%% Tresholds for no-target rejection
if noTarget == 1
    if (strcmp(classifier,'maxfreq'))
        tresh1 = min(amp_all,[],2) - std(amp_all,[],2);

        for fr = 1:length(f)
            i1 = fr + 1;
            if i1 > 3
                i1 = 1;
            end
            i2 = fr - 1;
            if i2 < 1
                i2 = 3;
            end
            for trial = 1:size(X{1},3)
                if strcmp(enhancement,'ica')
                    Xt = ( (X_none(:,:,trial))'* mixmat' )'; %Project trial on ica filter (enhancement)
                elseif strcmp(enhancement,'cca')
                    Xt = ( (X_none(chanSel,:,trial))'* WX{Pvector}(:,channels_number) )'; % Project trial on CCA filter (enhancement)
                elseif strcmp(enhancement,'none')
                    Xt = X_none(chanSel,:,trial); % taking the best channel with no enhancement    
                end
                bff = fft(Xt,1000);
                ff = Fs/2*linspace(0,1,length(bff)/2+1);
                fftamp = 2*abs(bff(1:length(bff)/2+1));
                fftamp = fftamp./sqrt(sum(fftamp.^2)); %normalize the frequency domain data
                tmp = 0;
                tmp_noise = 0;
                for harm = 1:harmonics
                    tmp = tmp + max(fftamp(find( ff >= (f(fr)*harm)-(0.5) & ff <= (f(fr)*harm)+(0.5)))); 
                    tmp_noise = tmp_noise + max(fftamp(find( ff >= (f(fr)*harm)-(1.5) & ff < (f(fr)*harm)-(0.5)))) + max(fftamp(find( ff > (f(fr)*harm)+(0.5) & ff <= (f(fr)*harm)+10.5)));
                end

                %____Get the fft amplitude of the other two frequecies_____
                tmp_others = 0;
                for harm = 1:harmonics
                    tmp_others = tmp_others + max(fftamp(find( ff >= (f(i1)*harm)-(0.5) & ff <= (f(i1)*harm)+(0.5)))) + max(fftamp(find( ff >= (f(i2)*harm)-(0.5) & ff <= (f(i2)*harm)+(0.5)))); 
                end
                %__________________________________________________________

                amp(trial) = tmp;  
                amp_others(trial) = tmp_others;
                total(trial) = sum(fftamp);
                amp_noise_none(trial) = tmp_noise;
            end
            amp_all_none(fr, :) = amp;
            amp_others_all_none(fr, :) = amp_others;
            total_all_none(fr, :) = total;
            amp_noise_all_none(fr, :) = amp_noise_none;
        end
        % figure, bar(amp_all_none);
        ratio1_none = amp_all_none./total_all_none;
        ratio2_none = amp_all_none./amp_others_all_none;
        ratio3_none = amp_all_none./amp_noise_all_none;
        [v1_none p1_none] = sort(ratio1_none,2);
        [v2_none p2_none] = sort(ratio2_none,2);
        [v3_none p3_none] = sort(ratio3_none,2);

        selected_trials_none1 = p1_none(:,length(p1_none)-4:length(p1_none)); % n = 5
        selected_ratios_none1 = v1_none(:,length(v1_none)-4:length(v1_none)); % n = 5

        selected_trials_none2 = p2_none(:,length(p2_none)-4:length(p2_none)); % n = 5
        selected_ratios_none2 = v2_none(:,length(v2_none)-4:length(v2_none)); % n = 5

        selected_ratios_none3 = v3_none(:,length(v3_none)-4:length(v3_none)); % n = 5

        tresh1 = selected_ratios_none1(:,length(selected_ratios_none1));
        tresh2 = selected_ratios_none2(:,length(selected_ratios_none2));
        tresh3_none = selected_ratios_none3(:,length(selected_ratios_none3));

    else
        tresh3 = 0;
        tresh2 = 0;
        tresh1 = 0;
    end
else
    tresh3 = 0;
    tresh2 = 0;
    tresh1 = 0;
end



function [dSorted,dIndex]  = distfun(Center, Train, K)
%DISTFUN Calculate distances from training points to test points.
dSorted = zeros(1,K);
dIndex = zeros(1,K);
Dk = 1 - (Train * Center');
[dSorted,dIndex] = getBestK(Dk,K);


function [sorted,index] = getBestK(Dk,K)
% sort if needed
if K>1
    [sorted,index] = sort(Dk);
    sorted = sorted(1:K);
    index = index(1:K);
else
    [sorted,index] = min(Dk);
end

% Sample = [Training1(1:10,:); Training2(1:10,:); Training3(1:10,:)];
% c = knnclassify(Sample, Training, Group, 12, 'cosine');

% clear Y
% Y = X_av;


