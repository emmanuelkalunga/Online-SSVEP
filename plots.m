clear all
% load classProb
load classProb_3class.mat; NbrSess = subNbrOfSess;
Labels = Labels+1;
sub11.Labels = Labels(:,:,1:NbrSess(11),11);
sub11.classProb = classProb(:,:,:,1:NbrSess(11),11);
sub12.Labels = Labels(:,:,1:NbrSess(12),12);
sub12.classProb = classProb(:,:,:,1:NbrSess(12),12);
color = ['g','r','c'];
% 
% for k = 1:4
%     tmp = mean(sub11.classProb(:,:,sub11.Labels(:,1,1)==k,1),3);
%     figure, plot(tmp', '-s', 'LineWidth',2, 'MarkerSize',5)
% end
% %frt = ['-sr';'-ob';'-dg';'-pk'; '-hm'; '-^c'];
%% PLOT CLASS PROBABILITY VS. NUMBER OF EPOCHS
%-- SUBJECT WITH LEAST PERFORMANCE
% for k = 1:4
%     figure
%     tmp = mean(sub11.classProb(:,:,sub11.Labels(:,1,1)==k,1),3);
%     for l = 1:4
%         if l == k
%             plot((1:numel(tmp(l,:))),tmp(l,:), '-sr', 'LineWidth',2, 'MarkerSize',4)
%         else
%             plot((1:numel(tmp(l,:))),tmp(l,:), '-sk', 'LineWidth',2, 'MarkerSize',4)
%         end 
%         hold on
%     end
% end
% xlabel('#Epochs');
% ylabel('Class probability');
% set(gca,'FontSize',14,'fontWeight','normal')
% set(findall(gcf,'type','text'),'FontSize',14,'fontWeight','normal')

%-- SUBJECT WITH BEST PERFOMANCE
legends = {'13Hz class', '21Hz class', '17Hz class'};
for k = 1:size(classProb,1)
    figure
    tmp = mean(sub12.classProb(:,:,sub12.Labels(:,1,1)==k,1),3);
    for l = 1:size(classProb,1)
        if l == k
            plot((1:numel(tmp(l,1:18))),tmp(l,1:18), ['-p' color(l)], 'LineWidth',2, 'MarkerSize',5, 'MarkerFaceColor', color(l), 'DisplayName', legends{l})
            legend('-DynamicLegend');
            legend boxoff
        else
            plot((1:numel(tmp(l,1:18))),tmp(l,1:18), ['-d' color(l)], 'LineWidth',1, 'MarkerSize',5, 'MarkerFaceColor', color(l), 'DisplayName', legends{l})
            legend('-DynamicLegend');
            legend boxoff
        end 
        hold all
    end
    xlabel('Epoch index');
    ylabel('Class probability');
    set(gca,'FontSize',14,'fontWeight','normal')
    set(findall(gcf,'type','text'),'FontSize',14,'fontWeight','normal')
end

%% PLOT AVERAGE CLASSIFICATION ERROR VS. NUMBER OF EPOCHS
for sub = 1:12
    clear er
    sub_labels = Labels(:,:,1:NbrSess(sub),sub);
    sub_prob = classProb(:,:,:,1:NbrSess(sub),sub); 
    for sess = 1:NbrSess(sub)
        for tr = 1:8*size(classProb,1)
            [v p] = max(sub_prob(:,:,tr,sess));
%             er(tr,:,sess) = abs(p - sub_labels(tr,:,sess));
%             er(tr,:,sess) = ceil(er(tr,:,sess)/4);
            er1(tr,:,sess,sub) = abs(p - sub_labels(tr,:,sess));
            er1(tr,:,sess,sub) = ceil(er1(tr,:,sess,sub)/size(classProb,1));
        end
    end
%     tmp = permute(er,[2 1 3]);
%     tmp2 = tmp(:,:);
%     errMean(sub,:) = mean(tmp2,2);
    
    e = squeeze(er1(:,:,1:NbrSess(sub),sub));
    tmp = permute(e,[2 1 3]);
    tmp2 = tmp(:,:);
    errMean(sub,:) = mean(tmp2,2);
    
    %figure, plot(errMean(sub,:))
end
mer = mean(errMean); %-- All subjects mean error
ser = std(errMean); %-- standard dev. of error accross subjects
figure, plot(mer(1:18), '-o','LineWidth',2)
xlabel('Epoch index');
ylabel('Average classification error');
set(gca,'FontSize',14,'fontWeight','normal')
set(findall(gcf,'type','text'),'FontSize',14,'fontWeight','normal')

%-- PLOT ERROR BAR
figure, errorbar(mer, ser, '-s','LineWidth',1, 'MarkerSize',10, 'MarkerFaceColor', 'b')
xlabel('Epoch index');
ylabel('Average classification error');
set(gca,'FontSize',14,'fontWeight','normal')
set(findall(gcf,'type','text'),'FontSize',14,'fontWeight','normal')


%% PLOT AVERAGE CLASSIFICATION ERROR VS. PROBABILITY THRESHOLD
clear all
% load classProb
load classProb_3class.mat; NbrSess = subNbrOfSess;
Labels = Labels+1;
probIdx = 1;
for prob = 0:0.1:1
    for sub = 1:12
        sub_labels = Labels(:,:,1:NbrSess(sub),sub);
        sub_prob = classProb(:,:,:,1:NbrSess(sub),sub); 
        for sess = 1:NbrSess(sub)
            for tr = 1:8*size(classProb,1)
                [v p] = max(sub_prob(:,:,tr,sess));
                v2 = find(v>prob);
                if (isempty(v2))
                    ytrial_pos = numel(v);
                    ytrial = 1;
                else
                    ytrial_pos = v2(1);
                    ytrial = p(ytrial_pos);
                end
                er0 = abs(ytrial - sub_labels(tr,1,sess));
                err(tr,sess,sub,probIdx) = ceil(er0/size(classProb,1));
                temps(tr,sess,sub,probIdx) = ytrial_pos*0.2;
            end
        end
    end
    probIdx = probIdx + 1;
end
for sub = 1:12
    tmp = squeeze(err(:,1:NbrSess(sub),sub,:));
    tmp2 = permute(tmp, [3, 1, 2]);
    tmp3 = tmp2(:,:);
    erMean(:,sub) = mean(tmp3,2);
    
    tmp = squeeze(temps(:,1:NbrSess(sub),sub,:));
    tmp2 = permute(tmp, [3, 1, 2]);
    tmp3 = tmp2(:,:);
    tempsMean(:,sub) = mean(tmp3,2);
    
%     e = squeeze(er1(:,:,:,sub));
%     tmp = permute(e,[2 1 3]);
%     tmp2 = tmp(:,:);
%     errMean(sub,:) = mean(tmp2,2);
    
end

prob = 0:0.1:0.9;

figure, plot(prob, mean(erMean(1:10,:),2), '--s','LineWidth',2)
xlabel('Probability threshold ');
ylabel('Average classification error');
set(gca,'FontSize',14,'fontWeight','normal')
set(findall(gcf,'type','text'),'FontSize',14,'fontWeight','normal')
%-- PLOT ERROR BAR
figure, errorbar(prob, mean(erMean(1:10,:),2), std(erMean(1:10,:),[],2), '-s','LineWidth',1, 'MarkerSize',10, 'MarkerFaceColor', 'b')
xlim([-0.1 1])
ylim([0.1 0.4])
xlabel('Probability threshold ');
ylabel('Average classification error');
set(gca,'FontSize',14,'fontWeight','normal')
set(findall(gcf,'type','text'),'FontSize',14,'fontWeight','normal')

% %-- PLOT AVERAGE ITR VS. PROBABILITY THRESHOLD
% po = 1-mean(erMean(1:10,:),2);
% tLen = mean(tempsMean(1:10,:),2);
% B = log2(4)+po.*log2(po)+(1-po).*log2((1-po)/(4-1))
% bBm = B.*(60./tLen); 
% figure, plot(bBm)


figure
%ylim([0 0.5])
[hAx,hLine1,hLine2] = plotyy(prob, mean(erMean(1:10,:),2), prob,mean(tempsMean(1:10,:),2),'plot');
set(get(hAx(1),'Ylabel'),'String','Average classification error') 
set(get(hAx(2),'Ylabel'),'String','Average classification time (s)') 
set(hLine1,'LineStyle','--','Marker', 's','LineWidth', 2)
set(hLine2,'LineStyle',':','Marker', 'o','LineWidth', 2)
xlabel('Probability threshold ')
set(gca,'FontSize',14,'fontWeight','normal')
set(hAx(2),'FontSize',14,'fontWeight','normal')
set(findall(gcf,'type','text'),'FontSize',14,'fontWeight','normal')
 
% %--PLot error bars
% hold(hAx(1),'on')
% errorbar(hAx(1), prob, mean(erMean(1:5,:),2), std(erMean(1:5,:),[],2), 's' )
% hold(hAx(1),'off')
% hold(hAx(2),'on')
% errorbar(hAx(2), alpha, resMean(:,4), resStd(:,4), 'o' )
% ylim(hAx(1),[20 max((round(1000*resMean(:,2)))/10+ (round(1000*resStd(:,2)))/10)])
% % ylim(hAx(2),[0 max(resMean(:,4)+ resStd(:,4))])
% ylim(hAx(2),[0 0.8])

%% PLOT IMPACT OF W (window size/tLen) on online clasification accuracy and ITR
load online_curve_tlen_3class_new.mat
meanAcc = mean(subAcMean,2);
stdAcc = std(subAcMean,[],2);
meanItr = mean(itr3,2);
stdItr = std(itr3,[],2);
figure
[hAx,hLine1,hLine2] = plotyy(tlen,(round(1000*meanAcc))/10,tlen,meanItr,'plot');
set(get(hAx(1),'Ylabel'),'String','Accuracy (%)') 
set(get(hAx(2),'Ylabel'),'String','ITR (bits/min)') 
set(hLine1,'LineStyle','-', 'LineWidth', 3)
set(hLine2,'LineStyle','-', 'LineWidth', 3)
xlabel('Window size (w) in sec ')
set(gca,'FontSize',14,'fontWeight','normal')
set(hAx(2),'FontSize',14,'fontWeight','normal')
set(findall(gcf,'type','text'),'FontSize',14,'fontWeight','normal')
%--PLot error bars
hold(hAx(1),'on')
errorbar(hAx(1), tlen, (round(1000*meanAcc))/10, (round(1000*stdAcc))/10)
hold(hAx(1),'off')
hold(hAx(2),'on')
errorbar(hAx(2), tlen, meanItr, stdItr)
ylim(hAx(1),[50 max((round(1000*meanAcc))/10+ (round(1000*stdAcc))/10)])
% ylim(hAx(2),[0 max(resMean(:,4)+ resStd(:,4))])
ylim(hAx(2),[0 max(meanItr)+max(stdItr)])
%##################################################################################################################################################################
%% PLOT FOR 4 CLASSES
%##################################################################################################################################################################
clear all
% load classProb
load classProb_4class.mat; NbrSess = subNbrOfSess;
Labels = Labels+1;
sub11.Labels = Labels(:,:,1:NbrSess(11),11);
sub11.classProb = classProb(:,:,:,1:NbrSess(11),11);
sub12.Labels = Labels(:,:,1:NbrSess(12),12);
sub12.classProb = classProb(:,:,:,1:NbrSess(12),12);
% 
% for k = 1:4
%     tmp = mean(sub11.classProb(:,:,sub11.Labels(:,1,1)==k,1),3);
%     figure, plot(tmp', '-s', 'LineWidth',2, 'MarkerSize',5)
% end
% %frt = ['-sr';'-ob';'-dg';'-pk'; '-hm'; '-^c'];
%% PLOT CLASS PROBABILITY VS. NUMBER OF EPOCHS
%-- SUBJECT WITH LEAST PERFORMANCE
% for k = 1:4
%     figure
%     tmp = mean(sub11.classProb(:,:,sub11.Labels(:,1,1)==k,1),3);
%     for l = 1:4
%         if l == k
%             plot((1:numel(tmp(l,:))),tmp(l,:), '-sr', 'LineWidth',2, 'MarkerSize',4)
%         else
%             plot((1:numel(tmp(l,:))),tmp(l,:), '-sk', 'LineWidth',2, 'MarkerSize',4)
%         end 
%         hold on
%     end
% end
% xlabel('Epoch index');
% ylabel('Class probability');
% set(gca,'FontSize',14,'fontWeight','normal')
% set(findall(gcf,'type','text'),'FontSize',14,'fontWeight','normal')

%-- SUBJECT WITH BEST PERFOMANCE
color = ['b','g','r','c'];
legends = {'Resting clas', '13Hz class', '21Hz class', '17Hz class'};
for k = 1:size(classProb,1)
    figure
    tmp = mean(sub12.classProb(:,:,sub12.Labels(:,1,1)==k,1),3);
    for l = 1:size(classProb,1)
        if l == k
            plot((1:numel(tmp(l,1:18))),tmp(l,1:18), ['-p' color(l)], 'LineWidth',2, 'MarkerSize',5, 'MarkerFaceColor', color(l), 'DisplayName', legends{l})
            legend('-DynamicLegend');
            %legend boxoff
        else
            plot((1:numel(tmp(l,1:18))),tmp(l,1:18), ['-d' color(l)], 'LineWidth',1, 'MarkerSize',5, 'MarkerFaceColor', color(l), 'DisplayName', legends{l})
            legend('-DynamicLegend');
            %legend boxoff
        end 
        hold all
    end
    xlabel('Epoch index');
    ylabel('Class probability');
    set(gca,'FontSize',14,'fontWeight','normal')
    set(findall(gcf,'type','text'),'FontSize',14,'fontWeight','normal')
end
%% PLOT AVERAGE CLASSIFICATION ERROR VS. NUMBER OF EPOCHS
for sub = 1:12
    clear er
    sub_labels = Labels(:,:,1:NbrSess(sub),sub);
    sub_prob = classProb(:,:,:,1:NbrSess(sub),sub); 
    for sess = 1:NbrSess(sub)
        for tr = 1:8*size(classProb,1)
            [v p] = max(sub_prob(:,:,tr,sess));
%             er(tr,:,sess) = abs(p - sub_labels(tr,:,sess));
%             er(tr,:,sess) = ceil(er(tr,:,sess)/4);
            er1(tr,:,sess,sub) = abs(p - sub_labels(tr,:,sess));
            er1(tr,:,sess,sub) = ceil(er1(tr,:,sess,sub)/size(classProb,1));
        end
    end
%     tmp = permute(er,[2 1 3]);
%     tmp2 = tmp(:,:);
%     errMean(sub,:) = mean(tmp2,2);
    
    e = squeeze(er1(:,:,1:NbrSess(sub),sub));
    tmp = permute(e,[2 1 3]);
    tmp2 = tmp(:,:);
    errMean(sub,:) = mean(tmp2,2);
    
    %figure, plot(errMean(sub,:))
end
mer = mean(errMean); %-- All subjects mean error
ser = std(errMean); %-- standard dev. of error accross subjects
figure, plot(mer(1:18), '-o','LineWidth',2)
xlabel('Epoch index');
ylabel('Average classification error');
set(gca,'FontSize',14,'fontWeight','normal')
set(findall(gcf,'type','text'),'FontSize',14,'fontWeight','normal')

%-- PLOT ERROR BAR
figure, errorbar(mer, ser, '-s','LineWidth',1, 'MarkerSize',10, 'MarkerFaceColor', 'b')
xlabel('Epoch index');
ylabel('Average classification error');
set(gca,'FontSize',14,'fontWeight','normal')
set(findall(gcf,'type','text'),'FontSize',14,'fontWeight','normal')

%% PLOT AVERAGE CLASSIFICATION ERROR VS. PROBABILITY THRESHOLD
clear all
% load classProb
load classProb_4class.mat; NbrSess = subNbrOfSess;
Labels = Labels+1;
probIdx = 1;
for prob = 0:0.1:1
    for sub = 1:12
        sub_labels = Labels(:,:,1:NbrSess(sub),sub);
        sub_prob = classProb(:,:,:,1:NbrSess(sub),sub); 
        for sess = 1:NbrSess(sub)
            for tr = 1:8*size(classProb,1)
                [v p] = max(sub_prob(:,:,tr,sess));
                v2 = find(v>prob);
                if (isempty(v2))
                    ytrial_pos = numel(v);
                    ytrial = 1;
                else
                    ytrial_pos = v2(1);
                    ytrial = p(ytrial_pos);
                end
                er0 = abs(ytrial - sub_labels(tr,1,sess));
                err(tr,sess,sub,probIdx) = ceil(er0/size(classProb,1));
                temps(tr,sess,sub,probIdx) = ytrial_pos*0.2;
            end
        end
    end
    probIdx = probIdx + 1;
end
for sub = 1:12
    tmp = squeeze(err(:,1:NbrSess(sub),sub,:));
    tmp2 = permute(tmp, [3, 1, 2]);
    tmp3 = tmp2(:,:);
    erMean(:,sub) = mean(tmp3,2);
    
    tmp = squeeze(temps(:,1:NbrSess(sub),sub,:));
    tmp2 = permute(tmp, [3, 1, 2]);
    tmp3 = tmp2(:,:);
    tempsMean(:,sub) = mean(tmp3,2);
    
end

prob = 0:0.1:0.9;

figure, plot(prob, mean(erMean(1:10,:),2), '--s','LineWidth',2)
xlabel('Probability threshold ');
ylabel('Average classification error');
set(gca,'FontSize',14,'fontWeight','normal')
set(findall(gcf,'type','text'),'FontSize',14,'fontWeight','normal')
%-- PLOT ERROR BAR
figure, errorbar(prob, mean(erMean(1:10,:),2), std(erMean(1:10,:),[],2), '-s','LineWidth',1, 'MarkerSize',10, 'MarkerFaceColor', 'b')
xlim([-0.1 1])
ylim([0.1 0.5])
xlabel('Probability threshold ');
ylabel('Average classification error');
set(gca,'FontSize',14,'fontWeight','normal')
set(findall(gcf,'type','text'),'FontSize',14,'fontWeight','normal')

% %-- PLOT AVERAGE ITR VS. PROBABILITY THRESHOLD
% po = 1-mean(erMean(1:10,:),2);
% tLen = mean(tempsMean(1:10,:),2);
% B = log2(4)+po.*log2(po)+(1-po).*log2((1-po)/(4-1))
% bBm = B.*(60./tLen); 
% figure, plot(bBm)


figure
%ylim([0 0.5])
[hAx,hLine1,hLine2] = plotyy(prob, mean(erMean(1:10,:),2), prob,mean(tempsMean(1:10,:),2),'plot');
set(get(hAx(1),'Ylabel'),'String','Average classification error') 
set(get(hAx(2),'Ylabel'),'String','Average classification time (s)') 
set(hLine1,'LineStyle','--','Marker', 's','LineWidth', 2)
set(hLine2,'LineStyle',':','Marker', 'o','LineWidth', 2)
xlabel('Probability threshold ')
set(gca,'FontSize',14,'fontWeight','normal')
set(hAx(2),'FontSize',14,'fontWeight','normal')
set(findall(gcf,'type','text'),'FontSize',14,'fontWeight','normal')
 
% %--PLot error bars
% hold(hAx(1),'on')
% errorbar(hAx(1), prob, mean(erMean(1:5,:),2), std(erMean(1:5,:),[],2), 's' )
% hold(hAx(1),'off')
% hold(hAx(2),'on')
% errorbar(hAx(2), alpha, resMean(:,4), resStd(:,4), 'o' )
% ylim(hAx(1),[20 max((round(1000*resMean(:,2)))/10+ (round(1000*resStd(:,2)))/10)])
% % ylim(hAx(2),[0 max(resMean(:,4)+ resStd(:,4))])
% ylim(hAx(2),[0 0.8])

%% PLOT IMPACT OF W (window size/tLen) on online clasification accuracy and ITR
load online_curve_tlen_4class.mat
meanAcc = mean(subAcMean,2);
stdAcc = std(subAcMean,[],2);
meanItr = mean(itr3,2);
stdItr = std(itr3,[],2);
figure
[hAx,hLine1,hLine2] = plotyy(tlen,(round(1000*meanAcc))/10,tlen,meanItr,'plot');
set(get(hAx(1),'Ylabel'),'String','Accuracy (%)') 
set(get(hAx(2),'Ylabel'),'String','ITR (bits/min)') 
set(hLine1,'LineStyle','-', 'LineWidth', 3)
set(hLine2,'LineStyle','-', 'LineWidth', 3)
xlabel('Window size (w) in sec ')
set(gca,'FontSize',14,'fontWeight','normal')
set(hAx(2),'FontSize',14,'fontWeight','normal')
set(findall(gcf,'type','text'),'FontSize',14,'fontWeight','normal')
%--PLot error bars
hold(hAx(1),'on')
errorbar(hAx(1), tlen, (round(1000*meanAcc))/10, (round(1000*stdAcc))/10)
hold(hAx(1),'off')
hold(hAx(2),'on')
errorbar(hAx(2), tlen, meanItr, stdItr)
ylim(hAx(1),[50 max((round(1000*meanAcc))/10+ (round(1000*stdAcc))/10)])
% ylim(hAx(2),[0 max(resMean(:,4)+ resStd(:,4))])
ylim(hAx(2),[0 max(meanItr)+max(stdItr)])
%% PLOT CONFUSION MATRIX AND ROC SPACE
clear all
load online_curve_potato_4class.mat
LabelAll = LabelAll+1;
Yall = Yall+1;
targets = zeros(numel(unique(LabelAll)), numel(LabelAll));
outputs = zeros(numel(unique(LabelAll)), numel(LabelAll));
for k = 1:numel(unique(LabelAll))
    lab = zeros(4,1);
    lab(k) = 1;
    targets(:,LabelAll==k) = repmat(lab,1,sum(LabelAll==k));
    outputs(:,Yall==k) = repmat(lab,1,sum(Yall==k));
    TPR(k) = CP{k}.Sensitivity;
    FPR(k) = 1-CP{k}.Specificity;
end
figure,
plotconfusion(targets, outputs);

chance = [0 1];
figure,
plot([0 1], chance,'--r','LineWidth', 3, 'DisplayName', 'Random guess')
%legend('Random guess')
legend('-DynamicLegend');
hold all
marker = ['h','p','s','d'];
legends = {'Resting class', '13Hz class', '21Hz class', '17Hz class'};
for k = 1:numel(unique(LabelAll))
    plot(FPR(k), TPR(k), marker(k), 'LineWidth', 3, 'MarkerSize',8, 'DisplayName', legends{k})
    %legend('cl')
end
plot(0,1, 'o', 'LineWidth', 3, 'MarkerSize',8, 'DisplayName', 'Perfect classification')
plot(1,0, 'o', 'LineWidth', 3, 'MarkerSize',8, 'DisplayName', 'Worst classification')

xlabel('FPR or (1-specificity)');
ylabel('TPR or sensitivity');
set(gca,'FontSize',14,'fontWeight','normal')
set(findall(gcf,'type','text'),'FontSize',14,'fontWeight','normal')
%% PLOT DELAYS IN TIME SIGNAL SYNCHRONIZATION

clear all
tLen = 8;
delay = -2;

for sub = 16:17  %- ploting two subjects (11 and 12)
    clear x_all H_all P X Pm PSD
     %-- Load data
     [S_all, H_all] = loaddata(sub); %Returns cells of data from all available sessions
     Fs = H_all{1}.SampleRate;
     nbrSessions = length(S_all);
     sessions = 1:nbrSessions;
     %- Preprocessing of all available sessions (Same for training and test data)
     % 1) Band pass filter
     for session = 1:nbrSessions
         x_all{session} = bandpass_filter_ext([12.95 13.05], [16.9 17.1], [20.9 21.1], S_all{session}, H_all{session}); %74.31
     end
     X = get_trials(x_all, H_all, tLen, delay);
     taxis = [delay:1/Fs:delay+tLen]; taxis = taxis(1:end-1);
     titles = [
        'Example trial from resting state';
        ' Example trial from 13Hz SSVEP  ';
        ' Example trial from 21Hz SSVEP  ';
        ' Example trial from 17Hz SSVEP  '];
    %klass = ['No'; '13'; '21'; '17'];
    set(groot,'defaultAxesColorOrder','remove')
    set(groot,'defaultAxesColorOrder',[0 1 0;1 0 0;0 1 1]);
    figure
    %set(groot,'defaultAxesColorOrder',[0 1 0;1 0 0;0 1 1]);
    set(gcf,'DefaultAxesColorOrder',[0 1 0;1 0 0;0 1 1])
    for cl = 1:size(X,2) 
        subplot(2,2,cl)
        if sub == 16
            %set(gca, 'ColorOrder', [0 1 0; 1 0 0;0 1 1], 'NextPlot', 'replacechildren');
            plot(taxis, squeeze(X{cl}([1 17 9],:,7))'),
            if cl > 2, xlabel('time(s)'); end
        end
        if sub == 17
            %set(gca, 'ColorOrder', [0 1 0; 1 0 0;0 1 1]);%, 'NextPlot', 'replacechildren');
            plot(taxis, squeeze(X{cl}([1 17 9],:,10))'),
            if cl > 2, xlabel('time(s)'); end
        end
        hold on
        if sub == 16
            plot([0 0],[-0.0021 0.0021], '>--k')
            ylim([-0.0021 0.0021])
        end
        if sub == 17
            plot([0 0],[-0.006 0.006], '>--k')
            ylim([-0.006 0.006])
        end
        if cl == 1 & sub == 17
            legend(['13Hz';'21Hz';'17Hz']);
            %legend boxoff 
        end
        title(titles(cl,:))
        yticks = get(gca,'ytick');
        set(gca,'yticklabel',yticks*1000);
        if mod(cl,2)==0
            set(gca,'YTick',[])
        end
        if cl < 3
            set(gca, 'XTick', [])
        end
        set(gca,'FontSize',12,'fontWeight','normal')
        set(findall(gcf,'type','text'),'FontSize',12,'fontWeight','normal')
        pbaspect([16/16 9/16 1]); %Set figure(plot) aspect ratio
    end
    %fpad = [0 0 0 0]; axpad = [0 0];
    spaceplots; %cutt off blank margins (from matlab filexchage)
end






