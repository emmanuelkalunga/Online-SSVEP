% Description:
% ------------
% Draw tables in neurocompinting articles
% Author: Emmanuel K. Kalunga

%% OFFLINE RESULTS TABLE

%-- Load Lin2007 Results
load('CCA/cca_Lin2007.mat') 
acLin = ac'*100;
itrLin = itr';
%-- Load Nakanishi2014 Results
load ('CCA/nakanishi2014.mat')
acNak = acSubMean*100;
itrNak = itr;
%-- Load MDRM Results
load('./results/offline_basic_potato_3class.mat')
acMDRM = resMatrix(:,3,1)*100;
itrMDRM = itr;
%-- Loas MDRM opt. Results 
load('./results/offline_opt_potato_3class.mat')
acMDRMopt = resMatrix(:,3,1)*100;
itrMDRMopt = itr;

acMDRMoptPotato = resMatrix(:,3,2)*100;

po = bsxfun(@min,resMatrix(:,3,2),0.9999);
B = log2(3)+po.*log2(po)+(1-po).*log2((1-po)/(3-1));
itrMDRMoptPotato = B*(60/tLen);

colsep = ' & ';
rowending = ' \\';
fileID = 1; %-- Default
%-- ACCURACY TABLE
acc = [acLin, acNak, acMDRM, acMDRMopt, acMDRMoptPotato];

rowhead = {'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'Mean', 'std'};
colhead = {'Lin', 'Nak', 'MDRM', 'MDRM opt.', 'MDRM-outliers'};
disp('---------------------------------------------------');
disp('                Accuracy (%)                       ');
disp('---------------------------------------------------');
displaytable([acc; mean(acc); std(acc)],colhead,10,{'.1f'},rowhead,fileID,colsep,rowending)
disp('---------------------------------------------------');

%-- ITR TABLE
itr = [itrLin, itrNak, itrMDRM, itrMDRMopt, itrMDRMoptPotato];

rowhead = {'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'Mean', 'std'};
colhead = {'Lin', 'Nak', 'MDRM', 'MDRM opt.', 'MDRM-outliers'}
disp('---------------------------------------------------');
disp('                Accuracy (%)                       ');
disp('---------------------------------------------------');
displaytable([itr; mean(itr); std(itr)],colhead,10,{'.1f'},rowhead,fileID,colsep,rowending)
disp('---------------------------------------------------');

%-- COMBINED ACCURACY AND ITR
accitr = [acLin, itrLin, acNak, itrNak, acMDRM, itrMDRM, acMDRMopt, itrMDRMopt, acMDRMoptPotato, itrMDRMoptPotato];

rowhead = {'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'Mean', 'std'};
colhead1 = {'Lin', 'Nak', 'MDRM', 'MDRM opt.', 'MDRM-outliers'};
colhead2 = {'acc(%)', 'itr(bpm)', 'acc(%)', 'itr(bpm)', 'acc(%)', 'itr(bpm)', 'acc(%)', 'itr(bpm)', 'acc(%)', 'itr(bpm)'};
disp('---------------------------------------------------');
disp('          Offline performance comparison           ');
disp('---------------------------------------------------');
displaytable([accitr; mean(accitr); std(accitr)],colhead2,8,{'.1f'},rowhead,fileID,colsep,rowending)
disp('---------------------------------------------------');

%save('online_curve_potato_3class.mat', 'itr', 'tLen2', 'resMatrix', 'resMean', 'ac');

%% ONLINE RESULTS TABLE
%-- Load online cum results
load('./results/online_cum_3class')
acCum = resMatrix(:,3)*100;
delCum = del_sub_all';
itrCum = itr';
%-- Load online cum+curve+potato
load('./results/online_curve_potato_3class.mat')
acCur = resMatrix(:,3)*100;
delCur = del_sub_all';
itrCur = itr';
%-- Load online cum+curve+potato
load('./results/online_curve_3class.mat')
acCurPot = resMatrix(:,3)*100;
delCurPot = del_sub_all';
itrCurPot = itr';
%-- COMBINED ACCURACY AND ITR
colsep = ' & ';
rowending = ' \\';
fileID = 1; %-- Default
accDelItr = [acCum, delCum, itrCum, acCur, delCur, itrCur, acCurPot, delCurPot, itrCurPot];

rowhead = {'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'Mean', 'std'};
colhead1 = {'prob', 'prob+curve', '-outliers'};
colhead2 = {'acc(%)', 'delay(s)', 'itr(bpm)', 'acc(%)', 'delay(s)', 'itr(bpm)', 'acc(%)', 'delay(s)', 'itr(bpm)'};
disp('---------------------------------------------------');
disp('          Online performance comparison           ');
disp('---------------------------------------------------');
displaytable([accDelItr; mean(accDelItr); std(accDelItr)],colhead2,8,{'.1f'},rowhead,fileID,colsep,rowending)
disp('---------------------------------------------------');

%###################################################################################################################################
%% TABLE FOR 4 CLASSES
%###################################################################################################################################
%% ONLINE RESULTS TABLE
%-- Load online cum results
load('./results/online_cum_4class.mat')
acCum = resMatrix(:,3)*100;
delCum = del_sub_all';
itrCum = itr';
%-- Load online cum+curve+potato
load('./results/online_curve_4class.mat')
acCur = resMatrix(:,3)*100;
delCur = del_sub_all';
itrCur = itr';
%-- Load online cum+curve+potato
load('./results/online_curve_potato_4class.mat')
acCurPot = resMatrix(:,3)*100;
delCurPot = del_sub_all';
itrCurPot = itr';
%-- COMBINED ACCURACY AND ITR
colsep = ' & ';
rowending = ' \\';
fileID = 1; %-- Default
accDelItr = [acCum, delCum, itrCum, acCur, delCur, itrCur, acCurPot, delCurPot, itrCurPot];

rowhead = {'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'Mean', 'std'};
colhead1 = {'prob', 'prob+curve', '-outliers'};
colhead2 = {'acc(%)', 'delay(s)', 'itr(bpm)', 'acc(%)', 'delay(s)', 'itr(bpm)', 'acc(%)', 'delay(s)', 'itr(bpm)'};
disp('---------------------------------------------------');
disp('          Online performance comparison           ');
disp('---------------------------------------------------');
displaytable([accDelItr; mean(accDelItr); std(accDelItr)],colhead2,8,{'.1f'},rowhead,fileID,colsep,rowending)
disp('---------------------------------------------------');
